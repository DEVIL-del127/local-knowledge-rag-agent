"""Run the frozen 100-question NLU V2 semantic benchmark with checkpoints."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from nlu_v2_validate import build_engine


PROJECT_ROOT = Path(__file__).resolve().parent
# This checked-in run artifact contains the frozen 1..100 source questions.
# It replaces the former machine-local 1..50 document, so the benchmark can be
# launched from any checkout without a personal external file.
DEFAULT_BENCHMARK_INPUT = PROJECT_ROOT / "output" / "nlu_v2_benchmark_100_20260825" / "results.json"

HEURISTIC_REQUIREMENTS = {
    "sequence_event": re.compile(r"连续|持续\s*(?:超过|至少|大于)|有序|顺序"),
    "set_operation": re.compile(r"交集|并集|差集|同时满足|去重"),
    "calculation": re.compile(
        r"计算|统计|平均|均值|总额|总和|数量|峰值|最大|最小|中位数|分位数|"
        r"相关系数|协方差|比值|比例|占比|增长|回撤|波动率|标准差|P\d+",
        re.I,
    ),
    "turn_context": re.compile(r"上一轮|本轮|同一轮|停止刚才|取消旧|新 turn|前一步|上一步", re.I),
}


def parse_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(source_cases, list):
            raise ValueError(f"JSON benchmark input must contain a cases list: {path}")
        result = []
        for item in source_cases:
            if not isinstance(item, dict) or not {"id", "query"} <= set(item):
                raise ValueError(f"invalid benchmark case in {path}")
            result.append({
                "id": int(item["id"]), "title": str(item.get("title", "")),
                "query": str(item["query"]), "source_file": str(path),
            })
        return result
    text = path.read_text(encoding="utf-8-sig")
    headings = list(re.finditer(r"(?m)^### 测试题\s+(\d+)[：:]\s*(.*)$", text))
    result = []
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[start:end]
        quote_lines = []
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                value = stripped[1:].strip()
                if value:
                    quote_lines.append(value)
        query = "\n".join(quote_lines).strip()
        if not query:
            paragraphs = [
                item.strip() for item in re.split(r"\n\s*\n", section)
                if item.strip() and not set(item.strip()) <= {"-"}
            ]
            query = paragraphs[0] if paragraphs else ""
        result.append({
            "id": int(heading.group(1)),
            "title": heading.group(2).strip(),
            "query": query,
            "source_file": str(path),
        })
    return result


def summarize_result(case: dict[str, Any], payload: dict[str, Any], elapsed: float) -> dict[str, Any]:
    ir = payload["understanding"]
    validation = payload["validation"]
    patch_report = payload.get("patch_report") or {}
    requirements = ir.get("requirements", [])
    satisfied = sum(item.get("status") == "satisfied" for item in ir.get("coverage", []))
    heuristic_gaps = heuristic_gap_codes(case["query"], ir)
    return {
        "id": case["id"],
        "title": case["title"],
        "elapsed_seconds": round(elapsed, 3),
        "status": validation["status"],
        "executable": validation["executable"],
        "compatibility_intent": ir["compatibility_intent"],
        "understanding_status": ir.get("understanding_status"),
        "binding_status": ir.get("binding_status"),
        "reliability": validation["reliability"],
        "sources": [item["identifier"] for item in ir["source_candidates"]],
        "projections": [item.get("canonical_id") or item.get("raw_name") for item in ir["projections"]],
        "event_count": len(ir.get("events", [])),
        "set_operation_count": len(ir.get("set_operations", [])),
        "calculation_count": len(ir.get("calculations", [])),
        "aggregate_count": len(ir.get("aggregates", [])),
        "comparison_count": len(ir.get("comparisons", [])),
        "turn_directive_count": len(ir.get("turn_directives", [])),
        "requirement_coverage": {
            "satisfied": satisfied, "total": len(requirements),
            "ratio": round(satisfied / len(requirements), 4) if requirements else 1.0,
        },
        "unresolved_codes": [item["code"] for item in ir.get("unresolved", [])],
        "validation_error_codes": [item["code"] for item in validation.get("errors", [])],
        "logical_status": payload["logical_plan"]["status"],
        "logical_node_types": [item["task_type"] for item in payload["logical_plan"]["nodes"]],
        "model_calls": payload["model_calls"],
        "patch": {
            "protocol": patch_report.get("protocol"),
            "accepted": len(patch_report.get("accepted", [])),
            "rejected": len(patch_report.get("rejected", [])),
            "closed_gaps": len(patch_report.get("closed_gap_ids", [])),
            "committed": bool(patch_report.get("committed", False)),
            "transaction_error": patch_report.get("transaction_error", ""),
            "concrete_gain": patch_report.get("concrete_gain_count", 0),
            "clarification_gain": patch_report.get("clarification_gain_count", 0),
        },
        "heuristic_gap_codes": heuristic_gaps,
        "false_valid_candidate": bool(validation["executable"] and heuristic_gaps),
    }


def heuristic_gap_codes(query: str, ir: dict[str, Any]) -> list[str]:
    gaps = []
    if HEURISTIC_REQUIREMENTS["sequence_event"].search(query) and not ir.get("events"):
        gaps.append("explicit_sequence_without_event")
    if HEURISTIC_REQUIREMENTS["set_operation"].search(query) and not ir.get("set_operations"):
        gaps.append("explicit_set_without_set_operation")
    has_computation = any((
        ir.get("calculations"), ir.get("aggregates"), ir.get("comparisons"), ir.get("metrics"),
    ))
    if HEURISTIC_REQUIREMENTS["calculation"].search(query) and not has_computation:
        gaps.append("explicit_calculation_without_compute_ir")
    if HEURISTIC_REQUIREMENTS["turn_context"].search(query) and not ir.get("turn_directives"):
        gaps.append("explicit_turn_context_without_directive")
    return gaps


def build_report(run: dict[str, Any]) -> str:
    summaries = run["summaries"]
    status_counts = Counter(item["status"] for item in summaries)
    error_counts = Counter(
        code for item in summaries for code in item["validation_error_codes"]
    )
    gap_counts = Counter(code for item in summaries for code in item["heuristic_gap_codes"])
    lines = [
        "# NLU V2 100 题语义基准运行报告",
        "",
        f"- Started: `{run['started_at']}`",
        f"- Finished: `{run.get('finished_at', '')}`",
        f"- Model: `{run['model']}`",
        f"- LLM enabled: `{run['llm_enabled']}`",
        f"- Isolated cases: `{run.get('isolate_cases', False)}`",
        f"- Completed: `{len(summaries)}/100`",
        f"- Runtime errors: `{len(run['runtime_errors'])}`",
        f"- Validation statuses: `{dict(status_counts)}`",
        f"- Executable: `{sum(item['executable'] for item in summaries)}`",
        f"- Model calls: `{sum(item['model_calls'] for item in summaries)}`",
        f"- Committed patches: `{sum(item['patch']['committed'] for item in summaries)}`",
        f"- Concrete Patch gains: `{sum(item['patch']['concrete_gain'] for item in summaries)}`",
        f"- Clarification gains: `{sum(item['patch']['clarification_gain'] for item in summaries)}`",
        f"- False-valid candidates: `{sum(item['false_valid_candidate'] for item in summaries)}`",
        "",
        "## Important Interpretation",
        "",
        "本报告是运行与结构审计，不是 Golden IR exact-match 准确率。原始 100 题没有机器可读的",
        "expected IR 标注；`false_valid_candidate` 是保守启发式告警，需要结合逐题 IR 人工复核。",
        "",
        "## Frequent Validation Errors",
        "",
    ]
    lines.extend(f"- `{code}`: {count}" for code, count in error_counts.most_common(20))
    lines.extend(["", "## Heuristic Semantic Gaps", ""])
    lines.extend(f"- `{code}`: {count}" for code, count in gap_counts.most_common())
    lines.extend(["", "## Per Case", ""])
    lines.append("| ID | Status | Exec | Calls | Patch | Req | Events/Sets/Calcs | Heuristic gaps |")
    lines.append("|---:|---|:---:|---:|:---:|---:|---|---|")
    for item in summaries:
        coverage = item["requirement_coverage"]
        structures = (
            f"{item['event_count']}/{item['set_operation_count']}/"
            f"{item['calculation_count'] + item['aggregate_count'] + item['comparison_count']}"
        )
        lines.append(
            f"| {item['id']} | {item['status']} | {str(item['executable']).lower()} | "
            f"{item['model_calls']} | {str(item['patch']['committed']).lower()} | "
            f"{coverage['satisfied']}/{coverage['total']} | {structures} | "
            f"{', '.join(item['heuristic_gap_codes']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def save_checkpoint(output_dir: Path, run: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "results.json"
    temp_path = output_dir / "results.json.tmp"
    temp_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(raw_path)
    (output_dir / "summary.md").write_text(build_report(run), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=None,
                        help="project-local Markdown or JSON question source; repeatable")
    parser.add_argument("--first", type=Path, default=DEFAULT_BENCHMARK_INPUT,
                        help="compatibility input; defaults to the checked-in project fixture")
    parser.add_argument("--second", type=Path, default=None,
                        help="optional additional Markdown/JSON question source")
    parser.add_argument("--output", type=Path, default=Path("output/nlu_v2_benchmark_100"))
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--isolate-cases", action="store_true",
        help="create one Engine per case so breaker/cache state cannot leak across cases",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    args = parser.parse_args()

    input_paths = args.input or [item for item in (args.first, args.second) if item is not None]
    if not all(item.is_file() for item in input_paths):
        missing = [str(item) for item in input_paths if not item.is_file()]
        raise FileNotFoundError(f"benchmark input does not exist: {missing}")
    cases_by_id: dict[int, dict[str, Any]] = {}
    for input_path in input_paths:
        for case in parse_questions(input_path):
            previous = cases_by_id.get(case["id"])
            if previous is not None and previous["query"] != case["query"]:
                raise ValueError(f"duplicate benchmark ID with different query: {case['id']}")
            cases_by_id[case["id"]] = case
    cases = [cases_by_id[item] for item in sorted(cases_by_id)]
    ids = [item["id"] for item in cases]
    if ids != list(range(1, 101)):
        raise ValueError(f"expected question IDs 1..100, got {ids}")
    selected = [item for item in cases if args.start <= item["id"] <= args.end]
    shared_engine = None if args.isolate_cases else build_engine(
        not args.no_llm, model=args.model
    )
    run = {
        "schema_version": "nlu_v2_benchmark_run_v1",
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": "",
        "model": args.model,
        "llm_enabled": not args.no_llm,
        "isolate_cases": args.isolate_cases,
        "question_sources": [str(item) for item in input_paths],
        "cases": [],
        "summaries": [],
        "runtime_errors": [],
    }
    save_checkpoint(args.output, run)
    for case in selected:
        started = time.perf_counter()
        try:
            engine = shared_engine or build_engine(not args.no_llm, model=args.model)
            result = engine.analyze(case["query"])
            payload = result.to_dict()
            elapsed = time.perf_counter() - started
            summary = summarize_result(case, payload, elapsed)
            run["cases"].append({**case, "result": payload})
            run["summaries"].append(summary)
            print(
                f"[{case['id']:03d}/100] {summary['status']:<19} "
                f"calls={summary['model_calls']} patch={summary['patch']['committed']} "
                f"gaps={len(summary['heuristic_gap_codes'])} {elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            error = {
                "id": case["id"], "title": case["title"],
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(elapsed, 3),
            }
            run["runtime_errors"].append(error)
            print(f"[{case['id']:03d}/100] ERROR {error['error']}", flush=True)
        save_checkpoint(args.output, run)
    run["finished_at"] = datetime.now().astimezone().isoformat()
    save_checkpoint(args.output, run)
    print(f"Saved benchmark to {args.output}", flush=True)
    return 0 if not run["runtime_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
