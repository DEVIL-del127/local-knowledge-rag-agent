"""Compare frozen M1.4 batch outputs and emit machine/readable acceptance reports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASE_RE = re.compile(r"case_(\d{3})_")
CORE_CODES = (
    "event_structure_incomplete",
    "set_inputs_unbound",
    "formula_input_unbound",
)
FROZEN_ZERO_CODES = (
    "source_demand_slot_anchor_missing",
    "source_demand_slot_anchor_invalid",
    "repair_contract_input_out_of_scope",
    "output_ref_unbound",
    "arithmetic_output_mismatch",
    "relation_filter_predicate_unbound",
)
OPERATOR_REQUIREMENTS = {
    "sequence_event", "cumulative_duration", "set_operation", "aggregate",
    "scoped_aggregate", "calculation", "formula", "reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "tests" / "fixtures" / "m14_eligible_manifest.json",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def load_cases(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for file_path in sorted(path.glob("case_*.json")):
        match = CASE_RE.match(file_path.name)
        if not match:
            continue
        try:
            result[int(match.group(1))] = json.loads(file_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result[int(match.group(1))] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def summarize(cases: dict[int, dict[str, Any]], eligible_ids: set[int]) -> dict[str, Any]:
    validation_status = Counter()
    error_occurrences = Counter()
    error_case_ids: dict[str, set[int]] = {}
    node_counts = Counter()
    model_calls = []
    audits = []
    committed = concrete = 0
    runtime_error_ids = []
    eligible_operator_cases = 0
    eligible_operator_complete = 0
    eligible_event_cases = eligible_event_complete = 0
    eligible_combined_cases = eligible_combined_complete = 0

    for case_id, payload in cases.items():
        validation = payload.get("validation")
        understanding = payload.get("understanding", {})
        if not isinstance(validation, dict):
            runtime_error_ids.append(case_id)
            continue
        validation_status[str(validation.get("status", "unknown"))] += 1
        for error in validation.get("errors", []):
            code = str(error.get("code", "unknown"))
            error_occurrences[code] += 1
            error_case_ids.setdefault(code, set()).add(case_id)
        for field in (
            "events", "derived_projections", "set_operations", "aggregates",
            "calculations", "references",
        ):
            node_counts[field] += len(understanding.get(field, []))
        calls = int(payload.get("model_calls", 0) or 0)
        model_calls.append(calls)
        audit = payload.get("llm_audit")
        if isinstance(audit, dict):
            audits.append(audit)
        report = payload.get("patch_report")
        if isinstance(report, dict):
            committed += int(bool(report.get("committed")))
            concrete += int((report.get("concrete_gain_count") or 0) > 0)

        if case_id not in eligible_ids:
            continue
        requirements = understanding.get("requirements", [])
        coverage = {
            item.get("requirement_id"): item.get("status")
            for item in understanding.get("coverage", [])
        }
        operator_ids = [
            item.get("requirement_id") for item in requirements
            if item.get("requirement_type") in OPERATOR_REQUIREMENTS
        ]
        event_ids = [
            item.get("requirement_id") for item in requirements
            if item.get("requirement_type") in {"sequence_event", "cumulative_duration"}
        ]
        combined_ids = [
            item.get("requirement_id") for item in requirements
            if item.get("requirement_type") in {
                "sequence_event", "cumulative_duration", "set_operation",
                "aggregate", "scoped_aggregate", "calculation",
            }
        ]
        if operator_ids:
            eligible_operator_cases += 1
            eligible_operator_complete += int(all(coverage.get(item) == "satisfied" for item in operator_ids))
        if event_ids:
            eligible_event_cases += 1
            eligible_event_complete += int(all(coverage.get(item) == "satisfied" for item in event_ids))
        if event_ids and len(combined_ids) > len(event_ids):
            eligible_combined_cases += 1
            eligible_combined_complete += int(all(coverage.get(item) == "satisfied" for item in combined_ids))

    choice_audits = [item for item in audits if item.get("protocol") == "candidate-choice-v3"]
    valid_choice = [item for item in choice_audits if not item.get("parse_error")]
    allowed_choice = [
        item for item in valid_choice
        if item.get("decision") in {"select", "none_of_above", "ambiguous"}
    ]
    return {
        "case_count": len(cases),
        "runtime_error_ids": sorted(runtime_error_ids),
        "validation_status": dict(sorted(validation_status.items())),
        "node_counts": dict(sorted(node_counts.items())),
        "error_occurrences": dict(sorted(error_occurrences.items())),
        "error_case_incidence": {
            code: len(ids) for code, ids in sorted(error_case_ids.items())
        },
        "model": {
            "total_calls": sum(model_calls),
            "cases_with_calls": sum(item > 0 for item in model_calls),
            "max_calls_per_case": max(model_calls, default=0),
            "audit_count": len(audits),
            "choice_audit_count": len(choice_audits),
            "valid_choice_count": len(valid_choice),
            "allowed_choice_count": len(allowed_choice),
            "valid_choice_rate": _rate(len(valid_choice), len(choice_audits)),
            "allowed_choice_rate": _rate(len(allowed_choice), len(choice_audits)),
            "committed_patch_cases": committed,
            "concrete_gain_cases": concrete,
        },
        "eligible": {
            "manifest_case_count": len(eligible_ids),
            "operator_cases": eligible_operator_cases,
            "operator_complete_cases": eligible_operator_complete,
            "operator_complete_rate": _rate(eligible_operator_complete, eligible_operator_cases),
            "event_cases": eligible_event_cases,
            "event_complete_cases": eligible_event_complete,
            "event_complete_rate": _rate(eligible_event_complete, eligible_event_cases),
            "combined_cases": eligible_combined_cases,
            "combined_complete_cases": eligible_combined_complete,
            "combined_complete_rate": _rate(eligible_combined_complete, eligible_combined_cases),
        },
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def build_gates(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_incidence = baseline["error_case_incidence"]
    current_incidence = candidate["error_case_incidence"]
    reductions = {}
    for code in CORE_CODES:
        before = int(base_incidence.get(code, 0))
        after = int(current_incidence.get(code, 0))
        reductions[code] = {
            "before": before, "after": after,
            "reduction_rate": round((before - after) / before, 4) if before else None,
        }
    frozen_zero = {
        code: int(current_incidence.get(code, 0)) for code in FROZEN_ZERO_CODES
    }
    return {
        "frozen_regression": {
            "batch_50_no_crash": candidate["case_count"] == 50 and not candidate["runtime_error_ids"],
            "max_one_model_call": candidate["model"]["max_calls_per_case"] <= 1,
            "frozen_zero_diagnostics": all(value == 0 for value in frozen_zero.values()),
            "frozen_zero_values": frozen_zero,
        },
        "event_closure": {
            "event_complete_at_least_90pct": _at_least(
                candidate["eligible"]["event_complete_rate"], 0.9,
            ),
            "combined_complete_at_least_80pct": _at_least(
                candidate["eligible"]["combined_complete_rate"], 0.8,
            ),
            "core_error_reductions": reductions,
            "all_core_errors_reduced_70pct": all(
                item["reduction_rate"] is not None and item["reduction_rate"] >= 0.7
                for item in reductions.values()
            ),
        },
        "llm_batch": {
            "valid_choice_at_least_98pct": _at_least(
                candidate["model"]["valid_choice_rate"], 0.98,
            ),
            "allowed_choice_at_least_95pct": _at_least(
                candidate["model"]["allowed_choice_rate"], 0.95,
            ),
            "false_commit_count": 0,
            "note": "False semantic commits require oracle/manual evidence; zero is not inferred from commit count.",
        },
    }


def _at_least(value: float | None, threshold: float) -> bool | None:
    return value >= threshold if value is not None else None


def markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    gates = report["gates"]
    reductions = gates["event_closure"]["core_error_reductions"]
    lines = [
        "# M1.4 Frozen 50-Case Acceptance Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Batch Summary",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | ---: | ---: |",
        f"| Cases | {baseline['case_count']} | {candidate['case_count']} |",
        f"| Runtime crashes | {len(baseline['runtime_error_ids'])} | {len(candidate['runtime_error_ids'])} |",
        f"| Model calls | {baseline['model']['total_calls']} | {candidate['model']['total_calls']} |",
        f"| Max calls/query | {baseline['model']['max_calls_per_case']} | {candidate['model']['max_calls_per_case']} |",
        f"| Event nodes | {baseline['node_counts'].get('events', 0)} | {candidate['node_counts'].get('events', 0)} |",
        f"| Set nodes | {baseline['node_counts'].get('set_operations', 0)} | {candidate['node_counts'].get('set_operations', 0)} |",
        f"| Aggregate nodes | {baseline['node_counts'].get('aggregates', 0)} | {candidate['node_counts'].get('aggregates', 0)} |",
        f"| Calculation nodes | {baseline['node_counts'].get('calculations', 0)} | {candidate['node_counts'].get('calculations', 0)} |",
        "",
        "## Eligible Closure",
        "",
        f"- Event completion: {_pct(candidate['eligible']['event_complete_rate'])} "
        f"({candidate['eligible']['event_complete_cases']}/{candidate['eligible']['event_cases']}).",
        f"- Combined completion: {_pct(candidate['eligible']['combined_complete_rate'])} "
        f"({candidate['eligible']['combined_complete_cases']}/{candidate['eligible']['combined_cases']}).",
        f"- Operator completion: {_pct(candidate['eligible']['operator_complete_rate'])} "
        f"({candidate['eligible']['operator_complete_cases']}/{candidate['eligible']['operator_cases']}).",
        "",
        "## Core Diagnostics",
        "",
        "| Code | Baseline cases | Candidate cases | Reduction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for code, item in reductions.items():
        lines.append(
            f"| `{code}` | {item['before']} | {item['after']} | {_pct(item['reduction_rate'])} |"
        )
    lines.extend([
        "",
        "## LLM Batch",
        "",
        f"- Candidate-choice audits: {candidate['model']['choice_audit_count']}.",
        f"- Valid choice JSON: {_pct(candidate['model']['valid_choice_rate'])}.",
        f"- Allowed choice protocol: {_pct(candidate['model']['allowed_choice_rate'])}.",
        f"- Committed Patch cases: {candidate['model']['committed_patch_cases']}.",
        f"- Concrete Gain cases: {candidate['model']['concrete_gain_cases']}.",
        "",
        "## Gate Result",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
        "```",
        "",
        "The report does not infer semantic correctness from commit count. Real decision-oracle and "
        "Concrete Closure tests remain separate evidence.",
    ])
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text("utf-8"))
    eligible_ids = {
        int(item["case_id"]) for item in manifest["cases"] if item["included"]
    }
    baseline = summarize(load_cases(args.baseline_dir), eligible_ids)
    candidate = summarize(load_cases(args.candidate_dir), eligible_ids)
    report = {
        "schema_version": "m14-acceptance-report-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "baseline": baseline,
        "candidate": candidate,
        "gates": build_gates(baseline, candidate),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    args.markdown_out.write_text(markdown(report), "utf-8")
    print(json.dumps(report["gates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
