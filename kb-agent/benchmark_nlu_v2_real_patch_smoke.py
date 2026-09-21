"""Run bounded real-qwen atomic Patch smoke cases after the offline gate.

This is deliberately not part of unit-test discovery: it contacts the locally
configured Ollama endpoint only when invoked explicitly.  Every case makes one
operation-specific request and never performs retrieval or PhysicalPlan binding.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider
from nlu_v2.patch_protocol import GapRequest, request_ir_digest


EVENT_QUERY = "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
SET_QUERY = (
    "监测设备的温度传感器。找出 2026年1月1日至今，温度连续超过 85℃ "
    "超过 10分钟 的所有时间段，以及 累计超过 80℃ 但未超过 85℃ 的总时长"
    "超过 6小时 的所有日期。输出这两类异常时间段的交集日期。"
)
AGGREGATE_QUERY = "设备传感器，计算这些记录的温度平均值。"
CALCULATION_QUERY = "设备传感器，计算温度的标准差。"
TURN_QUERY = "请将上一轮的论文查询改为英文，并且只保留有代码的论文。"


@dataclass(frozen=True, slots=True)
class SmokeCase:
    name: str
    query: str
    gap_type: str
    operations: tuple[str, ...]


CASES = (
    SmokeCase("event", EVENT_QUERY, "missing_event", ("add_event",)),
    SmokeCase("set", SET_QUERY, "missing_set_inputs", ("add_set_operation",)),
    SmokeCase("aggregate", AGGREGATE_QUERY, "missing_aggregate_scope", ("add_aggregate",)),
    SmokeCase("calculation", CALCULATION_QUERY, "missing_formula_input", ("add_calculation",)),
    SmokeCase("turn", TURN_QUERY, "missing_turn_directive", ("add_turn_directive",)),
)


def _prepare(engine: QueryUnderstandingEngine, case: SmokeCase, catalog):
    base = engine.analyze(case.query).understanding
    # Remove exactly one deterministic product, then let CoverageDiff and the
    # GapAnalyzer rebuild the real work order.  This preserves Requirement IDs,
    # source anchors, clauses and typed choice domains used in production.
    if case.name == "event":
        base.events = []
    elif case.name == "set":
        base.set_operations = []
    elif case.name == "aggregate":
        base.aggregates = []
    elif case.name == "calculation":
        base.calculations = []
    elif case.name == "turn":
        base.turn_directives = []
    engine.coverage_matcher.apply(base)
    gaps = engine.gap_analyzer.analyze(base, catalog)
    matching = [
        gap for gap in gaps
        if gap.gap_type == case.gap_type
        and any(operation in gap.allowed_operations for operation in case.operations)
    ]
    gap = next((item for item in matching if item.readiness == "ready"),
               matching[0] if matching else None)
    if gap is None:
        raise RuntimeError(f"no {case.gap_type} GapRequest generated for {case.name}")
    return base, gap


def run_case(case: SmokeCase, *, model: str, ollama_url: str | None) -> dict:
    engine = QueryUnderstandingEngine()
    catalog = engine.catalog_provider.snapshot()
    base, gap = _prepare(engine, case, catalog)
    extractor = BoundedLLMExtractor(
        OllamaOpenAIProvider(model=model, base_url=ollama_url),
        max_remote_attempts=1, allow_json_repair=False,
    )
    candidate = extractor.extract(
        base.query.normalized, engine._catalog_payload(catalog, base),
        engine._existing_ir_summary(base), gaps=[gap],
        base_ir_digest=request_ir_digest(base), enable_patch_v1=True,
        enable_atomic_patch_v2=True,
    )
    if candidate.patch is None:
        return {
            "case": case.name, "model_calls": candidate.attempts,
            "schema_valid": candidate.valid_json, "committed": False,
            "error": candidate.error or "empty_or_unparsed_patch",
            "raw_response_excerpt": candidate.raw_response_excerpt,
            "parsed_json_excerpt": candidate.parsed_json_excerpt,
        }
    _, report = engine._apply_patch_transaction(base, candidate.patch, [gap], catalog)
    return {
        "case": case.name, "model_calls": candidate.attempts,
        "schema_valid": candidate.valid_json, "protocol": candidate.protocol,
        "parsed_operation": [
            item.model_dump(mode="json") for item in candidate.patch.operations
        ],
        "accepted": report.accepted_count, "rejected": report.rejected_count,
        "rejection_reasons": [item.reason for item in report.rejected],
        "closed_gaps": report.closed_gap_count, "committed": report.committed,
        "rejection_gate": report.rejection_gate,
        "transaction_error": report.transaction_error,
        "coverage_before": (
            report.semantic_effect.coverage_before if report.semantic_effect else {}
        ),
        "coverage_after": (
            report.semantic_effect.coverage_after if report.semantic_effect else {}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--case", action="append", choices=[item.name for item in CASES])
    args = parser.parse_args()
    wanted = set(args.case or [item.name for item in CASES])
    reports = [run_case(item, model=args.model, ollama_url=args.ollama_url)
               for item in CASES if item.name in wanted]
    print(json.dumps({"model": args.model, "reports": reports}, ensure_ascii=False, indent=2))
    return 0 if all(item["committed"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
