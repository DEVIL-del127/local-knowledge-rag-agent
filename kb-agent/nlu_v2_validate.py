"""Standalone acceptance CLI for NLU V2. It never performs retrieval."""
from __future__ import annotations

import argparse
import json
import sys

from nlu_v2 import EngineConfig, QueryUnderstandingEngine
from nlu_v2.legacy_adapter import dual_run
from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider


def build_engine(enable_llm: bool = False, *, model: str | None = None,
                 ollama_url: str | None = None, allow_repair: bool = False,
                 review_complex: bool = False,
                 candidate_choice_v3: bool = False,
                 m15_shadow: bool = False) -> QueryUnderstandingEngine:
    extractor = None
    if enable_llm:
        extractor = BoundedLLMExtractor(
            OllamaOpenAIProvider(model=model, base_url=ollama_url),
            allow_json_repair=allow_repair,
        )
    return QueryUnderstandingEngine(
        config=EngineConfig(
            enable_llm=enable_llm, llm_on_complex=review_complex,
            enable_candidate_choice_v3=candidate_choice_v3,
            enable_m15_requirement_graph_shadow=m15_shadow,
            enable_m15_field_binding_shadow=m15_shadow,
            enable_m15_quantity_contract=m15_shadow,
            enable_m15_temporal_contracts=m15_shadow,
            enable_m15_typed_lineage=m15_shadow,
            enable_m15_logical_plan=m15_shadow,
        ),
        llm_extractor=extractor,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NLU V2 三层 IR 验证（不执行检索）")
    parser.add_argument("query", nargs="?", help="待解析问题")
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出完整 JSON")
    parser.add_argument("--legacy", action="store_true", help="显式启动旧 Router 并输出双轨比较")
    llm_mode = parser.add_mutually_exclusive_group()
    llm_mode.add_argument(
        "--llm", action="store_true", dest="llm",
        help="启用本地 Ollama 结构化兜底（默认启用）",
    )
    llm_mode.add_argument(
        "--no-llm", action="store_false", dest="llm",
        help="关闭 Ollama，仅运行本地规则和验证",
    )
    parser.set_defaults(llm=True)
    parser.add_argument("--llm-model", default=None, help="Ollama NLU 模型，默认 qwen2.5:7b")
    parser.add_argument(
        "--ollama-url", default=None,
        help="Ollama 地址；默认读取 KB_OLLAMA_URL，并自动识别 WSL 宿主机",
    )
    parser.add_argument(
        "--llm-repair", action="store_true",
        help="仅 legacy Atomic 协议允许一次 JSON 修复；Candidate Choice v3 始终零重试",
    )
    parser.add_argument(
        "--candidate-choice-v3", action="store_true",
        help="显式启用尚未通过批量发布 Gate 的 Candidate Choice v3",
    )
    parser.add_argument(
        "--m15-shadow", action="store_true",
        help="附加 M1.5 类型闭环审计视图；不替换当前 IR/LogicalPlan",
    )
    parser.add_argument(
        "--legacy-atomic-patch", action="store_true",
        help="回退到旧 Atomic Patch 模型输出协议",
    )
    parser.add_argument("--llm-review-complex", action="store_true", help="规则已完整时也调用一次 LLM 复核")
    parser.add_argument("--llm-health", action="store_true", help="检查 Ollama 和模型后退出")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.llm_health:
        try:
            health = OllamaOpenAIProvider(
                model=args.llm_model, base_url=args.ollama_url
            ).health()
            print(json.dumps(health, ensure_ascii=False, indent=2))
            return 0 if health["available"] else 2
        except Exception as exc:
            print(json.dumps({
                "available": False, "error": f"{type(exc).__name__}: {exc}"
            }, ensure_ascii=False, indent=2))
            return 2

    query = args.query or input("问题> ").strip()
    if not query:
        parser.error("query 不能为空")
    engine = build_engine(
        args.llm, model=args.llm_model, ollama_url=args.ollama_url,
        allow_repair=args.llm_repair, review_complex=args.llm_review_complex,
        candidate_choice_v3=args.candidate_choice_v3 and not args.legacy_atomic_patch,
        m15_shadow=args.m15_shadow,
    )
    try:
        if args.legacy:
            legacy_router = None
            legacy_error = ""
            try:
                from router import Router
                legacy_router = Router()
            except Exception as exc:
                legacy_error = f"{type(exc).__name__}: {exc}"
            payload = dual_run(query, engine, legacy_router).to_dict()
            if legacy_error:
                payload["comparison"]["legacy_error"] = legacy_error
        else:
            payload = engine.analyze(query).to_dict()
    except Exception as exc:
        error = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(error, ensure_ascii=False, indent=2) if args.as_json else error["error"])
        return 2
    if args.as_json or args.legacy:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_summary(payload)
    return 0


def _print_summary(payload: dict) -> None:
    understanding = payload["understanding"]
    validation = payload["validation"]
    logical = payload["logical_plan"]
    physical = payload["physical_plan"]
    print("\n── RequestIR / UnderstandingIR ──")
    print(f"兼容意图 : {understanding['compatibility_intent']}")
    print(f"数据源候选: {[item['identifier'] for item in understanding['source_candidates']]}")
    print(f"Schema假设: {_hypothesis_summary(understanding.get('schema_hypotheses', []))}")
    print(f"投影字段 : {[item['canonical_id'] or item['raw_name'] for item in understanding['projections']]}")
    print(f"指标     : {_metric_summary(understanding['metrics'])}")
    print(f"条件     : {_predicate_summary(understanding['filters'])}")
    print(f"时间窗口 : {_temporal_summary(understanding['temporal'])}")
    print(f"采样策略 : {_sampling_summary(understanding.get('sampling_policies', []))}")
    print(f"时序事件 : {_event_summary(understanding['events'])}")
    print(f"派生投影 : {_projection_summary(understanding.get('derived_projections', []))}")
    print(f"集合运算 : {_set_summary(understanding['set_operations'])}")
    print(f"作用域聚合: {_aggregate_summary(understanding.get('aggregates', []))}")
    print(f"结果比较 : {_comparison_summary(understanding.get('comparisons', []))}")
    print(f"算术表达式: {_arithmetic_summary(understanding.get('arithmetic', []))}")
    print(f"有序序列 : {_sequence_summary(understanding.get('sequences', []))}")
    print(f"状态递推 : {_ordered_fold_summary(understanding.get('ordered_folds', []))}")
    print(f"算子调用 : {_invocation_summary(understanding.get('operator_invocations', []))}")
    print(f"计算     : {_calculation_summary(understanding['calculations'])}")
    print(f"未解析   : {[item['message'] for item in understanding['unresolved']] or '（无）'}")
    claim_counts = {}
    for item in understanding.get("claims", []):
        key = f"{item['claim_type']}:{item['status']}"
        claim_counts[key] = claim_counts.get(key, 0) + 1
    requirements = understanding.get("requirements", [])
    coverage = understanding.get("coverage", [])
    covered = sum(item.get("status") == "satisfied" for item in coverage)
    print(f"语义声明 : {claim_counts or '（无）'}")
    uncovered = [item for item in coverage if item.get("status") != "satisfied"]
    print(f"需求覆盖 : {covered}/{len(requirements)}")
    if uncovered:
        print("覆盖差分 : " + "; ".join(
            f"{item.get('requirement_id')}:{','.join(item.get('missing_attributes', []))}"
            for item in uncovered
        ))
    print(f"理解状态 : {understanding.get('understanding_status', 'unknown')}")
    print(f"绑定状态 : {understanding.get('binding_status', 'unknown')}")
    print(f"能力状态 : {understanding.get('capability_status', 'unknown')}")
    print(f"执行状态 : {understanding.get('execution_status', 'unknown')}")
    print("\n── Validation ──")
    print(f"状态     : {validation['status']}")
    print(f"可执行   : {validation['executable']}")
    print(f"可靠度   : {validation['reliability']:.3f}（证据推导，不是模型自评）")
    dimensions = validation.get("dimensions", {})
    dimension_statuses = {
        name: value.get("status") for name, value in dimensions.items()
    }
    print(f"多维校验 : {dimension_statuses}")
    if validation["errors"]:
        print(f"问题     : {[item['message'] for item in validation['errors']]}")
    print("\n── LogicalPlan ──")
    print(f"状态     : {logical['status']}")
    for node in logical["nodes"]:
        print(f"{node['task_id']}: {node['task_type']} <- {node['depends_on']}")
    print("\n── PhysicalPlan ──")
    print(f"状态     : {physical['status']}（等待根 Agent SkillRegistry 绑定）")
    print(f"模型调用 : {payload['model_calls']} 次")
    patch_report = payload.get("patch_report")
    if patch_report:
        print(
            "语义Patch: "
            f"accepted={len(patch_report.get('accepted', []))}, "
            f"rejected={len(patch_report.get('rejected', []))}, "
            f"closed_gaps={len(patch_report.get('closed_gap_ids', []))}, "
            f"committed={patch_report.get('committed', False)}"
        )
        if patch_report.get("transaction_error"):
            print(f"Patch状态: {patch_report['transaction_error']}")
        if patch_report.get("rejection_gate"):
            print(f"拒绝 Gate: {patch_report['rejection_gate']}")
        rejected = patch_report.get("rejected", [])
        if rejected:
            print("Patch拒绝 : " + "; ".join(
                f"{item.get('operation_type')}={item.get('reason')}" for item in rejected
            ))
    llm_audit = payload.get("llm_audit")
    if llm_audit:
        print(
            "模型审计 : "
            f"model={llm_audit.get('model') or 'unknown'}, "
            f"operation={llm_audit.get('atomic_operation') or 'legacy'}, "
            f"prompt={llm_audit.get('prompt_chars', 0)} chars, "
            f"response={llm_audit.get('response_chars', 0)} chars, "
            f"elapsed={llm_audit.get('duration_ms', 0):.1f} ms, "
            f"stop={llm_audit.get('stop_reason') or 'unknown'}"
        )
        if llm_audit.get("parsed_json_excerpt"):
            print(f"解析 JSON: {llm_audit['parsed_json_excerpt']}")
        if llm_audit.get("rejected_gate"):
            print(f"审计 Gate: {llm_audit['rejected_gate']}")
        if llm_audit.get("parse_error"):
            print(f"解析错误 : {llm_audit['parse_error']}")
            print(f"原始响应 : {llm_audit.get('raw_response_excerpt') or '（无响应正文）'}")
    effect = payload.get("semantic_effect")
    if effect:
        print(
            "语义效果 : "
            f"profile={effect.get('acceptance_profile')}, "
            f"closed_requirements={len(effect.get('closed_requirement_ids', []))}, "
            f"reason={effect.get('commit_or_reject_reason')}"
        )
    closure = payload.get("event_closure_report")
    if closure:
        print(
            "事件闭环 : "
            f"structure={closure.get('structure_status')}, "
            f"binding={closure.get('binding_status')}, "
            f"execution={closure.get('execution_status')}, "
            f"edges={len(closure.get('producer_consumer_edges', []))}, "
            f"unresolved={len(closure.get('unresolved_dependencies', []))}"
        )
    clarification = payload.get("clarification_plan")
    if clarification:
        questions = clarification.get("questions", [])
        print("澄清状态 : " + clarification.get("state", {}).get("reason_kind", "pending"))
        for question in questions:
            suffix = f"（候选：{', '.join(question.get('candidates', []))}）" if question.get("candidates") else ""
            print(f"澄清问题 : {question.get('prompt', '')}{suffix}")
    print("轨迹     : " + " -> ".join(item["stage"] for item in payload["trace"]))


def _metric_summary(metrics: list[dict]) -> str | list[str]:
    values = [
        f"{item['aggregation']}({item['field']['canonical_id'] or item['field']['raw_name']})"
        for item in metrics
    ]
    return values or "（无）"


def _arithmetic_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['arithmetic_id']}={item['expression'].get('operator', item['expression'].get('kind'))}"
            for item in items] or "（无）"


def _sequence_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['sequence_id']}({len(item.get('steps', []))} steps)" for item in items] or "（无）"


def _ordered_fold_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['fold_id']}={'resettable' if item['transition'].get('reset_condition') else 'ordered'}"
            for item in items] or "（无）"


def _invocation_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['operator_id']}[{','.join(item.get('requirement_ids', []))}]" for item in items] or "（无）"


def _hypothesis_summary(items: list[dict]) -> str | list[str]:
    return [
        f"{item['raw_name']} -> {item['hypothesis_id']} ({item['declared_type']}, 不可执行)"
        for item in items
    ] or "（无）"


def _sampling_summary(items: list[dict]) -> str | list[str]:
    return [f"interval={item.get('expected_interval_seconds')}s, missing={item.get('missing_data_policy')}"
            for item in items] or "（无）"


def _projection_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['function']}({item['input_ref']['producer_id']}) -> {item['output']['producer_id']}"
            for item in items] or "（无）"


def _aggregate_summary(items: list[dict]) -> str | list[str]:
    return [
        f"{item['aggregate']['aggregate_id']}: {item['aggregate']['function']} "
        f"scope={item['scope']} -> {item['aggregate']['output']['producer_id']}"
        for item in items
    ] or "（无）"


def _comparison_summary(items: list[dict]) -> str | list[str]:
    return [f"{item['comparison_id']}: {item['operator']} -> {item['output']['producer_id']}"
            for item in items] or "（无）"


def _predicate_summary(node: dict | None) -> str:
    if not node:
        return "（无）"
    if node.get("kind") == "boolean":
        parts = [_predicate_summary(item) for item in node.get("children", [])]
        return "(" + f" {node.get('operator', 'and').upper()} ".join(parts) + ")"
    field = node.get("field") or {}
    value = node.get("value") or {}
    unit = f" {value.get('unit')}" if value.get("unit") else ""
    return (
        f"{field.get('canonical_id') or field.get('raw_name') or '?'} "
        f"{node.get('operator') or '?'} {value.get('value')}{unit}"
    )


def _temporal_summary(items: list[dict]) -> str | list[str]:
    values = []
    for item in items:
        if item.get("exact"):
            value = item["exact"]
        else:
            value = f"{item.get('from_value') or '?'} ~ {item.get('to_value') or '?'}"
        values.append(f"{value} [{item.get('granularity', '')}]")
    return values or "（无）"


def _event_summary(items: list[dict]) -> str | list[str]:
    values = []
    for item in items:
        metric = item["derived_metric"]["metric_id"].rsplit(".", 1)[-1]
        threshold = item["threshold"]
        values.append(
            f"{item['event_id']}: {metric}({_predicate_summary(item['condition'])}) "
            f"{threshold['operator']} {threshold['value']}{threshold['unit']} -> {item['output_name']}"
        )
    return values or "（无）"


def _set_summary(items: list[dict]) -> str | list[str]:
    values = [
        f"{item['operation']}({', '.join(item['inputs'])}) -> {item['output_name']}"
        for item in items
    ]
    return values or "（无）"


def _calculation_summary(items: list[dict]) -> str | list[str]:
    values = [
        f"{item['calculation_type']}: {item['expression']}"
        for item in items
    ]
    return values or "（无）"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
