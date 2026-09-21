"""Generate and validate the pre-production M1.5 Evidence Freeze artifacts.

The generator reads recorded M1.4 batch JSON only. It does not import NLU
production modules, CoverageMatcher, candidate compilation, or validators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASE_RE = re.compile(r"case_(\d{3})_")
FIXTURE_NAMES = (
    "m15_obligation_manifest.json",
    "m15_catalog_contract_matrix.json",
    "m15_operator_output_oracle.json",
    "m15_llm_choice_oracle.json",
    "m15_deterministic_baseline.json",
    "m15_acceptance_schema.json",
)
SUPPORTED_REQUIREMENTS = {
    "sequence_event",
    "cumulative_duration",
    "cumulative_aggregate",
    "derived_projection",
    "set_operation",
    "aggregate",
    "scoped_aggregate",
    "calculation",
    "formula",
    "reference",
}
SUPPORTED_CALCULATIONS = {
    "avg", "sum", "min", "max", "count", "stddev", "volatility",
    "ratio", "difference", "correlation", "cumulative_duration_ratio",
}
FIRST_FAILURE = {
    "sequence_event": "event_structure",
    "cumulative_duration": "duration_accumulation_structure",
    "cumulative_aggregate": "cumulative_value_structure",
    "derived_projection": "projection",
    "set_operation": "set_lineage",
    "aggregate": "aggregate_scope_grain",
    "scoped_aggregate": "aggregate_scope_grain",
    "calculation": "calculation_expression",
    "formula": "calculation_expression",
    "reference": "reference_lineage",
}
OPERATOR_MAP = {
    "sequence_event": "consecutive",
    "cumulative_duration": "cumulative_duration",
    "cumulative_aggregate": "cumulative_value",
    "derived_projection": "projection",
    "set_operation": "set_operation",
    "aggregate": "aggregate",
    "scoped_aggregate": "scoped_aggregate",
    "calculation": "calculation",
    "formula": "calculation",
    "reference": "reference",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_blob(payload))


def _json_blob(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _load_cases(source_dir: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    cases = []
    for path in sorted(source_dir.glob("case_*.json")):
        match = CASE_RE.match(path.name)
        if not match:
            continue
        cases.append((int(match.group(1)), path, json.loads(path.read_text("utf-8"))))
    if [item[0] for item in cases] != list(range(1, 51)):
        raise ValueError("source directory must contain case_001 through case_050 exactly once")
    return cases


def _requirement_span(requirement: dict[str, Any], demands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if requirement.get("span"):
        return requirement["span"]
    demand_spans = [
        demands[item]["span"]
        for item in requirement.get("expected_attributes", {}).get("source_demand_ids", [])
        if item in demands and demands[item].get("span")
    ]
    if not demand_spans:
        return {"start": -1, "end": -1, "text": requirement.get("text", "")}
    start = min(item["start"] for item in demand_spans)
    end = max(item["end"] for item in demand_spans)
    return {"start": start, "end": end, "text": requirement.get("text", "")}


def _calculation_name(requirement: dict[str, Any]) -> str:
    attrs = requirement.get("expected_attributes", {})
    return str(attrs.get("function") or requirement.get("operator_family") or "").lower()


def _scope(requirement: dict[str, Any]) -> tuple[bool, str]:
    kind = requirement.get("requirement_type", "")
    if kind not in SUPPORTED_REQUIREMENTS:
        return False, "operator_not_in_m15_scope"
    if kind in {"calculation", "formula"}:
        name = _calculation_name(requirement)
        if name not in SUPPORTED_CALCULATIONS:
            return False, f"unsupported_formula:{name or 'unknown'}"
    if kind == "derived_projection" and requirement.get("operator_family") not in {
        "period_duration", "projection", "date_projection", "interval_projection",
    }:
        return False, "unsupported_projection"
    return True, ""


def _output_role(requirement: dict[str, Any]) -> str:
    kind = requirement.get("requirement_type", "")
    text = requirement.get("text", "")
    if kind == "sequence_event":
        return "event_interval"
    if kind == "cumulative_duration":
        return "duration_value"
    if kind == "cumulative_aggregate":
        return "quantity_value"
    if kind == "set_operation":
        return "date_set" if "日期" in text else "interval_set"
    if kind == "derived_projection":
        return "duration_value" if "时长" in text else "date_projection"
    if kind in {"aggregate", "scoped_aggregate", "calculation", "formula"}:
        return "numeric_value"
    return "reference_target"


def _metric_membership(requirement: dict[str, Any], included: bool) -> list[str]:
    if not included:
        return []
    kind = requirement.get("requirement_type", "")
    values = {
        "sequence_event": ["event_structure"],
        "cumulative_duration": ["duration_accumulation"],
        "cumulative_aggregate": ["cumulative_value"],
        "derived_projection": ["event_projection_set"],
        "set_operation": ["event_projection_set"],
        "aggregate": ["full_combined"],
        "scoped_aggregate": ["full_combined"],
        "calculation": ["full_combined"],
        "formula": ["full_combined"],
        "reference": ["full_combined"],
    }.get(kind, [])
    return values + [f"diagnostic:{_diagnostic_code(kind)}"]


def _diagnostic_code(kind: str) -> str:
    if kind == "sequence_event":
        return "event_structure_incomplete"
    if kind in {"derived_projection", "set_operation"}:
        return "set_inputs_unbound"
    if kind in {"aggregate", "scoped_aggregate", "calculation", "formula", "reference"}:
        return "formula_input_unbound"
    if kind == "cumulative_duration":
        return "duration_accumulation_incomplete"
    if kind == "cumulative_aggregate":
        return "cumulative_value_incomplete"
    return "semantic_structure_incomplete"


def build_obligation_manifest(cases: list[tuple[int, Path, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    source_cases = []
    for case_id, path, payload in cases:
        ir = payload.get("understanding", {})
        requirements = ir.get("requirements", [])
        requirement_by_id = {item["requirement_id"]: item for item in requirements}
        demand_by_id = {item["demand_id"]: item for item in ir.get("source_demands", [])}
        coverage = {item["requirement_id"]: item for item in ir.get("coverage", [])}
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        source_cases.append({"case_id": case_id, "artifact": path.name, "sha256": source_sha})
        for requirement in requirements:
            kind = requirement.get("requirement_type", "")
            if kind not in SUPPORTED_REQUIREMENTS:
                continue
            source_ids = sorted(set(
                requirement.get("expected_attributes", {}).get("source_demand_ids", [])
            ))
            dependency_ids = requirement.get("dependencies", [])
            dependency_demand_ids = sorted({
                demand_id
                for dependency_id in dependency_ids
                for demand_id in requirement_by_id.get(dependency_id, {})
                .get("expected_attributes", {}).get("source_demand_ids", [])
            })
            canonical_operator = OPERATOR_MAP[kind]
            if kind in {"calculation", "formula"}:
                canonical_operator = _calculation_name(requirement) or canonical_operator
            role = _output_role(requirement)
            grain = requirement.get("expected_output_grain") or (
                "date" if role in {"date_set", "date_projection"} else
                "interval" if role in {"event_interval", "interval_set"} else "scalar"
            )
            identity = {
                "case_id": case_id,
                "clause_id": requirement.get("clause_id", ""),
                "source_demand_ids": source_ids,
                "operator": canonical_operator,
                "dependency_demand_ids": dependency_demand_ids,
                "output_role": role,
                "grain": grain,
            }
            obligation_id = "obl:sha256:" + _digest(identity)
            included, exclusion = _scope(requirement)
            current_coverage = coverage.get(requirement["requirement_id"], {})
            span = _requirement_span(requirement, demand_by_id)
            row = grouped.get(obligation_id)
            if row is None:
                grouped[obligation_id] = {
                    "obligation_id": obligation_id,
                    "case_id": case_id,
                    "source_sha256": source_sha,
                    "clause_id": requirement.get("clause_id", ""),
                    "source_spans": [span],
                    "source_demand_ids": source_ids,
                    "raw_requirement_ids": [requirement["requirement_id"]],
                    "operator_family": kind,
                    "canonical_operator": canonical_operator,
                    "dependency_demand_ids": dependency_demand_ids,
                    "dependency_obligation_ids": [],
                    "output_role": role,
                    "grain": grain,
                    "in_scope": included,
                    "exclusion_code": exclusion,
                    "expected_readiness_level": (
                        "structure_complete_binding_pending" if included else "out_of_scope"
                    ),
                    "expected_primary_failure": (
                        "none" if current_coverage.get("status") == "satisfied"
                        else FIRST_FAILURE[kind]
                    ),
                    "expected_secondary_diagnostics": sorted(set(
                        current_coverage.get("missing_attributes", [])
                    )),
                    "baseline_coverage_status": current_coverage.get("status", "missing"),
                    "metric_membership": _metric_membership(requirement, included),
                    "oracle_ref": f"operator:{canonical_operator}",
                }
            else:
                row["raw_requirement_ids"].append(requirement["requirement_id"])
                if span not in row["source_spans"]:
                    row["source_spans"].append(span)

    obligations = sorted(grouped.values(), key=lambda item: (item["case_id"], item["source_spans"][0]["start"], item["obligation_id"]))
    for item in obligations:
        item["raw_requirement_ids"] = sorted(set(item["raw_requirement_ids"]))
        item["source_spans"] = sorted(item["source_spans"], key=lambda value: (value["start"], value["end"], value["text"]))
    return {
        "schema_version": "m15-obligation-manifest-v1",
        "source": "recorded M1.4 50-case artifacts; independent of production compiler and Coverage",
        "first_failure_precedence": [
            "authority_evidence", "source_field_binding", "quantity_unit_typing",
            "predicate_typing", "event_or_accumulation_structure", "source_capability_policy",
            "projection", "set_lineage", "aggregate_scope_grain",
            "calculation_expression", "logical_plan_preflight", "optional_llm_decision",
        ],
        "metric_formulas": {
            "completion_rate": "oracle_correct_completed_obligations / frozen_included_obligations",
            "reduction_rate": "(baseline_failures - candidate_failures) / baseline_failures",
            "zero_denominator": "not_evaluated",
        },
        "source_cases": source_cases,
        "obligations": obligations,
    }


def build_catalog_matrix() -> dict[str, Any]:
    base_fields = {
        "device": [
            {"id": "telemetry.observed_at", "type": "datetime", "role": "timestamp"},
            {"id": "telemetry.device_id", "type": "keyword", "role": "partition"},
            {"id": "telemetry.temperature", "aliases": ["温度", "temp"], "type": "number", "unit": "celsius"},
            {"id": "telemetry.acceleration", "aliases": ["加速度"], "type": "number", "unit": "m/s2"},
        ],
        "manufacturing": [
            {"id": "machine.observed_at", "type": "datetime", "role": "timestamp"},
            {"id": "machine.machine_id", "type": "keyword", "role": "partition"},
            {"id": "machine.vibration", "aliases": ["振动", "振动幅度"], "type": "number", "unit": "mm/s"},
        ],
        "market": [
            {"id": "prices.trade_date", "type": "date", "role": "timestamp"},
            {"id": "prices.symbol", "type": "keyword", "role": "partition"},
            {"id": "prices.close", "aliases": ["收盘价", "price"], "type": "number", "unit": "currency:CNY"},
        ],
    }

    def row(identifier: str, group: str, variant: str, *, capabilities: list[str],
            sampling: dict[str, Any], policies: dict[str, Any], expected: str) -> dict[str, Any]:
        return {
            "contract_id": identifier,
            "matched_group": group,
            "variant": variant,
            "read_only": True,
            "fields": base_fields[group],
            "timestamp": next(item["id"] for item in base_fields[group] if item.get("role") == "timestamp"),
            "partition_keys": [next(item["id"] for item in base_fields[group] if item.get("role") == "partition")],
            "ordering_key": next(item["id"] for item in base_fields[group] if item.get("role") == "timestamp"),
            "timezone": "Asia/Shanghai",
            "sampling": sampling,
            "policies": policies,
            "capabilities": capabilities,
            "expected_readiness_level": expected,
        }

    complete_policies = {
        "max_gap": "1.5 * sampling_interval",
        "missing_data": "break_event",
        "boundary": "source_predicate_operator",
        "timezone_provenance": "source_contract",
    }
    rows = [
        row("telemetry-ready", "device", "positive",
            capabilities=["ordered_scan", "event_segmentation", "date_projection", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 2},
            policies=complete_policies, expected="logical_plan_ready"),
        row("telemetry-no-segmentation", "device", "missing_capability",
            capabilities=["ordered_scan", "date_projection", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 2},
            policies=complete_policies, expected="event_capability_pending"),
        row("telemetry-no-gap-policy", "device", "missing_policy",
            capabilities=["ordered_scan", "event_segmentation", "date_projection", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 2},
            policies={key: value for key, value in complete_policies.items() if key != "max_gap"},
            expected="event_policy_pending"),
        row("machine-duration-ready", "manufacturing", "positive",
            capabilities=["ordered_scan", "event_segmentation", "duration_accumulation", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 0.1},
            policies={**complete_policies, "integration_method": "sampled_duration_next_interval"},
            expected="logical_plan_ready"),
        row("machine-no-duration", "manufacturing", "missing_capability",
            capabilities=["ordered_scan", "event_segmentation", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 0.1},
            policies={**complete_policies, "integration_method": "sampled_duration_next_interval"},
            expected="duration_capability_pending"),
        row("machine-no-integration-policy", "manufacturing", "missing_policy",
            capabilities=["ordered_scan", "event_segmentation", "duration_accumulation", "aggregate"],
            sampling={"kind": "regular", "interval_seconds": 0.1},
            policies=complete_policies, expected="duration_policy_pending"),
        row("market-window-ready", "market", "positive",
            capabilities=["ordered_scan", "window_aggregate", "market_calendar", "event_segmentation"],
            sampling={"kind": "calendar_event", "calendar": "XSHG"},
            policies={**complete_policies, "window_semantics": "trading_day"},
            expected="logical_plan_ready"),
        row("market-no-calendar", "market", "missing_capability",
            capabilities=["ordered_scan", "window_aggregate", "event_segmentation"],
            sampling={"kind": "calendar_event", "calendar": None},
            policies={**complete_policies, "window_semantics": "trading_day"},
            expected="window_capability_pending"),
        row("market-no-window-policy", "market", "missing_policy",
            capabilities=["ordered_scan", "window_aggregate", "market_calendar", "event_segmentation"],
            sampling={"kind": "calendar_event", "calendar": "XSHG"},
            policies=complete_policies, expected="window_policy_pending"),
    ]
    return {"schema_version": "m15-catalog-contract-matrix-v1", "contracts": rows}


def build_operator_oracle() -> dict[str, Any]:
    return {
        "schema_version": "m15-operator-output-oracle-v1",
        "operator_type_matrix": [
            {"operator": "consecutive", "output": "relation<interval>"},
            {"operator": "cumulative_duration", "output": "relation<group_key,duration>"},
            {"operator": "cumulative_value", "output": "relation<group_key,quantity>"},
            {"operator": "project_local_date", "input": "relation<interval>", "output": "relation<date>"},
            {"operator": "set_intersection", "input": "two relations with identical element type/grain", "output": "relation<same>"},
        ],
        "case_dags": [
            {"case_id": 24, "nodes": ["ConsecutiveEvent", "ConsecutiveEvent", "SetIntersection<interval>", "DurationAccumulation<day,duration>", "PeriodDuration<day,duration>", "Ratio<number>"], "forbidden": ["CumulativeDuration->relation<interval>"]},
            {"case_id": 27, "nodes": ["ConsecutiveEvent", "CumulativeValue<day,distance>", "ProjectDate", "SetIntersection<date>", "Ratio<number>"], "forbidden": ["Distance->DurationAccumulation"]},
            {"case_id": 47, "nodes": ["ConsecutiveEvent", "ConsecutiveEvent", "SetIntersection<interval>", "DurationAccumulation<day,duration>", "PeriodDuration<day,duration>", "Ratio<number>"], "forbidden": ["SameProducerSet", "CumulativeDuration->relation<interval>"]},
        ],
        "negative_bindings": [
            {"name": "duration_as_interval", "producer": "DurationAccumulation", "consumer": "SetIntersection<interval>", "expected": "reject"},
            {"name": "distance_as_duration", "producer": "CumulativeValue<distance>", "consumer": "DurationRatio", "expected": "reject"},
            {"name": "same_set_producer", "producer_refs": ["event_a", "event_a"], "expected": "reject"},
            {"name": "mixed_set_grain", "producer_types": ["relation<date>", "relation<interval>"], "expected": "reject"},
        ],
        "date_projection_cases": [
            {"name": "cross_midnight", "interval": "[2026-08-01T23:59+08:00,2026-08-02T00:01+08:00)", "dates": ["2026-08-01", "2026-08-02"]},
            {"name": "exact_midnight_end", "interval": "[2026-08-01T23:00+08:00,2026-08-02T00:00+08:00)", "dates": ["2026-08-01"]},
            {"name": "window_clip", "interval": "[2026-07-31T23:00+08:00,2026-08-02T01:00+08:00)", "query_window": "[2026-08-01T00:00+08:00,2026-08-02T00:00+08:00)", "dates": ["2026-08-01"]},
            {"name": "deduplicate", "intervals": ["[2026-08-01T01:00+08:00,2026-08-01T02:00+08:00)", "[2026-08-01T03:00+08:00,2026-08-01T04:00+08:00)"], "dates": ["2026-08-01"]},
            {"name": "dst_naive_rejected", "timezone": "America/New_York", "interval": "[2026-11-01T01:15,2026-11-01T01:45)", "expected": "timezone_ambiguity"},
        ],
        "quantity_cases": [
            {"name": "absolute_celsius", "kind": "absolute_temperature", "transform": "affine", "expected": "supported"},
            {"name": "celsius_delta", "kind": "temperature_delta", "transform": "linear", "expected": "supported"},
            {"name": "db_same_semantic", "kind": "sound_level", "transform": "logarithmic", "expected": "same_unit_comparison_only"},
            {"name": "db_scalar_conversion", "kind": "sound_level", "operation": "multiply_scale", "expected": "reject"},
        ],
    }


def _candidate(summary: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"summary": summary, "semantic_payload": payload, "payload_digest": _digest(payload)}


def _choice_rows() -> list[dict[str, Any]]:
    specs: dict[str, list[tuple[str, str, list[tuple[str, dict[str, Any]]], str, int | None]]] = {
        "event": [
            ("temp_high", "温度连续超过85℃并持续10分钟", [("温度>85℃连续10分钟", {"op":"consecutive","field":"temperature","cmp":"gt","value":85,"unit":"celsius","duration_s":600}), ("温度<85℃连续10分钟", {"op":"consecutive","field":"temperature","cmp":"lt","value":85,"unit":"celsius","duration_s":600})], "select", 0),
            ("accel_high", "加速度连续超过2.5m/s²并持续5秒", [("加速度>2.5m/s²连续5秒", {"op":"consecutive","field":"acceleration","cmp":"gt","value":2.5,"unit":"m/s2","duration_s":5}), ("加速度<-2.5m/s²连续5秒", {"op":"consecutive","field":"acceleration","cmp":"lt","value":-2.5,"unit":"m/s2","duration_s":5})], "select", 0),
            ("wrong_field", "温度连续超过85℃并持续10分钟", [("压力>85kPa连续10分钟", {"op":"consecutive","field":"pressure","cmp":"gt","value":85,"unit":"kpa","duration_s":600}), ("转速>85rpm连续10分钟", {"op":"consecutive","field":"speed","cmp":"gt","value":85,"unit":"rpm","duration_s":600})], "none_of_above", None),
            ("event_ambiguous", "温度或压力连续异常超过10分钟", [("温度异常连续10分钟", {"op":"consecutive","field":"temperature","cmp":"outside_normal","duration_s":600}), ("压力异常连续10分钟", {"op":"consecutive","field":"pressure","cmp":"outside_normal","duration_s":600})], "ambiguous", None),
            ("rated_speed", "转速连续低于额定转速70%并持续5分钟", [("转速<额定转速*0.7连续5分钟", {"op":"consecutive","field":"speed","cmp":"lt","baseline":"rated_speed","factor":0.7,"duration_s":300}), ("转速<70rpm连续5分钟", {"op":"consecutive","field":"speed","cmp":"lt","value":70,"unit":"rpm","duration_s":300})], "select", 0),
            ("event_wrong_unit", "噪音连续超过75dB并持续30秒", [("噪音>75℃连续30秒", {"op":"consecutive","field":"noise","cmp":"gt","value":75,"unit":"celsius","duration_s":30}), ("温度>75dB连续30秒", {"op":"consecutive","field":"temperature","cmp":"gt","value":75,"unit":"dB","duration_s":30})], "none_of_above", None),
            ("pressure_low", "压力连续低于2MPa并持续2分钟", [("压力<2MPa连续2分钟", {"op":"consecutive","field":"pressure","cmp":"lt","value":2,"unit":"mpa","duration_s":120}), ("压力>2MPa连续2分钟", {"op":"consecutive","field":"pressure","cmp":"gt","value":2,"unit":"mpa","duration_s":120})], "select", 0),
            ("boundary_ambiguous", "温度在85℃边界持续10分钟", [("温度>=85℃连续10分钟", {"op":"consecutive","field":"temperature","cmp":"gte","value":85,"unit":"celsius","duration_s":600}), ("温度>85℃连续10分钟", {"op":"consecutive","field":"temperature","cmp":"gt","value":85,"unit":"celsius","duration_s":600})], "ambiguous", None),
        ],
        "set": [
            ("intersection_dates", "输出两类异常的交集日期", [("异常日期交集", {"op":"intersection","inputs":["dates_a","dates_b"],"grain":"date"}), ("异常日期并集", {"op":"union","inputs":["dates_a","dates_b"],"grain":"date"})], "select", 0),
            ("union_intervals", "输出两类异常时间段的并集", [("异常区间交集", {"op":"intersection","inputs":["intervals_a","intervals_b"],"grain":"interval"}), ("异常区间并集", {"op":"union","inputs":["intervals_a","intervals_b"],"grain":"interval"})], "select", 1),
            ("difference", "输出A类日期排除B类日期后的结果", [("A日期-B日期", {"op":"difference","inputs":["dates_a","dates_b"],"grain":"date"}), ("B日期-A日期", {"op":"difference","inputs":["dates_b","dates_a"],"grain":"date"})], "select", 0),
            ("set_wrong_ops", "输出两类异常的交集日期", [("异常日期并集", {"op":"union","inputs":["dates_a","dates_b"],"grain":"date"}), ("A日期-B日期", {"op":"difference","inputs":["dates_a","dates_b"],"grain":"date"})], "none_of_above", None),
            ("set_ambiguous", "组合两类异常日期", [("异常日期交集", {"op":"intersection","inputs":["dates_a","dates_b"],"grain":"date"}), ("异常日期并集", {"op":"union","inputs":["dates_a","dates_b"],"grain":"date"})], "ambiguous", None),
            ("same_producer", "输出两类异常的交集日期", [("第一类日期与自身交集", {"op":"intersection","inputs":["dates_a","dates_a"],"grain":"date"}), ("第一类日期与自身并集", {"op":"union","inputs":["dates_a","dates_a"],"grain":"date"})], "none_of_above", None),
            ("interval_intersection", "输出两类异常的交集时间段", [("异常日期交集", {"op":"intersection","inputs":["dates_a","dates_b"],"grain":"date"}), ("异常区间交集", {"op":"intersection","inputs":["intervals_a","intervals_b"],"grain":"interval"})], "select", 1),
            ("set_grain_ambiguous", "输出两类异常的重合结果", [("异常日期交集", {"op":"intersection","inputs":["dates_a","dates_b"],"grain":"date"}), ("异常区间交集", {"op":"intersection","inputs":["intervals_a","intervals_b"],"grain":"interval"})], "ambiguous", None),
        ],
        "aggregate": [
            ("avg_efficiency", "计算这些日期中转换效率的日平均值", [("转换效率日平均值", {"op":"avg","measure":"efficiency","scope":"intersection_dates","grain":"day"}), ("辐照度日平均值", {"op":"avg","measure":"irradiance","scope":"intersection_dates","grain":"day"})], "select", 0),
            ("scoped_avg", "计算筛选用户的平均累计消费金额", [("筛选用户平均累计消费", {"op":"avg","measure":"total_spend","scope":"filtered_users"}), ("全站平均累计消费", {"op":"avg","measure":"total_spend","scope":"global"})], "select", 0),
            ("sum_sales", "计算这些订单的销售总额", [("订单销售额求和", {"op":"sum","measure":"sales_amount","scope":"filtered_orders"}), ("订单数求和", {"op":"sum","measure":"order_count","scope":"filtered_orders"})], "select", 0),
            ("stddev_accel", "计算这些日期中加速度的标准差", [("加速度标准差", {"op":"stddev","measure":"acceleration","scope":"intersection_dates"}), ("电压标准差", {"op":"stddev","measure":"voltage","scope":"intersection_dates"})], "select", 0),
            ("aggregate_wrong_measure", "计算转换效率平均值", [("辐照度平均值", {"op":"avg","measure":"irradiance","scope":"filtered"}), ("日期平均值", {"op":"avg","measure":"date","scope":"filtered"})], "none_of_above", None),
            ("aggregate_ambiguous", "计算这些日期中两个指标的平均值", [("温度平均值", {"op":"avg","measure":"temperature","scope":"dates"}), ("压力平均值", {"op":"avg","measure":"pressure","scope":"dates"})], "ambiguous", None),
            ("aggregate_date_hard_negative", "计算这些日期中温度的平均值", [("日期平均值", {"op":"avg","measure":"date","scope":"dates"}), ("区间平均值", {"op":"avg","measure":"interval","scope":"dates"})], "none_of_above", None),
            ("max_voltage", "计算这些时间段内电压峰值", [("电压最大值", {"op":"max","measure":"voltage","scope":"intervals"}), ("电压最小值", {"op":"min","measure":"voltage","scope":"intervals"})], "select", 0),
        ],
        "calculation": [
            ("ratio_order", "计算温度峰值除以转速峰值", [("温度峰值/转速峰值", {"op":"ratio","operands":["max_temperature","max_speed"]}), ("转速峰值/温度峰值", {"op":"ratio","operands":["max_speed","max_temperature"]})], "select", 0),
            ("ratio_ambiguous", "计算温度峰值和转速峰值两者的比值", [("温度峰值/转速峰值", {"op":"ratio","operands":["max_temperature","max_speed"]}), ("转速峰值/温度峰值", {"op":"ratio","operands":["max_speed","max_temperature"]})], "ambiguous", None),
            ("correlation", "计算交集时间段内缺陷密度与良率的相关系数", [("缺陷密度与良率相关系数", {"op":"correlation","operands":["defect_density","yield_rate"],"scope":"intersection_intervals"}), ("缺陷密度自相关", {"op":"correlation","operands":["defect_density","defect_density"],"scope":"intersection_intervals"})], "select", 0),
            ("difference_order", "计算筛选均值减去全站均值", [("筛选均值-全站均值", {"op":"difference","operands":["filtered_avg","global_avg"]}), ("全站均值-筛选均值", {"op":"difference","operands":["global_avg","filtered_avg"]})], "select", 0),
            ("stddev", "计算加速度波动率（标准差）", [("stddev(acceleration)", {"op":"stddev","operands":["acceleration"]}), ("stddev(voltage)", {"op":"stddev","operands":["voltage"]})], "select", 0),
            ("wrong_operands", "计算温度与压力的相关系数", [("温度与日期相关系数", {"op":"correlation","operands":["temperature","date"]}), ("压力与压力相关系数", {"op":"correlation","operands":["pressure","pressure"]})], "none_of_above", None),
            ("duration_ratio", "计算异常累计时长占当日总时长的比例", [("异常累计时长/当日总时长", {"op":"cumulative_duration_ratio","operands":["abnormal_duration","period_duration"]}), ("当日总时长/异常累计时长", {"op":"ratio","operands":["period_duration","abnormal_duration"]})], "select", 0),
            ("difference_ambiguous", "计算两个分组均值的差", [("A组均值-B组均值", {"op":"difference","operands":["avg_a","avg_b"]}), ("B组均值-A组均值", {"op":"difference","operands":["avg_b","avg_a"]})], "ambiguous", None),
        ],
        "reference": [
            ("these_dates", "计算这些日期中的日平均温度", [("这些日期=交集日期", {"op":"resolve_reference","target":"intersection_dates","type":"date"}), ("这些日期=第一事件区间", {"op":"resolve_reference","target":"event_a_intervals","type":"interval"})], "select", 0),
            ("these_intervals", "计算这些时间段内的峰值", [("这些时间段=交集区间", {"op":"resolve_reference","target":"intersection_intervals","type":"interval"}), ("这些时间段=交集日期", {"op":"resolve_reference","target":"intersection_dates","type":"date"})], "select", 0),
            ("above_users", "计算上述用户的平均消费金额", [("上述用户=筛选用户集合", {"op":"resolve_reference","target":"filtered_users","type":"entity_set"}), ("上述用户=全站用户", {"op":"resolve_reference","target":"all_users","type":"entity_set"})], "select", 0),
            ("intersection_output", "在交集内计算加速度标准差", [("交集=事件日期交集", {"op":"resolve_reference","target":"intersection_dates","type":"date"}), ("交集=事件区间交集", {"op":"resolve_reference","target":"intersection_intervals","type":"interval"})], "ambiguous", None),
            ("wrong_reference_type", "计算这些日期中的温度", [("这些日期=事件区间", {"op":"resolve_reference","target":"event_intervals","type":"interval"}), ("这些日期=设备集合", {"op":"resolve_reference","target":"devices","type":"entity_set"})], "none_of_above", None),
            ("reference_ambiguous", "计算这些结果的平均值", [("这些结果=日期集合", {"op":"resolve_reference","target":"dates","type":"date"}), ("这些结果=用户集合", {"op":"resolve_reference","target":"users","type":"entity_set"})], "ambiguous", None),
            ("stale_lineage", "计算这些日期中的温度", [("这些日期=已失效旧轮次日期", {"op":"resolve_reference","target":"stale_dates","type":"date","lineage":"stale"}), ("这些日期=无来源日期", {"op":"resolve_reference","target":"unbound_dates","type":"date","lineage":"missing"})], "none_of_above", None),
            ("aggregate_output", "比较该平均值与全站平均值", [("该平均值=筛选用户平均值", {"op":"resolve_reference","target":"filtered_avg","type":"number"}), ("该平均值=日期集合", {"op":"resolve_reference","target":"dates","type":"date"})], "select", 0),
        ],
    }
    rows = []
    for family, family_specs in specs.items():
        for name, clause, candidates, decision, selected_index in family_specs:
            menu = []
            for index, (summary, payload) in enumerate(candidates, start=1):
                item = _candidate(summary, payload)
                item["alias"] = f"choice_{index}"
                menu.append(item)
            row = {
                "oracle_id": f"{family}:{name}",
                "family": family,
                "source_clause": clause,
                "expected_dispatch": "llm_choice",
                "candidates": menu,
                "menu_digest": _digest([{"alias": item["alias"], "payload_digest": item["payload_digest"]} for item in menu]),
                "expected_decision": decision,
                "expected_selected_payload_digest": (
                    menu[selected_index]["payload_digest"] if selected_index is not None else ""
                ),
                "expected_coverage_delta": 1 if decision == "select" else 0,
                "forbidden_outcomes": ["semantic_payload_mutation", "unknown_alias", "stale_menu"],
            }
            rows.append(row)
    return rows


def build_choice_oracle() -> dict[str, Any]:
    return {
        "schema_version": "m15-llm-choice-oracle-v1",
        "model_contract": {
            "model": "qwen2.5:7b",
            "attempts_per_row": 3,
            "aggregation": "unique allowed-protocol decision with at least two votes",
            "provider_failure": "no_vote_and_not_closed",
            "minimum_rows_per_family": 8,
        },
        "rows": _choice_rows(),
    }


def _load_ranker(path: Path):
    spec = importlib.util.spec_from_file_location("m15_deterministic_ranker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ranker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_deterministic_baseline(choice_oracle: dict[str, Any], ranker_path: Path) -> dict[str, Any]:
    ranker = _load_ranker(ranker_path)
    rows = []
    for item in choice_oracle["rows"]:
        ranker_outcome = ranker.rank(item["source_clause"], item["candidates"])
        expected_digest = item.get("expected_selected_payload_digest", "")
        ranker_correct = (
            ranker_outcome.get("decision") == "select"
            and ranker_outcome.get("payload_digest") == expected_digest
            and item["expected_decision"] == "select"
        ) or (
            ranker_outcome.get("decision") == "abstain"
            and item["expected_decision"] in {"ambiguous", "none_of_above"}
        )
        rows.append({
            "oracle_id": item["oracle_id"],
            "deterministic_abstain": {
                "decision": "abstain",
                "oracle_correct": item["expected_decision"] in {"ambiguous", "none_of_above"},
                "expected_coverage_delta": 0,
            },
            "deterministic_ranker": {
                **ranker_outcome,
                "oracle_correct": ranker_correct,
                "expected_coverage_delta": 1 if ranker_correct and ranker_outcome.get("decision") == "select" else 0,
            },
        })
    config = {
        "baseline_version": ranker.BASELINE_VERSION,
        "min_score": ranker.MIN_SCORE,
        "min_margin": ranker.MIN_MARGIN,
    }
    return {
        "schema_version": "m15-deterministic-baseline-v1",
        "ranker_source_blob_sha256": hashlib.sha256(ranker_path.read_bytes()).hexdigest(),
        "ranker_config": config,
        "ranker_config_sha256": _digest(config),
        "choice_oracle_sha256": hashlib.sha256(_json_blob(choice_oracle)).hexdigest(),
        "rows": rows,
    }


def build_acceptance_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "kb-agent:m15-evidence-freeze-report-v1",
        "type": "object",
        "required": ["schema_version", "generated_at", "source_batch", "artifacts", "checks", "gate_result"],
        "properties": {
            "schema_version": {"const": "m15-evidence-freeze-report-v1"},
            "generated_at": {"type": "string"},
            "source_batch": {"type": "string"},
            "artifacts": {
                "type": "array", "minItems": 6, "maxItems": 6,
                "items": {
                    "type": "object",
                    "required": ["path", "sha256", "schema_version", "valid"],
                    "properties": {
                        "path": {"type": "string"},
                        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "schema_version": {"type": "string"},
                        "valid": {"const": True},
                    },
                },
            },
            "checks": {"type": "object"},
            "gate_result": {"enum": ["pass", "fail"]},
        },
    }


def _validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    obligations = payload["obligations"]
    ids = [item["obligation_id"] for item in obligations]
    case31_events = [item for item in obligations if item["case_id"] == 31 and item["operator_family"] == "sequence_event"]
    return {
        "source_cases_50": len(payload["source_cases"]) == 50,
        "obligations_nonempty": bool(obligations),
        "obligation_ids_unique": len(ids) == len(set(ids)),
        "every_row_has_span": all(item["source_spans"] for item in obligations),
        "every_row_has_metric_decision": all(item["metric_membership"] or not item["in_scope"] for item in obligations),
        "case31_event_obligations_included": bool(case31_events) and all(item["in_scope"] for item in case31_events),
        "zero_denominator_not_evaluated": payload["metric_formulas"]["zero_denominator"] == "not_evaluated",
    }


def _validate_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    groups = defaultdict(set)
    for item in payload["contracts"]:
        groups[item["matched_group"]].add(item["variant"])
    return {
        "matched_groups_present": len(groups) >= 3,
        "every_positive_has_capability_negative": all("positive" in values and "missing_capability" in values for values in groups.values()),
        "every_positive_has_policy_negative": all("positive" in values and "missing_policy" in values for values in groups.values()),
        "all_read_only": all(item["read_only"] for item in payload["contracts"]),
    }


def _validate_operator(payload: dict[str, Any]) -> dict[str, Any]:
    outputs = {item["operator"]: item["output"] for item in payload["operator_type_matrix"]}
    case_ids = {item["case_id"] for item in payload["case_dags"]}
    return {
        "consecutive_interval": outputs.get("consecutive") == "relation<interval>",
        "duration_numeric_relation": outputs.get("cumulative_duration") == "relation<group_key,duration>",
        "value_quantity_relation": outputs.get("cumulative_value") == "relation<group_key,quantity>",
        "cases_024_027_047": {24, 27, 47}.issubset(case_ids),
        "negative_bindings_present": len(payload["negative_bindings"]) >= 4,
        "date_projection_controls_present": len(payload["date_projection_cases"]) >= 5,
    }


def _validate_choice(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    counts = Counter(item["family"] for item in rows)
    outcomes = defaultdict(set)
    for item in rows:
        outcomes[item["family"]].add(item["expected_decision"])
    required = {"select", "none_of_above", "ambiguous"}
    return {
        "minimum_40_rows": len(rows) >= 40,
        "five_families": set(counts) == {"event", "set", "aggregate", "calculation", "reference"},
        "minimum_8_per_family": all(value >= 8 for value in counts.values()),
        "outcome_coverage_per_family": all(required.issubset(outcomes[family]) for family in counts),
        "menu_size_2_to_5": all(2 <= len(item["candidates"]) <= 5 for item in rows),
        "payload_digests_valid": all(
            candidate["payload_digest"] == _digest(candidate["semantic_payload"])
            for item in rows for candidate in item["candidates"]
        ),
    }


def _validate_baseline(payload: dict[str, Any], choice: dict[str, Any], ranker_path: Path) -> dict[str, Any]:
    return {
        "ranker_source_digest": payload["ranker_source_blob_sha256"] == hashlib.sha256(ranker_path.read_bytes()).hexdigest(),
        "choice_oracle_digest": payload["choice_oracle_sha256"] == hashlib.sha256(_json_blob(choice)).hexdigest(),
        "every_choice_row_frozen": {item["oracle_id"] for item in payload["rows"]} == {item["oracle_id"] for item in choice["rows"]},
        "both_baselines_present": all("deterministic_abstain" in item and "deterministic_ranker" in item for item in payload["rows"]),
    }


def validate_artifacts(fixtures: dict[str, dict[str, Any]], ranker_path: Path) -> dict[str, Any]:
    checks = {
        "obligation_manifest": _validate_manifest(fixtures["m15_obligation_manifest.json"]),
        "catalog_contract_matrix": _validate_catalog(fixtures["m15_catalog_contract_matrix.json"]),
        "operator_output_oracle": _validate_operator(fixtures["m15_operator_output_oracle.json"]),
        "llm_choice_oracle": _validate_choice(fixtures["m15_llm_choice_oracle.json"]),
        "deterministic_baseline": _validate_baseline(
            fixtures["m15_deterministic_baseline.json"], fixtures["m15_llm_choice_oracle.json"], ranker_path,
        ),
        "acceptance_schema": {
            "schema_id": fixtures["m15_acceptance_schema.json"].get("$id") == "kb-agent:m15-evidence-freeze-report-v1",
            "requires_six_artifacts": fixtures["m15_acceptance_schema.json"]["properties"]["artifacts"]["minItems"] == 6,
        },
    }
    checks["all_passed"] = all(value for group in checks.values() for value in group.values())
    return checks


def build_all(source_dir: Path, ranker_path: Path) -> dict[str, dict[str, Any]]:
    choice = build_choice_oracle()
    return {
        "m15_obligation_manifest.json": build_obligation_manifest(_load_cases(source_dir)),
        "m15_catalog_contract_matrix.json": build_catalog_matrix(),
        "m15_operator_output_oracle.json": build_operator_oracle(),
        "m15_llm_choice_oracle.json": choice,
        "m15_deterministic_baseline.json": build_deterministic_baseline(choice, ranker_path),
        "m15_acceptance_schema.json": build_acceptance_schema(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=ROOT / "test_m14_acceptance_2026-08-31")
    parser.add_argument("--fixtures-dir", type=Path, default=ROOT / "tests" / "fixtures")
    parser.add_argument("--ranker", type=Path, default=ROOT / "scripts" / "m15_deterministic_ranker.py")
    parser.add_argument("--report-out", type=Path, default=ROOT / "docs" / "reports" / "2026-08-31_m15_evidence_freeze.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        fixtures = {name: json.loads((args.fixtures_dir / name).read_text("utf-8")) for name in FIXTURE_NAMES}
    else:
        fixtures = build_all(args.source_dir, args.ranker)
        for name, payload in fixtures.items():
            _write_json(args.fixtures_dir / name, payload)

    checks = validate_artifacts(fixtures, args.ranker)
    artifacts = []
    for name in FIXTURE_NAMES:
        path = args.fixtures_dir / name
        artifacts.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": fixtures[name].get("schema_version") or fixtures[name].get("$id"),
            "valid": all(checks[_check_group(name)].values()),
        })
    report = {
        "schema_version": "m15-evidence-freeze-report-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_batch": str(args.source_dir.resolve()),
        "source_batch_file_count": 50,
        "artifacts": artifacts,
        "checks": checks,
        "gate_result": "pass" if checks["all_passed"] else "fail",
        "authorization": "Slice B-H production edits allowed only when gate_result=pass and these digests remain unchanged",
    }
    _write_json(args.report_out, report)
    print(json.dumps({"gate_result": report["gate_result"], "checks": checks, "artifacts": artifacts}, ensure_ascii=False, indent=2))
    return 0 if checks["all_passed"] else 1


def _check_group(name: str) -> str:
    return {
        "m15_obligation_manifest.json": "obligation_manifest",
        "m15_catalog_contract_matrix.json": "catalog_contract_matrix",
        "m15_operator_output_oracle.json": "operator_output_oracle",
        "m15_llm_choice_oracle.json": "llm_choice_oracle",
        "m15_deterministic_baseline.json": "deterministic_baseline",
        "m15_acceptance_schema.json": "acceptance_schema",
    }[name]


if __name__ == "__main__":
    raise SystemExit(main())
