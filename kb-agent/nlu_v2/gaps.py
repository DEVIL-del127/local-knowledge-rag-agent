"""Deterministic gap analysis for bounded semantic Patch requests."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable

from .catalog import CatalogSnapshot
from .models import SourceSpan, UnderstandingIR
from .patch_protocol import GapRequest
from .semantic_repair import RepairUnit


_UNRESOLVED_TO_GAP = {
    "event_structure_incomplete": (
        "missing_event", ["condition", "duration", "metric"], ["add_event"]
    ),
    "set_inputs_unbound": (
        "missing_set_inputs", ["inputs", "operation"], ["add_set_operation"]
    ),
    "formula_input_unbound": (
        "missing_formula_input", ["input_expression"],
        ["add_calculation", "add_aggregate", "add_comparison"],
    ),
    "aggregate_scope_unbound": (
        "missing_aggregate_scope", ["scope", "scope_ref"], ["add_aggregate"]
    ),
    "reference_unresolved": (
        "unresolved_reference", ["target_ref"], ["resolve_reference"]
    ),
    "unknown_field": (
        "unresolved_symbol", ["field_binding"], ["add_ambiguity"]
    ),
}

_REQUIREMENT_TO_GAP = {
    "sequence_event": ("missing_event", ["condition", "duration", "metric"], ["add_event"]),
    "cumulative_duration": ("missing_event", ["condition", "duration", "metric"], ["add_event"]),
    "set_operation": ("missing_set_inputs", ["inputs", "operation"], ["add_set_operation"]),
    "calculation": ("missing_formula_input", ["input_expression"], ["add_calculation", "add_aggregate", "add_comparison"]),
    "formula": ("ambiguous_formula", ["formula"], ["add_calculation", "add_ambiguity"]),
    "scoped_aggregate": ("missing_aggregate_scope", ["scope", "input"], ["add_aggregate"]),
    "reference": ("unresolved_reference", ["target_ref"], ["resolve_reference", "add_ambiguity"]),
    "turn_directive": ("missing_turn_directive", ["directive"], ["add_turn_directive", "add_ambiguity"]),
    "predicate": ("missing_predicate", ["predicate"], ["add_ambiguity"]),
    "conversion": ("missing_conversion", ["conversion"], ["add_calculation", "add_ambiguity"]),
    "temporal_ambiguity": ("ambiguous_formula", ["century"], ["add_ambiguity"]),
    "output": ("missing_output", ["output_contract"], ["add_ambiguity"]),
}

_GAP_PRIORITY = {
    "missing_turn_directive": 10, "ambiguous_formula": 20,
    "missing_predicate": 30, "missing_event": 40, "missing_sequence": 40, "missing_set_inputs": 50,
    "missing_formula_input": 60, "missing_aggregate_scope": 70,
    "unresolved_reference": 80, "missing_conversion": 90,
    "unresolved_symbol": 100, "missing_output": 110,
}

_DURATION_EVIDENCE_RE = re.compile(
    r"(?:持续|连续|一直|保持|时长).{0,12}?\d+(?:\.\d+)?\s*"
    r"(?:毫秒|秒钟?|分钟|分|小时|时|天|日|个?交易日|min|h|s)", re.I,
)
_DETERMINISTIC_GAPS = {"unresolved_symbol", "missing_output", "missing_predicate"}


@dataclass(slots=True)
class GapAnalyzer:
    """Translate deterministic IR defects into minimal model work orders."""

    context_radius: int = 96

    def analyze(self, ir: UnderstandingIR, catalog: CatalogSnapshot) -> list[GapRequest]:
        query = ir.query.normalized
        catalog_symbols = self._catalog_symbols(ir, catalog)
        hypothesis_symbols = [item.hypothesis_id for item in ir.schema_hypotheses]
        metric_ids = self._metric_ids(ir, catalog)
        output_refs = self._output_refs(ir)
        gaps: list[GapRequest] = []
        coverage_by_id = {item.requirement_id: item for item in ir.coverage}

        for requirement in ir.requirements:
            if coverage_by_id.get(requirement.requirement_id) and (
                coverage_by_id[requirement.requirement_id].status == "satisfied"
            ):
                continue
            mapping = _REQUIREMENT_TO_GAP.get(requirement.requirement_type)
            if mapping is None:
                continue
            gap_type, slots, operations = mapping
            if requirement.requirement_type == "sequence_event" and self._is_ordered_sequence(ir, requirement):
                gap_type, slots, operations = (
                    "missing_sequence", ["steps", "partition_by", "order_by"], ["add_sequence"],
                )
            gaps.append(self._build(
                ir, gap_type, f"requirement:{requirement.requirement_id}",
                requirement.span, slots, operations, catalog_symbols, hypothesis_symbols,
                metric_ids, output_refs,
                {"requirement_type": requirement.requirement_type,
                 "text": requirement.text,
                 "attributes": requirement.expected_attributes},
                requirement_id=requirement.requirement_id,
                requirement_dependency_ids=list(requirement.dependencies),
                clause_id=requirement.clause_id,
            ))

        for index, item in enumerate(ir.unresolved):
            mapping = _UNRESOLVED_TO_GAP.get(item.code)
            if mapping is None or item.code == "write_operation":
                continue
            gap_type, slots, operations = mapping
            if gap_type == "unresolved_reference" and any(
                requirement.requirement_type == "reference"
                and requirement.span is not None and item.span is not None
                and requirement.span.start < item.span.end
                and item.span.start < requirement.span.end
                for requirement in ir.requirements
            ):
                continue
            gaps.append(self._build(
                ir, gap_type, f"unresolved:{index}:{item.code}", item.span,
                slots, operations, catalog_symbols, hypothesis_symbols,
                metric_ids, output_refs,
                {"code": item.code, "message": item.message,
                 "candidates": list(item.candidates),
                 "span": item.span.to_dict() if item.span else None},
            ))

        unresolved_codes = {item.code for item in ir.unresolved}
        for calculation in ir.calculations:
            missing = self._missing_calculation_slots(calculation, catalog)
            if not missing or "formula_input_unbound" in unresolved_codes:
                continue
            span = calculation.provenance[0].span if calculation.provenance else None
            gaps.append(self._build(
                ir, "missing_formula_input", f"calculation:{calculation.calculation_type}",
                span, missing, ["add_calculation", "add_aggregate", "add_comparison"],
                catalog_symbols, hypothesis_symbols, metric_ids, output_refs,
                {"calculation_type": calculation.calculation_type,
                 "expression": calculation.expression,
                 "parameters": calculation.parameters},
            ))

        for aggregate in ir.aggregates:
            if aggregate.scope != "relation" or aggregate.scope_ref is not None:
                continue
            spec = aggregate.aggregate
            gaps.append(self._build(
                ir, "missing_aggregate_scope", f"aggregate:{spec.aggregate_id}",
                self._first_span(spec.provenance), ["scope_ref"], ["add_aggregate"],
                catalog_symbols, hypothesis_symbols, metric_ids, output_refs,
                {"aggregate_id": spec.aggregate_id, "function": spec.function},
            ))

        for reference in ir.references:
            if reference.status == "resolved" and reference.target_task_id:
                continue
            if any(
                requirement.requirement_type == "reference"
                and requirement.span is not None and reference.span is not None
                and requirement.span.start < reference.span.end
                and reference.span.start < requirement.span.end
                for requirement in ir.requirements
            ):
                continue
            gaps.append(self._build(
                ir, "unresolved_reference", f"reference:{reference.reference_id}",
                reference.span, ["target_ref"], ["resolve_reference", "add_ambiguity"],
                catalog_symbols, hypothesis_symbols, metric_ids, output_refs,
                {"reference_id": reference.reference_id, "raw": reference.raw},
            ))

        gaps = self._dedupe(gaps)
        self._classify_readiness(ir, catalog, gaps)
        return sorted(gaps, key=lambda item: (
            _GAP_PRIORITY.get(item.gap_type, 999), item.gap_id,
        ))

    @staticmethod
    def repair_units(gaps: list[GapRequest]) -> list[RepairUnit]:
        """Expose dependency-connected repair work instead of flat Gap batches."""
        return RepairUnit.from_gaps(gaps)

    def _build(
        self, ir: UnderstandingIR, gap_type: str, target: str, span: SourceSpan | None,
        slots: list[str], operations: list[str], catalog_symbols: list[str],
        hypothesis_symbols: list[str], metrics: list[str], output_refs: list[str],
        summary: dict, requirement_id: str = "", clause_id: str = "",
        requirement_dependency_ids: list[str] | None = None,
    ) -> GapRequest:
        query_slice = self._query_slice(ir.query.normalized, span)
        resolved_clause_id = clause_id or self._clause_id(ir, span)
        identity = json.dumps(
            {"type": gap_type, "target": target, "slice": query_slice, "slots": slots},
            ensure_ascii=False, sort_keys=True,
        )
        gap_id = "gap_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        anchors = sorted({
            item.demand_id for item in ir.source_demands
            if (
                resolved_clause_id and item.clause_id == resolved_clause_id
            ) or (
                not resolved_clause_id and span is not None
                and item.span.start < span.end and span.start < item.span.end
            )
        })
        field_refs = sorted(dict.fromkeys([*catalog_symbols, *hypothesis_symbols]))
        return GapRequest(
            gap_id=gap_id,
            gap_type=gap_type,
            target_node_id=target,
            required_slots=slots,
            query_slice=query_slice,
            allowed_catalog_symbols=catalog_symbols,
            allowed_hypothesis_symbols=hypothesis_symbols,
            allowed_metrics=metrics if gap_type == "missing_event" else [],
            available_output_refs=output_refs,
            allowed_operations=operations,
            allowed_anchor_ids=anchors,
            existing_node_summary=summary,
            requirement_id=requirement_id,
            requirement_dependency_ids=list(requirement_dependency_ids or ()),
            clause_id=resolved_clause_id,
            choice_domain={
                "field_refs": field_refs,
                "metric_refs": list(metrics if gap_type == "missing_event" else ()),
                "output_refs": list(output_refs),
                **_semantic_choice_constraints(gap_type, query_slice, summary),
            },
            response_mode="semantic_choice",
        )

    def _classify_readiness(self, ir: UnderstandingIR, catalog: CatalogSnapshot,
                            gaps: list[GapRequest]) -> None:
        """Keep the model on resolvable leaf choices, never on missing inputs."""
        requirement_gaps = {
            gap.requirement_id: gap for gap in gaps if gap.requirement_id
        }
        coverage = {item.requirement_id: item.status for item in ir.coverage}
        catalog_units = sorted({
            str(field.unit) for source in catalog.sources for field in source.fields if field.unit
        } | {
            str(metric.unit) for source in catalog.sources
            for metric in source.derived_metrics if metric.unit
        } | {"second", "minute", "hour", "day", "ratio"})
        for gap in gaps:
            blockers: list[str] = []
            fields = gap.choice_domain.get("field_refs", [])
            outputs = gap.choice_domain.get("output_refs", [])
            gap.choice_domain["unit_refs"] = catalog_units

            if gap.gap_type in _DETERMINISTIC_GAPS:
                gap.readiness = "deterministic"
                gap.resolution_mode = "deterministic_terminal"
                gap.blocked_by = ["local_compiler_or_catalog"]
                continue

            for requirement_id in gap.requirement_dependency_ids:
                dependency = requirement_gaps.get(requirement_id)
                if (dependency is not None and dependency.gap_type in {
                    "missing_event", "missing_set_inputs", "missing_formula_input",
                    "missing_aggregate_scope", "missing_sequence",
                } and coverage.get(requirement_id) != "satisfied"):
                    blockers.append(dependency.gap_id)

            if not gap.allowed_anchor_ids:
                blockers.append("source_anchor")
            if gap.gap_type == "missing_event":
                if not fields:
                    blockers.append("field_binding")
                if not _DURATION_EVIDENCE_RE.search(gap.query_slice):
                    blockers.append("duration_evidence")
                if not gap.choice_domain.get("metric_refs"):
                    blockers.append("derived_metric")
            elif gap.gap_type == "missing_set_inputs":
                compatible = self._set_input_refs(ir, gap)
                gap.choice_domain["output_refs"] = compatible
                gap.available_output_refs = compatible
                if len(compatible) < 2:
                    blockers.append("two_typed_output_refs")
            elif gap.gap_type == "missing_formula_input":
                if not outputs and not fields:
                    blockers.append("numeric_input")
            elif gap.gap_type == "missing_aggregate_scope":
                if not fields and not outputs:
                    blockers.append("aggregate_input")
                if "scope_ref" in gap.required_slots and not outputs:
                    blockers.append("scope_output_ref")
            elif gap.gap_type == "unresolved_reference":
                if not outputs:
                    blockers.append("reference_target")
            elif gap.gap_type == "missing_sequence":
                if not fields:
                    blockers.append("sequence_fields")

            gap.blocked_by = sorted(dict.fromkeys(blockers))
            gap.readiness = "blocked" if gap.blocked_by else "ready"

    @staticmethod
    def _set_input_refs(ir: UnderstandingIR, gap: GapRequest) -> list[str]:
        wants_date = "日期" in gap.query_slice or "date" in str(gap.existing_node_summary).lower()
        date_refs = [item.output.ref_id for item in ir.derived_projections
                     if item.function == "local_date" or item.output.grain == "date"]
        interval_refs = [item.output_ref.ref_id for item in ir.events if item.output_ref]
        candidates = date_refs if wants_date and len(date_refs) >= 2 else interval_refs
        return sorted(dict.fromkeys(candidates))

    @staticmethod
    def _clause_id(ir: UnderstandingIR, span: SourceSpan | None) -> str:
        if span is None or ir.clause_graph is None:
            return ""
        return next((item.clause_id for item in ir.clause_graph.nodes
                     if item.span.start <= span.start < item.span.end), "")

    def _query_slice(self, query: str, span: SourceSpan | None) -> str:
        if span is None:
            return query[: self.context_radius * 2]
        start = max(0, span.start - self.context_radius)
        end = min(len(query), span.end + self.context_radius)
        return query[start:end]

    @staticmethod
    def _is_ordered_sequence(ir: UnderstandingIR, requirement) -> bool:
        if ir.sequences:
            return False
        text = ir.query.normalized
        return any(token in text for token in ("→", "->", "漏斗", "依次", "先后"))

    @staticmethod
    def _catalog_symbols(ir: UnderstandingIR, catalog: CatalogSnapshot) -> list[str]:
        preferred = {
            item.identifier for item in ir.source_candidates
            if catalog.source(item.identifier) is not None
        }
        if ir.schema_hypotheses and not preferred:
            return []
        fields = [
            field.field_id for source in catalog.sources
            if not preferred or source.source_id in preferred
            for field in source.fields
        ]
        return sorted(dict.fromkeys(fields))

    @staticmethod
    def _metric_ids(ir: UnderstandingIR, catalog: CatalogSnapshot) -> list[str]:
        preferred = {
            item.identifier for item in ir.source_candidates
            if catalog.source(item.identifier) is not None
        }
        values = [
            "semantic.consecutive_duration",
            "semantic.cumulative_duration",
        ]
        if not (ir.schema_hypotheses and not preferred):
            values.extend(
                metric.metric_id for source in catalog.sources
                if not preferred or source.source_id in preferred
                for metric in source.derived_metrics
            )
        values.extend(item.derived_metric.metric_id for item in ir.events)
        return sorted(dict.fromkeys(values))

    @staticmethod
    def _output_refs(ir: UnderstandingIR) -> list[str]:
        values = []
        values.extend(item.output_ref.ref_id for item in ir.events if item.output_ref)
        values.extend(item.output_ref.ref_id for item in ir.set_operations if item.output_ref)
        values.extend(item.output.ref_id for item in ir.derived_projections)
        values.extend(item.output.ref_id for item in ir.relation_filters)
        values.extend(item.aggregate.output.ref_id for item in ir.aggregates)
        values.extend(item.output.ref_id for item in ir.comparisons)
        values.extend(item.output.ref_id for item in ir.arithmetic)
        values.extend(item.output.ref_id for item in ir.conversions)
        values.extend(item.output.ref_id for item in ir.ordered_folds)
        values.extend(item.output.ref_id for item in ir.sequences)
        return sorted(dict.fromkeys(values))

    @staticmethod
    def _missing_calculation_slots(calculation, catalog: CatalogSnapshot) -> list[str]:
        if calculation.calculation_type == "volatility":
            field_id = str(calculation.parameters.get("input_field", ""))
            return [] if field_id and catalog.field(field_id) is not None else ["input_field"]
        if calculation.calculation_type in {"mom", "yoy"}:
            return [name for name in ("left", "right") if not calculation.parameters.get(name)]
        return [] if calculation.parameters else ["input_expression"]

    @staticmethod
    def _first_span(provenance: Iterable) -> SourceSpan | None:
        return next((item.span for item in provenance if item.span is not None), None)

    @staticmethod
    def _dedupe(gaps: list[GapRequest]) -> list[GapRequest]:
        result = []
        seen = set()
        for gap in gaps:
            key = (gap.gap_type, gap.target_node_id)
            if key in seen:
                continue
            seen.add(key)
            result.append(gap)
        return result


def _semantic_choice_constraints(gap_type: str, text: str, summary: dict) -> dict[str, list[str]]:
    """Derive small operation enums from source-backed Requirement attributes."""
    attributes = summary.get("attributes") if isinstance(summary.get("attributes"), dict) else {}
    result: dict[str, list[str]] = {}
    if gap_type == "missing_aggregate_scope":
        function = str(attributes.get("function", ""))
        if not function:
            function = next((value for pattern, value in (
                (r"平均|均值", "avg"), (r"总额|总和|合计|求和", "sum"),
                (r"最小|最低", "min"), (r"最大|最高|峰值", "max"),
                (r"标准差|波动率", "stddev"), (r"数量|次数|计数", "count"),
            ) if re.search(pattern, text)), "")
        scope = str(attributes.get("scope", ""))
        if not scope:
            if re.search(r"全站|全局|总体|全部", text):
                scope = "global"
            elif re.search(r"这些|上述|筛选|满足|交集|并集|重合", text):
                scope = "filtered"
        if function:
            result["functions"] = [function]
        if scope:
            result["scopes"] = [scope]
    elif gap_type in {"missing_formula_input", "ambiguous_formula"}:
        family = str(attributes.get("operator_family", ""))
        choices = next((value for pattern, value in (
            (r"标准差|波动率", ("volatility", "stddev")),
            (r"相关系数|相关性", ("correlation", "correlation")),
            (r"占比|比例|比值|相除", ("ratio", "ratio")),
            (r"差值|相减", ("difference", "difference")),
            (r"乘积", ("product", "product")),
            (r"最大值|最高值|峰值", ("maximum", "identity")),
            (r"同比|环比|增长率", ("growth", "growth_rate")),
        ) if re.search(pattern, text)), None)
        if choices is None and family in {
            "volatility", "correlation", "ratio", "difference", "product", "growth",
        }:
            template = {"volatility": "stddev", "growth": "growth_rate"}.get(family, family)
            choices = (family, template)
        if choices is not None:
            result["calculation_types"] = [choices[0]]
            result["formula_templates"] = [choices[1]]
    elif gap_type == "missing_set_inputs":
        operation = next((value for pattern, value in (
            (r"交集|重合", "intersection"), (r"并集", "union"), (r"差集", "difference"),
        ) if re.search(pattern, text)), "")
        if operation:
            result["set_operations"] = [operation]
    return result
