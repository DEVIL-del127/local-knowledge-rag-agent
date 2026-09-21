"""Locally compiled, role-typed semantic candidate menus."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .catalog import CatalogSnapshot
from .expression_signatures import (
    DEFAULT_EXPRESSION_SIGNATURES,
    ValueSignature,
)
from .field_roles import hypothesis_matches_field
from .models import RequirementSpec, SourceDemand, UnderstandingIR
from .patch_protocol import GapRequest, SemanticPatchV1, parse_atomic_patch_text, request_ir_digest
from .semantic_targets import OutputLineageIndex, SemanticTargetBuilder


CANDIDATE_COMPILER_VERSION = "m14-candidate-compiler-v1"
CANDIDATE_CHOICE_PROTOCOL = "candidate-choice-v3"


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    candidate_id: str
    operation_type: str
    semantic_payload: dict[str, Any]
    evidence_summary: str


@dataclass(frozen=True, slots=True)
class CandidateMenu:
    gap_id: str
    requirement_id: str
    source_clause: str
    target_summary: dict[str, Any]
    candidates: tuple[SemanticCandidate, ...]
    dispatch: str
    blocked_by: tuple[str, ...]
    source_clause_digest: str
    target_contract_digest: str
    base_lineage_digest: str
    candidate_menu_digest: str
    candidate_compiler_version: str = CANDIDATE_COMPILER_VERSION
    expression_signature_version: str = DEFAULT_EXPRESSION_SIGNATURES.VERSION

    def candidate(self, candidate_id: str) -> SemanticCandidate | None:
        return next((item for item in self.candidates if item.candidate_id == candidate_id), None)


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    decision: str
    candidate_id: str = ""
    error: str = ""


def classify_candidate_dispatch(candidate_count: int,
                                blockers: list[str] | tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Classify a frozen menu with the same policy used in production."""
    reasons = list(blockers)
    if reasons:
        return "blocked_by_upstream", tuple(sorted(set(reasons)))
    if candidate_count == 1:
        return "local_compile", ()
    if 2 <= candidate_count <= 5:
        return "llm_choice", ()
    if candidate_count == 0:
        reasons.append("no_typed_candidate")
        return "blocked_by_upstream", tuple(reasons)
    reasons.append("candidate_menu_too_large")
    return "clarify", tuple(reasons)


class SemanticCandidateCompiler:
    """Compile candidate payloads from exact local evidence and typed outputs."""

    def __init__(self, targets: SemanticTargetBuilder | None = None):
        self.targets = targets or SemanticTargetBuilder()

    def compile(self, ir: UnderstandingIR, gap: GapRequest,
                catalog: CatalogSnapshot) -> CandidateMenu:
        requirement = next(
            (item for item in ir.requirements if item.requirement_id == gap.requirement_id),
            None,
        )
        lineage = OutputLineageIndex.build(ir)
        source_clause = requirement.span.text if requirement and requirement.span else gap.query_slice
        target = self.targets.build(requirement, ir) if requirement else None
        blockers = list(gap.blocked_by)
        payloads: list[tuple[str, dict[str, Any], str]] = []
        if requirement is None:
            blockers.append("target_requirement")
        elif target is not None and target.grounding.enforceable and not target.ready:
            blockers.extend(
                f"role:{item.role}" for item in target.grounding.roles
                if item.status == "unproven"
            )
        elif gap.gap_type == "missing_event":
            payloads, extra = self._event_candidates(ir, requirement, gap, catalog)
            blockers.extend(extra)
        elif gap.gap_type == "missing_set_inputs":
            payloads, extra = self._set_candidates(ir, requirement, gap)
            blockers.extend(extra)
        elif gap.gap_type == "missing_aggregate_scope":
            payloads, extra = self._aggregate_candidates(
                ir, requirement, gap, catalog, lineage,
            )
            blockers.extend(extra)
        elif gap.gap_type == "missing_formula_input":
            payloads, extra = self._calculation_candidates(
                ir, requirement, gap, catalog, lineage,
            )
            blockers.extend(extra)
        elif gap.gap_type == "unresolved_reference":
            payloads, extra = self._reference_candidates(
                ir, requirement, gap, lineage,
            )
            blockers.extend(extra)
        else:
            blockers.append("unsupported_candidate_family")

        candidates = tuple(self._candidate(*item) for item in payloads)
        dispatch, normalized_blockers = classify_candidate_dispatch(
            len(candidates), blockers,
        )
        blockers = list(normalized_blockers)
        if dispatch == "blocked_by_upstream" and blockers:
            candidates = ()

        target_summary = {
            "requirement_id": requirement.requirement_id if requirement else "",
            "family": requirement.requirement_type if requirement else "",
            "operation": target.operation if target else "",
            "measure_fields": list(target.measure_fields) if target else [],
            "scope_requirement_ids": list(target.scope_requirement_ids) if target else [],
            "group_by_grain": target.group_by_grain if target else "",
        }
        source_digest = _digest(source_clause)
        target_digest = _digest(target_summary)
        lineage_digest = _digest({key: value.__dict__ if hasattr(value, "__dict__") else {
            "ref_id": value.ref_id, "producer_id": value.producer_id,
            "result_type": value.result_type, "unit": value.unit,
            "shape": value.shape, "grain": value.grain,
            "source_fields": value.source_fields, "source_anchor_ids": value.source_anchor_ids,
            "scope_ancestors": value.scope_ancestors, "input_refs": value.input_refs,
            "semantic_role": value.semantic_role,
        } for key, value in sorted(lineage.entries.items())})
        menu_digest = _digest([
            {"candidate_id": item.candidate_id, "payload": item.semantic_payload}
            for item in candidates
        ])
        return CandidateMenu(
            gap.gap_id, requirement.requirement_id if requirement else "", source_clause,
            target_summary, candidates, dispatch, tuple(sorted(set(blockers))),
            source_digest, target_digest, lineage_digest, menu_digest,
        )

    def compile_patch(self, candidate: SemanticCandidate, gap: GapRequest,
                      ir: UnderstandingIR) -> tuple[SemanticPatchV1 | None, str]:
        patch, _, error = parse_atomic_patch_text(
            json.dumps(candidate.semantic_payload, ensure_ascii=False),
            base_ir_digest=request_ir_digest(ir), gap=gap,
        )
        return patch, error

    @staticmethod
    def _candidate(operation_type: str, payload: dict[str, Any], summary: str) -> SemanticCandidate:
        canonical = {"operation_type": operation_type, "payload": payload}
        candidate_id = "cand:sha256:" + hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return SemanticCandidate(candidate_id, operation_type, payload, summary)

    def _event_candidates(self, ir: UnderstandingIR, requirement: RequirementSpec,
                          gap: GapRequest, catalog: CatalogSnapshot):
        demands = self._demands(requirement, ir)
        comparisons = [item for item in demands if item.demand_type == "comparison"]
        durations = [item for item in demands if item.demand_type == "duration_threshold"]
        if len(comparisons) != 1 or len(durations) != 1:
            return [], ["event_condition_or_duration_not_unique"]
        comparison, duration = comparisons[0], durations[0]
        field = next((item for item in ir.source_demands
                      if item.demand_id == comparison.field_anchor_id), None)
        field_ref = self._resolve_field_ref(
            str(field.attributes.get("field_text", field.text)) if field else "",
            gap, ir, catalog,
        )
        if not field_ref:
            return [], ["event_field_not_unique"]
        condition_unit = _canonical_unit(str(comparison.attributes.get("unit", "")))
        duration_unit = _duration_unit(str(duration.attributes.get("unit", "")))
        if not duration_unit:
            return [], ["event_duration_unit_unsupported"]
        payload = {
            "logic": "and",
            "conditions": [{
                "field_ref": field_ref,
                "operator": comparison.attributes.get("operator"),
                "value": comparison.attributes.get("value"),
                "unit": condition_unit,
                "right_ref": None,
                "multiplier": None,
            }],
            "duration_operator": duration.attributes.get("operator", "gt"),
            "duration_value": duration.attributes.get("value"),
            "duration_unit": duration_unit,
            "group_by": ["local_date"],
        }
        return [("add_event", payload, f"CONSECUTIVE({field_ref})")], []

    @staticmethod
    def _set_candidates(ir: UnderstandingIR, requirement: RequirementSpec, gap: GapRequest):
        refs = list(dict.fromkeys(gap.choice_domain.get("output_refs", [])))
        operations = list(dict.fromkeys(gap.choice_domain.get("set_operations", [])))
        if len(refs) != 2:
            return [], ["set_requires_two_distinct_outputs"]
        if len(operations) != 1:
            return [], ["set_operation_not_unique"]
        grain = "date" if all(
            next((item.output.grain for item in ir.derived_projections
                  if item.output.ref_id == ref), "") == "date" for ref in refs
        ) else "interval"
        payload = {"operation": operations[0], "inputs": refs, "granularity": grain}
        return [("add_set_operation", payload, f"{operations[0]}({', '.join(refs)})")], []

    def _aggregate_candidates(self, ir: UnderstandingIR, requirement: RequirementSpec,
                              gap: GapRequest, catalog: CatalogSnapshot,
                              lineage: OutputLineageIndex):
        target = self.targets.build(requirement, ir)
        if len(target.measure_fields) != 1:
            return [], ["aggregate_measure_not_unique"]
        measure_ref = self._resolve_field_ref(target.measure_fields[0], gap, ir, catalog)
        if not measure_ref:
            return [], ["aggregate_measure_not_bound"]
        functions = list(dict.fromkeys(gap.choice_domain.get("functions", [])))
        function = functions[0] if len(functions) == 1 else target.operation
        if function not in {"avg", "sum", "min", "max", "count", "stddev"}:
            return [], ["aggregate_function_not_supported"]
        value = _value_signature(measure_ref, ir, catalog, lineage)
        signature = DEFAULT_EXPRESSION_SIGNATURES.validate(
            function, [value], semantic_role="measure",
        )
        if not signature.valid:
            return [], [signature.diagnostic_code]
        set_refs = [
            item.output_ref.ref_id for item in ir.set_operations
            if item.output_ref and any(
                dependency.requirement_id in requirement.dependencies
                for dependency in ir.requirements
                if dependency.requirement_type == "set_operation"
            )
        ]
        if len(set_refs) > 1:
            return [], ["aggregate_scope_not_unique"]
        scope_ref = set_refs[0] if set_refs else None
        scope = "relation" if scope_ref else requirement.expected_scope or "global"
        payload = {
            "function": function,
            "input_ref": measure_ref,
            "scope": scope,
            "scope_ref": scope_ref,
            "group_by_refs": [scope_ref]
            if scope_ref and requirement.expected_output_grain == "day" else [],
        }
        return [("add_aggregate", payload,
                 f"{function}({measure_ref}) scope={scope_ref or scope}")], []

    def _calculation_candidates(self, ir: UnderstandingIR, requirement: RequirementSpec,
                                gap: GapRequest, catalog: CatalogSnapshot,
                                lineage: OutputLineageIndex):
        family = requirement.operator_family
        mapping = {
            "ratio": ("ratio", "ratio", 2),
            "difference": ("difference", "difference", 2),
            "correlation": ("correlation", "correlation", 2),
            "volatility": ("volatility", "stddev", 1),
            "maximum": ("maximum", "identity", 1),
        }
        if family not in mapping:
            return [], ["calculation_family_unsupported"]
        calculation_type, template, arity = mapping[family]
        refs = [
            item for item in self._dependency_output_refs(requirement, ir, lineage)
            if lineage.get(item) is not None
            and lineage.get(item).semantic_role in {"aggregate", "calculation", "value"}
            and lineage.get(item).result_type in {"number", "integer", "float"}
        ]
        if len(refs) != arity:
            field_names = [
                item.removeprefix("field:") for item in requirement.expected_inputs
                if item.startswith("field:")
            ]
            field_refs = [
                self._resolve_field_ref(item, gap, ir, catalog)
                for item in field_names
            ]
            if len(field_refs) == arity and all(field_refs) and len(set(field_refs)) == arity:
                refs = field_refs
        if len(refs) != arity:
            return [], ["calculation_inputs_not_unique"]
        signature = DEFAULT_EXPRESSION_SIGNATURES.validate(
            template,
            [_value_signature(item, ir, catalog, lineage) for item in refs],
            semantic_role="operand",
        )
        if not signature.valid:
            return [], [signature.diagnostic_code]
        ordered = [refs]
        if family == "ratio" and not self._ratio_order_is_explicit(requirement, ir):
            ordered = [refs, list(reversed(refs))]
        payloads = []
        for values in ordered:
            payloads.append(("add_calculation", {
                "calculation_type": calculation_type,
                "formula_template": template,
                "operands": values,
                "constants": [],
            }, f"{template}({', '.join(values)})"))
        return payloads, []

    @staticmethod
    def _ratio_order_is_explicit(requirement: RequirementSpec,
                                 ir: UnderstandingIR) -> bool:
        if re.search(r"占|除以|相除|/", requirement.text):
            return True
        by_id = {item.requirement_id: item for item in ir.requirements}
        dependencies = [
            by_id[item] for item in requirement.dependencies
            if item in by_id and by_id[item].span is not None
        ]
        dependencies.sort(key=lambda item: item.span.start)
        if len(dependencies) != 2 or requirement.span is None:
            return False
        left, right = dependencies
        if left.span.end > right.span.start or right.span.end > requirement.span.end:
            return False
        connector = ir.query.normalized[left.span.end:right.span.start]
        suffix = ir.query.normalized[right.span.end:requirement.span.end]
        return bool(
            re.search(r"与|和|及|/", connector)
            and "比值" in suffix
            and "两者" not in suffix
        )

    @staticmethod
    def _reference_candidates(ir: UnderstandingIR, requirement: RequirementSpec,
                              gap: GapRequest, lineage: OutputLineageIndex):
        refs = list(dict.fromkeys(gap.choice_domain.get("output_refs", [])))
        raw = " ".join((requirement.text, gap.query_slice,
                        str(gap.existing_node_summary.get("raw", ""))))
        if "日期" in raw:
            refs = [item for item in refs if lineage.get(item) is not None
                    and lineage.get(item).semantic_role == "date"]
        elif re.search(r"时间段|时段|区间", raw):
            refs = [item for item in refs if lineage.get(item) is not None
                    and lineage.get(item).semantic_role in {"event_interval", "set"}
                    and lineage.get(item).grain == "interval"]
        dependency_refs = SemanticCandidateCompiler._dependency_output_refs(
            requirement, ir, lineage,
        )
        if dependency_refs:
            narrowed = [item for item in refs if item in dependency_refs]
            if narrowed:
                refs = narrowed
        if not refs:
            return [], ["reference_target_not_unique"]
        return [
            ("resolve_reference", {"target_ref": item}, f"reference={item}")
            for item in refs
        ], []

    @staticmethod
    def _demands(requirement: RequirementSpec, ir: UnderstandingIR) -> list[SourceDemand]:
        by_id = {item.demand_id: item for item in ir.source_demands}
        pending = list(requirement.expected_attributes.get("source_demand_ids", []))
        seen = set()
        while pending:
            identifier = pending.pop()
            if identifier in seen or identifier not in by_id:
                continue
            seen.add(identifier)
            pending.extend(by_id[identifier].dependencies)
        return [by_id[item] for item in seen]

    @staticmethod
    def _resolve_field_ref(raw_name: str, gap: GapRequest, ir: UnderstandingIR,
                           catalog: CatalogSnapshot) -> str:
        catalog_values = [
            item for item in catalog.resolve_field(raw_name)
            if item in gap.allowed_catalog_symbols
        ]
        hypotheses = [
            hypothesis for hypothesis in ir.schema_hypotheses
            if hypothesis.hypothesis_id in gap.allowed_hypothesis_symbols
        ]
        exact = [
            hypothesis for hypothesis in hypotheses
            if hypothesis_matches_field(
                raw_name, hypothesis.raw_name, hypothesis.aliases,
                hypothesis.description, allow_suffix=False,
            )
        ]
        matched = exact or [
            hypothesis for hypothesis in hypotheses
            if hypothesis_matches_field(
                raw_name, hypothesis.raw_name, hypothesis.aliases,
                hypothesis.description,
            )
        ]
        # An inline query schema declaration is stronger evidence than a
        # generic Catalog alias. Keep Catalog values only when no exact local
        # declaration exists.
        values = ([] if exact else catalog_values)
        values.extend(item.hypothesis_id for item in matched)
        return values[0] if len(set(values)) == 1 else ""

    @staticmethod
    def _dependency_output_refs(requirement: RequirementSpec, ir: UnderstandingIR,
                                lineage: OutputLineageIndex) -> list[str]:
        coverage = {item.requirement_id: item for item in ir.coverage}
        requirements = {item.requirement_id: item for item in ir.requirements}
        refs = []
        for dependency_id in sorted(
            requirement.dependencies,
            key=lambda item: requirements[item].span.start
            if item in requirements and requirements[item].span else -1,
        ):
            item = coverage.get(dependency_id)
            if item is None:
                continue
            for node_id in item.covered_by:
                matches = [entry.ref_id for entry in lineage.entries.values()
                           if entry.producer_id == node_id or entry.ref_id == node_id]
                refs.extend(matches)
        return list(dict.fromkeys(refs))


def parse_candidate_decision(text: str, menu: CandidateMenu) -> CandidateDecision:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        decisions = re.findall(r'"decision"\s*:\s*"(select|none_of_above|ambiguous)"', value)
        candidates = re.findall(r'"candidate_id"\s*:\s*"(cand:sha256:[0-9a-f]{64})"', value)
        if len(decisions) != 1 or len(candidates) > 1:
            return CandidateDecision("invalid", error="json_syntax_failure")
        if decisions[0] == "select" and len(candidates) != 1:
            return CandidateDecision("invalid", error="json_syntax_failure")
        if decisions[0] != "select" and candidates:
            return CandidateDecision("invalid", error="json_syntax_failure")
        payload = {"decision": decisions[0]}
        if candidates:
            payload["candidate_id"] = candidates[0]
    if not isinstance(payload, dict) or set(payload) - {"decision", "candidate_id"}:
        return CandidateDecision("invalid", error="choice_schema_failure")
    decision = payload.get("decision")
    if decision not in {"select", "none_of_above", "ambiguous"}:
        return CandidateDecision("invalid", error="choice_schema_failure")
    candidate_id = str(payload.get("candidate_id", ""))
    if decision == "select":
        if not candidate_id or menu.candidate(candidate_id) is None:
            return CandidateDecision("invalid", candidate_id, "candidate_not_allowed")
    elif candidate_id:
        return CandidateDecision("invalid", candidate_id, "choice_schema_failure")
    return CandidateDecision(decision, candidate_id)


def _value_signature(identifier: str, ir: UnderstandingIR, catalog: CatalogSnapshot,
                     lineage: OutputLineageIndex) -> ValueSignature:
    field = catalog.field(identifier)
    if field:
        return ValueSignature(field.data_type, field.unit, "scalar", "measure")
    hypothesis = next((item for item in ir.schema_hypotheses
                       if item.hypothesis_id == identifier), None)
    if hypothesis:
        return ValueSignature(hypothesis.declared_type, hypothesis.unit, "scalar", "measure")
    output = lineage.get(identifier)
    if output:
        return ValueSignature(output.result_type, output.unit, output.shape, output.semantic_role)
    return ValueSignature("unknown")


def _canonical_unit(raw: str) -> str | None:
    return {
        "℃": "celsius", "°C": "celsius", "%": "percent",
        "MPa": "mpa", "kPa": "kpa", "Pa": "pa", "W": "watt",
        "kW": "kilowatt", "V": "volt", "m/s²": "m/s2", "m/s2": "m/s2",
        "km/h": "km/h", "m/s": "m/s",
    }.get(raw, raw.lower() if raw else None)


def _duration_unit(raw: str) -> str:
    return {
        "毫秒": "second", "秒": "second", "秒钟": "second", "s": "second",
        "分钟": "minute", "分": "minute", "min": "minute",
        "小时": "hour", "时": "hour", "h": "hour", "天": "day", "日": "day",
    }.get(raw, "")


def _digest(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
