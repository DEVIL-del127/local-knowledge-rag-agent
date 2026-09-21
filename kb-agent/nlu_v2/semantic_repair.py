"""Candidate-only semantic repair transactions and acceptance evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .catalog import CatalogSnapshot
from .coverage import CoverageMatcher
from .models import PatchReport, SemanticEffectReport, UnderstandingIR
from .patch_protocol import GapRequest, request_ir_digest
from .validator import IRValidator


@dataclass(frozen=True, slots=True)
class CompilationSnapshot:
    """The immutable compiler state against which a patch was requested."""

    ir_digest: str
    catalog_version: str
    source_demand_digest: str

    @classmethod
    def capture(cls, ir: UnderstandingIR) -> "CompilationSnapshot":
        source_payload = json.dumps([item.to_dict() for item in ir.source_demands],
                                    ensure_ascii=False, sort_keys=True, default=str)
        return cls(
            ir_digest=request_ir_digest(ir), catalog_version=ir.catalog_version,
            source_demand_digest=hashlib.sha256(source_payload.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class RepairUnit:
    """A connected group of gaps that must be assessed as one semantic unit."""

    unit_id: str
    gaps: tuple[GapRequest, ...]
    requirement_ids: tuple[str, ...]
    allowed_operations: tuple[str, ...]

    @classmethod
    def from_gaps(cls, gaps: Iterable[GapRequest]) -> list["RepairUnit"]:
        # A unit is the connected component of same-clause / same-output / DAG
        # dependent gaps.  It is deliberately not "the first gap".
        groups: list[list[GapRequest]] = []
        for gap in gaps:
            related = [group for group in groups if any(
                _gaps_connected(gap, member) for member in group
            )]
            if not related:
                groups.append([gap])
                continue
            merged = [gap]
            for group in related:
                merged.extend(group)
                groups.remove(group)
            groups.append(merged)
        result = []
        for items in groups:
            identity = "|".join(item.gap_id for item in items)
            operations = tuple(sorted({op for item in items for op in item.allowed_operations}))
            result.append(cls(
                unit_id="repair_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                gaps=tuple(items),
                requirement_ids=tuple(item.requirement_id for item in items if item.requirement_id),
                allowed_operations=operations,
            ))
        return result


@dataclass(frozen=True, slots=True)
class RepairContract:
    """Immutable limits for one candidate transaction."""

    snapshot: CompilationSnapshot
    repair_unit: RepairUnit
    acceptance_profile: str  # concrete / clarification
    allowed_anchor_ids: tuple[str, ...] = ()
    allowed_field_ids: tuple[str, ...] = ()
    allowed_operator_ids: tuple[str, ...] = ()
    allowed_unit_ids: tuple[str, ...] = ()
    allowed_input_refs: tuple[str, ...] = ()
    allowed_output_types: tuple[str, ...] = ()
    allowed_write_paths: tuple[str, ...] = ()
    expected_output_shape: str = ""
    protected_rule_node_ids: tuple[str, ...] = ()
    target_requirement_ids: tuple[str, ...] = ()

    @property
    def gap_ids(self) -> set[str]:
        return {item.gap_id for item in self.repair_unit.gaps}

    @classmethod
    def for_unit(cls, snapshot, unit, profile, ir, catalog):
        target_requirements = [item for item in ir.requirements if item.requirement_id in unit.requirement_ids]
        output_shapes = {item.expected_output_shape for item in target_requirements if item.expected_output_shape}
        output_types = {
            str(item.expected_attributes.get("expected_output_type", ""))
            for item in target_requirements if item.expected_attributes.get("expected_output_type")
        }
        anchor_ids = {value for gap in unit.gaps for value in gap.allowed_anchor_ids}
        input_refs = {"source_relation.rows"}
        input_refs.update(value for gap in unit.gaps for value in gap.available_output_refs)
        write_paths = {
            _operation_write_path(operation) for operation in unit.allowed_operations
            if _operation_write_path(operation)
        }
        catalog_units = {
            str(field.unit) for source in catalog.sources for field in source.fields if field.unit
        }
        catalog_units.update(
            str(metric.unit) for source in catalog.sources
            for metric in source.derived_metrics if metric.unit
        )
        catalog_units.update({
            "celsius", "fahrenheit", "m/s", "km/h", "percent", "currency",
            "count", "ratio", "second", "minute", "hour", "day",
        })
        return cls(
            snapshot, unit, profile,
            allowed_anchor_ids=tuple(sorted(anchor_ids)),
            allowed_field_ids=tuple(sorted({*[
                value for gap in unit.gaps for value in gap.allowed_catalog_symbols
            ], *[value for gap in unit.gaps for value in gap.allowed_hypothesis_symbols]})),
            allowed_operator_ids=tuple(unit.allowed_operations),
            allowed_unit_ids=tuple(sorted(catalog_units)),
            allowed_input_refs=tuple(sorted(input_refs)),
            allowed_output_types=tuple(sorted(output_types)) or ("number", "boolean", "relation", "event_interval", "date", "string", "datetime", "duration"),
            allowed_write_paths=tuple(sorted(write_paths)),
            expected_output_shape=next(iter(output_shapes)) if len(output_shapes) == 1 else "",
            protected_rule_node_ids=tuple(_rule_node_ids(ir)),
            target_requirement_ids=unit.requirement_ids,
        )


@dataclass(slots=True)
class GateDecision:
    accepted: bool
    effect: SemanticEffectReport
    candidate: UnderstandingIR


class SemanticAcceptanceGate:
    """Accept only effective, non-regressive candidate IRs.

    This gate is intentionally independent from Patch schema validation.  JSON
    validity means only that a candidate can be considered; this class decides
    whether it has semantic effect and may replace the canonical IR.
    """

    def __init__(self, *, validator: IRValidator | None = None,
                 coverage_matcher: CoverageMatcher | None = None):
        self.validator = validator or IRValidator()
        self.coverage_matcher = coverage_matcher or CoverageMatcher()

    def evaluate(self, baseline: UnderstandingIR, candidate: UnderstandingIR,
                 patch_report: PatchReport, contract: RepairContract,
                 catalog: CatalogSnapshot,
                 refresh: Callable[[UnderstandingIR], None] | None = None) -> GateDecision:
        before = copy.deepcopy(baseline)
        after = copy.deepcopy(candidate)
        if refresh:
            refresh(after)
        self.coverage_matcher.apply(before)
        self.coverage_matcher.apply(after)
        before_validation = self.validator.validate(before, catalog)
        after_validation = self.validator.validate(after, catalog)
        before_coverage = _coverage_map(before)
        after_coverage = _coverage_map(after)
        closed_requirements = sorted(
            requirement_id for requirement_id, status in after_coverage.items()
            if status == "satisfied" and before_coverage.get(requirement_id) != "satisfied"
        )
        regressed = sorted(
            requirement_id for requirement_id, status in before_coverage.items()
            if status == "satisfied" and after_coverage.get(requirement_id) != "satisfied"
        )
        baseline_errors = {(item.code, item.message) for item in before_validation.errors
                           if item.severity == "error"}
        profile = _profile(patch_report)
        new_errors = sorted({item.code for item in after_validation.errors if item.severity == "error"
                             and (item.code, item.message) not in baseline_errors})
        # An accepted clarification intentionally materializes as an open
        # semantic ambiguity and must keep execution blocked.  It is not a
        # candidate regression; all other newly introduced errors still reject
        # the transaction.
        if profile == "clarification":
            new_errors = [item for item in new_errors if item != "semantic_ambiguity"]
        effect = SemanticEffectReport(
            acceptance_profile=profile,
            closed_requirement_ids=closed_requirements,
            closed_gap_ids=list(patch_report.closed_gap_ids),
            added_nodes_and_edges=_added_nodes(baseline, candidate),
            resolved_diagnostics=_resolved_diagnostics(before_validation, after_validation),
            new_diagnostics=new_errors,
            coverage_before=before_coverage,
            coverage_after=after_coverage,
            base_digest=contract.snapshot.ir_digest,
            candidate_digest=request_ir_digest(after),
            concrete_gain=patch_report.concrete_gain_count,
            clarification_gain=patch_report.clarification_gain_count,
        )
        reason = self._reject_reason(
            profile, contract, patch_report, closed_requirements, regressed, new_errors,
            baseline, candidate, after_validation.executable,
        )
        if reason:
            effect.commit_or_reject_reason = reason
            return GateDecision(False, effect, baseline)
        effect.commit_or_reject_reason = "committed_" + profile
        effect.final_digest = effect.candidate_digest
        return GateDecision(True, effect, after)

    @staticmethod
    def _reject_reason(profile: str, contract: RepairContract, report: PatchReport,
                       closed_requirements: list[str], regressed: list[str],
                       new_errors: list[str], baseline: UnderstandingIR,
                       candidate: UnderstandingIR, candidate_executable: bool) -> str:
        if profile == "none":
            return "no_gain"
        if profile == "mixed":
            return "mixed_concrete_and_clarification_patch"
        if not set(report.closed_gap_ids) & contract.gap_ids:
            return "no_target_gap_closed"
        if regressed:
            return "covered_requirement_regressed:" + ",".join(regressed)
        if new_errors:
            return "new_fatal_validation_errors:" + ",".join(new_errors)
        if profile == "concrete":
            if contract.acceptance_profile != "concrete":
                return "unexpected_concrete_profile"
            # Compatibility callers that use the raw rule extractor do not
            # have a RequirementIR yet.  The engine always creates one before
            # calling the LLM, where the stronger coverage-gain rule applies.
            if baseline.requirements and not closed_requirements:
                return "no_requirement_coverage_gain"
            if contract.repair_unit.requirement_ids and not (
                set(closed_requirements) & set(contract.repair_unit.requirement_ids)
            ):
                return "target_requirement_not_closed"
            return ""
        if contract.acceptance_profile != "clarification":
            return "unexpected_clarification_profile"
        if candidate_executable:
            return "clarification_must_remain_execution_blocked"
        if _executable_shape(baseline) != _executable_shape(candidate):
            return "clarification_added_executable_ir"
        return ""


def contract_violation(contract: RepairContract, patch, ir: UnderstandingIR) -> str:
    """Reject a Patch before mutation when it escapes its RepairContract."""
    demands = {item.demand_id: item for item in ir.source_demands}
    gaps = {item.gap_id: item for item in contract.repair_unit.gaps}
    for operation in patch.operations:
        if operation.operation_type not in contract.allowed_operator_ids:
            return "repair_contract_operator_out_of_scope"
        if operation.gap_id not in contract.gap_ids:
            return "repair_contract_gap_out_of_scope"
        if _operation_write_path(operation.operation_type) not in contract.allowed_write_paths:
            return "repair_contract_write_path_out_of_scope"
        gap = gaps[operation.gap_id]
        anchors = getattr(operation, "anchor_ids", []) or []
        # Compatibility callers may start from a pre-ledger IR.  The engine
        # always has source demands; there, an absent local anchor is fatal.
        if ir.source_demands and not gap.allowed_anchor_ids:
            return "repair_contract_no_anchor_available"
        if gap.allowed_anchor_ids and not anchors:
            return "repair_contract_anchor_required"
        # V1 fixtures carry text evidence only.  When an operation explicitly
        # supplies anchor ids (the v2 contract form), enforce them here; text
        # spans remain separately validated by the Patch applier.
        for anchor_id in anchors:
            if (anchor_id not in gap.allowed_anchor_ids
                    or anchor_id not in contract.allowed_anchor_ids or anchor_id not in demands):
                return "repair_contract_anchor_out_of_scope"
        for identifier in _patch_identifiers(operation):
            if identifier and identifier not in contract.allowed_field_ids and identifier not in contract.allowed_input_refs:
                return "repair_contract_input_out_of_scope"
        for unit in _patch_units(operation):
            if unit and unit not in contract.allowed_unit_ids:
                return "repair_contract_unit_out_of_scope"
        for result_type in _patch_result_types(operation):
            if result_type and result_type != "unknown" and result_type not in contract.allowed_output_types:
                return "repair_contract_output_type_out_of_scope"
        output_shape = _operation_output_shape(operation)
        if contract.expected_output_shape and output_shape and output_shape != contract.expected_output_shape:
            return "repair_contract_output_shape_out_of_scope"
        if set(_patch_produced_node_ids(operation)) & set(contract.protected_rule_node_ids):
            return "repair_contract_protected_node_write"
    return ""


def _operation_write_path(operation_type: str) -> str:
    return {
        "add_event": "events", "add_set_operation": "set_operations",
        "add_aggregate": "aggregates", "add_comparison": "comparisons",
        "add_calculation": "calculations", "resolve_reference": "references",
        "add_ambiguity": "ambiguities", "add_sequence": "sequences",
        "add_turn_directive": "turn_directives",
    }.get(operation_type, "")


def _operation_output_shape(operation) -> str:
    if operation.operation_type == "add_event":
        return "event_set"
    if operation.operation_type == "add_set_operation":
        return "event_set"
    if operation.operation_type == "add_aggregate":
        return "relation" if operation.scope == "relation" else "scalar"
    if operation.operation_type in {"add_comparison", "add_calculation"}:
        return "scalar"
    if operation.operation_type == "add_sequence":
        return "event_set"
    return ""


def _patch_produced_node_ids(operation) -> list[str]:
    if operation.operation_type == "add_event":
        return [operation.event.event_id]
    if operation.operation_type == "add_aggregate":
        return [operation.aggregate_id]
    if operation.operation_type == "add_comparison":
        return [operation.comparison_id]
    if operation.operation_type == "add_calculation":
        return [f"calculation_{operation.operation_id}"]
    if operation.operation_type == "add_set_operation":
        return [f"set_{operation.operation_id}"]
    if operation.operation_type == "add_sequence":
        return [operation.sequence_id]
    return []


def _patch_identifiers(value) -> list[str]:
    """Collect only semantic IDs from typed Patch models, not textual IDs."""
    result = []
    if getattr(value, "kind", "") in {"catalog_symbol", "hypothesis_symbol", "output_ref"}:
        result.append(getattr(value, "identifier", ""))
    if hasattr(value, "field_id"):
        result.append(value.field_id)
    if hasattr(value, "target_ref"):
        result.append(value.target_ref)
    if hasattr(value, "scope_ref"):
        result.append(value.scope_ref or "")
    if hasattr(value, "inputs") and isinstance(value.inputs, list):
        result.extend(item for item in value.inputs if isinstance(item, str))
    for name in ("input", "left", "right", "expression", "right_expression", "rank_by"):
        child = getattr(value, name, None)
        if child is not None:
            result.extend(_patch_identifiers(child))
    for child in getattr(value, "arguments", []) or []:
        result.extend(_patch_identifiers(child))
    for child in getattr(value, "conditions", []) or []:
        result.extend(_patch_identifiers(child))
    return result


def _patch_units(value) -> list[str]:
    result = [str(getattr(value, "unit", "") or "")]
    for name in ("input", "left", "right", "expression", "right_expression"):
        child = getattr(value, name, None)
        if child is not None:
            result.extend(_patch_units(child))
    for child in [*(getattr(value, "arguments", []) or []),
                  *(getattr(value, "conditions", []) or [])]:
        result.extend(_patch_units(child))
    return result


def _patch_result_types(value) -> list[str]:
    result = [str(getattr(value, "result_type", "") or "")]
    for name in ("input", "left", "right", "expression", "right_expression"):
        child = getattr(value, name, None)
        if child is not None:
            result.extend(_patch_result_types(child))
    for child in [*(getattr(value, "arguments", []) or []),
                  *(getattr(value, "conditions", []) or [])]:
        result.extend(_patch_result_types(child))
    return result


def _profile(report: PatchReport) -> str:
    if report.concrete_gain_count and report.clarification_gain_count:
        return "mixed"
    if report.concrete_gain_count:
        return "concrete"
    if report.clarification_gain_count:
        return "clarification"
    return "none"


def _gaps_connected(left: GapRequest, right: GapRequest) -> bool:
    if left.requirement_id and left.requirement_id == right.requirement_id:
        return True
    if (left.requirement_id and left.requirement_id in right.requirement_dependency_ids) or (
            right.requirement_id and right.requirement_id in left.requirement_dependency_ids):
        return True
    if left.target_node_id and left.target_node_id == right.target_node_id:
        return True
    # available_output_refs are a local allow-list for model bindings, not DAG
    # edges.  In particular every gap can see source_relation.rows; using that
    # shared root used to collapse unrelated clauses into one repair request.
    analytic = {"missing_event", "missing_set_inputs", "missing_formula_input", "missing_aggregate_scope"}
    if left.clause_id and left.clause_id == right.clause_id and {
        left.gap_type, right.gap_type
    } <= analytic:
        return True
    return False


def _output_refs(ir: UnderstandingIR) -> list[str]:
    values = ["source_relation.rows"]
    values.extend(item.output_ref.ref_id for item in ir.events if item.output_ref)
    values.extend(item.output_ref.ref_id for item in ir.set_operations if item.output_ref)
    values.extend(item.aggregate.output.ref_id for item in ir.aggregates)
    values.extend(item.output.ref_id for item in ir.comparisons)
    values.extend(item.output.ref_id for item in ir.arithmetic)
    values.extend(item.output.ref_id for item in ir.conversions)
    values.extend(item.output.ref_id for item in ir.sequences)
    values.extend(item.output.ref_id for item in ir.group_by_operations)
    values.extend(item.output.ref_id for item in ir.top_k_operations)
    values.extend(item.output.ref_id for item in ir.window_aggregates)
    values.extend(item.output.ref_id for item in ir.cumulative_counts)
    values.extend(item.output.ref_id for item in ir.cumulative_durations)
    values.extend(item.output.ref_id for item in ir.calculations if item.output)
    return sorted(set(values))


def _rule_node_ids(ir: UnderstandingIR) -> list[str]:
    return [
        *[item.event_id for item in ir.events],
        *[item.output_name for item in ir.set_operations],
        *[item.aggregate.aggregate_id for item in ir.aggregates],
        *[item.group_id for item in ir.group_by_operations],
        *[item.top_k_id for item in ir.top_k_operations],
        *[item.window_id for item in ir.window_aggregates],
        *[item.count_id for item in ir.cumulative_counts],
        *[item.duration_id for item in ir.cumulative_durations],
        *[item.comparison_id for item in ir.comparisons],
        *[item.arithmetic_id for item in ir.arithmetic],
        *[item.sequence_id for item in ir.sequences],
        *[item.calculation_id for item in ir.calculations if item.calculation_id],
    ]


def _coverage_map(ir: UnderstandingIR) -> dict[str, str]:
    return {item.requirement_id: item.status for item in ir.coverage}


def _resolved_diagnostics(before, after) -> list[str]:
    current = {item.code for item in after.errors}
    return sorted({item.code for item in before.errors if item.code not in current})


def _added_nodes(before: UnderstandingIR, after: UnderstandingIR) -> list[str]:
    values: list[tuple[str, int, int]] = [
        ("events", len(before.events), len(after.events)),
        ("set_operations", len(before.set_operations), len(after.set_operations)),
        ("aggregates", len(before.aggregates), len(after.aggregates)),
        ("comparisons", len(before.comparisons), len(after.comparisons)),
        ("calculations", len(before.calculations), len(after.calculations)),
        ("sequences", len(before.sequences), len(after.sequences)),
        ("turn_directives", len(before.turn_directives), len(after.turn_directives)),
        ("ambiguities", len(before.ambiguities), len(after.ambiguities)),
    ]
    return [f"{kind}:{after_count - before_count}" for kind, before_count, after_count in values
            if after_count > before_count]


def _executable_shape(ir: UnderstandingIR) -> tuple[int, ...]:
    return (
        len(ir.events), len(ir.set_operations), len(ir.aggregates), len(ir.comparisons),
        len(ir.calculations), len(ir.arithmetic), len(ir.conversions), len(ir.ordered_folds),
        len(ir.sequences), len(ir.turn_directives),
    )
