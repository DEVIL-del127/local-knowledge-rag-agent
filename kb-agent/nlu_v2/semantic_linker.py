"""Deterministic dataflow closure for already-understood semantic nodes."""
from __future__ import annotations

import hashlib
import re

from .models import (
    DerivedProjectionSpec,
    ExpressionNode,
    OutputRef,
    Provenance,
    SetOperationSpec,
    SourceSpan,
    UnderstandingIR,
)


_SET_CUE = re.compile(r"交集|并集|差集|重合(?:时段|时间段|日期|区间)?")


class DeterministicSemanticLinker:
    """Add missing producer references without inventing business facts."""

    def link(self, ir: UnderstandingIR) -> None:
        self._ensure_event_outputs(ir)
        self._ensure_date_projections(ir)
        self._ensure_explicit_set(ir)
        self._rebind_scoped_aggregates(ir)
        self._remove_closed_set_diagnostics(ir)

    @staticmethod
    def _ensure_event_outputs(ir: UnderstandingIR) -> None:
        for event in ir.events:
            if event.output_ref is None:
                event.output_ref = OutputRef(
                    event.event_id, "intervals", "event_interval",
                    shape="event_set", grain="interval", keys=["local_date"],
                )
            if not event.output_name:
                event.output_name = event.output_ref.ref_id

    @staticmethod
    def _ensure_date_projections(ir: UnderstandingIR) -> None:
        existing = {item.input_ref.ref_id for item in ir.derived_projections}
        outputs = [event.output_ref for event in ir.events if event.output_ref]
        outputs.extend(item.output for item in ir.relation_filters)
        provenance = {
            event.output_ref.ref_id: event.provenance
            for event in ir.events if event.output_ref
        }
        provenance.update({item.output.ref_id: item.provenance for item in ir.relation_filters})
        for output in outputs:
            if output.ref_id in existing:
                continue
            projection_id = f"project_{output.producer_id}_date"
            ir.derived_projections.append(DerivedProjectionSpec(
                projection_id=projection_id,
                function="local_date",
                input_ref=output,
                output=OutputRef(
                    projection_id, "dates", "date", shape="event_set",
                    grain="date", keys=["local_date"], fields=["local_date"],
                ),
                provenance=list(provenance.get(output.ref_id, [])),
            ))
            existing.add(output.ref_id)

    @staticmethod
    def _ensure_explicit_set(ir: UnderstandingIR) -> None:
        requirement = next(
            (item for item in ir.requirements if item.requirement_type == "set_operation"),
            None,
        )
        if requirement is None:
            return
        match = _SET_CUE.search(ir.query.normalized)
        if match is None:
            return
        operation = {
            "交集": "intersection", "重合": "intersection",
            "并集": "union", "差集": "difference",
        }[next(key for key in ("交集", "重合", "并集", "差集") if key in match.group(0))]
        local = ir.query.normalized[match.start():match.end() + 8]
        granularity = "date" if "日期" in local else "interval"
        replacements = {
            item.input_ref.ref_id: item.output for item in ir.relation_filters
        }
        effective = [
            replacements.get(item.output_ref.ref_id, item.output_ref)
            for item in ir.events if item.output_ref
        ]
        if granularity == "date":
            projections = {item.input_ref.ref_id: item.output for item in ir.derived_projections}
            refs = [projections.get(item.ref_id) for item in effective]
        else:
            refs = effective
        refs = _unique_refs(refs)
        if len(refs) != 2:
            return
        existing = ir.set_operations[0] if ir.set_operations else None
        if existing is not None:
            existing.operation = operation
            existing.inputs = [item.ref_id for item in refs]
            existing.input_refs = refs
            existing.granularity = granularity
            if existing.output_ref is None or existing.output_ref.grain != granularity:
                existing.output_ref = _set_output(operation, granularity, requirement.requirement_id)
            return
        output = _set_output(operation, granularity, requirement.requirement_id)
        span = requirement.span or SourceSpan(match.start(), match.end(), match.group(0))
        ir.set_operations.append(SetOperationSpec(
            operation=operation,
            inputs=[item.ref_id for item in refs],
            output_name=output.ref_id,
            granularity=granularity,
            input_refs=refs,
            output_ref=output,
            provenance=[Provenance(source="requirement", rule="deterministic_set_link", span=span)],
        ))

    @staticmethod
    def _rebind_scoped_aggregates(ir: UnderstandingIR) -> None:
        """Bind aggregates only through an explicit Requirement dependency."""
        if not ir.set_operations or not ir.set_operations[0].output_ref:
            return
        set_ref = ir.set_operations[0].output_ref
        set_requirement = next(
            (item for item in ir.requirements if item.requirement_type == "set_operation"),
            None,
        )
        if set_requirement is None:
            return
        for item in ir.aggregates:
            span = next((value.span for value in item.aggregate.provenance if value.span), None)
            if span is None or item.scope == "global":
                continue
            requirements = [
                requirement for requirement in ir.requirements
                if requirement.requirement_type in {"aggregate", "scoped_aggregate"}
                and requirement.span is not None
                and requirement.span.start < span.end and span.start < requirement.span.end
            ]
            if len(requirements) != 1:
                continue
            dependency_proven = set_requirement.requirement_id in requirements[0].dependencies
            context_start = set_requirement.span.start if set_requirement.span else 0
            context = ir.query.normalized[context_start:span.end]
            explicitly_repeated = len(_SET_CUE.findall(context)) >= 2
            if not dependency_proven and not explicitly_repeated:
                continue
            requirement = requirements[0]
            item.scope = (requirement.expected_scope
                          or str(requirement.expected_attributes.get("scope", ""))
                          or "relation")
            item.scope_ref = set_ref
            expected_shape = (requirement.expected_output_shape
                              or str(requirement.expected_attributes.get(
                                  "expected_output_shape", ""
                              )))
            expected_grain = requirement.expected_output_grain or "scalar"
            grouped = expected_shape == "relation" or expected_grain != "scalar"
            if grouped:
                item.aggregate.output.shape = "relation"
                item.aggregate.output.grain = expected_grain
                item.aggregate.output.keys = list(set_ref.keys)
                if not item.group_by:
                    item.group_by = [ExpressionNode(
                        "output_ref", reference=set_ref.ref_id,
                        result_type="date", provenance=list(item.aggregate.provenance),
                    )]
            else:
                item.aggregate.output.shape = "scalar"
                item.aggregate.output.grain = "scalar"
                item.aggregate.output.keys = []

    @staticmethod
    def _remove_closed_set_diagnostics(ir: UnderstandingIR) -> None:
        if not ir.set_operations:
            return
        ir.unresolved = [item for item in ir.unresolved if item.code != "set_inputs_unbound"]


def _set_output(operation: str, granularity: str, requirement_id: str) -> OutputRef:
    suffix = hashlib.sha1(requirement_id.encode("utf-8")).hexdigest()[:10]
    result_type = "date" if granularity == "date" else "event_interval"
    return OutputRef(
        f"set_{operation}_{suffix}", "rows", result_type,
        shape="event_set", grain=granularity,
        keys=["local_date"] if granularity == "date" else [],
        fields=["local_date"] if granularity == "date" else [],
    )


def _unique_refs(refs: list[OutputRef | None]) -> list[OutputRef]:
    result: list[OutputRef] = []
    seen: set[str] = set()
    for item in refs:
        if item is not None and item.ref_id not in seen:
            seen.add(item.ref_id)
            result.append(item)
    return result
