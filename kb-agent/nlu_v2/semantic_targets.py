"""Derived semantic target and OutputRef lineage views over RequestIR."""
from __future__ import annotations

from dataclasses import dataclass

from .merge import walk_predicates
from .models import ExpressionNode, OutputRef, RequirementSpec, SourceSpan, UnderstandingIR
from .role_grounding import RoleGroundingAnalyzer, RoleGroundingProof


@dataclass(frozen=True, slots=True)
class OutputLineage:
    ref_id: str
    producer_id: str
    result_type: str
    unit: str | None
    shape: str
    grain: str
    source_fields: tuple[str, ...] = ()
    source_anchor_ids: tuple[str, ...] = ()
    scope_ancestors: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()
    semantic_role: str = "value"


class OutputLineageIndex:
    """Compute immutable lineage without changing the OutputRef wire schema."""

    def __init__(self, entries: dict[str, OutputLineage]):
        self.entries = entries

    def get(self, ref_id: str) -> OutputLineage | None:
        return self.entries.get(ref_id)

    @classmethod
    def build(cls, ir: UnderstandingIR) -> "OutputLineageIndex":
        entries: dict[str, OutputLineage] = {}
        demands = ir.source_demands
        hypothesis_aliases = {
            item.hypothesis_id: tuple(dict.fromkeys([item.raw_name, *item.aliases]))
            for item in ir.schema_hypotheses
        }

        def anchor_ids(spans: list[SourceSpan]) -> tuple[str, ...]:
            return tuple(sorted({
                demand.demand_id for demand in demands
                if any(span.start < demand.span.end and demand.span.start < span.end for span in spans)
            }))

        def add(output: OutputRef | None, *, fields=(), spans=(), inputs=(), role="value"):
            if output is None:
                return
            inherited_fields = set(fields)
            inherited_anchors = set(anchor_ids(list(spans)))
            scopes = set()
            for ref_id in inputs:
                parent = entries.get(ref_id)
                if parent is None:
                    continue
                inherited_fields.update(parent.source_fields)
                inherited_anchors.update(parent.source_anchor_ids)
                scopes.add(ref_id)
                scopes.update(parent.scope_ancestors)
            entries[output.ref_id] = OutputLineage(
                output.ref_id, output.producer_id, output.result_type, output.unit,
                output.shape, output.grain, tuple(sorted(inherited_fields)),
                tuple(sorted(inherited_anchors)), tuple(sorted(scopes)),
                tuple(inputs), role,
            )

        for event in ir.events:
            symbols = [node.field for node in walk_predicates(event.condition) if node.field]
            add(event.output_ref, fields=_symbol_ids(symbols, hypothesis_aliases),
                spans=_provenance_spans(event),
                role="event_interval")
        for item in ir.relation_filters:
            symbols = [node.field for node in walk_predicates(item.predicate) if node.field]
            add(item.output, fields=_symbol_ids(symbols, hypothesis_aliases),
                spans=_provenance_spans(item), inputs=(item.input_ref.ref_id,),
                role="event_interval")
        for item in ir.derived_projections:
            add(item.output, spans=_provenance_spans(item), inputs=(item.input_ref.ref_id,),
                role="date" if item.function == "local_date" else "projection")
        for item in ir.set_operations:
            refs = tuple(value.ref_id for value in item.input_refs) or tuple(item.inputs)
            add(item.output_ref, spans=_provenance_spans(item), inputs=refs, role="set")
        for item in ir.aggregates:
            expression_fields, expression_refs = _expression_lineage(
                item.aggregate.input, hypothesis_aliases,
            )
            refs = list(expression_refs)
            if item.scope_ref:
                refs.append(item.scope_ref.ref_id)
            add(item.aggregate.output, fields=expression_fields,
                spans=_provenance_spans(item.aggregate), inputs=tuple(refs), role="aggregate")
        for item in ir.comparisons:
            left_fields, left_refs = _expression_lineage(item.left, hypothesis_aliases)
            right_fields, right_refs = _expression_lineage(item.right, hypothesis_aliases)
            add(item.output, fields=left_fields + right_fields,
                spans=_provenance_spans(item), inputs=left_refs + right_refs, role="comparison")
        for item in ir.arithmetic:
            fields, refs = _expression_lineage(item.expression, hypothesis_aliases)
            add(item.output, fields=fields, spans=_provenance_spans(item),
                inputs=refs, role="calculation")
        for item in ir.calculations:
            if item.output:
                fields, refs = [], []
                for expression in item.inputs:
                    found_fields, found_refs = _expression_lineage(expression, hypothesis_aliases)
                    fields.extend(found_fields)
                    refs.extend(found_refs)
                add(item.output, fields=fields, spans=_provenance_spans(item),
                    inputs=tuple(refs), role="calculation")
        return cls(entries)


@dataclass(frozen=True, slots=True)
class SemanticTargetContract:
    requirement_id: str
    family: str
    operation: str
    measure_fields: tuple[str, ...]
    scope_requirement_ids: tuple[str, ...]
    group_by_grain: str
    expected_result_type: str
    expected_output_shape: str
    grounding: RoleGroundingProof

    @property
    def ready(self) -> bool:
        return self.grounding.complete


class SemanticTargetBuilder:
    def __init__(self, grounding: RoleGroundingAnalyzer | None = None):
        self.grounding = grounding or RoleGroundingAnalyzer()

    def build(self, requirement: RequirementSpec,
              ir: UnderstandingIR) -> SemanticTargetContract:
        proof = self.grounding.prove(requirement, ir)
        measure = proof.role("measure")
        operation = next(iter(requirement.expected_attributes.get("expected_functions", [])), "")
        return SemanticTargetContract(
            requirement_id=requirement.requirement_id,
            family=requirement.requirement_type,
            operation=operation or requirement.operator_family,
            measure_fields=measure.values if measure else (),
            scope_requirement_ids=tuple(requirement.dependencies),
            group_by_grain=requirement.expected_output_grain,
            expected_result_type=str(requirement.expected_attributes.get("expected_output_type", "")),
            expected_output_shape=requirement.expected_output_shape,
            grounding=proof,
        )


def _symbol_ids(symbols, hypothesis_aliases: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    values = set()
    aliases = hypothesis_aliases or {}
    for symbol in symbols:
        values.update(value for value in (symbol.raw_name, symbol.canonical_id or "") if value)
        for candidate in symbol.candidates:
            values.add(candidate.identifier)
            values.update(aliases.get(candidate.identifier, ()))
    return tuple(sorted(values))


def _provenance_spans(node) -> list[SourceSpan]:
    return [item.span for item in getattr(node, "provenance", []) if item.span]


def _expression_lineage(expression: ExpressionNode | None,
                        hypothesis_aliases: dict[str, tuple[str, ...]] | None = None,
                        ) -> tuple[list[str], tuple[str, ...]]:
    if expression is None:
        return [], ()
    fields = list(_symbol_ids(
        [expression.symbol] if expression.symbol else [], hypothesis_aliases,
    ))
    refs = [expression.reference] if expression.kind in {"output_ref", "reference"} and expression.reference else []
    for argument in expression.arguments:
        if isinstance(argument, ExpressionNode):
            child_fields, child_refs = _expression_lineage(argument, hypothesis_aliases)
            fields.extend(child_fields)
            refs.extend(child_refs)
    return list(dict.fromkeys(fields)), tuple(dict.fromkeys(refs))
