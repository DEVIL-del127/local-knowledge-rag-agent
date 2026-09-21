"""Strict typed producer/consumer lineage for M1.5 semantic closure."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .models import OutputRef, UnderstandingIR
from .semantic_targets import OutputLineageIndex


@dataclass(frozen=True, slots=True)
class TypedLineageNode:
    ref_id: str
    operator: str
    result_type: str
    shape: str
    grain: str
    input_refs: tuple[str, ...]
    status: str
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypedLineageReport:
    nodes: tuple[TypedLineageNode, ...]
    edges: tuple[tuple[str, str], ...]
    terminal_refs: tuple[str, ...]
    ready: bool
    blockers: tuple[str, ...]


class TypedLineageCompiler:
    def compile(self, ir: UnderstandingIR) -> TypedLineageReport:
        base = OutputLineageIndex.build(ir)
        nodes: list[TypedLineageNode] = []
        edges: set[tuple[str, str]] = set()
        global_blockers = []

        def add(output: OutputRef | None, operator: str, inputs=(), blockers=()):
            if output is None:
                global_blockers.append(f"{operator}:output_missing")
                return
            blockers = list(blockers)
            for input_ref in inputs:
                if input_ref not in base.entries and not any(item.ref_id == input_ref for item in nodes):
                    blockers.append(f"input_missing:{input_ref}")
                edges.add((input_ref, output.ref_id))
            nodes.append(TypedLineageNode(
                output.ref_id, operator, output.result_type, output.shape, output.grain,
                tuple(inputs), "ready" if not blockers else "blocked", tuple(dict.fromkeys(blockers)),
            ))
            global_blockers.extend(f"{output.ref_id}:{item}" for item in blockers)

        for event in ir.events:
            output = event.output_ref
            blockers = []
            if output and output.result_type not in {"interval", "event_interval"}:
                blockers.append("event_output_must_be_interval")
            if output and output.grain != "interval":
                blockers.append("event_grain_must_be_interval")
            add(output, "ConsecutiveEvent", blockers=blockers)
        for projection in ir.derived_projections:
            blockers = []
            if projection.function == "local_date":
                if projection.input_ref.result_type not in {"interval", "event_interval"}:
                    blockers.append("date_projection_requires_interval")
                if projection.output.result_type != "date" or projection.output.grain != "date":
                    blockers.append("date_projection_output_invalid")
            add(projection.output, "ProjectLocalDate", (projection.input_ref.ref_id,), blockers)
        for operation in ir.set_operations:
            refs = operation.input_refs
            blockers = []
            if len(refs) != 2:
                blockers.append("set_requires_two_inputs")
            elif refs[0].producer_id == refs[1].producer_id:
                blockers.append("same_producer_set")
            if refs and len({(item.result_type, item.grain) for item in refs}) != 1:
                blockers.append("mixed_set_type_or_grain")
            if operation.output_ref and refs:
                expected = (refs[0].result_type, refs[0].grain)
                actual = (operation.output_ref.result_type, operation.output_ref.grain)
                if expected != actual:
                    blockers.append("set_output_type_mismatch")
            add(operation.output_ref, f"Set{operation.operation.title()}",
                tuple(item.ref_id for item in refs), blockers)
        for duration in ir.cumulative_durations:
            blockers = []
            if duration.input_ref and duration.input_ref.result_type not in {"interval", "event_interval"}:
                blockers.append("duration_accumulation_requires_interval")
            if duration.output.result_type != "duration" or duration.output.shape != "relation":
                blockers.append("duration_accumulation_output_invalid")
            add(duration.output, "DurationAccumulation",
                (duration.input_ref.ref_id,) if duration.input_ref else (), blockers)
        for aggregate in ir.aggregates:
            if aggregate.aggregate.function in {"sum", "integral", "running_sum"}:
                blockers = []
                if aggregate.aggregate.output.result_type in {"duration", "interval", "event_interval"}:
                    blockers.append("cumulative_value_output_invalid")
                inputs = (aggregate.scope_ref.ref_id,) if aggregate.scope_ref else ()
                add(aggregate.aggregate.output, "CumulativeValue", inputs, blockers)
        for arithmetic in ir.arithmetic:
            refs = tuple(_expression_refs(arithmetic.expression))
            add(arithmetic.output, "TypedArithmetic", refs)
        for calculation in ir.calculations:
            if calculation.output:
                refs = tuple(ref for expression in calculation.inputs for ref in _expression_refs(expression))
                add(calculation.output, "Calculation", refs)

        consumed = {source for source, _ in edges}
        terminals = tuple(sorted(item.ref_id for item in nodes if item.ref_id not in consumed))
        return TypedLineageReport(
            tuple(nodes), tuple(sorted(edges)), terminals,
            bool(nodes) and not global_blockers,
            tuple(sorted(set(global_blockers))),
        )


def project_local_dates(intervals, timezone: str, query_window=None):
    """Project clipped half-open intervals to unique local calendar dates."""
    zone = ZoneInfo(timezone)
    projected = set()
    for start, end in intervals:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("timezone_ambiguity")
        if query_window:
            start = max(start, query_window[0])
            end = min(end, query_window[1])
        if start >= end:
            continue
        local_start = start.astimezone(zone)
        # Half-open end: exact midnight belongs only to the preceding date.
        local_last = (end - timedelta(microseconds=1)).astimezone(zone)
        day = local_start.date()
        while day <= local_last.date():
            projected.add(day.isoformat())
            day += timedelta(days=1)
    return tuple(sorted(projected))


def _expression_refs(expression):
    refs = []
    if expression.kind in {"reference", "output_ref"} and expression.reference:
        refs.append(expression.reference)
    for argument in expression.arguments:
        if hasattr(argument, "kind"):
            refs.extend(_expression_refs(argument))
    return refs
