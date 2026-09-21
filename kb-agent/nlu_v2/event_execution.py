"""Typed temporal/value execution contracts derived from authoritative IR."""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import CatalogSnapshot
from .field_binding import FieldRoleBinder
from .merge import walk_predicates
from .models import OutputRef, UnderstandingIR


@dataclass(frozen=True, slots=True)
class ConsecutiveEventContract:
    contract_id: str
    predicate_fields: tuple[str, ...]
    duration_seconds: float | None
    source_id: str
    timestamp_ref: str
    partition_refs: tuple[str, ...]
    sampling_policy: str
    missing_data_policy: str
    boundary_policy: str
    timezone: str
    output: OutputRef
    readiness: str
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurationAccumulationContract:
    contract_id: str
    input_ref: str
    group_grain: str
    integration_method: str
    output: OutputRef
    readiness: str
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CumulativeValueContract:
    contract_id: str
    measure_field: str
    aggregation: str
    group_grain: str
    output: OutputRef
    readiness: str
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventReadinessReport:
    consecutive_events: tuple[ConsecutiveEventContract, ...]
    duration_accumulations: tuple[DurationAccumulationContract, ...]
    cumulative_values: tuple[CumulativeValueContract, ...]
    false_ready_count: int


class EventExecutionCompiler:
    def __init__(self, binder: FieldRoleBinder | None = None):
        self.binder = binder or FieldRoleBinder()

    def compile(self, ir: UnderstandingIR, catalog: CatalogSnapshot) -> EventReadinessReport:
        binding = self.binder.bind(ir, catalog)
        bound = {item.catalog_id for item in binding.proofs if item.executable}
        hypotheses = {item.cleaned_name for item in binding.proofs if item.binding_state == "hypothesis_only"}
        source_id = _source_id(ir, catalog)
        source = catalog.source(source_id) if source_id else None
        events = []
        for item in ir.events:
            fields = tuple(sorted({
                node.field.canonical_id or node.field.raw_name
                for node in walk_predicates(item.condition) if node.field
            }))
            blockers = []
            if not fields or any(field not in bound for field in fields):
                blockers.append("field_binding_pending")
            if any(field in hypotheses for field in fields):
                blockers.append("schema_hypothesis_not_executable")
            if item.threshold is None:
                blockers.append("duration_missing")
            timestamp = item.sampling.order_by.canonical_id if item.sampling.order_by else ""
            if not timestamp:
                blockers.append("timestamp_missing")
            if not source:
                blockers.append("source_missing")
            elif not {"ordered_partition", "event_segmentation"}.issubset(set(source.capabilities)):
                blockers.append("source_capability_missing")
            if item.sampling.max_gap_seconds is None:
                blockers.append("max_gap_policy_missing")
            output = item.output_ref or OutputRef(item.event_id, "intervals", "interval", shape="relation", grain="interval")
            events.append(ConsecutiveEventContract(
                item.event_id, fields, item.threshold.normalized_seconds if item.threshold else None,
                source_id, timestamp, tuple(value.canonical_id or value.raw_name for value in item.sampling.partition_by),
                "regular" if item.sampling.expected_interval_seconds else "unknown",
                item.sampling.missing_data_policy, "clip_to_window", item.window.timezone,
                output, _readiness(blockers), tuple(dict.fromkeys(blockers)),
            ))
        durations = []
        for item in ir.cumulative_durations:
            blockers = []
            if item.input_ref and item.input_ref.result_type != "interval":
                blockers.append("duration_input_must_be_interval")
            if item.output.result_type not in {"number", "duration"} or item.output.unit != "second":
                blockers.append("duration_output_type_invalid")
            contract_output = OutputRef(
                item.output.producer_id, item.output.port, "duration", unit="second",
                shape="relation", grain=item.output.grain,
                keys=list(item.output.keys) or (["local_date"] if item.output.grain == "day" else []),
                fields=list(item.output.fields),
            )
            durations.append(DurationAccumulationContract(
                item.duration_id, item.input_ref.ref_id if item.input_ref else "predicate",
                item.output.grain, item.integration_method, contract_output,
                _readiness(blockers), tuple(blockers),
            ))
        cumulative_values = []
        for item in ir.aggregates:
            if item.aggregate.function not in {"sum", "integral", "running_sum", "count"}:
                continue
            symbol = item.aggregate.input.symbol
            measure = symbol.canonical_id or symbol.raw_name if symbol else ""
            blockers = []
            if item.aggregate.function == "count" and not measure:
                measure = "*"
            if measure != "*" and (not measure or measure not in bound):
                blockers.append("value_measure_binding_pending")
            if item.aggregate.output.result_type == "duration":
                blockers.append("value_output_cannot_be_duration")
            cumulative_values.append(CumulativeValueContract(
                item.aggregate.aggregate_id, measure, item.aggregate.function,
                item.aggregate.output.grain, item.aggregate.output,
                _readiness(blockers), tuple(blockers),
            ))
        contracts = [*events, *durations, *cumulative_values]
        false_ready = sum(
            item.readiness == "logical_plan_ready" and bool(item.blockers)
            for item in contracts
        )
        return EventReadinessReport(tuple(events), tuple(durations), tuple(cumulative_values), false_ready)


def _readiness(blockers):
    return "logical_plan_ready" if not blockers else "structure_complete_binding_pending"


def _source_id(ir, catalog):
    resolved = [item.identifier for item in ir.source_candidates if item.status == "resolved" and catalog.source(item.identifier)]
    if len(set(resolved)) == 1:
        return resolved[0]
    if ir.query_schema and len(ir.query_schema.source_ids) == 1:
        return ir.query_schema.source_ids[0]
    return ""
