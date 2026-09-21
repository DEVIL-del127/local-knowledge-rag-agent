"""Evidence-gated merge of untrusted model candidates into rule IR."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from .catalog import CatalogSnapshot
from .models import (
    CalculationSpec,
    CandidateRef,
    DerivedMetricCall,
    Diagnostic,
    DurationConstraint,
    EventSpec,
    GoalSpec,
    LiteralValue,
    MetricSpec,
    PredicateNode,
    Provenance,
    SamplingPolicy,
    SetOperationSpec,
    SourceSpan,
    SymbolRef,
    UnderstandingIR,
    UnresolvedItem,
    WindowSpec,
)


_GOALS = {"retrieve", "aggregate", "compare", "compute", "present"}
_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "contains", "exists"}
_AGGREGATIONS = {"none", "sum", "avg", "min", "max", "count"}
_CALCULATIONS = {"mom", "yoy", "growth", "difference", "ratio", "volatility"}
_DURATION_SECONDS = {
    "second": 1.0, "seconds": 1.0, "秒": 1.0, "秒钟": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0, "分钟": 60.0, "分": 60.0, "min": 60.0,
    "hour": 3600.0, "hours": 3600.0, "小时": 3600.0, "时": 3600.0, "h": 3600.0,
    "day": 86400.0, "days": 86400.0, "天": 86400.0,
}


def merge_llm_candidate(base: UnderstandingIR, payload: Mapping[str, Any],
                        catalog: CatalogSnapshot) -> UnderstandingIR:
    """Merge only catalog-valid nodes whose source span exactly matches the query."""
    query = base.query.normalized
    rejected = 0
    before_count = _semantic_node_count(base)

    existing_goals = {item.goal_type for item in base.goals}
    for item in _items(payload.get("goals")):
        goal = str(item.get("type") or item.get("goal_type") or "")
        span = _validated_span(item.get("span"), query)
        if goal in _GOALS and span and goal not in existing_goals:
            base.goals.append(GoalSpec(
                goal, span, [Provenance(source="llm", span=span, rule="candidate_goal")]
            ))
            existing_goals.add(goal)
        elif item:
            rejected += 1

    existing_sources = {item.identifier for item in base.source_candidates}
    for item in _items(payload.get("sources")):
        source_id = str(item.get("id") or item.get("source_id") or "")
        span = _validated_span(item.get("span"), query)
        if (catalog.source(source_id) and span
                and catalog.prove_source_alias(source_id, span.text)
                and source_id not in existing_sources):
            base.source_candidates.append(CandidateRef(
                source_id, source="llm", status="resolved"
            ))
            existing_sources.add(source_id)
        elif item:
            rejected += 1

    existing_metrics = {
        item.field.canonical_id for item in base.metrics if item.field.canonical_id
    }
    for item in _items(payload.get("metrics")):
        field_id = str(item.get("field_id") or "")
        aggregation = str(item.get("aggregation") or "none")
        span = _validated_span(item.get("span"), query)
        if (catalog.field(field_id) and span and catalog.prove_field_alias(field_id, span.text)
                and aggregation in _AGGREGATIONS
                and field_id not in existing_metrics):
            symbol = _symbol(field_id, span, catalog)
            base.metrics.append(MetricSpec(symbol, aggregation=aggregation,
                                           unit=item.get("unit")))
            existing_metrics.add(field_id)
        elif item:
            rejected += 1

    existing_projections = {
        item.canonical_id for item in base.projections if item.canonical_id
    }
    for item in _items(payload.get("projections")):
        field_id = str(item.get("field_id") or "")
        span = _validated_span(item.get("span"), query)
        if (catalog.field(field_id) and span and catalog.prove_field_alias(field_id, span.text)
                and field_id not in existing_projections):
            base.projections.append(_symbol(field_id, span, catalog))
            existing_projections.add(field_id)
        elif item:
            rejected += 1

    event_rejected, event_aliases, accepted_event_spans = _merge_events(
        base, payload, catalog, query
    )
    rejected += event_rejected
    set_rejected, set_added = _merge_set_operations(
        base, payload, query, event_aliases
    )
    rejected += set_rejected
    if accepted_event_spans:
        base.unresolved = [
            item for item in base.unresolved
            if not (
                item.span and any(
                    item.span.start < end and item.span.end > start
                    for start, end in accepted_event_spans
                )
            )
        ]
    if set_added:
        base.unresolved = [
            item for item in base.unresolved if item.code != "set_inputs_unbound"
        ]
        set_output = base.set_operations[-1].output_name
        for reference in base.references:
            if reference.status != "resolved" and reference.raw in {"该日期", "上述结果", "这些结果"}:
                reference.status = "resolved"
                reference.target_task_id = set_output
        for calculation in base.calculations:
            if calculation.calculation_type == "volatility" and not calculation.parameters.get(
                "filter_reference"
            ):
                calculation.parameters["filter_reference"] = set_output
        if set_output not in base.output.fields:
            base.output.fields.append(set_output)

    existing_filters = {
        _predicate_identity(node) for node in walk_predicates(base.filters)
        if node.field and node.field.canonical_id
    }
    new_filters = []
    for item in _items(payload.get("filters")):
        field_id = str(item.get("field_id") or "")
        operator = str(item.get("operator") or "")
        span = _validated_span(item.get("span"), query)
        field_spec = catalog.field(field_id)
        if field_spec and field_spec.data_type == "date" and base.temporal:
            continue
        identity = _candidate_predicate_identity(field_id, operator, item.get("value"))
        if (catalog.field(field_id) and span and catalog.prove_field_alias(field_id, span.text)
                and operator in _OPERATORS
                and identity not in existing_filters and "value" in item):
            value = item.get("value")
            value_type = "range" if operator == "between" else _value_type(value)
            new_filters.append(PredicateNode(
                operator=operator,
                field=_symbol(field_id, span, catalog),
                value=LiteralValue(value, value_type, unit=item.get("unit")),
                provenance=[Provenance(source="llm", rule="candidate_filter", span=span)],
            ))
            existing_filters.add(identity)
        elif item:
            rejected += 1
    base.filters = _combine(base.filters, new_filters)

    existing_calculations = {
        (item.calculation_type, item.expression,
         jsonable(item.parameters)) for item in base.calculations
    }
    for item in _items(payload.get("calculations")):
        span = _validated_span(item.get("span"), query)
        calc_type = str(item.get("type") or item.get("calculation_type") or "")
        expression = str(item.get("expression") or "")
        parameters = dict(item.get("parameters") or {})
        identity = (calc_type, expression, jsonable(parameters))
        if (span and calc_type in _CALCULATIONS and _safe_expression(expression)
                and identity not in existing_calculations):
            base.calculations.append(CalculationSpec(
                calculation_type=calc_type,
                expression=expression,
                parameters=parameters,
                provenance=[Provenance(source="llm", rule="candidate_calculation", span=span)],
            ))
            existing_calculations.add(identity)
        elif identity not in existing_calculations:
            rejected += 1

    for item in _items(payload.get("unresolved")):
        raw = str(item.get("name") or item.get("raw") or "")
        span = _validated_span(item.get("span"), query)
        if raw and span:
            base.unresolved.append(UnresolvedItem(
                code=str(item.get("code") or "llm_unresolved"),
                message=str(item.get("message") or f"无法绑定：{raw}"),
                span=span,
                candidates=[str(value) for value in item.get("candidates", [])],
            ))
        elif item:
            rejected += 1

    _infer_sources(base, catalog)
    if rejected:
        base.diagnostics.append(Diagnostic(
            severity="warning", code="llm_nodes_rejected",
            message=f"忽略 {rejected} 个缺少有效 Catalog ID 或原文证据的 LLM 候选节点",
        ))
    if _semantic_node_count(base) == before_count:
        base.diagnostics.append(Diagnostic(
            "info", "llm_no_gain", "结构化模型候选未增加可验证语义",
        ))
    return base


def walk_predicates(root: PredicateNode | None) -> Iterable[PredicateNode]:
    if root is None:
        return
    if root.kind == "boolean":
        for child in root.children:
            yield from walk_predicates(child)
        return
    yield root


def _validated_span(value: Any, query: str) -> SourceSpan | None:
    if not isinstance(value, Mapping):
        return None
    text = str(value.get("text", ""))
    if "start" in value and "end" in value:
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError):
            start, end = -1, -1
        if start >= 0 and end > start and end <= len(query) and query[start:end] == text:
            return SourceSpan(start, end, text)
    # Model-provided offsets are advisory. Exact unique text is the evidence;
    # canonical offsets are always derived locally when the offsets are stale.
    if text and text != "SPAN" and query.count(text) == 1:
        start = query.find(text)
        return SourceSpan(start, start + len(text), text)
    return None


def _recover_duration_span(event_span: SourceSpan, query: str,
                           expected_seconds: float) -> SourceSpan | None:
    """Recover omitted model evidence only from one exact normalized duration."""
    pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>秒钟?|分钟|分|小时|时|天|日|seconds?|minutes?|hours?|days?|s|min|h)"
        r"(?:\s*(?:以上|至少|不少于))?",
        re.I,
    )
    matches = []
    for match in pattern.finditer(event_span.text):
        unit = match.group("unit").lower()
        factor = _DURATION_SECONDS.get(unit)
        if factor is None:
            continue
        seconds = float(match.group("value")) * factor
        if abs(seconds - expected_seconds) > 1e-6:
            continue
        start = event_span.start + match.start()
        end = event_span.start + match.end()
        matches.append(SourceSpan(start, end, query[start:end]))
    return matches[0] if len(matches) == 1 else None


def _symbol(field_id: str, span: SourceSpan, catalog: CatalogSnapshot) -> SymbolRef:
    field = catalog.field(field_id)
    return SymbolRef(
        raw_name=span.text,
        canonical_id=field_id,
        candidates=[CandidateRef(field_id, source="llm")],
        status="resolved" if field else "unresolved",
        span=span,
    )


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _combine(existing: PredicateNode | None,
             additions: list[PredicateNode]) -> PredicateNode | None:
    if not additions:
        return existing
    nodes = list(walk_predicates(existing)) if existing else []
    nodes.extend(additions)
    if len(nodes) == 1:
        return nodes[0]
    return PredicateNode(kind="boolean", operator="and", children=nodes)


def _infer_sources(ir: UnderstandingIR, catalog: CatalogSnapshot) -> None:
    existing = {item.identifier for item in ir.source_candidates}
    symbols = list(ir.projections)
    symbols.extend(item.field for item in ir.metrics)
    symbols.extend(
        node.field for node in walk_predicates(ir.filters) if node.field is not None
    )
    symbols.extend(ir.grouping)
    for event in ir.events:
        symbols.extend(
            node.field for node in walk_predicates(event.condition) if node.field is not None
        )
    for symbol in symbols:
        source_id = catalog.source_for_field(symbol.canonical_id or "")
        if source_id and source_id not in existing:
            ir.source_candidates.append(CandidateRef(source_id, source="inferred"))
            existing.add(source_id)


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    return "string"


def _safe_expression(expression: str) -> bool:
    if not expression or "__" in expression:
        return False
    return all(character.isalnum() or character in "._+-*/()% \t" for character in expression)


def jsonable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _predicate_identity(node: PredicateNode) -> tuple[str, str, str]:
    field_id = node.field.canonical_id if node.field else ""
    value = node.value.value if node.value else None
    return _candidate_predicate_identity(field_id or "", node.operator, value)


def _candidate_predicate_identity(field_id: str, operator: str,
                                  value: Any) -> tuple[str, str, str]:
    return field_id, operator, jsonable(value)


def _merge_events(base: UnderstandingIR, payload: Mapping[str, Any],
                  catalog: CatalogSnapshot, query: str
                  ) -> tuple[int, dict[str, str], list[tuple[int, int]]]:
    aliases = {
        item.event_id: item.output_name for item in base.events
    }
    aliases.update({item.output_name: item.output_name for item in base.events})
    existing_spans = [
        (item.provenance[0].span.start, item.provenance[0].span.end)
        for item in base.events if item.provenance and item.provenance[0].span
    ]
    accepted_spans: list[tuple[int, int]] = []
    rejected = 0
    for index, item in enumerate(_items(payload.get("events")), start=1):
        event_span = _validated_span(item.get("span"), query)
        metric_id = str(item.get("metric_id") or "")
        metric = catalog.derived_metric(metric_id)
        if (not event_span or not metric or not metric_id.endswith(
            (".consecutive_duration", ".cumulative_duration")
        )):
            rejected += 1
            continue
        continuity_signal = any(
            word in event_span.text for word in ("连续", "一直", "持续", "保持")
        )
        cumulative_signal = any(
            word in event_span.text for word in ("累计", "总时长", "合计时长")
        )
        metric_source = catalog.source_for_derived_metric(metric_id)
        if continuity_signal and not cumulative_signal:
            expected_id = f"{metric_source}.consecutive_duration"
            expected_metric = catalog.derived_metric(expected_id)
            if expected_metric:
                metric_id, metric = expected_id, expected_metric
        elif cumulative_signal:
            expected_id = f"{metric_source}.cumulative_duration"
            expected_metric = catalog.derived_metric(expected_id)
            if expected_metric:
                metric_id, metric = expected_id, expected_metric
        if any(event_span.start < end and event_span.end > start for start, end in existing_spans):
            rejected += 1
            continue
        source_id = catalog.source_for_derived_metric(metric_id)
        source = catalog.source(source_id or "")
        conditions = []
        valid = True
        invalid_condition_count = 0
        for condition_item in _items(item.get("conditions")):
            span = _validated_span(condition_item.get("span"), query)
            field_id = str(condition_item.get("field_id") or "")
            operator = str(condition_item.get("operator") or "")
            if (source and field_id == source.metadata.get("timestamp_field")
                    and base.temporal):
                continue
            if (not span or catalog.field(field_id) is None
                    or not catalog.prove_field_alias(field_id, span.text)
                    or operator not in _OPERATORS
                    or "value" not in condition_item
                    or catalog.source_for_field(field_id) != source_id):
                invalid_condition_count += 1
                continue
            value = condition_item.get("value")
            conditions.append(PredicateNode(
                operator=operator,
                field=_symbol(field_id, span, catalog),
                value=LiteralValue(
                    value, "range" if operator == "between" else _value_type(value),
                    unit=condition_item.get("unit"),
                ),
                provenance=[Provenance(
                    source="llm", rule="candidate_event_condition", span=span,
                )],
            ))
        if not conditions:
            conditions = [
                node for node in walk_predicates(base.filters)
                if node.field and catalog.source_for_field(node.field.canonical_id or "") == source_id
                and node.provenance and node.provenance[0].span
                and node.provenance[0].span.start < event_span.end
                and node.provenance[0].span.end > event_span.start
            ]
        duration_item = item.get("duration")
        duration_span = _validated_span(
            duration_item.get("span") if isinstance(duration_item, Mapping) else None, query
        )
        try:
            duration_value = float(duration_item.get("value"))
            duration_unit = str(duration_item.get("unit") or "").lower()
            duration_operator = str(duration_item.get("operator") or "")
        except (AttributeError, TypeError, ValueError):
            valid = False
            duration_value, duration_unit, duration_operator = 0.0, "", ""
        if (not duration_span and valid and duration_value > 0
                and duration_unit in _DURATION_SECONDS):
            duration_span = _recover_duration_span(
                event_span, query,
                duration_value * _DURATION_SECONDS[duration_unit],
            )
        if (not conditions or not duration_span or duration_value <= 0
                or duration_unit not in _DURATION_SECONDS
                or duration_operator not in {"gt", "gte"}):
            valid = False
        if duration_span and any(word in duration_span.text for word in ("以上", "至少", "不少于")):
            duration_operator = "gte"
        if not valid or source is None:
            rejected += 1
            continue
        logic = str(item.get("logic") or "and")
        if logic not in {"and", "or"}:
            rejected += 1
            continue
        condition = conditions[0] if len(conditions) == 1 else PredicateNode(
            kind="boolean", operator=logic, children=conditions
        )
        temporal = base.temporal[0] if base.temporal else None
        metadata = source.metadata
        timestamp_id = str(metadata.get("timestamp_field", ""))
        partition_ids = [str(value) for value in metadata.get("partition_fields", [])]
        expected = metadata.get("expected_sampling_interval_seconds")
        expected_seconds = int(expected) if expected is not None else None
        sampling = SamplingPolicy(
            order_by=_catalog_symbol(timestamp_id, catalog) if timestamp_id else None,
            partition_by=[_catalog_symbol(value, catalog) for value in partition_ids],
            expected_interval_seconds=expected_seconds,
            max_gap_seconds=expected_seconds * int(metric.defaults.get("max_gap_multiplier", 2))
            if expected_seconds else None,
            missing_data_policy=metric.missing_data_policy or "break_segment",
            duplicate_policy="keep_last",
        )
        raw_id = str(item.get("event_id") or f"event_{index}")
        safe_id = "".join(character for character in raw_id if character.isalnum() or character == "_")[:40]
        event_id = f"llm_{safe_id or f'event_{index}'}"
        output_name = f"{event_id}_local_dates"
        event = EventSpec(
            event_id=event_id,
            condition=condition,
            derived_metric=DerivedMetricCall(
                metric_id=metric_id,
                arguments={"condition": "event_condition", **metric.defaults},
                result_type=metric.result_type,
                unit=metric.unit,
                provenance=[Provenance(
                    source="llm", rule="candidate_derived_metric", span=event_span,
                )],
            ),
            threshold=DurationConstraint(
                operator=duration_operator, value=duration_value, unit=duration_unit,
                normalized_seconds=duration_value * _DURATION_SECONDS[duration_unit],
                span=duration_span,
            ),
            window=WindowSpec(
                window_type="absolute",
                from_value=(temporal.from_value or temporal.exact) if temporal else None,
                to_value=(temporal.to_value or temporal.exact) if temporal else None,
                timezone=(temporal.timezone if temporal else "")
                or str(metadata.get("default_timezone", "Asia/Shanghai")),
                granularity=temporal.granularity if temporal else "minute",
            ),
            sampling=sampling,
            group_by=["local_date"],
            output_name=output_name,
            provenance=[Provenance(
                source="llm", rule="candidate_event", span=event_span,
            )],
        )
        base.events.append(event)
        rejected += invalid_condition_count
        aliases[raw_id] = output_name
        aliases[event_id] = output_name
        aliases[output_name] = output_name
        accepted_spans.append((event_span.start, event_span.end))
        existing_spans.append((event_span.start, event_span.end))
    return rejected, aliases, accepted_spans


def _merge_set_operations(base: UnderstandingIR, payload: Mapping[str, Any], query: str,
                          aliases: Mapping[str, str]) -> tuple[int, bool]:
    if base.set_operations:
        return len(_items(payload.get("set_operations"))), False
    rejected = 0
    for index, item in enumerate(_items(payload.get("set_operations")), start=1):
        span = _validated_span(item.get("span"), query)
        operation = str(item.get("operation") or "")
        raw_inputs = item.get("inputs") if isinstance(item.get("inputs"), list) else []
        inputs = [aliases.get(str(value), "") for value in raw_inputs]
        if (not span or operation not in {"intersection", "union", "difference"}
                or len(inputs) < 2 or any(not value for value in inputs)):
            rejected += 1
            continue
        base.set_operations.append(SetOperationSpec(
            operation=operation,
            inputs=inputs,
            output_name=f"llm_set_{index}_dates",
            granularity="date",
            provenance=[Provenance(
                source="llm", rule="candidate_set_operation", span=span,
            )],
        ))
        return rejected, True
    return rejected, False


def _catalog_symbol(field_id: str, catalog: CatalogSnapshot) -> SymbolRef:
    field = catalog.field(field_id)
    return SymbolRef(
        raw_name=field_id.rsplit(".", 1)[-1],
        canonical_id=field_id if field else None,
        candidates=[CandidateRef(field_id, source="catalog")] if field else [],
        status="resolved" if field else "unresolved",
    )


def _semantic_node_count(ir: UnderstandingIR) -> int:
    return sum((
        len(ir.goals), len(ir.source_candidates), len(ir.projections), len(ir.metrics),
        len(list(walk_predicates(ir.filters))), len(ir.events), len(ir.set_operations),
        len(ir.calculations), len(getattr(ir, "aggregates", [])),
        len(getattr(ir, "comparisons", [])),
    ))
