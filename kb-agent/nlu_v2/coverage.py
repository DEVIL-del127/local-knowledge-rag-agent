"""Strict matching between independent requirements and produced SemanticIR."""
from __future__ import annotations

from .field_roles import hypothesis_matches_field
from .merge import walk_predicates
from .models import (
    AmbiguitySpec, CoverageSpec, EventSpec, ExpressionNode, MetricSpec, OutputRef, PredicateNode,
    RequirementSpec, SourceSpan, UnderstandingIR,
)
from .semantic_targets import OutputLineageIndex, SemanticTargetBuilder


class CoverageMatcher:
    """Prove requirements through exact semantic attributes, never node type alone."""

    def __init__(self, *, enforce_semantic_target_gate: bool = True):
        self.enforce_semantic_target_gate = enforce_semantic_target_gate

    def apply(self, ir: UnderstandingIR) -> list[CoverageSpec]:
        self._active_ir = ir
        coverage = [self._match(item, ir) for item in ir.requirements]
        self._enforce_requirement_dependencies(ir.requirements, coverage, ir)
        ir.coverage = coverage
        self._link_operator_requirements(ir, coverage)
        self.refresh_ambiguities(ir)
        return coverage

    @staticmethod
    def _enforce_requirement_dependencies(requirements: list[RequirementSpec],
                                          coverage: list[CoverageSpec], ir: UnderstandingIR) -> None:
        """Require exact upstream OutputRef lineage, not a matching count."""
        by_requirement = {item.requirement_id: item for item in coverage}
        lineage = OutputLineageIndex.build(ir)
        changed = True
        while changed:
            changed = False
            for requirement in requirements:
                item = by_requirement.get(requirement.requirement_id)
                if item is None or item.status != "satisfied":
                    continue
                missing = [dependency for dependency in requirement.dependencies
                           if by_requirement.get(dependency, CoverageSpec(dependency)).status != "satisfied"]
                if missing:
                    item.status = "uncovered"
                    item.missing_attributes = [f"dependency:{value}" for value in missing]
                    item.reason = "required upstream semantics are uncovered"
                    changed = True
                    continue
                # The output contract itself is a presentation declaration,
                # not a DAG producer.  All executable downstream operators
                # must consume the exact OutputRef emitted by every direct
                # Requirement dependency.
                if requirement.requirement_type == "output":
                    continue
                if requirement.requirement_type in {
                    "field_reference", "predicate", "boolean_logic", "temporal",
                }:
                    continue
                required_refs = {
                    output.ref_id
                    for dependency in requirement.dependencies
                    for output in _outputs_for_coverage(by_requirement.get(dependency), ir)
                }
                if not required_refs:
                    continue
                candidates = _nodes_for_coverage(item, ir)
                if not any(required_refs <= _node_lineage_inputs(node, lineage)
                           for node in candidates):
                    item.status = "uncovered"
                    item.missing_attributes = [
                        "input_output_ref:" + value for value in sorted(required_refs)
                    ]
                    item.reason = "required upstream OutputRef lineage is absent"
                    changed = True

    def _match(self, requirement: RequirementSpec, ir: UnderstandingIR) -> CoverageSpec:
        kind = requirement.requirement_type
        matches: list[tuple[str, object]] = []
        if kind == "predicate":
            matches = [(node_id, node) for node_id, node in self._predicate_matches(requirement, ir)]
        elif kind == "boolean_logic":
            relation = requirement.expected_attributes.get("relation")
            roots = [ir.filters, *(item.condition for item in ir.events)]
            matches = [
                (f"boolean:{index}", node) for index, root in enumerate(roots)
                for node in _boolean_nodes(root)
                if (not relation or node.operator == relation)
                and (
                    self._node_has_evidence(node, requirement.span)
                    or _connector_between_children(node, requirement.span)
                )
            ]
            if relation == "and":
                matches.extend(
                    (f"set_boolean:{index}", item)
                    for index, item in enumerate(ir.set_operations)
                    if item.operation == "intersection"
                    and _connector_between_nodes(ir.events, requirement.span)
                )
                matches.extend(
                    (f"temporal_boolean:{index}", item)
                    for index, item in enumerate(ir.temporal)
                    if _connector_between_nodes(ir.temporal, requirement.span)
                )
                matches.extend(
                    (f"calculation_boolean:{index}", item)
                    for index, item in enumerate(ir.calculations)
                    if _connector_between_nodes(ir.calculations, requirement.span)
                )
                if any(" AND " in criterion for criterion in ir.output.criteria):
                    matches.append(("output_boolean", ir.output))
        elif kind == "field_reference":
            matches = [
                (f"field:{index}", symbol) for index, symbol in enumerate(_all_symbols(ir))
                if self._field_matches(requirement.expected_attributes.get("field_text", ""), symbol)
                and self._symbol_has_evidence(symbol, requirement.span)
            ]
        elif kind == "temporal":
            matches = [
                (f"temporal:{index}", item) for index, item in enumerate(ir.temporal)
                if self._node_has_evidence(item, requirement.span)
                and (not requirement.text or item.raw == requirement.text or _spans_overlap(_spans_for(item), requirement.span))
            ]
            matches.extend(
                (item.duration_id, item) for item in ir.cumulative_durations
                if requirement.text and requirement.text in str(item.scope.reference)
                and self._node_has_evidence(item, requirement.span)
            )
            matches.extend(
                (item.count_id, item) for item in ir.cumulative_counts
                if requirement.text and requirement.text in str(item.scope.reference)
                and self._node_has_evidence(item, requirement.span)
            )
        elif kind == "set_operation":
            wanted = self._set_operation(requirement.text)
            compatible = [
                item for item in ir.set_operations
                if not wanted or item.operation == wanted
            ]
            matches = [
                (item.output_name, item) for item in ir.set_operations
                if (not wanted or item.operation == wanted)
                and self._node_has_evidence(item, requirement.span)
            ]
            # Output clauses often restate an earlier Set operation (for
            # example, "输出重合时间段").  A single type-compatible producer
            # is deterministic; multiple producers remain uncovered.
            if not matches and len(compatible) == 1:
                matches = [(compatible[0].output_name, compatible[0])]
        elif kind in {"sequence_event", "event"}:
            matches = self._event_matches(requirement, ir)
        elif kind == "cumulative_count":
            matches = self._cumulative_count_matches(requirement, ir)
        elif kind == "cumulative_duration":
            matches = self._cumulative_duration_matches(requirement, ir)
        elif kind in {"aggregate", "scoped_aggregate", "cumulative_aggregate"}:
            matches = self._aggregate_matches(requirement, ir)
        elif kind == "derived_projection":
            matches = [
                (item.projection_id, item) for item in ir.derived_projections
                if requirement.operator_family == "period_duration"
                and item.function == "local_calendar_day_duration_seconds"
                and self._node_has_evidence(item, requirement.span)
            ]
        elif kind == "group_by":
            matches = self._group_matches(requirement, ir)
        elif kind == "top_k":
            matches = self._top_k_matches(requirement, ir)
        elif kind == "calculation":
            matches = self._calculation_matches(requirement, ir)
        elif kind == "conversion":
            matches = [
                (item.conversion_id, item) for item in ir.conversions
                if self._node_has_evidence(item, requirement.span)
            ]
        elif kind == "reference":
            matches = [(item.reference_id, item) for item in ir.references
                       if item.status == "resolved" and item.target_task_id]
        elif kind == "turn_directive":
            matches = [(item.directive_id, item) for item in ir.turn_directives
                       if self._node_has_evidence(item, requirement.span)]
        elif kind == "temporal_ambiguity":
            matches = [(item.ambiguity_id, item) for item in ir.ambiguities
                       if item.kind in {"temporal_century", "timezone_fold"}
                       and self._node_has_evidence(item, requirement.span)]
        elif kind == "satisfiability_check":
            matches = [(item.check_id, item) for item in ir.unsatisfiable
                       if self._node_has_evidence(item, requirement.span)]
        elif kind == "output":
            matches = self._output_matches(requirement, ir)

        strict_matches = [(node_id, node) for node_id, node in matches
                          if self._contract_matches(requirement, node, ir)]
        minimum = max(1, requirement.cardinality)
        status = "satisfied" if len(strict_matches) >= minimum else "uncovered"
        missing = [] if status == "satisfied" else self._missing_contract(requirement, matches, ir)
        return CoverageSpec(
            requirement_id=requirement.requirement_id,
            status=status,
            covered_by=[node_id for node_id, _ in strict_matches],
            missing_attributes=missing,
            reason="" if status == "satisfied" else "strict semantic contract is not fully matched",
            matched_attributes=self._matched_attributes(requirement, strict_matches),
        )

    def _predicate_matches(self, requirement: RequirementSpec,
                           ir: UnderstandingIR) -> list[tuple[str, PredicateNode]]:
        expected = requirement.expected_attributes
        expected_operator = expected.get("operator")
        expected_value = expected.get("value")
        expected_field = str(expected.get("field_text", ""))
        result = []
        roots = [
            ir.filters,
            *(item.condition for item in ir.events),
            *(item.predicate for item in ir.relation_filters),
        ]
        for index, predicate in enumerate(
            node for root in roots for node in walk_predicates(root) if node.kind != "boolean"
        ):
            if expected_operator and predicate.operator != expected_operator:
                continue
            if expected_value is not None:
                actual = predicate.value.value if predicate.value else None
                if isinstance(actual, list) or actual is None or not _values_equal(actual, expected_value):
                    continue
            if expected.get("unit"):
                actual_unit = _canonical_unit(predicate.value.unit if predicate.value else None)
                if actual_unit != _canonical_unit(expected.get("unit")):
                    continue
            if expected_field and not self._field_matches_in_ir(expected_field, predicate.field, ir):
                continue
            if not self._node_has_evidence(predicate, requirement.span):
                continue
            result.append((f"predicate:{index}", predicate))
        return result

    def _event_matches(self, requirement: RequirementSpec, ir: UnderstandingIR) -> list[tuple[str, object]]:
        family = requirement.operator_family or requirement.expected_attributes.get("operator_family", "")
        action = requirement.expected_attributes.get("expected_action")
        result = []
        for event in ir.events:
            metric = event.derived_metric.metric_id.lower()
            if family == "cumulative_duration" and not metric.endswith("cumulative_duration"):
                continue
            if family == "cumulative_count" and "cumulative_count" not in metric:
                continue
            if family == "event" and action:
                predicate_values = [str(node.value.value).lower() for node in walk_predicates(event.condition)
                                    if node.value is not None]
                if str(action).lower() not in predicate_values:
                    continue
            if self._node_has_evidence(event, requirement.span):
                result.append((event.event_id, event))
        for duration in ir.cumulative_durations:
            if family != "event" or not action:
                continue
            value = duration.condition.value.value if duration.condition.value is not None else None
            if str(value).lower() == str(action).lower() and self._node_has_evidence(duration, requirement.span):
                result.append((duration.duration_id, duration))
        return result

    def _aggregate_matches(self, requirement: RequirementSpec, ir: UnderstandingIR) -> list[tuple[str, object]]:
        target = SemanticTargetBuilder().build(requirement, ir)
        if self.enforce_semantic_target_gate and target.grounding.enforceable:
            measure = target.grounding.role("measure")
            scope = target.grounding.role("scope")
            grain = target.grounding.role("grain")
            if (any(item.startswith("field:") for item in requirement.expected_inputs)
                    and (measure is None or measure.status != "proven")):
                return []
            if requirement.dependencies and scope is not None and scope.status == "unproven":
                return []
            if (requirement.expected_output_grain not in {"", "scalar"}
                    and grain is not None and grain.status == "unproven"):
                return []
        expected = set(requirement.expected_attributes.get("expected_functions", []))
        result = []
        for aggregate in ir.aggregates:
            if expected and aggregate.aggregate.function not in expected:
                continue
            if requirement.operator_family == "quantile":
                expected_quantile = requirement.expected_attributes.get("quantile")
                actual_quantile = aggregate.aggregate.parameters.get("quantile")
                if expected_quantile is None or actual_quantile is None:
                    continue
                if abs(float(actual_quantile) - float(expected_quantile) / 100.0) > 1e-9:
                    continue
            if not self._node_has_evidence(aggregate.aggregate, requirement.span):
                continue
            result.append((aggregate.aggregate.aggregate_id, aggregate))
        # A flat projection metric has no relation/global/filtered scope and
        # therefore cannot prove a scoped aggregate obligation.
        if requirement.requirement_type not in {"scoped_aggregate", "cumulative_aggregate"}:
            for index, metric in enumerate(ir.metrics, start=1):
                if expected and metric.aggregation not in expected:
                    continue
                if not self._symbol_has_evidence(metric.field, requirement.span):
                    continue
                result.append((f"metric:{index}", metric))
        return result

    def _group_matches(self, requirement: RequirementSpec, ir: UnderstandingIR) -> list[tuple[str, object]]:
        expected_text = _group_text(requirement.text)
        expected_fields = [item[6:] for item in requirement.expected_inputs
                           if item.startswith("field:")]
        return [
            (item.group_id, item) for item in ir.group_by_operations
            if any(
                (expected_fields and any(
                    self._field_matches_in_ir(field, key.symbol, ir)
                    for field in expected_fields
                ))
                or (not expected_fields and (
                    not expected_text or self._field_matches(expected_text, key.symbol)
                ))
                for key in item.keys if key.symbol
            )
            and self._node_has_evidence(item, requirement.span)
        ]

    def _top_k_matches(self, requirement: RequirementSpec, ir: UnderstandingIR) -> list[tuple[str, object]]:
        expected = requirement.expected_attributes.get("expected_limit")
        return [
            (item.top_k_id, item) for item in ir.top_k_operations
            if expected is not None and item.limit == expected
            and self._node_has_evidence(item, requirement.span)
        ]

    def _cumulative_count_matches(self, requirement: RequirementSpec,
                                  ir: UnderstandingIR) -> list[tuple[str, object]]:
        expected = str(requirement.expected_attributes.get("expected_action", "")).lower()
        return [
            (item.count_id, item) for item in ir.cumulative_counts
            if item.action.value is not None
            and (not expected or str(item.action.value.value).lower() == expected)
            and self._node_has_evidence(item, requirement.span)
        ]

    def _cumulative_duration_matches(self, requirement: RequirementSpec,
                                     ir: UnderstandingIR) -> list[tuple[str, object]]:
        expected = str(requirement.expected_attributes.get("expected_action", "")).lower()
        return [
            (item.duration_id, item) for item in ir.cumulative_durations
            if (item.condition.value is not None or item.input_ref is not None)
            and (not expected or (
                item.condition.value is not None
                and str(item.condition.value.value).lower() == expected
            ))
            and self._node_has_evidence(item, requirement.span)
        ]

    def _calculation_matches(self, requirement: RequirementSpec,
                             ir: UnderstandingIR) -> list[tuple[str, object]]:
        family = requirement.operator_family or requirement.expected_attributes.get("operator_family", "")
        if family in {"", "calculation"}:
            # A bare "计算" cue does not identify an input/output formula and
            # therefore cannot prove that a calculation requirement is complete.
            return []
        aliases = {"difference": {"difference"}, "ratio": {"ratio"},
                   "correlation": {"correlation", "pearson"}}
        expected = aliases.get(family, {family})
        result = []
        for index, calculation in enumerate(ir.calculations, start=1):
            if calculation.calculation_type in expected and self._node_has_evidence(calculation, requirement.span):
                result.append((f"calculation:{index}", calculation))
        for arithmetic in ir.arithmetic:
            arithmetic_family = {
                "div": "ratio", "sub": "difference", "mul": "product",
            }.get(arithmetic.expression.operator, arithmetic.expression.operator)
            if arithmetic_family in expected and self._node_has_evidence(arithmetic, requirement.span):
                result.append((arithmetic.arithmetic_id, arithmetic))
        return result

    @staticmethod
    def _output_matches(requirement: RequirementSpec, ir: UnderstandingIR) -> list[tuple[str, object]]:
        return [("output_contract", ir.output)] if ir.output.format and (ir.output.fields or ir.output.limit) else []

    def _contract_matches(self, requirement: RequirementSpec, node: object, ir: UnderstandingIR) -> bool:
        if not self._source_anchor_match(requirement, node, ir):
            if requirement.requirement_type != "set_operation":
                return False
            wanted = self._set_operation(requirement.text)
            compatible = [item for item in ir.set_operations
                          if not wanted or item.operation == wanted]
            if len(compatible) != 1 or compatible[0] is not node:
                return False
        output = _node_output(node)
        if requirement.expected_output_shape and (output is None or output.shape != requirement.expected_output_shape):
            return False
        expected_type = str(requirement.expected_attributes.get("expected_output_type", ""))
        if expected_type and (output is None or output.result_type != expected_type):
            return False
        if requirement.expected_output_fields:
            if output is None or not self._output_fields_match(requirement, node, output, ir):
                return False
        if requirement.expected_output_grain and requirement.expected_output_grain != "scalar":
            if output is None or not self._grain_matches(
                    requirement.expected_output_grain, node, output):
                return False
        if requirement.expected_scope and _node_scope(node) != requirement.expected_scope:
            return False
        field_inputs = [item[6:] for item in requirement.expected_inputs if item.startswith("field:")]
        if field_inputs:
            symbols = _node_symbols(node)
            demands = {item.demand_id: item for item in ir.source_demands}
            input_demands = [
                demands[item] for item in requirement.expected_attributes.get(
                    "input_demand_ids", []
                ) if item in demands
            ]
            for field in field_inputs:
                if any(self._field_matches_in_ir(field, symbol, ir) for symbol in symbols):
                    continue
                contextual = False
                for demand in input_demands:
                    window = ir.query.normalized[
                        max(0, demand.span.start - 48):demand.span.end
                    ]
                    normalized_window = "".join(window.replace("`", "").split()).lower()
                    if any("".join((symbol.raw_name or "").replace("`", "").split()).lower()
                           in normalized_window for symbol in symbols if symbol.raw_name):
                        contextual = True
                        break
                if not contextual:
                    return False
        return True

    @staticmethod
    def _grain_matches(expected: str, node: object, output: OutputRef) -> bool:
        if output.grain == expected:
            return True
        if expected == "day" and output.grain == "group":
            if "local_date" in output.keys:
                return True
            return any(
                expression.result_type == "date"
                for expression in getattr(node, "group_by", [])
            )
        return False

    def _output_fields_match(self, requirement: RequirementSpec, node: object,
                             output: OutputRef, ir: UnderstandingIR) -> bool:
        expected = set(requirement.expected_output_fields)
        actual = set(output.fields)
        if expected <= actual:
            return True
        field_inputs = [item[6:] for item in requirement.expected_inputs if item.startswith("field:")]
        symbols = _node_symbols(node)
        for value in expected:
            if ":" not in value:
                if value not in actual and not any(
                    item.startswith(value + ":") for item in actual
                ):
                    return False
                continue
            function, field = value.split(":", 1)
            candidates = [item.split(":", 1)[1] for item in actual
                          if item.startswith(function + ":")]
            if not candidates:
                return False
            if field in field_inputs and symbols and any(
                self._field_matches_in_ir(field, symbol, ir) for symbol in symbols
            ):
                continue
            if field not in candidates:
                return False
        return True

    def _missing_contract(self, requirement: RequirementSpec,
                          type_matches: list[tuple[str, object]], ir: UnderstandingIR) -> list[str]:
        if not type_matches:
            family = requirement.operator_family or requirement.requirement_type
            if family == "calculation":
                return ["formula_input_output_contract"]
            return [family]
        missing = []
        if any(item.startswith("requirement:") for item in requirement.expected_inputs):
            missing.append("input_output_refs")
        if any(item.startswith("field:") for item in requirement.expected_inputs):
            missing.append("input_fields")
        if requirement.expected_output_shape:
            missing.append("output_shape")
        if requirement.expected_attributes.get("expected_output_type"):
            missing.append("output_type")
        if requirement.expected_output_fields:
            missing.append("output_fields")
        if requirement.expected_output_grain and requirement.expected_output_grain != "scalar":
            missing.append("output_grain")
        if requirement.expected_scope:
            missing.append("scope")
        return missing or ["source_anchor_contract"]

    @staticmethod
    def _matched_attributes(requirement: RequirementSpec,
                            matches: list[tuple[str, object]]) -> dict[str, object]:
        if not matches:
            return {}
        attrs = dict(requirement.expected_attributes)
        attrs["matched_nodes"] = [node_id for node_id, _ in matches]
        return attrs

    def _source_anchor_match(self, requirement: RequirementSpec, node: object,
                             ir: UnderstandingIR) -> bool:
        identifiers = requirement.expected_attributes.get("source_demand_ids", [])
        if not identifiers:
            return True
        demands = {item.demand_id: item for item in ir.source_demands}
        spans = _spans_for(node)
        for identifier in identifiers:
            demand = demands.get(identifier)
            if demand is None:
                return False
            if (demand.demand_type == "boolean" and requirement.span is not None
                    and not _spans_overlap([demand.span], requirement.span)):
                # Clause-level connectors may depend on several operators.
                # They are not evidence slots of an event that ends before
                # the connector; the downstream Boolean/Set requirement owns
                # those anchors independently.
                continue
            if demand.demand_type == "boolean" and requirement.expected_attributes.get("boolean_relations"):
                if (isinstance(node, EventSpec) and node.threshold is not None
                        and self._node_has_evidence(node, demand.span)):
                    # This connector is the explicit condition-to-duration
                    # conjunction inside one consecutive event.
                    continue
                if not self._boolean_demand_match(demand, ir):
                    return False
                continue
            if node is ir.output and demand.demand_type == "output":
                continue
            if demand.demand_type == "field":
                field_text = str(demand.attributes.get("field_text", demand.text))
                symbols = _node_symbols(node)
                window = ir.query.normalized[max(0, demand.span.start - 48):demand.span.end]
                normalized_window = "".join(window.replace("`", "").split()).lower()
                if (any(self._field_matches_in_ir(field_text, symbol, ir)
                        for symbol in symbols)
                        or any("".join((symbol.raw_name or "").replace("`", "").split()).lower()
                               in normalized_window
                               for symbol in symbols if symbol.raw_name)):
                    continue
            if not (_spans_overlap(spans, demand.span) or self._same_clause(spans, demand.span)):
                return False
        return True

    def _boolean_demand_match(self, demand, ir: UnderstandingIR) -> bool:
        """Prove an absorbed connector against the semantic graph edge it denotes."""
        relation = str(demand.attributes.get("relation", "and"))
        roots = [ir.filters, *(item.condition for item in ir.events)]
        for root in roots:
            for node in _boolean_nodes(root):
                if node.operator == relation and (
                    self._node_has_evidence(node, demand.span)
                    or _connector_between_children(node, demand.span)
                ):
                    return True
        set_operation = "union" if relation == "or" else "intersection"
        return any(
            item.operation == set_operation
            and (
                self._node_has_evidence(item, demand.span)
                or _connector_between_nodes(ir.events, demand.span)
            )
            for item in ir.set_operations
        )

    def _node_has_evidence(self, node: object, requirement_span: SourceSpan | None) -> bool:
        if requirement_span is None:
            return True
        spans = _spans_for(node)
        return _spans_overlap(spans, requirement_span) or self._same_clause(spans, requirement_span)

    @staticmethod
    def _symbol_has_evidence(symbol, span: SourceSpan | None) -> bool:
        return bool(span is None or (symbol.span and _spans_overlap([symbol.span], span)))

    @staticmethod
    def _field_matches(expected: str, symbol) -> bool:
        if symbol is None or not expected:
            return False
        expected = expected.strip("`").lower()
        actual = (symbol.raw_name or symbol.canonical_id or "").strip("`").lower()
        canonical = (symbol.canonical_id or "").rsplit(".", 1)[-1].lower()
        return expected == actual or expected == canonical

    def _field_matches_in_ir(self, expected: str, symbol, ir: UnderstandingIR) -> bool:
        if self._field_matches(expected, symbol):
            return True
        identifiers = {
            value.lower() for value in [
                symbol.canonical_id or "",
                *(item.identifier for item in getattr(symbol, "candidates", [])),
            ] if value
        }
        if any(
            hypothesis.hypothesis_id.lower() in identifiers
            and expected.strip("`").lower() in {
                hypothesis.raw_name.strip("`").lower(),
                *(item.strip("`").lower() for item in hypothesis.aliases),
            }
            for hypothesis in ir.schema_hypotheses
        ):
            return True
        expected_name = expected.strip("`").lower()
        suffix_matches = [
            hypothesis for hypothesis in ir.schema_hypotheses
            if hypothesis_matches_field(
                expected_name, hypothesis.raw_name, hypothesis.aliases,
                hypothesis.description,
            )
        ]
        if (len(suffix_matches) == 1
                and suffix_matches[0].hypothesis_id.lower() in identifiers):
            return True
        return any(
            ({value.lower() for value in [
                candidate.canonical_id or "",
                *(item.identifier for item in getattr(candidate, "candidates", [])),
            ] if value} & identifiers)
            and self._field_matches(expected, candidate)
            for candidate in _all_symbols(ir)
        )

    def _same_clause(self, spans: list[SourceSpan], expected: SourceSpan) -> bool:
        graph = self._active_ir.clause_graph if getattr(self, "_active_ir", None) else None
        if graph is None:
            return False
        clause = next((item for item in graph.nodes
                       if item.span.start <= expected.start < item.span.end), None)
        return bool(clause and any(clause.span.start <= item.start < clause.span.end for item in spans))

    @staticmethod
    def uncovered_critical(ir: UnderstandingIR) -> list[RequirementSpec]:
        coverage_by_id = {item.requirement_id: item for item in ir.coverage}
        return [item for item in ir.requirements if item.critical
                and coverage_by_id.get(item.requirement_id, CoverageSpec(item.requirement_id)).status != "satisfied"]

    @staticmethod
    def _set_operation(text: str) -> str:
        if "交集" in text or "重合" in text:
            return "intersection"
        if "并集" in text:
            return "union"
        if "差集" in text:
            return "difference"
        return ""

    @staticmethod
    def _link_operator_requirements(ir: UnderstandingIR, coverage: list[CoverageSpec]) -> None:
        covered_by = {}
        for item in coverage:
            for node_id in item.covered_by:
                covered_by.setdefault(node_id, []).append(item.requirement_id)
        for invocation in ir.operator_invocations:
            invocation.requirement_ids = sorted({requirement_id for node_id in invocation.input_refs
                                                  for requirement_id in covered_by.get(node_id, [])})

    @staticmethod
    def refresh_ambiguities(ir: UnderstandingIR) -> None:
        preserved = [item for item in ir.ambiguities
                     if not item.ambiguity_id.startswith(("amb_unresolved_", "amb_conflict_"))]
        generated = [
            AmbiguitySpec(f"amb_unresolved_{index}", item.code, item.message, item.span,
                          list(item.candidates))
            for index, item in enumerate(ir.unresolved, start=1)
        ]
        generated.extend(
            AmbiguitySpec(f"amb_conflict_{index}", "claim_conflict",
                          f"声明 {claim.claim_id} 与其他候选冲突", candidates=list(claim.conflicts_with))
            for index, claim in enumerate((item for item in ir.claims if item.status == "conflict"), start=1)
        )
        ir.ambiguities = _dedupe_ambiguities(preserved + generated)


def _all_symbols(ir: UnderstandingIR):
    values = list(ir.projections) + list(ir.grouping)
    values.extend(item.field for item in ir.metrics)
    for root in [ir.filters, *(event.condition for event in ir.events)]:
        values.extend(item.field for item in walk_predicates(root) if item.field)
    for aggregate in ir.aggregates:
        values.extend(_expression_symbols(aggregate.aggregate.input))
    for group in ir.group_by_operations:
        for key in group.keys:
            values.extend(_expression_symbols(key))
    for window in ir.window_aggregates:
        values.extend(_expression_symbols(window.input))
    unique, seen = [], set()
    for symbol in values:
        key = (symbol.raw_name, symbol.canonical_id, symbol.span.start if symbol.span else None)
        if key not in seen:
            seen.add(key)
            unique.append(symbol)
    return unique


def _node_output(node: object) -> OutputRef | None:
    if isinstance(node, OutputRef):
        return node
    if isinstance(node, MetricSpec) and node.aggregation != "none":
        identifier = node.field.canonical_id or node.field.raw_name
        return OutputRef(
            f"metric_{identifier}_{node.aggregation}", "value", "number",
            unit=node.unit, shape="scalar", grain="scalar",
            fields=[identifier, node.field.raw_name, node.aggregation],
        )
    if hasattr(node, "aggregate"):
        return getattr(node.aggregate, "output", None)
    return getattr(node, "output", None) or getattr(node, "output_ref", None)


def _node_scope(node: object) -> str:
    if isinstance(node, MetricSpec) and node.aggregation != "none":
        return "global"
    value = getattr(node, "scope", "")
    if isinstance(value, str):
        return value
    return getattr(value, "window_type", "")


def _coverage_nodes(ir: UnderstandingIR) -> dict[str, object]:
    """Rehydrate stable CoverageSpec node IDs into their concrete producers."""
    values: dict[str, object] = {
        **{item.event_id: item for item in ir.events},
        **{item.output_name: item for item in ir.set_operations},
        **{item.aggregate.aggregate_id: item for item in ir.aggregates},
        **{item.group_id: item for item in ir.group_by_operations},
        **{item.top_k_id: item for item in ir.top_k_operations},
        **{item.window_id: item for item in ir.window_aggregates},
        **{item.count_id: item for item in ir.cumulative_counts},
        **{item.duration_id: item for item in ir.cumulative_durations},
        **{item.projection_id: item for item in ir.derived_projections},
        **{item.comparison_id: item for item in ir.comparisons},
        **{item.arithmetic_id: item for item in ir.arithmetic},
        **{item.reference_id: item for item in ir.references},
    }
    values.update({f"metric:{index}": item for index, item in enumerate(ir.metrics, start=1)})
    values.update({f"calculation:{index}": item for index, item in enumerate(ir.calculations, start=1)})
    values.update({item.calculation_id: item for item in ir.calculations if item.calculation_id})
    return values


def _nodes_for_coverage(coverage: CoverageSpec | None, ir: UnderstandingIR) -> list[object]:
    if coverage is None:
        return []
    nodes = _coverage_nodes(ir)
    return [nodes[item] for item in coverage.covered_by if item in nodes]


def _outputs_for_coverage(coverage: CoverageSpec | None, ir: UnderstandingIR) -> list[OutputRef]:
    return [output for node in _nodes_for_coverage(coverage, ir)
            if (output := _node_output(node)) is not None]


def _node_input_refs(node: object) -> list[str]:
    refs: list[str] = []
    if hasattr(node, "target_task_id") and node.target_task_id:
        refs.append(str(node.target_task_id))
    if hasattr(node, "input_refs"):
        refs.extend(item.ref_id if isinstance(item, OutputRef) else str(item) for item in node.input_refs)
    if hasattr(node, "scope_ref") and node.scope_ref:
        refs.append(node.scope_ref.ref_id)
    if hasattr(node, "aggregate"):
        refs.extend(_expression_refs(node.aggregate.input))
    if hasattr(node, "input_ref") and node.input_ref:
        refs.append(node.input_ref.ref_id)
    if hasattr(node, "inputs"):
        for value in node.inputs:
            if isinstance(value, ExpressionNode):
                refs.extend(_expression_refs(value))
    if hasattr(node, "parameters") and isinstance(node.parameters, dict):
        for key in ("filter_reference", "scope_ref", "input_ref"):
            if node.parameters.get(key):
                refs.append(str(node.parameters[key]))
    if hasattr(node, "rank_by"):
        refs.extend(_expression_refs(node.rank_by))
    if hasattr(node, "condition") and isinstance(node.condition, PredicateNode):
        for predicate in walk_predicates(node.condition):
            if predicate.right_expression is not None:
                refs.extend(_expression_refs(predicate.right_expression))
    if hasattr(node, "expression") and isinstance(node.expression, ExpressionNode):
        refs.extend(_expression_refs(node.expression))
    return list(dict.fromkeys(refs))


def _node_symbols(node: object):
    symbols = []
    if isinstance(node, PredicateNode):
        if node.field:
            symbols.append(node.field)
        for child in node.children:
            symbols.extend(_node_symbols(child))
    if hasattr(node, "aggregate"):
        symbols.extend(_expression_symbols(node.aggregate.input))
    if hasattr(node, "keys"):
        for value in node.keys:
            if isinstance(value, ExpressionNode):
                symbols.extend(_expression_symbols(value))
    if hasattr(node, "input") and isinstance(node.input, ExpressionNode):
        symbols.extend(_expression_symbols(node.input))
    if hasattr(node, "inputs"):
        for value in node.inputs:
            if isinstance(value, ExpressionNode):
                symbols.extend(_expression_symbols(value))
    if hasattr(node, "rank_by") and isinstance(node.rank_by, ExpressionNode):
        symbols.extend(_expression_symbols(node.rank_by))
    if hasattr(node, "expression") and isinstance(node.expression, ExpressionNode):
        symbols.extend(_expression_symbols(node.expression))
    if hasattr(node, "condition"):
        symbols.extend(_node_symbols(node.condition))
    if hasattr(node, "raw_name") and hasattr(node, "status"):
        symbols.append(node)
    return symbols


def _expression_symbols(expression: ExpressionNode | None):
    if expression is None:
        return []
    values = [expression.symbol] if expression.symbol else []
    for argument in expression.arguments:
        if isinstance(argument, ExpressionNode):
            values.extend(_expression_symbols(argument))
    return values


def _node_lineage_inputs(node: object, lineage: OutputLineageIndex) -> set[str]:
    values = set(_node_input_refs(node))
    output = _node_output(node)
    entry = lineage.get(output.ref_id) if output is not None else None
    if entry is not None:
        values.update(entry.scope_ancestors)
    return values


def _expression_refs(expression: ExpressionNode | None) -> list[str]:
    if expression is None:
        return []
    refs = [expression.reference] if expression.kind in {"reference", "output_ref"} and expression.reference else []
    for argument in expression.arguments:
        if isinstance(argument, ExpressionNode):
            refs.extend(_expression_refs(argument))
    return refs


def _spans_for(node: object) -> list[SourceSpan]:
    result: list[SourceSpan] = []
    for provenance in getattr(node, "provenance", []) or []:
        if provenance.span:
            result.append(provenance.span)
    span = getattr(node, "span", None)
    if isinstance(span, SourceSpan):
        result.append(span)
    result.extend(
        item for item in (getattr(node, "evidence_spans", []) or [])
        if isinstance(item, SourceSpan)
    )
    if isinstance(node, PredicateNode):
        if node.field and node.field.span:
            result.append(node.field.span)
        for child in node.children:
            result.extend(_spans_for(child))
    if isinstance(node, MetricSpec) and node.field.span:
        result.append(node.field.span)
    if hasattr(node, "aggregate"):
        result.extend(_spans_for(node.aggregate))
    return result


def _connector_between_children(node: PredicateNode, expected: SourceSpan | None) -> bool:
    if expected is None or len(node.children) < 2:
        return False
    return _connector_between_nodes(node.children, expected)


def _boolean_nodes(root: PredicateNode | None):
    if root is None:
        return
    if root.kind == "boolean":
        yield root
        for child in root.children:
            yield from _boolean_nodes(child)


def _connector_between_nodes(nodes: list[object], expected: SourceSpan | None) -> bool:
    if expected is None:
        return False
    spans = [span for node in nodes for span in _spans_for(node)]
    has_left = any(span.end <= expected.start for span in spans)
    has_right = any(span.start >= expected.end for span in spans)
    return has_left and has_right


def _spans_overlap(spans: list[SourceSpan], expected: SourceSpan | None) -> bool:
    return bool(expected and any(item.start < expected.end and expected.start < item.end for item in spans))


def _values_equal(left, right) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).lower() == str(right).lower()


def _canonical_unit(value: str | None) -> str | None:
    if not value:
        return None
    return {
        "℃": "celsius", "°c": "celsius", "c": "celsius",
        "db": "decibel", "decibel": "decibel",
        "m/s²": "m/s^2", "m/s2": "m/s^2", "m/s^2": "m/s^2",
        "kw": "kilowatt", "w": "watt", "mw": "megawatt",
        "mwh": "megawatt_hour", "kv": "kilovolt", "v": "volt",
        "a": "ampere", "mpa": "megapascal", "kpa": "kilopascal",
        "pa": "pascal", "km/h": "km/h", "mm/s": "mm/s",
        "m": "meter", "km": "kilometer", "ms": "millisecond",
        "μm": "micrometer", "um": "micrometer", "吨": "tonne",
        "万元": "currency", "元": "currency", "bps": "basis_point",
        "m³/h": "cubic_meter_per_hour", "m3/h": "cubic_meter_per_hour",
        "%": "percent", "百分比": "percent",
        "w/m²": "watt_per_square_meter", "w/m2": "watt_per_square_meter",
        "个/平方米": "count_per_square_meter",
        "分钟": "minute", "分": "minute", "秒": "second", "秒钟": "second",
        "小时": "hour", "时": "hour", "天": "day", "日": "day",
        "美元": "currency", "欧元": "currency", "元": "currency", "人民币": "currency",
    }.get(value.lower(), value.lower())


def _group_text(text: str) -> str:
    if "按" not in text or "分组" not in text:
        return ""
    return text.split("按", 1)[1].split("分组", 1)[0].strip(" `（）()")


def _dedupe_ambiguities(items: list[AmbiguitySpec]) -> list[AmbiguitySpec]:
    result, seen = [], set()
    for item in items:
        key = (item.kind, item.span.start if item.span else None,
               item.span.end if item.span else None, item.message)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
