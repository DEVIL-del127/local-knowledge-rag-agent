"""Read-only, single-source logical planning."""
from __future__ import annotations

from .catalog import CatalogSnapshot
from .merge import walk_predicates
from .models import (
    LogicalPlan,
    OutputRef,
    PhysicalPlan,
    ResultField,
    TaskNode,
    UnderstandingIR,
    ValidationReport,
)


class LogicalPlanner:
    def plan(self, ir: UnderstandingIR, validation: ValidationReport,
             catalog: CatalogSnapshot) -> LogicalPlan:
        # Never trust a stale report supplied by another caller.
        from .validator import IRValidator
        current = IRValidator().validate(ir, catalog)
        if not current.executable:
            return LogicalPlan(
                schema_version=ir.schema_version, catalog_version=ir.catalog_version,
                status="unsupported" if current.status == "unsupported" else "blocked",
                reason="planner preflight: " + "; ".join(
                    item.message for item in current.errors[:3]
                ),
            )
        if not validation.executable:
            return LogicalPlan(
                schema_version=ir.schema_version,
                catalog_version=ir.catalog_version,
                status="unsupported" if validation.status == "unsupported" else "blocked",
                reason="; ".join(item.message for item in validation.errors[:3]),
            )

        if ir.literature_contract:
            return self._plan_literature(ir, catalog)

        source_id = self._source_id(ir, catalog)
        source = catalog.source(source_id) if source_id else None
        if not source:
            if not (ir.goals or ir.projections or ir.metrics or ir.events or ir.set_operations
                    or ir.calculations or ir.aggregates or ir.group_by_operations
                    or ir.top_k_operations or ir.window_aggregates or ir.cumulative_counts
                    or ir.cumulative_durations or ir.relation_filters):
                return LogicalPlan(
                    ir.schema_version, ir.catalog_version, status="ready",
                    reason="semantic-control request has no retrieval task",
                )
            return LogicalPlan(ir.schema_version, ir.catalog_version,
                               status="blocked", reason="source unavailable")

        if (ir.events or ir.ordered_folds or ir.sequences or ir.group_by_operations
                or ir.top_k_operations or ir.window_aggregates or ir.cumulative_counts
                or ir.cumulative_durations):
            return self._plan_structured_analytics(ir, source_id, source)

        nodes: list[TaskNode] = []
        retrieve = TaskNode(
            task_id="t1_retrieve", task_type="retrieve", source_id=source_id,
            inputs={
                "query": ir.query.normalized,
                "filters": ir.filters.to_dict() if ir.filters else None,
                "temporal": [item.to_dict() for item in ir.temporal],
                "fields": list(dict.fromkeys(
                    [item.canonical_id for item in ir.projections if item.canonical_id]
                    + [item.field.canonical_id for item in ir.metrics if item.field.canonical_id]
                )),
            },
            output_schema=self._retrieval_schema(ir, catalog),
            required_capabilities=["retrieve"] + (["filter"] if ir.filters or ir.temporal else []),
            requirement_ids=self._requirement_ids(ir, {"target", "constraint"}),
            security_scope_digest=ir.query.security_scope_digest,
        )
        nodes.append(retrieve)
        previous = retrieve.task_id

        aggregates = [item for item in ir.metrics if item.aggregation != "none"]
        if aggregates or any(item.goal_type == "aggregate" for item in ir.goals):
            aggregate = TaskNode(
                task_id="t2_aggregate", task_type="aggregate", depends_on=[previous],
                source_id=source_id,
                inputs={
                    "metrics": [item.to_dict() for item in aggregates],
                    "group_by": ir.output.group_by,
                    "time_periods": [self._period(item) for item in ir.temporal],
                },
                output_schema=[
                    ResultField(item.alias or item.field.canonical_id or item.field.raw_name,
                                "number", item.unit)
                    for item in (aggregates or ir.metrics)
                ],
                required_capabilities=["aggregate"],
                requirement_ids=self._requirement_ids(ir, {"target", "constraint"}),
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(aggregate)
            previous = aggregate.task_id

        calculation_ids = []
        aggregate_ids = []
        for aggregate in ir.aggregates:
            task_id = aggregate.aggregate.aggregate_id
            node = TaskNode(
                task_id=task_id, task_type="aggregate", depends_on=[previous],
                source_id=source_id, inputs=aggregate.to_dict(),
                output_schema=[ResultField(aggregate.aggregate.output.port,
                                           aggregate.aggregate.output.result_type)],
                required_capabilities=["aggregate"],
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(node)
            aggregate_ids.append(task_id)
        if aggregate_ids:
            previous = aggregate_ids[-1]

        for comparison in ir.comparisons:
            node = TaskNode(
                task_id=comparison.comparison_id, task_type="compare",
                depends_on=list(aggregate_ids) or [previous], source_id=source_id,
                inputs=comparison.to_dict(),
                output_schema=[ResultField(comparison.output.port, "boolean")],
                required_capabilities=["aggregate"],
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(node)
            previous = node.task_id
        for arithmetic in ir.arithmetic:
            node = TaskNode(
                task_id=arithmetic.arithmetic_id, task_type="calculate",
                depends_on=[previous], source_id=source_id,
                inputs=arithmetic.to_dict(),
                output_schema=[ResultField(arithmetic.output.port, arithmetic.output.result_type,
                                           arithmetic.output.unit)],
                required_capabilities=["math.expression"],
                requirement_ids=self._operator_requirement_ids(ir, {"ARITHMETIC"}),
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(node)
            calculation_ids.append(node.task_id)
            previous = node.task_id
        for conversion in ir.conversions:
            node = TaskNode(
                task_id=conversion.conversion_id, task_type="convert",
                depends_on=[previous], source_id=source_id,
                inputs=conversion.to_dict(),
                output_schema=[ResultField(conversion.output.port, conversion.output.result_type,
                                           conversion.output.unit)],
                required_capabilities=["unit.convert"],
                requirement_ids=self._operator_requirement_ids(ir, {"CONVERT_UNIT", "CONVERT_CURRENCY"}),
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(node)
            calculation_ids.append(node.task_id)
            previous = node.task_id
        for index, calculation in enumerate(ir.calculations, start=1):
            task_id = f"t{len(nodes) + 1}_compute_{index}"
            node = TaskNode(
                task_id=task_id, task_type="calculate", depends_on=[previous],
                source_id=source_id,
                inputs=calculation.to_dict(),
                output_schema=[ResultField(f"{calculation.calculation_type}_{index}", "number")],
                required_capabilities=["calculator"],
                requirement_ids=self._requirement_ids(ir, {"target"}, "calculation"),
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(node)
            calculation_ids.append(task_id)

        if any(item.goal_type == "compare" for item in ir.goals) and not ir.calculations:
            compare = TaskNode(
                task_id=f"t{len(nodes) + 1}_compare", task_type="compare",
                depends_on=[previous], source_id=source_id,
                inputs={"criteria": ir.output.criteria},
                output_schema=[ResultField("comparison", "object")],
                required_capabilities=["compare"],
                requirement_ids=self._requirement_ids(ir, {"target"}),
                security_scope_digest=ir.query.security_scope_digest,
            )
            nodes.append(compare)
            previous = compare.task_id
        elif calculation_ids:
            previous = calculation_ids[-1]

        if any(item.goal_type == "present" for item in ir.goals) or ir.output.criteria:
            dependencies = calculation_ids or [previous]
            nodes.append(TaskNode(
                task_id=f"t{len(nodes) + 1}_compose", task_type="compose",
                depends_on=list(dict.fromkeys(dependencies)), source_id=source_id,
                inputs=ir.output.to_dict(),
                output_schema=[ResultField("result", "object")],
                required_capabilities=[],
                requirement_ids=self._requirement_ids(ir, {"output"}),
                security_scope_digest=ir.query.security_scope_digest,
            ))

        missing = sorted({
            capability for node in nodes for capability in node.required_capabilities
            if capability not in source.capabilities
            and capability not in {"calculator", "compare", "structured_analytics", "math.expression", "unit.convert"}
        })
        if missing:
            for node in nodes:
                if any(item in missing for item in node.required_capabilities):
                    node.status = "blocked"
            return LogicalPlan(ir.schema_version, ir.catalog_version, nodes,
                               status="unsupported",
                               reason=f"数据源缺少能力：{', '.join(missing)}")
        if self._has_cycle(nodes):
            return LogicalPlan(ir.schema_version, ir.catalog_version, nodes,
                               status="blocked", reason="task dependency cycle")
        return LogicalPlan(ir.schema_version, ir.catalog_version, nodes, status="ready")

    def _plan_structured_analytics(self, ir: UnderstandingIR, source_id: str,
                                   source) -> LogicalPlan:
        """Keep domain operators inside one unbound analytics capability."""
        retrieve = TaskNode(
            task_id="t1_retrieve",
            task_type="retrieve",
            source_id=source_id,
            inputs={
                "query": ir.query.normalized,
                "temporal": [item.to_dict() for item in ir.temporal],
                "fields": list(dict.fromkeys([
                    field_id for event in ir.events
                    for field_id in _event_field_ids(event)
                ] + _analytic_field_ids(ir))),
                "scan_limits": {
                    "max_scan_points": source.metadata.get("max_scan_points"),
                    "scan_chunk_size": source.metadata.get("scan_chunk_size"),
                    "max_partitions": source.metadata.get("max_partitions"),
                    "overflow_policy": "reject_or_chunk",
                },
            },
            output_schema=self._retrieval_schema(ir, _SourceCatalogView(source)),
            required_capabilities=["retrieve", "filter"],
            requirement_ids=self._requirement_ids(ir, {"target", "constraint"}),
            security_scope_digest=ir.query.security_scope_digest,
        )
        analytics = TaskNode(
            task_id="t2_calculate",
            task_type="calculate",
            depends_on=[retrieve.task_id],
            source_id=source_id,
            inputs={
                "capability": "structured_analytics",
                "algebra": ["filter", "window", "aggregate", "set", "transform"],
                "events": [item.to_dict() for item in ir.events],
                "set_operations": [item.to_dict() for item in ir.set_operations],
                "derived_projections": [item.to_dict() for item in ir.derived_projections],
                "aggregates": [item.to_dict() for item in ir.aggregates],
                "group_by_operations": [item.to_dict() for item in ir.group_by_operations],
                "top_k_operations": [item.to_dict() for item in ir.top_k_operations],
                "window_aggregates": [item.to_dict() for item in ir.window_aggregates],
                "cumulative_counts": [item.to_dict() for item in ir.cumulative_counts],
                "cumulative_durations": [item.to_dict() for item in ir.cumulative_durations],
                "comparisons": [item.to_dict() for item in ir.comparisons],
                "calculations": [item.to_dict() for item in ir.calculations],
                "arithmetic": [item.to_dict() for item in ir.arithmetic],
                "relation_filters": [item.to_dict() for item in ir.relation_filters],
                "conversions": [item.to_dict() for item in ir.conversions],
                "ordered_folds": [item.to_dict() for item in ir.ordered_folds],
                "sequences": [item.to_dict() for item in ir.sequences],
                "operator_invocations": [item.to_dict() for item in ir.operator_invocations],
            },
            output_schema=[
                *[ResultField(item.output_name, "event_result") for item in ir.events],
                *[field for item in ir.aggregates
                  for field in _output_result_fields(item.aggregate.output)],
                *[field for item in ir.group_by_operations
                  for field in _output_result_fields(item.output)],
                *[field for item in ir.top_k_operations
                  for field in _output_result_fields(item.output)],
                *[field for item in ir.window_aggregates
                  for field in _output_result_fields(item.output)],
                *[field for item in ir.cumulative_counts
                  for field in _output_result_fields(item.output)],
                *[field for item in ir.cumulative_durations
                  for field in _output_result_fields(item.output)],
                *[ResultField(item.output_name, item.granularity)
                  for item in ir.set_operations],
                *[field for index, item in enumerate(ir.calculations, start=1)
                  for field in _output_result_fields(
                      item.output or OutputRef(f"calculation_{index}", "value", "number",
                                               fields=[f"{item.calculation_type}_{index}"])
                  )],
                *[ResultField(item.output.port, item.output.result_type, item.output.unit)
                  for item in ir.arithmetic],
                *[field for item in ir.relation_filters
                  for field in _output_result_fields(item.output)],
                *[ResultField(item.output.port, item.output.result_type, item.output.unit)
                  for item in ir.conversions],
                *[ResultField(item.output.port, item.output.result_type, item.output.unit)
                  for item in ir.sequences],
            ],
            required_capabilities=[
                "structured_analytics",
                *(["state.resettable_scan"] if any(
                    item.transition.reset_condition for item in ir.ordered_folds
                ) else []),
                *(["state.ordered_fold"] if any(
                    not item.transition.reset_condition for item in ir.ordered_folds
                ) else []),
                *(["timeseries.sequence"] if ir.sequences else []),
                *(["math.expression"] if ir.arithmetic else []),
                *(["unit.convert"] if ir.conversions else []),
                *(["read.group"] if ir.grouping or ir.output.group_by or ir.group_by_operations else []),
                *(["read.sort_top_k"] if ir.output.limit or ir.top_k_operations else []),
            ],
            requirement_ids=self._operator_requirement_ids(ir),
            security_scope_digest=ir.query.security_scope_digest,
        )
        compose = TaskNode(
            task_id="t3_compose",
            task_type="compose",
            depends_on=[analytics.task_id],
            source_id=source_id,
            inputs=ir.output.to_dict(),
            output_schema=[ResultField("result", "object")],
            requirement_ids=self._requirement_ids(ir, {"output"}),
            security_scope_digest=ir.query.security_scope_digest,
        )
        nodes = [retrieve, analytics, compose]
        if self._has_cycle(nodes):
            return LogicalPlan(ir.schema_version, ir.catalog_version, nodes,
                               status="blocked", reason="task dependency cycle")
        return LogicalPlan(ir.schema_version, ir.catalog_version, nodes, status="ready")

    @staticmethod
    def physical_shell(ir: UnderstandingIR) -> PhysicalPlan:
        return PhysicalPlan(schema_version=ir.schema_version, catalog_version=ir.catalog_version)

    @staticmethod
    def _requirement_ids(ir: UnderstandingIR, roles: set[str],
                         requirement_type: str | None = None) -> list[str]:
        return [
            item.requirement_id for item in ir.requirements
            if item.role in roles
            and (requirement_type is None or item.requirement_type == requirement_type)
        ]

    @staticmethod
    def _plan_literature(ir: UnderstandingIR, catalog: CatalogSnapshot) -> LogicalPlan:
        contract = dict(ir.literature_contract)
        task = str(contract.get("task") or "qa")
        source_id = str((ir.source_binding or {}).get("source_id") or "")
        source = catalog.source(source_id) if source_id else None
        if source is None:
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked", reason="source unavailable")
        common = {
            "accepted_ir_digest": ir.accepted_ir_digest,
            "literature_contract": contract,
        }
        if task == "inventory":
            nodes = [TaskNode(
                task_id="literature_inventory_1", task_type="inventory", source_id=source_id,
                inputs=common, required_capabilities=["inventory"],
                security_scope_digest=ir.query.security_scope_digest,
            )]
        else:
            reference = dict(contract.get("document_reference") or {})
            explicit = reference.get("kind") == "explicit_document"
            first_type = "resolve_document" if explicit else "discover_documents"
            nodes = [TaskNode(
                task_id="literature_select_1", task_type=first_type, source_id=source_id,
                inputs=common, required_capabilities=["retrieve"],
                security_scope_digest=ir.query.security_scope_digest,
            )]
            if task in {"qa", "summarize", "compare"}:
                count = 2 if task == "compare" else 1
                for index in range(count):
                    nodes.append(TaskNode(
                        task_id=f"literature_passages_{index + 1}",
                        task_type="retrieve_document_passages",
                        depends_on=["literature_select_1"], source_id=source_id,
                        inputs={**common, "result_dependency": f"literature_select_1[{index}]"},
                        required_capabilities=["retrieve"],
                        security_scope_digest=ir.query.security_scope_digest,
                    ))
        return LogicalPlan(
            schema_version=ir.schema_version, catalog_version=ir.catalog_version,
            nodes=nodes, status="ready", reason="accepted literature contract compiled",
        )

    @staticmethod
    def _operator_requirement_ids(ir: UnderstandingIR,
                                  operators: set[str] | None = None) -> list[str]:
        return sorted({
            requirement_id
            for invocation in ir.operator_invocations
            if operators is None or invocation.operator_id in operators
            for requirement_id in invocation.requirement_ids
        })

    @staticmethod
    def _source_id(ir: UnderstandingIR, catalog: CatalogSnapshot) -> str | None:
        ids = [
            item.identifier for item in ir.source_candidates
            if item.status == "resolved" and catalog.source(item.identifier)
        ]
        return ids[0] if len(set(ids)) == 1 else None

    @staticmethod
    def _retrieval_schema(ir: UnderstandingIR, catalog: CatalogSnapshot) -> list[ResultField]:
        fields = []
        symbols = list(ir.projections)
        symbols.extend(item.field for item in ir.metrics)
        seen = set()
        for symbol in symbols:
            field_id = symbol.canonical_id or symbol.raw_name
            if field_id in seen:
                continue
            seen.add(field_id)
            spec = catalog.field(symbol.canonical_id or "")
            fields.append(ResultField(
                field_id,
                spec.data_type if spec else "unknown",
                spec.unit if spec else None,
            ))
        return fields or [ResultField("document", "object")]

    @staticmethod
    def _period(item) -> str:
        return (item.from_value or item.exact or item.raw)[:7]

    @staticmethod
    def _has_cycle(nodes: list[TaskNode]) -> bool:
        graph = {node.task_id: node.depends_on for node in nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            if any(dependency in graph and visit(dependency) for dependency in graph.get(node_id, [])):
                return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in graph)


def _event_field_ids(event) -> list[str]:
    result = []
    for node in walk_predicates(event.condition):
        if node.field and node.field.canonical_id:
            result.append(node.field.canonical_id)
    if event.sampling.order_by and event.sampling.order_by.canonical_id:
        result.append(event.sampling.order_by.canonical_id)
    result.extend(
        item.canonical_id for item in event.sampling.partition_by if item.canonical_id
    )
    return result


def _analytic_field_ids(ir: UnderstandingIR) -> list[str]:
    """Collect catalog fields used by explicit analytic DAG nodes."""
    expressions = []
    for aggregate in ir.aggregates:
        expressions.append(aggregate.aggregate.input)
        expressions.extend(aggregate.group_by)
    for group in ir.group_by_operations:
        expressions.extend(group.keys)
    for window in ir.window_aggregates:
        expressions.append(window.input)
    for count in ir.cumulative_counts:
        expressions.extend(node.field for node in walk_predicates(count.action) if node.field)
    for duration in ir.cumulative_durations:
        expressions.extend(node.field for node in walk_predicates(duration.condition) if node.field)
    result = []
    for expression in expressions:
        symbol = getattr(expression, "symbol", expression)
        identifier = getattr(symbol, "canonical_id", None)
        if identifier:
            result.append(identifier)
    return result


def _output_result_fields(output: OutputRef) -> list[ResultField]:
    """Expose semantic OutputRef fields instead of the generic ``value`` port."""
    return [ResultField(name, output.result_type, output.unit)
            for name in (output.fields or [output.port])]


class _SourceCatalogView:
    """Small adapter used only to reuse retrieval schema generation."""

    def __init__(self, source):
        self._source = source

    def field(self, field_id: str):
        return next((item for item in self._source.fields if item.field_id == field_id), None)
