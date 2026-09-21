"""Read-only typed LogicalPlan compiler, disabled until M1.5 release gates pass."""
from __future__ import annotations

from .catalog import CatalogSnapshot
from .event_execution import EventReadinessReport
from .models import LogicalPlan, ResultField, TaskNode, UnderstandingIR
from .typed_lineage import TypedLineageReport


class M15LogicalPlanner:
    def plan(self, ir: UnderstandingIR, catalog: CatalogSnapshot,
             readiness: EventReadinessReport | None,
             lineage: TypedLineageReport | None) -> LogicalPlan:
        if readiness is None or lineage is None:
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked",
                               reason="M1.5 typed contracts unavailable")
        contracts = [*readiness.consecutive_events, *readiness.duration_accumulations,
                     *readiness.cumulative_values]
        if not contracts or any(item.readiness != "logical_plan_ready" for item in contracts):
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked",
                               reason="M1.5 binding or policy preflight incomplete")
        if not lineage.ready:
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked",
                               reason="M1.5 typed lineage preflight failed: " + "; ".join(lineage.blockers[:3]))
        source_ids = [item.identifier for item in ir.source_candidates
                      if item.status == "resolved" and catalog.source(item.identifier)]
        if len(set(source_ids)) != 1:
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked",
                               reason="M1.5 requires one bound read-only source")
        source_id = source_ids[0]
        source = catalog.source(source_id)
        if not source or not source.read_only:
            return LogicalPlan(ir.schema_version, ir.catalog_version, status="blocked",
                               reason="M1.5 source is unavailable or not read-only")
        nodes = [TaskNode(
            "m15_read", "retrieve", source_id=source_id,
            inputs={"temporal": [item.to_dict() for item in ir.temporal]},
            output_schema=[ResultField("rows", "relation")],
            required_capabilities=["retrieve"], binding_status="bound",
            security_scope_digest=ir.query.security_scope_digest,
        )]
        producer_tasks = {"source_relation.rows": "m15_read"}
        for index, item in enumerate(lineage.nodes, start=1):
            task_id = f"m15_{index}_{item.operator.lower()}"
            dependencies = sorted({producer_tasks[ref] for ref in item.input_refs if ref in producer_tasks})
            nodes.append(TaskNode(
                task_id, item.operator, depends_on=dependencies or [nodes[-1].task_id],
                source_id=source_id, inputs={"input_refs": list(item.input_refs)},
                output_schema=[ResultField(item.ref_id, item.result_type)],
                required_capabilities=[_capability(item.operator)], binding_status="bound",
                security_scope_digest=ir.query.security_scope_digest,
            ))
            producer_tasks[item.ref_id] = task_id
        nodes.append(TaskNode(
            "m15_compose", "compose",
            depends_on=[producer_tasks[item] for item in lineage.terminal_refs if item in producer_tasks],
            source_id=source_id, output_schema=[ResultField("result", "object")],
            binding_status="bound", security_scope_digest=ir.query.security_scope_digest,
        ))
        return LogicalPlan(ir.schema_version, ir.catalog_version, nodes, status="ready")


def _capability(operator):
    return {
        "ConsecutiveEvent": "event_segmentation",
        "ProjectLocalDate": "project_local_date",
        "SetIntersection": "set_operation",
        "SetUnion": "set_operation",
        "SetDifference": "set_operation",
        "DurationAccumulation": "duration_aggregation",
        "CumulativeValue": "aggregate",
        "TypedArithmetic": "math.expression",
        "Calculation": "calculator",
    }.get(operator, "structured_analytics")
