"""Semantic and safety validation for UnderstandingIR."""
from __future__ import annotations

import re

from .catalog import CatalogSnapshot, FieldSpec
from .merge import walk_predicates
from .models import (
    Diagnostic,
    ExpressionNode,
    LiteralValue,
    SymbolRef,
    UnderstandingIR,
    ValidationReport,
)
from .coverage import CoverageMatcher
from .expression_signatures import DEFAULT_EXPRESSION_SIGNATURES, ValueSignature


_NUMERIC_TYPES = {"integer", "long", "float", "double", "number"}
_UNIT_FAMILIES = {
    "celsius": "temperature", "fahrenheit": "temperature",
    "m/s": "speed", "km/h": "speed",
    "percent": "ratio", "ratio": "ratio",
    "currency": "currency", "cny": "currency", "usd": "currency",
    "mm": "length",
}
_CALCULATIONS = {
    "mom", "yoy", "growth", "difference", "ratio", "volatility", "covariance",
    "overlap_ratio", "conversion_rate", "median", "quantile", "correlation",
    "maximum",
}


def _field_text_from_span(field_text: str, anchor_text: str) -> bool:
    """Only allow the harmless removal of the enclosing backticks.

    In particular, this rejects the old forged shape ``span='连续'`` with
    ``field_text='温度'`` instead of trying to rescue it through nearby text.
    """
    if not field_text:
        return False
    anchor = anchor_text.strip()
    if anchor.startswith("`") and anchor.endswith("`"):
        anchor = anchor[1:-1]
    return field_text.strip() == anchor


def _values_equal(left, right) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _expression_output_refs(expression: ExpressionNode | None) -> list[str]:
    if expression is None:
        return []
    result = [expression.reference] if expression.kind in {"output_ref", "reference"} and expression.reference else []
    for argument in expression.arguments:
        if isinstance(argument, ExpressionNode):
            result.extend(_expression_output_refs(argument))
    return result


def _predicate_output_refs(predicate: PredicateNode | None) -> list[str]:
    """Return OutputRef dependencies embedded in an event/duration predicate."""
    if predicate is None:
        return []
    refs: list[str] = []
    for node in walk_predicates(predicate):
        if node.right_expression is not None:
            refs.extend(_expression_output_refs(node.right_expression))
        if node.field and node.field.ref_kind == "output":
            refs.append(node.field.canonical_id or node.field.raw_name)
    return list(dict.fromkeys(item for item in refs if item))


class IRValidator:
    def validate(self, ir: UnderstandingIR, catalog: CatalogSnapshot) -> ValidationReport:
        if ir.literature_contract:
            return self._validate_literature_contract(ir, catalog)
        errors: list[Diagnostic] = []
        unsupported = False
        resolved_source = None
        source_demands_valid = self._validate_source_demands(ir, errors)

        # RequirementIR is source evidence.  CoverageReport is the only place
        # where satisfaction may be recorded; never trust a mutable field on a
        # requirement as proof of semantic completeness.
        uncovered_requirements = CoverageMatcher.uncovered_critical(ir)
        for item in uncovered_requirements:
            errors.append(Diagnostic(
                "error", "requirement_uncovered",
                f"明确需求未被可靠覆盖：{item.text or item.requirement_type}", item.span,
            ))
        conflicting_claims = [item for item in ir.claims if item.status == "conflict"]
        if conflicting_claims:
            errors.append(Diagnostic(
                "error", "semantic_claim_conflict",
                f"存在 {len(conflicting_claims)} 个互斥语义声明，必须澄清",
            ))
        open_ambiguities = [item for item in ir.ambiguities if item.status == "open"]
        for item in open_ambiguities:
            errors.append(Diagnostic(
                "error", "semantic_ambiguity", item.message, item.span,
            ))
        for item in ir.unsatisfiable:
            errors.append(Diagnostic(
                "error", "unsatisfiable_constraints", item.reason,
                item.predicate_spans[0] if item.predicate_spans else None,
            ))

        symbols = self._symbols(ir)
        self._validate_query_schema_isolation(ir, symbols, catalog, errors)
        for symbol in symbols:
            if symbol.ref_kind == "output":
                continue
            if symbol.status != "resolved" or not symbol.canonical_id:
                errors.append(Diagnostic(
                    "error", "unresolved_symbol", f"字段未解析：{symbol.raw_name}", symbol.span,
                ))
            elif catalog.field(symbol.canonical_id) is None:
                errors.append(Diagnostic(
                    "error", "unknown_catalog_id", f"Catalog 不存在字段：{symbol.canonical_id}", symbol.span,
                ))

        for item in ir.unresolved:
            errors.append(Diagnostic("error", item.code, item.message, item.span))
            if item.code == "write_operation":
                unsupported = True

        context_blocked = any(item.code == "context_required" for item in ir.unresolved)
        source_ids = self._source_ids(ir, symbols, catalog)
        source_required = bool(symbols or ir.goals or ir.events or ir.set_operations or ir.metrics
                               or ir.calculations or ir.aggregates or ir.group_by_operations
                               or ir.top_k_operations or ir.window_aggregates or ir.cumulative_counts
                               or ir.cumulative_durations)
        if not source_ids:
            if source_required:
                errors.append(Diagnostic("error", "source_unresolved", "无法确定只读数据源"))
        elif len(source_ids) > 1:
            unsupported = not context_blocked
            errors.append(Diagnostic(
                "error", "multi_source_deferred" if context_blocked else "multi_source_unsupported",
                ("缺少会话上下文，暂不判定候选数据源" if context_blocked else
                 f"第一阶段不支持跨数据源查询：{', '.join(sorted(source_ids))}"),
            ))
        else:
            resolved_source = catalog.source(next(iter(source_ids)))
            if resolved_source is None:
                errors.append(Diagnostic("error", "source_missing", "数据源不在当前 Catalog"))
            elif not resolved_source.read_only:
                unsupported = True
                errors.append(Diagnostic("error", "write_source_blocked", "第一阶段只允许只读数据源"))

        predicate_roots = [ir.filters]
        predicate_roots.extend(event.condition for event in ir.events)
        for predicate in (
            item for root in predicate_roots for item in walk_predicates(root)
        ):
            if not predicate.field or not predicate.field.canonical_id:
                continue
            field = catalog.field(predicate.field.canonical_id)
            if not field:
                continue
            if predicate.operator not in field.allowed_operators and predicate.operator not in {
                "is_null", "is_not_null",
            }:
                errors.append(Diagnostic(
                    "error", "operator_type_mismatch",
                    f"字段 {field.field_id} 不支持操作符 {predicate.operator}",
                    predicate.field.span,
                ))
            if predicate.value:
                self._validate_value(field, predicate.value.unit, predicate.value.value_type,
                                     errors, predicate.field.span)
            if predicate.right_expression:
                inferred = _infer_expression(predicate.right_expression, {}, catalog, errors)
                field_type = "number" if field.data_type in _NUMERIC_TYPES else field.data_type
                inferred_type = "number" if inferred[0] in _NUMERIC_TYPES else inferred[0]
                if (field_type, field.unit) != (inferred_type, inferred[1]):
                    errors.append(Diagnostic(
                        "error", "dynamic_threshold_type_mismatch",
                        f"字段 {field.field_id} 与动态阈值类型或单位不兼容",
                        predicate.field.span,
                    ))

        for metric in ir.metrics:
            if not metric.field.canonical_id:
                continue
            field = catalog.field(metric.field.canonical_id)
            if field and metric.aggregation not in field.aggregations and metric.aggregation != "none":
                errors.append(Diagnostic(
                    "error", "aggregation_type_mismatch",
                    f"字段 {field.field_id} 不支持聚合 {metric.aggregation}", metric.field.span,
                ))

        event_outputs = {item.output_name for item in ir.events}
        output_refs = {
            item.output.ref_id: item.output for item in ir.derived_projections
        }
        # These operators produce typed OutputRefs independently of their
        # placement in the validation passes.  Make the references visible to
        # downstream set/comparison/expression checks while their own shape is
        # still validated below.
        output_refs.update({
            item.output_ref.ref_id: item.output_ref
            for item in ir.events if item.output_ref is not None
        })
        output_refs.update({item.output.ref_id: item.output for item in ir.arithmetic})
        output_refs.update({item.output.ref_id: item.output for item in ir.relation_filters})
        output_refs.update({item.output.ref_id: item.output for item in ir.sequences})
        output_refs.update({item.output.ref_id: item.output for item in ir.ordered_folds})
        output_refs.update({item.output.ref_id: item.output for item in ir.group_by_operations})
        output_refs.update({item.output.ref_id: item.output for item in ir.cumulative_durations})
        for event in ir.events:
            metric = catalog.derived_metric(event.derived_metric.metric_id)
            if metric is None:
                errors.append(Diagnostic(
                    "error", "derived_metric_missing",
                    f"Catalog 不存在派生指标：{event.derived_metric.metric_id}",
                ))
                continue
            if event.threshold is None:
                if not event.derived_metric.arguments.get("emit_all_intervals"):
                    errors.append(Diagnostic(
                        "error", "invalid_duration_threshold",
                        f"事件 {event.event_id} 缺少时长阈值或显式全区间策略",
                    ))
            elif (event.threshold.operator not in {"gt", "gte"}
                  or event.threshold.normalized_seconds <= 0):
                errors.append(Diagnostic(
                    "error", "invalid_duration_threshold",
                    f"事件 {event.event_id} 的时长阈值无效",
                ))
            if not event.window.from_value or not event.window.to_value:
                errors.append(Diagnostic(
                    "error", "event_window_unbound",
                    f"事件 {event.event_id} 缺少完整时间窗口",
                ))
            if not event.sampling.order_by or not event.sampling.order_by.canonical_id:
                errors.append(Diagnostic(
                    "error", "sampling_order_unbound",
                    f"事件 {event.event_id} 缺少时间排序字段",
                ))
            if event.derived_metric.metric_id.endswith(".consecutive_duration"):
                if not event.sampling.partition_by:
                    errors.append(Diagnostic(
                        "error", "sampling_partition_unbound",
                        f"事件 {event.event_id} 缺少设备分区字段",
                    ))
                if not event.sampling.max_gap_seconds:
                    errors.append(Diagnostic(
                        "error", "sampling_gap_unbound",
                        f"事件 {event.event_id} 缺少连续区间最大采样间隔",
                    ))
                if event.sampling.missing_data_policy != "break_segment":
                    errors.append(Diagnostic(
                        "error", "unsafe_continuity_policy",
                        "连续事件遇到缺失数据时必须中断区间",
                    ))
            if event.derived_metric.metric_id.endswith(".cumulative_duration"):
                integration = event.derived_metric.arguments.get("integration_method")
                if integration not in {"step", "linear"}:
                    errors.append(Diagnostic(
                        "error", "duration_integration_unbound",
                        f"累计时长缺少受支持的积分方法：{integration}",
                    ))
            if resolved_source:
                missing = [
                    item for item in metric.required_capabilities
                    if item not in resolved_source.capabilities
                ]
                if missing:
                    unsupported = True
                    errors.append(Diagnostic(
                        "error", "derived_capability_missing",
                        f"数据源缺少派生指标能力：{', '.join(missing)}",
                    ))

        set_outputs = set()
        set_output_refs = set()
        for operation in ir.set_operations:
            if operation.operation not in {"intersection", "union", "difference"}:
                unsupported = True
                errors.append(Diagnostic(
                    "error", "set_operation_unsupported",
                    f"不支持集合操作：{operation.operation}",
                ))
            if len(set(operation.inputs)) != len(operation.inputs):
                errors.append(Diagnostic(
                    "error", "set_inputs_not_distinct", "集合操作输入必须来自不同生产节点",
                ))
            missing_inputs = [
                item for item in operation.inputs
                if item not in event_outputs and item not in output_refs
            ]
            if missing_inputs:
                errors.append(Diagnostic(
                    "error", "set_input_unbound",
                    f"集合操作输入未绑定：{missing_inputs}",
                ))
            set_outputs.add(operation.output_name)
            if operation.output_ref:
                output_refs[operation.output_ref.ref_id] = operation.output_ref
                set_output_refs.add(operation.output_ref.ref_id)

        for group in ir.group_by_operations:
            if group.input_ref.ref_id != "source_relation.rows" and group.input_ref.ref_id not in output_refs:
                errors.append(Diagnostic("error", "group_input_unbound",
                                         f"分组 {group.group_id} 的输入未绑定"))
            if not group.keys:
                errors.append(Diagnostic("error", "group_keys_unbound",
                                         f"分组 {group.group_id} 缺少键"))
            for key in group.keys:
                _infer_expression(key, output_refs, catalog, errors)
            if group.output.shape != "relation" or group.output.grain != "group":
                errors.append(Diagnostic("error", "group_output_shape_mismatch",
                                         f"分组 {group.group_id} 必须输出 group 粒度 relation"))

        for aggregate in ir.aggregates:
            spec = aggregate.aggregate
            if spec.function not in {"sum", "avg", "min", "max", "count", "stddev", "quantile"}:
                errors.append(Diagnostic("error", "aggregate_unsupported",
                                         f"不支持聚合：{spec.function}"))
                continue
            inferred = _infer_expression(spec.input, output_refs, catalog, errors)
            expected_type, expected_unit = "number", None if spec.function == "count" else inferred[1]
            if spec.function != "quantile":
                signature = DEFAULT_EXPRESSION_SIGNATURES.validate(
                    spec.function,
                    [ValueSignature(
                        inferred[0], inferred[1],
                        _expression_shape(spec.input, output_refs), "measure",
                    )],
                    semantic_role="measure",
                )
                if not signature.valid:
                    errors.append(Diagnostic(
                        "error", "aggregate_signature_invalid",
                        f"聚合 {spec.aggregate_id} 输入不符合签名：{signature.message}",
                    ))
                else:
                    expected_type, expected_unit = signature.result_type, signature.unit
            if spec.output.result_type != expected_type or spec.output.unit != expected_unit:
                errors.append(Diagnostic(
                    "error", "aggregate_unit_mismatch", f"聚合 {spec.aggregate_id} 的输出类型或单位不一致",
                ))
            if aggregate.scope == "relation" and aggregate.scope_ref is None:
                errors.append(Diagnostic("error", "aggregate_scope_unbound",
                                         f"聚合 {spec.aggregate_id} 缺少 relation scope_ref"))
            if spec.function == "quantile":
                value = spec.parameters.get("quantile")
                if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
                    errors.append(Diagnostic("error", "quantile_parameter_invalid",
                                             f"聚合 {spec.aggregate_id} 缺少有效分位数"))
            output_refs[spec.output.ref_id] = spec.output

        for window in ir.window_aggregates:
            if window.function not in {"sum", "avg", "min", "max", "count", "stddev", "quantile"}:
                errors.append(Diagnostic("error", "window_aggregate_unsupported",
                                         f"窗口聚合 {window.window_id} 不支持 {window.function}"))
            if window.input_ref and window.input_ref.ref_id not in output_refs and window.input_ref.ref_id != "source_relation.rows":
                errors.append(Diagnostic("error", "window_input_unbound",
                                         f"窗口聚合 {window.window_id} 的输入未绑定"))
            inferred = _infer_expression(window.input, output_refs, catalog, errors)
            expected_unit = None if window.function == "count" else inferred[1]
            if (window.output.result_type, window.output.unit) != ("number", expected_unit):
                errors.append(Diagnostic("error", "window_aggregate_unit_mismatch",
                                         f"窗口聚合 {window.window_id} 的输出类型或单位不一致"))
            output_refs[window.output.ref_id] = window.output

        for count in ir.cumulative_counts:
            if count.action.operator != "eq" or count.action.field is None or count.action.value is None:
                errors.append(Diagnostic("error", "cumulative_count_action_unbound",
                                         f"累计计数 {count.count_id} 缺少明确 action 条件"))
            if count.scope.granularity not in {"day", "window"}:
                errors.append(Diagnostic("error", "cumulative_count_scope_invalid",
                                         f"累计计数 {count.count_id} 缺少明确时序范围"))
            if (count.output.result_type, count.output.unit) != ("number", "count"):
                errors.append(Diagnostic("error", "cumulative_count_output_mismatch",
                                         f"累计计数 {count.count_id} 输出必须为 count"))
            output_refs[count.output.ref_id] = count.output

        for duration in ir.cumulative_durations:
            if (duration.condition.operator not in {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in"}
                    or duration.condition.field is None
                    or duration.condition.value is None):
                errors.append(Diagnostic(
                    "error", "cumulative_duration_condition_unbound",
                    f"累计时长 {duration.duration_id} 缺少明确事件条件",
                ))
            if duration.scope.granularity not in {"day", "window", "interval"}:
                errors.append(Diagnostic(
                    "error", "cumulative_duration_scope_invalid",
                    f"累计时长 {duration.duration_id} 缺少明确时序范围",
                ))
            if duration.input_ref and duration.input_ref.ref_id != "source_relation.rows" and duration.input_ref.ref_id not in output_refs:
                errors.append(Diagnostic(
                    "error", "cumulative_duration_input_unbound",
                    f"累计时长 {duration.duration_id} 的输入未绑定",
                ))
            if duration.integration_method not in {"step", "linear"}:
                errors.append(Diagnostic(
                    "error", "duration_integration_unbound",
                    f"累计时长 {duration.duration_id} 缺少受支持的积分方法",
                ))
            if (duration.output.result_type, duration.output.unit) != ("number", "second"):
                errors.append(Diagnostic(
                    "error", "cumulative_duration_output_mismatch",
                    f"累计时长 {duration.duration_id} 输出必须为 second 数值",
                ))
            output_refs[duration.output.ref_id] = duration.output

        for arithmetic in ir.arithmetic:
            inferred = _infer_expression(arithmetic.expression, output_refs, catalog, errors)
            declared = (arithmetic.output.result_type, arithmetic.output.unit)
            if inferred != declared:
                errors.append(Diagnostic(
                    "error", "arithmetic_output_mismatch",
                    f"算术表达式 {arithmetic.arithmetic_id} 的输出声明与推导不一致",
                ))
            output_refs[arithmetic.output.ref_id] = arithmetic.output

        for relation_filter in ir.relation_filters:
            if relation_filter.input_ref.ref_id not in output_refs:
                errors.append(Diagnostic(
                    "error", "relation_filter_input_unbound",
                    f"关系过滤 {relation_filter.filter_id} 的输入未绑定",
                ))
            predicate_refs = _predicate_output_refs(relation_filter.predicate)
            if not predicate_refs or any(item not in output_refs for item in predicate_refs):
                errors.append(Diagnostic(
                    "error", "relation_filter_predicate_unbound",
                    f"关系过滤 {relation_filter.filter_id} 的谓词未绑定到计算输出",
                ))
            if relation_filter.output.shape not in {"relation", "event_set"}:
                errors.append(Diagnostic(
                    "error", "relation_filter_output_mismatch",
                    f"关系过滤 {relation_filter.filter_id} 必须输出 relation 或 event_set",
                ))
            output_refs[relation_filter.output.ref_id] = relation_filter.output

        for comparison in ir.comparisons:
            left = _infer_expression(comparison.left, output_refs, catalog, errors)
            right = _infer_expression(comparison.right, output_refs, catalog, errors)
            if comparison.operator not in {"gt", "gte", "lt", "lte", "eq", "ne"}:
                errors.append(Diagnostic("error", "comparison_unsupported",
                                         f"不支持比较操作：{comparison.operator}"))
            if left != right:
                errors.append(Diagnostic("error", "comparison_unit_mismatch",
                                         "比较两侧类型或单位不兼容"))
            output_refs[comparison.output.ref_id] = comparison.output

        for conversion in ir.conversions:
            inferred = _infer_expression(conversion.input, output_refs, catalog, errors)
            valid_conversion = (
                inferred[0] == "number" and conversion.output.result_type == "number"
            ) or (
                inferred[0] == "datetime" and conversion.output.result_type == "datetime"
                and conversion.target_unit.upper() == "UTC"
            )
            if not valid_conversion:
                errors.append(Diagnostic(
                    "error", "conversion_type_mismatch",
                    f"换算 {conversion.conversion_id} 只能处理数值输入",
                ))
            if not conversion.target_unit:
                errors.append(Diagnostic(
                    "error", "conversion_target_unbound",
                    f"换算 {conversion.conversion_id} 缺少目标单位",
                ))
            if conversion.missing_data_policy not in {"reject", "skip", "null"}:
                errors.append(Diagnostic(
                    "error", "conversion_missing_policy_invalid",
                    f"换算 {conversion.conversion_id} 的缺失策略无效",
                ))
            output_refs[conversion.output.ref_id] = conversion.output

        for fold in ir.ordered_folds:
            transition = fold.transition
            if not transition.partition_by:
                errors.append(Diagnostic(
                    "error", "state_partition_unbound", f"状态递推 {fold.fold_id} 缺少分区字段",
                ))
            if not transition.order_by.canonical_id and transition.order_by.ref_kind != "hypothesis":
                errors.append(Diagnostic(
                    "error", "state_order_unbound", f"状态递推 {fold.fold_id} 缺少排序字段",
                ))
            if transition.reset_condition is None and transition.reset_expression is not None:
                errors.append(Diagnostic(
                    "error", "state_reset_incomplete", f"状态递推 {fold.fold_id} 缺少重置条件",
                ))
            if transition.reset_condition is not None and transition.reset_expression is None:
                errors.append(Diagnostic(
                    "error", "state_reset_incomplete", f"状态递推 {fold.fold_id} 缺少重置表达式",
                ))
            if transition.missing_data_policy not in {"break", "carry", "reject"}:
                errors.append(Diagnostic(
                    "error", "state_missing_policy_invalid", f"状态递推 {fold.fold_id} 缺失策略无效",
                ))
            order_field = catalog.field(transition.order_by.canonical_id or "")
            if order_field and order_field.data_type not in {"date", "datetime", "integer", "long", "float", "double", "number", "string"}:
                errors.append(Diagnostic(
                    "error", "state_order_not_sortable", f"状态递推 {fold.fold_id} 的排序字段不可排序",
                ))
            initial = _infer_expression(transition.initial_state, output_refs, catalog, errors)
            state_outputs = dict(output_refs)
            state_outputs["previous_state"] = ExpressionNode(
                kind="literal", result_type=initial[0], unit=initial[1],
            )
            inferred_transition = _infer_expression(
                transition.transition_expression, state_outputs, catalog, errors,
            )
            if inferred_transition != initial:
                errors.append(Diagnostic(
                    "error", "state_transition_type_mismatch",
                    f"状态递推 {fold.fold_id} 的初始值与转移表达式类型或单位不兼容",
                ))
            if _references_previous_state(transition.initial_state):
                errors.append(Diagnostic(
                    "error", "state_previous_state_scope_invalid", "previous_state 只能出现在转移表达式中",
                ))
            if transition.reset_expression:
                reset = _infer_expression(transition.reset_expression, output_refs, catalog, errors)
                if reset != initial:
                    errors.append(Diagnostic(
                        "error", "state_reset_type_mismatch",
                        f"状态递推 {fold.fold_id} 的重置表达式类型或单位不兼容",
                    ))
                if _references_previous_state(transition.reset_expression):
                    errors.append(Diagnostic(
                        "error", "state_previous_state_scope_invalid", "previous_state 不能出现在重置表达式中",
                    ))
            if fold.output.grain != "partition_order":
                errors.append(Diagnostic(
                    "error", "state_output_grain_mismatch", f"状态递推 {fold.fold_id} 输出粒度必须为 partition_order",
                ))
            if fold.post_condition and not _post_condition_uses_fold_output(fold):
                errors.append(Diagnostic(
                    "error", "state_post_condition_scope_invalid", "状态递推后的条件只能引用当前扫描输出",
                ))
            output_refs[fold.output.ref_id] = fold.output

        for sequence in ir.sequences:
            if len(sequence.steps) < 2:
                errors.append(Diagnostic(
                    "error", "sequence_steps_incomplete", f"有序序列 {sequence.sequence_id} 至少需要两个步骤",
                ))
            if not sequence.partition_by:
                errors.append(Diagnostic(
                    "error", "sequence_partition_unbound", f"有序序列 {sequence.sequence_id} 缺少分区字段",
                ))
            if sequence.order_by is None:
                errors.append(Diagnostic(
                    "error", "sequence_order_unbound", f"有序序列 {sequence.sequence_id} 缺少排序字段",
                ))
            elif sequence.order_by.canonical_id:
                order_field = catalog.field(sequence.order_by.canonical_id)
                if order_field and order_field.data_type not in {"date", "datetime", "integer", "long", "float", "double", "number", "string"}:
                    errors.append(Diagnostic(
                        "error", "sequence_order_not_sortable", f"有序序列 {sequence.sequence_id} 的排序字段不可排序",
                    ))
            if sequence.max_duration and sequence.max_duration.normalized_seconds <= 0:
                errors.append(Diagnostic(
                    "error", "sequence_duration_invalid", f"有序序列 {sequence.sequence_id} 的时限无效",
                ))
            if sequence.missing_data_policy not in {"reject", "break", "skip"}:
                errors.append(Diagnostic(
                    "error", "sequence_missing_policy_invalid", f"有序序列 {sequence.sequence_id} 的缺失策略无效",
                ))
            output_refs[sequence.output.ref_id] = sequence.output

        trading_day_durations = [
            item.threshold for item in ir.events
            if item.threshold and item.threshold.unit in {"交易日", "个交易日"}
        ] + [
            item.max_duration for item in ir.sequences
            if item.max_duration and item.max_duration.unit in {"交易日", "个交易日"}
        ] + [
            item.post_duration for item in ir.ordered_folds
            if item.post_duration and item.post_duration.unit in {"交易日", "个交易日"}
        ]
        if trading_day_durations and (
            resolved_source is None or "market_calendar" not in resolved_source.capabilities
        ):
            errors.append(Diagnostic(
                "error", "trading_calendar_unbound",
                "交易日时长需要已绑定数据源声明 market_calendar 能力",
            ))

        periods = {
            (item.from_value or item.exact or "")[:7]
            for item in ir.temporal
            if item.from_value or item.exact
        }
        for calculation in ir.calculations:
            if calculation.calculation_type not in _CALCULATIONS or not _safe_expression(calculation.expression):
                unsupported = True
                errors.append(Diagnostic(
                    "error", "unsafe_calculation",
                    f"计算表达式不在安全白名单：{calculation.calculation_type}",
                ))
                continue
            if calculation.output is None:
                errors.append(Diagnostic(
                    "error", "calculation_output_unbound",
                    f"计算 {calculation.calculation_type} 缺少明确 OutputRef 输出",
                ))
            if not calculation.inputs:
                errors.append(Diagnostic(
                    "error", "formula_input_unbound",
                    f"计算 {calculation.calculation_type} 缺少明确 SymbolRef 或 OutputRef 输入",
                ))
            elif any(not _explicit_calculation_input(item) for item in calculation.inputs):
                errors.append(Diagnostic(
                    "error", "formula_input_invalid",
                    "Calculation 输入必须由明确 SymbolRef 或 OutputRef 组成",
                ))
            if calculation.calculation_type in {"correlation", "ratio", "difference"}:
                if len(calculation.inputs) < 2 or calculation.output is None:
                    errors.append(Diagnostic(
                        "error", "formula_input_unbound",
                        f"计算 {calculation.calculation_type} 必须有两个明确 OutputRef 输入及输出",
                    ))
                else:
                    inferred_inputs = [
                        _infer_expression(item, output_refs, catalog, errors)
                        for item in calculation.inputs
                    ]
                    if any(item.kind not in {"output_ref", "reference", "field"} for item in calculation.inputs):
                        errors.append(Diagnostic("error", "formula_input_invalid",
                                                 "Calculation 输入必须是明确 SymbolRef 或 OutputRef"))
                    if calculation.calculation_type == "correlation":
                        if any(item[0] not in _NUMERIC_TYPES for item in inferred_inputs) or calculation.output.unit != "ratio":
                            errors.append(Diagnostic("error", "calculation_output_mismatch",
                                                     "相关系数需要数值输入并输出无量纲 ratio"))
                    elif calculation.calculation_type == "ratio":
                        if len(set(inferred_inputs)) > 1 or calculation.output.unit != "ratio":
                            errors.append(Diagnostic("error", "calculation_output_mismatch",
                                                     "比值需要同类型同单位输入并输出 ratio"))
                    elif calculation.calculation_type == "difference":
                        if len(set(inferred_inputs)) > 1 or calculation.output.unit != inferred_inputs[0][1]:
                            errors.append(Diagnostic("error", "calculation_output_mismatch",
                                                     "差值需要同类型同单位输入并保留单位"))
            expression_ast = calculation.parameters.get("expression_ast")
            if expression_ast is not None:
                expression = _expression_from_dict(expression_ast)
                if expression is None:
                    errors.append(Diagnostic(
                        "error", "calculation_expression_invalid",
                        "原子计算 Patch 的表达式结构无效",
                    ))
                else:
                    _infer_expression(expression, output_refs, catalog, errors)
            if calculation.calculation_type in {"mom", "yoy"}:
                required = {str(calculation.parameters.get("left", "")),
                            str(calculation.parameters.get("right", ""))}
                if "" in required or not required.issubset(periods):
                    errors.append(Diagnostic(
                        "error", "formula_input_unbound",
                        f"公式输入未被时间窗口覆盖：{sorted(required - periods)}",
                    ))
            elif calculation.calculation_type == "volatility":
                input_field = str(calculation.parameters.get("input_field", ""))
                reference = str(calculation.parameters.get("filter_reference", ""))
                metric_id = str(calculation.parameters.get("derived_metric_id", ""))
                if not input_field or (
                    catalog.field(input_field) is None and not input_field.startswith("hyp:")
                ):
                    errors.append(Diagnostic(
                        "error", "formula_input_unbound", "波动率缺少有效输入字段",
                    ))
                elif input_field.startswith("hyp:"):
                    errors.append(Diagnostic(
                        "error", "formula_input_unbound", "波动率输入仍是未绑定 SchemaHypothesis",
                    ))
                if len(calculation.inputs) != 1 or not _calculation_input_matches_field(
                        calculation.inputs[0] if calculation.inputs else None, input_field):
                    errors.append(Diagnostic(
                        "error", "formula_input_unbound",
                        "波动率输入没有绑定到声明的 SymbolRef",
                    ))
                if reference and reference not in set_outputs and reference not in set_output_refs:
                    errors.append(Diagnostic(
                        "error", "formula_reference_unbound",
                        f"波动率筛选引用未绑定：{reference}",
                    ))
                if not metric_id or (
                    metric_id != "semantic.stddev" and catalog.derived_metric(metric_id) is None
                ):
                    errors.append(Diagnostic(
                        "error", "derived_metric_missing", "波动率派生指标未注册",
                    ))
            elif not calculation.parameters:
                errors.append(Diagnostic(
                    "error", "formula_input_unbound",
                    f"计算 {calculation.calculation_type} 缺少可绑定输入",
                ))
            if calculation.output is not None:
                output_refs[calculation.output.ref_id] = calculation.output

        for top_k in ir.top_k_operations:
            if top_k.input_ref.ref_id not in output_refs:
                errors.append(Diagnostic("error", "top_k_input_unbound",
                                         f"Top-K {top_k.top_k_id} 的输入未绑定"))
            if top_k.limit <= 0:
                errors.append(Diagnostic("error", "top_k_limit_invalid",
                                         f"Top-K {top_k.top_k_id} 的数量必须大于零"))
            inferred = _infer_expression(top_k.rank_by, output_refs, catalog, errors)
            if inferred[0] not in _NUMERIC_TYPES:
                errors.append(Diagnostic("error", "top_k_rank_invalid",
                                         f"Top-K {top_k.top_k_id} 的排序输入必须是数值"))
            if top_k.output.shape != "relation":
                errors.append(Diagnostic("error", "top_k_output_shape_mismatch",
                                         f"Top-K {top_k.top_k_id} 必须输出 relation"))
            output_refs[top_k.output.ref_id] = top_k.output

        for reference in ir.references:
            if reference.status != "resolved" or not reference.target_task_id:
                errors.append(Diagnostic(
                    "error", "reference_unresolved", f"引用未解析：{reference.raw}", reference.span,
                ))

        self._validate_output_dag(ir, errors)

        status = "valid"
        if errors:
            status = "unsupported" if unsupported else "needs_clarification"
        binding_error_codes = {
            "unresolved_symbol", "unknown_catalog_id", "source_unresolved",
            "source_missing", "multi_source_unsupported", "operator_type_mismatch",
            "value_type_mismatch", "unit_mismatch", "aggregation_type_mismatch",
            "formula_input_unbound", "formula_reference_unbound", "reference_unresolved",
            "conversion_target_unbound", "state_partition_unbound", "state_order_unbound",
            "sequence_partition_unbound", "sequence_order_unbound",
        }
        binding_complete = not ir.unresolved and not any(
            item.code in binding_error_codes for item in errors
        )
        reliability, dimensions = self._reliability_dimensions(
            ir, symbols, errors,
            uncovered_requirements=uncovered_requirements,
            conflicting_claims=conflicting_claims,
            source_demands_valid=source_demands_valid,
            binding_complete=binding_complete,
            executable=status == "valid",
        )
        ir.reliability = reliability
        understanding_complete = (
            not uncovered_requirements and not conflicting_claims and not open_ambiguities
            and not ir.unsatisfiable and source_demands_valid
        )
        ir.understanding_status = (
            "complete" if understanding_complete
            else "ambiguous" if ir.ambiguities or conflicting_claims else "partial"
        )
        ir.semantic_status = ir.understanding_status
        ir.binding_status = "bound" if binding_complete else (
            "partial" if symbols or ir.source_candidates else "unbound"
        )
        ir.capability_status = (
            "unsupported" if unsupported else "available" if binding_complete else "unbound"
        )
        ir.execution_status = "ready" if status == "valid" else "blocked"
        ir.diagnostics.extend(error for error in errors if error not in ir.diagnostics)
        return ValidationReport(
            status=status,
            executable=status == "valid",
            reliability=reliability,
            errors=errors,
            understanding_complete=understanding_complete,
            binding_complete=binding_complete,
            authorization_status=(
                "authorized_catalog" if ir.query.security_scope_digest else "not_evaluated"
            ),
            dimensions=dimensions,
        )

    @staticmethod
    def _validate_literature_contract(ir: UnderstandingIR, catalog: CatalogSnapshot) -> ValidationReport:
        errors: list[Diagnostic] = []
        unsupported = False
        blocking_unresolved = {
            "write_operation", "context_required", "reference_unresolved",
        }
        for item in ir.unresolved:
            if item.code in blocking_unresolved:
                errors.append(Diagnostic("error", item.code, item.message, item.span))
                unsupported = unsupported or item.code == "write_operation"
        binding = dict(ir.source_binding or {})
        error_code = str(binding.get("error_code") or "")
        if error_code:
            unsupported = error_code == "unsupported_capability"
            errors.append(Diagnostic(
                "error", error_code,
                "当前授权文献库不具备请求所需能力" if unsupported else "多个业务数据源会改变结果，必须明确范围",
            ))
        source_id = str(binding.get("source_id") or "")
        source = catalog.source(source_id) if source_id else None
        if not error_code and source is None:
            errors.append(Diagnostic("error", "catalog_configuration_error", "授权文献源未绑定到当前 Catalog"))
        elif source is not None and not source.read_only:
            unsupported = True
            errors.append(Diagnostic("error", "write_source_blocked", "文献任务只允许只读数据源"))
        task = str(ir.literature_contract.get("task") or "")
        if re.search(r"\bdoi\b", ir.query.normalized, re.I) and source is not None:
            has_doi = any(field.field_id.rsplit(".", 1)[-1].casefold() == "doi" for field in source.fields)
            if not has_doi:
                unsupported = True
                errors.append(Diagnostic("error", "unsupported_capability", "当前目录未声明 DOI 元数据能力"))
        status = "unsupported" if unsupported else "needs_clarification" if errors else "valid"
        executable = status == "valid"
        ir.understanding_status = "complete" if executable else "partial"
        ir.semantic_status = ir.understanding_status
        ir.binding_status = "bound" if source is not None else "unbound"
        ir.capability_status = "available" if executable else "unsupported" if unsupported else "unbound"
        ir.execution_status = "ready" if executable else "blocked"
        ir.reliability = 1.0 if executable else 0.0
        ir.diagnostics.extend(item for item in errors if item not in ir.diagnostics)
        return ValidationReport(
            status=status, executable=executable, reliability=ir.reliability,
            errors=errors, understanding_complete=executable,
            binding_complete=source is not None and not error_code,
            authorization_status="authorized_catalog" if ir.query.security_scope_digest else "not_evaluated",
            dimensions={
                "literature_contract": {"status": "accepted" if executable else "blocked", "task": task},
                "source_binding": {"status": "bound" if source is not None else "unbound", **binding},
            },
        )

    @staticmethod
    def _validate_source_demands(ir: UnderstandingIR, errors: list[Diagnostic]) -> bool:
        """Validate anchors, slot entailment, and Requirement consumption.

        The ledger is a source-of-truth contract.  It is insufficient for a
        comparison span to be textually valid: its declared field/value/unit/
        operator properties must each be backed by the correct independent
        anchor and the corresponding requirement must own that demand.
        """
        initial_errors = len(errors)
        seen = set()
        query = ir.query.normalized
        by_id = {item.demand_id: item for item in ir.source_demands}
        by_requirement = {item.requirement_id: item for item in ir.requirements}
        high_value_types = {
            "field", "quantity", "unit", "comparison", "comparison_operator", "boolean",
            "temporal", "action_value", "operator", "output",
        }
        for demand in ir.source_demands:
            if demand.demand_id in seen:
                errors.append(Diagnostic("error", "source_demand_duplicate",
                                         f"重复源需求锚点：{demand.demand_id}", demand.span))
                continue
            seen.add(demand.demand_id)
            if (demand.span.start < 0 or demand.span.end > len(query)
                    or query[demand.span.start:demand.span.end] != demand.text):
                errors.append(Diagnostic("error", "source_demand_anchor_invalid",
                                         "源需求锚点与原始查询不一致", demand.span))
            if demand.consumption_status not in {
                "mapped_to_requirement", "unresolved", "approved_noise",
            }:
                errors.append(Diagnostic("error", "source_demand_consumption_status_invalid",
                                         f"源需求状态非法：{demand.consumption_status}", demand.span))
            if demand.consumption_status == "approved_noise":
                if demand.demand_type in high_value_types:
                    errors.append(Diagnostic("error", "source_demand_noise_forbidden",
                                             f"高价值需求不得标记为噪声：{demand.text}", demand.span))
                elif not demand.approved_noise_reason:
                    errors.append(Diagnostic("error", "source_demand_noise_reason_missing",
                                             "批准噪声必须记录原因", demand.span))
            if demand.critical and demand.consumption_status != "mapped_to_requirement":
                errors.append(Diagnostic("error", "source_demand_unconsumed",
                                         f"高价值源需求未被 Requirement 消费：{demand.text}", demand.span))
            if demand.consumption_status == "mapped_to_requirement":
                if not demand.mapped_requirement_ids:
                    errors.append(Diagnostic("error", "source_demand_mapping_missing",
                                             "已消费源需求缺少 Requirement 映射", demand.span))
                for requirement_id in demand.mapped_requirement_ids:
                    requirement = by_requirement.get(requirement_id)
                    if requirement is None or demand.demand_id not in requirement.expected_attributes.get(
                        "source_demand_ids", []
                    ):
                        errors.append(Diagnostic("error", "source_demand_mapping_invalid",
                                                 f"源需求映射未被 Requirement 持有：{requirement_id}", demand.span))
            if demand.demand_type == "field":
                field_text = str(demand.attributes.get("field_text", ""))
                if not _field_text_from_span(field_text, demand.text):
                    errors.append(Diagnostic("error", "source_demand_field_anchor_mismatch",
                                             "字段属性没有由字段 Anchor 文本蕴含", demand.span))
            if demand.demand_type == "comparison":
                IRValidator._validate_comparison_slots(demand, by_id, errors)
        ids = seen
        for requirement in ir.requirements:
            for demand_id in requirement.expected_attributes.get("source_demand_ids", []):
                if demand_id not in ids:
                    errors.append(Diagnostic("error", "requirement_demand_missing",
                                             f"需求 {requirement.requirement_id} 引用了不存在的源需求" ,
                                             requirement.span))
        return len(errors) == initial_errors

    @staticmethod
    def _validate_comparison_slots(demand, by_id, errors: list[Diagnostic]) -> None:
        slots = {
            "field_anchor_id": {"field", "derived_metric"},
            "quantity_anchor_id": {"quantity"},
            "unit_anchor_id": {"unit"},
            "operator_anchor_id": {"comparison_operator"},
        }
        slot_nodes = {}
        for attr, expected_types in slots.items():
            identifier = getattr(demand, attr, "")
            if not identifier:
                # Unitless predicates are valid and have no source unit span.
                # Requiring a fabricated unit anchor would weaken provenance.
                if attr == "unit_anchor_id" and not demand.attributes.get("unit"):
                    continue
                # Duration thresholds such as "持续超过 8 分钟" constrain a
                # sequence operator rather than a business data field.  Their
                # field is supplied by the enclosing event condition, so no
                # fake field anchor may be manufactured from "持续".
                if (attr == "field_anchor_id"
                        and not demand.attributes.get("field_text")
                        and demand.attributes.get("unit") in {
                            "毫秒", "秒", "秒钟", "分钟", "分", "小时", "时", "天", "日",
                        }):
                    continue
                errors.append(Diagnostic("error", "source_demand_slot_anchor_missing",
                                         f"比较需求缺少 {attr}", demand.span))
                continue
            node = by_id.get(identifier)
            if node is None or node.demand_type not in expected_types:
                errors.append(Diagnostic("error", "source_demand_slot_anchor_invalid",
                                         f"比较需求 {attr} 未指向 {sorted(expected_types)} Anchor", demand.span))
                continue
            cross_clause_field = (
                attr == "field_anchor_id" and demand.attributes.get("inherited_field")
            )
            if ((node.clause_id != demand.clause_id and not cross_clause_field)
                    or identifier not in demand.dependencies):
                errors.append(Diagnostic("error", "source_demand_slot_dependency_invalid",
                                         f"比较需求未声明 {attr} 的同子句依赖", demand.span))
            slot_nodes[attr] = node
        field = slot_nodes.get("field_anchor_id")
        quantity = slot_nodes.get("quantity_anchor_id")
        unit = slot_nodes.get("unit_anchor_id")
        operator = slot_nodes.get("operator_anchor_id")
        if field and str(demand.attributes.get("field_text", "")) != str(field.attributes.get("field_text", "")):
            errors.append(Diagnostic("error", "source_demand_comparison_field_mismatch",
                                     "比较字段没有引用字段 Anchor 的文本", demand.span))
        if quantity and not _values_equal(demand.attributes.get("value"), quantity.attributes.get("value")):
            errors.append(Diagnostic("error", "source_demand_comparison_quantity_mismatch",
                                     "比较数值没有引用数值 Anchor", demand.span))
        if unit and str(demand.attributes.get("unit", "")) != str(unit.attributes.get("raw_unit", "")):
            errors.append(Diagnostic("error", "source_demand_comparison_unit_mismatch",
                                     "比较单位没有引用单位 Anchor", demand.span))
        if operator and str(demand.attributes.get("operator", "")) != str(operator.attributes.get("operator", "")):
            errors.append(Diagnostic("error", "source_demand_comparison_operator_mismatch",
                                     "比较操作符没有引用操作符 Anchor", demand.span))

    @staticmethod
    def _validate_query_schema_isolation(ir: UnderstandingIR, symbols: list[SymbolRef],
                                         catalog: CatalogSnapshot, errors: list[Diagnostic]) -> None:
        schema = ir.query_schema
        if schema is None:
            return
        explicit_fields = set(schema.field_ids)
        scoped_sources = set(schema.source_ids)
        for symbol in symbols:
            if symbol.ref_kind == "output" or symbol.status != "resolved" or not symbol.canonical_id:
                continue
            source_id = catalog.source_for_field(symbol.canonical_id)
            if symbol.canonical_id in explicit_fields:
                continue
            if scoped_sources and source_id in scoped_sources:
                continue
            errors.append(Diagnostic(
                "error", "query_schema_cross_source_binding",
                f"未指定数据源时禁止将 {symbol.raw_name} 直接绑定到 {symbol.canonical_id}", symbol.span,
            ))

    @staticmethod
    def _validate_output_dag(ir: UnderstandingIR, errors: list[Diagnostic]) -> None:
        """Validate OutputRef lineage and reject cyclic downstream plans."""
        # The relation scanned from the resolved source is the sole permitted
        # root.  It is a real input, not an accidentally-unbound OutputRef.
        graph: dict[str, list[str]] = {"source_relation.rows": []}

        def add(output, inputs=()):
            if output:
                graph[output.ref_id] = [item for item in inputs if item]

        for item in ir.events:
            add(item.output_ref)
        for item in ir.derived_projections:
            add(item.output, [item.input_ref.ref_id])
        for item in ir.set_operations:
            refs = [value.ref_id for value in item.input_refs] or [
                value for value in item.inputs if "." in value
            ]
            add(item.output_ref, refs)
        for item in ir.aggregates:
            add(item.aggregate.output, _expression_output_refs(item.aggregate.input) +
                ([item.scope_ref.ref_id] if item.scope_ref else []))
        for item in ir.group_by_operations:
            add(item.output, [item.input_ref.ref_id] + [
                ref for key in item.keys for ref in _expression_output_refs(key)
            ])
        for item in ir.window_aggregates:
            add(item.output, ([item.input_ref.ref_id] if item.input_ref else []) +
                _expression_output_refs(item.input))
        for item in ir.cumulative_counts:
            add(item.output)
        for item in ir.cumulative_durations:
            add(item.output, ([item.input_ref.ref_id] if item.input_ref else []) +
                _predicate_output_refs(item.condition))
        for item in ir.comparisons:
            add(item.output, _expression_output_refs(item.left) + _expression_output_refs(item.right))
        for item in ir.arithmetic:
            add(item.output, _expression_output_refs(item.expression))
        for item in ir.relation_filters:
            add(item.output, [item.input_ref.ref_id] + _predicate_output_refs(item.predicate))
        for item in ir.conversions:
            add(item.output, _expression_output_refs(item.input))
        for item in ir.calculations:
            if item.output:
                add(item.output, [ref for value in item.inputs
                                  for ref in _expression_output_refs(value)])
        for item in ir.top_k_operations:
            add(item.output, [item.input_ref.ref_id] + _expression_output_refs(item.rank_by))
        for item in ir.ordered_folds:
            add(item.output, _expression_output_refs(item.transition.initial_state) +
                _expression_output_refs(item.transition.transition_expression))
        for item in ir.sequences:
            add(item.output)

        for output, inputs in graph.items():
            for value in inputs:
                if value not in graph:
                    errors.append(Diagnostic("error", "output_ref_unbound",
                                             f"输出 {output} 引用了不存在的 {value}"))
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                errors.append(Diagnostic("error", "output_dag_cycle", f"OutputRef DAG 存在环：{node}"))
                return
            if node in visited:
                return
            visiting.add(node)
            for child in graph.get(node, []):
                if child in graph:
                    visit(child)
            visiting.remove(node)
            visited.add(node)

        for output in graph:
            visit(output)

    @staticmethod
    def _symbols(ir: UnderstandingIR) -> list[SymbolRef]:
        result = list(ir.projections)
        result.extend(item.field for item in ir.metrics)
        result.extend(item for item in ir.grouping)
        for aggregate in ir.aggregates:
            result.extend(_expression_symbols(aggregate.aggregate.input))
            for expression in aggregate.group_by:
                result.extend(_expression_symbols(expression))
        for group in ir.group_by_operations:
            for expression in group.keys:
                result.extend(_expression_symbols(expression))
        for window in ir.window_aggregates:
            result.extend(_expression_symbols(window.input))
        for count in ir.cumulative_counts:
            result.extend(node.field for node in walk_predicates(count.action) if node.field is not None)
        for duration in ir.cumulative_durations:
            result.extend(node.field for node in walk_predicates(duration.condition) if node.field is not None)
        for top_k in ir.top_k_operations:
            result.extend(_expression_symbols(top_k.rank_by))
        for comparison in ir.comparisons:
            result.extend(_expression_symbols(comparison.left))
            result.extend(_expression_symbols(comparison.right))
        for arithmetic in ir.arithmetic:
            result.extend(_expression_symbols(arithmetic.expression))
        for calculation in ir.calculations:
            for expression in calculation.inputs:
                result.extend(_expression_symbols(expression))
        for conversion in ir.conversions:
            result.extend(_expression_symbols(conversion.input))
        for fold in ir.ordered_folds:
            result.extend(fold.transition.partition_by)
            result.append(fold.transition.order_by)
            result.extend(_expression_symbols(fold.transition.initial_state))
            result.extend(_expression_symbols(fold.transition.transition_expression))
            if fold.transition.reset_expression:
                result.extend(_expression_symbols(fold.transition.reset_expression))
            if fold.transition.reset_condition:
                result.extend(node.field for node in walk_predicates(fold.transition.reset_condition)
                              if node.field is not None)
        result.extend(
            node.field for node in walk_predicates(ir.filters) if node.field is not None
        )
        for event in ir.events:
            result.extend(
                node.field for node in walk_predicates(event.condition) if node.field is not None
            )
            for node in walk_predicates(event.condition):
                if node.right_expression:
                    result.extend(_expression_symbols(node.right_expression))
            if event.sampling.order_by:
                result.append(event.sampling.order_by)
            result.extend(event.sampling.partition_by)
        for relation_filter in ir.relation_filters:
            result.extend(
                node.field for node in walk_predicates(relation_filter.predicate)
                if node.field is not None
            )
        for sequence in ir.sequences:
            result.extend(sequence.partition_by)
            if sequence.order_by:
                result.append(sequence.order_by)
            for step in sequence.steps:
                result.extend(node.field for node in walk_predicates(step) if node.field is not None)
                for node in walk_predicates(step):
                    if node.right_expression:
                        result.extend(_expression_symbols(node.right_expression))
        deduped = []
        seen = set()
        for symbol in result:
            key = (symbol.canonical_id, symbol.raw_name, symbol.span.start if symbol.span else None)
            if key not in seen:
                seen.add(key)
                deduped.append(symbol)
        return deduped

    @staticmethod
    def _source_ids(ir: UnderstandingIR, symbols: list[SymbolRef],
                    catalog: CatalogSnapshot) -> set[str]:
        source_ids = {
            item.identifier for item in ir.source_candidates
            if item.status == "resolved" and catalog.source(item.identifier)
        }
        source_ids.update(
            source_id for symbol in symbols
            for source_id in [catalog.source_for_field(symbol.canonical_id or "")]
            if source_id
        )
        return source_ids

    @staticmethod
    def _validate_value(field: FieldSpec, unit: str | None, value_type: str,
                        errors: list[Diagnostic], span) -> None:
        if field.data_type in _NUMERIC_TYPES and value_type not in {"number", "range", "list"}:
            errors.append(Diagnostic(
                "error", "value_type_mismatch", f"字段 {field.field_id} 需要数值条件", span,
            ))
        if not unit or not field.unit:
            return
        expected = _UNIT_FAMILIES.get(field.unit.lower(), field.unit.lower())
        actual = _UNIT_FAMILIES.get(unit.lower(), unit.lower())
        if expected != actual:
            errors.append(Diagnostic(
                "error", "unit_mismatch",
                f"字段 {field.field_id} 的单位 {field.unit} 与条件单位 {unit} 不兼容", span,
            ))

    @staticmethod
    def _reliability_dimensions(
            ir: UnderstandingIR, symbols: list[SymbolRef], errors: list[Diagnostic], *,
            uncovered_requirements, conflicting_claims, source_demands_valid: bool,
            binding_complete: bool, executable: bool,
    ) -> tuple[float, dict[str, dict[str, object]]]:
        """Expose diagnostic reliability factors instead of one opaque score."""
        resolved = sum(1 for item in symbols if item.status == "resolved" and item.canonical_id)
        symbol_ratio = resolved / max(1, len(symbols))
        evidenced = sum(1 for item in symbols if item.span and item.span.text)
        evidence_ratio = evidenced / max(1, len(symbols))
        coverage_total = len(ir.requirements)
        coverage_score = (
            (coverage_total - len(uncovered_requirements)) / coverage_total
            if coverage_total else 1.0
        )
        operator_requirements = [item for item in ir.requirements if (
            item.operator_family or item.requirement_type in {
                "aggregate", "calculation", "comparison", "group_by", "top_k",
                "sequence_event", "set_operation", "output",
            }
        )]
        coverage_by_id = {item.requirement_id: item.status for item in ir.coverage}
        operator_covered = sum(
            coverage_by_id.get(item.requirement_id) == "satisfied"
            for item in operator_requirements
        )
        operator_score = (
            operator_covered / len(operator_requirements)
            if operator_requirements else 1.0
        )
        semantic_error_codes = {
            "source_demand_unconsumed", "source_demand_noise_forbidden",
            "source_demand_anchor_invalid", "source_demand_slot_entailment_invalid",
            "requirement_uncovered", "claim_conflict", "unsatisfiable_constraints",
            "output_dag_cycle", "output_ref_unbound",
        }
        semantic_errors = sum(item.code in semantic_error_codes for item in errors)
        semantic_score = max(0.0, 1.0 - 0.2 * semantic_errors - 0.15 * len(conflicting_claims))
        if not source_demands_valid:
            semantic_score = min(semantic_score, 0.2)
        binding_score = symbol_ratio if symbols else (1.0 if binding_complete else 0.0)
        # Source evidence is a separate contribution: a fully bound catalog
        # cannot hide a field that lacks a textual anchor.
        binding_score = round(0.75 * binding_score + 0.25 * evidence_ratio, 3)
        dimensions: dict[str, dict[str, object]] = {
            "requirement_coverage": {
                "status": "pass" if coverage_score == 1.0 else "fail",
                "score": round(coverage_score, 3),
                "covered": coverage_total - len(uncovered_requirements),
                "total": coverage_total,
            },
            "semantic_consistency": {
                "status": "pass" if semantic_score == 1.0 else "fail",
                "score": round(semantic_score, 3),
                "semantic_errors": semantic_errors,
                "claim_conflicts": len(conflicting_claims),
                "source_demands_valid": source_demands_valid,
            },
            "schema_binding": {
                "status": "pass" if binding_complete else "fail",
                "score": binding_score,
                "resolved": resolved,
                "total": len(symbols),
                "evidence_ratio": round(evidence_ratio, 3),
            },
            "operator_completeness": {
                "status": "pass" if operator_score == 1.0 else "fail",
                "score": round(operator_score, 3),
                "covered": operator_covered,
                "total": len(operator_requirements),
            },
            "executability": {
                "status": "pass" if executable else "fail",
                "score": 1.0 if executable else 0.0,
            },
            # Engine fills this with a committed/rejected Patch outcome when
            # an LLM was actually involved.  This neutral value prevents a
            # no-LLM local compilation from being reported as a model failure.
            "llm_effect": {"status": "not_attempted", "score": 0.5},
        }
        score = (
            0.25 * coverage_score + 0.20 * semantic_score + 0.20 * binding_score
            + 0.15 * operator_score + 0.20 * (1.0 if executable else 0.0)
        )
        return round(max(0.0, min(0.98, score)), 3), dimensions


def _safe_expression(expression: str) -> bool:
    if not expression or "__" in expression:
        return False
    return all(character.isalnum() or character in "._,+-*/()% \t" for character in expression)


def _expression_symbols(expression: ExpressionNode) -> list[SymbolRef]:
    result = [expression.symbol] if expression.symbol else []
    for argument in expression.arguments:
        if isinstance(argument, ExpressionNode):
            result.extend(_expression_symbols(argument))
    return result


def _explicit_calculation_input(expression: ExpressionNode | None) -> bool:
    """Accept only expressions whose leaves are stable SymbolRef/OutputRef IDs."""
    if expression is None:
        return False
    if expression.kind == "field":
        return expression.symbol is not None and bool(
            expression.symbol.canonical_id or expression.symbol.candidates or expression.symbol.raw_name
        )
    if expression.kind in {"output_ref", "reference"}:
        return bool(expression.reference)
    if expression.kind in {"binary", "function"}:
        arguments = [item for item in expression.arguments if isinstance(item, ExpressionNode)]
        return bool(arguments) and any(
            _explicit_calculation_input(item) for item in arguments if item.kind != "literal"
        ) and all(
            item.kind == "literal" or _explicit_calculation_input(item) for item in arguments
        )
    return False


def _calculation_input_matches_field(expression: ExpressionNode | None, field_id: str) -> bool:
    if expression is None or expression.kind != "field" or expression.symbol is None or not field_id:
        return False
    symbol = expression.symbol
    identifiers = [symbol.canonical_id, symbol.raw_name]
    identifiers.extend(item.identifier for item in symbol.candidates)
    return field_id in {item for item in identifiers if item}


def _references_previous_state(expression: ExpressionNode) -> bool:
    return expression.reference == "previous_state" or any(
        _references_previous_state(item) for item in expression.arguments
        if isinstance(item, ExpressionNode)
    )


def _post_condition_uses_fold_output(fold) -> bool:
    allowed = fold.output.producer_id
    for node in walk_predicates(fold.post_condition):
        if node.kind == "boolean":
            continue
        if not node.field or node.field.ref_kind != "output":
            return False
        identifier = node.field.canonical_id or node.field.raw_name
        if not identifier.startswith(allowed + "."):
            return False
    return True


def _expression_from_dict(value: object) -> ExpressionNode | None:
    """Rehydrate a persisted atomic-Patch expression for local validation.

    ``CalculationSpec.expression`` remains a legacy display string, while
    atomic patches preserve their typed AST in ``expression_ast``.  Validation
    must use that typed form instead of trusting the rendered string.
    """
    if not isinstance(value, dict):
        return None
    raw_arguments = value.get("arguments", [])
    if not isinstance(raw_arguments, list):
        return None
    arguments = []
    for item in raw_arguments:
        node = _expression_from_dict(item)
        if node is None:
            return None
        arguments.append(node)
    raw_symbol = value.get("symbol")
    symbol = None
    if raw_symbol is not None:
        if not isinstance(raw_symbol, dict):
            return None
        symbol = SymbolRef(
            raw_name=str(raw_symbol.get("raw_name", "")),
            canonical_id=raw_symbol.get("canonical_id"),
            status=str(raw_symbol.get("status", "unresolved")),
            ref_kind=str(raw_symbol.get("ref_kind", "catalog")),
        )
    raw_literal = value.get("literal")
    literal = None
    if raw_literal is not None:
        if not isinstance(raw_literal, dict):
            return None
        literal = LiteralValue(
            raw_literal.get("value"), str(raw_literal.get("value_type", "unknown")),
            raw_literal.get("unit"), raw_literal.get("raw_unit"),
        )
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    return ExpressionNode(
        kind=kind,
        operator=str(value.get("operator", "")),
        arguments=arguments,
        symbol=symbol,
        literal=literal,
        reference=str(value.get("reference", "")),
        result_type=str(value.get("result_type", "unknown")),
        unit=value.get("unit"),
    )


def _expression_shape(expression: ExpressionNode, outputs: dict) -> str:
    if expression.kind in {"output_ref", "reference"}:
        output = outputs.get(expression.reference)
        return output.shape if output is not None else "scalar"
    return "scalar"


def _infer_expression(expression: ExpressionNode, outputs: dict, catalog: CatalogSnapshot,
                      errors: list[Diagnostic]) -> tuple[str, str | None]:
    if expression.kind == "field":
        field = catalog.field(expression.symbol.canonical_id or "") if expression.symbol else None
        inferred = (field.data_type if field else expression.result_type,
                    field.unit if field else expression.unit)
    elif expression.kind == "literal" and expression.literal:
        inferred = (expression.literal.value_type, expression.literal.unit)
    elif expression.kind in {"output_ref", "reference"}:
        output = outputs.get(expression.reference)
        if not output:
            errors.append(Diagnostic("error", "expression_reference_unbound",
                                     f"表达式引用未绑定：{expression.reference}"))
            inferred = ("unknown", None)
        else:
            inferred = (output.result_type, output.unit)
    elif expression.kind == "function":
        supported = {"moving_avg", "window_ratio", "stddev", "correlation", "max"}
        if expression.operator not in supported:
            errors.append(Diagnostic(
                "error", "expression_function_unsupported",
                f"表达式函数不受支持：{expression.operator}",
            ))
        arguments = [
            _infer_expression(argument, outputs, catalog, errors)
            for argument in expression.arguments if isinstance(argument, ExpressionNode)
        ]
        if expression.operator == "moving_avg":
            valid_width = len(arguments) == 2 and (
                (arguments[1][0] in _NUMERIC_TYPES and arguments[1][1] is None)
                or (arguments[1][0] == "duration"
                    and arguments[1][1] in {"millisecond", "second", "minute", "hour", "day"})
            )
            if (len(arguments) != 2 or arguments[0][0] not in _NUMERIC_TYPES
                    or not valid_width):
                errors.append(Diagnostic(
                    "error", "window_function_shape_invalid",
                    "moving_avg 需要数值输入及无量纲或时长窗口宽度",
                ))
            inferred = arguments[0] if arguments else ("unknown", None)
        elif expression.operator == "window_ratio":
            if (len(arguments) != 2 or arguments[0][0] not in _NUMERIC_TYPES
                    or arguments[0] != arguments[1]):
                errors.append(Diagnostic(
                    "error", "window_function_shape_invalid", "window_ratio 需要两个同类型同单位窗口输入",
                ))
            inferred = ("number", "ratio")
        elif expression.operator in {"stddev", "max"}:
            if len(arguments) != 1 or arguments[0][0] not in _NUMERIC_TYPES:
                errors.append(Diagnostic(
                    "error", "function_shape_invalid",
                    f"{expression.operator} 需要一个数值输入",
                ))
            inferred = arguments[0] if arguments else ("unknown", None)
        elif expression.operator == "correlation":
            if len(arguments) != 2 or any(item[0] not in _NUMERIC_TYPES for item in arguments):
                errors.append(Diagnostic(
                    "error", "function_shape_invalid", "correlation 需要两个数值输入",
                ))
            inferred = ("number", "ratio")
        else:
            inferred = (expression.result_type, expression.unit)
    elif expression.kind == "binary" and len(expression.arguments) == 2:
        left = _infer_expression(expression.arguments[0], outputs, catalog, errors)
        right = _infer_expression(expression.arguments[1], outputs, catalog, errors)
        if expression.operator in {"add", "sub"}:
            if left != right:
                errors.append(Diagnostic("error", "expression_unit_mismatch",
                                         "加减两侧类型或单位不兼容"))
            inferred = left
        elif expression.operator == "mul":
            if left[1] is None and left[0] == "number":
                inferred = right
            elif right[1] is None and right[0] == "number":
                inferred = left
            else:
                errors.append(Diagnostic("error", "compound_unit_unsupported",
                                         "M1 仅允许数值与无量纲标量相乘"))
                inferred = ("unknown", None)
        elif expression.operator == "div":
            if right[1] is None and right[0] == "number":
                inferred = left
            elif left == right:
                inferred = ("number", "ratio")
            else:
                errors.append(Diagnostic("error", "compound_unit_unsupported",
                                         "M1 不支持该复合单位除法"))
                inferred = ("unknown", None)
        else:
            errors.append(Diagnostic("error", "expression_operator_unsupported",
                                     f"表达式操作符不受支持：{expression.operator}"))
            inferred = ("unknown", None)
    else:
        errors.append(Diagnostic("error", "expression_shape_invalid", "表达式结构无效"))
        inferred = ("unknown", None)
    if (expression.result_type not in {"", "unknown"}
            and (expression.result_type, expression.unit) != inferred):
        errors.append(Diagnostic("error", "expression_declaration_mismatch",
                                 "表达式声明类型/单位与本地推导不一致"))
    return inferred
