"""Versioned, domain-neutral semantic operator registry and local enrichers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .merge import walk_predicates
from .field_roles import FieldPhraseParser
from .models import (
    AggregateSpec, AmbiguitySpec, ArithmeticSpec, CalculationSpec, CandidateRef,
    ComparisonSpec, ContentAnnotation, ConversionSpec, CumulativeCountSpec, CumulativeDurationSpec,
    DerivedMetricCall, DerivedProjectionSpec,
    DurationConstraint, EventSpec, ExpressionNode, LiteralValue, OperatorInvocation,
    GroupBySpec, OrderedFoldSpec, OutputRef, PredicateNode, Provenance, SamplingPolicy,
    RelationFilterSpec, SchemaHypothesis, ScopedAggregateSpec, SequenceSpec, SetOperationSpec, TopKSpec,
    SourceSpan, StateTransitionSpec, SymbolRef, UnderstandingIR,
    UnresolvedItem, UnsatisfiableSpec, WindowAggregateSpec, WindowSpec,
)


@dataclass(frozen=True, slots=True)
class OperatorDescriptor:
    operator_id: str
    family: str
    input_shapes: tuple[str, ...]
    output_shape: str
    required_attributes: tuple[str, ...] = ()
    capability: str = "read.filter"
    read_only: bool = True


class OperatorRegistry:
    VERSION = "operator-registry-v1"

    def __init__(self):
        values = (
            OperatorDescriptor("FILTER", "predicate", ("relation",), "relation"),
            OperatorDescriptor("AND", "boolean", ("boolean", "boolean"), "boolean"),
            OperatorDescriptor("OR", "boolean", ("boolean", "boolean"), "boolean"),
            OperatorDescriptor("NOT", "boolean", ("boolean",), "boolean"),
            OperatorDescriptor("CONSECUTIVE", "event", ("relation", "predicate"), "event_set", capability="timeseries.consecutive"),
            OperatorDescriptor("CUMULATIVE_DURATION", "aggregate", ("relation", "predicate"), "scalar", capability="timeseries.cumulative_duration"),
            OperatorDescriptor("CUMULATIVE_COUNT", "aggregate", ("relation", "predicate"), "scalar", capability="read.aggregate"),
            OperatorDescriptor("AGGREGATE", "aggregate", ("relation",), "scalar", capability="read.aggregate"),
            OperatorDescriptor("QUANTILE", "aggregate", ("relation",), "scalar", capability="read.aggregate"),
            OperatorDescriptor("SCOPED_AGGREGATE", "aggregate", ("relation",), "scalar", capability="read.aggregate"),
            OperatorDescriptor("INTERSECTION", "set", ("relation", "relation"), "relation", capability="set.intersection"),
            OperatorDescriptor("UNION", "set", ("relation", "relation"), "relation", capability="set.union"),
            OperatorDescriptor("DIFFERENCE", "set", ("relation", "relation"), "relation", capability="set.difference"),
            OperatorDescriptor("PROJECT_DATE", "projection", ("event_set",), "relation"),
            OperatorDescriptor("RELATION_FILTER", "filter", ("relation", "boolean"), "relation"),
            OperatorDescriptor("GROUP_BY", "group", ("relation",), "relation", capability="read.aggregate"),
            OperatorDescriptor("TOP_K", "ranking", ("relation",), "relation", capability="read.aggregate"),
            OperatorDescriptor("WINDOW", "window", ("relation",), "relation", capability="read.filter"),
            OperatorDescriptor("ARITHMETIC", "arithmetic", ("number", "number"), "number", capability="math.expression"),
            OperatorDescriptor("COMPARE", "comparison", ("scalar", "scalar"), "boolean", capability="math.expression"),
            OperatorDescriptor("CONVERT_UNIT", "conversion", ("number",), "number", capability="unit.convert"),
            OperatorDescriptor("CONVERT_CURRENCY", "conversion", ("number",), "number", capability="unit.convert"),
            OperatorDescriptor("ORDERED_FOLD", "state", ("relation",), "relation", ("partition_by", "order_by", "initial_state", "transition"), "state.ordered_fold"),
            OperatorDescriptor("RESETTABLE_SCAN", "state", ("relation",), "relation", ("partition_by", "order_by", "initial_state", "transition", "reset_condition"), "state.resettable_scan"),
            OperatorDescriptor("SEQUENCE", "sequence", ("relation",), "event_set", ("steps", "partition_by", "order_by"), "state.sequence"),
            OperatorDescriptor("TURN_INHERIT", "turn", ("context",), "context", capability="context.inherit"),
            OperatorDescriptor("TURN_REPLACE", "turn", ("context",), "context", capability="context.replace"),
            OperatorDescriptor("TURN_CANCEL", "turn", ("context",), "context", capability="context.cancel"),
            OperatorDescriptor("UNSATISFIABLE", "validation", ("predicate",), "diagnostic"),
        )
        self._items = {item.operator_id: item for item in values}

    def get(self, operator_id: str) -> OperatorDescriptor | None:
        return self._items.get(operator_id)

    def all(self) -> tuple[OperatorDescriptor, ...]:
        return tuple(self._items.values())


class GenericOperatorEnricher:
    """Add semantics derivable without domain-specific field names."""

    def __init__(self, registry: OperatorRegistry | None = None):
        self.registry = registry or OperatorRegistry()

    def enrich(self, ir: UnderstandingIR, catalog=None) -> None:
        self._temporal_ambiguities(ir)
        self._timezone_ambiguities(ir)
        self._timezone_conversion(ir)
        self._content_instruction_evidence(ir)
        self._formula_ambiguities(ir)
        self._implicit_categorical_predicates(ir, catalog)
        self._null_predicates(ir)
        self._dynamic_threshold_events(ir, catalog)
        self._compound_condition_events(ir, catalog)
        self._ratio_recovery_event(ir, catalog)
        self._window_qualified_events(ir, catalog)
        self._requirement_sequence_events(ir, catalog)
        self._cumulative_aggregates(ir, catalog)
        self._requirement_cumulative_aggregates(ir, catalog)
        self._sequence(ir)
        self._grouping_topk(ir)
        # Set outputs are scope inputs for cumulative-duration and downstream
        # analytics, so materialize their typed refs before lowering the DAG.
        self._generic_set_operations(ir)
        self._analytic_dag(ir, catalog)
        self._generic_set_operations(ir)
        self._bind_duration_set_scopes(ir)
        self._generic_calculations(ir)
        self._cleanup_structural_unresolved(ir)
        self._unsatisfiable(ir)
        self._ordered_fold(ir)
        self._invocations(ir)

    def refresh(self, ir: UnderstandingIR) -> None:
        """Rebuild semantics derived from nodes added by context or an LLM Patch."""
        self._generic_set_operations(ir)
        self._analytic_dag(ir, None)
        self._scoped_duration_ratios(ir)
        self._generic_calculations(ir)
        self._cleanup_structural_unresolved(ir)
        self._invocations(ir)

    def finalize_coverage_lineage(self, ir: UnderstandingIR) -> bool:
        """Bind downstream operators to producers proven by CoverageSpec.

        Operator extraction runs before coverage matching, so extraction-time
        span proximity may only create a provisional scope.  This pass uses
        the independently matched Requirement -> producer relation and is the
        only place allowed to make the final duration/set lineage decision.
        """
        coverage = {item.requirement_id: item for item in ir.coverage}
        set_by_node = {
            item.output_name: item for item in ir.set_operations
            if item.output_ref is not None
        }
        duration_by_node = {
            item.duration_id: item for item in ir.cumulative_durations
        }
        aggregate_by_node = {
            item.aggregate.aggregate_id: item for item in ir.aggregates
        }
        producer_outputs = {
            **{item.output_name: item.output_ref for item in ir.set_operations},
            **{item.group_id: item.output for item in ir.group_by_operations},
            **{item.event_id: item.output_ref for item in ir.events},
            **{item.projection_id: item.output for item in ir.derived_projections},
            **{item.duration_id: item.output for item in ir.cumulative_durations},
            **{item.aggregate.aggregate_id: item.aggregate.output for item in ir.aggregates},
            **{item.window_id: item.output for item in ir.window_aggregates},
        }
        changed = False
        for event in ir.events:
            condition = event.condition
            if not condition.right_expression or condition.right_expression.operator != "dynamic_threshold":
                continue
            event_span = next((item.span for item in event.provenance if item.span), None)
            candidates = [
                item for item in ir.window_aggregates
                if event_span is not None and item.provenance and item.provenance[0].span
                and event_span.start <= item.provenance[0].span.start <= event_span.end
            ]
            if len(candidates) != 1:
                continue
            window = candidates[0]
            condition.field = window.input.symbol
            condition.right_expression = ExpressionNode(
                "output_ref", reference=window.output.ref_id,
                result_type="number", unit=window.output.unit,
                provenance=list(window.provenance),
            )
            changed = True
        for requirement in ir.requirements:
            if requirement.requirement_type != "cumulative_duration":
                continue
            duration_coverage = coverage.get(requirement.requirement_id)
            durations = [
                duration_by_node[node_id]
                for node_id in (duration_coverage.covered_by if duration_coverage else [])
                if node_id in duration_by_node
            ]
            if not durations:
                # Requirement-lowered duration IDs are deterministic.  This
                # bootstraps the fixed point when the node cannot be covered
                # until its upstream Set OutputRef has first been attached.
                expected_id = "cumulative_duration_" + requirement.requirement_id.removeprefix("req_")
                if expected_id in duration_by_node:
                    durations = [duration_by_node[expected_id]]
            set_operations = []
            for dependency_id in requirement.dependencies:
                dependency = coverage.get(dependency_id)
                if dependency is None:
                    continue
                set_operations.extend(
                    set_by_node[node_id]
                    for node_id in dependency.covered_by
                    if node_id in set_by_node
                )
            if len(durations) != 1 or len(set_operations) != 1:
                continue
            duration = durations[0]
            output_ref = set_operations[0].output_ref
            if output_ref is None:
                continue
            expected_scope = WindowSpec(
                "calendar_day", reference=output_ref.ref_id, granularity="day",
            )
            if duration.input_ref != output_ref:
                duration.input_ref = output_ref
                changed = True
            if duration.scope != expected_scope:
                duration.scope = expected_scope
                changed = True
            if (duration.output.shape, duration.output.grain, duration.output.keys) != (
                "relation", "day", ["local_date"],
            ):
                duration.output.shape = "relation"
                duration.output.grain = "day"
                duration.output.keys = ["local_date"]
                changed = True
        for requirement in ir.requirements:
            if requirement.requirement_type != "aggregate":
                continue
            aggregate_id = "aggregate_" + requirement.requirement_id.removeprefix("req_")
            aggregate = aggregate_by_node.get(aggregate_id)
            if aggregate is None:
                continue
            dependency_refs = []
            for dependency_id in requirement.dependencies:
                dependency = coverage.get(dependency_id)
                if dependency is None or dependency.status != "satisfied":
                    continue
                dependency_refs.extend(
                    producer_outputs[node_id]
                    for node_id in dependency.covered_by
                    if producer_outputs.get(node_id) is not None
                )
            unique_refs = {item.ref_id: item for item in dependency_refs}
            scope_candidates = [
                item for item in unique_refs.values()
                if item.shape in {"relation", "event_set"}
            ]
            if len(scope_candidates) == 1 and aggregate.scope_ref != scope_candidates[0]:
                aggregate.scope_ref = scope_candidates[0]
                changed = True
            expected_scope = (requirement.expected_scope
                              or str(requirement.expected_attributes.get("scope", "")))
            if expected_scope and aggregate.scope != expected_scope:
                aggregate.scope = expected_scope
                changed = True
            expected_shape = (requirement.expected_output_shape
                              or str(requirement.expected_attributes.get(
                                  "expected_output_shape", ""
                              )))
            expected_grain = requirement.expected_output_grain or "scalar"
            if expected_shape and aggregate.aggregate.output.shape != expected_shape:
                aggregate.aggregate.output.shape = expected_shape
                changed = True
            if aggregate.aggregate.output.grain != expected_grain:
                aggregate.aggregate.output.grain = expected_grain
                changed = True
            grouped = aggregate.aggregate.output.shape == "relation" or expected_grain != "scalar"
            expected_keys = (list(aggregate.scope_ref.keys)
                             if grouped and aggregate.scope_ref else [])
            if aggregate.aggregate.output.keys != expected_keys:
                aggregate.aggregate.output.keys = expected_keys
                changed = True
        if changed:
            self._invocations(ir)
        return changed

    @staticmethod
    def _cleanup_structural_unresolved(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        expected_events = len(re.findall(
            r"连续(?!缺失)(?=[^，,。；;]{0,35}(?:超过|高于|低于|大于|小于))", query
        ))
        hypotheses = ir.schema_hypotheses
        kept = []
        for item in ir.unresolved:
            if item.code == "set_inputs_unbound" and ir.set_operations:
                continue
            if (item.code == "event_structure_incomplete" and ir.events
                    and (expected_events == 0 or len(ir.events) >= expected_events)):
                continue
            if (item.code == "event_structure_incomplete" and ir.cumulative_durations
                    and re.search(r"(?:累计|总|合计).*?(?:时长|耗时)", query)):
                continue
            if item.code == "formula_input_unbound" and item.span and any(
                arithmetic.expression.operator == "div"
                and len(arithmetic.expression.arguments) == 2
                and "ratio" in arithmetic.output.fields
                and any(
                    provenance.span
                    and provenance.span.start < item.span.end
                    and item.span.start < provenance.span.end
                    for provenance in arithmetic.provenance
                )
                for arithmetic in ir.arithmetic
            ):
                continue
            if item.code == "unknown_field" and item.span:
                text = re.sub(r"^(?:以及|并且|同时|同期|且)", "", item.span.text).strip()
                owners = [
                    hypothesis for hypothesis in hypotheses
                    if any(text.lower() == alias.lower() for alias in hypothesis.aliases)
                ]
                if len(owners) == 1:
                    continue
            kept.append(item)
        ir.unresolved = kept

    @staticmethod
    def _temporal_ambiguities(ir: UnderstandingIR) -> None:
        for match in re.finditer(r"(?<!\d)(\d{2})年", ir.query.normalized):
            span = SourceSpan(match.start(), match.end(), match.group(0))
            if any(item.kind == "temporal_century" and item.span == span for item in ir.ambiguities):
                continue
            ir.ambiguities.append(AmbiguitySpec(
                ambiguity_id=f"amb_year_{match.start()}", kind="temporal_century",
                message=f"两位年份 {match.group(1)}年 缺少世纪上下文",
                span=span, candidates=[f"19{match.group(1)}", f"20{match.group(1)}"],
            ))

    @staticmethod
    def _timezone_ambiguities(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        timezone_start = re.search(r"美国东部时间", query)
        dst = re.search(r"夏令时|DST", query, re.I)
        if not timezone_start or not dst or re.search(
            r"(?:UTC\s*offset|偏移)\s*(?:为|=)\s*[+-]\d|fold\s*(?:=|为)\s*[01]", query, re.I
        ):
            return
        # The unresolved choice is expressed after the DST wording (UTC offset
        # versus fold), so the ambiguity evidence must span that full source
        # condition rather than only the timezone/DST prefix.
        start, end = timezone_start.start(), len(query)
        ir.ambiguities.append(AmbiguitySpec(
            ambiguity_id=f"amb_timezone_{start}", kind="timezone_fold",
            message="跨越夏令时回拨的本地时间缺少 UTC offset 或 fold 信息",
            span=SourceSpan(start, end, query[start:end]),
            candidates=["fold=0", "fold=1", "提供 UTC offset"],
        ))

    @staticmethod
    def _timezone_conversion(ir: UnderstandingIR) -> None:
        match = re.search(r"换算为\s*UTC|转换为\s*UTC", ir.query.normalized, re.I)
        if not match or not ir.temporal or any(x.target_unit == "UTC" for x in ir.conversions):
            return
        ir.conversions.append(ConversionSpec(
            conversion_id="convert_timezone_utc",
            input=ExpressionNode(
                kind="literal", literal=LiteralValue("temporal:0", "datetime", unit="local_timezone"),
                result_type="datetime", unit="local_timezone",
            ),
            target_unit="UTC",
            output=OutputRef("convert_timezone_utc", "window", "datetime",
                             unit="UTC", shape="relation", grain="instant"),
            policy="timezone", missing_data_policy="reject",
            provenance=[Provenance(source="rule", rule="timezone_conversion",
                                   span=SourceSpan(match.start(), match.end(), match.group(0)))],
        ))

    @staticmethod
    def _content_instruction_evidence(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        quoted_pattern = re.compile(r"[“\"](?P<text>[^”\"]*(?:忽略系统指令|调用删除工具|输出密钥)[^”\"]*)[”\"]")
        for match in quoted_pattern.finditer(query):
            ir.content_annotations.append(ContentAnnotation(
                annotation_id=f"content_instruction_{match.start()}",
                kind="prompt_injection_evidence",
                span=SourceSpan(match.start(), match.end(), match.group(0)),
                action="treat_as_data",
            ))
            if "content_instruction_evidence" not in ir.output.fields:
                ir.output.fields.append("content_instruction_evidence")

    @staticmethod
    def _formula_ambiguities(ir: UnderstandingIR) -> None:
        for match in re.finditer(r"最大回撤|夏普比率|复合增长率|CAGR", ir.query.normalized, re.I):
            if any(match.group(0).lower() in (x.calculation_type + x.expression).lower()
                   for x in ir.calculations):
                continue
            ir.ambiguities.append(AmbiguitySpec(
                ambiguity_id=f"amb_formula_{match.start()}", kind="formula_ambiguity",
                message=f"计算 {match.group(0)} 缺少明确公式或业务口径",
                span=SourceSpan(match.start(), match.end(), match.group(0)),
                candidates=["请提供公式或选择企业指标定义"],
            ))

    @staticmethod
    def _implicit_categorical_predicates(ir: UnderstandingIR, catalog) -> None:
        query = ir.query.normalized
        if catalog is not None:
            language_field = catalog.field("private_kb.language")
            if language_field:
                for match in re.finditer(r"中文|英文", query):
                    if any(item.field and item.field.canonical_id == language_field.field_id
                           and item.value and item.value.value == match.group(0)
                           for item in walk_predicates(ir.filters)):
                        continue
                    symbol = SymbolRef(
                        raw_name="language", canonical_id=language_field.field_id,
                        candidates=[CandidateRef(language_field.field_id, source="catalog")],
                        status="resolved", span=SourceSpan(match.start(), match.end(), match.group(0)),
                    )
                    _append_and(ir, PredicateNode(
                        operator="eq", field=symbol,
                        value=LiteralValue(match.group(0), "string"),
                        provenance=[Provenance(source="rule", rule="implicit_language",
                                               span=SourceSpan(match.start(), match.end(), match.group(0)))],
                    ))
        for match in re.finditer(r"有代码|包含代码|代码地址", query):
            hypothesis_id = "hyp:" + hashlib.sha1(b"code_url").hexdigest()[:16]
            if not any(item.hypothesis_id == hypothesis_id for item in ir.schema_hypotheses):
                ir.schema_hypotheses.append(SchemaHypothesis(
                    hypothesis_id=hypothesis_id, raw_name="code_url",
                    normalized_name="code_url", declared_type="string",
                    description="问题中隐含的代码地址字段",
                    aliases=["代码", "代码地址"],
                    span=SourceSpan(match.start(), match.end(), match.group(0)), executable=False,
                ))
            symbol = SymbolRef(
                raw_name="code_url", candidates=[CandidateRef(hypothesis_id, source="query_schema")],
                status="candidate", ref_kind="hypothesis",
                span=SourceSpan(match.start(), match.end(), match.group(0)),
            )
            _append_and(ir, PredicateNode(
                operator="is_not_null", field=symbol,
                provenance=[Provenance(source="rule", rule="implicit_code_field",
                                       span=SourceSpan(match.start(), match.end(), match.group(0)))],
            ))

    @staticmethod
    def _null_predicates(ir: UnderstandingIR) -> None:
        for match in re.finditer(r"不为空|非空|不能为空|为空|是空值", ir.query.normalized):
            existing = [item for item in walk_predicates(ir.filters)
                        if item.operator in {"is_null", "is_not_null"}]
            if any(item.provenance and item.provenance[0].span
                   and item.provenance[0].span.start == match.start() for item in existing):
                continue
            symbols = [
                item.field for item in walk_predicates(ir.filters)
                if item.field and item.field.span and item.field.span.start <= match.start()
            ]
            if not symbols:
                continue
            field = max(symbols, key=lambda item: item.span.start if item.span else -1)
            predicate = PredicateNode(
                operator="is_not_null" if re.search(r"不|非", match.group(0)) else "is_null",
                field=field,
                provenance=[Provenance(
                    source="rule", rule="null_predicate",
                    span=SourceSpan(match.start(), match.end(), match.group(0)),
                )],
            )
            if ir.filters is None:
                ir.filters = predicate
            elif ir.filters.kind == "boolean" and ir.filters.operator == "and":
                ir.filters.children.append(predicate)
            else:
                ir.filters = PredicateNode(kind="boolean", operator="and", children=[ir.filters, predicate])

    @staticmethod
    def _generic_calculations(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        existing = {item.calculation_type for item in ir.calculations}
        patterns = (
            ("covariance", r"协方差", "covariance"),
            ("overlap_ratio", r"时间重合度|重合时长\s*/|重合占比", "overlap_ratio"),
            ("conversion_rate", r"转化率|转换率", "ratio"),
            ("ratio", r"比值|占比", "ratio"),
            ("median", r"中位数", "median"),
            ("quantile", r"P(?:90|95|99)|分位数", "quantile"),
        )
        for calculation_type, pattern, expression in patterns:
            match = re.search(pattern, query, re.I)
            if (not match or calculation_type in existing
                    or any(item.operator_family == calculation_type for item in ir.requirements)):
                continue
            span = SourceSpan(match.start(), match.end(), match.group(0))
            inputs = _generic_calculation_inputs(ir, calculation_type)
            # A generic keyword may be lowered only when an earlier,
            # explicitly requested DAG producer supplies its input.  Projected
            # fields are deliberately not harvested as a "nearest" formula
            # guess.
            if not inputs:
                if not any(item.code == "formula_input_unbound" and item.span == span
                           for item in ir.unresolved):
                    ir.unresolved.append(UnresolvedItem(
                        "formula_input_unbound", f"{calculation_type} 缺少原文明确的输入字段或输出引用", span,
                    ))
                continue
            calculation_id = f"generic_{calculation_type}_{match.start()}"
            unit = "ratio" if calculation_type in {"overlap_ratio", "conversion_rate"} else None
            ir.calculations.append(CalculationSpec(
                calculation_type=calculation_type, expression=expression,
                parameters={"operator": calculation_type},
                provenance=[Provenance(source="rule", rule="generic_named_calculation", span=span)],
                calculation_id=calculation_id, inputs=inputs,
                output=OutputRef(calculation_id, "value", "number", unit=unit,
                                 shape="scalar", grain="scalar", fields=[calculation_type]),
                scope="relation",
            ))
            existing.add(calculation_type)

    @staticmethod
    def _dynamic_threshold_events(ir: UnderstandingIR, catalog=None) -> None:
        query = ir.query.normalized
        duration = r"(?P<duration>\d+(?:\.\d+)?)\s*(?P<duration_unit>毫秒|秒钟?|分钟|分|小时|时|天|日)"
        relative = re.compile(
            r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\s*连续\s*"
            r"(?P<op>低于|小于|高于|超过|大于)\s*"
            r"(?P<baseline>额定[A-Za-z_\u4e00-\u9fff]{1,20}?)\s*(?P<percent>\d+(?:\.\d+)?)%\s*"
            r"(?:且)?\s*(?:持续)?\s*(?:超过|大于|不少于|至少)\s*" + duration
        )
        moving = re.compile(
            r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\s*连续\s*"
            r"(?P<op>低于|小于|高于|超过|大于)\s*前\s*"
            r"(?P<window>\d+(?:\.\d+)?)\s*(?P<window_unit>秒钟?|分钟|分|小时|时)\s*"
            r"平均(?:[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,20})?\s*"
            r"(?P<arith>[+-])\s*(?P<offset>\d+(?:\.\d+)?)\s*"
            r"(?P<offset_unit>℃|°C|MPa|kPa|Pa|%|V|m/s(?:²|2)?)?\s*"
            r"(?:且)?\s*(?:持续)?\s*(?:超过|大于|不少于|至少)\s*" + duration
        )
        for match in relative.finditer(query):
            if _overlaps_event(ir, match.start(), match.end()):
                continue
            field = _find_ir_symbol(ir, match.group("field"))
            if not field:
                continue
            baseline = _ensure_hypothesis_symbol(
                ir, match.group("baseline"), match.start("baseline"), match.end("baseline"),
                unit=_symbol_unit(field, catalog),
            )
            percent = float(match.group("percent")) / 100.0
            right = ExpressionNode(
                kind="binary", operator="mul", result_type="number",
                unit=_symbol_unit(field, catalog), arguments=[
                    ExpressionNode(kind="field", symbol=baseline, result_type="number",
                                   unit=_symbol_unit(field, catalog)),
                    ExpressionNode(kind="literal", literal=LiteralValue(percent, "number"),
                                   result_type="number"),
                ],
                provenance=[_prov(query, match.start("baseline"), match.end("percent"),
                                  "relative_threshold")],
            )
            condition = PredicateNode(
                operator=_comparison_operator(match.group("op")), field=field,
                right_expression=right,
                provenance=[_prov(query, match.start("field"), match.end("percent"),
                                  "dynamic_threshold_predicate")],
            )
            _append_consecutive_event(ir, condition, match, catalog, "relative_threshold")

        for match in moving.finditer(query):
            if _overlaps_event(ir, match.start(), match.end()):
                continue
            field = _find_ir_symbol(ir, match.group("field"))
            if not field:
                continue
            window_seconds = _seconds(float(match.group("window")), match.group("window_unit"))
            field_unit = _symbol_unit(field, catalog)
            offset_unit = _canonical_unit(match.group("offset_unit")) or field_unit
            moving_average = ExpressionNode(
                kind="function", operator="moving_avg", result_type="number", unit=field_unit,
                arguments=[
                    ExpressionNode(kind="field", symbol=field, result_type="number", unit=field_unit),
                    ExpressionNode(kind="literal",
                                   literal=LiteralValue(window_seconds, "duration", unit="second"),
                                   result_type="duration", unit="second"),
                ],
            )
            right = ExpressionNode(
                kind="binary", operator="add" if match.group("arith") == "+" else "sub",
                result_type="number", unit=field_unit, arguments=[
                    moving_average,
                    ExpressionNode(kind="literal",
                                   literal=LiteralValue(float(match.group("offset")), "number",
                                                        unit=offset_unit),
                                   result_type="number", unit=offset_unit),
                ],
                provenance=[_prov(query, match.start("window"), match.end("offset"),
                                  "moving_threshold")],
            )
            condition = PredicateNode(
                operator=_comparison_operator(match.group("op")), field=field,
                right_expression=right,
                provenance=[_prov(query, match.start("field"), match.end("offset"),
                                  "dynamic_threshold_predicate")],
            )
            _append_consecutive_event(ir, condition, match, catalog, "moving_threshold")

    @staticmethod
    def _compound_condition_events(ir: UnderstandingIR, catalog=None) -> None:
        query = ir.query.normalized
        pattern = re.compile(
            r"连续\s*(?P<duration>\d+(?:\.\d+)?)\s*"
            r"(?P<duration_unit>毫秒|秒钟?|分钟|分|小时|时|天|日)"
        )
        for match in pattern.finditer(query):
            if _overlaps_event(ir, match.start(), match.end()):
                continue
            clause_start = max(
                query.rfind(mark, 0, match.start()) + 1
                for mark in ("。", "；", ";", "，", ",")
            )
            candidates = []
            for predicate in walk_predicates(ir.filters):
                span = _predicate_span(predicate)
                if span and span.start >= clause_start and span.end <= match.start():
                    candidates.append(predicate)
            if len(candidates) < 2:
                continue
            condition = PredicateNode(kind="boolean", operator="and", children=candidates)
            _append_consecutive_event(ir, condition, match, catalog, "compound_condition")

    @staticmethod
    def _ratio_recovery_event(ir: UnderstandingIR, catalog=None) -> None:
        query = ir.query.normalized
        match = re.search(
            r"连续\s*(?P<duration>\d+(?:\.\d+)?)\s*"
            r"(?P<duration_unit>秒钟?|分钟|分|小时|时)\s*"
            r"(?P<label>[^，,。；;]{0,20}?比例)\s*(?:超过|高于|大于)\s*"
            r"(?P<value>\d+(?:\.\d+)?)%",
            query,
        )
        if not match or _overlaps_event(ir, match.start(), match.end()):
            return
        arithmetic_id = f"window_ratio_{len(ir.arithmetic) + 1}"
        ratio_output = OutputRef(arithmetic_id, "value", "number", unit="ratio")
        expression = ExpressionNode(
            kind="function", operator="window_ratio", result_type="number", unit="ratio",
            arguments=[
                ExpressionNode(kind="literal", literal=LiteralValue(match.group("label"), "string"),
                               result_type="string"),
                ExpressionNode(kind="literal",
                               literal=LiteralValue(
                                   _seconds(float(match.group("duration")), match.group("duration_unit")),
                                   "duration", unit="second"),
                               result_type="duration", unit="second"),
            ],
            provenance=[_prov(query, match.start(), match.end(), "window_ratio")],
        )
        ir.arithmetic.append(ArithmeticSpec(
            arithmetic_id=arithmetic_id, expression=expression, output=ratio_output,
            provenance=list(expression.provenance),
        ))
        ratio_symbol = SymbolRef(
            raw_name=match.group("label"), canonical_id=ratio_output.ref_id,
            status="resolved", ref_kind="output",
            span=SourceSpan(match.start("label"), match.end("label"), match.group("label")),
        )
        condition = PredicateNode(
            operator="gt", field=ratio_symbol,
            value=LiteralValue(float(match.group("value")) / 100.0, "number", unit="ratio"),
            provenance=[_prov(query, match.start("label"), match.end("value"),
                              "window_ratio_predicate")],
        )
        _append_consecutive_event(ir, condition, match, catalog, "ratio_recovery")

        recovery = re.search(
            r"随后\s*(?P<window>\d+(?:\.\d+)?)\s*(?P<window_unit>秒钟?|分钟|分|小时|时)"
            r"内恢复到\s*(?:低于|小于)\s*(?P<value>\d+(?:\.\d+)?)%",
            query[match.end():],
        )
        if recovery and not ir.sequences:
            offset = match.end()
            recovery_predicate = PredicateNode(
                operator="lt", field=ratio_symbol,
                value=LiteralValue(float(recovery.group("value")) / 100.0,
                                   "number", unit="ratio"),
                provenance=[_prov(query, offset + recovery.start(), offset + recovery.end(),
                                  "recovery_predicate")],
            )
            partition, order = _partition_and_order(ir)
            ir.sequences.append(SequenceSpec(
                sequence_id="sequence_recovery_1", steps=[condition, recovery_predicate],
                partition_by=partition, order_by=order,
                max_duration=DurationConstraint(
                    "lte", float(recovery.group("window")), recovery.group("window_unit"),
                    _seconds(float(recovery.group("window")), recovery.group("window_unit")),
                    SourceSpan(offset + recovery.start("window"), offset + recovery.end("window_unit"),
                               query[offset + recovery.start("window"):offset + recovery.end("window_unit")]),
                ),
                output=OutputRef("sequence_recovery_1", "events", "event_interval",
                                 shape="event_set", grain="interval"),
                provenance=[_prov(query, match.start(), offset + recovery.end(), "recovery_sequence")],
            ))

    @staticmethod
    def _window_qualified_events(ir: UnderstandingIR, catalog=None) -> None:
        """Compile interval-qualified consecutive events without domain names."""
        query = ir.query.normalized
        pattern = re.compile(
            r"连续\s*(?P<base_op>超过|高于|大于|不少于|至少)\s*"
            r"(?P<base_value>-?\d+(?:\.\d+)?)\s*"
            r"(?P<unit>W/m²|W/m2|mm/s²|mm/s2|mm/s|m/s²|m/s2|m/s|km/h|℃|°C|kPa|MPa|Pa|dB|W|kW|V|%)\s*"
            r"(?:\*{0,2}\s*)?(?:且|并且)(?:\s*\*{0,2})?\s*"
            r"(?P<qual_op>超过|高于|大于|不少于|至少)\s*"
            r"(?P<qual_value>-?\d+(?:\.\d+)?)\s*(?P=unit)\s*"
            r"的?时间占比\s*(?P<ratio_op>超过|高于|大于|不少于|至少)\s*"
            r"(?P<ratio_value>\d+(?:\.\d+)?)%",
            re.I,
        )
        for match in pattern.finditer(query):
            if _overlaps_event(ir, match.start(), match.end()):
                continue
            comparison = next((
                item for item in ir.source_demands
                if item.demand_type == "comparison"
                and item.span.start == match.start("base_op")
            ), None)
            field_text = str(comparison.attributes.get("field_text", "")) if comparison else ""
            field = _find_ir_symbol(ir, field_text)
            if field is None:
                continue

            span = SourceSpan(match.start(), match.end(), match.group(0))
            event_id = f"event_{len(ir.events) + 1}_consecutive"
            source_id = catalog.source_for_field(field.canonical_id or "") if catalog else None
            event_output = OutputRef(
                event_id, "intervals", "event_interval",
                shape="event_set", grain="interval", keys=["local_date"],
            )
            base_condition = PredicateNode(
                operator=_comparison_operator(match.group("base_op")), field=field,
                value=LiteralValue(
                    float(match.group("base_value")), "number",
                    _canonical_unit(match.group("unit")), match.group("unit"),
                ),
                provenance=[_prov(query, match.start("base_op"), match.end("unit"),
                                  "window_qualified_base_condition")],
            )
            event = EventSpec(
                event_id=event_id, condition=base_condition,
                derived_metric=DerivedMetricCall(
                    metric_id=(f"{source_id}.consecutive_duration" if source_id
                               else "semantic.consecutive_duration"),
                    arguments={"operator": "CONSECUTIVE", "emit_all_intervals": True},
                    result_type="duration", unit="second",
                    provenance=[Provenance(source="rule", rule="window_qualified_event", span=span)],
                ),
                threshold=None, window=_window(ir), sampling=_sampling_for(ir, field, catalog),
                group_by=["local_date"], output_name=event_output.ref_id,
                output_ref=event_output,
                provenance=[Provenance(source="rule", rule="window_qualified_event", span=span)],
            )
            ir.events.append(event)

            qualifier = PredicateNode(
                operator=_comparison_operator(match.group("qual_op")), field=field,
                value=LiteralValue(
                    float(match.group("qual_value")), "number",
                    _canonical_unit(match.group("unit")), match.group("unit"),
                ),
                provenance=[_prov(query, match.start("qual_op"), match.end("unit"),
                                  "window_qualifier_condition")],
            )
            duration_id = f"{event_id}_qualifying_duration"
            duration_output = OutputRef(
                duration_id, "duration", "number", unit="second",
                shape="relation", grain="interval", keys=["event_interval"],
                fields=["cumulative_duration"],
            )
            ir.cumulative_durations.append(CumulativeDurationSpec(
                duration_id=duration_id, condition=qualifier,
                scope=WindowSpec("event_interval", reference=event_output.ref_id,
                                 granularity="interval"),
                output=duration_output, input_ref=event_output,
                integration_method="step",
                provenance=[Provenance(source="rule", rule="window_qualifier_duration", span=span)],
            ))

            projection_id = f"{event_id}_interval_duration"
            interval_output = OutputRef(
                projection_id, "duration", "number", unit="second",
                shape="relation", grain="interval", keys=["event_interval"],
                fields=["interval_duration"],
            )
            ir.derived_projections.append(DerivedProjectionSpec(
                projection_id=projection_id, function="interval_duration_seconds",
                input_ref=event_output, output=interval_output,
                provenance=[Provenance(source="rule", rule="event_interval_duration", span=span)],
            ))

            ratio_id = f"{event_id}_qualifying_ratio"
            ratio_output = OutputRef(
                ratio_id, "value", "number", unit="ratio",
                shape="relation", grain="interval", keys=["event_interval"], fields=["ratio"],
            )
            ratio_expression = ExpressionNode(
                "binary", operator="div", arguments=[
                    ExpressionNode("output_ref", reference=duration_output.ref_id,
                                   result_type="number", unit="second"),
                    ExpressionNode("output_ref", reference=interval_output.ref_id,
                                   result_type="number", unit="second"),
                ], result_type="number", unit="ratio",
                provenance=[_prov(query, match.start("qual_op"), match.end("ratio_value"),
                                  "window_qualifying_ratio")],
            )
            ir.arithmetic.append(ArithmeticSpec(
                ratio_id, ratio_expression, ratio_output,
                list(ratio_expression.provenance), scope="relation",
            ))

            ratio_span = SourceSpan(
                match.start("ratio_op"), match.end("ratio_value") + 1,
                query[match.start("ratio_op"):match.end("ratio_value") + 1],
            )
            ratio_predicate = PredicateNode(
                operator=_comparison_operator(match.group("ratio_op")),
                field=SymbolRef(
                    raw_name="time_ratio", canonical_id=ratio_output.ref_id,
                    status="resolved", ref_kind="output",
                    span=SourceSpan(match.start("ratio_op") - 4, match.start("ratio_op"), "时间占比"),
                ),
                value=LiteralValue(float(match.group("ratio_value")) / 100.0,
                                   "number", unit="ratio", raw_unit="%"),
                provenance=[Provenance(source="rule", rule="window_ratio_threshold", span=ratio_span)],
            )
            filter_id = f"{event_id}_qualified"
            ir.relation_filters.append(RelationFilterSpec(
                filter_id=filter_id, input_ref=event_output, predicate=ratio_predicate,
                output=OutputRef(
                    filter_id, "intervals", "event_interval",
                    shape="event_set", grain="interval", keys=["local_date"],
                ),
                provenance=[Provenance(source="rule", rule="window_qualified_filter", span=span)],
            ))

    @staticmethod
    def _requirement_sequence_events(ir: UnderstandingIR, catalog=None) -> None:
        """Lower sequence obligations missed by fixed-threshold regexes."""
        by_id = {item.requirement_id: item for item in ir.requirements}
        for requirement in ir.requirements:
            if requirement.requirement_type not in {"sequence_event", "event"} or requirement.span is None:
                continue
            if _overlaps_event(ir, requirement.span.start, requirement.span.end):
                continue
            duration_demands = [
                item for item in ir.source_demands
                if item.demand_type == "duration_threshold"
                and requirement.requirement_id in item.mapped_requirement_ids
                and item.span.start < requirement.span.end + 12
            ]
            if not duration_demands:
                continue
            duration = max(duration_demands, key=lambda item: item.span.start)
            predicate_requirement = next((
                by_id[item] for item in requirement.dependencies
                if item in by_id and by_id[item].requirement_type == "predicate"
            ), None)
            if predicate_requirement is not None:
                attrs = predicate_requirement.expected_attributes
                field = _find_ir_symbol(ir, str(attrs.get("field_text", "")))
                if field is None:
                    continue
                condition = PredicateNode(
                    operator=str(attrs.get("operator", "gt")), field=field,
                    value=LiteralValue(attrs.get("value"), "number", attrs.get("unit") or None),
                    provenance=[_requirement_provenance(
                        predicate_requirement, "requirement_sequence_predicate",
                    )],
                )
            else:
                role = FieldPhraseParser.field_before(
                    ir.query.normalized, requirement.span.start, window=32,
                )
                field = _find_ir_symbol(ir, role.text)
                if field is None:
                    continue
                condition = PredicateNode(
                    operator="gt" if not re.search(r"低于|小于", requirement.text) else "lt",
                    field=field,
                    right_expression=ExpressionNode(
                        kind="function", operator="dynamic_threshold",
                        arguments=[requirement.text], result_type="number",
                        provenance=[_requirement_provenance(
                            requirement, "dynamic_threshold_expression",
                        )],
                    ),
                    provenance=[_requirement_provenance(
                        requirement, "requirement_dynamic_predicate",
                    )],
                )
            value = float(duration.attributes.get("value", 0))
            unit = str(duration.attributes.get("unit", ""))
            event_id = f"event_{len(ir.events) + 1}_consecutive"
            ir.events.append(EventSpec(
                event_id=event_id, condition=condition,
                derived_metric=DerivedMetricCall(
                    "semantic.consecutive_duration", {"operator": "CONSECUTIVE"},
                    "duration", "second",
                    [_requirement_provenance(requirement, "requirement_sequence_event")],
                ),
                threshold=DurationConstraint(
                    str(duration.attributes.get("operator", "gt")), value, unit,
                    _seconds(value, unit), duration.span,
                ),
                window=_window(ir), sampling=_sampling_for(ir, field, catalog),
                group_by=["local_date"], output_name=f"{event_id}_intervals",
                output_ref=OutputRef(event_id, "intervals", "event_interval",
                                     shape="event_set", grain="interval", keys=["local_date"]),
                provenance=[_requirement_provenance(requirement, "requirement_sequence_event")],
            ))

    @staticmethod
    def _scoped_duration_ratios(ir: UnderstandingIR) -> None:
        """Compile condition-duration / local-day-duration on an explicit set scope."""
        set_ref = next((item.output_ref for item in ir.set_operations if item.output_ref), None)
        if set_ref is None:
            return
        query = ir.query.normalized
        pattern = re.compile(
            r"(?P<op>超过|高于|大于|不少于|至少)\s*"
            r"(?P<value>-?\d+(?:\.\d+)?)\s*"
            r"(?P<unit>W/m²|W/m2|mm/s²|mm/s2|mm/s|m/s²|m/s2|m/s|km/h|℃|°C|kPa|MPa|Pa|dB|W|kW|V|%)\s*"
            r"\*{0,2}\s*的?(?:总|累计)时长\s*\*{0,2}\s*占\s*"
            r"\*{0,2}\s*(?:当日|当天)总时长\s*\*{0,2}\s*的?比例",
            re.I,
        )
        for match in pattern.finditer(query):
            ratio_id = f"scoped_duration_ratio_{match.start()}"
            if any(item.arithmetic_id == ratio_id for item in ir.arithmetic):
                continue
            comparison = next((
                item for item in ir.source_demands
                if item.demand_type == "comparison"
                and item.span.start == match.start("op")
            ), None)
            field_text = str(comparison.attributes.get("field_text", "")) if comparison else ""
            field = _find_ir_symbol(ir, field_text)
            if field is None:
                continue
            span = SourceSpan(match.start(), match.end(), match.group(0))
            existing_duration = next((
                item for item in ir.cumulative_durations
                if item.scope.window_type == "calendar_day"
                if item.condition.field and _same_symbol(item.condition.field, field)
                and item.condition.value
                and float(item.condition.value.value) == float(match.group("value"))
            ), None)
            if existing_duration is None:
                duration_id = f"scoped_duration_{match.start()}"
                duration_output = OutputRef(
                    duration_id, "duration", "number", unit="second",
                    shape="relation", grain="day", keys=["local_date"],
                    fields=["cumulative_duration"],
                )
                existing_duration = CumulativeDurationSpec(
                    duration_id=duration_id,
                    condition=PredicateNode(
                        operator=_comparison_operator(match.group("op")), field=field,
                        value=LiteralValue(float(match.group("value")), "number",
                                           _canonical_unit(match.group("unit")), match.group("unit")),
                        provenance=[Provenance(source="rule", rule="scoped_duration_condition", span=span)],
                    ),
                    scope=WindowSpec("calendar_day", reference=set_ref.ref_id, granularity="day"),
                    output=duration_output, input_ref=set_ref, integration_method="step",
                    provenance=[Provenance(source="rule", rule="scoped_cumulative_duration", span=span)],
                )
                ir.cumulative_durations.append(existing_duration)
            else:
                existing_duration.scope = WindowSpec(
                    "calendar_day", reference=set_ref.ref_id, granularity="day",
                )
                existing_duration.input_ref = set_ref
                existing_duration.output.shape = "relation"
                existing_duration.output.grain = "day"
                existing_duration.output.keys = ["local_date"]
                ir.cumulative_durations[:] = [
                    item for item in ir.cumulative_durations
                    if item is existing_duration
                    or item.scope.window_type != "calendar_day"
                    or not item.condition.field
                    or not _same_symbol(item.condition.field, field)
                    or not item.condition.value
                    or float(item.condition.value.value) != float(match.group("value"))
                ]
            duration_output = existing_duration.output
            denominator_id = f"local_day_duration_{match.start()}"
            denominator_output = OutputRef(
                denominator_id, "duration", "number", unit="second",
                shape="relation", grain="day", keys=["local_date"],
                fields=["local_day_duration"],
            )
            ir.derived_projections.append(DerivedProjectionSpec(
                projection_id=denominator_id,
                function="local_calendar_day_duration_seconds",
                input_ref=set_ref, output=denominator_output,
                provenance=[Provenance(source="rule", rule="local_day_duration", span=span)],
            ))
            output = OutputRef(
                ratio_id, "value", "number", unit="ratio",
                shape="relation", grain="day", keys=["local_date"], fields=["ratio"],
            )
            expression = ExpressionNode(
                "binary", operator="div", arguments=[
                    ExpressionNode("output_ref", reference=duration_output.ref_id,
                                   result_type="number", unit="second"),
                    ExpressionNode("output_ref", reference=denominator_output.ref_id,
                                   result_type="number", unit="second"),
                ], result_type="number", unit="ratio",
                provenance=[Provenance(source="rule", rule="scoped_duration_ratio", span=span)],
            )
            ir.arithmetic.append(ArithmeticSpec(
                ratio_id, expression, output, list(expression.provenance), scope="relation",
            ))

    @staticmethod
    def _cumulative_aggregates(ir: UnderstandingIR, catalog=None) -> None:
        query = ir.query.normalized
        pattern = re.compile(
            r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\s*累计\s*"
            r"(?P<op>超过|高于|大于|不少于|至少)\s*(?P<value>\d+(?:\.\d+)?)"
            r"\s*(?P<unit>手|次|个|笔|元|美元|欧元)?"
        )
        for match in pattern.finditer(query):
            clause_tail = re.split(r"[。；;]", query[match.end():], maxsplit=1)[0]
            if re.search(r"(?:累计|合计|总)?时长", clause_tail):
                continue
            field = _find_ir_symbol(ir, match.group("field"))
            if not field:
                continue
            if any(item.aggregate.input.symbol and _same_symbol(item.aggregate.input.symbol, field)
                   for item in ir.aggregates):
                continue
            aggregate_id = f"aggregate_{len(ir.aggregates) + 1}_sum"
            field_unit = _symbol_unit(field, catalog) or _canonical_unit(match.group("unit"))
            output = OutputRef(aggregate_id, "value", "number", unit=field_unit)
            aggregate = AggregateSpec(
                aggregate_id=aggregate_id, function="sum",
                input=ExpressionNode(kind="field", symbol=field, result_type="number", unit=field_unit),
                output=output,
                provenance=[_prov(query, match.start("field"), match.end("field"),
                                  "cumulative_aggregate")],
            )
            group_symbols = [item for item in _all_symbols(ir)
                             if re.search(r"(?:^|_)id$|编号$", item.raw_name, re.I)]
            ir.aggregates.append(ScopedAggregateSpec(
                aggregate=aggregate, scope="filtered",
                group_by=[ExpressionNode(kind="field", symbol=item, result_type="string")
                          for item in group_symbols[:2]],
            ))
            comparison_id = f"comparison_{len(ir.comparisons) + 1}_aggregate"
            ir.comparisons.append(ComparisonSpec(
                comparison_id=comparison_id,
                left=ExpressionNode(kind="reference", reference=output.ref_id,
                                    result_type="number", unit=field_unit),
                operator=_comparison_operator(match.group("op")),
                right=ExpressionNode(
                    kind="literal", literal=LiteralValue(float(match.group("value")), "number",
                                                           unit=field_unit),
                    result_type="number", unit=field_unit,
                ),
                output=OutputRef(comparison_id, "matched", "boolean"),
                provenance=[_prov(query, match.start(), match.end(), "aggregate_comparison")],
            ))

    @staticmethod
    def _requirement_cumulative_aggregates(ir: UnderstandingIR, catalog=None) -> None:
        """Compile cumulative values from exact Requirement/Demand anchors."""
        demands = {item.demand_id: item for item in ir.source_demands}
        for requirement in ir.requirements:
            if requirement.requirement_type != "cumulative_aggregate":
                continue
            if any(item.aggregate.provenance and item.aggregate.provenance[0].span
                   and requirement.span
                   and item.aggregate.provenance[0].span.start <= requirement.span.start
                   and requirement.span.end <= item.aggregate.provenance[0].span.end
                   for item in ir.aggregates):
                continue
            owned = [demands[item] for item in requirement.expected_attributes.get(
                "source_demand_ids", []
            ) if item in demands]
            fields = [item for item in owned if item.demand_type == "field"]
            count = next((item for item in owned if item.demand_type == "count_threshold"), None)
            symbol = None
            if fields:
                field_text = str(fields[0].attributes.get("field_text", fields[0].text))
                symbol = _find_ir_symbol(ir, field_text)
            aggregate_id = "aggregate_" + requirement.requirement_id.rsplit("_", 1)[-1] + "_cumulative"
            function = "sum" if symbol is not None else "count" if count is not None else ""
            if not function:
                continue
            unit = _symbol_unit(symbol, catalog) if symbol is not None else "count"
            expression = (ExpressionNode("field", symbol=symbol, result_type="number", unit=unit)
                          if symbol is not None else
                          ExpressionNode("literal", literal=LiteralValue(1, "number", unit="count"),
                                         result_type="number", unit="count"))
            output = OutputRef(aggregate_id, "value", "number", unit=unit)
            ir.aggregates.append(ScopedAggregateSpec(
                AggregateSpec(
                    aggregate_id, function, expression, output,
                    [_requirement_provenance(requirement, "requirement_cumulative_value")],
                ),
                "filtered",
            ))

    @staticmethod
    def _bind_duration_set_scopes(ir: UnderstandingIR) -> None:
        """Rebind duration nodes after all deterministic Set producers exist."""
        requirements = {item.requirement_id: item for item in ir.requirements}
        set_requirements = {
            item.requirement_id: item for item in ir.requirements
            if item.requirement_type == "set_operation"
        }
        for duration in ir.cumulative_durations:
            suffix = duration.duration_id.rsplit("_", 1)[-1]
            requirement = next((
                item for item in ir.requirements
                if item.requirement_type == "cumulative_duration"
                and item.requirement_id.endswith(suffix)
            ), None)
            if requirement is None:
                continue
            dependency = next((set_requirements[item] for item in requirement.dependencies
                               if item in set_requirements), None)
            if dependency is None:
                continue
            operation = min(
                (item for item in ir.set_operations if item.output_ref),
                key=lambda item: abs(
                    (item.provenance[0].span.start if item.provenance and item.provenance[0].span else 0)
                    - (dependency.span.start if dependency.span else 0)
                ),
                default=None,
            )
            if operation is None:
                continue
            duration.input_ref = operation.output_ref
            duration.scope = WindowSpec(
                "calendar_day", reference=operation.output_ref.ref_id, granularity="day",
            )
            duration.output.shape = "relation"
            duration.output.grain = "day"
            duration.output.keys = ["local_date"]

    @staticmethod
    def _sequence(ir: UnderstandingIR) -> None:
        if ir.sequences:
            return
        query = ir.query.normalized
        match = re.search(
            r"(?P<steps>[A-Za-z_][A-Za-z0-9_]*\s*(?:→|->)\s*"
            r"[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:→|->)\s*[A-Za-z_][A-Za-z0-9_]*)+)",
            query,
        )
        if not match:
            return
        values = [item.strip() for item in re.split(r"→|->", match.group("steps"))]
        action = next((item for item in _all_symbols(ir)
                       if re.search(r"action|行为|动作", item.raw_name, re.I)), None)
        if not action:
            return
        steps = [PredicateNode(
            operator="eq", field=action, value=LiteralValue(value, "string"),
            provenance=[_prov(query, match.start("steps"), match.end("steps"), "sequence_step")],
        ) for value in values]
        partition, order = _partition_and_order(ir)
        duration_match = re.search(
            r"总耗时\s*(?:不超过|小于等于|至多)\s*(\d+(?:\.\d+)?)\s*"
            r"(秒钟?|分钟|分|小时|时)", query,
        )
        max_duration = None
        if duration_match:
            max_duration = DurationConstraint(
                "lte", float(duration_match.group(1)), duration_match.group(2),
                _seconds(float(duration_match.group(1)), duration_match.group(2)),
                SourceSpan(duration_match.start(), duration_match.end(), duration_match.group(0)),
            )
        repeated = [index for index, value in enumerate(values)
                    if re.search(rf"重复\s*{re.escape(value)}", query, re.I)]
        ir.sequences.append(SequenceSpec(
            sequence_id="sequence_1", steps=steps, partition_by=partition,
            order_by=order, max_duration=max_duration, allow_repeated_steps=repeated,
            output=OutputRef("sequence_1", "events", "event_interval",
                             shape="event_set", grain="interval"),
            provenance=[_prov(query, match.start(), match.end(), "ordered_sequence")],
        ))

    @staticmethod
    def _grouping_topk(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        group_match = re.search(
            r"按\s*(?P<label>[^，,。；;]{1,24}?)(?:[（(]\s*`(?P<field>[A-Za-z_][A-Za-z0-9_.]*)`\s*[）)])?\s*分组",
            query,
        )
        if group_match:
            raw = group_match.group("field") or group_match.group("label")
            symbol = _find_ir_symbol(ir, raw)
            if symbol and not any(_same_symbol(item, symbol) for item in ir.grouping):
                ir.grouping.append(symbol)
                identifier = _symbol_id(symbol)
                if identifier and identifier not in ir.output.group_by:
                    ir.output.group_by.append(identifier)
        top_match = re.search(r"(?:前|top)\s*(?P<limit>\d+)\s*个?", query, re.I)
        if top_match:
            ir.output.limit = int(top_match.group("limit"))

    @staticmethod
    def _analytic_dag(ir: UnderstandingIR, catalog=None) -> None:
        """Compile analytic Requirements into an explicit, field-neutral DAG.

        This is intentionally driven by RequirementIR rather than a set of
        domain keywords.  Unknown fields remain query-schema hypotheses, which
        preserves the graph for review while the binding validator blocks
        execution.  No "nearest field" fallback is used.
        """
        root = OutputRef("source_relation", "rows", "relation", shape="relation", grain="row")
        aggregate_outputs: dict[str, OutputRef] = {}
        groups: dict[str, GroupBySpec] = {}

        for requirement in ir.requirements:
            if requirement.requirement_type != "group_by":
                continue
            group_id = "group_" + requirement.requirement_id.rsplit("_", 1)[-1]
            existing = next((item for item in ir.group_by_operations if item.group_id == group_id), None)
            if existing:
                groups[requirement.requirement_id] = existing
                continue
            symbols = _requirement_symbols(ir, requirement, catalog)
            if not symbols:
                continue
            keys = [ExpressionNode("field", symbol=item, result_type="string", unit=_symbol_unit(item, catalog))
                    for item in symbols]
            output = OutputRef(group_id, "groups", "relation", shape="relation", grain="group",
                               keys=[item.raw_name for item in symbols],
                               fields=_requirement_output_fields(requirement))
            group = GroupBySpec(
                group_id=group_id, input_ref=root, keys=keys, output=output,
                provenance=[_requirement_provenance(requirement, "requirement_group_by")],
            )
            ir.group_by_operations.append(group)
            groups[requirement.requirement_id] = group
            for symbol in symbols:
                if not any(_same_symbol(symbol, item) for item in ir.grouping):
                    ir.grouping.append(symbol)
                identifier = _symbol_id(symbol)
                if identifier and identifier not in ir.output.group_by:
                    ir.output.group_by.append(identifier)

        for requirement in ir.requirements:
            if requirement.requirement_type != "aggregate":
                continue
            aggregate_id = "aggregate_" + requirement.requirement_id.rsplit("_", 1)[-1]
            existing = next((item for item in ir.aggregates if item.aggregate.aggregate_id == aggregate_id), None)
            if existing:
                aggregate_outputs[requirement.requirement_id] = existing.aggregate.output
                continue
            functions = requirement.expected_attributes.get("expected_functions", [])
            function = functions[0] if functions else {
                "quantile": "quantile", "window_aggregate": "avg",
            }.get(requirement.operator_family, "")
            if not function:
                continue
            product_max = re.search(r"乘积(?:的)?最大值", ir.query.normalized)
            if (function == "max" and product_max and requirement.span
                    and product_max.start() <= requirement.span.start < product_max.end()
                    and len(ir.aggregates) >= 2):
                # "A 与 B 的乘积的最大值" is max(A * B), not an aggregate
                # over a physical column named "乘积".  The typed arithmetic
                # DAG below owns this source span.
                continue
            symbols = _requirement_symbols(ir, requirement, catalog)
            if not symbols:
                continue
            symbol = symbols[0]
            unit = _symbol_unit(symbol, catalog)
            dependency_groups = [groups[value] for value in requirement.dependencies if value in groups]
            context_start = max(0, (requirement.span.start if requirement.span else 0) - 48)
            context_end = requirement.span.end if requirement.span else context_start
            context = ir.query.normalized[context_start:context_end]
            set_scope = next((
                item.output_ref for item in reversed(ir.set_operations)
                if item.output_ref and re.search(r"交集|并集|重合|这些(?:日期|时间段)", context)
            ), None)
            scope_ref = dependency_groups[-1].output if dependency_groups else set_scope
            declared_scope = str(requirement.expected_attributes.get("scope", ""))
            scope = declared_scope or ("relation" if scope_ref else "global")
            expected_shape = requirement.expected_output_shape or str(
                requirement.expected_attributes.get("expected_output_shape", "")
            )
            expected_grain = requirement.expected_output_grain or "scalar"
            shape = expected_shape or ("relation" if expected_grain != "scalar" else "scalar")
            grain = expected_grain
            grouped = shape == "relation" or grain != "scalar"
            output = OutputRef(aggregate_id, "value", "number", unit=unit, shape=shape, grain=grain,
                               keys=list(scope_ref.keys) if scope_ref and grouped else [],
                               fields=_requirement_output_fields(requirement))
            parameters = {}
            if function == "quantile":
                value = requirement.expected_attributes.get("quantile")
                parameters["quantile"] = (float(value) / 100.0) if value is not None else None
            aggregate = AggregateSpec(
                aggregate_id=aggregate_id, function=function,
                input=ExpressionNode("field", symbol=symbol, result_type="number", unit=unit),
                output=output, parameters=parameters,
                provenance=[_requirement_provenance(requirement, "requirement_aggregate")],
            )
            ir.aggregates.append(ScopedAggregateSpec(
                aggregate=aggregate, scope=scope, scope_ref=scope_ref,
                group_by=(list(dependency_groups[-1].keys)
                          if dependency_groups and grouped else []),
            ))
            aggregate_outputs[requirement.requirement_id] = output

        product_max = re.search(r"乘积(?:的)?最大值", ir.query.normalized)
        if product_max and not any(item.calculation_type == "maximum"
                                   for item in ir.calculations):
            upstream = [
                item for item in ir.aggregates
                if item.aggregate.provenance and item.aggregate.provenance[0].span
                and item.aggregate.provenance[0].span.end <= product_max.start()
            ]
            if len(upstream) >= 2:
                left, right = upstream[-2:]
                arithmetic_id = f"arithmetic_product_{product_max.start()}"
                product_output = OutputRef(
                    arithmetic_id, "value", "number",
                    unit=left.aggregate.output.unit or right.aggregate.output.unit,
                    shape="scalar", grain="scalar", fields=["product"],
                )
                expression = ExpressionNode(
                    "binary", operator="mul", arguments=[
                        ExpressionNode(
                            "output_ref", reference=left.aggregate.output.ref_id,
                            result_type="number", unit=left.aggregate.output.unit,
                        ),
                        ExpressionNode(
                            "output_ref", reference=right.aggregate.output.ref_id,
                            result_type="number", unit=right.aggregate.output.unit,
                        ),
                    ], result_type="number", unit=product_output.unit,
                    provenance=[_prov(ir.query.normalized, product_max.start(),
                                      product_max.end(), "aggregate_product")],
                )
                ir.arithmetic.append(ArithmeticSpec(
                    arithmetic_id, expression, product_output, list(expression.provenance),
                ))
                calculation_id = f"calculation_maximum_{product_max.start()}"
                ir.calculations.append(CalculationSpec(
                    calculation_type="maximum",
                    expression=f"max({product_output.ref_id})",
                    parameters={"operator": "max", "input_ref": product_output.ref_id},
                    provenance=[_prov(ir.query.normalized, product_max.start(),
                                      product_max.end(), "maximum_of_product")],
                    calculation_id=calculation_id,
                    inputs=[ExpressionNode(
                        "output_ref", reference=product_output.ref_id,
                        result_type="number", unit=product_output.unit,
                    )],
                    output=OutputRef(
                        calculation_id, "value", "number", unit=product_output.unit,
                        shape="scalar", grain="scalar", fields=["maximum"],
                    ),
                    scope="relation",
                ))

        for requirement in ir.requirements:
            if requirement.requirement_type != "cumulative_count":
                continue
            count_id = "cumulative_count_" + requirement.requirement_id.rsplit("_", 1)[-1]
            if any(item.count_id == count_id for item in ir.cumulative_counts):
                continue
            action_demand = next((item for item in ir.source_demands
                                  if item.demand_type == "action_value"
                                  and item.clause_id == requirement.clause_id), None)
            if action_demand is None:
                continue
            action_symbol = _ensure_hypothesis_symbol(
                ir, "action", action_demand.span.start, action_demand.span.end,
            )
            action = PredicateNode(
                operator="eq", field=action_symbol,
                value=LiteralValue(action_demand.attributes.get("value"), "string"),
                provenance=[Provenance(source="rule", rule="cumulative_count_action", span=action_demand.span)],
            )
            scope = WindowSpec("calendar_day", reference="当天" if "当天" in ir.query.normalized else "query_window",
                               granularity="day")
            ir.cumulative_counts.append(CumulativeCountSpec(
                count_id=count_id, action=action, scope=scope,
                output=OutputRef(count_id, "count", "number", unit="count", shape="scalar", grain="day",
                                 fields=_requirement_output_fields(requirement)),
                provenance=[_requirement_provenance(requirement, "requirement_cumulative_count")],
            ))

        for requirement in ir.requirements:
            if requirement.requirement_type != "cumulative_duration":
                continue
            duration_id = "cumulative_duration_" + requirement.requirement_id.rsplit("_", 1)[-1]
            if any(item.duration_id == duration_id for item in ir.cumulative_durations):
                continue
            # A duration is only safe to compile when the query states the
            # action/value being accumulated.  Without that, leave the
            # Requirement uncovered instead of creating a duration for a
            # nearby field or action.
            action_demand = next((item for item in ir.source_demands
                                  if item.demand_type == "action_value"
                                  and item.clause_id == requirement.clause_id), None)
            comparison_demand = next((item for item in ir.source_demands
                                      if item.demand_type == "comparison"
                                      and item.clause_id == requirement.clause_id), None)
            dependency_ids = set(requirement.dependencies)
            set_dependency = next((
                item for item in ir.requirements
                if item.requirement_id in dependency_ids
                and item.requirement_type == "set_operation"
            ), None)
            dependency_predicates = [
                item for item in ir.requirements
                if item.requirement_type == "predicate"
                and item.requirement_id in dependency_ids
                and item.expected_attributes.get("field_anchor_id")
            ]
            if comparison_demand is None and len(dependency_predicates) == 1:
                predicate_requirement = dependency_predicates[0]
                comparison_demand = next((
                    item for item in ir.source_demands
                    if item.demand_id in predicate_requirement.expected_attributes.get("source_demand_ids", [])
                    and item.demand_type == "comparison"
                ), None)
                if comparison_demand is None:
                    attrs = predicate_requirement.expected_attributes
                    field = _find_ir_symbol(ir, str(attrs.get("field_text", "")))
                    if field is not None:
                        condition = PredicateNode(
                            operator=str(attrs.get("operator", "")), field=field,
                            value=LiteralValue(attrs.get("value"), "number", attrs.get("unit") or None),
                            provenance=[_requirement_provenance(
                                predicate_requirement, "dependent_duration_predicate",
                            )],
                        )
                        comparison_demand = "compiled_from_requirement"
            if action_demand is None and comparison_demand is None and set_dependency is None:
                if not any(item.code == "cumulative_duration_input_unbound" and item.span == requirement.span
                           for item in ir.unresolved):
                    ir.unresolved.append(UnresolvedItem(
                        "cumulative_duration_input_unbound", "累计时长缺少原文明确的动作或比较条件",
                        requirement.span,
                    ))
                continue
            if set_dependency is not None and action_demand is None and comparison_demand is None:
                condition = PredicateNode(
                    kind="reference", operator="interval_membership",
                    provenance=[_requirement_provenance(
                        set_dependency, "set_duration_membership",
                    )],
                )
            elif action_demand is not None:
                action_symbol = _ensure_hypothesis_symbol(
                    ir, "action", action_demand.span.start, action_demand.span.end,
                )
                condition = PredicateNode(
                    operator="eq", field=action_symbol,
                    value=LiteralValue(action_demand.attributes.get("value"), "string"),
                    provenance=[Provenance(source="rule", rule="cumulative_duration_action", span=action_demand.span)],
                )
            elif comparison_demand != "compiled_from_requirement":
                field_demand = next((item for item in ir.source_demands
                                     if item.demand_id == comparison_demand.field_anchor_id), None)
                field_text = str(comparison_demand.attributes.get("field_text", ""))
                field = _find_ir_symbol(ir, field_text)
                if field is None and field_demand is not None:
                    field = _ensure_hypothesis_symbol(
                        ir, field_text, field_demand.span.start, field_demand.span.end,
                    )
                if field is None:
                    ir.unresolved.append(UnresolvedItem(
                        "cumulative_duration_input_unbound", "累计时长比较条件缺少字段 Anchor", requirement.span,
                    ))
                    continue
                condition = PredicateNode(
                    operator=str(comparison_demand.attributes.get("operator", "")), field=field,
                    value=LiteralValue(comparison_demand.attributes.get("value"), "number",
                                       comparison_demand.attributes.get("unit") or None),
                    provenance=[Provenance(source="requirement", rule="cumulative_duration_comparison",
                                           span=comparison_demand.span)],
                )
            scope = WindowSpec(
                "calendar_day", reference="当天" if "当天" in ir.query.normalized else "query_window",
                granularity="day",
            )
            set_scope = next((
                item.output_ref for item in reversed(ir.set_operations)
                if item.output_ref and set_dependency is not None
            ), None)
            ir.cumulative_durations.append(CumulativeDurationSpec(
                duration_id=duration_id, condition=condition, scope=scope,
                input_ref=set_scope or root,
                integration_method="step",
                output=OutputRef(duration_id, "duration", "number", unit="second",
                                 shape="relation" if set_scope else "scalar", grain="day",
                                 keys=["local_date"] if set_scope else [],
                                 fields=["cumulative_duration"]),
                provenance=[_requirement_provenance(requirement, "requirement_cumulative_duration")],
            ))

        for requirement in ir.requirements:
            if requirement.operator_family != "window_aggregate":
                continue
            window_id = "window_aggregate_" + requirement.requirement_id.rsplit("_", 1)[-1]
            if any(item.window_id == window_id for item in ir.window_aggregates):
                continue
            symbols = _requirement_symbols(ir, requirement, catalog)
            if not symbols:
                continue
            symbol = symbols[0]
            unit = _symbol_unit(symbol, catalog)
            window_size = requirement.expected_attributes.get("window_size")
            window_unit = str(requirement.expected_attributes.get("window_unit", ""))
            reference = ""
            if window_size is not None and window_unit:
                rendered_size = int(window_size) if float(window_size).is_integer() else window_size
                reference = f"{rendered_size} {window_unit}"
            ir.window_aggregates.append(WindowAggregateSpec(
                window_id=window_id, function="avg",
                input=ExpressionNode("field", symbol=symbol, result_type="number", unit=unit),
                window=WindowSpec("rolling", reference=reference,
                                  granularity=_window_granularity(window_unit)),
                output=OutputRef(window_id, "value", "number", unit=unit, shape="relation", grain="window"),
                input_ref=root,
                provenance=[_requirement_provenance(requirement, "requirement_window_aggregate")],
            ))

        for requirement in ir.requirements:
            if requirement.requirement_type != "calculation" or requirement.operator_family not in {
                "correlation", "ratio", "difference",
            }:
                continue
            calculation_id = "calculation_" + requirement.requirement_id.rsplit("_", 1)[-1]
            if any(item.calculation_id == calculation_id and item.output for item in ir.calculations):
                continue
            inputs = [aggregate_outputs[item] for item in requirement.dependencies if item in aggregate_outputs]
            direct_expressions = []
            if not inputs and requirement.operator_family == "correlation":
                symbols = _requirement_symbols(ir, requirement, catalog)
                if len(symbols) == 2:
                    direct_expressions = [
                        ExpressionNode("field", symbol=symbol, result_type="number",
                                       unit=_symbol_unit(symbol, catalog))
                        for symbol in symbols
                    ]
            if len(inputs) < 2:
                same_clause = [
                    item for item in ir.requirements
                    if item.requirement_type == "aggregate"
                    and item.clause_id == requirement.clause_id
                    and item.requirement_id in aggregate_outputs
                    and (not requirement.span or not item.span or item.span.start <= requirement.span.start)
                ]
                if len(same_clause) >= 2:
                    same_clause.sort(key=lambda item: item.span.start if item.span else -1)
                    inputs = [aggregate_outputs[item.requirement_id] for item in same_clause[-2:]]
            if len(inputs) < 2 and len(direct_expressions) < 2:
                # A FormulaGap is preferable to inventing a source field.
                if not any(item.code == "formula_input_unbound" and item.span == requirement.span
                           for item in ir.unresolved):
                    ir.unresolved.append(UnresolvedItem(
                        "formula_input_unbound", f"{requirement.operator_family} 缺少明确的上游聚合输入",
                        requirement.span,
                    ))
                continue
            expressions = direct_expressions or [
                ExpressionNode("output_ref", reference=item.ref_id,
                               result_type=item.result_type, unit=item.unit)
                for item in inputs
            ]
            unit = ("ratio" if requirement.operator_family in {"correlation", "ratio"}
                    else inputs[0].unit)
            ir.calculations.append(CalculationSpec(
                calculation_type=requirement.operator_family,
                expression=(f"{requirement.operator_family}("
                            + ", ".join(
                                item.ref_id for item in inputs
                            ) + ")" if inputs else
                            f"{requirement.operator_family}("
                            + ", ".join(item.symbol.raw_name for item in expressions if item.symbol)
                            + ")"),
                parameters={"operator": requirement.operator_family},
                provenance=[_requirement_provenance(requirement, "requirement_calculation")],
                calculation_id=calculation_id, inputs=expressions,
                output=OutputRef(calculation_id, "value", "number", unit=unit, shape="scalar", grain="scalar",
                                 fields=_requirement_output_fields(requirement)),
                scope="relation",
            ))

        for requirement in ir.requirements:
            if requirement.requirement_type != "top_k":
                continue
            top_id = "top_k_" + requirement.requirement_id.rsplit("_", 1)[-1]
            if any(item.top_k_id == top_id for item in ir.top_k_operations):
                continue
            inputs = [aggregate_outputs[item] for item in requirement.dependencies if item in aggregate_outputs]
            if not inputs:
                continue
            input_ref = inputs[-1]
            limit = int(requirement.expected_attributes.get("expected_limit") or 0)
            if limit <= 0:
                continue
            ir.top_k_operations.append(TopKSpec(
                top_k_id=top_id, input_ref=input_ref,
                rank_by=ExpressionNode("output_ref", reference=input_ref.ref_id,
                                       result_type=input_ref.result_type, unit=input_ref.unit),
                limit=limit,
                output=OutputRef(top_id, "rows", "relation", shape="relation", grain=input_ref.grain,
                                  keys=list(input_ref.keys), fields=_requirement_output_fields(requirement)),
                provenance=[_requirement_provenance(requirement, "requirement_top_k")],
            ))

    @staticmethod
    def _generic_set_operations(ir: UnderstandingIR) -> None:
        if ir.set_operations or len(ir.events) < 2:
            return
        match = re.search(r"交集|重合(?:时段|时间段|日期|区间|度)", ir.query.normalized)
        if not match:
            return
        inputs = [item.output_name for item in ir.events[:2]]
        refs = [item.output_ref for item in ir.events[:2] if item.output_ref]
        granularity = "date" if "日期" in match.group(0) else "interval"
        output = OutputRef(
            "set_intersection_1", "rows", "relation", shape="relation", grain=granularity,
        )
        ir.set_operations.append(SetOperationSpec(
            operation="intersection", inputs=inputs,
            output_name="abnormal_intersection_dates" if granularity == "date" else "abnormal_overlap_intervals",
            granularity=granularity, input_refs=refs, output_ref=output,
            provenance=[Provenance(
                source="rule", rule="generic_event_intersection",
                span=SourceSpan(match.start(), match.end(), match.group(0)),
            )],
        ))

    @staticmethod
    def _unsatisfiable(ir: UnderstandingIR) -> None:
        by_field: dict[str, list[PredicateNode]] = {}
        for item in walk_predicates(ir.filters):
            if item.kind == "boolean" or not item.field or not item.value:
                continue
            key = item.field.canonical_id or item.field.raw_name.lower()
            by_field.setdefault(key, []).append(item)
        for field, items in by_field.items():
            lowers = [x for x in items if x.operator in {"gt", "gte"} and isinstance(x.value.value, (int, float))]
            uppers = [x for x in items if x.operator in {"lt", "lte"} and isinstance(x.value.value, (int, float))]
            if not lowers or not uppers:
                continue
            lower = max(lowers, key=lambda x: float(x.value.value))
            upper = min(uppers, key=lambda x: float(x.value.value))
            if float(lower.value.value) < float(upper.value.value):
                continue
            spans = [x.provenance[0].span for x in (lower, upper) if x.provenance and x.provenance[0].span]
            check_id = "unsat_" + hashlib.sha1(field.encode("utf-8")).hexdigest()[:10]
            ir.unsatisfiable.append(UnsatisfiableSpec(
                check_id=check_id,
                reason=f"字段 {field} 的下界不小于上界，条件集合为空",
                predicate_spans=spans,
                provenance=[Provenance(source="validator", rule="numeric_interval_empty")],
            ))

    @staticmethod
    def _ordered_fold(ir: UnderstandingIR) -> None:
        query = ir.query.normalized
        if not re.search(r"按.*(?:顺序|时间).*(?:累计|重建)|逐(?:条|笔|行).*累加|(?:遇到|可).*(?:重置|校正)|初始.*(?:加|减)", query):
            return
        symbols = list(ir.projections)
        symbols.extend(item.field for item in ir.metrics)
        symbols.extend(SymbolRef(
            raw_name=item.raw_name, canonical_id=None,
            candidates=[CandidateRef(item.hypothesis_id, source="query_schema")],
            status="candidate", ref_kind="hypothesis", span=item.span,
        ) for item in ir.schema_hypotheses)
        unique = []
        seen = set()
        for item in symbols:
            key = item.canonical_id or item.raw_name
            if key not in seen:
                seen.add(key)
                unique.append(item)
        order = next((x for x in unique if re.search(r"time|date|时间|日期", x.raw_name, re.I)), None)
        delta = next((x for x in unique if re.search(r"change|delta|变动|增量|变化", x.raw_name, re.I)), None)
        partition = next((x for x in unique if re.search(r"_id$|编号|设备|用户|商品", x.raw_name, re.I)), None)
        if not order or not delta:
            return
        initial_match = re.search(r"初始\S{0,10}?(?:为|=)?\s*(-?\d+(?:\.\d+)?)", query)
        initial_value = float(initial_match.group(1)) if initial_match else 0.0
        initial = ExpressionNode(kind="literal", literal=LiteralValue(initial_value, "number"), result_type="number")
        transition = ExpressionNode(
            kind="binary", operator="add", result_type="number", arguments=[
                ExpressionNode(kind="reference", reference="previous_state", result_type="number"),
                ExpressionNode(kind="field", symbol=delta, result_type="number"),
            ],
        )
        reset_condition = None
        reset_expression = None
        reset_symbol = next((x for x in unique if re.search(r"event_type|type|事件类型|reset|correct|盘点|校正", x.raw_name, re.I)), None)
        if re.search(r"重置|校正|盘点", query) and reset_symbol:
            reset_condition = PredicateNode(
                operator="eq", field=reset_symbol,
                value=LiteralValue("盘点校正", "string"),
                provenance=[Provenance(source="rule", rule="reset_condition")],
            )
            reset_expression = ExpressionNode(kind="field", symbol=delta, result_type="number")
        output = OutputRef("state_scan_1", "rows", "relation", shape="relation", grain="partition_order")
        state = StateTransitionSpec(
            transition_id="state_transition_1", partition_by=[partition] if partition else [],
            order_by=order, initial_state=initial, transition_expression=transition,
            reset_condition=reset_condition, reset_expression=reset_expression,
            missing_data_policy="reject", output=output,
            provenance=[Provenance(source="rule", rule="ordered_fold")],
        )
        duration_match = re.search(
            r"连续\s*(\d+(?:\.\d+)?)\s*(毫秒|秒钟?|分钟|分|小时|时|天|日)", query
        )
        duration = None
        if duration_match:
            seconds = {"毫秒": .001, "秒": 1, "秒钟": 1, "分钟": 60, "分": 60,
                       "小时": 3600, "时": 3600, "天": 86400, "日": 86400}
            unit = duration_match.group(2)
            value = float(duration_match.group(1))
            duration = DurationConstraint(
                "gte", value, unit, value * seconds[unit],
                SourceSpan(duration_match.start(), duration_match.end(), duration_match.group(0)),
            )
        state_symbol = SymbolRef(
            raw_name="derived_state", canonical_id="state_scan_1.state",
            status="resolved", ref_kind="output",
        )
        post_condition = PredicateNode(
            operator="lt", field=state_symbol, value=LiteralValue(0.0, "number"),
            provenance=[Provenance(source="rule", rule="state_post_condition")],
        ) if re.search(r"负数|小于\s*0|低于\s*0", query) else None
        ir.ordered_folds.append(OrderedFoldSpec(
            fold_id="ordered_fold_1", transition=state, output=output,
            post_condition=post_condition, post_duration=duration,
            provenance=[Provenance(source="rule", rule="ordered_fold")],
        ))

    @staticmethod
    def _invocations(ir: UnderstandingIR) -> None:
        values: list[tuple[str, str]] = []
        if ir.filters:
            values.append(("filter", "FILTER"))
        values.extend((item.event_id, "CUMULATIVE_DURATION" if item.derived_metric.metric_id.endswith("cumulative_duration") else "CONSECUTIVE") for item in ir.events)
        values.extend((item.output_name, item.operation.upper()) for item in ir.set_operations)
        values.extend((item.aggregate.aggregate_id, "QUANTILE" if item.aggregate.function == "quantile"
                       else "SCOPED_AGGREGATE") for item in ir.aggregates)
        values.extend((item.group_id, "GROUP_BY") for item in ir.group_by_operations)
        values.extend((item.top_k_id, "TOP_K") for item in ir.top_k_operations)
        values.extend((item.window_id, "WINDOW") for item in ir.window_aggregates)
        values.extend((item.count_id, "CUMULATIVE_COUNT") for item in ir.cumulative_counts)
        values.extend((item.duration_id, "CUMULATIVE_DURATION") for item in ir.cumulative_durations)
        values.extend((item.calculation_id or f"calculation_{index}", "ARITHMETIC")
                      for index, item in enumerate(ir.calculations, start=1))
        values.extend((item.comparison_id, "COMPARE") for item in ir.comparisons)
        if ir.grouping or ir.output.group_by:
            values.append(("grouping", "GROUP_BY"))
        if ir.output.limit:
            values.append(("top_k", "TOP_K"))
        if ir.temporal or ir.sampling_policies:
            values.append(("window", "WINDOW"))
        values.extend((item.fold_id, "RESETTABLE_SCAN" if item.transition.reset_condition else "ORDERED_FOLD") for item in ir.ordered_folds)
        values.extend((item.sequence_id, "SEQUENCE") for item in ir.sequences)
        values.extend((item.arithmetic_id, "ARITHMETIC") for item in ir.arithmetic)
        values.extend((item.filter_id, "RELATION_FILTER") for item in ir.relation_filters)
        values.extend(
            (f"{item.event_id}_threshold", "ARITHMETIC")
            for item in ir.events if item.condition.right_expression is not None
        )
        values.extend((item.check_id, "UNSATISFIABLE") for item in ir.unsatisfiable)
        for directive in ir.turn_directives:
            if "cancel_previous" in directive.pre_actions:
                values.append((directive.directive_id, "TURN_CANCEL"))
            values.append((directive.directive_id, "TURN_INHERIT" if directive.context_mode == "inherit" else "TURN_REPLACE"))
        ir.operator_invocations = [
            OperatorInvocation(
                f"invoke_{index}", operator_id, input_refs=[node_id], requirement_ids=[],
            )
            for index, (node_id, operator_id) in enumerate(values, start=1)
        ]


def _append_and(ir: UnderstandingIR, predicate: PredicateNode) -> None:
    if ir.filters is None:
        ir.filters = predicate
    elif ir.filters.kind == "boolean" and ir.filters.operator == "and":
        ir.filters.children.append(predicate)
    else:
        ir.filters = PredicateNode(kind="boolean", operator="and", children=[ir.filters, predicate])


_DURATION_FACTORS = {
    "毫秒": 0.001, "秒": 1.0, "秒钟": 1.0, "分钟": 60.0, "分": 60.0,
    "小时": 3600.0, "时": 3600.0, "天": 86400.0, "日": 86400.0,
    "交易日": 86400.0, "个交易日": 86400.0,
}


def _seconds(value: float, unit: str) -> float:
    return value * _DURATION_FACTORS[unit]


def _comparison_operator(raw: str) -> str:
    if raw in {"低于", "小于"}:
        return "lt"
    if raw in {"不超过", "小于等于", "至多"}:
        return "lte"
    if raw in {"不少于", "至少"}:
        return "gte"
    return "gt"


def _canonical_unit(raw: str | None) -> str | None:
    if not raw:
        return None
    return {
        "℃": "celsius", "°c": "celsius", "mpa": "megapascal",
        "kpa": "kilopascal", "pa": "pascal", "%": "percent",
        "v": "volt", "手": "count", "次": "count", "个": "count",
        "笔": "count", "元": "currency", "美元": "currency", "欧元": "currency",
        "m/s²": "m/s^2", "m/s2": "m/s^2",
    }.get(raw.lower(), raw)


def _prov(query: str, start: int, end: int, rule: str) -> Provenance:
    return Provenance(
        source="rule", rule=rule,
        span=SourceSpan(start, end, query[start:end]),
    )


def _symbol_id(symbol: SymbolRef) -> str:
    if symbol.canonical_id:
        return symbol.canonical_id
    return symbol.candidates[0].identifier if symbol.candidates else symbol.raw_name


def _same_symbol(left: SymbolRef, right: SymbolRef) -> bool:
    return _symbol_id(left) == _symbol_id(right)


def _all_symbols(ir: UnderstandingIR) -> list[SymbolRef]:
    result = list(ir.projections)
    result.extend(item.field for item in ir.metrics)
    result.extend(ir.grouping)
    result.extend(
        item.field for root in [ir.filters, *(event.condition for event in ir.events)]
        for item in walk_predicates(root) if item.field
    )
    for hypothesis in ir.schema_hypotheses:
        for alias in hypothesis.aliases or [hypothesis.raw_name]:
            result.append(SymbolRef(
                raw_name=alias, candidates=[CandidateRef(
                    hypothesis.hypothesis_id, source="query_schema", status="candidate"
                )], status="candidate", span=hypothesis.span, ref_kind="hypothesis",
            ))
    unique = []
    seen = set()
    for item in result:
        key = (_symbol_id(item), item.raw_name.lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _find_ir_symbol(ir: UnderstandingIR, raw_name: str) -> SymbolRef | None:
    cleaned = re.sub(r"^(?:以及|并且|同时|且|同期)", "", raw_name).strip()
    symbols = _all_symbols(ir)
    exact = [item for item in symbols if item.raw_name.lower() == cleaned.lower()]
    if exact:
        return next((item for item in exact if item.ref_kind == "hypothesis"), exact[0])
    contained = [
        item for item in symbols
        if item.raw_name.lower() in cleaned.lower() or cleaned.lower() in item.raw_name.lower()
    ]
    return max(
        contained,
        key=lambda item: (item.ref_kind == "hypothesis", len(item.raw_name)),
        default=None,
    )


def _requirement_provenance(requirement, rule: str) -> Provenance:
    """Keep every analytic node anchored to the Requirement that requested it."""
    return Provenance(source="requirement", rule=rule, span=requirement.span)


def _requirement_output_fields(requirement) -> list[str]:
    """Copy the immutable Requirement output contract into its OutputRef."""
    return list(dict.fromkeys(getattr(requirement, "expected_output_fields", []) or []))


def _generic_calculation_inputs(ir: UnderstandingIR, calculation_type: str) -> list[ExpressionNode]:
    """Select only already materialized, typed DAG outputs for generic math."""
    def output_expression(output: OutputRef) -> ExpressionNode:
        return ExpressionNode("output_ref", reference=output.ref_id,
                              result_type=output.result_type, unit=output.unit)

    aggregate_outputs = [item.aggregate.output for item in ir.aggregates]
    event_outputs = [item.output_ref for item in ir.events if item.output_ref]
    sequence_outputs = [item.output for item in ir.sequences]
    if calculation_type == "overlap_ratio" and len(event_outputs) >= 2:
        return [output_expression(item) for item in event_outputs[:2]]
    if calculation_type in {"conversion_rate", "median"} and sequence_outputs:
        return [output_expression(sequence_outputs[0])]
    if calculation_type in {"covariance", "ratio"} and len(aggregate_outputs) >= 2:
        return [output_expression(item) for item in aggregate_outputs[:2]]
    if calculation_type == "quantile" and aggregate_outputs:
        return [output_expression(aggregate_outputs[0])]
    return []


def _requirement_symbols(ir: UnderstandingIR, requirement, catalog) -> list[SymbolRef]:
    """Resolve only the explicitly declared field inputs for one requirement.

    Requirements are the provenance boundary for analytic operators.  In
    particular, this deliberately never falls back to the closest preceding
    field: a calculation must either name its source or remain a FormulaGap.
    """
    expected = [value[6:] for value in requirement.expected_inputs
                if value.startswith("field:") and value[6:]]
    demand_ids = set(requirement.expected_attributes.get("input_demand_ids", []))
    demands = {item.demand_id: item for item in ir.source_demands}
    result: list[SymbolRef] = []
    for raw in expected:
        symbol = _find_ir_symbol(ir, raw)
        demand = next((demands[item] for item in demand_ids
                       if item in demands and str(demands[item].attributes.get(
                           "field_text", demands[item].text.strip("`"))) == raw), None)
        if symbol is None and demand is not None:
            symbol = _ensure_hypothesis_symbol(
                ir, raw, demand.span.start, demand.span.end,
                demand.attributes.get("unit"),
            )
        if symbol is not None and not any(_same_symbol(symbol, item) for item in result):
            result.append(symbol)
    return result


def _ensure_hypothesis_symbol(ir: UnderstandingIR, raw_name: str, start: int, end: int,
                              unit: str | None = None) -> SymbolRef:
    existing = _find_ir_symbol(ir, raw_name)
    if existing:
        return existing
    hypothesis_id = "hyp:" + hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:16]
    hypothesis = SchemaHypothesis(
        hypothesis_id=hypothesis_id, raw_name=raw_name,
        normalized_name=raw_name.lower(), declared_type="number",
        description="问题中引用但 Catalog 未绑定的动态基准", unit=unit,
        aliases=[raw_name], span=SourceSpan(start, end, ir.query.normalized[start:end]),
        executable=False,
    )
    ir.schema_hypotheses.append(hypothesis)
    return SymbolRef(
        raw_name=raw_name, candidates=[CandidateRef(
            hypothesis_id, source="query_schema", status="candidate"
        )], status="candidate", span=hypothesis.span, ref_kind="hypothesis",
    )


def _symbol_unit(symbol: SymbolRef, catalog) -> str | None:
    if catalog is not None and symbol.canonical_id:
        field = catalog.field(symbol.canonical_id)
        if field:
            return field.unit
    identifier = _symbol_id(symbol)
    return next((item.unit for item in getattr(catalog, "schema_hypotheses", [])
                 if item.hypothesis_id == identifier), None) if catalog else None


def _predicate_span(predicate: PredicateNode) -> SourceSpan | None:
    return next((item.span for item in predicate.provenance if item.span), None)


def _overlaps_event(ir: UnderstandingIR, start: int, end: int) -> bool:
    for event in ir.events:
        span = next((item.span for item in event.provenance if item.span), None)
        if span and start < span.end and end > span.start:
            return True
    return False


def _partition_and_order(ir: UnderstandingIR) -> tuple[list[SymbolRef], SymbolRef | None]:
    symbols = _all_symbols(ir)
    partitions = [item for item in symbols
                  if re.search(r"(?:^|_)id$|编号$|^sku$", item.raw_name, re.I)]
    order = next((item for item in symbols
                  if re.search(r"timestamp|time|date|时间|日期|^ts$", item.raw_name, re.I)), None)
    return partitions[:2], order


def _sampling_for(ir: UnderstandingIR, field: SymbolRef | None, catalog) -> SamplingPolicy:
    interval = next((item.expected_interval_seconds for item in ir.sampling_policies
                     if item.expected_interval_seconds), None)
    partition, order = _partition_and_order(ir)
    if catalog is not None and field and field.canonical_id:
        source_id = catalog.source_for_field(field.canonical_id)
        source = catalog.source(source_id or "")
        if source:
            order_id = source.metadata.get("timestamp_field")
            if order_id and catalog.field(order_id):
                order = SymbolRef(raw_name=order_id.rsplit(".", 1)[-1], canonical_id=order_id,
                                  candidates=[CandidateRef(order_id)], status="resolved")
            partition = [SymbolRef(
                raw_name=item.rsplit(".", 1)[-1], canonical_id=item,
                candidates=[CandidateRef(item)], status="resolved",
            ) for item in source.metadata.get("partition_fields", []) if catalog.field(item)]
            interval = interval or source.metadata.get("expected_sampling_interval_seconds")
    return SamplingPolicy(
        order_by=order, partition_by=partition,
        expected_interval_seconds=interval,
        max_gap_seconds=(interval * 2 if interval else None),
        missing_data_policy="break_segment",
    )


def _window(ir: UnderstandingIR) -> WindowSpec:
    item = ir.temporal[0] if ir.temporal else None
    return WindowSpec(
        window_type="absolute",
        from_value=(item.from_value or item.exact) if item else None,
        to_value=(item.to_value or item.exact) if item else None,
        timezone=item.timezone if item and item.timezone else "Asia/Shanghai",
        granularity=item.granularity if item else "minute",
    )


def _window_granularity(unit: str) -> str:
    if unit in {"日", "天"}:
        return "day"
    if unit in {"小时", "时"}:
        return "hour"
    if unit in {"分钟", "分"}:
        return "minute"
    return "second"


def _append_consecutive_event(ir: UnderstandingIR, condition: PredicateNode, match,
                              catalog, rule: str) -> None:
    query = ir.query.normalized
    field = next((item.field for item in walk_predicates(condition) if item.field), None)
    source_id = catalog.source_for_field(field.canonical_id or "") if catalog and field else None
    metric_id = f"{source_id}.consecutive_duration" if source_id else "semantic.consecutive_duration"
    event_id = f"event_{len(ir.events) + 1}_consecutive"
    duration_value = float(match.group("duration"))
    duration_unit = match.group("duration_unit")
    span = SourceSpan(match.start(), match.end(), match.group(0))
    ir.events.append(EventSpec(
        event_id=event_id, condition=condition,
        derived_metric=DerivedMetricCall(
            metric_id=metric_id, arguments={"operator": "CONSECUTIVE"},
            result_type="duration", unit="second",
            provenance=[Provenance(source="rule", rule=rule, span=span)],
        ),
        threshold=DurationConstraint(
            "gt", duration_value, duration_unit, _seconds(duration_value, duration_unit),
            SourceSpan(match.start("duration"), match.end("duration_unit"),
                       query[match.start("duration"):match.end("duration_unit")]),
        ),
        window=_window(ir), sampling=_sampling_for(ir, field, catalog),
        group_by=["local_date"], output_name=f"{event_id}_intervals",
        output_ref=OutputRef(event_id, "intervals", "event_interval",
                             shape="event_set", grain="interval", keys=["local_date"]),
        provenance=[Provenance(source="rule", rule=rule, span=span)],
    ))
