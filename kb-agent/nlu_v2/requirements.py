"""Independent requirements extracted from query text, never from SemanticIR claims."""
from __future__ import annotations

import hashlib
import re

from .field_roles import FieldPhraseParser
from .models import ClauseGraph, RequirementSpec, SourceDemand, SourceSpan


_CUES = (
    ("set_operation", "target", "set", re.compile(r"交集|并集|差集|重合(?:时段|时间段|日期|区间|度)")),
    ("boolean_logic", "constraint", "boolean", re.compile(r"同时满足")),
    ("sequence_event", "constraint", "event", re.compile(r"连续|一直|保持(?:了)?\s*\d|持续\s*\d|上穿|下穿|金叉|死叉")),
    ("cumulative_aggregate", "constraint", "aggregate", re.compile(
        r"[`A-Za-z_\u4e00-\u9fff]{1,24}累计(?:超过|高于|大于|不少于|至少)"
    )),
    ("derived_projection", "target", "period_duration", re.compile(r"(?:当日|当天|自然日)总时长")),
    ("cumulative_duration", "constraint", "cumulative_duration", re.compile(
        r"(?:累计|(?<!当日)(?<!当天)(?<!自然日)总|合计)"
        r"(?:[A-Za-z_\u4e00-\u9fff]{0,12})?(?:时长|耗时)"
    )),
    ("sampling", "context", "sampling", re.compile(r"每\s*\d+(?:\.\d+)?\s*(?:毫秒|秒|分钟|分|小时|时)(?:记录|采样|上报)?(?:一次)?")),
    ("scoped_aggregate", "target", "aggregate", re.compile(r"全站平均|总体平均|筛选后平均|这些.+平均|平均累计")),
    ("calculation", "target", "calculation", re.compile(r"同比|环比|增长率|波动率|标准差|相关性|相关系数|计算")),
    ("formula", "target", "formula", re.compile(r"最大回撤|夏普比率|复合增长率|CAGR|分位数|(?<![A-Za-z])P\d+", re.I)),
    ("conversion", "constraint", "conversion", re.compile(r"汇率|换算|折算|转换为|统一为|人民币|美元|欧元|CNY|USD|EUR", re.I)),
    ("reference", "constraint", "reference", re.compile(
        r"(?:该|这些|上述|以上|前述)(?:日期|时间段|区间|结果|记录|用户|文档|事件|集合)|"
        r"(?:这|上述|以上)两类(?:异常)?(?:日期|时间段|区间|结果|事件)|"
        r"前一步|上一步|刚才第[一二三四五六七八九十\d]+"
    )),
    ("turn_directive", "constraint", "turn", re.compile(r"停止刚才|取消(?:旧|上一个|前一个)?|上一轮|改成|替换|不是.+?是|先说.+?随后")),
    ("output", "output", "output", re.compile(r"最终输出|输出|返回|列出|告诉我|提取|找出")),
)
_NUMERIC_RE = re.compile(
    r"(?P<op>大于等于|不低于|不少于|至少|小于等于|不高于|不超过|未超过|至多|"
    r"大于|高于|超过|小于|低于|少于|不等于|等于|>=|<=|!=|>|<|=)\s*"
    r"(?P<number>-?\d+(?:\.\d+)?)"
)
_FIELD_TOKEN_RE = re.compile(r"[`A-Za-z_\u4e00-\u9fff][`A-Za-z0-9_.\u4e00-\u9fff]{0,30}$")
_NULL_RE = re.compile(r"(?P<op>不为空|非空|不能为空|为空|是空值|为null|not\s+null|null)", re.I)
_TWO_DIGIT_YEAR_RE = re.compile(r"(?<!\d)(?P<year>\d{2})年")


class RequirementExtractor:
    """Extract obligations using only raw text and the independently built ClauseGraph."""

    def extract(self, query: str, graph: ClauseGraph, ledger=None) -> list[RequirementSpec]:
        self._query_text = query
        requirements: list[RequirementSpec] = []
        inherited_field = ""
        for node in (item for item in graph.nodes if item.instruction_scope == "active"):
            for kind, role, family, pattern in _CUES:
                for match in pattern.finditer(node.text):
                    if kind in {"calculation", "formula"} and (
                        node.role == "output" or _is_output_reference(query, node)
                    ):
                        continue
                    effective_family = _calculation_family(match.group(0)) if kind == "calculation" else family
                    requirements.append(self._make(
                        kind, role, effective_family,
                        _absolute_span(node, match.start(), match.end()), node.clause_id,
                    ))
            # The ledger owns numeric comparison evidence when present.  It
            # supplies four slot anchors; recreating a separate broad-prefix
            # requirement here loses that evidence relationship.
            if ledger is None:
                numeric, inherited_field = self._numeric_requirements(node, inherited_field)
                requirements.extend(numeric)
            nulls, inherited_field = self._null_requirements(node, inherited_field)
            requirements.extend(nulls)
            for match in re.finditer(r"中文|英文", node.text):
                requirements.append(self._make(
                    "predicate", "constraint", "predicate",
                    _absolute_span(node, match.start(), match.end()), node.clause_id,
                    {"field_text": "language", "operator": "eq", "value": match.group(0)},
                ))
            for match in re.finditer(r"有代码|包含代码|代码地址", node.text):
                requirements.append(self._make(
                    "predicate", "constraint", "predicate",
                    _absolute_span(node, match.start(), match.end()), node.clause_id,
                    {"field_text": "code_url", "operator": "is_not_null"},
                ))
            for match in _TWO_DIGIT_YEAR_RE.finditer(node.text):
                requirements.append(self._make(
                    "temporal_ambiguity", "constraint", "ambiguity",
                    _absolute_span(node, match.start(), match.end()), node.clause_id,
                    {"year_text": match.group("year")},
                ))
        requirements.extend(self._satisfiability_requirements(requirements, query))
        timezone_start = re.search(r"美国东部时间", query)
        dst = re.search(r"夏令时|DST", query, re.I)
        if timezone_start and dst:
            # The ambiguity is determined by the complete stated condition
            # (including the "offset or fold" branch), not merely the words
            # "夏令时".  Keep the entire source evidence range for slot-level
            # consumption and coverage.
            start, end = timezone_start.start(), len(query)
            text = query[start:end]
            if not re.search(r"(?:UTC\s*offset|偏移)\s*(?:为|=)\s*[+-]\d|fold\s*(?:为|=)\s*[01]", query, re.I):
                requirements.append(self._make(
                    "temporal_ambiguity", "constraint", "ambiguity",
                    SourceSpan(start, end, text),
                    next((x.clause_id for x in graph.nodes
                          if x.span.start <= start < x.span.end), ""),
                    {"kind": "dst_fold"},
                ))
        requirements = self._dedupe(requirements)
        requirements = self._apply_source_demands(requirements, list(getattr(ledger, "demands", [])))
        requirements = self._attach_clause_dependencies(requirements, graph)
        return self._complete_output_contracts(self._wire_semantic_dependencies(requirements))

    @staticmethod
    def _complete_output_contracts(requirements: list[RequirementSpec]) -> list[RequirementSpec]:
        """Materialize output contracts before SemanticIR lowering.

        A requirement used to state only an operator family.  These stable
        semantic field labels let the lowerer and Coverage independently agree
        on which value is produced without encoding local node IDs in the
        query-facing RequirementIR.
        """
        for item in requirements:
            attrs = item.expected_attributes
            family = item.operator_family
            input_fields = [value[6:] for value in item.expected_inputs if value.startswith("field:")]
            expected_type = ""
            fields: list[str] = []
            if item.requirement_type == "aggregate":
                function = next(iter(attrs.get("expected_functions", [])), family or "aggregate")
                fields = [f"{function}:{field}" for field in input_fields] or [str(function)]
                expected_type = "number"
            elif item.requirement_type == "group_by":
                fields = [f"group:{field}" for field in input_fields] or ["group"]
                expected_type = "relation"
            elif item.requirement_type == "top_k":
                fields = ["top_k"]
                expected_type = "relation"
            elif item.requirement_type == "cumulative_count":
                action = str(attrs.get("expected_action", ""))
                fields = [f"count:{action}"] if action else ["count"]
                expected_type = "number"
            elif item.requirement_type == "cumulative_duration":
                fields = ["cumulative_duration"]
                expected_type = "number"
            elif item.requirement_type == "calculation" and family not in {"", "calculation"}:
                fields = [family]
                expected_type = "number"
            elif item.requirement_type == "set_operation":
                # Set/Event producers are typed by their OutputRef shape and
                # result type; unlike analytic value producers, the query does
                # not name a stable output value field.
                expected_type = "relation"
            elif item.requirement_type == "sequence_event":
                expected_type = "event_interval"
            if fields:
                item.expected_output_fields = list(dict.fromkeys(fields))
                attrs["expected_output_type"] = expected_type
        return requirements

    @staticmethod
    def _attach_clause_dependencies(requirements: list[RequirementSpec], graph: ClauseGraph) -> list[RequirementSpec]:
        """Materialize source-clause dataflow as RequirementIR dependencies."""
        by_clause: dict[str, list[RequirementSpec]] = {}
        for item in requirements:
            if item.clause_id:
                by_clause.setdefault(item.clause_id, []).append(item)
        dataflow = {"USES_OUTPUT", "DEPENDS_ON", "THEN", "INHERITS", "SAME_SCOPE"}
        for edge in graph.edges:
            if edge.relation not in dataflow:
                continue
            upstream = [item.requirement_id for item in by_clause.get(edge.source_id, [])]
            if edge.relation == "SAME_SCOPE":
                upstream = [
                    item.requirement_id for item in by_clause.get(edge.source_id, [])
                    if item.requirement_type in {
                        "predicate", "sequence_event", "event", "derived_projection",
                        "set_operation", "group_by", "aggregate", "scoped_aggregate",
                        "calculation",
                    }
                ]
            for target in by_clause.get(edge.target_id, []):
                target.dependencies = sorted(set(target.dependencies) | set(upstream))
        return requirements

    def _apply_source_demands(self, requirements: list[RequirementSpec],
                               demands: list[SourceDemand]) -> list[RequirementSpec]:
        """Create compiler obligations for every semantic source demand.

        Demand consumption and Requirement coverage are intentionally distinct:
        a demand is consumed once a requirement owns it, but completion still
        requires that requirement to be proven against the produced IR.
        """
        specialized_calculation_clauses = {
            item.clause_id for item in demands
            if item.demand_type == "operator"
            and (
                item.attributes.get("operator_family") in {"correlation", "ratio", "difference"}
                or _calculation_family(item.text) != "calculation"
            )
        }
        specialized_calculation_clauses.update(
            item.clause_id for item in requirements
            if item.requirement_type == "calculation"
            and item.operator_family not in {"", "calculation"}
        )
        # If a phrase contains both “计算” and a named formula, the named
        # formula owns the obligation.  Retaining an independent bare
        # calculation requirement would make a complete correlation falsely
        # incomplete, while conversely it is never accepted as formula proof.
        requirements = [item for item in requirements if not (
            item.requirement_type == "calculation" and item.operator_family == "calculation"
            and item.clause_id in specialized_calculation_clauses
        )]
        additions: list[RequirementSpec] = []
        by_id = {item.demand_id: item for item in demands}
        for demand in demands:
            if demand.demand_type == "boolean":
                # A boolean inside a turn-revision instruction or inside the
                # stated DST ambiguity condition is consumed by that explicit
                # semantic control node.  It is still source-owned, but it is
                # not falsely modeled as a data-filter boolean with no
                # PredicateNode to prove it.
                owner = next((item for item in requirements if (
                    item.clause_id == demand.clause_id
                    and item.requirement_type in {"turn_directive", "temporal_ambiguity"}
                )), None) or next((item for item in requirements if (
                    item.requirement_type == "turn_directive"
                )), None) or next((item for item in requirements if (
                    item.requirement_type == "temporal_ambiguity"
                )), None)
                if owner is not None:
                    owner.expected_attributes["source_demand_ids"] = sorted(set(
                        owner.expected_attributes.get("source_demand_ids", []) + [demand.demand_id]
                    ))
                    continue
            requirement = self._requirement_for_demand(demand, by_id)
            if requirement is not None:
                additions.append(requirement)

        combined = self._dedupe([*requirements, *additions])
        for clause in specialized_calculation_clauses:
            named = next((
                item for item in combined
                if item.clause_id == clause and item.requirement_type == "calculation"
                and item.operator_family not in {"", "calculation"}
            ), None)
            if named is None:
                continue
            generic = [
                item for item in combined
                if item.clause_id == clause and item.requirement_type == "calculation"
                and item.operator_family in {"", "calculation"}
            ]
            for item in generic:
                named.expected_attributes["source_demand_ids"] = sorted(set(
                    named.expected_attributes.get("source_demand_ids", [])
                    + item.expected_attributes.get("source_demand_ids", [])
                ))
                combined.remove(item)
        # A duration threshold is a slot of the enclosing sequence/cumulative
        # event, not an independent field predicate.  Attach its exact anchors
        # to the event Requirement so CoverageDiff does not demand a fake field.
        for demand in demands:
            if demand.demand_type != "duration_threshold":
                continue
            owners = [
                item for item in combined
                if item.clause_id == demand.clause_id
                and item.requirement_type in {"sequence_event", "cumulative_duration"}
            ]
            owner = min(
                owners,
                key=lambda item: abs((item.span.start if item.span else demand.span.start)
                                     - demand.span.start),
                default=None,
            )
            if owner is not None:
                local_dependencies = [
                    identifier for identifier in demand.dependencies
                    if identifier in by_id and owner.span is not None
                    and owner.span.start <= by_id[identifier].span.start
                    and by_id[identifier].span.end <= max(owner.span.end, demand.span.end)
                ]
                owner.expected_attributes["source_demand_ids"] = sorted(set(
                    owner.expected_attributes.get("source_demand_ids", [])
                    + [demand.demand_id, *local_dependencies]
                ))
        for requirement in combined:
            if not requirement.span:
                continue
            # Preserve all explicit dependency anchors for a composite demand,
            # even when only the comparison span overlaps the requirement.
            demand_ids = set(requirement.expected_attributes.get("source_demand_ids", []))
            for demand in demands:
                if _overlaps(demand.span, requirement.span) and demand.clause_id == requirement.clause_id:
                    demand_ids.add(demand.demand_id)
            if demand_ids:
                requirement.expected_attributes["source_demand_ids"] = sorted(demand_ids)

        for demand in demands:
            mapped = [item.requirement_id for item in combined
                      if demand.demand_id in item.expected_attributes.get("source_demand_ids", [])]
            if (not mapped and demand.demand_type == "operator"
                    and demand.attributes.get("operator_family") == "calculation"):
                owner = next((item for item in combined
                              if item.clause_id == demand.clause_id
                              and item.operator_family in {"correlation", "ratio", "difference"}), None)
                if owner is not None:
                    owner.expected_attributes["source_demand_ids"] = sorted(set(
                        owner.expected_attributes.get("source_demand_ids", []) + [demand.demand_id]
                    ))
                    mapped = [owner.requirement_id]
            if mapped:
                demand.consumption_status = "mapped_to_requirement"
                demand.mapped_requirement_ids = sorted(set(mapped))
                demand.approved_noise_reason = ""
            elif demand.consumption_status != "approved_noise":
                demand.consumption_status = "unresolved"
                demand.mapped_requirement_ids = []
        return combined

    def _requirement_for_demand(self, demand: SourceDemand,
                                by_id: dict[str, SourceDemand]) -> RequirementSpec | None:
        attrs = dict(demand.attributes)
        owned = [demand.demand_id, *demand.dependencies]
        attrs["source_demand_ids"] = sorted(set(owned))
        if demand.demand_type == "comparison":
            attrs.update({
                "field_anchor_id": demand.field_anchor_id,
                "quantity_anchor_id": demand.quantity_anchor_id,
                "unit_anchor_id": demand.unit_anchor_id,
                "operator_anchor_id": demand.operator_anchor_id,
            })
            field_anchor = by_id.get(demand.field_anchor_id)
            if field_anchor and field_anchor.demand_type == "derived_metric":
                attrs["derived_operator"] = field_anchor.attributes.get("derived_operator", "")
                if attrs.get("derived_operator") == "time_ratio" and attrs.get("unit") == "%":
                    attrs["value"] = float(attrs["value"]) / 100.0
                    attrs["unit"] = "ratio"
            return self._make("predicate", "constraint", "predicate", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "field":
            attrs["field_text"] = demand.attributes.get("field_text", demand.text.strip("`"))
            return self._make("field_reference", "context", "field", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "boolean":
            return self._make("boolean_logic", "constraint", "boolean", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "temporal":
            return self._make("temporal", "context", "temporal", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "output":
            return self._make("output", "output", "output", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "action_value":
            attrs["expected_action"] = demand.attributes.get("value", "")
            return self._make("event", "constraint", "event", demand.span, demand.clause_id, attrs)
        if demand.demand_type == "count_threshold":
            return self._make(
                "count_threshold", "constraint", "count_threshold",
                demand.span, demand.clause_id, attrs,
            )
        if demand.demand_type != "operator":
            return None
        family = str(demand.attributes.get("operator_family", ""))
        if family == "calculation":
            family = _calculation_family(demand.text)
        mapping = {
            "set_operation": ("set_operation", "target", "set"),
            "sequence": ("sequence_event", "constraint", "event"),
            "cumulative_count": ("cumulative_count", "target", "cumulative_count"),
            "cumulative_duration": ("cumulative_duration", "constraint", "event"),
            "period_duration": ("derived_projection", "target", "period_duration"),
            "aggregate_avg": ("aggregate", "target", "aggregate"),
            "aggregate_sum": ("aggregate", "target", "aggregate"),
            "peak": ("aggregate", "target", "aggregate"),
            "quantile": ("aggregate", "target", "quantile"),
            "window_aggregate": ("aggregate", "target", "window_aggregate"),
            "group_by": ("group_by", "target", "group_by"),
            "top_k": ("top_k", "output", "top_k"),
            "correlation": ("calculation", "target", "correlation"),
            "ratio": ("calculation", "target", "ratio"),
            "difference": ("calculation", "target", "difference"),
            "calculation": ("calculation", "target", "calculation"),
            "conversion": ("conversion", "constraint", "conversion"),
        }.get(family)
        if mapping is None and family in {
            "mom", "yoy", "growth", "volatility", "correlation",
            "max_drawdown", "sharpe", "cagr",
        }:
            mapping = ("calculation", "target", family)
        if mapping is None:
            return None
        kind, role, operator_family = mapping
        attrs["operator_family"] = family
        function = attrs.get("function")
        if function:
            attrs["expected_functions"] = [function]
        if family == "top_k":
            attrs["expected_limit"] = attrs.get("limit")
            attrs["expected_output_shape"] = "relation"
        if family == "group_by":
            attrs["expected_output_shape"] = "relation"
        if family in {"aggregate_avg", "aggregate_sum", "peak", "quantile", "correlation", "ratio", "difference",
                      "cumulative_count", "cumulative_duration"}:
            attrs.setdefault("expected_output_shape", "scalar")
        requirement = self._make(kind, role, operator_family, demand.span, demand.clause_id, attrs)
        clause_fields = [item for item in by_id.values()
                         if item.demand_type == "field" and item.clause_id == demand.clause_id]
        clause_fields.sort(key=lambda item: (item.span.start, item.span.end))
        # An analytic cue owns only its syntactically attached field.  Giving
        # every nearby field to each aggregate was another form of implicit
        # lineage guessing ("平均功率与平均风速" became avg(power), avg(power)).
        if family in {"aggregate_avg", "aggregate_sum", "peak", "quantile", "window_aggregate"}:
            dependency_fields = [
                by_id[item] for item in demand.dependencies
                if item in by_id and by_id[item].demand_type == "field"
            ]
            following = [item for item in clause_fields
                         if demand.span.end <= item.span.start <= demand.span.end + 16]
            preceding = [item for item in clause_fields
                         if demand.span.start - 16 <= item.span.end <= demand.span.start]
            # Suffix operators bind their left field (温度峰值); prefix
            # operators bind their right field (平均温度).
            if family in {"peak", "quantile"}:
                direct_following = [
                    item for item in following
                    if 0 <= item.span.start - demand.span.end <= 4
                ]
                input_demands = (dependency_fields[:1] or direct_following[:1]
                                 or preceding[-1:] or following[:1])
            elif family == "window_aggregate":
                meaningful_preceding = [
                    item for item in preceding
                    if str(item.attributes.get("field_text", item.text.strip("`"))) not in {"线", "量"}
                ]
                input_demands = dependency_fields[:1] or meaningful_preceding[-1:]
            else:
                input_demands = following[:1] or preceding[-1:]
        elif family == "group_by":
            enclosed = [item for item in clause_fields
                        if demand.span.start <= item.span.start and item.span.end <= demand.span.end]
            preceding = [item for item in clause_fields
                         if demand.span.start - 16 <= item.span.end <= demand.span.start]
            input_demands = enclosed[:1] or preceding[-1:]
        elif family in {"correlation", "ratio", "difference"}:
            input_demands = [
                by_id[item] for item in demand.dependencies
                if item in by_id and by_id[item].demand_type == "field"
            ]
        else:
            input_demands = []
        if input_demands:
            input_ids = sorted({item.demand_id for item in input_demands})
            requirement.expected_attributes["input_demand_ids"] = input_ids
            requirement.expected_attributes["source_demand_ids"] = sorted(set(
                requirement.expected_attributes.get("source_demand_ids", [])
            ) | set(input_ids))
            requirement.expected_inputs = [
                "field:" + str(item.attributes.get("field_text", item.text.strip("`")))
                for item in input_demands
            ]
        elif family == "window_aggregate":
            prefix = getattr(self, "_query_text", "")[:demand.span.start]
            attached = re.search(
                r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\s*"
                r"连续\s*(?:高于|超过|大于|低于|小于)\s*\d+(?:\.\d+)?\s*$",
                prefix,
            )
            if attached:
                field_text = re.sub(r"^(?:以及|并且|同时|和|与|及|、)+", "", attached.group("field"))
                requirement.expected_inputs = ["field:" + field_text]
                requirement.expected_attributes["field_text"] = field_text
        group_by_grain = str(attrs.get("group_by_grain", ""))
        requirement.expected_output_shape = (
            "relation" if group_by_grain else str(attrs.get("expected_output_shape", ""))
        )
        requirement.expected_output_grain = (
            "group" if family in {"group_by", "top_k"}
            else "day" if family in {"cumulative_count", "cumulative_duration"}
            else group_by_grain if group_by_grain
            else "scalar"
        )
        requirement.expected_scope = (
            "grouped" if family == "group_by"
            else "calendar_day" if family in {"cumulative_count", "cumulative_duration"}
            else "relation" if family in {"correlation", "ratio", "difference"}
            else str(attrs.get("scope", "global"))
            if family in {"aggregate_avg", "aggregate_sum", "peak", "quantile"}
            else ""
        )
        return requirement

    @staticmethod
    def _wire_semantic_dependencies(requirements: list[RequirementSpec]) -> list[RequirementSpec]:
        """Declare source-order dependency edges needed by nested operators.

        This stays domain-neutral: a final ratio/correlation consumes preceding
        aggregate requirements in the same active clause; Top-K consumes a
        grouping/aggregate result.  Coverage later requires the corresponding
        OutputRef edges in the concrete DAG.
        """
        by_clause: dict[str, list[RequirementSpec]] = {}
        for item in requirements:
            by_clause.setdefault(item.clause_id, []).append(item)
        for items in by_clause.values():
            items.sort(key=lambda item: item.span.start if item.span else -1)
            for index, item in enumerate(items):
                prior = items[:index]
                if item.operator_family in {"correlation", "ratio", "difference"}:
                    inputs = [x.requirement_id for x in prior if x.requirement_type == "aggregate"]
                    item.dependencies = sorted(set(item.dependencies) | set(inputs))
                    item.expected_inputs = sorted(set(item.expected_inputs) | {
                        "requirement:" + value for value in inputs
                    })
                elif item.requirement_type == "aggregate":
                    groups = [x.requirement_id for x in prior if x.requirement_type == "group_by"]
                    if groups:
                        item.dependencies = sorted(set(item.dependencies) | set(groups))
                        item.expected_inputs = sorted(set(item.expected_inputs) | {
                            "requirement:" + value for value in groups
                        })
                        item.expected_scope = "relation"
                        item.expected_output_shape = "relation"
                        item.expected_output_grain = "group"
                elif item.requirement_type == "top_k":
                    aggregates = [x.requirement_id for x in prior if x.requirement_type == "aggregate"]
                    inputs = aggregates or [x.requirement_id for x in prior if x.requirement_type == "group_by"]
                    item.dependencies = sorted(set(item.dependencies) | set(inputs))
                    # Top-K consumes the immediately ranked relation.  A
                    # group-by that feeds that aggregate remains a transitive
                    # dependency, not a fictitious second direct OutputRef.
                    item.expected_inputs = sorted(
                        {value for value in item.expected_inputs if not value.startswith("requirement:")}
                        | {"requirement:" + value for value in inputs}
                    )
                elif item.requirement_type == "output":
                    inputs = [x.requirement_id for x in prior if x.role != "context"]
                    item.dependencies = sorted(set(item.dependencies) | set(inputs))
                    item.expected_inputs = sorted(set(item.expected_inputs) | {
                        "requirement:" + value for value in inputs
                    })
        # ClauseGraph intentionally preserves punctuation as lightweight
        # context clauses.  A final Top-K/output phrase can therefore live in
        # the immediately following clause while still consuming the preceding
        # grouping/aggregate DAG.  Recover only backward, source-ordered edges.
        ordered = sorted(requirements, key=lambda item: item.span.start if item.span else -1)
        for index, item in enumerate(ordered):
            if item.requirement_type == "set_operation" and not item.dependencies:
                producers = [
                    value for value in ordered[:index]
                    if value.requirement_type in {"sequence_event", "event"}
                ]
                # A binary set operation has one deterministic producer pair
                # only when exactly two event Requirements precede it.
                if len(producers) == 2:
                    inputs = [value.requirement_id for value in producers]
                    item.dependencies = sorted(set(item.dependencies) | set(inputs))
                    item.expected_inputs = sorted(set(item.expected_inputs) | {
                        "requirement:" + value for value in inputs
                    })
            if item.requirement_type not in {"top_k", "output"} or item.dependencies:
                continue
            prior = ordered[:index]
            accepted = {"group_by", "aggregate"} if item.requirement_type == "top_k" else {
                "predicate", "field_reference", "event", "sequence_event", "set_operation",
                "aggregate", "calculation", "group_by", "top_k",
            }
            inputs = [x.requirement_id for x in prior if x.requirement_type in accepted]
            item.dependencies = sorted(set(item.dependencies) | set(inputs))
            item.expected_inputs = sorted(set(item.expected_inputs) | {
                "requirement:" + value for value in inputs
            })
        return requirements

    def _numeric_requirements(self, node, inherited_field: str) -> tuple[list[RequirementSpec], str]:
        result = []
        last_field = inherited_field
        previous_end = 0
        for match in _NUMERIC_RE.finditer(node.text):
            suffix = node.text[match.end():match.end() + 8]
            prefix_window = node.text[max(0, match.start() - 12):match.start()]
            is_duration = re.match(r"\s*(?:毫秒|秒钟?|分钟|分|小时|时|天|日|min|h|s)", suffix, re.I)
            if is_duration and (
                re.search(r"持续|时长|保持", prefix_window)
                or (node.role == "event" and previous_end > 0)
            ):
                previous_end = match.end()
                continue
            prefix = node.text[previous_end:match.start()]
            candidate = _field_from_prefix(prefix)
            if candidate:
                last_field = candidate
            start = max(previous_end, match.start() - len(candidate or ""))
            result.append(self._make(
                "predicate", "constraint", "predicate",
                _absolute_span(node, start, match.end()), node.clause_id,
                {"field_text": last_field, "operator": _operator(match.group("op")),
                 "value": float(match.group("number"))},
            ))
            previous_end = match.end()
        return result, last_field

    def _null_requirements(self, node, inherited_field: str) -> tuple[list[RequirementSpec], str]:
        result = []
        for match in _NULL_RE.finditer(node.text):
            field = _field_from_prefix(node.text[:match.start()]) or inherited_field
            if field:
                inherited_field = field
            result.append(self._make(
                "predicate", "constraint", "predicate",
                _absolute_span(node, max(0, match.start() - len(field)), match.end()), node.clause_id,
                {"field_text": field,
                 "operator": "is_not_null" if re.search(r"不|非|not", match.group("op"), re.I) else "is_null"},
            ))
        return result, inherited_field

    def _satisfiability_requirements(self, requirements: list[RequirementSpec],
                                     query: str) -> list[RequirementSpec]:
        result = []
        by_field: dict[str, list[RequirementSpec]] = {}
        for item in requirements:
            field = str(item.expected_attributes.get("field_text", "")).strip().lower()
            if item.requirement_type == "predicate" and field:
                by_field.setdefault(field, []).append(item)
        for field, items in by_field.items():
            lower = [x for x in items if x.expected_attributes.get("operator") in {"gt", "gte"}]
            upper = [x for x in items if x.expected_attributes.get("operator") in {"lt", "lte"}]
            if not lower or not upper:
                continue
            compatible_pairs = [
                (left, right) for left in lower for right in upper
                if left.span and right.span and not re.search(
                    r"以及|或者|\bOR\b", query[
                        min(left.span.end, right.span.end):max(left.span.start, right.span.start)
                    ], re.I,
                )
            ]
            if not compatible_pairs:
                continue
            lower = [item[0] for item in compatible_pairs]
            upper = [item[1] for item in compatible_pairs]
            maximum_lower = max(float(x.expected_attributes["value"]) for x in lower)
            minimum_upper = min(float(x.expected_attributes["value"]) for x in upper)
            if maximum_lower < minimum_upper:
                continue
            spans = [x.span for x in lower + upper if x.span]
            span = SourceSpan(min(x.start for x in spans), max(x.end for x in spans), "")
            result.append(self._make(
                "satisfiability_check", "constraint", "unsatisfiable", span,
                items[0].clause_id, {"field_text": field},
            ))
        return result

    @staticmethod
    def _make(kind: str, role: str, family: str, span: SourceSpan, clause_id: str,
              attributes: dict | None = None) -> RequirementSpec:
        identity = f"{kind}|{span.start}:{span.end}|{span.text}|{clause_id}"
        return RequirementSpec(
            requirement_id="req_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12],
            requirement_type=kind, role=role, text=span.text, span=span,
            status="uncovered", critical=True, clause_id=clause_id,
            operator_family=family, expected_attributes=attributes or {},
        )

    @staticmethod
    def _dedupe(items: list[RequirementSpec]) -> list[RequirementSpec]:
        result: list[RequirementSpec] = []
        seen: dict[tuple, RequirementSpec] = {}
        for item in items:
            key = (item.requirement_type, item.span.start if item.span else -1,
                   item.span.end if item.span else -1)
            existing = seen.get(key)
            if existing is None:
                seen[key] = item
                result.append(item)
                continue
            attrs = dict(existing.expected_attributes)
            attrs.update({key: value for key, value in item.expected_attributes.items()
                          if value is not None and value != "" and value != []})
            attrs["source_demand_ids"] = sorted(set(
                existing.expected_attributes.get("source_demand_ids", [])
            ) | set(item.expected_attributes.get("source_demand_ids", [])))
            existing.expected_attributes = attrs
            existing.expected_inputs = sorted(set(existing.expected_inputs) | set(item.expected_inputs))
            existing.expected_output_fields = sorted(set(existing.expected_output_fields) | set(item.expected_output_fields))
            existing.cardinality = max(existing.cardinality, item.cardinality)
            if existing.operator_family in {"", "calculation", "aggregate"} and item.operator_family not in {"", "calculation", "aggregate"}:
                existing.operator_family = item.operator_family
            if not existing.expected_scope and item.expected_scope:
                existing.expected_scope = item.expected_scope
            if not existing.expected_output_shape and item.expected_output_shape:
                existing.expected_output_shape = item.expected_output_shape
            if not existing.expected_output_grain and item.expected_output_grain:
                existing.expected_output_grain = item.expected_output_grain
        return result


def _absolute_span(node, start: int, end: int) -> SourceSpan:
    absolute_start = node.span.start + start
    absolute_end = node.span.start + end
    return SourceSpan(absolute_start, absolute_end, node.text[start:end])


def _overlaps(left: SourceSpan, right: SourceSpan | None) -> bool:
    return bool(right and left.start < right.end and right.start < left.end)


def _field_from_prefix(prefix: str) -> str:
    role = FieldPhraseParser.classify(prefix)
    return role.text if role.role == "field" else ""


def _operator(raw: str) -> str:
    if raw in {"大于等于", "不低于", "不少于", "至少", ">="}:
        return "gte"
    if raw in {"小于等于", "不高于", "不超过", "未超过", "至多", "<="}:
        return "lte"
    if raw in {"大于", "高于", "超过", ">"}:
        return "gt"
    if raw in {"小于", "低于", "少于", "<"}:
        return "lt"
    if raw in {"不等于", "!="}:
        return "ne"
    return "eq"


def _calculation_family(raw: str) -> str:
    lowered = raw.lower()
    if "环比" in raw:
        return "mom"
    if "同比" in raw:
        return "yoy"
    if "波动率" in raw or "标准差" in raw:
        return "volatility"
    if "增长率" in raw:
        return "growth"
    if "相关" in raw:
        return "correlation"
    if "最大回撤" in raw:
        return "max_drawdown"
    if "夏普" in raw:
        return "sharpe"
    if "cagr" in lowered or "复合增长率" in raw:
        return "cagr"
    return "calculation"


def _is_output_reference(query: str, node) -> bool:
    """Recognize a result-format clause that refers to prior calculations.

    Clause splitting intentionally stays lossless, so a short continuation after
    ``最终输出:`` may be typed as a calculation clause.  It is still an output
    contract rather than an additional calculation obligation.
    """
    prefix = query[max(0, node.span.start - 96):node.span.start]
    last_output = max(prefix.rfind("输出"), prefix.rfind("返回"), prefix.rfind("列出"))
    last_boundary = max(prefix.rfind("。"), prefix.rfind("；"), prefix.rfind(";"))
    return last_output > last_boundary
