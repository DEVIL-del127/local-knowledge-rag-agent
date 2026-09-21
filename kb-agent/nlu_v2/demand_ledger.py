"""Lossless, slot-anchored source-demand ledger for the semantic compiler."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .field_roles import FieldPhraseParser
from .models import ClauseGraph, SourceDemand, SourceSpan


_COMPARISON = re.compile(
    r"(?P<operator>大于等于|不低于|不少于|至少|小于等于|不高于|不超过|未超过|至多|"
    r"大于|高于|超过|小于|低于|少于|不等于|等于|>=|<=|!=|>|<|=)\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)"
)
_UNITS = re.compile(
    r"个/平方米|W/m²|W/m2|mm/s²|mm/s2|mm/s|m/s²|m/s2|m/s|km/h|℃|°C|kPa|MPa|Pa|dB|W|kW|V|%|"
    r"美元|欧元|人民币|元|毫秒|秒钟?|分钟|分|小时|时|个交易日|交易日|天|日|次|笔|个",
    re.I,
)
_TEMPORAL = re.compile(
    r"\d{2,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:至|到|[-—~])?"
    r"(?:\d{2,4}年)?(?:\d{1,2}月(?:\d{1,2}日)?)?|今天|昨日|至今|"
    r"最近\s*\d+\s*[天日月年]|当天|当日"
)
_OPERATORS = (
    ("turn_directive", re.compile(r"停止刚才|取消(?:旧|上一个|前一个)?|上一轮|改成|改为|替换|不是.+?是|先说.+?随后")),
    ("set_operation", re.compile(r"交集|并集|差集|重合(?:时段|时间段|日期|区间)?")),
    ("sequence", re.compile(r"连续|一直|保持(?:了)?|持续|依次|先后|→|->")),
    ("cumulative_count", re.compile(r"(?:当天|当日)?累计(?:点击|浏览|查看|加购|购买|下单)?.{0,4}次数|累计次数")),
    ("period_duration", re.compile(r"(?:当日|当天|自然日)总时长")),
    ("cumulative_duration", re.compile(
        r"(?:累计|(?<!当日)(?<!当天)(?<!自然日)总|合计)"
        r"(?:[A-Za-z_\u4e00-\u9fff]{0,12})?(?:时长|耗时)"
    )),
    ("aggregate_avg", re.compile(r"日均|月均|周均|年均|日平均值?|月平均值?|周平均值?|年平均值?|平均值?|均值")),
    ("aggregate_sum", re.compile(r"总额|总和|合计|求和|总(?=[A-Za-z_\u4e00-\u9fff]{1,16}(?:金额|数量|时长))")),
    ("peak", re.compile(r"峰值|最高值?|最大值?")),
    ("quantile", re.compile(r"(?:\d{1,3}\s*)?分位数|(?<![A-Za-z])P\d{1,3}" , re.I)),
    ("correlation", re.compile(r"相关系数|相关性")),
    ("ratio", re.compile(r"比值|比例|相除|占比")),
    ("difference", re.compile(r"均值差(?:异)?|差值|相减")),
    ("group_by", re.compile(r"按\s*[^，,。；;]{1,24}?\s*分组")),
    ("top_k", re.compile(r"(?:前|top)\s*\d+\s*(?:条|个|项|名)?", re.I)),
    ("window_aggregate", re.compile(r"滑动(?:窗口)?|窗口(?:内)?平均|移动平均")),
    ("calculation", re.compile(r"计算|同比|环比|增长率|波动率|标准差|最大回撤|夏普比率")),
    ("conversion", re.compile(r"换算|折算|转换为|汇率")),
)
_OUTPUT = re.compile(r"最终输出|输出|返回|列出|提取|找出|告诉我")
# Bare “并” frequently joins analytic stages (“峰值并取 Top3”), not a
# boolean predicate.  Preserve explicit conjunctions such as 并且 / 且.
_BOOLEAN = re.compile(r"以及|并且|同时|且|而且|或者|或|但是|但|和")
_BACKTICK_FIELD = re.compile(r"`(?P<field>[A-Za-z_][A-Za-z0-9_.]*)`")
_METRIC_FIELD_PATTERNS = (
    re.compile(r"(?:日均|月均|周均|年均|平均|均值|峰值|最高值?|最大值?|最小值?|分位数|波动率|标准差|总(?!额|和))\s*\*{0,2}(?P<field>`[A-Za-z_][A-Za-z0-9_.]*`|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\*{0,2}\s*(?=以及|并且|同时|和|与|及|、|，|,|。|的相关|的比|的比例|的乘积|占|相关|并|后|$)"),
    re.compile(r"(?P<field>`[A-Za-z_][A-Za-z0-9_.]*`|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)(?:的)?\s*\*{0,2}(?:日均|月均|周均|年均|日平均值?|月平均值?|周平均值?|年平均值?|平均值?|均值|峰值|最大值?|最小值?|(?:\d{1,3}\s*)?分位数|波动率|标准差)\*{0,2}"),
    re.compile(r"按\s*(?P<field>`[A-Za-z_][A-Za-z0-9_.]*`|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,24}?)\s*分组"),
)
_CORRELATION_FIELDS = re.compile(
    r"(?P<left>`[A-Za-z_][A-Za-z0-9_.]*`|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,30}?)"
    r"\s*(?:与|和|及|、)\s*"
    r"(?P<right>`[A-Za-z_][A-Za-z0-9_.]*`|[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,30}?)"
    r"(?:的)?\s*\*{0,2}(?:相关系数|相关性)\*{0,2}"
)


@dataclass(slots=True)
class AnchorMap:
    """Stable source anchors with duplicate suppression by source identity."""

    query: str
    demands: list[SourceDemand] = field(default_factory=list)
    _keys: set[tuple[str, int, int, str]] = field(default_factory=set)

    def add(self, demand_type: str, start: int, end: int, *, clause_id: str = "",
            attributes: dict | None = None, dependencies: list[str] | None = None,
            critical: bool = True) -> SourceDemand | None:
        if start < 0 or end <= start or end > len(self.query):
            return None
        text = self.query[start:end]
        key = (demand_type, start, end, text)
        if key in self._keys:
            return next((item for item in self.demands
                         if (item.demand_type, item.span.start, item.span.end, item.text) == key), None)
        self._keys.add(key)
        identity = f"{demand_type}|{start}|{end}|{text}|{clause_id}"
        item = SourceDemand(
            demand_id="demand_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            demand_type=demand_type, text=text, span=SourceSpan(start, end, text),
            clause_id=clause_id, attributes=dict(attributes or {}),
            dependencies=list(dependencies or ()), critical=critical,
        )
        self.demands.append(item)
        return item

    def within(self, span: SourceSpan) -> list[SourceDemand]:
        return [item for item in self.demands
                if item.span.start < span.end and span.start < item.span.end]


class SourceDemandLedger:
    """Record every source obligation before Catalog binding or model repair.

    Comparisons are represented by four independently anchored slots.  The
    composite comparison records their IDs and depends on them; no property is
    allowed to be copied from an unrelated nearby span.
    """

    def build(self, query: str, graph: ClauseGraph) -> AnchorMap:
        anchors = AnchorMap(query)
        for node in graph.nodes:
            if node.instruction_scope != "active":
                continue
            self._node_demands(anchors, node.text, node.span.start, node.clause_id)
        for edge in graph.edges:
            if (edge.relation.lower() in {"and", "or", "not", "and_not", "then"}
                    and edge.span and edge.span.text.strip()):
                anchors.add("boolean", edge.span.start, edge.span.end, clause_id="",
                            attributes={"relation": edge.relation})
        return anchors

    def _node_demands(self, anchors: AnchorMap, text: str, offset: int, clause_id: str) -> None:
        for match in _BACKTICK_FIELD.finditer(text):
            anchors.add("field", offset + match.start(), offset + match.end(), clause_id=clause_id,
                        attributes={"field_text": match.group("field"), "explicit_schema": True})
        for pattern in _METRIC_FIELD_PATTERNS:
            for match in pattern.finditer(text):
                raw_field = match.group("field")
                scoped_field = re.sub(
                    r"^.*?(?:日期|时间段|时段|区间|窗口)内", "", raw_field,
                )
                relative = raw_field.rfind(scoped_field) if scoped_field else 0
                role = FieldPhraseParser.classify(
                    scoped_field or raw_field, match.start("field") + max(relative, 0),
                )
                if role.role != "field" or not role.text or role.text == "值":
                    continue
                if re.search(r"是否|高于|低于|超过|不超过|全站|总体|筛选|这些|上述", role.text):
                    # Scope/comparison language to the left of a suffix
                    # aggregate is not part of its measure field.
                    continue
                formula_suffix = re.search(
                    r"(?:波动率|标准差|相关系数|相关性|增长率|同比|环比)$", role.text,
                )
                if formula_suffix:
                    raw_field = role.text[:formula_suffix.start()].rstrip("的 ")
                    if not raw_field:
                        continue
                    role = type(role)(
                        role.role, raw_field, role.start,
                        role.start + len(raw_field), raw_field,
                    )
                anchors.add("field", offset + role.start, offset + role.end, clause_id=clause_id,
                            attributes={"field_text": role.normalized or role.text})

        for match in _CORRELATION_FIELDS.finditer(text):
            for group in ("left", "right"):
                raw = match.group(group).strip("` ")
                value = re.sub(
                    r"^.*?(?:日期|时间段|时段|区间|窗口)内", "", raw,
                )
                role = FieldPhraseParser.classify(value)
                if role.role != "field" or not role.text:
                    continue
                relative = match.group(group).rfind(role.text)
                start = match.start(group) + max(relative, 0)
                anchors.add(
                    "field", offset + start, offset + start + len(role.text),
                    clause_id=clause_id,
                    attributes={"field_text": role.normalized or role.text},
                )

        # Action enums are values, not numeric fields.  Keep them as explicit
        # demands so a future event/count operator must consume them.
        for action in _action_mentions(text):
            anchors.add("action_value", offset + action.start, offset + action.end, clause_id=clause_id,
                        attributes={"value": action.normalized})

        for match in _COMPARISON.finditer(text):
            self._comparison_demands(anchors, text, offset, clause_id, match)

        for match in _TEMPORAL.finditer(text):
            anchors.add("temporal", offset + match.start(), offset + match.end(), clause_id=clause_id,
                        attributes={"raw": match.group(0)})
        schema_spans = [match.span() for match in re.finditer(
            r"`[A-Za-z_][A-Za-z0-9_.]*`\s*[（(][^）)]*[）)]", text,
        )]
        for family, pattern in _OPERATORS:
            for match in pattern.finditer(text):
                if any(start <= match.start() < end for start, end in schema_spans):
                    continue
                if family == "aggregate_avg" and re.match(
                    r"(?:(?!与|和|及|、|以及|并且|的)[^，,。；;]){0,24}?"
                    r"(?:波动率|标准差|相关系数|相关性|增长率|同比|环比)",
                    text[match.end():],
                ):
                    continue
                if family == "aggregate_avg" and match.group(0) == "平均" and re.match(
                    r"[^，,。；;]{1,24}?(?:超过|高于|大于|低于|小于|等于|不超过)",
                    text[match.end():],
                ):
                    continue
                context = text[max(0, match.start() - 36):match.start()]
                effective_family = family
                window_match = None
                if family == "aggregate_avg":
                    window_match = re.search(
                        r"(?P<size>\d+(?:\.\d+)?)\s*$",
                        text[:match.start()],
                    )
                    if window_match and re.match(r"(?:线|量)", text[match.end():]):
                        effective_family = "window_aggregate"
                attributes = {"operator_family": effective_family,
                              **_operator_attributes(effective_family, match.group(0))}
                if window_match is not None and effective_family == "window_aggregate":
                    attributes.update({
                        "window_size": float(window_match.group("size")),
                        "window_unit": "日",
                        "function": "avg",
                    })
                if re.search(r"全(?:站|局|部|体|量)", context):
                    attributes["scope"] = "global"
                elif re.search(r"这些|上述|筛选|满足|交集|并集|重合", context):
                    attributes["scope"] = "filtered"
                operator = anchors.add(
                    "operator", offset + match.start(), offset + match.end(),
                    clause_id=clause_id, attributes=attributes,
                )
                if operator is not None and effective_family in {
                    "aggregate_avg", "aggregate_sum", "peak", "quantile",
                }:
                    fields = [item for item in anchors.demands
                              if item.demand_type == "field" and item.clause_id == clause_id]
                    nearest = min(fields, key=lambda item: min(
                        abs(item.span.end - (offset + match.start())),
                        abs(item.span.start - (offset + match.end())),
                    ), default=None)
                    if nearest is not None and min(
                        abs(nearest.span.end - (offset + match.start())),
                        abs(nearest.span.start - (offset + match.end())),
                    ) <= 40:
                        operator.dependencies.append(nearest.demand_id)
                if operator is not None and family == "correlation":
                    fields = [
                        item for item in anchors.demands
                        if item.demand_type == "field"
                        and item.clause_id == clause_id
                        and item.span.end <= offset + match.start()
                        and offset + match.start() - item.span.end <= 64
                    ]
                    for item in fields[-2:]:
                        if item.demand_id not in operator.dependencies:
                            operator.dependencies.append(item.demand_id)
        for match in _OUTPUT.finditer(text):
            anchors.add("output", offset + match.start(), offset + match.end(), clause_id=clause_id)
        for match in _BOOLEAN.finditer(text):
            # In a qualitative comparison request (for example comparing
            # methods and conclusions), conjunction joins requested output
            # facets rather than data predicates.  With no field/comparison
            # anchor in the clause it must not create an uncovered filter.
            if re.search(r"比较|对比|compare", text, re.I) and not any(
                item.clause_id == clause_id and item.demand_type in {
                    "field", "comparison", "comparison_operator"
                } for item in anchors.demands
            ):
                continue
            anchors.add("boolean", offset + match.start(), offset + match.end(), clause_id=clause_id,
                        attributes={"relation": _boolean_relation(match.group(0))})

    def _comparison_demands(self, anchors: AnchorMap, text: str, offset: int,
                            clause_id: str, match: re.Match[str]) -> None:
        unit_start = match.end()
        while unit_start < len(text) and text[unit_start].isspace():
            unit_start += 1
        unit_match = _UNITS.match(text, unit_start)
        unit = unit_match.group(0) if unit_match else ""
        duration_units = {
            "毫秒", "秒", "秒钟", "分钟", "分", "小时", "时",
            "个交易日", "交易日", "天", "日",
        }
        prefix = text[max(0, match.start() - 32):match.start()]
        duration_threshold = unit in duration_units and bool(re.search(
            r"连续|持续|总时长|累计时长|合计时长|时长", prefix
        ))
        count_threshold = unit in {"次", "个", "笔"} and bool(re.search(
            r"连续|持续|累计|次数|连续订单", prefix
        ))

        role = FieldPhraseParser.field_before(text, match.start())
        field_anchor = None
        if not duration_threshold and not count_threshold and role.role == "field" and role.text:
            field_anchor = anchors.add(
                "field", offset + role.start, offset + role.end, clause_id=clause_id,
                attributes={"field_text": role.normalized or role.text},
            )
        elif (not duration_threshold and not count_threshold
              and role.role == "derived_operator" and role.text):
            field_anchor = anchors.add(
                "derived_metric", offset + role.start, offset + role.end,
                clause_id=clause_id,
                attributes={
                    "field_text": role.normalized or role.text,
                    "derived_operator": role.normalized or role.text,
                },
            )
        if not duration_threshold and not count_threshold and field_anchor is None:
            # Elliptical bounds ("温度超过80℃但未超过85℃") inherit only a
            # preceding field anchor in the same clause.  This is provenance,
            # not Catalog binding, and never crosses clause boundaries.
            field_anchor = next((
                item for item in reversed(anchors.demands)
                if item.demand_type == "field" and item.clause_id == clause_id
                and item.span.end <= offset + match.start()
            ), None)
        if not duration_threshold and not count_threshold and field_anchor is None:
            global_start = offset + match.start()
            prior = next((
                item for item in reversed(anchors.demands)
                if item.demand_type == "field" and item.span.end <= global_start
                and global_start - item.span.end <= 160
            ), None)
            if prior is not None:
                bridge = anchors.query[prior.span.end:global_start]
                if "。" not in bridge and re.search(r"累计|总时长|以及|并且|且|但", bridge):
                    field_anchor = prior
                    inherited_field = True
                else:
                    inherited_field = False
            else:
                inherited_field = False
        else:
            inherited_field = False
        operator_anchor = anchors.add(
            "comparison_operator", offset + match.start("operator"), offset + match.end("operator"),
            clause_id=clause_id, attributes={"operator": _operator(match.group("operator"))},
        )
        quantity = anchors.add(
            "quantity", offset + match.start("value"), offset + match.end("value"), clause_id=clause_id,
            attributes={"value": float(match.group("value"))},
        )
        unit_anchor = None
        end = offset + match.end()
        if unit_match:
            unit_anchor = anchors.add(
                "unit", offset + unit_match.start(), offset + unit_match.end(), clause_id=clause_id,
                attributes={"raw_unit": unit}, dependencies=[quantity.demand_id] if quantity else [],
            )
            end = offset + unit_match.end()
        dependencies = [item.demand_id for item in (field_anchor, operator_anchor, quantity, unit_anchor) if item]
        comparison = anchors.add(
            ("duration_threshold" if duration_threshold else
             "count_threshold" if count_threshold else "comparison"),
            offset + match.start(), end, clause_id=clause_id,
            attributes={
                "operator": _operator(match.group("operator")),
                "value": float(match.group("value")), "unit": unit,
                "field_text": field_anchor.attributes.get("field_text", "") if field_anchor else "",
                "inherited_field": inherited_field,
            }, dependencies=dependencies,
        )
        if comparison:
            comparison.field_anchor_id = field_anchor.demand_id if field_anchor else ""
            comparison.operator_anchor_id = operator_anchor.demand_id if operator_anchor else ""
            comparison.quantity_anchor_id = quantity.demand_id if quantity else ""
            comparison.unit_anchor_id = unit_anchor.demand_id if unit_anchor else ""


def _action_mentions(text: str):
    # ``action_value`` returns the first match.  Advance deliberately to retain
    # all actions in a conjunction without allowing zero-length loops.
    offset = 0
    while offset < len(text):
        role = FieldPhraseParser.action_value(text[offset:])
        if role is None:
            return
        yield type(role)(role.role, role.text, role.start + offset, role.end + offset, role.normalized)
        offset += max(role.end, 1)


def _operator_attributes(family: str, text: str) -> dict[str, object]:
    values: dict[str, object] = {}
    if family == "top_k":
        number = re.search(r"\d+", text)
        values["limit"] = int(number.group(0)) if number else None
    if family == "quantile":
        number = re.search(r"\d+", text)
        values["quantile"] = int(number.group(0)) if number else None
    if family == "aggregate_avg":
        values["function"] = "avg"
        grains = {"日": "day", "天": "day", "月": "month", "周": "week", "年": "year"}
        group_by_grain = next((value for marker, value in grains.items()
                               if text.startswith(marker)), "")
        if group_by_grain:
            values["group_by_grain"] = group_by_grain
    elif family == "aggregate_sum":
        values["function"] = "sum"
    elif family == "peak":
        values["function"] = "max"
    elif family == "correlation":
        values["function"] = "correlation"
    elif family == "ratio":
        values["function"] = "ratio"
    elif family == "difference":
        values["function"] = "difference"
    return values


def _operator(raw: str) -> str:
    if raw in {"小于", "低于", "少于", "<"}:
        return "lt"
    if raw in {"小于等于", "不高于", "不超过", "未超过", "至多", "<="}:
        return "lte"
    if raw in {"大于等于", "不低于", "不少于", "至少", ">="}:
        return "gte"
    if raw in {"不等于", "!="}:
        return "ne"
    if raw in {"等于", "="}:
        return "eq"
    return "gt"


def _boolean_relation(raw: str) -> str:
    if raw in {"或者", "或"}:
        return "or"
    if raw in {"但是", "但"}:
        # Negation is already encoded by the following comparison operator
        # (for example 未超过 -> lte); the connector itself remains AND.
        return "and"
    return "and"
