"""Deterministic extraction of reusable atomic query syntax."""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

try:
    # Installed distribution: legacy pure-rule helpers live in a namespaced
    # compatibility package so they cannot collide with unrelated top-level modules.
    from nlu_legacy.cleaner import clean
    from nlu_legacy.nlu import parse_time_ranges, parse_year_month_matrix
except ImportError:
    # Source checkout compatibility for the historical kb-agent test harness.
    from cleaner import clean
    from nlu import parse_time_ranges, parse_year_month_matrix

from .catalog import CatalogSnapshot, FieldMatch
from .field_roles import FieldPhraseParser
from .models import (
    CalculationSpec,
    AggregateSpec,
    ComparisonSpec,
    CandidateRef,
    DerivedMetricCall,
    DurationConstraint,
    DerivedProjectionSpec,
    EventSpec,
    GoalSpec,
    LiteralValue,
    MetricSpec,
    OutputContract,
    OutputRef,
    PredicateNode,
    Provenance,
    QueryEnvelope,
    ReferenceSpec,
    SamplingPolicy,
    SchemaHypothesis,
    ScopedAggregateSpec,
    ExpressionNode,
    SetOperationSpec,
    SourceSpan,
    SymbolRef,
    TemporalItem,
    UnderstandingIR,
    UnresolvedItem,
    WindowSpec,
)


_GOAL_PATTERNS = {
    "retrieve": re.compile(r"查询|查找|找出|找一下|提取|列出|哪些|数据|文献|论文", re.I),
    "aggregate": re.compile(r"总额|总和|合计|平均|均值|最大|最小|多少|几篇|数量|统计", re.I),
    "compare": re.compile(r"对比|比较|\bvs\.?\b|同比|环比|相比", re.I),
    "compute": re.compile(r"计算|增长率|同比|环比|差值|波动率|公式", re.I),
    "present": re.compile(r"最终输出|输出|返回|展示|告诉我", re.I),
}

_AGGREGATIONS = (
    (re.compile(r"总额|总和|合计|求和"), "sum"),
    (re.compile(r"平均|均值"), "avg"),
    (re.compile(r"最大|最高"), "max"),
    (re.compile(r"最小|最低"), "min"),
    (re.compile(r"多少|几篇|数量|计数|统计"), "count"),
)

_OPERATOR_PATTERNS = (
    (re.compile(r"(?:介于|在)?\s*(-?\d+(?:\.\d+)?)\s*(?:到|至|~|～|-)\s*(-?\d+(?:\.\d+)?)"), "between"),
    (re.compile(r"(?:大于等于|不低于|不少于|至少|>=|≥)\s*(-?\d+(?:\.\d+)?)"), "gte"),
    (re.compile(r"(?:小于等于|不高于|不超过|至多|<=|≤)\s*(-?\d+(?:\.\d+)?)"), "lte"),
    (re.compile(r"(?:大于|高于|超过|>)\s*(-?\d+(?:\.\d+)?)"), "gt"),
    (re.compile(r"(?:小于|低于|少于|<)\s*(-?\d+(?:\.\d+)?)"), "lt"),
    (re.compile(r"(?:不等于|不是|!=|≠)\s*(-?\d+(?:\.\d+)?)"), "ne"),
    (re.compile(r"(?:等于|为|=)\s*(-?\d+(?:\.\d+)?)"), "eq"),
    (re.compile(r"^\s*(-?\d+(?:\.\d+)?)"), "eq"),
)

_UNIT_PATTERNS = (
    (re.compile(r"摄氏度|℃|°C", re.I), "celsius"),
    (re.compile(r"华氏度|℉|°F", re.I), "fahrenheit"),
    (re.compile(r"W/m(?:²|2)", re.I), "watt_per_square_meter"),
    (re.compile(r"mm/s", re.I), "mm/s"),
    (re.compile(r"m/s(?:²|2)", re.I), "m/s^2"),
    (re.compile(r"米每秒|m/s", re.I), "m/s"),
    (re.compile(r"公里每小时|km/h", re.I), "km/h"),
    (re.compile(r"MPa", re.I), "megapascal"),
    (re.compile(r"kPa", re.I), "kilopascal"),
    (re.compile(r"Pa", re.I), "pascal"),
    (re.compile(r"dB", re.I), "decibel"),
    (re.compile(r"MWh", re.I), "megawatt_hour"),
    (re.compile(r"MW", re.I), "megawatt"),
    (re.compile(r"kW", re.I), "kilowatt"),
    (re.compile(r"W", re.I), "watt"),
    (re.compile(r"kV", re.I), "kilovolt"),
    (re.compile(r"V", re.I), "volt"),
    (re.compile(r"A", re.I), "ampere"),
    (re.compile(r"m(?:³|3)/h", re.I), "cubic_meter_per_hour"),
    (re.compile(r"bps", re.I), "basis_point"),
    (re.compile(r"ms", re.I), "millisecond"),
    (re.compile(r"μm|um", re.I), "micrometer"),
    (re.compile(r"km", re.I), "kilometer"),
    (re.compile(r"m", re.I), "meter"),
    (re.compile(r"吨"), "tonne"),
    (re.compile(r"万元"), "currency"),
    (re.compile(r"毫米|mm", re.I), "mm"),
    (re.compile(r"百分比|百分之|%"), "percent"),
    (re.compile(r"人民币|元|CNY", re.I), "currency"),
)

_REFERENCE_RE = re.compile(r"该日期|上述结果|前一步|上一步|这两个窗口|前述数据|这些结果")
_WRITE_OPERATION_RE = re.compile(r"删除|移除|更新|修改|写入|新增|添加|上传|导入|清空|重建(?:索引|数据库|知识库)")
_UNKNOWN_NUMERIC_FIELD_RE = re.compile(
    r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{1,24}?)\s*"
    r"(?P<op>大于等于|不低于|不少于|至少|小于等于|不高于|不超过|至多|大于|高于|超过|小于|低于|少于|不等于|等于|>=|<=|!=|>|<|=)\s*"
    r"(?P<number>-?\d+(?:\.\d+)?)"
)
_DURATION_UNIT_SECONDS = {
    "毫秒": 0.001,
    "秒": 1.0, "秒钟": 1.0, "s": 1.0,
    "分钟": 60.0, "分": 60.0, "min": 60.0,
    "小时": 3600.0, "时": 3600.0, "h": 3600.0,
    "天": 86400.0, "日": 86400.0,
    "交易日": 86400.0, "个交易日": 86400.0,
}
_CONTINUOUS_EVENT_RE = re.compile(
    r"连续\s*(?:大于|高于|超过|>)\s*(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<value_unit>摄氏度|华氏度|℃|℉|°C|°F)?\s*"
    r"(?:持续)?(?:大于|超过|达)\s*(?P<duration>\d+(?:\.\d+)?)\s*"
    r"(?P<duration_unit>秒钟?|分钟|分|小时|时|天|日|min|h|s)", re.I,
)
_CUMULATIVE_EVENT_RE = re.compile(
    r"累计\s*(?:大于|高于|超过|>)\s*(?P<lower>-?\d+(?:\.\d+)?)\s*"
    r"(?P<lower_unit>摄氏度|华氏度|℃|℉|°C|°F)?\s*"
    r"(?:但)?\s*(?:未超过|不超过|小于等于|<=|≤)\s*(?P<upper>-?\d+(?:\.\d+)?)\s*"
    r"(?P<upper_unit>摄氏度|华氏度|℃|℉|°C|°F)?\s*"
    r"(?:的)?\s*(?:累计|总)?时长\s*(?:大于|超过|>)\s*"
    r"(?P<duration>\d+(?:\.\d+)?)\s*"
    r"(?P<duration_unit>秒钟?|分钟|分|小时|时|天|日|min|h|s)", re.I,
)
_CONTINUITY_SIGNAL_RE = re.compile(r"连续|持续|一直|保持")
_INLINE_FIELD_RE = re.compile(
    r"`(?P<name>[A-Za-z_][A-Za-z0-9_.]*)`"
    r"(?:\s*[（(](?P<description>[^）)]+)[）)])?"
)
_SAMPLING_RE = re.compile(
    r"每\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>毫秒|秒|分钟|分|小时|时)"
    r"(?:记录|采样|上报)?(?:一次)?"
)
_GENERIC_CONTINUOUS_EVENT_RE = re.compile(
    r"(?P<field>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,30}?"
    r"(?:\s+[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,20})?)\s*"
    r"连续\s*(?P<op>大于等于|不低于|超过|大于|高于|低于|小于|>=|<=|>|<)\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<value_unit>L/100km|m(?:³|3)/h|个/平方米|W/m(?:²|2)|mm/s|m/s(?:²|2)?|km/h|"
    r"kV|V|kPa|Pa|MPa|dB|MWh|MW|kW|W|℃|°C|%|bps|ms|μm|um|km|m|吨|万元|元|A|"
    r"次/分钟|条/分钟|辆/分钟|车/分钟|次|手)?\s*"
    r"(?:[（(][^）)]*[）)])?\s*(?:且)?\s*(?:持续)?\s*"
    r"(?:超过|大于|不少于|至少|>)\s*(?P<duration>\d+(?:\.\d+)?)\s*"
    r"(?P<duration_unit>毫秒|秒钟?|分钟|分|小时|时|天|日|个?交易日|min|h|s)", re.I,
)


def _is_semantic_control_query(query: str) -> bool:
    """Recognize compiler-control and quoted-content safety messages.

    The business words inside these messages are examples or quoted document
    content.  Treating them as an implicit Catalog request would violate the
    query-local schema isolation rule.
    """
    turn_revision = bool(re.search(r"同一轮|最终\s*IR", query)) and bool(re.search(
        r"不是.+?是|先说.+?随后|只能保留", query,
    ))
    quoted_content = bool(re.search(
        r"检索到(?:的)?文档包含(?:文本)?[“\"].*(?:忽略系统指令|调用删除工具|输出密钥)", query,
    )) and bool(re.search(r"(?:不能变成|只能作为).*(?:Agent指令|待总结内容)", query))
    return turn_revision or quoted_content


@dataclass(slots=True)
class RuleExtraction:
    ir: UnderstandingIR
    evidence_nodes: int = 0


class AtomicRuleExtractor:
    """Extract syntax that is independent of any hard-coded business domain."""

    def extract(self, query: str, catalog: CatalogSnapshot) -> RuleExtraction:
        normalized = clean(query).cleaned
        ir = UnderstandingIR(
            query=QueryEnvelope(raw=query, normalized=normalized),
            catalog_version=catalog.version,
        )
        ir.goals = self._goals(normalized)
        # These messages ask the compiler to represent a turn correction or a
        # quoted-content safety decision.  They are not a request to retrieve
        # the business terms mentioned as examples, so Catalog candidate names
        # must not create a spurious source-binding obligation.
        semantic_control = _is_semantic_control_query(normalized)
        if semantic_control:
            ir.goals = []
        ir.schema_hypotheses = self._schema_hypotheses(normalized)
        mentions = catalog.find_field_mentions(normalized)
        if semantic_control:
            mentions = []
        for mention in mentions:
            owners = [
                item for item in ir.schema_hypotheses
                if any(mention.alias.lower() in alias.lower() for alias in item.aliases)
            ]
            if len(owners) == 1 and mention.alias not in owners[0].aliases:
                owners[0].aliases.append(mention.alias)
        hypothesis_symbols = self._hypothesis_symbols(normalized, ir.schema_hypotheses)
        hypothesis_aliases = {
            alias.lower()
            for item in ir.schema_hypotheses
            for alias in item.aliases
            if alias
        }
        # Inline schema declarations are evidence for an unbound hypothesis, not
        # proof that an identically named field belongs to a Catalog source.
        mentions = [item for item in mentions if item.alias.lower() not in hypothesis_aliases]
        local_schema = catalog.query_schema(
            normalized, hypotheses=[item.hypothesis_id for item in ir.schema_hypotheses],
        )
        ir.query_schema = local_schema
        preferred_sources = list(local_schema.source_ids)
        source_spans = _source_alias_spans(normalized, catalog)
        mentions = [
            item for item in mentions
            if not any(start <= item.start and item.end <= end for start, end in source_spans)
        ]
        symbols = [
            self._symbol_for_mention(item, catalog, preferred_sources, set(local_schema.field_ids))
            for item in mentions
        ]
        ir.projections = _dedupe_symbols(symbols + self._requested_hypothesis_projections(
            normalized, hypothesis_symbols
        ))
        ir.source_candidates = self._source_candidates(normalized, symbols, catalog)
        ir.metrics = self._metrics(normalized, mentions, symbols, catalog)
        ir.temporal = self._temporal(normalized)
        ir.sampling_policies = self._sampling_policies(normalized)
        all_symbols = symbols + hypothesis_symbols
        ir.events = self._events(
            normalized, mentions, all_symbols, ir.temporal, catalog,
            ir.sampling_policies[0] if ir.sampling_policies else None,
        )
        ir.derived_projections = self._date_projections(ir.events)
        ir.set_operations = self._set_operations(normalized, ir.events, ir.derived_projections)
        if ir.set_operations:
            ir.projections = [item for item in ir.projections if item.raw_name not in {"日期", "时间"}]
        if (_CONTINUITY_SIGNAL_RE.search(normalized) and not any(
            item.derived_metric.metric_id.endswith(".consecutive_duration") for item in ir.events
        )):
            match = _CONTINUITY_SIGNAL_RE.search(normalized)
            ir.unresolved.append(UnresolvedItem(
                code="event_structure_incomplete",
                message="连续事件缺少可绑定的条件或持续时长",
                span=_span(normalized, match.start(), match.end()) if match else None,
            ))
        if re.search(r"累计.*?时长", normalized) and not any(
            item.derived_metric.metric_id.endswith(".cumulative_duration") for item in ir.events
        ):
            match = re.search(r"累计", normalized)
            ir.unresolved.append(UnresolvedItem(
                code="event_structure_incomplete",
                message="累计事件缺少可绑定的条件、统计窗口或总时长",
                span=_span(normalized, match.start(), match.end()) if match else None,
            ))
        if "交集" in normalized and not ir.set_operations:
            match = re.search(r"交集", normalized)
            ir.unresolved.append(UnresolvedItem(
                code="set_inputs_unbound",
                message="集合交集缺少至少两个已解析事件输出",
                span=_span(normalized, match.start(), match.end()) if match else None,
            ))
        covered_ranges = [
            (item.provenance[0].span.start, item.provenance[0].span.end)
            for item in ir.events if item.provenance and item.provenance[0].span
        ]
        predicates, unresolved = self._predicates(
            normalized, mentions, symbols, catalog, ignored_ranges=covered_ranges
        )
        hypothesis_predicates = self._hypothesis_predicates(
            normalized, hypothesis_symbols, covered_ranges
        )
        hypothesis_ranges = [
            (item.provenance[0].span.start, item.provenance[0].span.end)
            for item in hypothesis_predicates
            if item.provenance and item.provenance[0].span
        ]
        predicates = [
            item for item in predicates
            if not (
                item.provenance and item.provenance[0].span
                and any(item.provenance[0].span.start < end and start < item.provenance[0].span.end
                        for start, end in hypothesis_ranges)
            )
        ]
        hypothesis_names = {item.raw_name.lower() for item in hypothesis_symbols}
        predicates = [
            item for item in predicates
            if not (item.field and item.field.status == "unresolved"
                    and item.field.raw_name.lower() in hypothesis_names)
        ]
        unresolved = [
            item for item in unresolved
            if not (item.span and item.span.text.strip().lower() in hypothesis_names)
        ]
        predicates.extend(hypothesis_predicates)
        predicates.extend(self._catalog_categorical_shortcuts(normalized, catalog, predicates))
        ir.filters = self._boolean_tree(normalized, predicates)
        ir.unresolved.extend(unresolved)
        ir.calculations = self._calculations(
            normalized, all_symbols, ir.set_operations, catalog
        )
        for calculation in ir.calculations:
            if calculation.parameters.get("formula_gap") == "input_field_unbound":
                ir.unresolved.append(UnresolvedItem(
                    code="formula_input_unbound",
                    message="公式缺少原文明确修饰的输入字段",
                    span=next((item.span for item in calculation.provenance if item.span), None),
                ))
        ir.aggregates, ir.comparisons = self._scoped_aggregates_and_comparisons(
            normalized, all_symbols
        )
        ir.references = self._references(normalized, ir.set_operations)
        write_match = _active_write_match(normalized)
        if write_match:
            ir.unresolved.append(UnresolvedItem(
                code="write_operation",
                message="第一阶段禁止写入或修改数据源",
                span=_span(normalized, write_match.start(), write_match.end()),
            ))
        ir.output = self._output(normalized, ir)
        ir.compatibility_intent = self._compatibility_intent(ir)
        evidence = (
            len(mentions) + len(predicates) + len(ir.temporal)
            + len(ir.events) + len(ir.set_operations) + len(ir.calculations)
        )
        return RuleExtraction(ir=ir, evidence_nodes=evidence)

    @staticmethod
    def _schema_hypotheses(query: str) -> list[SchemaHypothesis]:
        result = []
        seen = set()
        for match in _INLINE_FIELD_RE.finditer(query):
            name = match.group("name")
            if name in seen:
                continue
            seen.add(name)
            description = (match.group("description") or "").strip()
            aliases = [name]
            label_match = re.search(
                r"(?P<label>[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,20}"
                r"(?:\s+[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]{0,20})?)\s*[（(]\s*`"
                + re.escape(name) + r"`", query
            )
            if label_match:
                raw_label = re.sub(
                    r"^(?:包含|以及|并且|同时|和|与|及|、)+", "",
                    label_match.group("label"),
                )
                cleaned_label = _clean_field_phrase(raw_label) or raw_label
                if cleaned_label:
                    aliases.append(cleaned_label)
            plain_label = re.search(
                r"(?:^|[、，,；;。\s])(?P<label>[\u4e00-\u9fff]{1,10})\s*`"
                + re.escape(name) + r"`", query
            )
            if plain_label:
                label = _clean_field_phrase(plain_label.group("label"))
                if label not in {"包含", "记录", "字段", "数据", "每秒记录"}:
                    aliases.append(label)
            if description:
                aliases.extend(
                    item.strip() for item in re.split(r"[，,、/]", description)
                    if item.strip()
                )
            declared_type = "string" if re.search(r"(?:^|_)id$", name, re.I) else "number"
            unit = "currency" if re.search(r"金额|消费|spend|price|cost", description + name, re.I) else None
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            result.append(SchemaHypothesis(
                hypothesis_id=f"hyp:{digest}", raw_name=name,
                normalized_name=name.lower(), declared_type=declared_type,
                description=description, unit=unit, aliases=list(dict.fromkeys(aliases)),
                span=_span(query, match.start(), match.end()), executable=False,
            ))
        return result

    @staticmethod
    def _hypothesis_symbols(query: str, hypotheses: list[SchemaHypothesis]) -> list[SymbolRef]:
        result = []
        for item in hypotheses:
            aliases = sorted(item.aliases, key=len, reverse=True)
            for alias in aliases:
                start = query.find(alias)
                if start < 0:
                    continue
                result.append(SymbolRef(
                    raw_name=alias, canonical_id=None,
                    candidates=[CandidateRef(item.hypothesis_id, source="query_schema", status="candidate")],
                    status="candidate", span=_span(query, start, start + len(alias)),
                    ref_kind="hypothesis",
                ))
        return _dedupe_symbols(result)

    @staticmethod
    def _requested_hypothesis_projections(query: str,
                                          symbols: list[SymbolRef]) -> list[SymbolRef]:
        marker = re.search(r"(?:最终)?输出[：:]?", query)
        if not marker:
            return []
        tail = re.split(r"并(?:且)?计算|计算", query[marker.end():], maxsplit=1)[0]
        result = [item for item in symbols if item.raw_name in tail]
        for item in symbols:
            spaced = item.raw_name.replace("_", " ")
            if spaced.lower() in tail.lower() and item not in result:
                result.append(item)
        id_symbols = [item for item in symbols if re.search(r"(?:^|_)id$", item.raw_name, re.I)]
        if re.search(r"\bID\b", tail, re.I) and len(id_symbols) == 1 and id_symbols[0] not in result:
            result.append(id_symbols[0])
        return result

    @staticmethod
    def _sampling_policies(query: str) -> list[SamplingPolicy]:
        match = _SAMPLING_RE.search(query)
        if not match:
            return []
        factors = {"毫秒": 0.001, "秒": 1, "分钟": 60, "分": 60, "小时": 3600, "时": 3600}
        seconds = float(match.group("value")) * factors[match.group("unit")]
        return [SamplingPolicy(expected_interval_seconds=seconds)]

    def _hypothesis_predicates(self, query: str, symbols: list[SymbolRef],
                               ignored_ranges: list[tuple[int, int]]) -> list[PredicateNode]:
        result = []
        used = set()
        for symbol in sorted(symbols, key=lambda item: len(item.raw_name), reverse=True):
            if symbol.raw_name in used:
                continue
            for match in re.finditer(re.escape(symbol.raw_name), query, re.I):
                suffix = query[match.end():match.end() + 48]
                values = self._operator_values(suffix)
                if not values:
                    continue
                operator, numbers, op_start, op_end = values[0]
                start, end = match.start(), match.end() + op_end
                if any(start < old_end and end > old_start for old_start, old_end in ignored_ranges):
                    continue
                raw_unit, unit = self._unit(query[match.end() + op_start:end + 10])
                result.append(PredicateNode(
                    operator=operator, field=symbol,
                    value=LiteralValue(numbers if operator == "between" else numbers[0],
                                       "range" if operator == "between" else "number",
                                       unit=unit, raw_unit=raw_unit),
                    provenance=[Provenance(source="rule", rule="hypothesis_predicate",
                                           span=_span(query, start, end))],
                ))
                used.add(symbol.raw_name)
                break
        return result

    @staticmethod
    def _goals(query: str) -> list[GoalSpec]:
        goals: list[GoalSpec] = []
        for goal_type, pattern in _GOAL_PATTERNS.items():
            match = pattern.search(query)
            if match:
                goals.append(GoalSpec(goal_type, _span(query, match.start(), match.end())))
        if not goals:
            goals.append(GoalSpec("retrieve"))
        if any(goal.goal_type in {"aggregate", "compare", "compute"} for goal in goals):
            if not any(goal.goal_type == "retrieve" for goal in goals):
                goals.insert(0, GoalSpec("retrieve"))
        return goals

    @staticmethod
    def _symbol_for_mention(mention: FieldMatch, catalog: CatalogSnapshot,
                            preferred_sources: list[str], explicit_field_ids: set[str] | None = None) -> SymbolRef:
        field_ids = list(mention.field_ids)
        explicit_field_ids = explicit_field_ids or set()
        explicitly_named = [item for item in field_ids if item in explicit_field_ids]
        if explicitly_named:
            field_ids = explicitly_named
        elif len(preferred_sources) == 1:
            scoped = [
                item for item in field_ids
                if catalog.source_for_field(item) == preferred_sources[0]
            ]
            if scoped:
                field_ids = scoped
        # Generic Catalog aliases are never an implicit resolved binding.  An
        # explicit query field or a uniquely named query source is required.
        resolved = bool(explicitly_named or (len(preferred_sources) == 1 and len(field_ids) == 1))
        candidates = [CandidateRef(item, source="catalog", status="resolved" if resolved else "candidate")
                      for item in field_ids]
        return SymbolRef(
            raw_name=mention.alias,
            canonical_id=field_ids[0] if resolved and len(field_ids) == 1 else None,
            candidates=candidates,
            status="resolved" if resolved and len(field_ids) == 1 else "candidate",
            span=SourceSpan(mention.start, mention.end, mention.alias),
        )

    @staticmethod
    def _source_candidates(query: str, symbols: Iterable[SymbolRef],
                           catalog: CatalogSnapshot) -> list[CandidateRef]:
        symbols = list(symbols)
        source_ids = catalog.source_alias_matches(query)
        source_ids.extend(
            source_id for symbol in symbols
            if symbol.canonical_id
            for source_id in [catalog.source_for_field(symbol.canonical_id)]
            if source_id
        )
        owner_sets = [
            {catalog.source_for_field(candidate.identifier) for candidate in symbol.candidates
             if catalog.source_for_field(candidate.identifier)}
            for symbol in symbols
        ]
        owner_sets = [owners for owners in owner_sets if owners]
        if len(owner_sets) > 1 and not set.intersection(*owner_sets):
            # The query explicitly combines fields whose Catalog owners cannot
            # belong to one source.  Surface the complete read-only source set
            # so the validator can report unsupported multi-source semantics.
            source_ids.extend(source_id for owners in owner_sets for source_id in sorted(owners))
        return [CandidateRef(item, source="catalog") for item in dict.fromkeys(source_ids)]

    @staticmethod
    def _metrics(query: str, mentions: list[FieldMatch], symbols: list[SymbolRef],
                 catalog: CatalogSnapshot) -> list[MetricSpec]:
        metrics: list[MetricSpec] = []
        for mention, symbol in zip(mentions, symbols):
            window = query[max(0, mention.start - 12):min(len(query), mention.end + 12)]
            aggregation = "none"
            for pattern, value in _AGGREGATIONS:
                if pattern.search(window):
                    aggregation = value
                    break
            # A field ending in "总额/总量" describes an additive requested metric.
            if aggregation == "none" and re.search(r"总额|总量|合计", mention.alias):
                aggregation = "sum"
            if aggregation == "none":
                continue
            field_spec = catalog.field(symbol.canonical_id or "")
            metrics.append(MetricSpec(
                field=symbol, aggregation=aggregation,
                unit=field_spec.unit if field_spec else None,
            ))
        return _dedupe_metrics(metrics)

    def _predicates(self, query: str, mentions: list[FieldMatch], symbols: list[SymbolRef],
                    catalog: CatalogSnapshot,
                    ignored_ranges: list[tuple[int, int]] | None = None
                    ) -> tuple[list[PredicateNode], list[UnresolvedItem]]:
        predicates: list[PredicateNode] = []
        occupied: list[tuple[int, int]] = list(ignored_ranges or [])
        for index, (mention, symbol) in enumerate(zip(mentions, symbols)):
            if any(mention.start < end and mention.end > start for start, end in occupied):
                continue
            next_start = mentions[index + 1].start if index + 1 < len(mentions) else len(query)
            suffix = query[mention.end:min(next_start, mention.end + 60)]
            parsed_values = self._operator_values(suffix)
            field_spec = catalog.field(symbol.canonical_id or "")
            for parsed_index, (operator, values, op_start, op_end) in enumerate(parsed_values):
                absolute_start = mention.start if parsed_index == 0 else mention.end + op_start
                absolute_end = mention.end + op_end
                if any(absolute_start < end and absolute_end > start for start, end in occupied):
                    continue
                raw_unit, unit = self._unit(query[mention.end + op_start:absolute_end + 12])
                values, unit = _normalize_values(
                    values, unit, field_spec.unit if field_spec else None
                )
                literal_value = values if operator == "between" else values[0]
                literal_type = "range" if operator == "between" else "number"
                provenance = Provenance(
                    source="rule", rule="numeric_predicate",
                    span=_span(query, absolute_start, absolute_end),
                )
                predicates.append(PredicateNode(
                    operator=operator, field=symbol,
                    value=LiteralValue(literal_value, literal_type, unit=unit, raw_unit=raw_unit),
                    provenance=[provenance],
                ))
                occupied.append((absolute_start, absolute_end))
            if not parsed_values and field_spec and field_spec.data_type in {"string", "text", "keyword"}:
                parsed_string = self._string_operator_value(suffix)
                if parsed_string:
                    operator, value, op_start, op_end = parsed_string
                    absolute_end = mention.end + op_end
                    predicates.append(PredicateNode(
                        operator=operator, field=symbol,
                        value=LiteralValue(value, "list" if isinstance(value, list) else "string"),
                        provenance=[Provenance(
                            source="rule", rule="categorical_predicate",
                            span=_span(query, mention.start, absolute_end),
                        )],
                    ))
                    occupied.append((mention.start, absolute_end))

        unresolved: list[UnresolvedItem] = []
        for match in _UNKNOWN_NUMERIC_FIELD_RE.finditer(query):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            raw_name = _strip_query_prefix(match.group("field"))
            role = FieldPhraseParser.classify(raw_name)
            if role.role != "field":
                continue
            raw_name = role.text
            if not raw_name:
                continue
            field_ids = catalog.resolve_field(raw_name)
            if field_ids:
                continue
            field_start = match.start("field") + (len(match.group("field")) - len(raw_name))
            symbol = SymbolRef(
                raw_name=raw_name, status="unresolved",
                span=_span(query, field_start, match.start("op")),
            )
            operator = _operator_from_text(match.group("op"))
            predicates.append(PredicateNode(
                operator=operator,
                field=symbol,
                value=LiteralValue(float(match.group("number")), "number"),
                provenance=[Provenance(
                    source="rule", rule="unknown_numeric_predicate",
                    span=_span(query, field_start, match.end()),
                )],
            ))
            unresolved.append(UnresolvedItem(
                code="unknown_field",
                message=f"Catalog 中无法唯一绑定字段：{raw_name}",
                span=symbol.span,
            ))
        return predicates, unresolved

    @staticmethod
    def _catalog_categorical_shortcuts(
        query: str, catalog: CatalogSnapshot, existing: list[PredicateNode],
    ) -> list[PredicateNode]:
        """Bind common categorical adjectives as predicates when catalog fields exist."""
        result: list[PredicateNode] = []
        shortcuts = (
            (r"英文", "language", "英文"),
            (r"中文", "language", "中文"),
            (r"期刊论文", "document_type", "期刊论文"),
        )
        for pattern, field_name, value in shortcuts:
            match = re.search(pattern, query)
            field_ids = catalog.resolve_field(field_name)
            if not match or len(field_ids) != 1:
                continue
            if any(item.field and item.field.canonical_id == field_ids[0] for item in existing):
                continue
            symbol = SymbolRef(
                raw_name=field_name, canonical_id=field_ids[0], status="resolved",
                candidates=[CandidateRef(field_ids[0], source="catalog", status="resolved")],
                span=_span(query, match.start(), match.end()),
            )
            result.append(PredicateNode(
                operator="eq", field=symbol, value=LiteralValue(value, "string"),
                provenance=[Provenance(
                    source="rule", rule="categorical_shortcut",
                    span=_span(query, match.start(), match.end()),
                )],
            ))
        return result

    @staticmethod
    def _operator_values(text: str) -> list[tuple[str, list[float], int, int]]:
        candidates = []
        for priority, (pattern, operator) in enumerate(_OPERATOR_PATTERNS):
            for match in pattern.finditer(text):
                if match.start() > 6 and not re.search(r"(?:且|并且|同时|,|，|、)\s*$", text[:match.start()]):
                    continue
                candidates.append((match.start(), match.end(), priority, operator, match))
        selected = []
        for start, end, priority, operator, match in sorted(candidates, key=lambda item: (item[0], item[2])):
            if any(start < old_end and end > old_start for old_start, old_end, *_ in selected):
                continue
            selected.append((start, end, priority, operator, match))
        return [
            (operator, [float(item) for item in match.groups()], start, end)
            for start, end, _, operator, match in sorted(selected)
        ]

    @staticmethod
    def _string_operator_value(text: str) -> tuple[str, str | list[str], int, int] | None:
        patterns = (
            (re.compile(r"(?:属于|在)\s*([^。；;]+?)(?=且|并且|以及|。|；|;|$)"), "in"),
            (re.compile(r"(?:包含|含有)\s*([^，,。；;且]+)"), "contains"),
            (re.compile(r"(?:不等于|不是)\s*([^，,。；;且]+)"), "ne"),
            (re.compile(r"(?:等于|为|是)\s*([^，,。；;且]+)"), "eq"),
        )
        for pattern, operator in patterns:
            match = pattern.search(text)
            if not match or match.start() > 6:
                continue
            raw = match.group(1).strip()
            raw = re.sub(r"的?(?:数据|记录|文献|论文|资料)$", "", raw).strip()
            if not raw:
                continue
            if operator == "in":
                values = [item.strip() for item in re.split(r"[、,，/]|或", raw) if item.strip()]
                return operator, values, match.start(), match.end()
            return operator, raw, match.start(), match.end()
        return None

    @staticmethod
    def _unit(text: str) -> tuple[str | None, str | None]:
        for pattern, canonical in _UNIT_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0), canonical
        return None, None

    def _events(self, query: str, mentions: list[FieldMatch], symbols: list[SymbolRef],
                temporal: list[TemporalItem], catalog: CatalogSnapshot,
                query_sampling: SamplingPolicy | None = None) -> list[EventSpec]:
        events: list[EventSpec] = []
        for match in _GENERIC_CONTINUOUS_EVENT_RE.finditer(query):
            raw_field = _clean_field_phrase(match.group("field"))
            symbol = _find_symbol(raw_field, symbols)
            if not symbol:
                continue
            source_id = catalog.source_for_field(symbol.canonical_id or "")
            field = catalog.field(symbol.canonical_id or "")
            raw_unit, unit = self._unit(match.group("value_unit") or "")
            if unit is None and match.group("value_unit"):
                raw_unit = match.group("value_unit")
                unit = _canonical_generic_unit(raw_unit)
            values, unit = _normalize_values(
                [float(match.group("value"))], unit, field.unit if field else None
            )
            operator = _operator_from_text(match.group("op"))
            condition = PredicateNode(
                operator=operator, field=symbol,
                value=LiteralValue(values[0], "number", unit=unit, raw_unit=raw_unit),
                provenance=[Provenance(
                    source="rule", rule="generic_event_condition",
                    span=_span(query, match.start("field"), match.end("value")),
                )],
            )
            event_id = f"event_{len(events) + 1}_consecutive"
            output_name = f"{event_id}_intervals"
            metric_id = f"{source_id}.consecutive_duration" if source_id else "semantic.consecutive_duration"
            sampling = _sampling_policy(source_id, catalog)
            if query_sampling and query_sampling.expected_interval_seconds:
                sampling.expected_interval_seconds = query_sampling.expected_interval_seconds
                sampling.max_gap_seconds = query_sampling.expected_interval_seconds * 2
            event_span = _span(query, match.start("field"), match.end())
            events.append(EventSpec(
                event_id=event_id, condition=condition,
                derived_metric=DerivedMetricCall(
                    metric_id=metric_id, arguments={"operator": "CONSECUTIVE"},
                    result_type="duration", unit="second",
                    provenance=[Provenance(source="rule", rule="generic_consecutive",
                                           span=event_span)],
                ),
                threshold=_duration_constraint(query, match),
                window=_window_from_temporal(temporal), sampling=sampling,
                group_by=["local_date"], output_name=output_name,
                output_ref=OutputRef(event_id, "intervals", "event_interval",
                                     shape="event_set", grain="interval",
                                     keys=["local_date"]),
                provenance=[Provenance(source="rule", rule="generic_continuous_event",
                                       span=event_span)],
            ))

        for match in _CONTINUOUS_EVENT_RE.finditer(query):
            symbol = _nearest_numeric_symbol(match.start(), mentions, symbols, catalog)
            if not symbol or not symbol.canonical_id:
                continue
            if any(item.provenance and item.provenance[0].span
                   and item.provenance[0].span.start <= match.start() < item.provenance[0].span.end
                   for item in events):
                continue
            field = catalog.field(symbol.canonical_id)
            raw_unit, unit = self._unit(match.group("value_unit") or "")
            values, unit = _normalize_values(
                [float(match.group("value"))], unit, field.unit if field else None
            )
            condition_span = _span(query, symbol.span.start, match.end("value")) if symbol.span else None
            condition = PredicateNode(
                operator="gt", field=symbol,
                value=LiteralValue(values[0], "number", unit=unit, raw_unit=raw_unit),
                provenance=[Provenance(
                    source="rule", rule="event_condition",
                    span=condition_span,
                )],
            )
            duration = _duration_constraint(query, match)
            event_start = symbol.span.start if symbol.span else match.start()
            event_span = _span(query, event_start, match.end())
            source_id = catalog.source_for_field(symbol.canonical_id)
            metric_spec = catalog.derived_metric(f"{source_id}.consecutive_duration")
            events.append(EventSpec(
                event_id=f"event_{len(events) + 1}_consecutive",
                condition=condition,
                derived_metric=DerivedMetricCall(
                    metric_id=f"{source_id}.consecutive_duration",
                    arguments={
                        "condition": "event_condition",
                        **(metric_spec.defaults if metric_spec else {}),
                    },
                    result_type="duration", unit="second",
                    provenance=[Provenance(
                        source="rule", rule="derived_metric_consecutive", span=event_span,
                    )],
                ),
                threshold=duration,
                window=_window_from_temporal(temporal),
                sampling=_sampling_policy(source_id, catalog),
                group_by=["local_date"],
                output_name=f"event_{len(events) + 1}_local_dates",
                output_ref=OutputRef(
                    f"event_{len(events) + 1}_consecutive", "intervals",
                    "event_interval", shape="event_set", grain="interval",
                    keys=["local_date"],
                ),
                provenance=[Provenance(
                    source="rule", rule="continuous_event", span=event_span,
                )],
            ))

        for match in _CUMULATIVE_EVENT_RE.finditer(query):
            symbol = _nearest_numeric_symbol(match.start(), mentions, symbols, catalog)
            if not symbol or not symbol.canonical_id:
                continue
            field = catalog.field(symbol.canonical_id)
            lower_raw, lower_unit = self._unit(match.group("lower_unit") or "")
            upper_raw, upper_unit = self._unit(match.group("upper_unit") or "")
            if upper_unit is None:
                upper_unit = lower_unit
                upper_raw = lower_raw
            lower, lower_unit = _normalize_values(
                [float(match.group("lower"))], lower_unit, field.unit if field else None
            )
            upper, upper_unit = _normalize_values(
                [float(match.group("upper"))], upper_unit, field.unit if field else None
            )
            lower_predicate = PredicateNode(
                operator="gt", field=symbol,
                value=LiteralValue(lower[0], "number", lower_unit, lower_raw),
                provenance=[Provenance(
                    source="rule", rule="event_lower_bound",
                    span=_span(query, match.start(), match.end("lower")),
                )],
            )
            upper_predicate = PredicateNode(
                operator="lte", field=symbol,
                value=LiteralValue(upper[0], "number", upper_unit, upper_raw),
                provenance=[Provenance(
                    source="rule", rule="event_upper_bound",
                    span=_span(query, match.start("upper"), match.end("upper")),
                )],
            )
            condition = PredicateNode(
                kind="boolean", operator="and", children=[lower_predicate, upper_predicate]
            )
            event_span = _span(query, match.start(), match.end())
            source_id = catalog.source_for_field(symbol.canonical_id)
            metric_spec = catalog.derived_metric(f"{source_id}.cumulative_duration")
            sampling = _sampling_policy(source_id, catalog)
            if metric_spec and metric_spec.missing_data_policy:
                sampling.missing_data_policy = metric_spec.missing_data_policy
            events.append(EventSpec(
                event_id=f"event_{len(events) + 1}_cumulative",
                condition=condition,
                derived_metric=DerivedMetricCall(
                    metric_id=f"{source_id}.cumulative_duration",
                    arguments={
                        "condition": "event_condition", "window": "calendar_day",
                        **(metric_spec.defaults if metric_spec else {}),
                    },
                    result_type="duration", unit="second",
                    provenance=[Provenance(
                        source="rule", rule="derived_metric_cumulative", span=event_span,
                    )],
                ),
                threshold=_duration_constraint(query, match),
                window=_window_from_temporal(temporal),
                sampling=sampling,
                group_by=["local_date"],
                output_name=f"event_{len(events) + 1}_local_dates",
                output_ref=OutputRef(
                    f"event_{len(events) + 1}_cumulative", "dates", "date",
                    shape="event_set", grain="date", keys=["local_date"],
                ),
                provenance=[Provenance(
                    source="rule", rule="cumulative_event", span=event_span,
                )],
            ))
        return events

    @staticmethod
    def _date_projections(events: list[EventSpec]) -> list[DerivedProjectionSpec]:
        result = []
        for event in events:
            if not event.output_ref:
                continue
            producer = f"project_{event.event_id}_date"
            result.append(DerivedProjectionSpec(
                projection_id=producer, function="local_date",
                input_ref=event.output_ref,
                output=OutputRef(producer, "dates", "date", shape="event_set",
                                 grain="date", keys=["local_date"]),
                provenance=list(event.provenance),
            ))
        return result

    @staticmethod
    def _set_operations(query: str, events: list[EventSpec],
                        projections: list[DerivedProjectionSpec]) -> list[SetOperationSpec]:
        match = re.search(r"交集(?:日期)?", query)
        if not match or len(events) < 2:
            return []
        refs = [item.output for item in projections]
        return [SetOperationSpec(
            operation="intersection",
            inputs=[item.ref_id for item in refs],
            output_name="abnormal_intersection_dates",
            granularity="date",
            input_refs=refs,
            output_ref=OutputRef(
                "set_intersection_1", "dates", "date",
                shape="event_set", grain="date", keys=["local_date"],
            ),
            provenance=[Provenance(
                source="rule", rule="set_intersection",
                span=_span(query, match.start(), match.end()),
            )],
        )]

    @staticmethod
    def _temporal(query: str) -> list[TemporalItem]:
        parsed = parse_year_month_matrix(query)
        matrix_mode = bool(parsed)
        if not parsed:
            now = datetime.now()
            parsed = parse_time_ranges(
                query, now.year, now.month, now.day, now.hour, now.minute
            )
        items = []
        for item in parsed:
            if getattr(item, "invalid_reason", ""):
                continue
            raw = getattr(item, "raw", "") or ""
            start = query.find(raw) if raw else -1
            evidence_raw = raw
            if start < 0 and raw:
                # Matrix expansion synthesizes values such as "2025年6月". Keep
                # the generated value while anchoring it to the source year.
                year_match = re.match(r"((?:19|20)\d{2})年", raw)
                if year_match:
                    evidence_raw = year_match.group(0)
                    start = query.find(evidence_raw)
            evidence_spans = []
            if raw:
                for match in re.finditer(re.escape(raw), query):
                    evidence_spans.append(_span(query, match.start(), match.end()))
            if matrix_mode:
                year_match = re.match(r"((?:19|20)\d{2})年", raw)
                if year_match:
                    source_year = re.search(re.escape(year_match.group(0)), query)
                    if source_year:
                        year_span = _span(query, source_year.start(), source_year.end())
                        if year_span not in evidence_spans:
                            evidence_spans.append(year_span)
            if start >= 0 and not evidence_spans:
                evidence_spans.append(_span(query, start, start + len(evidence_raw)))
            from_value = getattr(item, "from_", None)
            month_since = re.fullmatch(r"((?:19|20)\d{2})年(\d{1,2})月以来", raw)
            if month_since:
                from_value = f"{int(month_since.group(1)):04d}-{int(month_since.group(2)):02d}-01"
            items.append(TemporalItem(
                operator=getattr(item, "op", ""),
                from_value=from_value,
                to_value=getattr(item, "to", None),
                exact=getattr(item, "exact", None),
                granularity=getattr(item, "granularity", "year") or "year",
                timezone=getattr(item, "timezone", "") or "",
                raw=raw,
                span=_span(query, start, start + len(evidence_raw)) if start >= 0 else None,
                evidence_spans=evidence_spans,
            ))
        exact_range = re.search(
            r"(?P<y1>(?:19|20)\d{2})年(?P<m1>\d{1,2})月(?P<d1>\d{1,2})日\s*"
            r"(?:至|到|~|～)\s*(?:(?P<y2>(?:19|20)\d{2})年)?"
            r"(?P<m2>\d{1,2})月(?P<d2>\d{1,2})日", query
        )
        if exact_range:
            y2 = int(exact_range.group("y2") or exact_range.group("y1"))
            precise = TemporalItem(
                operator="between",
                from_value=f"{int(exact_range.group('y1')):04d}-{int(exact_range.group('m1')):02d}-{int(exact_range.group('d1')):02d}",
                to_value=f"{y2:04d}-{int(exact_range.group('m2')):02d}-{int(exact_range.group('d2')):02d}",
                granularity="day", raw=exact_range.group(0),
                span=_span(query, exact_range.start(), exact_range.end()),
            )
            items = [item for item in items if not (
                item.span and item.span.start < exact_range.end() and item.span.end > exact_range.start()
            )]
            items.insert(0, precise)
        return items

    @staticmethod
    def _calculations(query: str, symbols: list[SymbolRef],
                      set_operations: list[SetOperationSpec],
                      catalog: CatalogSnapshot) -> list[CalculationSpec]:
        marker_re = re.compile(r"环比|同比")
        pair_re = re.compile(
            r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(?:vs\.?|对比|相比)\s*"
            r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月",
            re.I,
        )
        markers = list(marker_re.finditer(query))
        calculations = []
        for index, match in enumerate(pair_re.finditer(query), start=1):
            prior = [item for item in markers if item.start() < match.start()]
            calculation_type = prior[-1].group(0) if prior else "difference"
            type_id = {"环比": "mom", "同比": "yoy"}.get(calculation_type, "difference")
            left = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
            right = f"{int(match.group(3)):04d}-{int(match.group(4)):02d}"
            input_symbol = (
                _formula_input_symbol(query, match, symbols)
                or _unique_numeric_period_symbol(symbols, catalog)
            )
            input_id = _symbol_identifier(input_symbol) if input_symbol else ""
            parameters = {"left": left, "right": right, "comparison_id": index,
                          "input_field": input_id}
            if input_symbol is None:
                parameters["formula_gap"] = "input_field_unbound"
            provenance = [Provenance(
                source="rule", rule="period_comparison",
                span=_span(query, match.start(), match.end()),
            )]
            if prior:
                marker = prior[-1]
                provenance.append(Provenance(
                    source="rule", rule="period_comparison_family",
                    span=_span(query, marker.start(), marker.end()),
                ))
            for family_marker in markers:
                marker_type = {"环比": "mom", "同比": "yoy"}.get(family_marker.group(0))
                marker_span = _span(query, family_marker.start(), family_marker.end())
                if marker_type == type_id and all(item.span != marker_span for item in provenance):
                    provenance.append(Provenance(
                        source="rule", rule="period_comparison_reference",
                        span=marker_span,
                    ))
            calculations.append(CalculationSpec(
                calculation_type=type_id,
                expression=f"({left} - {right}) / {right}",
                parameters=parameters,
                provenance=provenance,
                calculation_id=f"calculation_{type_id}_{index}",
                inputs=[ExpressionNode("field", symbol=input_symbol, result_type="number")]
                if input_symbol is not None else [],
                output=OutputRef(f"calculation_{type_id}_{index}", "value", "number",
                                 shape="scalar", grain="scalar", fields=[type_id]),
            ))
        if not calculations:
            for raw, type_id in (("波动率", "volatility"), ("增长率", "growth"), ("差值", "difference")):
                match = re.search(raw, query)
                if match:
                    parameters = {}
                    expression = raw
                    input_symbol = None
                    if type_id == "volatility":
                        input_symbol = _formula_input_symbol(query, match, symbols)
                        input_id = _symbol_identifier(input_symbol) if input_symbol else ""
                        method = "stddev"
                        parameters = {
                            "input_field": input_id,
                            "filter_reference": (
                                set_operations[0].output_ref.ref_id
                                if set_operations and set_operations[0].output_ref else ""
                            ),
                            "window": "month_to_date" if "当月截至目前" in query else "query_window",
                            "group_by": "local_date" if re.search(r"日均|每日|按日", query) else "",
                            "method": method,
                            "derived_metric_id": "semantic.stddev",
                        }
                        if input_symbol is None:
                            parameters["formula_gap"] = "input_field_unbound"
                        expression = f"{method}({input_symbol.raw_name if input_symbol else ''})"
                    calculations.append(CalculationSpec(
                        type_id, expression, parameters=parameters,
                        provenance=[Provenance(source="rule", rule="named_formula",
                                               span=_span(query, match.start(), match.end()))],
                        calculation_id=f"calculation_{type_id}_{match.start()}",
                        inputs=[ExpressionNode("field", symbol=input_symbol, result_type="number")]
                        if input_symbol is not None else [],
                        output=OutputRef(
                            f"calculation_{type_id}_{match.start()}", "value", "number",
                            unit=(catalog.field(input_symbol.canonical_id or "").unit
                                  if input_symbol and input_symbol.canonical_id
                                  and catalog.field(input_symbol.canonical_id or "") else None),
                            shape="scalar", grain="scalar", fields=[type_id],
                        ),
                    ))
        return calculations

    @staticmethod
    def _scoped_aggregates_and_comparisons(
        query: str, symbols: list[SymbolRef]
    ) -> tuple[list[ScopedAggregateSpec], list[ComparisonSpec]]:
        factor_match = re.search(r"全(?:站|局|部).*?平均值(?:的)?\s*(\d+(?:\.\d+)?)\s*倍", query)
        compare_match = re.search(r"是否?(?:高于|大于|超过)", query)
        if not factor_match or not compare_match:
            return [], []
        field_symbol = _nearest_mentioned_symbol(query, compare_match.start(), symbols)
        if not field_symbol:
            return [], []
        unit = "currency" if re.search(r"金额|消费|spend|price|cost", field_symbol.raw_name, re.I) else None
        field_expr = ExpressionNode(
            kind="field", symbol=field_symbol, result_type="number", unit=unit,
        )
        filtered_output = OutputRef("agg_filtered_avg", "value", "number", unit=unit)
        global_output = OutputRef("agg_global_avg", "value", "number", unit=unit)
        filtered = ScopedAggregateSpec(
            aggregate=AggregateSpec(
                "agg_filtered_avg", "avg", field_expr, filtered_output,
                [Provenance(source="rule", rule="scoped_filtered_avg",
                            span=_span(query, max(0, compare_match.start() - 20), compare_match.start()))],
            ),
            scope="filtered",
        )
        global_aggregate = ScopedAggregateSpec(
            aggregate=AggregateSpec(
                "agg_global_avg", "avg", field_expr, global_output,
                [Provenance(source="rule", rule="scoped_global_avg",
                            span=_span(query, factor_match.start(), factor_match.end()))],
            ),
            scope="global",
        )
        factor = float(factor_match.group(1))
        left = ExpressionNode(
            kind="output_ref", reference=filtered_output.ref_id,
            result_type="number", unit=unit,
        )
        right = ExpressionNode(
            kind="binary", operator="mul", arguments=[
                ExpressionNode(kind="output_ref", reference=global_output.ref_id,
                               result_type="number", unit=unit),
                ExpressionNode(kind="literal", literal=LiteralValue(factor, "number"),
                               result_type="number", unit=None),
            ], result_type="number", unit=unit,
        )
        comparison = ComparisonSpec(
            "compare_filtered_to_global", left, "gt", right,
            OutputRef("compare_filtered_to_global", "result", "boolean"),
            [Provenance(source="rule", rule="scoped_average_comparison",
                        span=_span(query, compare_match.start(), factor_match.end()))],
        )
        return [filtered, global_aggregate], [comparison]

    @staticmethod
    def _references(query: str, set_operations: list[SetOperationSpec]) -> list[ReferenceSpec]:
        return [
            ReferenceSpec(
                reference_id=f"ref_{index}", raw=match.group(0),
                target_task_id=(set_operations[0].output_ref.ref_id
                                if set_operations and set_operations[0].output_ref else None),
                status="resolved" if set_operations else "unresolved",
                span=_span(query, match.start(), match.end()),
            )
            for index, match in enumerate(_REFERENCE_RE.finditer(query), start=1)
        ]

    @staticmethod
    def _output(query: str, ir: UnderstandingIR) -> OutputContract:
        output = OutputContract()
        limit = re.search(r"(?:前|top\s*)(\d+)\s*(?:条|篇|个)?", query, re.I)
        if limit:
            output.limit = int(limit.group(1))
        if "分别" in query and ir.temporal:
            output.group_by.append("time_period")
        positive = re.search(r"同时.*?正向增长", query)
        if positive:
            families = list(dict.fromkeys(
                item.calculation_type for item in ir.calculations
                if item.calculation_type in {"mom", "yoy", "growth"}
            ))
            output.criteria.append(
                " AND ".join(f"{family} > 0" for family in families)
                if families else "growth > 0"
            )
            if len(families) > 1:
                output.group_by.append("comparison_period")
            output.provenance.append(Provenance(
                source="rule", rule="positive_growth_output_filter",
                span=_span(query, positive.start(), positive.end()),
            ))
        elif "正向增长" in query:
            output.criteria.append("growth > 0")
        if ir.set_operations:
            output.fields = [ir.set_operations[0].output_name]
        else:
            output.fields = list(dict.fromkeys(
                item.canonical_id or item.raw_name for item in ir.projections
            ))
        if any(item.calculation_type == "volatility" for item in ir.calculations):
            calculation = next(item for item in ir.calculations
                               if item.calculation_type == "volatility")
            raw_input = calculation.parameters.get("input_field") or "value"
            output.fields.append(f"daily_{str(raw_input).split('.')[-1]}_volatility")
            output.group_by.append("local_date")
        output.fields = list(dict.fromkeys(output.fields))
        output.group_by = list(dict.fromkeys(output.group_by))
        return output

    @staticmethod
    def _boolean_tree(query: str, predicates: list[PredicateNode]) -> PredicateNode | None:
        if not predicates:
            return None
        if len(predicates) == 1:
            return predicates[0]
        ordered = sorted(predicates, key=lambda item: _predicate_bounds(item)[0])
        groups: list[list[PredicateNode]] = [[ordered[0]]]
        previous_end = _predicate_bounds(ordered[0])[1]
        for predicate in ordered[1:]:
            current_start, current_end = _predicate_bounds(predicate)
            connector = query[previous_end:current_start]
            if re.search(r"或者|或", connector):
                groups.append([predicate])
            else:
                groups[-1].append(predicate)
            previous_end = current_end
        and_nodes = [
            group[0] if len(group) == 1
            else PredicateNode(kind="boolean", operator="and", children=group)
            for group in groups
        ]
        return and_nodes[0] if len(and_nodes) == 1 else PredicateNode(
            kind="boolean", operator="or", children=and_nodes
        )

    @staticmethod
    def _compatibility_intent(ir: UnderstandingIR) -> str:
        goal_types = {item.goal_type for item in ir.goals}
        if "compute" in goal_types:
            return "numerical_calculation"
        if "aggregate" in goal_types:
            return "inventory"
        if ir.filters or ir.temporal:
            return "attribute_filter"
        return "semantic_retrieval"


def _span(query: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(start, end, query[start:end])


def _active_write_match(query: str):
    quoted = [
        (item.start(), item.end())
        for pattern in (re.compile(r"`[^`]*`"), re.compile(r"“[^”]*”|\"[^\"]*\""))
        for item in pattern.finditer(query)
    ]
    for match in _WRITE_OPERATION_RE.finditer(query):
        if any(match.start() >= start and match.end() <= end for start, end in quoted):
            continue
        return match
    return None


def _predicate_bounds(predicate: PredicateNode) -> tuple[int, int]:
    span = predicate.provenance[0].span if predicate.provenance else None
    return (span.start, span.end) if span else (0, 0)


def _operator_from_text(text: str) -> str:
    for pattern, operator in _OPERATOR_PATTERNS[1:]:
        if pattern.match(f"{text}0"):
            return operator
    return {
        "大于": "gt", "高于": "gt", "超过": "gt", ">": "gt",
        "小于": "lt", "低于": "lt", "少于": "lt", "<": "lt",
        "等于": "eq", "=": "eq", "不等于": "ne", "!=": "ne",
        "大于等于": "gte", ">=": "gte", "小于等于": "lte", "<=": "lte",
    }.get(text, "eq")


def _strip_query_prefix(value: str) -> str:
    value = re.split(r"以及|并且|同时|且|，|,|。|；|;", value)[-1]
    value = re.sub(r"^(?:请|帮我|查询|查找|找出|筛选|提取|以及|并且|同时|且)+", "", value)
    value = re.sub(r"(?:连续|持续)$", "", value)
    return value.strip()


def _dedupe_symbols(symbols: list[SymbolRef]) -> list[SymbolRef]:
    result: list[SymbolRef] = []
    seen: set[tuple[str | None, str]] = set()
    for symbol in symbols:
        key = (symbol.canonical_id, symbol.raw_name if not symbol.canonical_id else "")
        if key not in seen:
            seen.add(key)
            result.append(symbol)
    return result


def _source_alias_spans(query: str, catalog: CatalogSnapshot) -> list[tuple[int, int]]:
    lowered = query.lower()
    spans = []
    for source in catalog.sources:
        for alias in sorted(source.aliases, key=len, reverse=True):
            start = lowered.find(alias.lower())
            while start >= 0:
                spans.append((start, start + len(alias)))
                start = lowered.find(alias.lower(), start + len(alias))
    return spans


def _nearest_numeric_symbol(position: int, mentions: list[FieldMatch],
                            symbols: list[SymbolRef], catalog: CatalogSnapshot
                            ) -> SymbolRef | None:
    candidates = []
    for mention, symbol in zip(mentions, symbols):
        field = catalog.field(symbol.canonical_id or "")
        if mention.end <= position and field and field.data_type in {
            "integer", "long", "float", "double", "number"
        }:
            candidates.append((mention.end, symbol))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _duration_constraint(query: str, match: re.Match) -> DurationConstraint:
    value = float(match.group("duration"))
    raw_unit = match.group("duration_unit")
    unit_key = raw_unit.lower() if raw_unit.lower() in _DURATION_UNIT_SECONDS else raw_unit
    multiplier = _DURATION_UNIT_SECONDS[unit_key]
    return DurationConstraint(
        operator="gt", value=value, unit=raw_unit,
        normalized_seconds=value * multiplier,
        span=_span(query, match.start("duration"), match.end("duration_unit")),
    )


def _window_from_temporal(temporal: list[TemporalItem]) -> WindowSpec:
    if not temporal:
        return WindowSpec(window_type="absolute")
    item = temporal[0]
    return WindowSpec(
        window_type="absolute",
        from_value=item.from_value or item.exact,
        to_value=item.to_value or item.exact,
        timezone=item.timezone or "Asia/Shanghai",
        granularity=item.granularity,
    )


def _sampling_policy(source_id: str | None, catalog: CatalogSnapshot) -> SamplingPolicy:
    source = catalog.source(source_id or "")
    metadata = source.metadata if source else {}
    timestamp_id = str(metadata.get("timestamp_field", ""))
    partition_ids = [str(item) for item in metadata.get("partition_fields", [])]
    expected = metadata.get("expected_sampling_interval_seconds")
    expected_seconds = int(expected) if expected is not None else None
    return SamplingPolicy(
        order_by=_catalog_symbol(timestamp_id, catalog) if timestamp_id else None,
        partition_by=[_catalog_symbol(item, catalog) for item in partition_ids],
        expected_interval_seconds=expected_seconds,
        max_gap_seconds=expected_seconds * 2 if expected_seconds else None,
        missing_data_policy="break_segment",
        duplicate_policy="keep_last",
    )


def _catalog_symbol(field_id: str, catalog: CatalogSnapshot) -> SymbolRef:
    field = catalog.field(field_id)
    return SymbolRef(
        raw_name=field_id.rsplit(".", 1)[-1],
        canonical_id=field_id if field else None,
        candidates=[CandidateRef(field_id, source="catalog")] if field else [],
        status="resolved" if field else "unresolved",
    )


def _dedupe_metrics(metrics: list[MetricSpec]) -> list[MetricSpec]:
    result: list[MetricSpec] = []
    seen: set[tuple[str, str]] = set()
    for metric in metrics:
        key = (metric.field.canonical_id or metric.field.raw_name, metric.aggregation)
        if key not in seen:
            seen.add(key)
            result.append(metric)
    return result


def _normalize_values(values: list[float], unit: str | None,
                      target_unit: str | None) -> tuple[list[float], str | None]:
    if not unit or not target_unit or unit == target_unit:
        return values, unit or target_unit
    if unit == "km/h" and target_unit == "m/s":
        return [round(value / 3.6, 8) for value in values], target_unit
    if unit == "m/s" and target_unit == "km/h":
        return [round(value * 3.6, 8) for value in values], target_unit
    if unit == "fahrenheit" and target_unit == "celsius":
        return [round((value - 32.0) * 5.0 / 9.0, 8) for value in values], target_unit
    if unit == "celsius" and target_unit == "fahrenheit":
        return [round(value * 9.0 / 5.0 + 32.0, 8) for value in values], target_unit
    return values, unit


def _clean_field_phrase(raw: str) -> str:
    return FieldPhraseParser.clean_field(raw)


def _find_symbol(raw_name: str, symbols: list[SymbolRef]) -> SymbolRef | None:
    if not raw_name:
        return None
    lowered = raw_name.lower()
    exact = [item for item in symbols if item.raw_name.lower() == lowered]
    if exact:
        return next((item for item in exact if item.ref_kind == "hypothesis"), exact[0])
    contained = [item for item in symbols
                 if item.raw_name.lower() in lowered or lowered in item.raw_name.lower()]
    return max(
        contained,
        key=lambda item: (item.ref_kind == "hypothesis", len(item.raw_name)),
        default=None,
    )


def _canonical_generic_unit(raw: str) -> str | None:
    value = raw.lower().replace("²", "2")
    return {
        "m/s2": "m/s^2", "km/h": "km/h", "pa": "pascal",
        "kpa": "kilopascal", "mpa": "megapascal", "w": "watt",
        "kw": "kilowatt", "mw": "megawatt", "mwh": "megawatt_hour",
        "kv": "kilovolt", "%": "percent", "v": "volt", "a": "ampere",
        "m": "meter", "km": "kilometer", "ms": "millisecond",
        "μm": "micrometer", "um": "micrometer", "吨": "tonne",
        "万元": "currency", "元": "currency", "bps": "basis_point",
        "m3/h": "cubic_meter_per_hour", "m³/h": "cubic_meter_per_hour",
        "w/m2": "watt_per_square_meter", "mm/s": "mm/s", "db": "decibel",
        "℃": "celsius", "°c": "celsius", "个/平方米": "count_per_square_meter",
    }.get(value)


def _symbol_identifier(symbol: SymbolRef | None) -> str:
    if not symbol:
        return ""
    if symbol.ref_kind == "catalog":
        return symbol.canonical_id or ""
    return symbol.candidates[0].identifier if symbol.candidates else ""


def _formula_input_symbol(query: str, formula_match, symbols: list[SymbolRef]) -> SymbolRef | None:
    """Return only a field that syntactically modifies the named formula.

    `价格波动率` is explicit evidence for price; an unrelated earlier `成交量`
    is not.  This narrow attachment check replaces the former nearest-field
    heuristic and intentionally returns None for a bare `波动率`.
    """
    prefix = query[max(0, formula_match.start() - 48):formula_match.start()]
    candidates = []
    for symbol in symbols:
        raw = symbol.raw_name.strip("`")
        if not raw or re.search(r"日期|时间|timestamp|(?:^|_)id$", raw, re.I):
            continue
        attachment = re.search(rf"(?:^|[、，,；;\s])?{re.escape(raw)}(?:的)?$", prefix, re.I)
        if attachment:
            candidates.append((len(raw), symbol))
    if not candidates:
        return None
    selected = max(candidates, key=lambda value: value[0])[1]
    identifier = _symbol_identifier(selected)
    return max(
        (item for item in symbols if identifier and _symbol_identifier(item) == identifier),
        key=lambda item: (bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", item.raw_name)),
                          len(item.raw_name)),
        default=selected,
    )


def _unique_numeric_period_symbol(
    symbols: list[SymbolRef], catalog: CatalogSnapshot,
) -> SymbolRef | None:
    """Infer a period-comparison input only when one numeric field is possible."""
    candidates: dict[str, SymbolRef] = {}
    for symbol in symbols:
        identifier = _symbol_identifier(symbol)
        field = catalog.field(identifier)
        if not field or field.data_type not in {"integer", "long", "float", "double", "number"}:
            continue
        if re.search(r"日期|时间|timestamp|(?:^|_)id$", symbol.raw_name, re.I):
            continue
        candidates.setdefault(identifier, symbol)
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _nearest_semantic_symbol(position: int, symbols: list[SymbolRef]) -> SymbolRef | None:
    candidates = [
        item for item in symbols
        if item.span and item.span.start < position
        and not re.search(r"日期|时间|timestamp|(?:^|_)id$", item.raw_name, re.I)
    ]
    return max(candidates, key=lambda item: item.span.start, default=None)


def _nearest_mentioned_symbol(query: str, position: int,
                              symbols: list[SymbolRef]) -> SymbolRef | None:
    candidates = []
    for item in symbols:
        if re.search(r"日期|时间|timestamp|(?:^|_)id$", item.raw_name, re.I):
            continue
        found = query.rfind(item.raw_name, 0, position)
        if found >= 0:
            candidates.append((found, len(item.raw_name), item))
    return max(candidates, key=lambda row: (row[0], row[1]))[2] if candidates else None
