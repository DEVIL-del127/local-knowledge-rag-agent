"""Role-aware field phrase parsing shared by source and rule extractors.

This module deliberately separates a data field from action values and
calculation language.  A previous broad "text before comparison" expression
was able to turn connector and modifier fragments into fields, which then
polluted both the demand ledger and schema binding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_CONNECTOR_RE = re.compile(r"(?:以及|并且|并|同时|而且|或者|或|和|与|及|且|、|,|，|;|；|。|：|:)")
_LEADING_RE = re.compile(
    r"^(?:请|找出|查询|筛选|提取|计算|求|其中|在|于|从|对|将|把|以及|和|与|及|"
    r"并|该日期|这些(?:日期|记录|用户|结果|数据)?|当月截至目前|日均|每日|每月|每年|当日|当天|本日|"
    r"累计|连续|持续|一直|保持|所有|各个|每个|的)+"
)
_TRAILING_RE = re.compile(
    r"(?:的?(?:值|数值|次数|记录|数据)|累计|连续|持续|一直|保持|的总时长|总时长|时长)$"
)
_SOURCE_PREFIX_RE = re.compile(r"^.*?(?:数据源|数据表|数据|表|索引|传感器|设备|记录)中")
_TIME_RATIO_RE = re.compile(r"(?:的)?时间占比$|占用时间(?:的)?比例$|时长占比$")
_ACTION_PATTERNS = (
    (re.compile(r"(?:add[_ -]?to[_ -]?cart|加购|加入购物车)", re.I), "add_to_cart"),
    (re.compile(r"(?:purchase|购买|下单|成交)", re.I), "purchase"),
    (re.compile(r"(?:click|点击)", re.I), "click"),
    (re.compile(r"(?:view|浏览|查看)", re.I), "view"),
)
_NON_FIELDS = {
    "", "累计", "连续", "持续", "总时长", "的总时长", "时间占比", "的时间占比",
    "当天", "当日", "当天累计", "但", "但未", "并", "和", "以及", "且", "内",
}
_DYNAMIC_REFERENCE_RE = re.compile(
    r"(?:额定|标称|基准|参考|目标|阈值|上限|下限|限值|baseline|rated|nominal|reference|target)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class FieldRole:
    """One phrase classified before any Catalog lookup occurs."""

    role: str  # field / action_value / derived_operator / noise
    text: str
    start: int
    end: int
    normalized: str = ""


class FieldPhraseParser:
    """Extract field candidates while refusing connector/modifier pseudo-fields."""

    @classmethod
    def field_before(cls, text: str, position: int, *, window: int = 48) -> FieldRole:
        start = max(0, position - window)
        fragment = text[start:position]
        # A later connector starts a new syntactic role; it must never remain
        # prepended to the following field ("和主轴转速").
        connector = list(_CONNECTOR_RE.finditer(fragment))
        if connector:
            start += connector[-1].end()
            fragment = text[start:position]
        return cls.classify(fragment, start)

    @classmethod
    def classify(cls, raw: str, start: int = 0) -> FieldRole:
        original = raw
        value = raw.strip()
        left_trim = len(raw) - len(raw.lstrip())
        absolute_start = start + left_trim
        if not value:
            return FieldRole("noise", "", absolute_start, absolute_start)

        markdown = re.match(r"[*_]+", value)
        if markdown:
            absolute_start += markdown.end()
            value = value[markdown.end():]
        value = value.rstrip("*_ ")

        # Explicit query fields have the strongest textual evidence.  Preserve
        # the exact interior text but anchor the complete backtick expression.
        exact = re.search(r"`([A-Za-z_][A-Za-z0-9_.]*)`\s*$", value)
        if exact:
            begin = absolute_start + exact.start()
            return FieldRole("field", exact.group(1), begin, begin + len(exact.group(0)), exact.group(1))

        value = _SOURCE_PREFIX_RE.sub("", value)
        value = re.sub(r"^.*?(?:记录一次|采样一次|上报一次)", "", value)
        # "24小时内连续浏览" is an action expression, not the fake field
        # "内连续浏览".  Keep the action for event/cumulative-count parsing.
        value = re.sub(r"^.*?(?:\d+(?:\.\d+)?\s*(?:秒|分钟|分|小时|时|天|日))内", "", value)
        value = _LEADING_RE.sub("", value).strip()
        # A time window before a predicate scopes the query; it is never part
        # of the following field name ("2026年1月1日至今温度一直高于...").
        value = re.sub(
            r"^(?:(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
            r"(?:至今|到今天|以来|起)?|今天|昨日|至今)\s*", "", value,
        )
        value = _LEADING_RE.sub("", value).strip()
        if _TIME_RATIO_RE.search(value):
            phrase = _TIME_RATIO_RE.search(value).group(0)
            begin = start + original.rfind(phrase)
            return FieldRole("derived_operator", phrase, begin, begin + len(phrase), "time_ratio")
        # Action words inside an amount/price phrase describe a numeric field
        # ("购买金额"), not an action enum ("购买").
        if re.search(r"(?:金额|价格|费用|成本|收入|支出)$", value):
            value = _TRAILING_RE.sub("", value).strip(" 的")
            begin = start + original.rfind(value)
            return FieldRole("field", value, begin, begin + len(value), value)
        for pattern, action in _ACTION_PATTERNS:
            match = pattern.search(value)
            if match:
                begin = start + original.rfind(match.group(0))
                return FieldRole("action_value", match.group(0), begin, begin + len(match.group(0)), action)

        value = _TRAILING_RE.sub("", value).strip(" 的")
        value = re.sub(r"^(?:内|中)", "", value).strip()
        if value in _NON_FIELDS or len(value) > 32:
            return FieldRole("noise", value, absolute_start, absolute_start + len(value))
        # Reject pure numeric and unit material before it can become a schema
        # hypothesis.  A mixed Chinese/identifier phrase remains admissible.
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z/%²]+)?", value):
            return FieldRole("noise", value, absolute_start, absolute_start + len(value))
        begin = start + original.rfind(value)
        return FieldRole("field", value, begin, begin + len(value), value)

    @classmethod
    def action_value(cls, text: str) -> FieldRole | None:
        for pattern, action in _ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return FieldRole("action_value", match.group(0), match.start(), match.end(), action)
        return None

    @classmethod
    def clean_field(cls, raw: str) -> str:
        """Compatibility helper for callers that only need a field string."""
        return cls.classify(raw).text if cls.classify(raw).role == "field" else ""


def hypothesis_matches_field(
    expected: str,
    raw_name: str,
    aliases: list[str] | tuple[str, ...],
    description: str = "",
    *,
    allow_suffix: bool = True,
) -> bool:
    """Match a query field without letting a dynamic baseline shadow a measure.

    Industrial queries often contain both a measured value and a relative
    reference, for example ``speed``/``主轴转速`` and ``额定转速``. A shortened
    output phrase such as ``转速峰值`` may use a unique suffix alias, but it must
    not bind to the dynamic reference unless the phrase names that role.
    """
    normalized = expected.strip("` ").lower()
    names = {
        value.strip("` ").lower()
        for value in (raw_name, *aliases)
        if value and value.strip("` ")
    }
    if not normalized or not names:
        return False
    if normalized in names:
        return True
    if not allow_suffix or len(normalized) < 2:
        return False
    dynamic_reference = bool(
        _DYNAMIC_REFERENCE_RE.search(raw_name)
        or _DYNAMIC_REFERENCE_RE.search(description)
    )
    if dynamic_reference and not _DYNAMIC_REFERENCE_RE.search(normalized):
        return False
    return any(value.endswith(normalized) for value in names)
