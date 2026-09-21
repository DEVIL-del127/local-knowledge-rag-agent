"""Lossless clause segmentation used by independent requirement extraction."""
from __future__ import annotations

import hashlib
import re

from .models import ClauseEdge, ClauseGraph, ClauseNode, SourceSpan


_BOUNDARY_RE = re.compile(r"[。！？!?；;\n]+")
_CONNECTOR_RE = re.compile(r"以及|并且|同时|而且|然后|再|或者|或|但|但是|且|并|\b(?:and|or|then)\b", re.I)
_ROLE_PATTERNS = (
    ("turn_directive", re.compile(r"停止刚才|取消(?:旧|上一个|前一个)?|上一轮|上一步|前一步|改成|替换|那.+呢")),
    ("source_declaration", re.compile(r"数据源|数据表|索引|来自|使用.+数据")),
    ("schema_declaration", re.compile(r"包含|字段|每\s*\d+(?:\.\d+)?\s*(?:毫秒|秒|分|分钟|小时).*(?:记录|采样|上报)|`[^`]+`")),
    ("set_operation", re.compile(r"交集|并集|差集|重合(?:时段|时间段|日期|区间)?")),
    ("event", re.compile(r"连续|持续|(?:累计|总|合计)(?:[A-Za-z_\u4e00-\u9fff]{0,12})?(?:时长|耗时)|金叉|死叉|上穿|下穿")),
    ("aggregate", re.compile(r"平均|均值|总额|总和|合计")),
    ("calculation", re.compile(r"计算|同比|环比|增长率|波动率|标准差|相关|回撤|分位数")),
    ("reference", re.compile(r"该日期|上述(?:结果|日期|文档)|这些结果|前一步|上一步")),
    ("output", re.compile(r"最终输出|输出|返回|列出|提取|找出|告诉我")),
    ("comparison", re.compile(r"大于|小于|超过|低于|高于|等于|不为空|非空")),
    ("filter", re.compile(r"排除|仅|只要")),
)


class ClauseGraphBuilder:
    """Build a small graph without rewriting source text or offsets."""

    def build(self, query: str) -> ClauseGraph:
        quoted = _quoted_ranges(query)
        nodes: list[ClauseNode] = []
        edges: list[ClauseEdge] = []
        for start, end in _segments(query):
            raw = query[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            text = raw.strip()
            if not text:
                continue
            start += left_trim
            end = start + len(text)
            scope = "quoted" if _inside_any(start, end, quoted) else "active"
            node_id = "clause_" + hashlib.sha1(
                f"{start}:{end}:{text}".encode("utf-8")
            ).hexdigest()[:12]
            nodes.append(ClauseNode(
                clause_id=node_id,
                span=SourceSpan(start, end, text),
                text=text,
                role="quoted_content" if scope == "quoted" else _role(text),
                polarity="negative" if re.search(r"不要|排除|不包含|非", text) else "positive",
                instruction_scope=scope,
                group_id=_group_id(query, start),
            ))
        for previous, current in zip(nodes, nodes[1:]):
            between = query[previous.span.end:current.span.start]
            edges.append(ClauseEdge(
                previous.clause_id, current.clause_id, _relation(between or current.text[:8]),
                SourceSpan(previous.span.end, current.span.start, between),
            ))
        return ClauseGraph(nodes=nodes, edges=edges)


def _segments(query: str) -> list[tuple[int, int]]:
    boundaries = {0, len(query)}
    for match in _BOUNDARY_RE.finditer(query):
        boundaries.update((match.start(), match.end()))
    for match in _CONNECTOR_RE.finditer(query):
        boundaries.add(match.start())
    ordered = sorted(boundaries)
    return [(left, right) for left, right in zip(ordered, ordered[1:]) if right > left]


def _quoted_ranges(query: str) -> list[tuple[int, int]]:
    patterns = (
        re.compile(r"```[\s\S]*?```"), re.compile(r"`[^`]*`"),
        re.compile(r"“[^”]*”|\"[^\"]*\"|‘[^’]*’|'[^']*'"),
    )
    return sorted((item.start(), item.end()) for pattern in patterns for item in pattern.finditer(query))


def _inside_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start >= left and end <= right for left, right in ranges)


def _role(text: str) -> str:
    return next((role for role, pattern in _ROLE_PATTERNS if pattern.search(text)), "context")


def _relation(text: str) -> str:
    if re.search(r"上述|该(?:日期|结果)|引用", text):
        return "USES_OUTPUT"
    if re.search(r"继承|沿用", text):
        return "INHERITS"
    if re.search(r"替换|改成|不是.+?是", text):
        return "REPLACES"
    if re.search(r"同一|相同|这些", text):
        return "SAME_SCOPE"
    if re.search(r"或者|\bOR\b|或", text, re.I):
        return "OR"
    if re.search(r"然后|再|\bTHEN\b", text, re.I):
        return "THEN"
    if re.search(r"但|但是", text):
        return "NOT"
    if re.search(r"由于|依赖|基于", text):
        return "DEPENDS_ON"
    return "AND"


def _group_id(query: str, position: int) -> str:
    depth = 0
    for char in query[:position]:
        if char in "([（":
            depth += 1
        elif char in ")]）":
            depth = max(0, depth - 1)
    return f"group_depth_{depth}"
