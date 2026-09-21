from __future__ import annotations

import re
from dataclasses import dataclass

from agent.literature_ir import (
    DocumentReference, LiteratureRequestIR, LiteratureTask, ReferenceKind, RequestedSection,
)
from agent.temporal_parser import parse_temporal_constraint


@dataclass(frozen=True, slots=True)
class LiteratureSemantics:
    request: LiteratureRequestIR


def analyze_literature(query: str) -> LiteratureSemantics:
    """The sole literature semantics entry point used after DomainRouter."""
    raw = str(query).strip()
    lowered = raw.casefold()
    title, filename, author = _explicit_identity(raw)
    reference = _reference(raw, title=title, filename=filename, author=author)
    task = _task(lowered, reference=reference)
    comparison_objects = _comparison_objects(raw) if task == LiteratureTask.COMPARE else ()
    canonical, sense, terms = _topic(raw, title=title)
    requested_sections = _sections(raw)
    if task == LiteratureTask.COMPARE and not requested_sections:
        requested_sections = (RequestedSection.METHOD, RequestedSection.RESULT)
    comparison_dimensions = tuple(item.value for item in requested_sections)
    request = LiteratureRequestIR(
        raw_query=raw,
        task=task,
        topic_terms=terms,
        canonical_topic=canonical,
        sense_id=sense,
        temporal=parse_temporal_constraint(raw),
        document_reference=reference,
        requested_sections=requested_sections,
        language="en" if re.search(r"英文|english", raw, re.I) else
                 ("zh" if "中文" in raw else None),
        document_type="journal_article" if re.search(r"期刊论文|journal paper", raw, re.I) else
                      ("conference_paper" if re.search(r"会议论文|conference paper", raw, re.I) else None),
        result_limit=20 if task in {LiteratureTask.ENUMERATE, LiteratureTask.INVENTORY} else 8,
        required_object_refs=(comparison_objects or (("object_1", "object_2") if task == LiteratureTask.COMPARE else ())),
        selection_policy=(
            "explicit_pair" if comparison_objects
            else "prior_result_order" if task == LiteratureTask.COMPARE and reference.kind != ReferenceKind.NONE
            else "discovery_rank_top_two" if task == LiteratureTask.COMPARE else None
        ),
        comparison_dimensions=(comparison_dimensions or (("method", "result") if task == LiteratureTask.COMPARE else ())),
        comparison_object_queries=comparison_objects,
    )
    return LiteratureSemantics(request)


def _task(text: str, *, reference: DocumentReference) -> LiteratureTask:
    positive = re.sub(
        r"(?:不要|不需要|(?:^|请|先|[,，;；。！？\s])别)\s*(?:总结|概括|比较|对比|对照|分析)",
        "",
        text,
    )
    if re.search(r"知识库|论文库|文献库|库里|库内", positive) and re.search(r"哪些|有什么|列出|多少|库存", positive):
        return LiteratureTask.INVENTORY
    if re.search(r"比较|对比|对照|区别|difference|compare", positive):
        return LiteratureTask.COMPARE
    if re.search(r"总结|综述|概括|主要讲|讲的什么|讲什么|分别讲|summary|summarize", positive):
        return LiteratureTask.SUMMARIZE
    if re.search(r"哪篇|有没有|定位|文件名|doi", positive):
        return LiteratureTask.LOCATE
    if (reference.kind == ReferenceKind.NONE
            and re.search(r"论文|papers?\b|文献", positive)
            and re.search(r"哪些|有什么|列出|查|检索|搜|找", positive)):
        return LiteratureTask.ENUMERATE
    return LiteratureTask.QA


def _explicit_identity(text: str) -> tuple[str | None, str | None, str | None]:
    filename_match = re.search(r"([^《》\n\r]{2,180}?\.pdf)", text, re.I)
    filename = filename_match.group(1).strip(" '\"") if filename_match else None
    quoted = re.search(r"《([^》]{2,180})》", text)
    title = quoted.group(1).strip() if quoted else None
    if not title and filename:
        title = re.sub(r"\.pdf$", "", filename, flags=re.I)
    if not title:
        suffix = re.search(r"^(.{4,160}?)(?:主要讲(?:的)?什么|讲的什么|讲什么|用了|使用了|的方法|的实验|的结论)$", text.strip())
        candidate = suffix.group(1).strip(" _") if suffix else ""
        if (suffix and len(candidate) >= 8
                and not re.search(r"^(?:这|该|它|第二|前两|上述|总结|概括|比较|对比)", candidate)):
            title = candidate
    author = None
    if title:
        stem = title
        split = re.match(r"(.+?)[_＿]([\u4e00-\u9fff]{2,4})$", stem)
        if split:
            title, author = split.group(1).strip(), split.group(2)
        else:
            match = re.search(r"([\u4e00-\u9fff]{2,4})(?:的|所写的).{0,30}(?:论文|文章)", text)
            author = match.group(1) if match else None
    return title, filename, author


def _comparison_objects(text: str) -> tuple[str, ...]:
    quoted = tuple(dict.fromkeys(
        item.strip() for item in re.findall(r"《([^》]{2,180})》", text) if item.strip()
    ))
    return quoted if len(quoted) == 2 else ()


def _reference(text: str, *, title: str | None, filename: str | None, author: str | None) -> DocumentReference:
    if title or filename:
        return DocumentReference(ReferenceKind.EXPLICIT_DOCUMENT, title=title, author=author, filename=filename)
    ordinal = re.search(r"第\s*([一二两三四五六七八九十\d]+)\s*篇", text)
    if ordinal:
        return DocumentReference(ReferenceKind.ORDINAL, ordinals=(_number(ordinal.group(1)),), expected_count=1)
    if re.search(r"这两篇|这2篇|前两篇|前2篇|它们|二者", text):
        return DocumentReference(ReferenceKind.PRIOR_RESULTS, expected_count=2)
    if re.search(r"上述论文|上述文献|这些论文|这些文献", text):
        return DocumentReference(ReferenceKind.PRIOR_RESULTS)
    if re.search(r"这篇|该论文|刚才那篇|它(?:的|用|采)", text):
        return DocumentReference(ReferenceKind.CURRENT_DOCUMENT, expected_count=1)
    if re.search(r"^(?:实验结果|结果|结论|方法|优化目标).*(?:怎么样|如何|是什么|有哪些|呢)?[？?]?$", text.strip()):
        return DocumentReference(ReferenceKind.CURRENT_DOCUMENT, expected_count=1)
    return DocumentReference()


def _topic(text: str, *, title: str | None) -> tuple[str | None, str | None, tuple[str, ...]]:
    if re.search(r"(?<![a-z])esn(?![a-z])|echo\s+state\s+network|回声状态网络", text, re.I):
        return "Echo State Network", "echo-state-network", (
            "Echo State Network", "ESN", "回声状态网络", "reservoir computing",
        )
    if re.search(r"(?<![a-z])mcmc(?![a-z])|markov\s+chain\s+monte\s+carlo|马尔科夫链蒙特卡", text, re.I):
        return "Markov Chain Monte Carlo", "markov-chain-monte-carlo", (
            "MCMC", "Markov Chain Monte Carlo", "马尔科夫链蒙特卡洛",
        )
    return (title, None, (title,)) if title else (None, None, ())


def _sections(text: str) -> tuple[RequestedSection, ...]:
    found = []
    mapping = (
        (r"摘要|abstract", RequestedSection.ABSTRACT),
        (r"引言|introduction", RequestedSection.INTRODUCTION),
        (r"方法|怎么做|用了什么|优化目标|method", RequestedSection.METHOD),
        (r"实验|experiment", RequestedSection.EXPERIMENT),
        (r"结果|性能|result", RequestedSection.RESULT),
        (r"结论|conclusion", RequestedSection.CONCLUSION),
        (r"参考文献|references", RequestedSection.REFERENCES),
    )
    for pattern, section in mapping:
        if re.search(pattern, text, re.I):
            found.append(section)
    return tuple(found)


def _number(value: str) -> int:
    if value.isdigit():
        return int(value)
    return {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}[value]
