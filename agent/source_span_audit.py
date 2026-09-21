from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class SpanAssignment:
    kind: str
    text: str
    start: int
    end: int
    owner: str
    requirement_ids: tuple[str, ...] = ()
    plan_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceSpanAudit:
    assignments: tuple[SpanAssignment, ...]
    unassigned: tuple[SpanAssignment, ...]
    complete: bool
    schema_version: str = "source-span-audit-v2"

    def to_dict(self) -> dict[str, object]:
        return {"assignments": [asdict(item) for item in self.assignments],
                "unassigned": [asdict(item) for item in self.unassigned],
                "complete": self.complete, "schema_version": self.schema_version}


_PATTERNS = (
    ("negation", re.compile(r"不要|别|不需要|排除|不含")),
    ("year", re.compile(r"(?<!\d)(?:19|20)\d{2}年?")),
    ("quantity", re.compile(r"前(?:两|二|三|四|五|\d+)篇|\d+篇")),
    ("object", re.compile(r"第(?:一|二|三|四|五|\d+)篇|这篇|那篇|它|上述")),
    ("action", re.compile(r"检索|查找|找(?:一下)?|搜索|总结|概括|比较|对比|分析")),
)


def audit_source_spans(
    query: str,
    requirement_ledger: Iterable[dict[str, Any]] | None = None,
    accepted_requirements: Iterable[dict[str, Any]] | None = None,
    compiled_plan: dict[str, Any] | None = None,
    literature_contract: dict[str, Any] | None = None,
) -> SourceSpanAudit:
    ledger = list(requirement_ledger or [])
    accepted_ids = {
        str(item.get("requirement_id") or "") for item in (accepted_requirements or [])
        if str(item.get("status") or "") not in {"rejected", "uncovered"}
    }
    nodes = list((compiled_plan or {}).get("nodes") or [])
    assignments: list[SpanAssignment] = []
    actions: list[str] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(query or ""):
            text = match.group(0)
            owner = _owner(kind, text)
            if kind == "action" and re.search(r"不要|别|不需要", (query or "")[max(0, match.start() - 4):match.start()]):
                owner = "negative_constraint"
            requirement_ids = tuple(sorted({
                str(item.get("requirement_id") or "") for item in ledger
                if _overlaps(match.start(), match.end(), item.get("span"))
                and (not accepted_ids or str(item.get("requirement_id") or "") in accepted_ids)
            } - {""}))
            plan_node_ids = tuple(
                str(node.get("task_id") or "") for node in nodes
                if _node_owns(owner, str(node.get("task_type") or ""), literature_contract or {})
            )
            item = SpanAssignment(
                kind, text, match.start(), match.end(), owner,
                requirement_ids=requirement_ids,
                plan_node_ids=tuple(item for item in plan_node_ids if item),
            )
            assignments.append(item)
            if kind == "action" and owner != "negative_constraint":
                actions.append(owner)
    allowed = not actions or actions in (["find"], ["summarize"], ["compare"],
                                          ["find", "summarize"], ["find", "compare"])
    unassigned = [] if allowed else [item for item in assignments if item.kind == "action"]
    if compiled_plan is not None and allowed:
        unassigned = [
            item for item in assignments
            if item.owner not in {"negative_constraint"} and not item.plan_node_ids
            and item.kind in {"action", "quantity", "object"}
        ]
    return SourceSpanAudit(tuple(sorted(assignments, key=lambda item: (item.start, item.end, item.kind))),
                           tuple(unassigned), allowed and not unassigned)


def _overlaps(start: int, end: int, span: Any) -> bool:
    if not isinstance(span, dict):
        return False
    return max(start, int(span.get("start", -1))) < min(end, int(span.get("end", -1)))


def _node_owns(owner: str, task_type: str, contract: dict[str, Any]) -> bool:
    contract_task = contract.get("task")
    contract_task = str(getattr(contract_task, "value", contract_task) or "")
    if owner == "find":
        return task_type in {"discover_documents", "resolve_document", "retrieve"}
    if owner in {"summarize", "compare"}:
        return (
            task_type in {"discover_documents", "resolve_document", "retrieve_document_passages"}
            and contract_task == owner
        )
    if owner in {"result_reference", "object_reference"}:
        return task_type in {"discover_documents", "resolve_document", "retrieve_document_passages"}
    if owner == "year_filter":
        return task_type in {"discover_documents", "retrieve"}
    return owner == "negative_constraint"


def _owner(kind: str, text: str) -> str:
    if kind == "action":
        if re.search(r"检索|查找|找|搜索", text):
            return "find"
        if re.search(r"总结|概括", text):
            return "summarize"
        if re.search(r"比较|对比", text):
            return "compare"
        return "unsupported_action"
    return {"negation": "negative_constraint", "year": "year_filter",
            "quantity": "result_reference", "object": "object_reference"}[kind]
