from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.literature_ir import LiteratureRequestIR


@dataclass(frozen=True, slots=True)
class GroundedAnswerRequest:
    original_query: str
    task: str
    selected_documents: tuple[str, ...]
    evidence_by_document: dict[str, tuple[dict[str, Any], ...]]
    requested_sections: tuple[str, ...]
    citation_policy: str = "filename-and-page-once"


def build_grounded_answer_request(
    request: LiteratureRequestIR, observations: list[dict[str, Any]],
) -> GroundedAnswerRequest:
    grouped: dict[str, list[dict[str, Any]]] = {}
    selected = tuple(request.document_reference.document_ids)
    for observation in observations:
        for item in list((observation.get("result") or {}).get("evidence") or []):
            document_id = str(item.get("document_id") or "")
            if selected and document_id not in selected:
                raise ValueError("evidence outside selected document scope")
            grouped.setdefault(document_id, []).append(dict(item))
    return GroundedAnswerRequest(
        original_query=request.raw_query, task=request.task.value,
        selected_documents=selected,
        evidence_by_document={key: tuple(value) for key, value in grouped.items()},
        requested_sections=tuple(item.value for item in request.requested_sections),
    )


def render_grounded_context(request: GroundedAnswerRequest) -> str:
    sections = [
        f"任务：{request.task}", f"原始问题：{request.original_query}",
        "规则：只能依据以下按文档隔离的证据回答；缺少方法或结果证据时必须明确说明；禁止根据标题推测。",
    ]
    for document_id, evidence in request.evidence_by_document.items():
        sections.append(f"\n[文档 document_id={document_id}]")
        for item in evidence:
            source = str(item.get("filename") or "未知文件")
            page = item.get("page")
            label = f"{source}#p{page}" if page else source
            text = str(item.get("text") or item.get("content") or item.get("highlights") or "")
            sections.append(f"[{label}] {text}")
    return "\n".join(sections)
