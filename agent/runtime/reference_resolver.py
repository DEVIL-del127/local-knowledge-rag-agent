from __future__ import annotations

from dataclasses import replace

from agent.literature_ir import DocumentReference, LiteratureRequestIR, ReferenceKind
from agent.runtime.turn_artifacts import TurnResultArtifact


class ReferenceResolutionError(ValueError):
    pass


def resolve_reference(
    request: LiteratureRequestIR, artifact: TurnResultArtifact | None,
    *, generation_id: str,
) -> LiteratureRequestIR:
    reference = request.document_reference
    if reference.document_ids:
        return request
    if reference.kind in {ReferenceKind.NONE, ReferenceKind.EXPLICIT_DOCUMENT}:
        return request
    if artifact is None:
        raise ReferenceResolutionError("上一轮没有可定位的论文，请先指定论文标题。")
    if artifact.generation_id != generation_id:
        raise ReferenceResolutionError("知识库版本已变化，请重新检索论文后再使用指代。")
    documents = list(artifact.documents)
    if reference.kind == ReferenceKind.CURRENT_DOCUMENT:
        if len(documents) != 1:
            raise ReferenceResolutionError("上一轮不是唯一论文，请明确指定论文标题。")
        selected = documents
    elif reference.kind == ReferenceKind.ORDINAL:
        if not reference.ordinals or max(reference.ordinals) > len(documents):
            raise ReferenceResolutionError("上一轮没有对应序号的论文，请重新指定。")
        selected = [documents[index - 1] for index in reference.ordinals]
    else:
        count = reference.expected_count
        if count is not None and len(documents) < count:
            raise ReferenceResolutionError(f"上一轮没有{count}篇可定位论文，请先重新检索。")
        selected = documents[:count] if count else documents
    resolved = replace(reference, document_ids=tuple(item.document_id for item in selected))
    return replace(request, document_reference=resolved)
