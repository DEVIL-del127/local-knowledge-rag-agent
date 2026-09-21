from __future__ import annotations

from agent.literature_ir import LiteratureRequestIR, LiteratureTask, ReferenceKind
from agent.retrieval_plan import RetrievalPlan, RetrievalStage, RetrievalStep


def plan_retrieval(request: LiteratureRequestIR) -> RetrievalPlan:
    ref = request.document_reference
    if request.task == LiteratureTask.INVENTORY:
        return RetrievalPlan((RetrievalStep(
            RetrievalStage.INVENTORY, request.raw_query, limit=request.result_limit,
        ),))
    if request.task == LiteratureTask.ENUMERATE:
        return RetrievalPlan((RetrievalStep(
            RetrievalStage.DOCUMENT_DISCOVERY, request.raw_query,
            topic_terms=request.topic_terms, temporal=request.temporal, limit=request.result_limit,
        ),))
    if request.task == LiteratureTask.COMPARE and request.comparison_object_queries:
        return RetrievalPlan(tuple(
            RetrievalStep(
                RetrievalStage.DOCUMENT_RESOLUTION, identity,
                topic_terms=(), limit=request.result_limit,
            )
            for identity in request.comparison_object_queries
        ))
    if ref.kind == ReferenceKind.EXPLICIT_DOCUMENT and not ref.document_ids:
        identity = ref.filename or ref.title or request.raw_query
        return RetrievalPlan((RetrievalStep(
            RetrievalStage.DOCUMENT_RESOLUTION, identity,
            topic_terms=request.topic_terms, limit=request.result_limit,
        ),))
    if ref.document_ids:
        return RetrievalPlan(tuple(RetrievalStep(
            RetrievalStage.PASSAGE_RETRIEVAL, request.raw_query,
            document_ids=(document_id,), requested_sections=request.requested_sections,
            limit=request.result_limit,
        ) for document_id in ref.document_ids))
    # Topic-only summarize/compare is deliberately two-phase. Discovery chooses
    # concrete documents; the runtime then retrieves each selected document in
    # a separate, document-ID-scoped call.
    return RetrievalPlan((RetrievalStep(
        RetrievalStage.DOCUMENT_DISCOVERY, request.raw_query,
        topic_terms=request.topic_terms, temporal=request.temporal, limit=request.result_limit,
    ),))
