from datetime import datetime, timedelta, timezone

import pytest

from agent.literature_ir import (
    BoundType, DocumentReference, LiteratureRequestIR, LiteratureTask,
    ReferenceKind, RequestedSection,
)
from agent.retrieval_plan import RetrievalStage
from agent.retrieval_planner import plan_retrieval
from agent.runtime.node_services import _bind_read_only_calls
from ingestion.chunking.structural import structural_chunks
from agent.evidence_bundle import build_grounded_answer_request, render_grounded_context
from agent.runtime.reference_resolver import ReferenceResolutionError, resolve_reference
from agent.runtime.turn_artifacts import DocumentResult, TurnResultArtifact
from agent.temporal_parser import parse_temporal_constraint
from nlu_v2.literature_semantics import analyze_literature


@pytest.mark.parametrize("text,expected", [
    ("2023年之后", {"gt": 2023}),
    ("2023年以来", {"gte": 2023}),
    ("截至2023年", {"lte": 2023}),
    ("2023年前", {"lt": 2023}),
    ("2019到2025年", {"gte": 2019, "lte": 2025}),
    ("2023年", {"gte": 2023, "lte": 2023}),
])
def test_temporal_bounds(text, expected):
    assert parse_temporal_constraint(text).to_es_range() == expected


def test_relative_time_is_reproducible():
    now = datetime(2026, 9, 9, tzinfo=timezone(timedelta(hours=8)))
    value = parse_temporal_constraint("近三年", now=now)
    assert value.to_es_range() == {"gte": 2024, "lte": 2026}
    assert value.relative_expression == "近三年"
    assert value.resolved_at and value.timezone == "Asia/Shanghai"


def _artifact(count=2):
    docs = tuple(DocumentResult(f"d{i}", f"p{i}.pdf", f"P{i}", (), 2024, i)
                 for i in range(1, count + 1))
    return TurnResultArtifact("t1", "enumerate", "g1", "snap", docs, "ESN", {})


def _request(reference, task=LiteratureTask.SUMMARIZE):
    return LiteratureRequestIR("这两篇分别讲什么", task,
        document_reference=reference, requested_sections=(RequestedSection.ABSTRACT,))


def test_prior_two_resolves_in_rank_order_and_plans_per_document():
    request = _request(DocumentReference(ReferenceKind.PRIOR_RESULTS, expected_count=2))
    resolved = resolve_reference(request, _artifact(), generation_id="g1")
    assert resolved.document_reference.document_ids == ("d1", "d2")
    plan = plan_retrieval(resolved)
    assert [step.document_ids for step in plan.steps] == [("d1",), ("d2",)]
    assert all(step.stage == RetrievalStage.PASSAGE_RETRIEVAL for step in plan.steps)


def test_ordinal_and_generation_fail_closed():
    request = _request(DocumentReference(ReferenceKind.ORDINAL, ordinals=(2,)))
    assert resolve_reference(request, _artifact(), generation_id="g1").document_reference.document_ids == ("d2",)
    with pytest.raises(ReferenceResolutionError):
        resolve_reference(request, _artifact(), generation_id="g2")


def test_current_document_question_with_naxie_is_qa_not_enumeration():
    request = analyze_literature("它用了哪些优化目标").request
    assert request.task == LiteratureTask.QA
    assert request.document_reference.kind == ReferenceKind.CURRENT_DOCUMENT


def test_qa_with_youshenme_is_not_document_enumeration():
    assert analyze_literature("ESN的谱半径有什么作用").request.task == LiteratureTask.QA


def test_topic_only_summary_starts_with_document_discovery():
    request = LiteratureRequestIR(
        "总结ESN文献", LiteratureTask.SUMMARIZE,
        topic_terms=("ESN",), canonical_topic="Echo State Network",
    )
    plan = plan_retrieval(request)
    assert [step.stage for step in plan.steps] == [RetrievalStage.DOCUMENT_DISCOVERY]


def test_exact_title_resolves_before_passage_search():
    title = "一种基于多目标优化的ESN网络架构搜索方法"
    request = LiteratureRequestIR(title + "主要讲什么", LiteratureTask.SUMMARIZE,
        topic_terms=("Echo State Network",), canonical_topic="Echo State Network",
        document_reference=DocumentReference(ReferenceKind.EXPLICIT_DOCUMENT, title=title))
    plan = plan_retrieval(request)
    assert plan.steps[0].stage == RetrievalStage.DOCUMENT_RESOLUTION
    assert plan.steps[0].query == title


def test_nlu_preserves_exact_title_and_topic_separately():
    value = analyze_literature("一种基于多目标优化的ESN网络架构搜索方法_张昭昭主要讲什么").request
    assert value.task == LiteratureTask.SUMMARIZE
    assert value.document_reference.title == "一种基于多目标优化的ESN网络架构搜索方法"
    assert value.document_reference.author == "张昭昭"
    assert value.canonical_topic == "Echo State Network"


def test_nlu_multiturn_reference_and_section():
    value = analyze_literature("第二篇用了什么方法").request
    assert value.document_reference.ordinals == (2,)
    assert value.requested_sections == (RequestedSection.METHOD,)
    assert value.task == LiteratureTask.QA


def test_nlu_respects_open_year_for_enumeration():
    value = analyze_literature("帮我查一下2023年之后的ESN文献").request
    assert value.task == LiteratureTask.ENUMERATE
    assert value.temporal.lower_bound == BoundType.OPEN


def test_binder_is_pure_plan_conversion():
    request = analyze_literature("帮我查一下2023年之后的ESN文献").request
    calls = _bind_read_only_calls(plan_retrieval(request))
    assert len(calls) == 1
    assert calls[0].skill_name == "discover_documents"
    assert calls[0].arguments["query"] == request.raw_query
    assert calls[0].arguments["temporal_range"] == {"gt": 2023}


def test_chunking_marks_references_and_abstract():
    chunks = structural_chunks(
        document_id="d", content_hash="h",
        markdown="# 摘要\n\n核心贡献。\n\n# 参考文献\n\n[1] cited work",
        generation="g", parser_version="p", embedding_version="e",
    )
    assert chunks[0].is_abstract and chunks[0].section_type == "abstract"
    assert chunks[1].is_reference and chunks[1].section_type == "references"


def test_grounded_bundle_rejects_cross_document_pollution():
    request = LiteratureRequestIR(
        "总结这篇", LiteratureTask.SUMMARIZE,
        document_reference=DocumentReference(ReferenceKind.EXPLICIT_DOCUMENT, document_ids=("d1",)),
    )
    with pytest.raises(ValueError):
        build_grounded_answer_request(request, [{"result": {"evidence": [{"document_id": "d2"}]}}])
    bundle = build_grounded_answer_request(request, [{"result": {"evidence": [
        {"document_id": "d1", "filename": "p.pdf", "page": 2, "text": "method evidence"},
    ]}}])
    rendered = render_grounded_context(bundle)
    assert "[文档 document_id=d1]" in rendered and "p.pdf#p2" in rendered


def test_inventory_has_dedicated_retrieval_stage():
    request = analyze_literature("库里有哪些论文").request
    plan = plan_retrieval(request)
    assert request.task == LiteratureTask.INVENTORY
    assert [step.stage for step in plan.steps] == [RetrievalStage.INVENTORY]
    calls = _bind_read_only_calls(plan)
    assert calls[0].skill_name == "kb_diagnose"
    assert calls[0].arguments["operation"] == "inventory"


@pytest.mark.parametrize("query", ["概括ESN的方法", "比较ESN论文的实验"])
def test_topic_summary_or_compare_is_not_misread_as_document_title(query):
    request = analyze_literature(query).request
    assert request.document_reference.kind == ReferenceKind.NONE
