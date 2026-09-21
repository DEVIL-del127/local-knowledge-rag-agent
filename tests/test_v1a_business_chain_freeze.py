import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.literature_ir import LiteratureTask
from agent.routing_service import RoutingService
from agent.runtime.models import AgentState, BusinessOutcome, ExecutionBudget
from agent.runtime.node_services import (
    RuntimeNodeServices, _comparison_coverage, _finalize_business_outcome,
)
from agent.source_span_audit import audit_source_spans
from nlu_v2.literature_semantics import analyze_literature


ROOT = Path(__file__).resolve().parents[1]


def _sealed_cases(version="v1"):
    path = ROOT / f"data/routing/v1a/business_chain_sealed_{version}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize("case", _sealed_cases(), ids=lambda item: item["id"])
def test_sealed_business_chain_expression(case):
    previous = None
    if case.get("context") == "two_results":
        previous = SimpleNamespace(
            accepted_task_state={"task": "literature", "current_object_ids": ["d1", "d2"]},
            needs_revalidation=False,
        )
    decision, hint = RoutingService().route(case["query"], previous=previous)
    assert decision.domain.value == case["expected_domain"]
    if case.get("expected_mode"):
        modes = [item.mode.value for item in hint.context_proposals if item.valid]
        assert modes == [case["expected_mode"]]
    if case.get("expected_task"):
        assert analyze_literature(case["query"]).request.task.value == case["expected_task"]


@pytest.mark.parametrize("case", _sealed_cases("v2"), ids=lambda item: item["id"])
def test_sealed_business_chain_holdout_v2(case):
    test_sealed_business_chain_expression(case)


@pytest.mark.parametrize("case", _sealed_cases("v3"), ids=lambda item: item["id"])
def test_sealed_business_chain_holdout_v3(case):
    test_sealed_business_chain_expression(case)


@pytest.mark.parametrize("case", _sealed_cases("v4"), ids=lambda item: item["id"])
def test_sealed_business_chain_holdout_v4(case):
    test_sealed_business_chain_expression(case)


def test_source_span_audit_links_obligation_to_plan_node():
    query = "找ESN论文再比较前两篇"
    ledger = [{
        "requirement_id": "req-find", "status": "covered",
        "span": {"start": 0, "end": 1, "text": "找"},
    }]
    plan = {"nodes": [
        {"task_id": "select", "task_type": "discover_documents"},
        {"task_id": "compare-evidence", "task_type": "retrieve_document_passages"},
    ]}
    audit = audit_source_spans(
        query, ledger, ledger, plan, {"task": "compare"},
    )
    assert audit.complete
    find = next(item for item in audit.assignments if item.owner == "find")
    compare = next(item for item in audit.assignments if item.owner == "compare")
    assert find.requirement_ids == ("req-find",)
    assert find.plan_node_ids == ("select",)
    assert compare.plan_node_ids == ("select", "compare-evidence")


def test_source_span_audit_links_two_phase_compare_to_discovery_node():
    audit = audit_source_spans(
        "比较两篇ESN论文", [], [],
        {"nodes": [{"task_id": "select", "task_type": "discover_documents"}]},
        {"task": "compare"},
    )
    assert audit.complete
    assert next(item for item in audit.assignments if item.owner == "compare").plan_node_ids == ("select",)


def test_source_span_audit_accepts_enum_task_from_engine_contract():
    audit = audit_source_spans(
        "比较两篇ESN论文", [], [],
        {"nodes": [{"task_id": "select", "task_type": "discover_documents"}]},
        {"task": LiteratureTask.COMPARE},
    )
    assert audit.complete


def test_compare_contract_declares_selection_dimensions_and_objects():
    request = analyze_literature("比较前两篇ESN论文的方法和结论").request
    assert request.task == LiteratureTask.COMPARE
    assert request.required_object_refs == ("object_1", "object_2")
    assert request.selection_policy == "prior_result_order"
    assert request.comparison_dimensions == ("method", "conclusion")


def test_compare_contract_defaults_to_retrievable_dimensions():
    request = analyze_literature("ESN与回声状态网络的区别").request
    assert tuple(item.value for item in request.requested_sections) == ("method", "result")
    assert request.comparison_dimensions == ("method", "result")


def test_explicit_pair_compare_compiles_two_resolution_steps():
    from agent.retrieval_planner import plan_retrieval

    request = analyze_literature(
        "比较《Adaptive model based on ESN for anomaly detection in industrial systems》"
        "和《一种基于多目标优化的ESN网络架构搜索方法_张昭昭》的方法和结论"
    ).request
    assert request.selection_policy == "explicit_pair"
    assert len(request.comparison_object_queries) == 2
    plan = plan_retrieval(request)
    assert [step.stage.value for step in plan.steps] == [
        "document_resolution", "document_resolution",
    ]
    assert [step.query for step in plan.steps] == list(request.comparison_object_queries)


def test_runtime_budget_can_complete_two_document_comparison():
    assert ExecutionBudget().max_tool_calls >= 5


def test_business_finalizer_rejects_false_answer_and_false_empty_result():
    state = AgentState("r", "u", "s", "t")
    with pytest.raises(ValueError, match="grounded evidence"):
        _finalize_business_outcome(state, BusinessOutcome.ANSWERED, "done", evidence_count=0)
    with pytest.raises(ValueError, match="backend call"):
        _finalize_business_outcome(state, BusinessOutcome.NO_MATCH, "empty", backend_called=False)


def test_business_finalizer_records_terminal_failure():
    state = AgentState("r", "u", "s", "t")
    _finalize_business_outcome(state, BusinessOutcome.FAILED, "runtime_binding_unavailable")
    assert state.business_outcome == BusinessOutcome.FAILED.value
    assert state.reason_code == "runtime_binding_unavailable"


def test_business_finalizer_requires_complete_compare_coverage():
    state = AgentState("r", "u", "s", "t")
    with pytest.raises(ValueError, match="complete object coverage"):
        _finalize_business_outcome(
            state, BusinessOutcome.ANSWERED, "done", backend_called=True,
            evidence_count=2, comparison_coverage={"complete": False},
        )


def test_compare_dimension_coverage_requires_each_dimension_for_each_document():
    evidence = [
        {"document_id": "d1", "section_type": "method"},
        {"document_id": "d1", "section_type": "conclusion"},
        {"document_id": "d2", "section_type": "method"},
    ]
    dimensions = ("method", "conclusion")
    coverage = _comparison_coverage(
        evidence, required_count=2, dimensions=dimensions,
        selection_policy="prior_result_order",
    )
    assert coverage["dimension_coverage"] == {
        "d1": {"method": True, "conclusion": True},
        "d2": {"method": True, "conclusion": False},
    }
    assert not coverage["complete"]


def test_gateway_compatibility_finds_legacy_read_only_skill():
    gateway = object()
    skill = SimpleNamespace(gateway=gateway)
    registry = SimpleNamespace(
        has=lambda name: name == "search_private_kb",
        get=lambda name: skill,
    )
    assert RuntimeNodeServices._gateway(SimpleNamespace(registry=registry)) is gateway


def test_production_build_application_wires_v1a_router(monkeypatch, tmp_path):
    from agent.app_settings import AppSettings
    from agent.application import build_application

    monkeypatch.setenv("AGENT_EXECUTION_PROFILE", "enforce")
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_DB", "test")
    monkeypatch.setenv("AGENT_USER_ID", "v1a-smoke")
    monkeypatch.setenv("SEMANTIC_COMPILER_LLM", "0")
    monkeypatch.setenv("AGENT_LLM_CLARIFICATION", "0")
    app = build_application(AppSettings.from_env(ROOT))
    try:
        assert app.agent.mode == "enforce"
        assert app.agent.coordinator.routing_service.semantic_router is not None
        assert app.agent.coordinator.compiler.require_active_generation
    finally:
        app.close()
    assert app.closed
