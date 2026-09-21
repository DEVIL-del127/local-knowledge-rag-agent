from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.routing_contracts import Applicability, ContextMode
from agent.routing_service import RoutingService
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.domain_router import RequestDomain
from agent.runtime.models import AgentState, PendingClarification
from agent.runtime.node_services import RuntimeNodeServices
from web.database_app import app


def test_greeting_does_not_truncate_literature_request():
    decision, hint = RoutingService().route("你好，找ESN论文")
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert hint.applicability == Applicability.BUSINESS


def test_unknown_is_not_silently_general_chat():
    decision, hint = RoutingService().route("帮我处理一下")
    assert decision.domain == RequestDomain.CLARIFY_DOMAIN
    assert hint.applicability == Applicability.UNKNOWN
    assert hint.requires_interpretation


def test_unsafe_mutation_and_contextless_followup_fail_closed():
    decision, _ = RoutingService().route("修改数据库记录")
    assert decision.domain == RequestDomain.UNSUPPORTED
    decision, _ = RoutingService().route("分析它")
    assert decision.domain == RequestDomain.CLARIFY_DOMAIN


def test_explicit_general_writing_task_is_not_misrouted_to_kb():
    decision, _ = RoutingService().route("请调整这段话，让表达更客气")
    assert decision.domain == RequestDomain.GENERAL_CHAT


def test_reference_without_bound_object_is_invalid_candidate():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": []},
        needs_revalidation=False,
    )
    _, hint = RoutingService().route("比较第二篇", previous=previous, literature_reference="ordinal")
    inherited = [item for item in hint.context_proposals if item.mode == ContextMode.INHERIT]
    assert inherited and not inherited[0].valid
    assert inherited[0].reason_code == "referenced_object_missing"


def test_explicit_topic_change_does_not_inherit_literature_state():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["doc-1"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("换个话题，写一封邮件", previous=previous)
    assert decision.domain == RequestDomain.GENERAL_CHAT
    assert [item.mode for item in hint.context_proposals] == [ContextMode.NEW]


def test_bound_comparison_routes_to_kb_with_unique_inherit_candidate():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1", "d2"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("它和刚才那个比呢", previous=previous)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert decision.reason_code == "kb_bound_object_reference"
    assert [item.mode for item in hint.context_proposals if item.valid] == [ContextMode.INHERIT]


def test_two_valid_context_candidates_with_different_effects_clarify():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("换成Transformer", previous=previous)
    assert decision.domain == RequestDomain.CLARIFY_DOMAIN
    assert hint.reason_code == "context_candidate_conflict"


def test_complete_new_topic_with_year_is_new_request_not_patch_conflict():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["old-doc"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("查找2024年Transformer论文", previous=previous)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert [item.mode for item in hint.context_proposals if item.valid] == [ContextMode.NEW]


def test_inherited_object_and_condition_form_one_effective_patch():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("这篇只看2024年的", previous=previous)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    valid = [item for item in hint.context_proposals if item.valid]
    assert len(valid) == 1
    assert valid[0].mode == ContextMode.PATCH
    assert valid[0].object_ids == ("d1",)


def test_explicit_new_topic_does_not_rebind_old_object_even_with_pronoun():
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["old-doc"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route(
        "换个话题，查找Transformer论文，那个方向近年有哪些工作", previous=previous,
    )
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert [item.mode for item in hint.context_proposals if item.valid] == [ContextMode.NEW]


@pytest.mark.parametrize("query", ["那2024年的呢", "2023年的呢"])
def test_year_fragment_is_a_patch_only_with_active_results(query):
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1", "d2"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route(query, previous=previous)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert [item.mode for item in hint.context_proposals if item.valid] == [ContextMode.PATCH]


@pytest.mark.parametrize("query", ["总结一下", "对比一下", "比较前两篇ESN论文的方法和结论"])
def test_context_actions_inherit_active_result_objects(query):
    previous = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1", "d2"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route(query, previous=previous)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert any(item.valid and item.object_ids == ("d1", "d2") for item in hint.context_proposals)


def test_continue_depends_on_active_context_and_empty_result_cannot_bind_second_item():
    active = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": ["d1"]},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("继续", previous=active)
    assert decision.domain == RequestDomain.KB_DOCUMENT
    assert [item.mode for item in hint.context_proposals if item.valid] == [ContextMode.INHERIT]

    empty = SimpleNamespace(
        accepted_task_state={"task": "literature", "current_object_ids": []},
        needs_revalidation=False,
    )
    decision, hint = RoutingService().route("第二篇", previous=empty)
    assert decision.domain == RequestDomain.CLARIFY_DOMAIN
    inherited = [item for item in hint.context_proposals if item.mode == ContextMode.INHERIT]
    assert inherited and not inherited[0].valid


def test_backend_creates_distinct_conversations_and_requires_request_identity():
    client = TestClient(app)
    first = client.post("/api/conversations", json={"database": "main"})
    second = client.post("/api/conversations", json={"database": "main"})
    assert first.status_code == second.status_code == 201
    assert first.json()["conversation_id"] != second.json()["conversation_id"]
    missing = client.post("/api/chat", json={"message": "你好", "database": "main", "history": []})
    assert missing.status_code == 422


def test_new_state_copies_accepted_routing_lifecycle_fields():
    previous = AgentState("old", "u", "s", "t")
    previous.accepted_task_state = {"task": "find", "current_object_ids": ["d1"]}
    previous.dependency_router_versions = ["router-v1"]
    previous.needs_revalidation = True
    previous.routing_hint = {"reason_code": "old"}
    service = SimpleNamespace(store=SimpleNamespace(_current_operation=ContextVar("op", default=None)))
    state = RuntimeNodeServices._new_state(service, "u", "s", "t", "继续", previous=previous)
    assert state.accepted_task_state == previous.accepted_task_state
    assert state.accepted_task_state is not previous.accepted_task_state
    assert state.dependency_router_versions == ["router-v1"]
    assert state.needs_revalidation is True
    assert state.routing_hint == {"reason_code": "old"}


def test_remember_reply_writes_and_clears_accepted_task_state():
    state = AgentState("r", "u", "s", "t", raw_query="找ESN论文")
    state.context_snapshot = {
        "literature_ir": {"task": "find", "canonical_topic": "ESN"},
        "current_document_ids": ["d1"],
        "last_result_artifact_ref": {"artifact_id": "a1"},
    }
    state.domain_decision = {"router_version": "router-v1"}
    state.routing_hint = {"context_proposals": []}
    reply = SimpleNamespace(
        answer="ok", intent=SimpleNamespace(intent=SimpleNamespace(value="kb")),
        suggested_actions=[],
    )
    RuntimeNodeServices._remember_reply(SimpleNamespace(), state, reply, domain=RequestDomain.KB_DOCUMENT)
    assert state.accepted_task_state["current_object_ids"] == ["d1"]
    assert state.dependency_router_versions == ["router-v1"]
    RuntimeNodeServices._remember_reply(SimpleNamespace(), state, reply, domain=RequestDomain.GENERAL_CHAT)
    assert state.accepted_task_state == {}
    assert state.dependency_router_versions == []
    assert state.context_snapshot == {}


def test_two_conversations_keep_different_selected_documents(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    first = AgentState("r1", "u", "browser-1", "browser-1")
    first.accepted_task_state = {"task": "literature", "current_object_ids": ["doc-a"]}
    second = AgentState("r2", "u", "browser-2", "browser-2")
    second.accepted_task_state = {"task": "literature", "current_object_ids": ["doc-b"]}
    store.save(first)
    store.save(second)
    assert store.load("u", "browser-1", "browser-1").accepted_task_state["current_object_ids"] == ["doc-a"]
    assert store.load("u", "browser-2", "browser-2").accepted_task_state["current_object_ids"] == ["doc-b"]


def test_new_independent_request_does_not_answer_old_clarification(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    state = AgentState("old", "u", "s", "s")
    state.pending_clarification = PendingClarification(
        "旧问题", "q1", "请选择论文", "candidate_id", ["a", "b"],
    )
    store.save(state)
    runtime = RuntimeCoordinator(compiler=object(), checkpoint_store=store)
    reply = runtime.handle("1+1等于几", legacy=object(), user_id="u", session_id="s", request_id="new")
    assert reply.intent.reason == "bounded_expression_complete"
    assert store.load("u", "s", "s").pending_clarification is None


def test_replayed_request_does_not_reencode_semantic_candidate(tmp_path):
    class Semantic:
        def __init__(self):
            self.calls = 0

        def route(self, query):
            self.calls += 1
            return SimpleNamespace(
                category="general_chat", score=0.99, margin=0.5,
                auto_release=True, reason_code="released",
            )

    semantic = Semantic()
    routing = RoutingService(semantic_router=semantic)
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    runtime = RuntimeCoordinator(
        compiler=object(), checkpoint_store=store, routing_service=routing,
    )
    request = dict(legacy=object(), user_id="u", session_id="s", request_id="same")
    first = runtime.handle("帮我处理一下", **request)
    replay = runtime.handle("帮我处理一下", **request)
    assert first.intent.reason == "general_chat_provider_unavailable"
    assert replay.intent.reason == "runtime_operation_replayed"
    assert semantic.calls == 1
