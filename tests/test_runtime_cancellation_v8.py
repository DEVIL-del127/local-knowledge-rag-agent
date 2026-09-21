import pytest


def test_cancelled_public_worker_returns_typed_reply_without_late_commit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    entered, release = Event(), Event()
    def invoke(**kwargs):
        entered.set()
        assert release.wait(10)
        return "late answer"
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(compiler=object(), checkpoint_store=store)
    legacy = SimpleNamespace(deepseek=SimpleNamespace(invoke_text=invoke))
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(runtime.handle, "写一段生日祝福", legacy=legacy,
                             user_id="u", session_id="s", request_id="old")
        try:
            assert entered.wait(10)
            cancelled = runtime.handle("取消", legacy=legacy, user_id="u", session_id="s", request_id="cancel")
            assert cancelled.intent.reason == "pending_action_cancelled"
        finally:
            release.set()
        assert worker.result(timeout=10).intent.reason == "runtime_operation_cancelled"
    assert not store.load("u", "s", "s").last_substantive_turn

from agent.agent_limits import BudgetExceededError
from agent.agent_skills import SkillRegistry, SkillSpec
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.operations import OperationConflict
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.models import AgentState, PendingClarification
from agent.runtime.dialogue import DialogueActResolver, DialogueActKind


@pytest.mark.parametrize("message", ["简单介绍一下Transformer", "举个例子解释贝叶斯定理", "通俗解释一下梯度下降"])
def test_style_with_new_subject_is_not_prior_answer_action(message):
    act = DialogueActResolver().resolve(message, pending=None, has_prior_substantive_turn=True)
    assert act.kind == DialogueActKind.NEW_REQUEST


@pytest.mark.parametrize("message", ["简单讲讲", "通俗一点", "换一种说法"])
def test_unbound_style_still_targets_previous_answer(message):
    act = DialogueActResolver().resolve(message, pending=None, has_prior_substantive_turn=True)
    assert act.kind == DialogueActKind.REPHRASE


def claim(store, request, **kwargs):
    return store.begin_operation(user_id="u", session_id="s", thread_id="s", request_id=request,
                                  payload={"request": request}, expected_revision=0, **kwargs)


def test_cancel_fences_old_commit_and_tools(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    old = claim(store, "old")
    cancel = claim(store, "cancel", cancel_existing=True)
    assert cancel.status == "claimed"
    with pytest.raises(OperationConflict):
        store.complete_operation(old, {"answer": "old"})
    with pytest.raises(OperationConflict):
        store.reserve_tool_attempt(old)
    store.complete_operation(cancel, {"answer": "cancelled"})
    assert claim(store, "old").status == "cancelled"
    newer = claim(store, "new")
    assert claim(store, "cancel", cancel_existing=True).status == "succeeded"
    store.reserve_tool_attempt(newer)  # A replayed cancellation cannot cancel newer work.


def test_registry_budget_checked_before_invoke(tmp_path):
    class Skill:
        spec = SkillSpec("fake", "synthetic")
        calls = 0
        def invoke(self, arguments):
            self.calls += 1
            return {}
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    operation = claim(store, "request", max_tool_calls=1)
    skill, registry = Skill(), SkillRegistry()
    registry.register(skill)
    with store.operation_scope(operation):
        registry.execute("fake", {})
        with pytest.raises(BudgetExceededError):
            registry.execute("fake", {})
    assert skill.calls == 1


@pytest.mark.parametrize("message", ["取消", "1+1等于几", "帮助"])
def test_pending_clarification_does_not_capture_controls(tmp_path, message):
    class Compiler:
        def resume(self, *args, **kwargs):
            raise AssertionError("old clarification captured a new control")
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    state = AgentState("previous", "u", "s", "s")
    state.pending_clarification = PendingClarification("old", "q", "choose", "candidate_id", ["A", "B"])
    store.save(state)
    reply = RuntimeCoordinator(compiler=Compiler(), checkpoint_store=store).handle(
        message, legacy=object(), user_id="u", session_id="s", request_id="next")
    assert reply.answer
    assert store.load("u", "s", "s").pending_clarification is None


def test_expired_clarification_cannot_resume(tmp_path):
    class Compiler:
        def resume(self, *args, **kwargs):
            raise AssertionError("expired clarification resumed")
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    state = AgentState("previous", "u", "s", "s")
    state.pending_clarification = PendingClarification("old", "q", "choose", "candidate_id", ["A"],
                                                      expires_at="2000-01-01T00:00:00+00:00")
    store.save(state)
    reply = RuntimeCoordinator(compiler=Compiler(), checkpoint_store=store).handle(
        "A", legacy=object(), user_id="u", session_id="s", request_id="next")
    assert reply.intent.reason == "clarification_expired"
    assert store.load("u", "s", "s").pending_clarification is None
