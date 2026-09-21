from __future__ import annotations

import multiprocessing as mp
import time
import sqlite3
import pytest
from pathlib import Path
from dataclasses import asdict, replace
from types import SimpleNamespace
from core.retrieval_gateway import GenerationSnapshot

from agent.agent_models import AgentReply, IntentPrediction, IntentType
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.dialogue import PendingInteraction
from agent.runtime.models import AgentState
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.agent_limits import BudgetExceededError
from agent.model_call_ledger import ModelCallLedger, ModelCallState, ModelCallConflict


class NeverCompiler:
    def compile(self, *_args, **_kwargs):
        raise AssertionError("output action must not invoke compiler")


class CountedAnswer:
    def __init__(self, counter):
        self.counter = counter
        self.registry = SimpleNamespace(
            has=lambda name: name == "search_private_kb",
            get=lambda name: SimpleNamespace(gateway=SimpleNamespace(
                verify_snapshot=lambda snapshot, force: None)),
        )

    def answer_from_observations(self, *_args, **_kwargs):
        client = DeepSeekClient(DeepSeekSettings(api_key="test-only", thinking=False))
        def fake_post(*_args):
            with self.counter.get_lock():
                self.counter.value += 1
            time.sleep(0.05)
            return {"choices": [{"message": {"content": "synthetic-example"}}]}
        client._post = fake_post
        answer = client.invoke_text(system_prompt="synthetic", user_prompt="synthetic", user_id="u")
        return AgentReply(
            answer=answer,
            intent=IntentPrediction(IntentType.KB_TUTOR, False, 1.0, "synthetic"),
        )


def seed(path):
    store = SQLiteCheckpointStore(path)
    state = AgentState("prior", "u", "s", "s")
    state.last_substantive_turn = {"turn_id": "prior", "domain": "kb_document", "answer": "prior"}
    state.pending_interaction = PendingInteraction(
        action_id="a", action_type="example", payload={"style": "example"},
        source_turn_id="prior", expires_at_epoch=4102444800,
    )
    state.observations = [{"synthetic": True}]
    snapshot = GenerationSnapshot("synthetic", 1, "index", "collection", "model", 3)
    snapshot = replace(snapshot, snapshot_digest=snapshot.canonical_digest())
    state.context_snapshot["generation_snapshot"] = asdict(snapshot)
    state.request_execution_snapshot = {"generation_snapshot_digest": snapshot.snapshot_digest,
                                        "cache_digest": "original"}
    store.save(state)
    return store


def accept_worker(path, counter, ready, release, results, number):
    coordinator = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=SQLiteCheckpointStore(path))
    ready.put(number)
    release.wait(30)
    try:
        reply = coordinator.handle(
            "行", legacy=CountedAnswer(counter), user_id="u", session_id="s",
            request_id=f"accept-{number}",
        )
        results.put(reply.intent.reason)
    except Exception as exc:
        results.put(f"{type(exc).__name__}:{getattr(exc, 'sqlite_errorname', '')}:{exc}")


def test_v8_32_processes_accept_same_action_once(tmp_path):
    path = str(tmp_path / "state.sqlite3")
    seed(path)
    ctx = mp.get_context("spawn")
    counter, ready, release, results = ctx.Value("i", 0), ctx.Queue(), ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=accept_worker, args=(path, counter, ready, release, results, n)) for n in range(32)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            ready.get(timeout=90)
        release.set()
        outcomes = [results.get(timeout=60) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
        assert all(worker.exitcode == 0 for worker in workers)
        assert counter.value == 1
        assert set(outcomes) <= {"dialogue_action_consumed", "runtime_operation_in_progress", "runtime_operation_replayed"}
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


def test_v8_public_request_replays_after_new_coordinator(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = seed(path)
    counter = mp.get_context("spawn").Value("i", 0)
    request = dict(legacy=CountedAnswer(counter), user_id="u", session_id="s", request_id="same")
    first = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store).handle("行", **request)
    second = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=SQLiteCheckpointStore(path)).handle("行", **request)
    assert first.answer == second.answer == "synthetic-example"
    assert second.intent.reason == "runtime_operation_replayed"
    assert counter.value == 1


def test_v8_changed_payload_for_same_public_id_is_rejected(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    coordinator = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store)
    args = dict(legacy=object(), user_id="u", session_id="s", request_id="same")
    assert coordinator.handle("1+1等于几", **args).answer == "1+1 = 2"
    assert coordinator.handle("2+2等于几", **args).intent.reason == "runtime_operation_conflict"


@pytest.mark.parametrize("damage", ["missing", "tampered", "execution_mismatch"])
def test_output_action_refuses_unverified_evidence_before_model(tmp_path, damage):
    store = seed(tmp_path / "state.sqlite3")
    state = store.load("u", "s", "s")
    if damage == "missing":
        state.context_snapshot.pop("generation_snapshot")
    elif damage == "tampered":
        state.context_snapshot["generation_snapshot"]["generation_id"] = "changed"
    else:
        state.request_execution_snapshot["generation_snapshot_digest"] = "changed"
    store.save(state)
    counter = mp.get_context("spawn").Value("i", 0)
    reply = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store).handle(
        "行", legacy=CountedAnswer(counter), user_id="u", session_id="s")
    assert reply.intent.reason == "dialogue_snapshot_unavailable"
    assert counter.value == 0


def test_v8_empty_response_retry_cannot_exceed_one_attempt(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    claim = store.begin_operation(user_id="u",session_id="s",thread_id="s",request_id="r",
                                  payload={"synthetic": True},expected_revision=0,max_model_calls=1)
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only", thinking=False))
    sends = []
    def fake_post(*_args):
        sends.append(1)
        return {"choices": [{"message": {"content": ""}}]}
    client._post = fake_post
    with store.operation_scope(claim), pytest.raises(BudgetExceededError):
        client.invoke_text(system_prompt="synthetic", user_prompt="synthetic", user_id="u")
    assert sends == [1]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM runtime_attempt_reservations").fetchone()[0] == 1


def test_v8_expired_owner_is_not_automatically_reexecuted(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    args = dict(user_id="u",session_id="s",thread_id="s",request_id="r",payload={},expected_revision=0)
    claim = store.begin_operation(**args)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE runtime_operations SET lease_until=0 WHERE operation_id=?", (claim.operation_id,))
    restarted = SQLiteCheckpointStore(store.path)
    assert restarted.begin_operation(**args).status == "unknown"


def test_v8_checkpoint_and_reply_commit_roll_back_together(tmp_path):
    store = seed(tmp_path / "state.sqlite3")
    args = dict(user_id="u",session_id="s",thread_id="s",request_id="r",payload={},expected_revision=1)
    claim = store.begin_operation(**args)
    state = store.load("u", "s", "s")
    with store.operation_scope(claim):
        state.pending_interaction.state = "consumed"
        store.save(state)
        assert store.load("u", "s", "s").pending_interaction.state == "pending"
        with sqlite3.connect(store.path) as connection:
            connection.execute("CREATE TRIGGER fail_reply BEFORE UPDATE OF result ON runtime_operations BEGIN SELECT RAISE(ABORT,'synthetic'); END")
        with pytest.raises(sqlite3.IntegrityError):
            store.complete_operation(claim, {"answer":"synthetic"})
    assert store.load("u", "s", "s").pending_interaction.state == "pending"


def test_v8_lease_expiry_between_prepare_and_dispatch_is_rejected(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    claim = store.begin_operation(user_id="u",session_id="s",thread_id="s",request_id="r",payload={},expected_revision=0)
    ledger = ModelCallLedger(store.path)
    record = ledger.prepare(provider="fake",purpose="answer",logical_request_id="r",attempt=1,
                            endpoint_identity="fake",model="fake",payload_digest="fake",estimated_tokens=1,
                            session_id="s",operation_id=claim.operation_id,operation_owner=claim.owner)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE runtime_operations SET lease_until=0")
    with pytest.raises(ModelCallConflict):
        ledger.transition(record.call_id,expected=ModelCallState.PREPARED,target=ModelCallState.DISPATCH_INTENT,
                          operation_id=claim.operation_id,operation_owner=claim.owner)
    assert ledger.get(record.call_id).state == ModelCallState.PREPARED


def test_v8_replay_is_not_admitted_to_memory():
    from memory.memory_manager import MemoryManager
    prediction = IntentPrediction(IntentType.KB_TUTOR,False,1.0,"synthetic",reason="runtime_operation_replayed")
    assert MemoryManager._admit_turn(prediction) is False


def owned_attempt(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "owned.sqlite3")
    claim = store.begin_operation(user_id="u", session_id="s", thread_id="s",
                                  request_id="r", payload={}, expected_revision=0)
    ledger = ModelCallLedger(store.path)
    record = ledger.prepare(provider="fake", purpose="answer", logical_request_id="r", attempt=1,
                            endpoint_identity="fake", model="fake", payload_digest="fake",
                            estimated_tokens=1, session_id="s", operation_id=claim.operation_id,
                            operation_owner=claim.owner)
    return store, claim, ledger, record


def test_v8_dispatch_cannot_omit_bound_owner(tmp_path):
    _, _, ledger, record = owned_attempt(tmp_path)
    with pytest.raises(ModelCallConflict):
        ledger.transition(record.call_id, expected=ModelCallState.PREPARED,
                          target=ModelCallState.DISPATCH_INTENT)
    assert ledger.get(record.call_id).state == ModelCallState.PREPARED


def test_v8_dispatch_cannot_borrow_another_live_operation(tmp_path):
    store, _, ledger, record = owned_attempt(tmp_path)
    other = store.begin_operation(user_id="u", session_id="other", thread_id="other",
                                  request_id="r2", payload={}, expected_revision=0)
    with pytest.raises(ModelCallConflict):
        ledger.transition(record.call_id, expected=ModelCallState.PREPARED,
                          target=ModelCallState.DISPATCH_INTENT,
                          operation_id=other.operation_id, operation_owner=other.owner)


def test_v8_recovery_preserves_live_owner_then_recovers_expired(tmp_path):
    store, claim, ledger, record = owned_attempt(tmp_path)
    ledger.transition(record.call_id, expected=ModelCallState.PREPARED,
                      target=ModelCallState.DISPATCH_INTENT,
                      operation_id=claim.operation_id, operation_owner=claim.owner)
    assert ledger.recover_uncertain() == 0
    assert ledger.get(record.call_id).state == ModelCallState.DISPATCH_INTENT
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE runtime_operations SET lease_until=0")
    assert ledger.recover_uncertain() == 1
    assert ledger.get(record.call_id).state == ModelCallState.OUTCOME_UNKNOWN
    assert ledger.recover_uncertain() == 0
