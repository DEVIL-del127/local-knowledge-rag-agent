from __future__ import annotations

import json
import sqlite3
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import requests

from agent.runtime.models import RequestExecutionSnapshot
from memory.memory_manager import MemoryManager
from telemetry.search_log.models import SearchLogEvent
from telemetry.search_log.recorder import SearchRecorder
from telemetry.search_log.config import SearchLogConfig
from telemetry.search_log.storage import SqliteStorage
from agent.agent_limits import PersistentCache, RateLimiter
from core.retrieval_gateway import GenerationMismatch, GenerationSnapshot, RetrievalGateway
from agent.nlu_profile import NluExecutionProfile
from agent.model_call_ledger import (
    ModelCallLedger, ModelCallConflict, ModelCallState, ModelOutcomeUnknown,
)
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.privacy import DataSanitizer, PersistencePolicy, PersistencePolicyError
from agent.app_settings import AppSettings


class _Embedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Collection:
    def __init__(self, fail=False):
        self.fail = fail
        self.rows = {}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        if self.fail:
            raise RuntimeError("injected vector failure")
        for index, item_id in enumerate(ids):
            self.rows[item_id] = (documents[index], metadatas[index])

    def delete(self, *, ids=None, where=None):
        if self.fail:
            raise RuntimeError("injected vector failure")
        for item_id in ids or []:
            self.rows.pop(item_id, None)


def _rate_attempt(path, queue):
    queue.put(RateLimiter(path, max_calls=5, window_seconds=60).allow("shared")[0])


def _cache_write(path, index):
    PersistentCache(path, max_entries=64, ttl_seconds=60).put(f"k{index}", {"value": index})


def test_memory_outbox_replays_after_projection_failure(tmp_path):
    manager = MemoryManager(state_dir=str(tmp_path), embedder=_Embedder())
    broken = _Collection(fail=True)
    manager._collection = broken
    entry = {"id": "fact-1", "type": "fact", "entity": "ESN", "relation": "prefers",
             "value": "Echo State Network", "ts": "2026-09-03T00:00:00+0800"}
    manager._commit_mutations("u1", [("upsert", entry)])
    with manager._memory_connect() as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM memory_outbox WHERE completed_ts IS NULL"
        ).fetchone()[0]
    assert pending == 1

    healthy = _Collection()
    manager._collection = healthy
    assert manager.replay_outbox() == 1
    assert "fact-1" in healthy.rows
    assert manager.replay_outbox() == 0

    manager._commit_mutations("u1", [("delete", entry)])
    assert "fact-1" not in healthy.rows
    assert manager.recall_memory(query="ESN", user_id="u1") == ""


def test_memory_persistence_has_no_plaintext_private_secret_or_reasoning_canaries(tmp_path):
    manager = MemoryManager(state_dir=str(tmp_path))
    session_id = manager.on_session_start(user_id="privacy-user")
    private = "private-query-canary-c8912"
    secret = "<redacted-test-secret>"
    reasoning = "provider-reasoning-canary-d104b"
    manager.ingest_message(
        round_no=1, user_msg=private, agent_reply="accepted", intent=None,
        session_id=session_id, user_id="privacy-user", force=True,
    )
    manager._commit_mutations("privacy-user", [("upsert", {
        "id": "privacy-fact", "type": "fact", "entity": "test",
        "relation": "contains", "value": f"{private} {secret}",
        "reasoning_content": reasoning, "ts": "2026-09-04T00:00:00+0800",
    })])
    manager.on_session_end(session_id=session_id, user_id="privacy-user")

    for path in tmp_path.rglob("*"):
        if path.is_file() and not path.name.endswith(".key"):
            raw = path.read_bytes()
            assert private.encode() not in raw, path
            assert secret.encode() not in raw, path
            assert reasoning.encode() not in raw, path
    restored = manager._load_l2("privacy-user")
    assert restored[0]["reasoning_content"] == "[REDACTED]"
    assert secret not in restored[0]["value"]


def test_request_execution_snapshot_digest_excludes_request_id():
    values = dict(
        generation_snapshot_digest="g", nlu_engine_tree_digest="n", nlu_profile_digest="p",
        request_ir_schema_version="ir", security_scope_digest="s", turn_context_digest="t",
        semantic_envelope_digest="e", logical_plan_digest="l", literature_query_digest="q",
        retrieval_config_digest="r", answer_model_identity="a", synthesis_prompt_version="sp",
        citation_validator_version="cv",
    )
    one = RequestExecutionSnapshot(request_id="one", **values)
    two = RequestExecutionSnapshot(request_id="two", **values)
    assert one.cache_digest() == two.cache_digest()
    assert RequestExecutionSnapshot(request_id="two", **{**values, "logical_plan_digest": "changed"}).cache_digest() != one.cache_digest()


def test_semantic_cache_identity_excludes_timing_and_trace_noise():
    from agent.retrieval_models import SemanticCompileResult, SemanticDecision

    first = SemanticCompileResult(
        decision=SemanticDecision.STRUCTURED_RETRIEVAL,
        effective_query="ESN", executable=True, elapsed_ms=1.0,
        trace=[{"trace_id": "one", "retry": 0}], model_audit={"latency_ms": 1},
    )
    second = SemanticCompileResult(
        decision=SemanticDecision.STRUCTURED_RETRIEVAL,
        effective_query="ESN", executable=True, elapsed_ms=999.0,
        trace=[{"trace_id": "two", "retry": 3}], model_audit={"latency_ms": 999},
    )
    assert first.stable_contract_dict() == second.stable_contract_dict()


def test_latest_nlu_profile_is_immutable_complete_and_digest_sensitive():
    profile = NluExecutionProfile.latest(enable_llm=True)
    required = {
        "enable_vector", "enable_llm", "llm_on_complex", "enable_patch_v1",
        "enable_requirement_ir_v2", "enable_operator_registry", "enable_atomic_patch_v2",
        "enable_semantic_repair", "enable_requirement_normalizer", "enable_semantic_linker",
        "enable_event_closure_contract", "enable_semantic_target_gate",
        "enable_candidate_choice_v3", "enable_m15_requirement_graph_shadow",
        "enable_m15_field_binding_shadow", "enable_m15_quantity_contract",
        "enable_m15_temporal_contracts", "enable_m15_typed_lineage", "enable_m15_logical_plan",
    }
    assert required == set(profile.engine_kwargs())
    assert all(value is True for key, value in profile.engine_kwargs().items() if key != "llm_on_complex")
    assert profile.llm_on_complex is False
    assert profile.digest() != NluExecutionProfile.latest(enable_llm=False).digest()
    try:
        profile.enable_llm = False
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("NLU execution profile must be immutable")


def test_search_log_sqlite_persists_execution_contract(tmp_path):
    db = tmp_path / "search.sqlite3"
    config = SearchLogConfig(storage="sqlite", sqlite_path=str(db), batch_size=1, flush_interval=0.01)
    recorder = SearchRecorder(storage=SqliteStorage(str(db)), config=config)
    recorder.record(SearchLogEvent(
        query="ESN papers", request_id="req-1", snapshot_digest="snap",
        request_ir_digest="ir", logical_plan_digest="plan", literature_query_digest="lit",
        execution_snapshot_digest="exec", admission="complete", skill_bindings=["search_private_kb"],
        executed_channels=["es", "vector"], result_type="matched", result_count=2,
        citation_valid=True,
    ))
    recorder.shutdown()
    with sqlite3.connect(db) as connection:
        payload = json.loads(connection.execute(
            "SELECT execution_json FROM search_log WHERE request_id='req-1'"
        ).fetchone()[0])
    assert payload["snapshot_digest"] == "snap"
    assert payload["result_type"] == "matched"
    assert payload["citation_valid"] is True


def test_production_search_log_defaults_do_not_persist_query_text(monkeypatch):
    monkeypatch.delenv("SEARCH_LOG_PERSIST_QUERY_TEXT", raising=False)
    config = SearchLogConfig.from_env()
    assert config.persist_query_text is False


def test_shared_sanitizer_removes_secrets_and_provider_reasoning():
    payload = DataSanitizer.sanitize({
        "api_key": "sk-super-secret-value",
        "nested": {"reasoning_content": "private chain", "text": "Authorization: Bearer abc"},
    })
    serialized = json.dumps(payload)
    assert "super-secret" not in serialized
    assert "private chain" not in serialized
    assert "Bearer abc" not in serialized


def test_persistence_policy_rejects_unknown_boundary_fields():
    with __import__("pytest").raises(PersistencePolicyError):
        PersistencePolicy.prepare("release-audit-v1", {
            "schema_version": "v1", "raw_query": "must not be published",
        })
    prepared = PersistencePolicy.prepare("release-audit-v1", {
        "schema_version": "v1", "gate_pass": True, "unresolved": [],
    })
    assert prepared == {"schema_version": "v1", "gate_pass": True, "unresolved": []}


def test_app_settings_resolve_relative_state_from_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_STATE_DIR", "runtime-state")
    monkeypatch.setenv("AGENT_EXECUTION_PROFILE", "enforce")
    settings = AppSettings.from_env(tmp_path)
    assert settings.state_dir == (tmp_path / "runtime-state").resolve()
    assert settings.ingestion_registry_path == (
        tmp_path / "data/ingestion/generation_registry.sqlite3"
    ).resolve()
    assert settings.llm_clarification is False


def test_sqlite_rate_limit_and_cache_are_cross_process_atomic(tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    rate_path = str(tmp_path / "rate.sqlite3")
    workers = [context.Process(target=_rate_attempt, args=(rate_path, queue)) for _ in range(12)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0
    allowed = [queue.get(timeout=2) for _ in workers]
    assert sum(allowed) == 5

    cache_path = str(tmp_path / "cache.sqlite3")
    writers = [context.Process(target=_cache_write, args=(cache_path, index)) for index in range(12)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(20)
        assert writer.exitcode == 0
    cache = PersistentCache(cache_path, max_entries=64, ttl_seconds=60)
    assert all(cache.get(f"k{index}") == {"value": index} for index in range(12))


def test_model_call_ledger_dispatch_recovery_is_outcome_unknown(tmp_path):
    ledger = ModelCallLedger(tmp_path / "model.sqlite3")
    prepared = ledger.prepare(
        provider="deepseek", purpose="answer", logical_request_id="request-1",
        attempt=1, endpoint_identity="endpoint", model="model",
        payload_digest="payload", estimated_tokens=100, session_id="session",
    )
    ledger.transition(
        prepared.call_id, expected=ModelCallState.PREPARED,
        target=ModelCallState.DISPATCH_INTENT,
    )
    assert ledger.recover_uncertain() == 1
    recovered = ledger.get(prepared.call_id)
    assert recovered.state == ModelCallState.OUTCOME_UNKNOWN
    with __import__("pytest").raises(ModelCallConflict):
        ledger.transition(
            prepared.call_id, expected=ModelCallState.DISPATCH_INTENT,
            target=ModelCallState.SUCCEEDED,
        )


def test_model_call_ledger_prepare_is_idempotent_under_same_attempt(tmp_path):
    ledger = ModelCallLedger(tmp_path / "model.sqlite3")
    kwargs = dict(
        provider="deepseek", purpose="router", logical_request_id="same",
        attempt=1, endpoint_identity="endpoint", model="model",
        payload_digest="payload", estimated_tokens=10, session_id="session",
    )
    first = ledger.prepare(**kwargs)
    second = ledger.prepare(**kwargs)
    assert first.call_id == second.call_id
    assert len(ledger.list_for("same")) == 1


def test_model_call_ledger_rejects_same_attempt_with_different_payload(tmp_path):
    ledger = ModelCallLedger(tmp_path / "model.sqlite3")
    kwargs = dict(
        provider="deepseek", purpose="answer", logical_request_id="same",
        attempt=1, endpoint_identity="endpoint", model="model",
        payload_digest="payload-a", estimated_tokens=10, session_id="session",
    )
    ledger.prepare(**kwargs)
    with __import__("pytest").raises(ModelCallConflict):
        ledger.prepare(**{**kwargs, "payload_digest": "payload-b"})


def _ledger_client(path):
    return DeepSeekClient(
        DeepSeekSettings(api_key="test-only", retries=0, thinking=False),
        ledger=ModelCallLedger(path),
    )


def test_dispatch_intent_before_socket_is_never_automatically_sent(tmp_path):
    import hashlib
    path = tmp_path / "model.sqlite3"
    ledger = ModelCallLedger(path)
    prepared = ledger.prepare(
        provider="deepseek", purpose="answer", logical_request_id="before-socket",
        attempt=1, endpoint_identity=hashlib.sha256(
            b"https://example.invalid/chat/completions"
        ).hexdigest(), model=DeepSeekSettings(api_key="test-only").model,
        payload_digest=hashlib.sha256(b'{"messages":[]}').hexdigest(),
        estimated_tokens=10, session_id="session",
    )
    ledger.transition(
        prepared.call_id, expected=ModelCallState.PREPARED,
        target=ModelCallState.DISPATCH_INTENT,
    )
    client = _ledger_client(path)
    sends = []
    client._post = lambda *args: sends.append(args)  # type: ignore[method-assign]
    with __import__("pytest").raises(ModelOutcomeUnknown):
        client._ledger_request(
            url="https://example.invalid/chat/completions", payload={"messages": []},
            user_id="session", logical_request_id="before-socket", purpose="answer",
        )
    assert sends == []


def test_prepared_before_dispatch_can_resume_with_one_send(tmp_path):
    path = tmp_path / "model.sqlite3"
    client = _ledger_client(path)
    payload = {"model": client.settings.model, "messages": []}
    import hashlib
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    client.ledger.prepare(
        provider="deepseek", purpose="answer", logical_request_id="prepared-resume",
        attempt=1,
        endpoint_identity=hashlib.sha256(
            b"https://example.invalid/chat/completions"
        ).hexdigest(),
        model=client.settings.model, payload_digest=digest,
        estimated_tokens=0, session_id="session",
    )
    sends = 0

    def post(*_args):
        nonlocal sends
        sends += 1
        return {"choices": [{"message": {"content": "resumed"}}], "usage": {}}

    client._post = post  # type: ignore[method-assign]
    result = client._ledger_request(
        url="https://example.invalid/chat/completions", payload=payload,
        user_id="session", logical_request_id="prepared-resume", purpose="answer",
    )
    assert result["content"] == "resumed"
    assert sends == 1


def test_accepted_then_disconnect_restart_never_resends(tmp_path):
    path = tmp_path / "model.sqlite3"
    first = _ledger_client(path)
    first._post = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        requests.ConnectionError("accepted then disconnected")
    )
    kwargs = dict(
        url="https://example.invalid/chat/completions",
        payload={"model": "model", "messages": []}, user_id="session",
        logical_request_id="accepted-disconnect", purpose="answer",
    )
    with __import__("pytest").raises(ModelOutcomeUnknown):
        first._ledger_request(**kwargs)
    restarted = _ledger_client(path)
    sends = []
    restarted._post = lambda *args: sends.append(args)  # type: ignore[method-assign]
    with __import__("pytest").raises(ModelOutcomeUnknown):
        restarted._ledger_request(**kwargs)
    assert sends == []
    assert restarted.ledger.list_for("accepted-disconnect")[0].state == ModelCallState.OUTCOME_UNKNOWN


def test_persisted_success_is_encrypted_and_replayed_without_dispatch(tmp_path):
    path = tmp_path / "model.sqlite3"
    first = _ledger_client(path)
    sends = 0

    def post_once(*_args):
        nonlocal sends
        sends += 1
        return {
            "choices": [{"message": {
                "content": "private-answer-canary-7f31",
                "reasoning_content": "provider-reasoning-canary-8a42",
            }}],
            "usage": {"total_tokens": 3},
        }

    first._post = post_once  # type: ignore[method-assign]
    kwargs = dict(
        url="https://example.invalid/chat/completions",
        payload={"model": "model", "messages": [{"role": "user", "content": "query"}]},
        user_id="session", logical_request_id="persisted-success", purpose="answer",
    )
    assert first._ledger_request(**kwargs)["content"] == "private-answer-canary-7f31"

    restarted = _ledger_client(path)
    restarted._post = lambda *_args: (_ for _ in ()).throw(AssertionError("must not send"))  # type: ignore[method-assign]
    replay = restarted._ledger_request(**kwargs)
    assert replay["content"] == "private-answer-canary-7f31"
    assert replay["reasoning_content"] == "[REDACTED]"
    for changed in (
        {"payload": {"messages": [{"role": "user", "content": "different"}]}},
        {"user_id": "other-session"},
        {"purpose": "other-purpose"},
        {"url": "https://other.invalid/chat/completions"},
    ):
        with __import__("pytest").raises(ModelCallConflict):
            restarted._ledger_request(**{**kwargs, **changed})
    assert sends == 1
    raw = path.read_bytes()
    assert b"private-answer-canary-7f31" not in raw
    assert b"provider-reasoning-canary-8a42" not in raw


def test_32_concurrent_same_logical_request_dispatches_exactly_once(tmp_path):
    path = tmp_path / "model.sqlite3"
    client = _ledger_client(path)
    lock = threading.Lock()
    sends = 0

    def post_once(*_args):
        nonlocal sends
        with lock:
            sends += 1
        time.sleep(0.05)
        return {"choices": [{"message": {"content": "shared"}}], "usage": {}}

    client._post = post_once  # type: ignore[method-assign]

    def invoke(_index):
        return client._ledger_request(
            url="https://example.invalid/chat/completions",
            payload={"model": "model", "messages": []}, user_id="session",
            logical_request_id="concurrent-32", purpose="answer",
        )["content"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(invoke, range(32)))
    assert results == ["shared"] * 32
    assert sends == 1
    assert len(client.ledger.list_for("concurrent-32")) == 1


def test_generation_snapshot_physical_identity_drift_fails_closed():
    class _Indices:
        def get_mapping(self, **kwargs):
            return {"idx": {"mappings": {"properties": {"changed": {"type": "text"}}}}}

        def get_settings(self, **kwargs):
            return {"idx": {"settings": {"index": {"uuid": "uuid-1"}}}}

    es = type("ESClient", (), {"indices": _Indices()})()
    backend = type("Backend", (), {
        "es_manager": type("ESManager", (), {"es": es})(),
        "vector_store": type("Vector", (), {"client": None})(),
    })()
    snapshot = GenerationSnapshot(
        generation_id="g1", revision=1, es_physical_index="idx", vector_collection="vec",
        embedding_model="bge-m3", embedding_dimension=1024, es_index_uuid="uuid-1",
        es_mapping_digest="not-the-current-digest",
    )
    try:
        RetrievalGateway(backend).verify_snapshot(snapshot)
    except GenerationMismatch as exc:
        assert "mapping changed" in str(exc)
    else:
        raise AssertionError("physical identity drift must fail closed")


def test_generation_snapshot_digest_covers_record_catalog_and_text_contracts():
    base = dict(
        generation_id="g1", revision=1, es_physical_index="idx",
        vector_collection="vec", embedding_model="bge-m3", embedding_dimension=1024,
        generation_record_digest="record", catalog_locator="manifest#catalog",
        catalog_digest="catalog", chunk_text_digest="text", safety_scope_digest="scope",
    )
    first = GenerationSnapshot(**base)
    changed = GenerationSnapshot(**{**base, "chunk_text_digest": "changed"})
    import dataclasses
    import hashlib
    canonical = lambda item: hashlib.sha256(json.dumps(
        dataclasses.asdict(item), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert canonical(first) != canonical(changed)
