import hashlib
import json
import time

import numpy as np
import pytest

import agent.semantic_compiler_adapter as compiler_module
import nlu_v2.engine as engine_module
from agent.retrieval_models import SemanticDecision
from agent.semantic_compiler_adapter import SemanticCompilerAdapter
from agent.semantic_matrix import EncodingStatus, EncodingUnavailable, SemanticMatrixRouter
from agent.source_span_audit import audit_source_spans


def _package(tmp_path, *, enabled=False):
    matrix = np.asarray([[1.0, 0.0], [0.99, 0.1], [0.98, -0.1],
                         [0.0, 1.0], [0.1, 0.99], [-0.1, 0.98]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(tmp_path / "matrix.npy", matrix, allow_pickle=False)
    examples = [{"category": "find", "cluster_id": f"f{i}"} for i in range(3)] + [
        {"category": "chat", "cluster_id": f"c{i}"} for i in range(3)]
    payload = {"matrix_file": "matrix.npy", "matrix_digest": hashlib.sha256(matrix.tobytes()).hexdigest(),
               "examples": examples, "categories": {
                   "find": {"enabled": enabled, "threshold": 0.9, "margin": 0.2},
                   "chat": {"enabled": False, "threshold": 0.9, "margin": 0.2}}}
    path = tmp_path / "package.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _semantic_backend():
    class Indices:
        def get_mapping(self, *, index):
            return {index: {"mappings": {"_meta": {"version": "test-v1"}, "properties": {
                "filename": {"type": "keyword"}, "content": {"type": "text"},
                "publication_year": {"type": "integer"},
            }}}}
    es = type("ES", (), {"indices": Indices()})()
    manager = type("Manager", (), {"es": es, "index_name": "documents"})()
    return type("Backend", (), {"es_manager": manager})()


def test_fixed_matrix_aggregates_three_clusters_but_stays_candidate_when_disabled(tmp_path):
    result = SemanticMatrixRouter(_package(tmp_path), lambda _: [1.0, 0.0]).route("找论文")
    assert result.category == "find" and result.score > 0.98 and result.margin > 0.9
    assert not result.auto_release and result.reason_code == "category_disabled"


def test_fixed_matrix_releases_only_explicitly_enabled_category(tmp_path):
    router = SemanticMatrixRouter(_package(tmp_path, enabled=True), lambda _: [1.0, 0.0])
    assert router.route("找论文").auto_release


def test_encoding_timeout_keeps_slot_unavailable_until_late_work_finishes(tmp_path):
    def slow(_):
        time.sleep(0.08)
        return [1.0, 0.0]
    router = SemanticMatrixRouter(_package(tmp_path), slow, timeout_seconds=0.01)
    with pytest.raises(EncodingUnavailable, match="encoding_timeout"):
        router.route("找论文")
    assert router.state()["timeout_pending"]
    with pytest.raises(EncodingUnavailable, match="encoding_slot_unavailable"):
        router.route("另一个请求")
    time.sleep(0.1)
    state = router.state()
    assert not state["timeout_pending"]
    assert state["status"] == EncodingStatus.FINISHED.value
    assert state["late_completions"] == 1 and state["last_completion_late"]


def test_three_timeouts_open_circuit_then_allow_exactly_one_probe(tmp_path):
    def slow(_):
        time.sleep(0.03)
        return [1.0, 0.0]
    router = SemanticMatrixRouter(
        _package(tmp_path), slow, timeout_seconds=0.005, circuit_seconds=0.1,
    )
    for _ in range(3):
        with pytest.raises(EncodingUnavailable, match="encoding_timeout"):
            router.route("找论文")
        time.sleep(0.04)
    assert router.state()["circuit_open"]
    with pytest.raises(EncodingUnavailable, match="encoding_circuit_open"):
        router.route("仍在熔断")
    time.sleep(0.11)
    with pytest.raises(EncodingUnavailable, match="encoding_timeout"):
        router.route("单次探针")
    with pytest.raises(EncodingUnavailable, match="encoding_slot_unavailable"):
        router.route("探针期间不能并发")


def test_health_confirmation_recovers_finished_circuit(tmp_path):
    router = SemanticMatrixRouter(_package(tmp_path), lambda _: [1.0, 0.0])
    router._consecutive_timeouts = 3
    router._opened_at = time.monotonic()
    router.confirm_service_recovery()
    assert router.state()["status"] == EncodingStatus.IDLE.value
    assert router.state()["consecutive_timeouts"] == 0


def test_source_span_audit_assigns_negation_and_allowed_find_compare_chain():
    audit = audit_source_spans("不要总结，找论文再比较前两篇")
    assert audit.complete
    assert {"negative_constraint", "find", "compare", "result_reference"} <= {
        item.owner for item in audit.assignments}


def test_source_span_audit_blocks_unsupported_action_combination():
    audit = audit_source_spans("总结后比较这篇论文")
    assert not audit.complete and audit.unassigned


def test_followup_compiler_cannot_override_source_span_block():
    adapter = SemanticCompilerAdapter(search_backend=_semantic_backend(), enable_llm=False)
    first = adapter.compile("2019年到2025年之间关于ESN的论文有哪些")
    followup = adapter.compile_with_context(
        "总结后比较这篇论文", first.to_dict(), previous_turn_id="turn-1",
    )
    assert followup.decision == SemanticDecision.BLOCKED
    assert not followup.executable
    assert followup.validation_status == "source_span_incomplete"
    assert "source_span_unassigned_action" in followup.diagnostics


def test_compile_path_runs_literature_projection_once(monkeypatch):
    calls = 0
    original = engine_module.analyze_literature

    def counted(query):
        nonlocal calls
        calls += 1
        return original(query)

    monkeypatch.setattr(engine_module, "analyze_literature", counted)
    adapter = SemanticCompilerAdapter(search_backend=_semantic_backend(), enable_llm=False)
    adapter.compile("找ESN论文")
    assert calls == 1
    assert not hasattr(compiler_module, "analyze_literature")


def test_compiler_exposes_one_accepted_literature_ir_and_catalog_binding():
    adapter = SemanticCompilerAdapter(search_backend=_semantic_backend(), enable_llm=False)
    result = adapter.compile("ESN的谱半径有什么作用")
    accepted = result.contract_reports["accepted_literature_ir"]
    assert result.executable
    assert accepted["digest"] == result.request_ir["accepted_ir_digest"]
    assert accepted["source_binding"]["binding_origin"] == "current_authorized_catalog"
    assert all(
        node["inputs"]["accepted_ir_digest"] == accepted["digest"]
        for node in result.logical_plan["nodes"]
    )
