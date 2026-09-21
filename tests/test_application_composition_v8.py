from types import SimpleNamespace

import pytest

from agent.lazy_backend import LazySearchBackend
from agent.application import AgentApplication
from agent.agent_service import PrivateKnowledgeAgent, AgentSettings
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.checkpoint_store import SQLiteCheckpointStore


def test_non_kb_registration_and_calculation_do_not_initialize_backend(tmp_path):
    def forbidden():
        raise AssertionError("backend initialized for non-KB request")
    backend = LazySearchBackend(forbidden, label="test", pdf_dir=str(tmp_path), index_name="synthetic")
    legacy = PrivateKnowledgeAgent(search_backend=backend, settings=AgentSettings(state_dir=str(tmp_path)))
    runtime = RuntimeCoordinator(compiler=object(), checkpoint_store=SQLiteCheckpointStore(tmp_path / "runtime.sqlite3"))
    assert runtime.handle("1+1等于几", legacy=legacy).answer == "1+1 = 2"
    assert runtime.handle("帮助", legacy=legacy).answer
    assert backend._value is None


def test_application_closes_owned_services_once_even_on_failure():
    calls = []
    def end(**kwargs):
        calls.append("memory")
        raise RuntimeError("synthetic")
    app = AgentApplication(None, "test", "u", SimpleNamespace(on_session_end=end), "s",
                           SimpleNamespace(shutdown=lambda: calls.append("recorder")),
                           SimpleNamespace(shutdown=lambda: calls.append("bus")))
    with pytest.raises(RuntimeError, match="synthetic"):
        app.close()
    app.close()
    assert calls == ["memory", "bus", "recorder"]


def test_real_agent_wires_general_and_output_generation_without_backend(tmp_path):
    from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
    def forbidden():
        raise AssertionError("private backend accessed")
    backend = LazySearchBackend(forbidden, label="test", pdf_dir=str(tmp_path), index_name="synthetic")
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"))
    sent = []
    def post(url, payload):
        sent.append(payload)
        return {"choices": [{"message": {"content": f"generated-{len(sent)}"}}]}
    client._post = post
    legacy = PrivateKnowledgeAgent(search_backend=backend, deepseek_client=client,
                                   settings=AgentSettings(state_dir=str(tmp_path)))
    runtime = RuntimeCoordinator(compiler=object(), checkpoint_store=SQLiteCheckpointStore(tmp_path / "runtime.sqlite3"))
    reply = runtime.handle("写一段生日祝福", legacy=legacy)
    assert reply.answer == "generated-1"
    reply = runtime.handle("换一种说法", legacy=legacy)
    assert reply.answer == "generated-2"
    assert len(sent) == 2
    assert backend._value is None
