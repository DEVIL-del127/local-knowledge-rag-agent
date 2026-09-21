from types import SimpleNamespace
import sqlite3

from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator


class NeverCompiler:
    def compile(self, *args, **kwargs):
        raise AssertionError("general task reached private compiler")


def test_general_task_generates_once_without_private_context(tmp_path):
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only", thinking=True))
    sent = []
    def post(url, payload):
        sent.append(payload)
        return {"choices": [{"message": {"content": "synthetic translation"}}]}
    client._post = post
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    runtime = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store)
    reply = runtime.handle("翻译这句话：早上好", legacy=SimpleNamespace(deepseek=client),
                           user_id="u", session_id="s", request_id="r")
    assert reply.answer == "synthetic translation"
    assert len(sent) == 1
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert len(sent[0]["messages"]) == 2
    assert not reply.evidence
    replay = runtime.handle("翻译这句话：早上好", legacy=SimpleNamespace(deepseek=client),
                            user_id="u", session_id="s", request_id="r")
    assert replay.answer == reply.answer
    assert len(sent) == 1


def test_general_empty_response_is_not_retried_or_remembered(tmp_path):
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"))
    sent = []
    def post(*args):
        sent.append(1)
        return {"choices": [{"message": {"content": ""}}]}
    client._post = post
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    reply = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store).handle(
        "写一段生日祝福", legacy=SimpleNamespace(deepseek=client), user_id="u", session_id="s")
    assert reply.intent.reason == "general_chat_generation_failed"
    assert len(sent) == 1
    assert store.load("u", "s", "s").last_substantive_turn == {}


def test_rephrase_generates_new_answer_once_and_replays(tmp_path):
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"))
    sent = []
    def post(url, payload):
        sent.append(payload)
        return {"choices": [{"message": {"content": "first" if len(sent) == 1 else "rewritten"}}]}
    client._post = post
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    runtime = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store)
    args = dict(legacy=SimpleNamespace(deepseek=client), user_id="u", session_id="s")
    runtime.handle("写一段生日祝福", request_id="first", **args)
    reply = runtime.handle("换一种说法", request_id="rewrite", **args)
    assert reply.answer == "rewritten"
    assert len(sent) == 2
    assert "first" in sent[1]["messages"][-1]["content"]
    assert runtime.handle("换一种说法", request_id="rewrite", **args).answer == "rewritten"
    assert len(sent) == 2
    assert store.load("u", "s", "s").last_substantive_turn["answer"] == "rewritten"


def test_general_answer_clears_old_private_evidence(tmp_path):
    from agent.runtime.models import AgentState
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    state = AgentState("prior", "u", "s", "s")
    state.observations = [{"private": "canary"}]
    state.context_snapshot = {"literature": {"canonical_topic": "private-canary"}}
    state.last_substantive_turn = {"domain": "kb_document", "answer": "private-canary"}
    store.save(state)
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"))
    sent = []
    def post(url, payload):
        sent.append(payload)
        return {"choices": [{"message": {"content": "public answer"}}]}
    client._post = post
    runtime = RuntimeCoordinator(compiler=NeverCompiler(), checkpoint_store=store)
    args = dict(legacy=SimpleNamespace(deepseek=client), user_id="u", session_id="s")
    runtime.handle("写一段生日祝福", **args)
    loaded = store.load("u", "s", "s")
    assert loaded.observations == []
    assert loaded.context_snapshot == {}
    runtime.handle("换一种说法", **args)
    assert len(sent) == 2
    assert "canary" not in str(sent)
