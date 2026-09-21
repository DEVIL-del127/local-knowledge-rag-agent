import sys
import time
from pathlib import Path

import pytest

from agent.agent_limits import BudgetExceededError
from agent.budgeted_nlu_provider import BudgetedNluProvider
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.model_call_ledger import ModelOutcomeUnknown
from agent.runtime.checkpoint_store import SQLiteCheckpointStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kb-agent"))
from nlu_v2.llm_extractor import ProviderResponse


class Provider:
    model = "synthetic"
    base_url = "http://synthetic.invalid"
    max_response_tokens = 64
    max_context_tokens = 1024

    def __init__(self):
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        assert kwargs["attempt_budget"] == 1
        return ProviderResponse("{}")


def test_nlu_and_answer_share_pre_dispatch_budget(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    claim = store.begin_operation(user_id="u", session_id="s", thread_id="s", request_id="r",
                                  payload={}, expected_revision=0, max_model_calls=1)
    provider = Provider()
    wrapper = BudgetedNluProvider(provider, require_operation=True)
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"))
    sent = []
    client._post = lambda *args: sent.append(1)
    with store.operation_scope(claim):
        args = dict(deadline=time.monotonic() + 30, attempt_budget=1)
        assert wrapper.complete("synthetic", **args).text == "{}"
        assert wrapper.complete("synthetic", **args).attempts == 0
        with pytest.raises(BudgetExceededError):
            client.invoke_text(system_prompt="synthetic", user_prompt="answer", user_id="u")
    assert provider.calls == 1
    assert sent == []


def test_nlu_repair_budget_is_separate_and_checked_before_provider(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    claim = store.begin_operation(user_id="u", session_id="s", thread_id="s", request_id="r",
                                  payload={}, expected_revision=0, max_model_calls=3, max_repair_calls=0)
    provider = Provider()
    with store.operation_scope(claim), pytest.raises(BudgetExceededError):
        BudgetedNluProvider(provider).complete("repair", deadline=time.monotonic()+30,
                                               attempt_budget=1, purpose="nlu_repair")
    assert provider.calls == 0


def test_enforce_nlu_rejects_unscoped_external_calls():
    provider = Provider()
    with pytest.raises(RuntimeError, match="nlu_operation_required"):
        BudgetedNluProvider(provider, require_operation=True).complete(
            "synthetic", deadline=time.monotonic()+30, attempt_budget=1)
    assert provider.calls == 0
