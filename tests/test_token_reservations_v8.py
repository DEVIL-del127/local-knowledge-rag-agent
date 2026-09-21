import sqlite3
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.agent_limits import BudgetExceededError
from agent.model_call_ledger import ModelCallLedger, ModelCallState


@pytest.mark.parametrize("content", ["broken", "[]", '{"date":"bad","users":{}}',
                                    '{"date":"2026-09-05","users":{"u":{"used":-1,"calls":1}}}'])
def test_corrupt_legacy_usage_fails_closed(tmp_path, content):
    from agent.agent_limits import TokenBudget
    path = tmp_path / "usage.json"
    path.write_text(content, encoding="utf-8")
    budget = TokenBudget(str(path))
    allowed, reason = budget.check_and_consume(1)
    assert not allowed
    assert reason
    assert path.read_text(encoding="utf-8") == content


def process_reserve(path, ready, release, results, number):
    ledger = ModelCallLedger(path)
    ready.put(number)
    release.wait(30)
    try:
        reserve(ledger, str(number))
        results.put("reserved")
    except BudgetExceededError:
        results.put("denied")
    except Exception as exc:
        results.put(type(exc).__name__)


def test_multiprocess_reservations_share_ceiling(tmp_path):
    path = str(tmp_path / "ledger.sqlite3")
    ModelCallLedger(path)
    ctx = mp.get_context("spawn")
    ready, release, results = ctx.Queue(), ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=process_reserve, args=(path, ready, release, results, n)) for n in range(16)]
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
        assert outcomes.count("reserved") == 1
        assert outcomes.count("denied") == 15
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1
        assert connection.execute("SELECT MAX(used) FROM token_budget_buckets").fetchone()[0] == 6


@pytest.mark.parametrize("session_limit,daily_limit", [(0, 100000), (100000, 0)])
def test_public_client_budget_denial_sends_nothing(tmp_path, session_limit, daily_limit):
    from types import SimpleNamespace
    from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
    from agent.runtime.checkpoint_store import SQLiteCheckpointStore

    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    claim = store.begin_operation(user_id="u", session_id="s", thread_id="s",
                                  request_id="r", payload={}, expected_revision=0)
    # Legacy guard intentionally permits the request: exercise the new atomic gate.
    budget = SimpleNamespace(session_limit=session_limit, daily_limit=daily_limit,
                             check_and_consume=lambda *a, **k: (True, None))
    client = DeepSeekClient(DeepSeekSettings(api_key="test-only"), budget=budget)
    sends = []
    client._post = lambda *a, **k: sends.append(1)
    with store.operation_scope(claim), pytest.raises(BudgetExceededError):
        client.invoke_text(system_prompt="synthetic", user_prompt="synthetic")
    assert sends == []
    with sqlite3.connect(store.path) as connection:
        for table in ("model_calls", "runtime_attempt_reservations", "token_budget_reservations", "token_budget_buckets"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def reserve(ledger, request, tokens=6, **kwargs):
    return ledger.prepare(provider="fake", purpose="answer", logical_request_id=request,
                          attempt=1, endpoint_identity="fake", model="fake", payload_digest="fake",
                          estimated_tokens=tokens, session_id="s",
                          token_scopes={"session": 10, "daily": 10}, daily_scope="daily", **kwargs)


def test_restart_and_replay_do_not_reset_budget(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = ModelCallLedger(path)
    first = reserve(ledger, "a")
    restarted = ModelCallLedger(path)
    assert reserve(restarted, "a").call_id == first.call_id
    with pytest.raises(BudgetExceededError):
        reserve(restarted, "b")
    assert restarted.list_for("b") == []


def test_concurrent_reservations_share_ceiling(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ModelCallLedger(path)
    def attempt(n):
        try:
            reserve(ModelCallLedger(path), str(n))
            return True
        except BudgetExceededError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(16))) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1
        assert connection.execute("SELECT MAX(used) FROM token_budget_buckets").fetchone()[0] == 6


def test_overrun_is_charged_and_prevents_next_call(tmp_path):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    record = reserve(ledger, "a", 3)
    ledger.transition(record.call_id, expected=ModelCallState.PREPARED, target=ModelCallState.DISPATCH_INTENT)
    ledger.transition(record.call_id, expected=ModelCallState.DISPATCH_INTENT,
                      target=ModelCallState.SUCCEEDED, actual_tokens=12)
    with pytest.raises(BudgetExceededError):
        reserve(ledger, "b", 1)
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT MIN(used) FROM token_budget_buckets").fetchone()[0] == 12


def test_daily_rollover_keeps_session_reservation(tmp_path, monkeypatch):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    monkeypatch.setattr("agent.model_call_ledger.time.time", lambda: 86400)
    reserve(ledger, "a")
    monkeypatch.setattr("agent.model_call_ledger.time.time", lambda: 172800)
    with pytest.raises(BudgetExceededError):
        reserve(ledger, "b")


def test_failed_daily_check_rolls_back_session_reservation(tmp_path):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    reserve(ledger, "a")
    with pytest.raises(BudgetExceededError):
        ledger.prepare(provider="fake", purpose="answer", logical_request_id="b", attempt=1,
                       endpoint_identity="fake", model="fake", payload_digest="fake", estimated_tokens=6,
                       session_id="s2", token_scopes={"other-session": 10, "daily": 10}, daily_scope="daily")
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT 1 FROM token_budget_buckets WHERE scope='other-session'").fetchone() is None
