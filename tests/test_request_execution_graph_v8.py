from types import SimpleNamespace

import pytest

from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.checkpoint_store import SQLiteCheckpointStore


@pytest.mark.parametrize("query,reason", [("1+1等于几", "bounded_expression_complete"),
                                          ("帮助", "help_complete"), ("取消", "pending_action_cancelled")])
def test_deterministic_requests_share_graph_admission_and_commit(tmp_path, query, reason):
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(compiler=object(), checkpoint_store=store)
    reply = runtime.handle(query, legacy=object(), request_id="r")
    assert reply.intent.reason == reason
    trace = reply.semantic_trace[-1]["runtime_execution_graph"]
    assert trace[0] == "admit"
    assert trace[1] == "route"
    assert trace[-1] == "commit"
    replay = runtime.handle(query, legacy=object(), request_id="r")
    assert replay.intent.reason == "runtime_operation_replayed"
    assert replay.semantic_trace == reply.semantic_trace
