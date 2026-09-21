import sqlite3

from agent.memory_commands import parse_memory_command
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from memory.memory_manager import MemoryManager


def test_passive_turn_does_not_persist_and_session_is_sqlite(tmp_path):
    memory = MemoryManager(state_dir=str(tmp_path))
    session = memory.on_session_start(user_id="u")
    memory.ingest_message(round_no=1, user_msg="我喜欢红色", agent_reply="好的",
                          intent=None, session_id=session, user_id="u")
    assert memory._load_session(session)["facts"] == []
    assert not (tmp_path / "sessions" / f"{session}.json").exists()
    with sqlite3.connect(tmp_path / "memory" / "memory.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM memory_session").fetchone()[0] == 1


def test_explicit_save_is_idempotent_inside_request_commit(tmp_path):
    memory = MemoryManager(state_dir=str(tmp_path / "memory"))
    store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")
    claim = store.begin_operation(user_id="u", session_id="s", thread_id="s",
                                  request_id="r", payload={}, expected_revision=0)
    command, arguments = parse_memory_command("请记住：我喜欢红色")
    with store.operation_scope(claim):
        saved = memory.admin(command=command, user_id="u", args=arguments)
    assert memory.admin(command="stats", user_id="u")["total"] == 0
    store.complete_operation(claim, {"answer": "ok"})
    store.project_memory_events(memory)
    store.project_memory_events(memory)
    listed = memory.admin(command="list", user_id="u")["items"]
    assert [(item["id"], item["value"]) for item in listed] == [(saved["id"], "我喜欢红色")]


def test_ambiguous_destructive_memory_command_is_rejected():
    try:
        parse_memory_command("删除这条记忆")
    except ValueError as exc:
        assert "完整ID" in str(exc)
    else:
        raise AssertionError("ambiguous delete was accepted")
