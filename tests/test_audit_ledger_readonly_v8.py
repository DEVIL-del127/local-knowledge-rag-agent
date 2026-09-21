import sqlite3
import json
import pytest

from scripts.build_agent_architecture_audit import _ledger_counts
from scripts.build_agent_architecture_audit import _read_active


def test_missing_audit_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"
    assert _ledger_counts(path) == {}
    assert not path.exists()


def test_uninitialized_audit_database_is_unchanged(tmp_path):
    path = tmp_path / "empty.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    before = path.read_bytes()
    assert _ledger_counts(path) == {}
    assert path.read_bytes() == before


def test_audit_counts_without_modifying_database(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE model_calls (state TEXT)")
        connection.executemany("INSERT INTO model_calls VALUES (?)", [("prepared",), ("prepared",), ("unknown",)])
    before = path.read_bytes()
    assert _ledger_counts(path) == {"prepared": 2, "unknown": 1}
    assert path.read_bytes() == before


def test_audit_active_reader_does_not_initialize_legacy_registry(tmp_path):
    path = tmp_path / "registry.sqlite3"
    payload = dict(generation_id="g", state="active", es_physical_index="i", vector_collection="c")
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE registry_meta (singleton INTEGER, active_generation TEXT, revision INTEGER)")
        connection.execute("INSERT INTO registry_meta VALUES (1, 'g', 11)")
        connection.execute("CREATE TABLE generations (generation_id TEXT, payload TEXT)")
        connection.execute("INSERT INTO generations VALUES (?, ?)", ("g", json.dumps(payload)))
    before = path.read_bytes()
    record, revision = _read_active(path)
    assert record.generation_id == "g"
    assert revision == 11
    assert path.read_bytes() == before
    missing = tmp_path / "absent.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        _read_active(missing)
    assert not missing.exists()
