import sqlite3
import hashlib

import pytest

from agent.model_call_ledger import ModelCallLedger, ModelCallConflict
from agent.token_budget_migration import import_frozen_baseline, validate_legacy_source


def validate(raw, **kwargs):
    return validate_legacy_source(raw, expected_digest=hashlib.sha256(raw).hexdigest(),
                                  source_timezone="Asia/Shanghai", **kwargs)


def test_source_validation_preserves_exact_day_and_usage():
    raw = b'{"date":"2026-09-05","users":{"u":{"used":12,"calls":2}}}'
    assert validate(raw) == {"date": "2026-09-05", "users": {"u": {"used": 12, "calls": 2}}}


@pytest.mark.parametrize("raw", [
    b'{"date":"2026-02-30","users":{}}',
    b'{"date":"2026-09-05","users":{},"users":{}}',
    b'{"date":"2026-09-05","users":{"u":{"used":true,"calls":2}}}',
    b'{"date":"2026-09-05","users":{"u":{"used":-1,"calls":2}}}',
    b'{"date":"2026-09-05","users":{},"secret":"synthetic"}',
    b'[]', b'\xff',
])
def test_source_rejects_ambiguous_or_invalid_bytes(raw):
    with pytest.raises(ValueError, match="^invalid legacy usage source$"):
        validate(raw)


def test_source_digest_and_timezone_must_match():
    raw = b'{"date":"2026-09-05","users":{}}'
    with pytest.raises(ValueError, match="changed after freeze"):
        validate_legacy_source(raw, expected_digest="0" * 64, source_timezone="Asia/Shanghai")
    with pytest.raises(ValueError, match="timezone"):
        validate_legacy_source(raw, expected_digest=hashlib.sha256(raw).hexdigest(), source_timezone="UTC")


def baseline():
    return dict(source_id="a" * 64, source_digest="b" * 64,
                buckets={"c" * 64: {"used": 12, "ceiling": 10}})


def test_import_preserves_overrun_and_restart_is_idempotent(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    assert import_frozen_baseline(ModelCallLedger(path), **baseline()) == "imported"
    assert import_frozen_baseline(ModelCallLedger(path), **baseline()) == "already_imported"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT used,ceiling FROM token_budget_buckets").fetchall() == [(12, 10)]


@pytest.mark.parametrize("change", ["digest", "mapping"])
def test_reimport_changed_source_or_mapping_rejected(tmp_path, change):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    args = baseline()
    import_frozen_baseline(ledger, **args)
    if change == "digest":
        args["source_digest"] = "d" * 64
    else:
        args["buckets"]["c" * 64]["used"] = 1
    with pytest.raises(ModelCallConflict):
        import_frozen_baseline(ledger, **args)


def test_nonempty_target_is_not_merged(tmp_path):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("INSERT INTO token_budget_buckets VALUES('existing',3,10)")
    with pytest.raises(ModelCallConflict):
        import_frozen_baseline(ledger, **baseline())
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT scope,used FROM token_budget_buckets").fetchall() == [("existing", 3)]


def test_marker_failure_rolls_back_entire_import(tmp_path):
    ledger = ModelCallLedger(tmp_path / "ledger.sqlite3")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("CREATE TABLE token_budget_migrations(source_id TEXT PRIMARY KEY,binding TEXT NOT NULL,schema_version INTEGER NOT NULL)")
        connection.execute("CREATE TRIGGER fail_marker BEFORE INSERT ON token_budget_migrations BEGIN SELECT RAISE(ABORT,'synthetic'); END")
    with pytest.raises(sqlite3.IntegrityError):
        import_frozen_baseline(ledger, **baseline())
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM token_budget_buckets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM token_budget_migrations").fetchone()[0] == 0
