import hashlib

import pytest

from agent.privacy import PersistenceCipher, protect_legacy_key


class Protector:
    def protect(self, payload, *, context):
        return b"protected:" + context + b":" + payload

    def unprotect(self, payload, *, context):
        prefix = b"protected:" + context + b":"
        if not payload.startswith(prefix):
            raise ValueError("context mismatch")
        return payload[len(prefix):]


def test_legacy_key_migration_is_recoverable_and_runtime_readable(tmp_path, monkeypatch):
    key = b"k" * 32
    key_path = tmp_path / "runtime.sqlite3.key"
    key_path.write_bytes(key)
    backup = tmp_path / "runtime.key.dpapi-backup"
    protector = Protector()
    monkeypatch.setattr(PersistenceCipher, "_key_protection", staticmethod(lambda: protector))
    monkeypatch.setattr(
        "agent.protected_backup.create_windows_backup",
        lambda raw, destination, *, context: (destination.write_bytes(b"backup") or "digest"),
    )
    result = protect_legacy_key(
        key_path, backup, expected_sha256=hashlib.sha256(key).hexdigest(),
    )
    assert result["migrated"] is True
    assert key_path.read_bytes().startswith(PersistenceCipher.key_prefix)
    PersistenceCipher(key_path)


def test_legacy_key_migration_rejects_changed_source(tmp_path, monkeypatch):
    key_path = tmp_path / "runtime.sqlite3.key"
    key_path.write_bytes(b"x" * 32)
    monkeypatch.setattr(PersistenceCipher, "_key_protection", staticmethod(Protector))
    with pytest.raises(ValueError, match="changed"):
        protect_legacy_key(key_path, tmp_path / "backup", expected_sha256="0" * 64)
    assert key_path.read_bytes() == b"x" * 32
    assert not (tmp_path / "backup").exists()
