"""Explicit checkpoint-envelope migration. Never called by application startup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from agent.privacy import PersistenceCipher
from agent.protected_backup import create_windows_backup


def _snapshot_bytes(connection: sqlite3.Connection) -> bytes:
    """Bundle the recoverable database and WAL bytes without a plaintext temp file."""
    row = connection.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise ValueError("checkpoint database path unavailable")
    database = Path(row[2])
    parts = []
    for suffix in ("", "-wal"):
        path = Path(str(database) + suffix)
        payload = path.read_bytes() if path.is_file() else b""
        name = ("database" if not suffix else "wal").encode("ascii")
        parts.append(len(name).to_bytes(2, "big") + name + len(payload).to_bytes(8, "big") + payload)
    return b"study-sqlite-bundle-v1\0" + b"".join(parts)


def migrate_checkpoint_envelopes(database: Path, backup: Path, *, expected_sha256: str) -> dict:
    """Migrate plaintext rows under an exclusive transaction after a DPAPI backup.

    Caller must hold the application's maintenance window and verify backup ACLs.
    Existing runtime processes must not retain keys or pending operations.
    The expected digest binds the inspected database serialization, including WAL.
    """
    database, backup = Path(database).resolve(), Path(backup).absolute()
    if not database.is_file() or backup.exists() or database == backup:
        raise ValueError("invalid checkpoint migration targets")
    key_path = database.with_suffix(database.suffix + ".key")
    if not key_path.is_file():
        raise ValueError("checkpoint key missing; migration cannot invent a key")
    cipher = PersistenceCipher(key_path)
    connection = sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=3)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "runtime_operations" in tables and connection.execute(
            "SELECT 1 FROM runtime_operations WHERE status='running' LIMIT 1"
        ).fetchone():
            raise ValueError("running operations prevent checkpoint migration")
        original = _snapshot_bytes(connection)
        if hashlib.sha256(original).hexdigest() != expected_sha256:
            raise ValueError("checkpoint changed since migration inventory")
        rows = connection.execute("SELECT checkpoint_key,payload,revision FROM checkpoints").fetchall()
        replacements = []
        from agent.runtime.models import AgentState
        for key, payload, revision in rows:
            aad = f"runtime-checkpoint-v1:{key}:{int(revision)}"
            if isinstance(payload, str) and payload.startswith(cipher.prefix):
                cipher.decrypt_json(payload, aad=aad)
                continue
            decoded = json.loads(payload)
            state = AgentState.from_dict(decoded)
            if "\x1f".join((state.user_id, state.session_id, state.thread_id)) != key:
                raise ValueError("legacy checkpoint scope mismatch")
            envelope = cipher.encrypt_json(decoded, aad=aad)
            cipher.decrypt_json(envelope, aad=aad)
            replacements.append((envelope, key, revision))
        if not replacements:
            connection.rollback()
            return {"migrated": 0, "backup_created": False}
        backup_digest = create_windows_backup(
            original, backup, context=b"study-checkpoint-backup-v1",
        )
        connection.executemany(
            "UPDATE checkpoints SET payload=? WHERE checkpoint_key=? AND revision=?", replacements,
        )
        connection.commit()
        return {"migrated": len(replacements), "backup_created": True, "backup_sha256": backup_digest}
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def checkpoint_inventory(database: Path) -> dict:
    """Read-only inventory; no payload or secret is returned."""
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        digest = hashlib.sha256(_snapshot_bytes(connection)).hexdigest()
        count = connection.execute("SELECT COUNT(*) FROM checkpoints WHERE payload NOT LIKE 'enc-v1:%'").fetchone()[0]
        return {"database_sha256": digest, "plaintext_rows": count}
