from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import threading
import contextvars
from contextlib import contextmanager
from pathlib import Path

from .models import AgentState
from agent.privacy import PersistenceCipher
from .operations import OperationStoreMixin


class CheckpointConflict(RuntimeError):
    pass


class SQLiteCheckpointStore(OperationStoreMixin):
    """Small local checkpoint store keyed by user/session/thread."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._current_operation = contextvars.ContextVar("runtime_operation", default=None)
        self._cipher = PersistenceCipher(self.path.with_suffix(self.path.suffix + ".key"))
        self._initialize()

    @staticmethod
    def key(user_id: str, session_id: str, thread_id: str) -> str:
        return "\x1f".join((user_id, session_id, thread_id))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            # WAL persists on disk; do not renegotiate it on every read/claim.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "checkpoint_key TEXT PRIMARY KEY, payload TEXT NOT NULL, revision INTEGER NOT NULL)"
            )
            self._initialize_operations(connection)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runtime_snapshot_objects("
                "digest TEXT PRIMARY KEY,payload TEXT NOT NULL,byte_length INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runtime_snapshot_pins("
                "checkpoint_key TEXT PRIMARY KEY,digest TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS turn_result_artifacts("
                "digest TEXT PRIMARY KEY,payload TEXT NOT NULL,schema_version TEXT NOT NULL)"
            )

    def load(self, user_id: str, session_id: str, thread_id: str) -> AgentState | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload, revision FROM checkpoints WHERE checkpoint_key = ?",
                (self.key(user_id, session_id, thread_id),),
            ).fetchone()
        if not row:
            return None
        checkpoint_key = self.key(user_id, session_id, thread_id)
        state = AgentState.from_dict(self._cipher.decrypt_json(
            row[0], aad=f"runtime-checkpoint-v1:{checkpoint_key}:{int(row[1])}",
        ))
        state.checkpoint_revision = int(row[1])
        self._migrate_legacy_literature_context(state)
        return state

    def _migrate_legacy_literature_context(self, state: AgentState) -> None:
        """One-way in-memory migration; old semantic state is never executed."""
        context = state.context_snapshot
        legacy_results = context.pop("result_set", None)
        context.pop("literature", None)
        if not legacy_results or context.get("last_result_artifact_ref"):
            return
        if not all(isinstance(item, dict) and item.get("document_id") for item in legacy_results):
            return
        from .turn_artifacts import DocumentResult, TurnResultArtifact
        snapshot = context.get("generation_snapshot") or {}
        generation_id = str(snapshot.get("generation_id") or state.ingestion_generation or "")
        if not generation_id:
            return
        documents = tuple(DocumentResult(
            document_id=str(item["document_id"]), filename=str(item.get("filename") or ""),
            title=str(item.get("title") or ""), authors=tuple(item.get("authors") or ()),
            publication_year=item.get("publication_year"), rank=index,
        ) for index, item in enumerate(legacy_results, 1))
        artifact = TurnResultArtifact(
            turn_id=state.request_id, task="legacy_migration", generation_id=generation_id,
            snapshot_digest=str(snapshot.get("snapshot_digest") or ""), documents=documents,
            topic=None, filters={"migration": "legacy-result-set-v1"},
        )
        digest = self.put_turn_result_artifact(artifact)
        context["last_result_artifact_ref"] = {"digest": digest, "schema_version": artifact.schema_version}

    def save(self, state: AgentState, *, expected_revision: int | None = None) -> int:
        staged = self.stage_operation_state(state, expected_revision=expected_revision)
        if staged is not None:
            return staged
        expected = state.checkpoint_revision if expected_revision is None else int(expected_revision)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                revision = self._save_in_connection(connection, state, expected)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        state.checkpoint_revision = revision
        return revision

    def _save_in_connection(self, connection, state: AgentState, expected: int) -> int:
        checkpoint_key = self.key(state.user_id, state.session_id, state.thread_id)
        row = connection.execute("SELECT revision FROM checkpoints WHERE checkpoint_key=?", (checkpoint_key,)).fetchone()
        actual = int(row[0]) if row else 0
        if actual != expected:
            raise CheckpointConflict(f"checkpoint revision mismatch: expected={expected}, actual={actual}")
        revision = actual + 1
        snapshot = state.context_snapshot.get("generation_snapshot")
        if snapshot:
            from core.retrieval_gateway import GenerationSnapshot
            try:
                GenerationSnapshot.restore(snapshot)
            except Exception:
                # Preserve legacy/corrupt input for the runtime's fail-closed
                # recovery path; never mint a trusted ref for invalid bytes.
                state.context_snapshot.pop("generation_snapshot_ref", None)
                connection.execute("DELETE FROM runtime_snapshot_pins WHERE checkpoint_key=?", (checkpoint_key,))
            else:
                raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                digest = hashlib.sha256(raw).hexdigest()
                connection.execute(
                    "INSERT OR IGNORE INTO runtime_snapshot_objects VALUES(?,?,?)",
                    (digest, self._cipher.encrypt_json(snapshot, aad=f"runtime-snapshot-v1:{digest}"), len(raw)),
                )
                connection.execute(
                    "INSERT INTO runtime_snapshot_pins VALUES(?,?) ON CONFLICT(checkpoint_key) DO UPDATE SET digest=excluded.digest",
                    (checkpoint_key, digest),
                )
                state.context_snapshot["generation_snapshot_ref"] = {
                    "store_id": "runtime-checkpoints-v1", "locator_scheme": "sqlite-sha256-v1",
                    "locator": digest, "digest_algorithm": "sha256", "digest": digest,
                    "byte_length": len(raw), "schema_version": snapshot["snapshot_schema_version"],
                }
        else:
            state.context_snapshot.pop("generation_snapshot_ref", None)
            connection.execute("DELETE FROM runtime_snapshot_pins WHERE checkpoint_key=?", (checkpoint_key,))
        payload = self._cipher.encrypt_json(state.to_dict(), aad=f"runtime-checkpoint-v1:{checkpoint_key}:{revision}")
        connection.execute(
            "INSERT INTO checkpoints VALUES(?,?,?) ON CONFLICT(checkpoint_key) DO UPDATE "
            "SET payload=excluded.payload,revision=excluded.revision", (checkpoint_key, payload, revision),
        )
        return revision

    def resolve_generation_snapshot(self, state: AgentState):
        ref = state.context_snapshot.get("generation_snapshot_ref")
        if ref is None:
            # Retained pre-reference checkpoints remain self-verifying; explicit
            # migration may attach refs, but reads never rewrite old state.
            return state.context_snapshot.get("generation_snapshot")
        expected_keys = {"store_id", "locator_scheme", "locator", "digest_algorithm", "digest", "byte_length", "schema_version"}
        if (not isinstance(ref, dict) or set(ref) != expected_keys
                or ref["store_id"] != "runtime-checkpoints-v1" or ref["locator_scheme"] != "sqlite-sha256-v1"
                or ref["digest_algorithm"] != "sha256" or ref["locator"] != ref["digest"]):
            raise CheckpointConflict("invalid generation snapshot reference")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT o.payload,o.byte_length FROM runtime_snapshot_objects o JOIN runtime_snapshot_pins p "
                "ON p.digest=o.digest WHERE p.checkpoint_key=? AND o.digest=?",
                (self.key(state.user_id, state.session_id, state.thread_id), ref["digest"]),
            ).fetchone()
        if row is None:
            raise CheckpointConflict("pinned generation snapshot object unavailable")
        value = self._cipher.decrypt_json(row[0], aad=f"runtime-snapshot-v1:{ref['digest']}")
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if (len(raw) != row[1] or len(raw) != ref["byte_length"] or hashlib.sha256(raw).hexdigest() != ref["digest"]
                or value.get("snapshot_schema_version") != ref["schema_version"]
                or value != state.context_snapshot.get("generation_snapshot")):
            raise CheckpointConflict("generation snapshot object identity mismatch")
        return value

    def delete(self, user_id: str, session_id: str, thread_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM checkpoints WHERE checkpoint_key = ?",
                (self.key(user_id, session_id, thread_id),),
            )
            connection.execute("DELETE FROM runtime_snapshot_pins WHERE checkpoint_key=?",
                               (self.key(user_id, session_id, thread_id),))
            connection.commit()

    def put_turn_result_artifact(self, artifact) -> str:
        payload = artifact.to_dict()
        digest = artifact.artifact_digest
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO turn_result_artifacts VALUES(?,?,?)",
                (digest, self._cipher.encrypt_json(payload, aad=f"turn-artifact-v1:{digest}"), artifact.schema_version),
            )
        return digest

    def get_turn_result_artifact(self, digest: str):
        from .turn_artifacts import TurnResultArtifact
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM turn_result_artifacts WHERE digest=?", (str(digest),),
            ).fetchone()
        if row is None:
            return None
        return TurnResultArtifact.restore(self._cipher.decrypt_json(row[0], aad=f"turn-artifact-v1:{digest}"))
