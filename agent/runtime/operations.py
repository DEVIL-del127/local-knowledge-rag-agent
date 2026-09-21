from __future__ import annotations

import copy
import hashlib
import hmac
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from agent.model_execution_context import operation_context
from agent.agent_limits import BudgetExceededError


class OperationConflict(RuntimeError):
    code = "runtime_operation_conflict"


@dataclass
class OperationClaim:
    operation_id: str
    scope: str
    owner: str
    checkpoint_revision: int
    status: str
    result: dict[str, Any] | None = None
    pending_state: Any = None
    memory_events: list[dict[str, Any]] = field(default_factory=list)


class OperationStoreMixin:
    """Durable admission and reply/checkpoint commit, in the checkpoint database.

    An expired owner is conservatively unknown; this slice never automatically
    re-executes an operation that may have performed an external effect.
    """

    def _initialize_operations(self, connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_memory_outbox ("
            "event_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, payload TEXT NOT NULL, "
            "completed INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_operations ("
            "operation_id TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL, "
            "status TEXT NOT NULL, checkpoint_revision INTEGER NOT NULL, "
            "lease_until REAL NOT NULL, result_until REAL NOT NULL, result TEXT)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS runtime_operation_owner "
            "ON runtime_operations(scope) WHERE status='running'"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_request_aliases ("
            "request_key TEXT PRIMARY KEY, operation_id TEXT NOT NULL, "
            "payload_hmac TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_operation_limits ("
            "operation_id TEXT PRIMARY KEY, model_calls INTEGER NOT NULL, repair_calls INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_attempt_reservations ("
            "call_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, is_repair INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_tool_budgets ("
            "operation_id TEXT PRIMARY KEY, used INTEGER NOT NULL, ceiling INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_node_cursor ("
            "operation_id TEXT PRIMARY KEY, owner TEXT NOT NULL, node TEXT NOT NULL, "
            "status TEXT NOT NULL, updated_ts REAL NOT NULL)"
        )

    @contextmanager
    def node_scope(self, claim: OperationClaim, node: str):
        """Fence every node before execution and record only non-sensitive metadata.

        A cursor is diagnostic, not permission to replay an interrupted effect.
        Recovery still treats an expired operation as outcome-unknown.
        """
        allowed = {"route", "output_action", "non_kb", "resume", "pin_generation",
                   "understand", "bind", "retrieve", "validate_evidence", "synthesize",
                   "verify_citations"}
        if node not in allowed:
            raise OperationConflict("unregistered execution node")
        def mark(status):
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if not connection.execute(
                    "SELECT 1 FROM runtime_operations WHERE operation_id=? AND owner=? "
                    "AND scope=? AND status='running' AND lease_until>?",
                    (claim.operation_id, claim.owner, claim.scope, time.time()),
                ).fetchone():
                    raise OperationConflict("node operation owner expired or cancelled")
                connection.execute(
                    "INSERT INTO runtime_node_cursor VALUES(?,?,?,?,?) "
                    "ON CONFLICT(operation_id) DO UPDATE SET owner=excluded.owner,node=excluded.node,"
                    "status=excluded.status,updated_ts=excluded.updated_ts",
                    (claim.operation_id, claim.owner, node, status, time.time()),
                )
                connection.commit()
        mark("running")
        with self.operation_scope(claim):
            yield
        mark("completed")

    def reserve_tool_attempt(self, claim: OperationClaim) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute(
                "SELECT 1 FROM runtime_operations WHERE operation_id=? AND owner=? "
                "AND status='running' AND lease_until>?", (claim.operation_id, claim.owner, time.time())
            ).fetchone():
                raise OperationConflict("tool operation owner expired or cancelled")
            changed = connection.execute(
                "UPDATE runtime_tool_budgets SET used=used+1 WHERE operation_id=? AND used<ceiling",
                (claim.operation_id,),
            ).rowcount
            if changed != 1:
                raise BudgetExceededError("runtime_tool_attempt_budget_exhausted")
            connection.commit()

    def _operation_hmac(self, value: Any) -> str:
        # Use the existing private-store key during the compatibility slice.
        # S5 replaces its provider; do not introduce a public/fixed HMAC secret.
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return self._cipher.identity_hmac(payload)

    def begin_operation(
        self, *, user_id: str, session_id: str, thread_id: str, request_id: str,
        payload: dict[str, Any], expected_revision: int, action_id: str | None = None,
        lease_seconds: float = 300.0, result_ttl_seconds: float = 86400.0,
        max_model_calls: int = 2, max_repair_calls: int = 1,
        max_tool_calls: int = 5,
        cancel_existing: bool = False,
    ) -> OperationClaim:
        if not request_id or len(request_id) > 256 or lease_seconds <= 0 or result_ttl_seconds <= 0:
            raise OperationConflict("invalid operation identity or lifetime")
        if any(type(limit) is not int or limit < 0 for limit in (max_model_calls, max_repair_calls, max_tool_calls)):
            raise OperationConflict("invalid model attempt limits")
        scope = self.key(user_id, session_id, thread_id)
        request_key = self._operation_hmac([scope, "request", request_id])
        fingerprint = self._operation_hmac(payload)
        op_id = self._operation_hmac([scope, "action" if action_id else "request", action_id or request_id])
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            try:
                alias = connection.execute(
                    "SELECT operation_id,payload_hmac FROM runtime_request_aliases WHERE request_key=?",
                    (request_key,),
                ).fetchone()
                if alias:
                    if not hmac.compare_digest(alias[1], fingerprint):
                        raise OperationConflict("request identity is already bound to another payload")
                    op_id = alias[0]
                existing = connection.execute(
                    "SELECT owner,status,checkpoint_revision,lease_until,result_until,result "
                    "FROM runtime_operations WHERE operation_id=?", (op_id,),
                ).fetchone()
                if existing:
                    owner, status, revision, lease_until, result_until, result = existing
                    if status == "running" and lease_until <= now:
                        status = "unknown"
                        connection.execute("UPDATE runtime_operations SET status=? WHERE operation_id=?",
                                           (status, op_id))
                    if status == "succeeded" and result_until <= now:
                        status, result = "expired", None
                        connection.execute("UPDATE runtime_operations SET status=?,result=NULL WHERE operation_id=?",
                                           (status, op_id))
                    connection.execute(
                        "INSERT OR IGNORE INTO runtime_request_aliases VALUES(?,?,?)",
                        (request_key, op_id, fingerprint),
                    )
                    decoded = self._cipher.decrypt_json(result, aad=f"runtime-operation-v1:{op_id}") if result else None
                    connection.commit()
                    return OperationClaim(op_id, scope, owner, revision, status, decoded)
                live = connection.execute(
                    "SELECT operation_id,lease_until FROM runtime_operations WHERE scope=? AND status='running'",
                    (scope,),
                ).fetchone()
                if live and cancel_existing:
                    connection.execute("UPDATE runtime_operations SET status='cancelled' WHERE operation_id=?", (live[0],))
                    live = None
                if live and live[1] > now:
                    connection.commit()
                    return OperationClaim(op_id, scope, "", expected_revision, "busy")
                if live:
                    connection.execute("UPDATE runtime_operations SET status='unknown' WHERE operation_id=?", (live[0],))
                current = connection.execute("SELECT revision FROM checkpoints WHERE checkpoint_key=?", (scope,)).fetchone()
                if (current[0] if current else 0) != expected_revision:
                    connection.commit()
                    return OperationClaim(op_id, scope, "", expected_revision, "busy")
                owner = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO runtime_operations VALUES(?,?,?,?,?,?,?,NULL)",
                    (op_id, scope, owner, "running", expected_revision, now + lease_seconds, now + result_ttl_seconds),
                )
                connection.execute("INSERT INTO runtime_request_aliases VALUES(?,?,?)", (request_key, op_id, fingerprint))
                connection.execute("INSERT INTO runtime_operation_limits VALUES(?,?,?)", (op_id, max_model_calls, max_repair_calls))
                connection.execute("INSERT INTO runtime_tool_budgets VALUES(?,0,?)", (op_id, max_tool_calls))
                connection.commit()
                return OperationClaim(op_id, scope, owner, expected_revision, "claimed")
            except Exception:
                connection.rollback()
                raise

    @contextmanager
    def operation_scope(self, claim: OperationClaim):
        if claim.status != "claimed" or self._current_operation.get() is not None:
            raise OperationConflict("operation cannot enter execution scope")
        token = self._current_operation.set(claim)
        execution_token = operation_context.set((self, claim))
        try:
            yield claim
        finally:
            operation_context.reset(execution_token)
            self._current_operation.reset(token)

    def stage_operation_state(self, state, *, expected_revision: int | None = None) -> int | None:
        claim = self._current_operation.get()
        if claim is None:
            return None
        expected = state.checkpoint_revision if expected_revision is None else expected_revision
        if self.key(state.user_id, state.session_id, state.thread_id) != claim.scope or expected != claim.checkpoint_revision:
            raise OperationConflict("checkpoint does not belong to operation")
        claim.pending_state = copy.deepcopy(state)
        return expected + 1

    def complete_operation(self, claim: OperationClaim, reply: dict[str, Any]) -> None:
        encoded = self._cipher.encrypt_json(reply, aad=f"runtime-operation-v1:{claim.operation_id}")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owned = connection.execute(
                    "SELECT 1 FROM runtime_operations WHERE operation_id=? AND owner=? AND status='running' AND lease_until>?",
                    (claim.operation_id, claim.owner, time.time()),
                ).fetchone()
                if not owned:
                    raise OperationConflict("operation owner is no longer valid")
                if claim.pending_state is not None:
                    self._save_in_connection(connection, claim.pending_state, claim.checkpoint_revision)
                for position, event in enumerate(claim.memory_events):
                    event_id = self._operation_hmac([claim.operation_id, "memory", position])
                    connection.execute(
                        "INSERT INTO runtime_memory_outbox(event_id,operation_id,payload) VALUES(?,?,?)",
                        (event_id, claim.operation_id, self._cipher.encrypt_json(
                            event, aad=f"runtime-memory-v1:{event_id}")),
                    )
                connection.execute(
                    "UPDATE runtime_operations SET status='succeeded',result=? WHERE operation_id=? AND owner=?",
                    (encoded, claim.operation_id, claim.owner),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def stage_memory_mutations(self, claim, user_id, mutations):
        if self._current_operation.get() is not claim:
            raise OperationConflict("memory mutation outside owned request")
        if not claim.scope.startswith(str(user_id) + "\x1f"):
            raise OperationConflict("memory user does not own request")
        claim.memory_events.append(copy.deepcopy({"user_id": user_id, "mutations": mutations}))

    def project_memory_events(self, memory, *, limit=100):
        """Replay committed intents only; receiver deduplicates in its own transaction."""
        for _ in range(limit):
            with self._lock, self._connect() as connection:
                # Serialize projectors across processes so an older upsert cannot
                # overtake a later deletion. Receiver performs SQLite I/O only.
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT event_id,payload FROM runtime_memory_outbox WHERE completed=0 ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row is None:
                    connection.commit()
                    return
                event_id, payload = row
                event = self._cipher.decrypt_json(payload, aad=f"runtime-memory-v1:{event_id}")
                memory.apply_request_event(event_id, event)
                connection.execute("UPDATE runtime_memory_outbox SET completed=1,payload='' WHERE event_id=?", (event_id,))
                connection.commit()
        with self._lock, self._connect() as connection:
            if connection.execute("SELECT 1 FROM runtime_memory_outbox WHERE completed=0 LIMIT 1").fetchone():
                raise OperationConflict("memory projection backlog remains")

    def abandon_operation(self, claim: OperationClaim) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runtime_operations SET status='unknown' WHERE operation_id=? AND owner=? AND status='running'",
                (claim.operation_id, claim.owner),
            )

    def operation_status(self, claim: OperationClaim) -> str:
        """Read only the caller's owned operation; never expose another scope."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM runtime_operations WHERE operation_id=? AND owner=? AND scope=?",
                (claim.operation_id, claim.owner, claim.scope),
            ).fetchone()
        return str(row[0]) if row else "unknown"
