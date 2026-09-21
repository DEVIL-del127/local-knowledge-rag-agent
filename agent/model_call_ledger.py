from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

from agent.privacy import PersistenceCipher
from agent.agent_limits import BudgetExceededError


class ModelCallState(str, Enum):
    PREPARED = "prepared"
    DISPATCH_INTENT = "dispatch_intent"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE_NOT_SENT = "failed_retryable_not_sent"
    FAILED_FINAL = "failed_final"
    OUTCOME_UNKNOWN = "outcome_unknown"


TERMINAL_STATES = {
    ModelCallState.SUCCEEDED,
    ModelCallState.FAILED_RETRYABLE_NOT_SENT,
    ModelCallState.FAILED_FINAL,
    ModelCallState.OUTCOME_UNKNOWN,
}


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    call_id: str
    provider: str
    purpose: str
    logical_request_id: str
    attempt: int
    state: ModelCallState
    endpoint_identity: str
    model: str
    payload_digest: str
    estimated_tokens: int
    session_id: str
    created_at: float
    updated_at: float
    response_digest: str = ""
    actual_tokens: int = 0
    error_class: str = ""
    response_envelope: str = ""
    revision: int = 1


class ModelCallConflict(RuntimeError):
    pass


class ModelOutcomeUnknown(RuntimeError):
    code = "model_outcome_unknown"


class ModelCallLedger:
    """Crash-safe accounting boundary for every external model HTTP attempt."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = PersistenceCipher(self.path.with_suffix(self.path.suffix + ".key"))
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_calls ("
                "call_id TEXT PRIMARY KEY, provider TEXT NOT NULL, purpose TEXT NOT NULL, "
                "logical_request_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
                "state TEXT NOT NULL, payload TEXT NOT NULL, "
                "UNIQUE(provider,purpose,logical_request_id,attempt))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS token_budget_buckets ("
                "scope TEXT PRIMARY KEY, used INTEGER NOT NULL, ceiling INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS token_budget_reservations ("
                "call_id TEXT NOT NULL, scope TEXT NOT NULL, charged INTEGER NOT NULL, "
                "PRIMARY KEY(call_id,scope))"
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _call_id(provider: str, purpose: str, logical_request_id: str, attempt: int) -> str:
        return hashlib.sha256(
            f"{provider}\0{purpose}\0{logical_request_id}\0{attempt}".encode("utf-8")
        ).hexdigest()

    def prepare(
        self, *, provider: str, purpose: str, logical_request_id: str, attempt: int,
        endpoint_identity: str, model: str, payload_digest: str,
        estimated_tokens: int, session_id: str,
        operation_id: str | None = None, operation_owner: str | None = None,
        token_scopes: dict[str, int] | None = None,
        daily_scope: str | None = None,
    ) -> ModelCallRecord:
        if type(estimated_tokens) is not int or estimated_tokens < 0:
            raise ValueError("invalid token reservation")
        if token_scopes is not None and (not token_scopes or any(
            not scope or type(limit) is not int or limit < 0 for scope, limit in token_scopes.items()
        )):
            raise ValueError("invalid token budget scopes")
        now = time.time()
        record = ModelCallRecord(
            call_id=self._call_id(provider, purpose, logical_request_id, attempt),
            provider=provider, purpose=purpose, logical_request_id=logical_request_id,
            attempt=int(attempt), state=ModelCallState.PREPARED,
            endpoint_identity=endpoint_identity, model=model,
            payload_digest=payload_digest, estimated_tokens=int(estimated_tokens),
            session_id=session_id, created_at=now, updated_at=now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()  # Lease checks must use time after lock acquisition.
            effective_scopes = dict(token_scopes or {})
            if daily_scope is not None:
                if daily_scope not in effective_scopes:
                    raise ValueError("daily scope missing from token policy")
                day = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date().isoformat()
                effective_scopes[f"{daily_scope}:{day}"] = effective_scopes.pop(daily_scope)
            if operation_id is not None:
                valid = connection.execute(
                    "SELECT 1 FROM runtime_operations WHERE operation_id=? AND owner=? "
                    "AND status='running' AND lease_until>?",
                    (operation_id, operation_owner, now),
                ).fetchone()
                if not valid:
                    raise ModelCallConflict("runtime operation no longer owns the attempt")
            row = connection.execute(
                "SELECT payload FROM model_calls WHERE call_id=?", (record.call_id,)
            ).fetchone()
            if row is not None:
                existing = self._decode(row[0])
                if self._bound_operation(connection, record.call_id) != operation_id:
                    raise ModelCallConflict("model attempt belongs to another operation")
                connection.commit()
                identity = (
                    existing.endpoint_identity == endpoint_identity
                    and existing.model == model
                    and existing.payload_digest == payload_digest
                    and existing.session_id == session_id
                )
                if not identity:
                    raise ModelCallConflict(
                        "logical model attempt identity does not match persisted record"
                    )
                return existing
            if operation_id is not None:
                limits = connection.execute(
                    "SELECT model_calls,repair_calls FROM runtime_operation_limits WHERE operation_id=?", (operation_id,),
                ).fetchone()
                used, repairs = connection.execute(
                    "SELECT COUNT(*),COALESCE(SUM(is_repair),0) FROM runtime_attempt_reservations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                is_repair = int(purpose in {"repair", "candidate", "nlu_repair"})
                if limits is None or used >= limits[0] or (is_repair and repairs >= limits[1]):
                    raise BudgetExceededError("runtime_model_attempt_budget_exhausted")
                connection.execute("INSERT INTO runtime_attempt_reservations VALUES(?,?,?)", (record.call_id, operation_id, is_repair))
            for scope, ceiling in effective_scopes.items():
                row = connection.execute(
                    "SELECT used,ceiling FROM token_budget_buckets WHERE scope=?", (scope,)
                ).fetchone()
                used = row[0] if row else 0
                ceiling = min(ceiling, row[1]) if row else ceiling
                if used + estimated_tokens > ceiling:
                    raise BudgetExceededError("runtime_token_budget_exhausted")
                connection.execute(
                    "INSERT INTO token_budget_buckets VALUES(?,?,?) ON CONFLICT(scope) DO UPDATE "
                    "SET used=excluded.used,ceiling=excluded.ceiling", (scope, used + estimated_tokens, ceiling)
                )
                connection.execute("INSERT INTO token_budget_reservations VALUES(?,?,?)",
                                   (record.call_id, scope, estimated_tokens))
            connection.execute(
                "INSERT INTO model_calls(call_id,provider,purpose,logical_request_id,attempt,state,payload) "
                "VALUES(?,?,?,?,?,?,?)",
                (record.call_id, provider, purpose, logical_request_id, int(attempt),
                 record.state.value, self._encode(record)),
            )
            connection.commit()
        return record

    @staticmethod
    def _bound_operation(connection, call_id: str) -> str | None:
        # Legacy standalone ledgers have no runtime tables.
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_attempt_reservations'"
        ).fetchone() is None:
            return None
        row = connection.execute(
            "SELECT operation_id FROM runtime_attempt_reservations WHERE call_id=?", (call_id,)
        ).fetchone()
        return row[0] if row else None

    def transition(
        self, call_id: str, *, expected: ModelCallState, target: ModelCallState,
        response_digest: str = "", actual_tokens: int = 0, error_class: str = "",
        response_payload: dict[str, Any] | None = None,
        operation_id: str | None = None, operation_owner: str | None = None,
        recovery_only: bool = False,
    ) -> ModelCallRecord:
        allowed = {
            ModelCallState.PREPARED: {ModelCallState.DISPATCH_INTENT,
                                      ModelCallState.FAILED_RETRYABLE_NOT_SENT,
                                      ModelCallState.FAILED_FINAL},
            ModelCallState.DISPATCH_INTENT: TERMINAL_STATES,
        }
        if target not in allowed.get(expected, set()):
            raise ModelCallConflict(f"invalid model call transition: {expected.value}->{target.value}")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            bound_operation = self._bound_operation(connection, call_id)
            if target == ModelCallState.DISPATCH_INTENT and bound_operation != operation_id:
                raise ModelCallConflict("dispatch identity does not own model attempt")
            if recovery_only:
                if expected != ModelCallState.DISPATCH_INTENT or target != ModelCallState.OUTCOME_UNKNOWN:
                    raise ModelCallConflict("invalid recovery transition")
                if bound_operation is not None and connection.execute(
                    "SELECT 1 FROM runtime_operations WHERE operation_id=? "
                    "AND status='running' AND lease_until>?", (bound_operation, time.time()),
                ).fetchone():
                    raise ModelCallConflict("cannot recover a live operation")
            if target == ModelCallState.DISPATCH_INTENT and operation_id is not None:
                valid = connection.execute(
                    "SELECT 1 FROM runtime_operations WHERE operation_id=? AND owner=? "
                    "AND status='running' AND lease_until>?", (operation_id, operation_owner, time.time()),
                ).fetchone()
                if not valid:
                    raise ModelCallConflict("operation lease expired before dispatch")
            row = connection.execute(
                "SELECT payload FROM model_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ModelCallConflict("model call record missing")
            current = self._decode(row[0])
            if current.state != expected:
                connection.rollback()
                raise ModelCallConflict(
                    f"model call state changed: expected={expected.value}, actual={current.state.value}"
                )
            if type(actual_tokens) is not int or actual_tokens < 0:
                raise ValueError("invalid actual token usage")
            # Unknown/absent usage never releases a reservation. Charge overruns
            # in full so subsequent admission is blocked rather than hiding cost.
            for scope, charged in connection.execute(
                "SELECT scope,charged FROM token_budget_reservations WHERE call_id=?", (call_id,)
            ).fetchall():
                if actual_tokens > charged:
                    connection.execute("UPDATE token_budget_buckets SET used=used+? WHERE scope=?",
                                       (actual_tokens - charged, scope))
                    connection.execute("UPDATE token_budget_reservations SET charged=? WHERE call_id=? AND scope=?",
                                       (actual_tokens, call_id, scope))
            updated = ModelCallRecord(
                **{**asdict(current), "state": target, "updated_at": time.time(),
                   "response_digest": response_digest, "actual_tokens": int(actual_tokens),
                   "error_class": error_class,
                   "response_envelope": (
                       self._cipher.encrypt_json(
                           response_payload, aad=f"model-call-response-v1:{call_id}"
                       ) if response_payload is not None else current.response_envelope
                   ),
                   "revision": current.revision + 1}
            )
            changed = connection.execute(
                "UPDATE model_calls SET state=?,payload=? WHERE call_id=? AND state=?",
                (target.value, self._encode(updated), call_id, expected.value),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise ModelCallConflict("model call CAS failed")
            connection.commit()
            return updated

    def recover_uncertain(self) -> int:
        """Conservatively close dispatch intents left behind by a crashed worker."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT call_id FROM model_calls WHERE state=?",
                (ModelCallState.DISPATCH_INTENT.value,),
            ).fetchall()
        count = 0
        for (call_id,) in rows:
            try:
                self.transition(
                    call_id, expected=ModelCallState.DISPATCH_INTENT,
                    target=ModelCallState.OUTCOME_UNKNOWN,
                    error_class="recovered_dispatch_intent",
                    recovery_only=True,
                )
                count += 1
            except ModelCallConflict:
                pass
        return count

    def get(self, call_id: str) -> ModelCallRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM model_calls WHERE call_id=?", (call_id,)
            ).fetchone()
        return self._decode(row[0]) if row else None

    def list_for(self, logical_request_id: str) -> list[ModelCallRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM model_calls WHERE logical_request_id=? ORDER BY attempt",
                (logical_request_id,),
            ).fetchall()
        return [self._decode(row[0]) for row in rows]

    def load_response(self, record: ModelCallRecord) -> dict[str, Any]:
        if record.state != ModelCallState.SUCCEEDED or not record.response_envelope:
            raise ModelCallConflict("persisted model response is unavailable")
        value = self._cipher.decrypt_json(
            record.response_envelope, aad=f"model-call-response-v1:{record.call_id}"
        )
        if not isinstance(value, dict):
            raise ModelCallConflict("persisted model response has invalid shape")
        return value

    @staticmethod
    def _encode(record: ModelCallRecord) -> str:
        payload = asdict(record)
        payload["state"] = record.state.value
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode(payload: str) -> ModelCallRecord:
        value: dict[str, Any] = json.loads(payload)
        value["state"] = ModelCallState(value["state"])
        return ModelCallRecord(**value)
