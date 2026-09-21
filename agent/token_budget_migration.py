"""Offline primitive for importing an already verified, frozen usage baseline.

No source file IO or automatic startup migration: the caller must establish the
stop-write boundary and protected recovery copy before production use.
"""
from __future__ import annotations

import hashlib
import json
import re
import hmac
from datetime import date

from agent.model_call_ledger import ModelCallConflict


def validate_legacy_source(raw: bytes, *, expected_digest: str, source_timezone: str) -> dict:
    """Validate frozen bytes only; never infer timezone or missing session usage.

    Returned user counters are private in-memory data, not a logging/report DTO.
    This validates content, not proof that the producer has stopped writing.
    """
    if not isinstance(raw, bytes) or len(raw) > 4 * 1024 * 1024:
        raise ValueError("invalid legacy source size")
    if source_timezone != "Asia/Shanghai":
        raise ValueError("legacy timezone requires explicit reconciliation")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("invalid expected source digest")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_digest):
        raise ValueError("legacy source changed after freeze")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate legacy key")
            obj[key] = value
        return obj

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        if not isinstance(value, dict) or set(value) != {"date", "users"}:
            raise ValueError()
        day = value["date"]
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError()
        if not isinstance(value["users"], dict):
            raise ValueError()
        for user, counters in value["users"].items():
            if not isinstance(user, str) or not user or "\x1f" in user:
                raise ValueError()
            if not isinstance(counters, dict) or set(counters) != {"used", "calls"}:
                raise ValueError()
            if any(type(counters[field]) is not int or not 0 <= counters[field] <= 2**63 - 1
                   for field in ("used", "calls")):
                raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        # Do not include source content, keys or decoder exception repr.
        raise ValueError("invalid legacy usage source") from None


def import_frozen_baseline(ledger, *, source_id: str, source_digest: str,
                           buckets: dict[str, dict[str, int]]) -> str:
    """Atomically seed an empty ledger; a matching replay is a no-op.

    source_id is an opaque source identity, not a path or user name. Digests bind
    exact frozen source bytes and normalized mapped scopes. Populated targets
    are rejected because old and new accounting may overlap.
    """
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (source_id, source_digest)):
        raise ValueError("invalid migration source identity")
    if not isinstance(buckets, dict) or not buckets:
        raise ValueError("missing verified token baseline")
    normalized = {}
    for scope, value in buckets.items():
        if not isinstance(scope, str) or not re.fullmatch(r"[0-9a-f]{64}(?::\d{4}-\d{2}-\d{2})?", scope):
            raise ValueError("invalid mapped token scope")
        if not isinstance(value, dict) or set(value) != {"used", "ceiling"} or any(
            type(value[key]) is not int or value[key] < 0 for key in ("used", "ceiling")
        ):
            raise ValueError("invalid token baseline counters")
        normalized[scope] = dict(value)
    payload = json.dumps({"schema": 1, "source_digest": source_digest, "buckets": normalized},
                         sort_keys=True, separators=(",", ":"))
    binding = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with ledger._lock, ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS token_budget_migrations ("
            "source_id TEXT PRIMARY KEY, binding TEXT NOT NULL, schema_version INTEGER NOT NULL)"
        )
        existing = connection.execute(
            "SELECT binding,schema_version FROM token_budget_migrations WHERE source_id=?", (source_id,)
        ).fetchone()
        if existing:
            if existing != (binding, 1):
                raise ModelCallConflict("migration source or mapping changed")
            connection.commit()
            return "already_imported"
        for table in ("model_calls", "token_budget_buckets", "token_budget_reservations", "token_budget_migrations"):
            if connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise ModelCallConflict("migration requires an empty accounting target")
        connection.executemany("INSERT INTO token_budget_buckets VALUES(?,?,?)",
                               [(scope, value["used"], value["ceiling"]) for scope, value in normalized.items()])
        connection.execute("INSERT INTO token_budget_migrations VALUES(?,?,1)", (source_id, binding))
        connection.commit()
    return "imported"
