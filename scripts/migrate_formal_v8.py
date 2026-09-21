"""Explicit, idempotent formal-state migration for the v8 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from agent.app_settings import AppSettings
from agent.checkpoint_migration import checkpoint_inventory, migrate_checkpoint_envelopes
from agent.model_call_ledger import ModelCallLedger
from agent.privacy import PersistenceCipher, protect_legacy_key
from agent.protected_backup import create_windows_backup
from agent.token_budget_migration import import_frozen_baseline, validate_legacy_source
from core.contract_store import canonical_json
from ingestion.pipeline.catalog_migration import attach_legacy_catalog
from ingestion.pipeline.generation_registry import GenerationRegistry


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env(ROOT)
    state = settings.state_dir
    runtime_db = state / "checkpoints/runtime.sqlite3"
    usage_path = state / "token_usage.json"
    registry_path = settings.ingestion_registry_path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = state / "migration_backups" / stamp

    inventory = checkpoint_inventory(runtime_db)
    raw_keys = [p for p in sorted(state.rglob("*.key")) if p.is_file() and len(p.read_bytes()) == 32]
    reader = GenerationRegistry(registry_path, read_only=True)
    active, revision = reader.get_active()
    if active is None:
        raise RuntimeError("ACTIVE generation missing")
    record_digest = hashlib.sha256(canonical_json(active.to_dict())).hexdigest()
    summary = {
        "apply": args.apply,
        "checkpoint_plaintext_rows": inventory["plaintext_rows"],
        "raw_key_count": len(raw_keys),
        "legacy_usage_exists": usage_path.is_file(),
        "active_generation": active.generation_id,
        "active_revision": revision,
        "active_manifest_digest": active.manifest_hash,
    }
    if not args.apply:
        print(json.dumps(summary, sort_keys=True))
        return

    backup_root.mkdir(parents=True, exist_ok=False)
    results: dict[str, object] = {}
    if inventory["plaintext_rows"]:
        runtime_key = runtime_db.with_suffix(runtime_db.suffix + ".key")
        if not runtime_key.exists():
            PersistenceCipher(runtime_key)
            results["runtime_key"] = "provisioned_os_protected"
        results["checkpoints"] = migrate_checkpoint_envelopes(
            runtime_db, backup_root / "runtime.sqlite3.dpapi-backup",
            expected_sha256=inventory["database_sha256"],
        )

    # Protect existing keys only after data conversion, preserving exact key bytes.
    key_results = []
    for index, path in enumerate(raw_keys, 1):
        key_results.append(protect_legacy_key(
            path, backup_root / f"legacy-key-{index}.dpapi-backup",
            expected_sha256=digest(path),
        ))
    results["keys"] = key_results

    if usage_path.is_file():
        raw = usage_path.read_bytes()
        source_digest = hashlib.sha256(raw).hexdigest()
        value = validate_legacy_source(raw, expected_digest=source_digest,
                                       source_timezone="Asia/Shanghai")
        create_windows_backup(raw, backup_root / "token-usage.dpapi-backup",
                              context=b"study-token-usage-backup-v1")
        cipher = PersistenceCipher(runtime_db.with_suffix(runtime_db.suffix + ".key"))
        ceiling = int(os.environ.get("AGENT_DAILY_TOKEN_BUDGET", "500000"))
        buckets = {}
        for user, counters in value["users"].items():
            payload = json.dumps(["token-daily-v1", user], ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
            scope = cipher.identity_hmac(payload) + ":" + value["date"]
            buckets[scope] = {"used": counters["used"], "ceiling": ceiling}
        if buckets:
            source_id = hashlib.sha256(b"study-legacy-token-usage-v1:" + bytes.fromhex(source_digest)).hexdigest()
            results["token_usage"] = import_frozen_baseline(
                ModelCallLedger(runtime_db), source_id=source_id,
                source_digest=source_digest, buckets=buckets,
            )
        else:
            results["token_usage"] = "empty_source"

    es = Elasticsearch(f"http://{settings.es_host}:{settings.es_port}", request_timeout=30)
    try:
        results["catalog"] = attach_legacy_catalog(
            registry_path, es_client=es,
            backup_path=backup_root / "generation-registry.dpapi-backup",
            expected_generation=active.generation_id, expected_revision=revision,
            expected_record_digest=record_digest,
        )
    finally:
        es.close()
    results["backup_directory"] = str(backup_root)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
