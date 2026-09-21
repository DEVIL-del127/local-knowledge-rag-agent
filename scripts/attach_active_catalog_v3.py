from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.contract_store import canonical_json
from ingestion.pipeline.catalog_migration import attach_legacy_catalog
from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "data/ingestion/generation_registry.sqlite3"
    registry = GenerationRegistry(registry_path, read_only=True)
    active, revision = registry.get_active()
    if active is None:
        raise RuntimeError("ACTIVE generation missing")
    record_digest = hashlib.sha256(canonical_json(active.to_dict())).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "data/ingestion/catalog_backups" / f"{stamp}.dpapi-backup"
    backup.parent.mkdir(parents=True, exist_ok=True)
    system = PDFSearchSystem(db="main")
    result = attach_legacy_catalog(
        registry_path, system.es_manager.es, backup_path=backup,
        expected_generation=active.generation_id, expected_revision=revision,
        expected_record_digest=record_digest,
    )
    print(json.dumps({"generation_id": active.generation_id, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
