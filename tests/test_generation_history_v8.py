import hashlib
import json
import sqlite3

import pytest

from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationRecord, GenerationState, GenerationConflict
from dataclasses import asdict, replace
from core.retrieval_gateway import GenerationSnapshot, GenerationMismatch


def test_snapshot_canonical_roundtrip_includes_default_fields():
    snapshot = GenerationSnapshot("g", 1, "es", "vector", "model", 3)
    snapshot = replace(snapshot, snapshot_digest=snapshot.canonical_digest())
    assert GenerationSnapshot.restore(asdict(snapshot)) == snapshot
    changed = asdict(snapshot)
    changed["embedding_normalization"] = "different"
    with pytest.raises(GenerationMismatch):
        GenerationSnapshot.restore(changed)


def test_v2_snapshot_digest_is_not_silently_redefined_by_v3_defaults():
    old = GenerationSnapshot("g", 1, "es", "vector", "model", 3,
                             snapshot_schema_version="generation-snapshot-v2")
    payload = asdict(old)
    payload.pop("catalog_ref")
    payload.pop("snapshot_digest")
    payload["snapshot_digest"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest()
    restored = GenerationSnapshot.restore(payload)
    assert GenerationSnapshot.restore(asdict(restored)) == restored
    with pytest.raises(ValueError, match="snapshot v3"):
        replace(restored, catalog_ref={"unverified": True}).canonical_digest()


def test_retired_record_keeps_original_revision_across_restart(tmp_path):
    path = tmp_path / "registry.sqlite3"
    registry = GenerationRegistry(path)
    for name in ("g1", "g2"):
        registry.put(GenerationRecord(name, GenerationState.PREPARED, "es-" + name, "vector-" + name))
    registry.activate("g1", expected_active=None, expected_revision=0)
    original, _ = registry.get_active()
    digest = hashlib.sha256(json.dumps(original.to_dict(), ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    registry.activate("g2", expected_active="g1", expected_revision=1)
    restarted = GenerationRegistry(path)
    assert restarted.get("g1").state == GenerationState.RETIRED
    historic = restarted.get_record_revision("g1", original.record_revision, expected_digest=digest)
    assert historic.to_dict() == original.to_dict()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE generation_record_revisions SET payload='{}' WHERE generation_id='g1' AND revision=?",
                           (original.record_revision,))
    with pytest.raises(GenerationConflict, match="digest mismatch"):
        restarted.get_record_revision("g1", original.record_revision, expected_digest=digest)
