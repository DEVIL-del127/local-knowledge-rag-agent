import hashlib
import sqlite3

import pytest

from core.contract_store import ContractStore, canonical_json
from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationRecord, GenerationState, GenerationConflict


def test_catalog_binding_survives_retirement_without_changing_original_record(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    store = ContractStore(objects, store_id="catalogs")
    ref = store.publish({"schema_version": "catalog-contract-v2", "sources": []})
    registry = GenerationRegistry(tmp_path / "registry.sqlite3")
    for name in ("g1", "g2"):
        registry.put(GenerationRecord(name, GenerationState.PREPARED, name, name, manifest_hash="a"*64))
    registry.activate("g1", expected_active=None, expected_revision=0)
    original, revision = registry.get_active()
    args = dict(generation_id="g1", record_digest=hashlib.sha256(canonical_json(original.to_dict())).hexdigest(),
                manifest_digest="a"*64, mapping_digest="b"*64, builder_digest="c"*64)
    bound = dict(**args, catalog_ref=ref, catalog_version="v1", store=store)
    digest = registry.register_catalog_attestation(**bound)
    assert registry.get_active() == (original, revision)
    assert registry.register_catalog_attestation(**bound) == digest
    with pytest.raises(GenerationConflict, match="conflict"):
        registry.register_catalog_attestation(**{**bound, "catalog_version": "changed"})
    registry.activate("g2", expected_active="g1", expected_revision=revision)
    registry = GenerationRegistry(tmp_path / "registry.sqlite3")
    assert registry.get_catalog_attestation(**args)["catalog_ref"]["digest"] == ref.digest
    with sqlite3.connect(registry.path) as connection:
        connection.execute("UPDATE catalog_attestations SET payload='{}'")
    with pytest.raises(GenerationConflict, match="digest mismatch"):
        registry.get_catalog_attestation(**args)
