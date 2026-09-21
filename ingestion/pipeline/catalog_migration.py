"""Explicit legacy Catalog attachment; never alters ACTIVE or manifest bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from core.catalog_provider import MainProjectCatalogReader, catalog_builder_digest
from core.contract_store import ContractStore, canonical_json
from agent.protected_backup import create_windows_backup
from .generation_registry import GenerationRegistry, GenerationConflict
from .manifest import ManifestRef, verify_manifest_ref
from agent.checkpoint_migration import _snapshot_bytes


def attach_legacy_catalog(registry_path: Path, es_client, *, backup_path: Path,
                          expected_generation: str, expected_revision: int,
                          expected_record_digest: str) -> dict:
    """Caller must hold a maintenance window and protect the backup directory ACL.

    Only registry schema/history and an append-only binding are written. Object
    publication may leave an unreferenced object if registration subsequently fails.
    """
    registry_path = Path(registry_path).resolve()
    reader = GenerationRegistry(registry_path, read_only=True)
    record, revision = reader.get_active()
    if (record is None or record.generation_id != expected_generation or revision != expected_revision
            or hashlib.sha256(canonical_json(record.to_dict())).hexdigest() != expected_record_digest):
        raise GenerationConflict("legacy catalog source changed")
    manifest = verify_manifest_ref(reader.manifest_root, ManifestRef(
        locator=record.manifest_locator, digest=record.manifest_hash,
        byte_length=record.manifest_byte_length, schema_version=record.manifest_schema_version,
        store_id=record.manifest_store_id, locator_scheme=record.manifest_locator_scheme,
        digest_algorithm=record.manifest_digest_algorithm,
    ))
    mapping_response = es_client.indices.get_mapping(index=record.es_physical_index)
    mapping = getattr(mapping_response, "body", mapping_response)
    mapping_digest = hashlib.sha256(canonical_json(mapping)).hexdigest()
    legacy_mapping_digest = hashlib.sha256(json.dumps(
        mapping_response, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    settings_response = es_client.indices.get_settings(index=record.es_physical_index)
    settings = getattr(settings_response, "body", settings_response)
    actual_uuid = settings[record.es_physical_index]["settings"]["index"]["uuid"]
    info_response = es_client.info()
    info = getattr(info_response, "body", info_response)
    if (manifest.get("es_mapping_digest") not in {mapping_digest, legacy_mapping_digest}
            or manifest.get("es_index_uuid") != actual_uuid
            or manifest.get("es_cluster_id") != info.get("cluster_uuid")):
        raise GenerationConflict("legacy catalog physical source mismatch")
    # Reader consumes the exact mapping already verified, not a second mutable fetch.
    class FrozenIndices:
        def get_mapping(self, **kwargs):
            return mapping
    class FrozenClient:
        indices = FrozenIndices()
    catalog = MainProjectCatalogReader(FrozenClient(), record.es_physical_index).snapshot()
    with sqlite3.connect(registry_path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        backup_digest = create_windows_backup(_snapshot_bytes(connection), backup_path,
                                               context=b"study-generation-registry-backup-v1")
    writer = GenerationRegistry(registry_path)
    current, current_revision = writer.get_active()
    if current_revision != revision or current.to_dict() != record.to_dict():
        raise GenerationConflict("legacy catalog source changed after backup")
    current_mapping_response = es_client.indices.get_mapping(index=record.es_physical_index)
    current_mapping = getattr(current_mapping_response, "body", current_mapping_response)
    if hashlib.sha256(canonical_json(current_mapping)).hexdigest() != mapping_digest:
        raise GenerationConflict("legacy catalog mapping changed before publication")
    root = registry_path.parent / "contracts"
    root.mkdir(exist_ok=True)
    store = ContractStore(root, store_id="local-contracts-v1")
    reference = store.publish(catalog.identity_payload())
    digest = writer.register_catalog_attestation(
        generation_id=record.generation_id, record_digest=expected_record_digest,
        manifest_digest=record.manifest_hash, mapping_digest=mapping_digest,
        builder_digest=catalog_builder_digest(), catalog_ref=reference,
        catalog_version=catalog.version, store=store,
    )
    return {"binding_digest": digest, "catalog_digest": reference.digest, "backup_sha256": backup_digest,
            "active_revision": revision, "manifest_digest": record.manifest_hash}
