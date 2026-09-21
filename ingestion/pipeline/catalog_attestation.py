"""Append-only catalog bindings for already verified historical generation records."""
from dataclasses import asdict
import hashlib
import json
import re

from core.contract_store import ContractRef, canonical_json


class CatalogAttestationMixin:
    @staticmethod
    def _initialize_catalog_attestations(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS catalog_attestations ("
                           "source_key TEXT PRIMARY KEY, digest TEXT NOT NULL, payload TEXT NOT NULL)")

    @staticmethod
    def _catalog_source(generation_id, record_digest, manifest_digest, mapping_digest, builder_digest):
        for value in (record_digest, manifest_digest, mapping_digest, builder_digest):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("invalid_catalog_source_digest")
        source = dict(generation_id=generation_id, record_digest=record_digest,
                      manifest_digest=manifest_digest, mapping_digest=mapping_digest, builder_digest=builder_digest)
        return source, hashlib.sha256(canonical_json(source)).hexdigest()

    def register_catalog_attestation(self, *, generation_id, record_digest, manifest_digest,
                                     mapping_digest, builder_digest, catalog_ref, catalog_version, store):
        from .generation_registry import GenerationConflict, GenerationRecord, GenerationState
        source, key = self._catalog_source(generation_id, record_digest, manifest_digest, mapping_digest, builder_digest)
        catalog_ref.validate()
        value = store.resolve(catalog_ref)
        if value.get("schema_version") != "catalog-contract-v2" or not isinstance(catalog_version, str) or not catalog_version:
            raise GenerationConflict("invalid catalog attestation contract")
        payload = {"schema_version": "legacy-catalog-attestation-v1", "source": source,
                   "catalog_ref": asdict(catalog_ref), "catalog_version": catalog_version}
        raw = canonical_json(payload).decode("utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT payload FROM generations WHERE generation_id=?", (generation_id,)).fetchone()
            if current is None:
                raise GenerationConflict("catalog source generation missing")
            record = GenerationRecord.from_dict(json.loads(current[0]))
            if record.state not in {GenerationState.ACTIVE, GenerationState.RETIRED}:
                raise GenerationConflict("catalog source generation unreadable")
            existing = connection.execute("SELECT digest,payload FROM catalog_attestations WHERE source_key=?", (key,)).fetchone()
            if existing:
                if existing != (digest, raw):
                    raise GenerationConflict("immutable catalog binding conflict")
                connection.commit()
                return digest
            if record.manifest_hash != manifest_digest or hashlib.sha256(canonical_json(record.to_dict())).hexdigest() != record_digest:
                raise GenerationConflict("catalog source changed before registration")
            self._archive_record(connection, record)
            connection.execute("INSERT INTO catalog_attestations VALUES(?,?,?)", (key, digest, raw))
            connection.commit()
        return digest

    def get_catalog_attestation(self, *, generation_id, record_digest, manifest_digest, mapping_digest, builder_digest):
        from .generation_registry import GenerationConflict, GenerationRecord, GenerationState
        source, key = self._catalog_source(generation_id, record_digest, manifest_digest, mapping_digest, builder_digest)
        with self._connect() as connection:
            connection.execute("BEGIN")
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalog_attestations'"
            ).fetchone():
                raise GenerationConflict("catalog attestation schema requires explicit maintenance")
            current = connection.execute("SELECT payload FROM generations WHERE generation_id=?", (generation_id,)).fetchone()
            row = connection.execute("SELECT digest,payload FROM catalog_attestations WHERE source_key=?", (key,)).fetchone()
            if current is None or GenerationRecord.from_dict(json.loads(current[0])).state not in {GenerationState.ACTIVE, GenerationState.RETIRED}:
                raise GenerationConflict("catalog source generation unreadable")
        if row is None:
            return None
        digest, raw = row
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
            raise GenerationConflict("catalog binding digest mismatch")
        payload = json.loads(raw)
        if payload.get("source") != source or payload.get("schema_version") != "legacy-catalog-attestation-v1":
            raise GenerationConflict("catalog binding source mismatch")
        ContractRef(**payload["catalog_ref"]).validate()
        return payload
