from __future__ import annotations

import uuid
import hashlib
import json
import time
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from core.kb_status import KnowledgeBaseStatus, SearchOutcome, SearchStatus


@dataclass(frozen=True, slots=True)
class ManifestIdentity:
    locator: str
    digest: str
    byte_length: int
    schema_version: str
    store_id: str = "local-manifest-store-v1"
    locator_scheme: str = "store-relative-v1"
    digest_algorithm: str = "sha256"


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    generation_id: str
    revision: int
    es_physical_index: str
    vector_collection: str
    embedding_model: str
    embedding_dimension: int
    generation_record_revision: int = 0
    generation_record_digest: str = ""
    manifest_locator: str = ""
    manifest_hash: str = ""
    manifest_schema_version: str = ""
    manifest_byte_length: int = 0
    manifest_store_id: str = "local-manifest-store-v1"
    manifest_locator_scheme: str = "store-relative-v1"
    manifest_digest_algorithm: str = "sha256"
    embedding_provider: str = ""
    embedding_model_revision: str = ""
    catalog_version: str = ""
    catalog_digest: str = ""
    catalog_store_id: str = "generation-manifest-catalog-v1"
    catalog_locator_scheme: str = "manifest-fragment-v1"
    catalog_locator: str = ""
    catalog_digest_algorithm: str = "sha256"
    document_count: int = 0
    chunk_count: int = 0
    document_set_digest: str = ""
    accepted_document_set_digest: str = ""
    chunk_set_digest: str = ""
    chunk_text_digest: str = ""
    safety_scope_digest: str = ""
    retrieval_profile: str = "rrf-v1"
    retrieval_profile_id: str = "rrf-v1"
    snapshot_schema_version: str = "generation-snapshot-v3"
    snapshot_digest: str = ""
    active_since_utc: str = ""
    es_cluster_id: str = ""
    es_index_uuid: str = ""
    es_mapping_digest: str = ""
    es_settings_digest: str = ""
    vector_backend: str = "chroma"
    vector_store_id: str = ""
    vector_collection_id: str = ""
    vector_metadata_digest: str = ""
    filesystem_discovered_count: int = 0
    accepted_document_count: int = 0
    quarantined_document_count: int = 0
    es_document_count: int = 0
    vector_document_count: int = 0
    es_document_set_digest: str = ""
    vector_document_set_digest: str = ""
    parser_name: str = ""
    parser_version: str = ""
    parser_config_digest: str = ""
    metadata_extractor_version: str = "bibliographic-v2"
    metadata_extractor_config_digest: str = ""
    chunker_name: str = "structural"
    chunker_version: str = "v2"
    chunker_config_digest: str = ""
    embedding_endpoint_identity: str = ""
    embedding_normalization: str = "provider-default"
    vector_distance_metric: str = "cosine"
    catalog_schema_version: str = "catalog-contract-v1"
    literature_query_schema_version: str = "literature-request-ir-v3"
    retrieval_config_digest: str = ""
    fusion_algorithm: str = "rrf"
    ranker_version: str = "rrf-v1"
    reranker_version: str = "none"
    registry_revision: int = 0
    manifest: ManifestIdentity | None = None
    catalog_ref: dict[str, Any] | None = None
    generation_record_ref: dict[str, Any] | None = None

    def canonical_digest(self) -> str:
        payload = asdict(self)
        payload.pop("snapshot_digest")
        if self.snapshot_schema_version in {"generation-snapshot-v2", "generation-snapshot-v3"}:
            if self.generation_record_ref is not None:
                raise ValueError("record reference requires snapshot v4")
            payload.pop("generation_record_ref")
        if self.snapshot_schema_version == "generation-snapshot-v2":
            if self.catalog_ref is not None:
                raise ValueError("catalog reference requires snapshot v3")
            payload.pop("catalog_ref")
        elif self.snapshot_schema_version not in {"generation-snapshot-v3", "generation-snapshot-v4"}:
            raise ValueError("unsupported generation snapshot schema")
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    @classmethod
    def restore(cls, payload: dict[str, Any]) -> "GenerationSnapshot":
        values = dict(payload)
        if isinstance(values.get("manifest"), dict):
            values["manifest"] = ManifestIdentity(**values["manifest"])
        result = cls(**values)
        compatibility = dict(payload)
        compatibility.pop("snapshot_digest", None)
        variants = [compatibility]
        # Dataclass round-trips add later optional fields as None. Accept every
        # historical presence form, but never ignore a non-null newer contract.
        removable = [name for name in ("catalog_ref", "generation_record_ref")
                     if compatibility.get(name) is None]
        for mask in range(1, 1 << len(removable)):
            variant = dict(compatibility)
            for position, name in enumerate(removable):
                if mask & (1 << position):
                    variant.pop(name, None)
            variants.append(variant)
        compatibility_digests = {hashlib.sha256(json.dumps(
            variant, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest() for variant in variants}
        if (not result.snapshot_digest
                or result.snapshot_digest not in {result.canonical_digest(), *compatibility_digests}):
            raise GenerationMismatch("persisted generation snapshot digest mismatch")
        return result


class GenerationMismatch(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedGenerationContext:
    snapshot: GenerationSnapshot
    verified_at: float
    integrity_token: Any
    kb_status: KnowledgeBaseStatus


class RetrievalGateway:
    """Preflight-first read gateway with request-fixed ingestion generation."""

    def __init__(self, backend: Any, *, generation_registry: Any | None = None,
                 identity_cache_ttl_seconds: float = 30.0) -> None:
        self.backend = backend
        self.registry = generation_registry
        self.identity_cache_ttl_seconds = max(0.0, float(identity_cache_ttl_seconds))
        self._identity_cache_lock = threading.RLock()
        self._cached_snapshot: GenerationSnapshot | None = None
        self._cached_snapshot_at = 0.0
        self._verified_snapshot_digest = ""
        self._verified_snapshot_at = 0.0
        self._integrity_monitor = None
        self._integrity_tokens: dict[str, str] = {}
        self._verified_contexts: dict[str, VerifiedGenerationContext] = {}

    def _change_token(self, index):
        if getattr(self.registry, "path", None) is None:
            return None
        from core.backend_integrity import BackendIntegrityMonitor, IntegrityCapabilityMissing
        with self._identity_cache_lock:
            if self._integrity_monitor is None:
                self._integrity_monitor = BackendIntegrityMonitor(self.backend)
            try:
                return self._integrity_monitor.token(index)
            except IntegrityCapabilityMissing as exc:
                raise GenerationMismatch("backend_integrity_capability_missing") from exc

    def close(self):
        with self._identity_cache_lock:
            if self._integrity_monitor is not None:
                self._integrity_monitor.close()
                self._integrity_monitor = None
            self._integrity_tokens.clear()
            self._cached_snapshot = None
            self._verified_snapshot_digest = ""

    @staticmethod
    def _es_document_tuples(es_client: Any, index_name: str) -> list[tuple[str, str, str]]:
        """Read the complete ES document identity set without a 10k truncation."""
        if callable(getattr(es_client, "options", None)):
            from elasticsearch import helpers
            hits = helpers.scan(
                es_client,
                index=index_name,
                query={
                    "query": {"match_all": {}},
                    "_source": ["document_id", "content_hash", "ingestion_generation"],
                },
                preserve_order=False,
            )
        else:
            response = es_client.search(index=index_name, body={
                "query": {"match_all": {}},
                "_source": ["document_id", "content_hash", "ingestion_generation"],
                "size": 10000,
            })
            hits = response.get("hits", {}).get("hits", [])
        return sorted({
            (str(source.get("document_id", "")), str(source.get("content_hash", "")),
             str(source.get("ingestion_generation", "")))
            for hit in hits
            for source in [hit.get("_source") or {}]
        })

    def pin(self, *, force_refresh: bool = False) -> GenerationSnapshot | None:
        if self.registry is None:
            return None
        active, revision = self.registry.get_active()
        if active is None:
            return None
        now = time.monotonic()
        with self._identity_cache_lock:
            cached = self._cached_snapshot
            if (not force_refresh and cached is not None
                    and cached.generation_id == active.generation_id
                    and cached.registry_revision == revision
                    and now - self._cached_snapshot_at <= self.identity_cache_ttl_seconds):
                self.verify_snapshot(cached)
                return cached
        strict_identity = getattr(self.registry, "path", None) is not None
        initial_token = self._change_token(active.es_physical_index)
        if strict_identity and not active.manifest_locator:
            raise GenerationMismatch("ACTIVE generation has no immutable manifest locator")
        manifest_payload = {}
        if active.manifest_locator:
            from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref
            manifest_root = getattr(self.registry, "manifest_root", None) or self.registry.path.parent / "manifests"
            manifest_payload = verify_manifest_ref(manifest_root, ManifestRef(
                locator=active.manifest_locator,
                digest=active.manifest_hash,
                byte_length=active.manifest_byte_length,
                schema_version=active.manifest_schema_version or "ingestion-manifest-v2",
                store_id=getattr(active, "manifest_store_id", "local-manifest-store-v1"),
                locator_scheme=getattr(active, "manifest_locator_scheme", "store-relative-v1"),
                digest_algorithm=getattr(active, "manifest_digest_algorithm", "sha256"),
            ))
        es_client = getattr(self.backend.es_manager, "es", None)
        es_info_response = es_client.info() if es_client is not None else {}
        mapping_response = es_client.indices.get_mapping(index=active.es_physical_index) if es_client is not None else {}
        settings_response = es_client.indices.get_settings(index=active.es_physical_index) if es_client is not None else {}
        es_info = self._response_body(es_info_response)
        mapping = self._response_body(mapping_response)
        settings = self._response_body(settings_response)
        index_meta = settings.get(active.es_physical_index, {}).get("settings", {}).get("index", {})
        vector_client = getattr(self.backend.vector_store, "client", None)
        collection = vector_client.get_collection(active.vector_collection) if vector_client is not None else None
        accepted = [item for item in manifest_payload.get("documents", []) if item.get("status") == "accepted"]
        quarantined = [item for item in manifest_payload.get("documents", []) if item.get("status") != "accepted"]
        canonical_digest = lambda value: hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()
        expected_document_tuples = sorted(
            (str(item.get("document_id", "")), str(item.get("source_hash", "")), active.generation_id)
            for item in accepted
        )
        es_document_tuples = expected_document_tuples
        if es_client is not None and callable(getattr(es_client, "search", None)):
            es_document_tuples = self._es_document_tuples(
                es_client, active.es_physical_index,
            )
        vector_document_tuples = expected_document_tuples
        vector_chunk_count = int(getattr(active, "chunk_count", 0))
        vector_chunk_set_digest = str(manifest_payload.get("chunk_set_digest", ""))
        if collection is not None and callable(getattr(collection, "get", None)):
            vector_payload = collection.get(include=["metadatas", "documents"])
            vector_metadata = [dict(item or {}) for item in vector_payload.get("metadatas") or []]
            vector_documents = [str(item or "") for item in vector_payload.get("documents") or []]
            vector_chunk_count = len(vector_metadata)
            vector_document_tuples = sorted({
                (str(item.get("document_id", "")), str(item.get("content_hash", "")),
                 str(item.get("ingestion_generation", "")))
                for item in vector_metadata
            })
            if len(vector_documents) == len(vector_metadata):
                chunk_rows = sorted(
                    f"{item.get('document_id', '')}:{item.get('content_hash', '')}:"
                    f"{item.get('chunk_index', '')}:"
                    f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
                    for item, text in zip(vector_metadata, vector_documents)
                )
                vector_chunk_set_digest = hashlib.sha256(
                    "\n".join(chunk_rows).encode("utf-8")
                ).hexdigest()
        if es_document_tuples != expected_document_tuples:
            raise GenerationMismatch("ACTIVE Elasticsearch document set differs from immutable manifest")
        if vector_document_tuples != expected_document_tuples:
            raise GenerationMismatch("ACTIVE vector document set differs from immutable manifest")
        if vector_chunk_count != int(getattr(active, "chunk_count", 0)):
            raise GenerationMismatch("ACTIVE vector chunk count differs from generation record")
        document_tuple_digest = lambda rows: hashlib.sha256(
            "\n".join(":".join(row) for row in rows).encode("utf-8")
        ).hexdigest()
        es_document_set_digest = document_tuple_digest(es_document_tuples)
        vector_document_set_digest = document_tuple_digest(vector_document_tuples)
        mapping_digest = canonical_digest(mapping)
        settings_digest = canonical_digest(settings)
        from ingestion.pipeline.manifest import vector_store_identity, is_legacy_vector_store_identity
        vector_store_id = vector_store_identity()
        vector_collection_id = str(getattr(collection, "id", ""))
        vector_metadata_digest = canonical_digest(getattr(collection, "metadata", {}) or {})
        physical_identity = {
            "generation_id": active.generation_id,
            "es_physical_index": active.es_physical_index,
            "vector_collection": active.vector_collection,
            "es_cluster_id": str(es_info.get("cluster_uuid", "")),
            "es_index_uuid": str(index_meta.get("uuid", "")),
            "es_mapping_digest": mapping_digest,
            "es_settings_digest": settings_digest,
            "vector_backend": "chroma",
            "vector_store_id": vector_store_id,
            "vector_collection_id": vector_collection_id,
            "vector_metadata_digest": vector_metadata_digest,
            "accepted_document_count": len(expected_document_tuples),
            "es_document_count": len(es_document_tuples),
            "vector_document_count": len(vector_document_tuples),
            "vector_chunk_count": vector_chunk_count,
            "accepted_document_set_digest": document_tuple_digest(expected_document_tuples),
            "es_document_set_digest": es_document_set_digest,
            "vector_document_set_digest": vector_document_set_digest,
            "chunk_set_digest": vector_chunk_set_digest,
        }
        if strict_identity:
            historical_response_digests = {
                "es_mapping_digest": canonical_digest(mapping_response),
                "es_settings_digest": canonical_digest(settings_response),
            }
            mismatches = [
                name for name, actual in physical_identity.items()
                if name in manifest_payload
                and manifest_payload.get(name) != actual
                and manifest_payload.get(name) != historical_response_digests.get(name)
                and not (name == "vector_store_id" and
                         is_legacy_vector_store_identity(manifest_payload.get(name)))
            ]
            if mismatches:
                raise GenerationMismatch(
                    "ACTIVE physical identity differs from immutable manifest: "
                    + ", ".join(sorted(mismatches))
                )
        endpoint_identity = canonical_digest({
            "provider": getattr(active, "embedding_provider", "") or "ollama",
            "endpoint": str(getattr(self.backend.embedder, "base_url", "")),
        })
        values = dict(
            generation_id=active.generation_id,
            revision=revision,
            registry_revision=revision,
            es_physical_index=active.es_physical_index,
            vector_collection=active.vector_collection,
            embedding_model=active.embedding_model,
            embedding_dimension=active.embedding_dimension,
            generation_record_revision=getattr(active, "record_revision", 0),
            generation_record_digest=canonical_digest(active.to_dict()),
            manifest_locator=getattr(active, "manifest_locator", ""),
            manifest_hash=getattr(active, "manifest_hash", ""),
            manifest_schema_version=getattr(active, "manifest_schema_version", ""),
            manifest_byte_length=getattr(active, "manifest_byte_length", 0),
            manifest_store_id=getattr(active, "manifest_store_id", "local-manifest-store-v1"),
            manifest_locator_scheme=getattr(active, "manifest_locator_scheme", "store-relative-v1"),
            manifest_digest_algorithm=getattr(active, "manifest_digest_algorithm", "sha256"),
            embedding_provider=getattr(active, "embedding_provider", "") or "ollama",
            embedding_model_revision=getattr(active, "embedding_model_revision", "") or active.embedding_model,
            catalog_version=getattr(active, "catalog_version", "") or "catalog-contract-v1",
            catalog_digest=getattr(active, "catalog_digest", "") or canonical_digest({
                "schema": "catalog-contract-v1", "mapping": mapping,
            }),
            catalog_store_id="generation-manifest-catalog-v1",
            catalog_locator_scheme="manifest-fragment-v1",
            catalog_locator=(
                f"{getattr(active, 'manifest_locator', '')}#catalog"
                if getattr(active, "manifest_locator", "") else ""
            ),
            catalog_digest_algorithm="sha256",
            document_count=getattr(active, "document_count", 0),
            chunk_count=getattr(active, "chunk_count", 0),
            document_set_digest=getattr(active, "document_set_digest", ""),
            accepted_document_set_digest=getattr(active, "document_set_digest", ""),
            chunk_set_digest=getattr(active, "chunk_set_digest", ""),
            chunk_text_digest=vector_chunk_set_digest,
            safety_scope_digest=canonical_digest({
                "source_inventory_root_identity": manifest_payload.get(
                    "source_inventory_root_identity", ""
                ),
                "generation_id": active.generation_id,
                "read_only": True,
            }),
            active_since_utc=getattr(active, "activated_at_utc", ""),
            es_cluster_id=str(es_info.get("cluster_uuid", "")),
            es_index_uuid=str(index_meta.get("uuid", "")),
            es_mapping_digest=mapping_digest,
            es_settings_digest=settings_digest,
            vector_backend=str(manifest_payload.get("vector_backend") or "chroma"),
            vector_store_id=vector_store_id,
            vector_collection_id=vector_collection_id,
            vector_metadata_digest=vector_metadata_digest,
            filesystem_discovered_count=int(manifest_payload.get("discovered_file_count", len(manifest_payload.get("documents", [])))),
            accepted_document_count=int(manifest_payload.get("accepted_document_count", len(accepted))),
            quarantined_document_count=int(manifest_payload.get("quarantined_document_count", len(quarantined))),
            es_document_count=len(es_document_tuples),
            vector_document_count=len(vector_document_tuples),
            es_document_set_digest=es_document_set_digest,
            vector_document_set_digest=vector_document_set_digest,
            parser_name=str(manifest_payload.get("parser_name", "")),
            parser_version=str(manifest_payload.get("parser_version", "")),
            parser_config_digest=str(manifest_payload.get("parser_config_digest", "")),
            metadata_extractor_version=str(manifest_payload.get("metadata_extractor_version", "")),
            metadata_extractor_config_digest=str(manifest_payload.get("metadata_extractor_config_digest", "")),
            chunker_name=str(manifest_payload.get("chunker_name", "")),
            chunker_version=str(manifest_payload.get("chunker_version", "")),
            chunker_config_digest=str(manifest_payload.get("chunker_config_digest", "")),
            embedding_endpoint_identity=endpoint_identity,
            embedding_normalization=str(manifest_payload.get("embedding_normalization", "")),
            vector_distance_metric=str(manifest_payload.get("vector_distance_metric", "")),
            catalog_schema_version=str(manifest_payload.get("catalog_schema_version", "")),
            retrieval_config_digest=hashlib.sha256(b"document-first+rrf-k60+strict-sense-v2").hexdigest(),
            manifest=ManifestIdentity(
                locator=getattr(active, "manifest_locator", ""),
                digest=getattr(active, "manifest_hash", ""),
                byte_length=int(getattr(active, "manifest_byte_length", 0)),
                schema_version=getattr(active, "manifest_schema_version", ""),
                store_id=getattr(active, "manifest_store_id", "local-manifest-store-v1"),
                locator_scheme=getattr(active, "manifest_locator_scheme", "store-relative-v1"),
                digest_algorithm=getattr(active, "manifest_digest_algorithm", "sha256"),
            ),
        )
        catalog_ref = manifest_payload.get("catalog_ref")
        catalog_version = values["catalog_version"]
        if catalog_ref is None and callable(getattr(self.registry, "get_catalog_attestation", None)):
            from core.catalog_provider import catalog_builder_digest
            attestation = self.registry.get_catalog_attestation(
                generation_id=active.generation_id, record_digest=values["generation_record_digest"],
                manifest_digest=active.manifest_hash, mapping_digest=mapping_digest,
                builder_digest=catalog_builder_digest(),
            )
            if attestation:
                catalog_ref = attestation["catalog_ref"]
                catalog_version = attestation["catalog_version"]
        if catalog_ref is not None:
            from core.contract_store import ContractRef, ContractStore
            ref = ContractRef(**catalog_ref)
            contract = ContractStore(self.registry.path.parent / "contracts", store_id="local-contracts-v1").resolve(ref)
            if contract.get("schema_version") != "catalog-contract-v2":
                raise GenerationMismatch("catalog object schema mismatch")
            values.update(catalog_ref=asdict(ref), catalog_digest=ref.digest, catalog_version=catalog_version,
                          catalog_store_id=ref.store_id, catalog_locator_scheme=ref.locator_scheme,
                          catalog_locator=ref.locator, catalog_digest_algorithm=ref.digest_algorithm,
                          catalog_schema_version=ref.schema_version)
        elif strict_identity:
            raise GenerationMismatch("catalog contract requires explicit maintenance; no resolvable CatalogRef")
        if strict_identity:
            record_raw = json.dumps(active.to_dict(), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")
            values.update(snapshot_schema_version="generation-snapshot-v4", generation_record_ref={
                "store_id": "generation-registry-v1", "locator_scheme": "generation-revision-v1",
                "locator": {"generation_id": active.generation_id, "revision": active.record_revision},
                "digest_algorithm": "sha256", "digest": hashlib.sha256(record_raw).hexdigest(),
                "byte_length": len(record_raw), "schema_version": "generation-record-v1",
            })
        snapshot = GenerationSnapshot(**values)
        snapshot = replace(snapshot, snapshot_digest=snapshot.canonical_digest())
        required = (
            "generation_id", "es_physical_index",
            "vector_collection", "embedding_provider", "embedding_model",
            "embedding_model_revision", "embedding_dimension", "catalog_version",
            "catalog_digest", "retrieval_config_digest",
            "generation_record_digest", "catalog_store_id",
            "catalog_locator_scheme",
            "catalog_digest_algorithm",
            "manifest_store_id", "manifest_locator_scheme", "manifest_digest_algorithm",
        )
        if strict_identity:
            required += (
                "catalog_locator",
                "es_cluster_id", "es_index_uuid", "es_mapping_digest",
                "es_settings_digest", "vector_backend", "vector_store_id",
                "vector_collection_id", "vector_metadata_digest",
                "accepted_document_set_digest", "es_document_set_digest",
                "vector_document_set_digest", "chunk_set_digest", "parser_name",
                "chunk_text_digest", "safety_scope_digest",
                "parser_version", "parser_config_digest", "metadata_extractor_version",
                "metadata_extractor_config_digest", "chunker_name", "chunker_version",
                "chunker_config_digest", "embedding_endpoint_identity",
                "embedding_normalization", "vector_distance_metric",
                "catalog_schema_version", "literature_query_schema_version",
                "retrieval_profile_id", "fusion_algorithm", "ranker_version",
                "reranker_version", "manifest_locator", "manifest_hash",
            )
        missing = [name for name in required if getattr(snapshot, name, None) in {None, "", 0}]
        if missing:
            raise GenerationMismatch(f"GenerationSnapshot missing required identity: {missing}")
        if initial_token != self._change_token(snapshot.es_physical_index):
            raise GenerationMismatch("backend changed during generation pin")
        with self._identity_cache_lock:
            if initial_token is not None:
                self._integrity_tokens[snapshot.snapshot_digest] = initial_token
            self._cached_snapshot = snapshot
            self._cached_snapshot_at = time.monotonic()
            # pin() has just performed the complete physical/manifest identity scan.
            self._verified_snapshot_digest = snapshot.snapshot_digest
            self._verified_snapshot_at = self._cached_snapshot_at
        return snapshot

    def pin_verified(self) -> VerifiedGenerationContext | None:
        """Perform the complete identity scan once and reuse it for this request."""
        snapshot = self.pin()
        if snapshot is None:
            return None
        status = self.status(snapshot)
        context = VerifiedGenerationContext(
            snapshot=snapshot, verified_at=time.time(),
            integrity_token=self._integrity_tokens.get(snapshot.snapshot_digest),
            kb_status=status,
        )
        with self._identity_cache_lock:
            self._verified_contexts[snapshot.snapshot_digest] = context
        return context

    def verified_context(self, snapshot: GenerationSnapshot) -> VerifiedGenerationContext | None:
        with self._identity_cache_lock:
            return self._verified_contexts.get(snapshot.snapshot_digest)

    def assert_integrity(self, context: VerifiedGenerationContext) -> None:
        current = self._change_token(context.snapshot.es_physical_index)
        if context.integrity_token is not None and current != context.integrity_token:
            raise GenerationMismatch("backend changed during verified request")

    @staticmethod
    def _canonical_digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _response_body(response):
        return getattr(response, "body", response)

    def verify_snapshot(self, snapshot: GenerationSnapshot, *, force: bool = False) -> None:
        """Verify immutable physical identities without consulting current ACTIVE."""
        if (getattr(self.registry, "path", None) is not None
                and (not snapshot.snapshot_digest or snapshot.canonical_digest() != snapshot.snapshot_digest)):
            raise GenerationMismatch("snapshot canonical digest mismatch")
        if snapshot.generation_record_ref is not None:
            ref = snapshot.generation_record_ref
            expected = {"store_id", "locator_scheme", "locator", "digest_algorithm", "digest", "byte_length", "schema_version"}
            if (set(ref) != expected or ref["store_id"] != "generation-registry-v1"
                    or ref["locator_scheme"] != "generation-revision-v1"
                    or ref["locator"] != {"generation_id": snapshot.generation_id, "revision": snapshot.generation_record_revision}
                    or ref["digest_algorithm"] != "sha256" or ref["digest"] != snapshot.generation_record_digest
                    or ref["schema_version"] != "generation-record-v1"):
                raise GenerationMismatch("generation record reference mismatch")
            record = self.registry.get_record_revision(snapshot.generation_id, snapshot.generation_record_revision,
                                                       expected_digest=ref["digest"])
            raw = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(raw) != ref["byte_length"]:
                raise GenerationMismatch("generation record reference length mismatch")
        if snapshot.catalog_ref is not None:
            from core.contract_store import ContractRef, ContractStore
            ref = ContractRef(**snapshot.catalog_ref)
            if (ref.digest != snapshot.catalog_digest or ref.locator != snapshot.catalog_locator
                    or ref.store_id != snapshot.catalog_store_id
                    or ref.locator_scheme != snapshot.catalog_locator_scheme
                    or ref.digest_algorithm != snapshot.catalog_digest_algorithm
                    or ref.schema_version != snapshot.catalog_schema_version):
                raise GenerationMismatch("snapshot catalog reference mismatch")
            ContractStore(self.registry.path.parent / "contracts", store_id="local-contracts-v1").resolve(ref)
        if snapshot.generation_record_revision and callable(getattr(self.registry, "get", None)):
            current = self.registry.get(snapshot.generation_id)
            state = getattr(getattr(current, "state", None), "value", "")
            if current is None or state not in {"active", "retired"}:
                raise GenerationMismatch("pinned generation is no longer readable")
            if current.record_revision == snapshot.generation_record_revision:
                if self._canonical_digest(current.to_dict()) != snapshot.generation_record_digest:
                    raise GenerationMismatch("pinned generation record changed")
            else:
                try:
                    self.registry.get_record_revision(snapshot.generation_id, snapshot.generation_record_revision,
                                                      expected_digest=snapshot.generation_record_digest)
                except Exception:
                    raise GenerationMismatch("pinned generation history unavailable") from None
        current_token = self._change_token(snapshot.es_physical_index)
        with self._identity_cache_lock:
            previous_token = self._integrity_tokens.get(snapshot.snapshot_digest)
            if previous_token is not None and current_token != previous_token:
                raise GenerationMismatch("backend changed since snapshot validation")
            if (not force and snapshot.snapshot_digest
                    and snapshot.snapshot_digest == self._verified_snapshot_digest
                    and (previous_token is not None or time.monotonic() - self._verified_snapshot_at
                         <= self.identity_cache_ttl_seconds)):
                return
        es_client = getattr(self.backend.es_manager, "es", None)
        mapping = {}
        settings = {}
        if es_client is not None and snapshot.es_index_uuid:
            mapping = self._response_body(es_client.indices.get_mapping(index=snapshot.es_physical_index))
            settings = self._response_body(es_client.indices.get_settings(index=snapshot.es_physical_index))
            current_uuid = str(settings.get(snapshot.es_physical_index, {}).get(
                "settings", {}
            ).get("index", {}).get("uuid", ""))
            info = self._response_body(es_client.info()) if callable(getattr(es_client, "info", None)) else {}
            checks = {
                "Elasticsearch cluster UUID": (
                    str(info.get("cluster_uuid", "")), snapshot.es_cluster_id,
                ),
                "Elasticsearch index UUID": (current_uuid, snapshot.es_index_uuid),
                "Elasticsearch mapping": (
                    self._canonical_digest(mapping), snapshot.es_mapping_digest,
                ),
                "Elasticsearch settings": (
                    self._canonical_digest(settings), snapshot.es_settings_digest,
                ),
            }
            changed = [
                f"{name} changed" for name, pair in checks.items()
                if pair[1] not in {None, ""} and pair[0] != pair[1]
            ]
            if changed:
                raise GenerationMismatch("pinned identity changed: " + ", ".join(changed))
        vector_client = getattr(self.backend.vector_store, "client", None)
        collection = None
        if vector_client is not None and snapshot.vector_collection_id:
            collection = vector_client.get_collection(snapshot.vector_collection)
            from ingestion.pipeline.manifest import vector_store_identity
            vector_store_id = vector_store_identity()
            checks = {
                "vector store": (vector_store_id, snapshot.vector_store_id),
                "vector collection": (
                    str(getattr(collection, "id", "")), snapshot.vector_collection_id,
                ),
                "vector metadata": (
                    self._canonical_digest(getattr(collection, "metadata", {}) or {}),
                    snapshot.vector_metadata_digest,
                ),
            }
            changed = [
                f"{name} changed" for name, pair in checks.items()
                if pair[1] not in {None, ""} and pair[0] != pair[1]
            ]
            if changed:
                raise GenerationMismatch("pinned identity changed: " + ", ".join(changed))

        # A real registry-backed snapshot carries a complete ManifestRef.  Scan
        # both physical stores because neither ES nor Chroma is assumed to be
        # immutable merely from its locator.
        if snapshot.manifest is None or not snapshot.manifest.locator:
            return
        from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref

        manifest_root = getattr(self.registry, "manifest_root", None)
        if manifest_root is None:
            raise GenerationMismatch("manifest root unavailable for pinned snapshot")
        manifest = verify_manifest_ref(manifest_root, ManifestRef(
            locator=snapshot.manifest.locator,
            digest=snapshot.manifest.digest,
            byte_length=snapshot.manifest.byte_length,
            store_id=snapshot.manifest.store_id,
            locator_scheme=snapshot.manifest.locator_scheme,
            digest_algorithm=snapshot.manifest.digest_algorithm,
            schema_version=snapshot.manifest.schema_version,
        ))
        manifest_checks = {
            "generation_id": snapshot.generation_id,
            "es_physical_index": snapshot.es_physical_index,
            "vector_collection": snapshot.vector_collection,
            "es_cluster_id": snapshot.es_cluster_id,
            "es_index_uuid": snapshot.es_index_uuid,
            "es_mapping_digest": snapshot.es_mapping_digest,
            "es_settings_digest": snapshot.es_settings_digest,
            "vector_backend": snapshot.vector_backend,
            "vector_store_id": snapshot.vector_store_id,
            "vector_collection_id": snapshot.vector_collection_id,
            "vector_metadata_digest": snapshot.vector_metadata_digest,
            "discovered_file_count": snapshot.filesystem_discovered_count,
            "accepted_document_count": snapshot.accepted_document_count,
            "quarantined_document_count": snapshot.quarantined_document_count,
            "es_document_count": snapshot.es_document_count,
            "vector_document_count": snapshot.vector_document_count,
            "vector_chunk_count": snapshot.chunk_count,
            "accepted_document_set_digest": snapshot.accepted_document_set_digest,
            "es_document_set_digest": snapshot.es_document_set_digest,
            "vector_document_set_digest": snapshot.vector_document_set_digest,
            "chunk_set_digest": snapshot.chunk_set_digest,
            "parser_name": snapshot.parser_name,
            "parser_version": snapshot.parser_version,
            "parser_config_digest": snapshot.parser_config_digest,
            "metadata_extractor_version": snapshot.metadata_extractor_version,
            "metadata_extractor_config_digest": snapshot.metadata_extractor_config_digest,
            "chunker_name": snapshot.chunker_name,
            "chunker_version": snapshot.chunker_version,
            "chunker_config_digest": snapshot.chunker_config_digest,
            "embedding_provider": snapshot.embedding_provider,
            "embedding_model": snapshot.embedding_model,
            "embedding_model_revision": snapshot.embedding_model_revision,
            "embedding_dimension": snapshot.embedding_dimension,
            "embedding_normalization": snapshot.embedding_normalization,
            "vector_distance_metric": snapshot.vector_distance_metric,
            "catalog_schema_version": snapshot.catalog_schema_version,
            "catalog_version": snapshot.catalog_version,
            "catalog_digest": snapshot.catalog_digest,
        }
        if snapshot.catalog_ref is not None and manifest.get("catalog_ref") is None:
            # Legacy immutable manifests describe their original Catalog. The
            # separately registered attestation binds the new contract without
            # rewriting those original bytes or confusing their two identities.
            from core.catalog_provider import catalog_builder_digest
            binding = self.registry.get_catalog_attestation(
                generation_id=snapshot.generation_id,
                record_digest=snapshot.generation_record_digest,
                manifest_digest=snapshot.manifest_hash,
                mapping_digest=snapshot.es_mapping_digest,
                builder_digest=catalog_builder_digest(),
            )
            if (binding is None or binding.get("catalog_ref") != snapshot.catalog_ref
                    or binding.get("catalog_version") != snapshot.catalog_version):
                raise GenerationMismatch("historical catalog attestation unavailable or changed")
            for key in ("catalog_schema_version", "catalog_version", "catalog_digest"):
                manifest_checks.pop(key)
        mismatches = [
            name for name, expected in manifest_checks.items()
            if manifest.get(name) != expected
        ]
        if mismatches:
            raise GenerationMismatch(
                "pinned snapshot differs from immutable manifest: "
                + ", ".join(sorted(mismatches))
            )

        es_rows = self._es_document_tuples(es_client, snapshot.es_physical_index)
        vector_payload = collection.get(include=["metadatas", "documents"])
        vector_metadata = [dict(item or {}) for item in vector_payload.get("metadatas") or []]
        vector_documents = [str(item or "") for item in vector_payload.get("documents") or []]
        vector_rows = sorted({
            (str(item.get("document_id", "")), str(item.get("content_hash", "")),
             str(item.get("ingestion_generation", "")))
            for item in vector_metadata
        })
        set_digest = lambda rows: hashlib.sha256(
            "\n".join(":".join(row) for row in rows).encode("utf-8")
        ).hexdigest()
        physical_checks = {
            "ES document count": (len(es_rows), snapshot.es_document_count),
            "vector document count": (len(vector_rows), snapshot.vector_document_count),
            "vector chunk count": (len(vector_metadata), snapshot.chunk_count),
            "ES document set": (set_digest(es_rows), snapshot.es_document_set_digest),
            "vector document set": (
                set_digest(vector_rows), snapshot.vector_document_set_digest,
            ),
        }
        if len(vector_documents) != len(vector_metadata):
            raise GenerationMismatch("vector chunk texts unavailable for digest verification")
        chunk_rows = sorted(
            f"{item.get('document_id', '')}:{item.get('content_hash', '')}:"
            f"{item.get('chunk_index', '')}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            for item, text in zip(vector_metadata, vector_documents)
        )
        physical_checks["vector chunk set"] = (
            hashlib.sha256("\n".join(chunk_rows).encode("utf-8")).hexdigest(),
            snapshot.chunk_set_digest,
        )
        changed = [name for name, pair in physical_checks.items() if pair[0] != pair[1]]
        if changed:
            raise GenerationMismatch("pinned physical set changed: " + ", ".join(changed))
        if current_token != self._change_token(snapshot.es_physical_index):
            raise GenerationMismatch("backend changed during snapshot validation")
        with self._identity_cache_lock:
            if current_token is not None:
                self._integrity_tokens[snapshot.snapshot_digest] = current_token
            self._verified_snapshot_digest = snapshot.snapshot_digest
            self._verified_snapshot_at = time.monotonic()

    def status(self, snapshot: GenerationSnapshot | None = None) -> KnowledgeBaseStatus:
        pinned = snapshot if snapshot is not None else self.pin()
        return self.backend.es_manager.inspect_status(
            vector_store=getattr(self.backend, "vector_store", None),
            embedding_model=(
                pinned.embedding_model if pinned else
                str(getattr(getattr(self.backend, "embedder", None), "model", ""))
            ),
            embedding_dimension=pinned.embedding_dimension if pinned else 1024,
            generation=pinned.generation_id if pinned else "",
            index_name=pinned.es_physical_index if pinned else None,
            vector_collection=pinned.vector_collection if pinned else None,
        )

    def inventory(self, snapshot: GenerationSnapshot | None = None, *, cursor: int = 0,
                  limit: int = 50) -> dict[str, Any]:
        pinned = snapshot if snapshot is not None else self.pin()
        if pinned is None:
            raise GenerationMismatch("inventory requires an ACTIVE generation")
        self.verify_snapshot(pinned)
        from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref
        manifest_root = getattr(self.registry, "manifest_root", None) or self.registry.path.parent / "manifests"
        manifest = verify_manifest_ref(manifest_root, ManifestRef(
            locator=pinned.manifest_locator,
            digest=pinned.manifest_hash,
            byte_length=pinned.manifest_byte_length,
            store_id=pinned.manifest_store_id,
            locator_scheme=pinned.manifest_locator_scheme,
            digest_algorithm=pinned.manifest_digest_algorithm,
            schema_version=pinned.manifest_schema_version or "ingestion-manifest-v2",
        ))
        manifest_docs = list(manifest.get("documents") or [])
        accepted = {str(item.get("filename")) for item in manifest_docs if item.get("status") == "accepted"}
        quarantined = {str(item.get("filename")) for item in manifest_docs if item.get("status") != "accepted"}
        pdf_dir = Path(str(getattr(self.backend, "pdf_dir", "")))
        filesystem = {item.name for item in pdf_dir.glob("*.pdf")} if pdf_dir.is_dir() else set()
        response = self.backend.es_manager.es.search(
            index=pinned.es_physical_index,
            body={"query": {"match_all": {}}, "_source": ["filename"], "size": 10000},
        )
        es_files = {str((item.get("_source") or {}).get("filename"))
                    for item in response.get("hits", {}).get("hits", [])}
        collection = self.backend.vector_store.client.get_collection(pinned.vector_collection)
        vector_payload = collection.get(include=["metadatas"])
        vector_files = {str((item or {}).get("filename"))
                        for item in vector_payload.get("metadatas") or []}
        ordered_accepted = sorted(accepted)
        start = max(0, int(cursor))
        page_limit = max(1, min(int(limit), 100))
        items = ordered_accepted[start:start + page_limit]
        next_cursor = start + len(items) if start + len(items) < len(ordered_accepted) else None
        return {
            "generation_id": pinned.generation_id,
            "filesystem": sorted(filesystem), "manifest_accepted": sorted(accepted),
            "manifest_quarantined": sorted(quarantined), "es_indexed": sorted(es_files),
            "vector_indexed": sorted(vector_files),
            "not_in_manifest": sorted(filesystem - accepted - quarantined),
            "accepted_missing_es": sorted(accepted - es_files),
            "accepted_missing_vector": sorted(accepted - vector_files),
            "counts": {"filesystem": len(filesystem), "accepted": len(accepted),
                       "quarantined": len(quarantined), "es": len(es_files),
                       "vector_documents": len(vector_files)},
            "total_documents": len(ordered_accepted), "items": items,
            "next_cursor": next_cursor, "has_more": next_cursor is not None,
            "truncated": next_cursor is not None, "complete": next_cursor is None,
        }

    def find_document(
        self, identity: str, *, snapshot: GenerationSnapshot | None = None
    ) -> SearchOutcome:
        trace_id = uuid.uuid4().hex
        snapshot = snapshot if snapshot is not None else self.pin()
        verified = self.verified_context(snapshot) if snapshot is not None else None
        if snapshot is not None and verified is None:
            self.verify_snapshot(snapshot)
        status = verified.kb_status if verified is not None else self.status(snapshot)
        blocked = _preflight_failure(status, trace_id)
        if blocked:
            return blocked
        matches = self.backend.es_manager.find_document(
            identity, index_name=snapshot.es_physical_index if snapshot else None
        )
        _validate_generation(matches, snapshot)
        if verified is not None:
            self.assert_integrity(verified)
        elif snapshot is not None:
            self.verify_snapshot(snapshot)
        return SearchOutcome(
            SearchStatus.MATCHED if matches else SearchStatus.DOCUMENT_ABSENT,
            evidence=matches,
            executed_channels=["es"],
            trace_id=trace_id,
            generation=status.ingestion_generation,
        )

    def search(self, query: str, *, top_k: int = 10) -> SearchOutcome:
        trace_id = uuid.uuid4().hex
        snapshot = self.pin()
        if self.registry is not None and snapshot is None:
            return SearchOutcome(SearchStatus.INDEX_MISSING, diagnostics=["no ACTIVE generation"], trace_id=trace_id)
        status = self.status(snapshot)
        blocked = _preflight_failure(status, trace_id)
        if blocked:
            return blocked
        if snapshot is None:
            outcome = self.backend.es_manager.search_bm25_outcome(
                query, size=top_k, trace_id=trace_id
            )
        else:
            try:
                self.verify_snapshot(snapshot)
                evidence = self.backend.es_manager._search_bm25_strict(
                    query, size=top_k, index_name=snapshot.es_physical_index
                )
                _validate_generation(evidence, snapshot)
                self.verify_snapshot(snapshot)
                outcome = SearchOutcome(
                    SearchStatus.MATCHED if evidence else SearchStatus.NO_MATCH,
                    evidence=evidence,
                    executed_channels=["es"],
                    trace_id=trace_id,
                )
            except Exception as exc:
                return SearchOutcome(
                    SearchStatus.FAILED,
                    diagnostics=[f"{type(exc).__name__}: {exc}"],
                    trace_id=trace_id,
                    generation=snapshot.generation_id,
                )
        outcome.generation = status.ingestion_generation
        return outcome

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        snapshot: GenerationSnapshot | None = None,
        mode: str = "hybrid",
        topic_terms: list[str] | None = None,
        temporal_range: dict[str, int] | None = None,
        sense_id: str = "",
        language: str = "",
        document_type: str = "",
        document_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> SearchOutcome:
        snapshot = snapshot if snapshot is not None else self.pin()
        verified = self.verified_context(snapshot) if snapshot is not None else None
        if self.registry is not None and snapshot is None:
            return SearchOutcome(
                SearchStatus.INDEX_MISSING, diagnostics=["no ACTIVE generation"],
                trace_id=uuid.uuid4().hex,
            )
        if snapshot is None:
            result = self.backend.hybrid_search(query, top_n=top_k, include_chunks=True)
            docs, chunks = result if isinstance(result, tuple) else (result, [])
            return SearchOutcome(
                SearchStatus.MATCHED if docs or chunks else SearchStatus.NO_MATCH,
                evidence=[{"channel": "bm25", **item} for item in docs]
                + [{"channel": "vector", **item} for item in chunks],
                executed_channels=["es", "vector"],
                trace_id=uuid.uuid4().hex,
            )
        stage_latency_ms: dict[str, float] = {}
        started = time.perf_counter()
        try:
            if verified is None:
                self.verify_snapshot(snapshot)
        except Exception as exc:
            stage_latency_ms["snapshot_verify"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            return SearchOutcome(
                SearchStatus.FAILED, diagnostics=[f"generation_mismatch: {exc}"],
                trace_id=uuid.uuid4().hex, generation=snapshot.generation_id,
                stage_latency_ms=stage_latency_ms,
            )
        stage_latency_ms["snapshot_verify"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        started = time.perf_counter()
        status = verified.kb_status if verified is not None else self.status(snapshot)
        stage_latency_ms["preflight"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        trace_id = uuid.uuid4().hex
        blocked = _preflight_failure(status, trace_id)
        if blocked:
            blocked.generation = snapshot.generation_id
            return blocked
        if mode != "document" and (
            not status.vector_collection_exists or status.vector_chunk_count <= 0
        ):
            return SearchOutcome(
                SearchStatus.FAILED,
                diagnostics=["vector collection is missing or empty"],
                trace_id=trace_id, generation=snapshot.generation_id,
                stage_latency_ms=stage_latency_ms,
            )
        try:
            started = time.perf_counter()
            docs = self.backend.es_manager._search_bm25_strict(
                " ".join(topic_terms or [query]), size=top_k * 2,
                index_name=snapshot.es_physical_index,
                temporal_range=temporal_range,
                document_first=mode == "document",
                language=language, document_type=document_type,
                document_ids=document_ids,
            )
            docs = [item for item in docs if _matches_literature_sense(item, sense_id)]
            stage_latency_ms["document_search"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            if mode == "document":
                chunks = []
                stage_latency_ms["passage_search"] = 0.0
            else:
                vector_kwargs = {"top_k": top_k * 2}
                if document_ids:
                    vector_kwargs["document_ids"] = document_ids
                vector_kwargs["section_types"] = section_types
                vector_kwargs["exclude_references"] = "references" not in (section_types or [])
                started = time.perf_counter()
                chunks = self.backend.vector_store.search_collection(
                    snapshot.vector_collection, query, **vector_kwargs
                )
                stage_latency_ms["passage_search"] = round(
                    (time.perf_counter() - started) * 1000.0, 3
                )
            _validate_generation(docs, snapshot)
            _validate_generation(chunks, snapshot)
        except Exception as exc:
            return SearchOutcome(
                SearchStatus.FAILED,
                diagnostics=[f"{type(exc).__name__}: {exc}"],
                trace_id=trace_id,
                generation=snapshot.generation_id,
                stage_latency_ms=stage_latency_ms,
            )
        started = time.perf_counter()
        ranked_docs = _rrf_documents(docs, chunks, top_k)
        if verified is not None:
            self.assert_integrity(verified)
        else:
            self.verify_snapshot(snapshot)
        stage_latency_ms["fusion"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        return SearchOutcome(
            SearchStatus.MATCHED if docs or chunks else SearchStatus.NO_MATCH,
            evidence=[{"channel": "bm25", **item} for item in ranked_docs]
            + [{"channel": "vector", **item} for item in chunks],
            executed_channels=["es", "vector"],
            trace_id=trace_id,
            generation=snapshot.generation_id,
            stage_latency_ms=stage_latency_ms,
        )


def _matches_literature_sense(item: dict[str, Any], sense_id: str) -> bool:
    if not sense_id:
        return True
    haystack = " ".join(str(item.get(key, "")) for key in (
        "title", "filename", "topic_aliases_text", "content", "highlights",
    )).casefold()
    if sense_id == "echo-state-network":
        heading = " ".join(str(item.get(key, "")) for key in ("title", "filename")).casefold()
        if any(token in heading for token in (
            "echo belief network", "回声信念网络", "enterprise social network",
            "enterprise social networks", "edible swiftlet", "swiftlet's nest",
        )):
            return False
        return "echo state network" in haystack or "echo-state network" in haystack or "回声状态网络" in haystack
    if sense_id == "markov-chain-monte-carlo":
        heading = " ".join(str(item.get(key, "")) for key in ("title", "filename")).casefold()
        return "markov chain monte carlo" in heading or "马尔科夫链蒙特卡洛" in heading or "mcmc" in heading
    return True


def _validate_generation(
    evidence: list[dict[str, Any]], snapshot: GenerationSnapshot | None
) -> None:
    if snapshot is None:
        return
    mismatches = [
        str(item.get("generation", ""))
        for item in evidence
        if str(item.get("generation", "")) != snapshot.generation_id
    ]
    if mismatches:
        raise GenerationMismatch(
            f"observation generation mismatch: expected={snapshot.generation_id}, got={mismatches[:3]}"
        )


def _preflight_failure(status: KnowledgeBaseStatus, trace_id: str) -> SearchOutcome | None:
    if not status.provider_available:
        return SearchOutcome(
            SearchStatus.PROVIDER_UNAVAILABLE,
            diagnostics=[status.health_reason],
            trace_id=trace_id,
        )
    if not status.index_exists:
        return SearchOutcome(SearchStatus.INDEX_MISSING, trace_id=trace_id)
    if status.document_count == 0:
        return SearchOutcome(SearchStatus.EMPTY_SOURCE, trace_id=trace_id)
    return None


def _rrf_documents(docs: list[dict[str, Any]], chunks: list[dict[str, Any]],
                   top_k: int, *, rrf_k: int = 60) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(docs, start=1):
        key = str(item.get("document_id") or item.get("filename") or "")
        if not key:
            continue
        row = fused.setdefault(key, dict(item, rrf=0.0))
        row["rrf"] = float(row.get("rrf", 0.0)) + 1.0 / (rrf_k + rank)
    seen_vector = set()
    for rank, item in enumerate(chunks, start=1):
        key = str(item.get("document_id") or item.get("doc_id") or item.get("filename") or "")
        if not key or key in seen_vector:
            continue
        seen_vector.add(key)
        row = fused.setdefault(key, {
            "document_id": item.get("document_id") or item.get("doc_id", ""),
            "filename": item.get("filename", ""), "highlights": [], "rrf": 0.0,
            "generation": item.get("generation") or item.get("ingestion_generation", ""),
        })
        row["rrf"] = float(row.get("rrf", 0.0)) + 1.0 / (rrf_k + rank)
        row.setdefault("vec_text", str(item.get("text", ""))[:500])
    return sorted(
        fused.values(),
        key=lambda item: (-float(item.get("rrf", 0.0)), str(item.get("filename", ""))),
    )[:max(1, int(top_k))]
