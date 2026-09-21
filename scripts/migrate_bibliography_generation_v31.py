from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from elasticsearch import helpers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval_gateway import RetrievalGateway
from ingestion.pipeline.coordinator import (
    BIBLIOGRAPHIC_METADATA_VERSION,
    _bibliographic_metadata_config_digest,
    _canonical_digest,
    _extract_bibliographic,
)
from ingestion.pipeline.generation_registry import (
    GenerationRecord, GenerationRegistry, GenerationState,
)
from ingestion.pipeline.manifest import IngestionManifest, ManifestDocument, ManifestRef, verify_manifest_ref
from ingestion.pipeline.stage import ElasticsearchStagingSink
from main import PDFSearchSystem


def _set_digest(rows: list[tuple[str, str, str]]) -> str:
    return hashlib.sha256(
        "\n".join(":".join(row) for row in sorted(rows)).encode("utf-8")
    ).hexdigest()


def _chunk_digest(metadata: list[dict], documents: list[str]) -> str:
    rows = sorted(
        f"{item.get('document_id', '')}:{item.get('content_hash', '')}:"
        f"{item.get('chunk_index', '')}:"
        f"{hashlib.sha256(str(text or '').encode('utf-8')).hexdigest()}"
        for item, text in zip(metadata, documents)
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _generation_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    return f"g{stamp}-{uuid.uuid4().hex[:8]}"


def main() -> int:
    system = PDFSearchSystem(db="main")
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    active, registry_revision = registry.get_active()
    if active is None:
        raise SystemExit("no ACTIVE generation")

    # Refuse to clone an already-inconsistent source generation.
    gateway = RetrievalGateway(system, generation_registry=registry)
    source_snapshot = gateway.pin()
    if source_snapshot is None:
        raise SystemExit("ACTIVE generation cannot be pinned")
    gateway.verify_snapshot(source_snapshot)
    source_manifest = verify_manifest_ref(registry.manifest_root, ManifestRef(
        locator=active.manifest_locator, digest=active.manifest_hash,
        byte_length=active.manifest_byte_length,
        store_id=active.manifest_store_id,
        locator_scheme=active.manifest_locator_scheme,
        digest_algorithm=active.manifest_digest_algorithm,
        schema_version=active.manifest_schema_version or "ingestion-manifest-v2",
    ))

    generation_id = _generation_id()
    es_index = f"pdf_documents_staging_{generation_id}"
    vector_collection = f"pdf_chunks_staging_{generation_id}"
    record = GenerationRecord(
        generation_id=generation_id,
        state=GenerationState.BUILDING,
        es_physical_index=es_index,
        vector_collection=vector_collection,
        parser_version=active.parser_version,
        embedding_provider=active.embedding_provider,
        embedding_model=active.embedding_model,
        embedding_model_revision=active.embedding_model_revision,
        embedding_dimension=active.embedding_dimension,
    )
    registry.put(record)

    try:
        source_hits = list(helpers.scan(
            system.es_manager.es,
            index=active.es_physical_index,
            query={"query": {"match_all": {}}},
            preserve_order=False,
        ))
        bibliography: dict[str, dict] = {}
        es_sources: list[dict] = []
        for hit in source_hits:
            source = dict(hit.get("_source") or {})
            document_id = str(source.get("document_id", ""))
            metadata = _extract_bibliographic(
                Path(str(source.get("filename", ""))), str(source.get("content", "")),
            )
            bibliography[document_id] = metadata
            source.update(metadata)
            source["ingestion_generation"] = generation_id
            es_sources.append(source)

        es_sink = ElasticsearchStagingSink(system.es_manager.es, es_index)
        es_sink.create()
        success, _ = helpers.bulk(system.es_manager.es, [
            {"_index": es_index, "_id": item["document_id"], "_source": item}
            for item in es_sources
        ], stats_only=True)
        if int(success) != len(es_sources):
            raise RuntimeError("Elasticsearch clone count mismatch")
        system.es_manager.es.indices.refresh(index=es_index)
        # The settings digest is captured only after applying the write block.
        system.es_manager.es.indices.put_settings(
            index=es_index, settings={"index.blocks.write": True},
        )

        source_collection = system.vector_store.client.get_collection(active.vector_collection)
        source_vectors = source_collection.get(
            include=["metadatas", "documents", "embeddings"],
        )
        ids = list(source_vectors.get("ids") or [])
        metadata = [dict(item or {}) for item in source_vectors.get("metadatas") or []]
        documents = [str(item or "") for item in source_vectors.get("documents") or []]
        embeddings_raw = source_vectors.get("embeddings")
        embeddings = embeddings_raw.tolist() if hasattr(embeddings_raw, "tolist") else list(embeddings_raw or [])
        if not (len(ids) == len(metadata) == len(documents) == len(embeddings)):
            raise RuntimeError("source vector payload is incomplete")
        updated_metadata = []
        for item in metadata:
            merged = dict(item)
            merged.update(bibliography.get(str(item.get("document_id", "")), {}))
            merged["ingestion_generation"] = generation_id
            updated_metadata.append(merged)
        target_collection = system.vector_store.client.create_collection(
            vector_collection,
            metadata={
                "hnsw:space": "cosine",
                "ingestion_contract": "v2",
                "generation_id": generation_id,
                "embedding_model": active.embedding_model,
                "embedding_dimension": active.embedding_dimension,
            },
        )
        for start in range(0, len(ids), 128):
            stop = start + 128
            target_collection.upsert(
                ids=ids[start:stop], embeddings=embeddings[start:stop],
                documents=documents[start:stop], metadatas=updated_metadata[start:stop],
            )

        es_rows = sorted({(
            str(item.get("document_id", "")), str(item.get("content_hash", "")), generation_id,
        ) for item in es_sources})
        vector_rows = sorted({(
            str(item.get("document_id", "")), str(item.get("content_hash", "")), generation_id,
        ) for item in updated_metadata})
        accepted_digest = _set_digest(es_rows)
        if es_rows != vector_rows:
            raise RuntimeError("cloned ES/vector document sets differ")
        chunk_set_digest = _chunk_digest(updated_metadata, documents)
        if chunk_set_digest != str(source_manifest.get("chunk_set_digest", "")):
            raise RuntimeError("cloned vector chunk set differs from source manifest")

        mapping = system.es_manager.es.indices.get_mapping(index=es_index)
        settings = system.es_manager.es.indices.get_settings(index=es_index)
        index_meta = settings[es_index]["settings"]["index"]
        vector_store_id = _canonical_digest({
            "client": type(system.vector_store.client).__qualname__,
            "persist_dir": str(system.vector_store.persist_dir),
        })
        now = datetime.now(timezone.utc).isoformat()
        manifest_documents = []
        for raw in source_manifest.get("documents", []):
            item = dict(raw)
            item["bibliographic_metadata"] = bibliography.get(
                str(item.get("document_id", "")), dict(item.get("bibliographic_metadata") or {}),
            )
            item["metadata_digest"] = _canonical_digest(item["bibliographic_metadata"])
            manifest_documents.append(ManifestDocument(**item))
        accepted_documents = [item for item in manifest_documents if item.status == "accepted"]
        quarantined_documents = [item for item in manifest_documents if item.status != "accepted"]
        expected_rows = sorted({(
            item.document_id, item.source_hash, generation_id,
        ) for item in accepted_documents})
        if es_rows != expected_rows:
            raise RuntimeError("cloned ES set differs from accepted manifest documents")
        manifest = IngestionManifest(
            generation_id=generation_id,
            embedding_model=active.embedding_model,
            embedding_dimension=active.embedding_dimension,
            documents=manifest_documents,
            status="prepared",
            created_at_utc=now,
            prepared_at_utc=now,
            parser_name=str(source_manifest.get("parser_name", "")),
            parser_version=str(source_manifest.get("parser_version", "")),
            parser_config_digest=str(source_manifest.get("parser_config_digest", "")),
            metadata_extractor_version=BIBLIOGRAPHIC_METADATA_VERSION,
            metadata_extractor_config_digest=_bibliographic_metadata_config_digest(),
            chunker_name=str(source_manifest.get("chunker_name", "structural")),
            chunker_version=str(source_manifest.get("chunker_version", "structural-v2")),
            chunker_config_digest=str(source_manifest.get("chunker_config_digest", "")),
            embedding_provider=active.embedding_provider,
            embedding_model_revision=active.embedding_model_revision,
            embedding_normalization=str(source_manifest.get("embedding_normalization", "provider-default")),
            vector_distance_metric=str(source_manifest.get("vector_distance_metric", "cosine")),
            es_physical_index=es_index,
            es_index_uuid=str(index_meta.get("uuid", "")),
            es_mapping_digest=_canonical_digest(mapping),
            es_cluster_id=str(system.es_manager.es.info().get("cluster_uuid", "")),
            es_settings_digest=_canonical_digest(settings),
            vector_collection=vector_collection,
            vector_collection_id=str(target_collection.id),
            vector_backend="chroma",
            vector_store_id=vector_store_id,
            vector_metadata_digest=_canonical_digest(target_collection.metadata or {}),
            catalog_schema_version="catalog-contract-v1",
            catalog_version="catalog-contract-v1",
            catalog_digest=_canonical_digest({"schema": "catalog-contract-v1", "mapping": mapping}),
            accepted_document_set_digest=accepted_digest,
            es_document_set_digest=accepted_digest,
            vector_document_set_digest=accepted_digest,
            chunk_set_digest=chunk_set_digest,
            source_inventory_root_identity=_canonical_digest(str(Path(system.pdf_dir).resolve())),
            discovered_file_count=len(manifest_documents),
            accepted_document_count=len(accepted_documents),
            quarantined_document_count=len(quarantined_documents),
            es_document_count=len(es_rows),
            vector_document_count=len(vector_rows),
            vector_chunk_count=len(updated_metadata),
        )

        reference = manifest.write_content_addressed(registry.manifest_root)
        record = registry.get(generation_id)
        record.state = GenerationState.VALIDATED
        record.document_count = len(es_rows)
        record.chunk_count = len(updated_metadata)
        record.document_set_digest = accepted_digest
        record.chunk_set_digest = chunk_set_digest
        record.catalog_version = manifest.catalog_version
        record.catalog_digest = manifest.catalog_digest
        registry.put(record)
        record = registry.prepare(
            generation_id,
            manifest_hash=reference.digest,
            manifest_locator=reference.locator,
            manifest_byte_length=reference.byte_length,
            manifest_schema_version=reference.schema_version,
        )
        new_revision = registry.activate(
            generation_id,
            expected_active=active.generation_id,
            expected_revision=registry_revision,
            expected_record_revision=record.record_revision,
        )
        try:
            verified_gateway = RetrievalGateway(system, generation_registry=registry)
            verified = verified_gateway.pin()
            if verified is None:
                raise RuntimeError("activated generation cannot be pinned")
            verified_gateway.verify_snapshot(verified)
        except Exception:
            rollback_target = registry.get(active.generation_id)
            registry.rollback(
                active.generation_id,
                expected_active=generation_id,
                expected_revision=new_revision,
                expected_record_revision=rollback_target.record_revision,
            )
            raise
        print(json.dumps({
            "generation_id": verified.generation_id,
            "registry_revision": new_revision,
            "manifest_locator": verified.manifest.locator,
            "manifest_digest": verified.manifest.digest,
            "manifest_byte_length": verified.manifest.byte_length,
            "document_count": verified.accepted_document_count,
            "chunk_count": verified.chunk_count,
            "document_set_digest": verified.accepted_document_set_digest,
            "chunk_set_digest": verified.chunk_set_digest,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failed = registry.get(generation_id)
        if failed is not None and failed.state in {
            GenerationState.BUILDING, GenerationState.VALIDATED, GenerationState.PREPARED,
            GenerationState.ACTIVE, GenerationState.RETIRED,
        }:
            failed.state = (
                GenerationState.FAILED
                if failed.state in {
                    GenerationState.BUILDING, GenerationState.VALIDATED,
                    GenerationState.PREPARED,
                }
                else GenerationState.CORRUPT
            )
            failed.failure_reason = f"bibliography generation migration failed: {type(exc).__name__}: {exc}"
            registry.put(failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
