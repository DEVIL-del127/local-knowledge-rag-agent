from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationState
from ingestion.pipeline.manifest import IngestionManifest, ManifestDocument, ManifestRef, verify_manifest_ref
from main import PDFSearchSystem


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_id", nargs="?")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = GenerationRegistry(root / "data/ingestion/generation_registry.sqlite3")
    record = registry.get(args.generation_id) if args.generation_id else next(
        (item for item in reversed(registry.list_all()) if item.state == GenerationState.PREPARED), None
    )
    if record is None or record.state not in {GenerationState.PREPARED, GenerationState.RETIRED}:
        raise SystemExit("no PREPARED or RETIRED generation")
    payload = verify_manifest_ref(registry.manifest_root, ManifestRef(
        locator=record.manifest_locator, digest=record.manifest_hash,
        byte_length=record.manifest_byte_length,
        store_id=record.manifest_store_id,
        locator_scheme=record.manifest_locator_scheme,
        digest_algorithm=record.manifest_digest_algorithm,
        schema_version=record.manifest_schema_version or "ingestion-manifest-v2",
    ))
    system = PDFSearchSystem(db="main")
    mapping = system.es_manager.es.indices.get_mapping(index=record.es_physical_index)
    settings = system.es_manager.es.indices.get_settings(index=record.es_physical_index)
    collection = system.vector_store.client.get_collection(record.vector_collection)
    es_records = system.es_manager.es.search(index=record.es_physical_index, body={
        "query": {"match_all": {}}, "size": 10000,
        "_source": ["document_id", "content_hash", "ingestion_generation"],
    }).get("hits", {}).get("hits", [])
    vector_payload = collection.get(include=["metadatas"])
    vector_metadata = [dict(item or {}) for item in vector_payload.get("metadatas") or []]
    es_docs = sorted({(
        str((hit.get("_source") or {}).get("document_id", "")),
        str((hit.get("_source") or {}).get("content_hash", "")),
        str((hit.get("_source") or {}).get("ingestion_generation", "")),
    ) for hit in es_records})
    vector_docs = sorted({(
        str(item.get("document_id", "")), str(item.get("content_hash", "")),
        str(item.get("ingestion_generation", "")),
    ) for item in vector_metadata})
    set_digest = lambda rows: hashlib.sha256(
        "\n".join(":".join(row) for row in rows).encode("utf-8")
    ).hexdigest()
    index_meta = settings[record.es_physical_index]["settings"]["index"]
    payload.update({
        "metadata_extractor_version": "bibliographic-v2",
        "metadata_extractor_config_digest": hashlib.sha256(
            b"bibliographic-v2:pdf-metadata+frontmatter+filename"
        ).hexdigest(),
        "embedding_provider": "ollama",
        "embedding_model_revision": record.embedding_model,
        "embedding_normalization": "provider-default",
        "vector_distance_metric": "cosine",
        "chunker_name": "structural", "chunker_version": "structural-v2",
        "es_index_uuid": str(index_meta.get("uuid", "")),
        "es_mapping_digest": digest(mapping),
        "es_cluster_id": str(system.es_manager.es.info().get("cluster_uuid", "")),
        "es_settings_digest": digest(settings),
        "es_document_count": len(es_docs),
        "es_document_set_digest": set_digest(es_docs),
        "vector_backend": "chroma",
        "vector_store_id": digest({
            "client": type(system.vector_store.client).__qualname__,
            "persist_dir": str(system.vector_store.persist_dir),
        }),
        "vector_collection_id": str(collection.id),
        "vector_metadata_digest": digest(collection.metadata or {}),
        "vector_document_count": len(vector_docs),
        "vector_chunk_count": len(vector_metadata),
        "vector_document_set_digest": set_digest(vector_docs),
        "catalog_version": "catalog-contract-v1",
        "catalog_schema_version": "catalog-contract-v1",
        "catalog_digest": digest({"schema": "catalog-contract-v1", "mapping": mapping}),
        "discovered_file_count": len(payload.get("documents", [])),
        "accepted_document_count": sum(
            item.get("status") == "accepted" for item in payload.get("documents", [])
        ),
        "quarantined_document_count": sum(
            item.get("status") != "accepted" for item in payload.get("documents", [])
        ),
        "source_inventory_root_identity": digest(str(Path(system.pdf_dir).resolve())),
    })
    manifest = IngestionManifest(**{
        **payload,
        "documents": [ManifestDocument(**item) for item in payload.get("documents", [])],
    })
    reference = manifest.write_content_addressed(registry.manifest_root)
    record.embedding_provider = payload["embedding_provider"]
    record.embedding_model_revision = payload["embedding_model_revision"]
    record.catalog_version = payload["catalog_version"]
    record.catalog_digest = payload["catalog_digest"]
    registry.put(record)
    if record.state == GenerationState.PREPARED:
        updated = registry.prepare(
            record.generation_id, manifest_hash=reference.digest,
            manifest_locator=reference.locator, manifest_byte_length=reference.byte_length,
            manifest_schema_version=reference.schema_version,
        )
    else:
        record.manifest_hash = reference.digest
        record.manifest_locator = reference.locator
        record.manifest_byte_length = reference.byte_length
        record.manifest_schema_version = reference.schema_version
        registry.put(record)
        updated = registry.get(record.generation_id)
    print(json.dumps({
        "generation_id": updated.generation_id, "record_revision": updated.record_revision,
        "manifest_locator": updated.manifest_locator, "manifest_digest": updated.manifest_hash,
        "manifest_byte_length": updated.manifest_byte_length,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
