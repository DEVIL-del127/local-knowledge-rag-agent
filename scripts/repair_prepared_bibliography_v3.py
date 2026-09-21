from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.pipeline.coordinator import _canonical_digest, _extract_bibliographic
from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationState
from ingestion.pipeline.manifest import IngestionManifest, ManifestDocument, ManifestRef, verify_manifest_ref
from main import PDFSearchSystem


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair bibliography only on an immutable PREPARED generation")
    parser.add_argument("generation_id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = GenerationRegistry(root / "data/ingestion/generation_registry.sqlite3")
    record = registry.get(args.generation_id)
    if record is None or record.state not in {GenerationState.PREPARED, GenerationState.RETIRED}:
        raise SystemExit("target must be PREPARED or RETIRED")
    payload = verify_manifest_ref(registry.manifest_root, ManifestRef(
        locator=record.manifest_locator, digest=record.manifest_hash,
        byte_length=record.manifest_byte_length,
        store_id=record.manifest_store_id,
        locator_scheme=record.manifest_locator_scheme,
        digest_algorithm=record.manifest_digest_algorithm,
        schema_version=record.manifest_schema_version or "ingestion-manifest-v2",
    ))
    system = PDFSearchSystem(db="main")
    response = system.es_manager.es.search(index=record.es_physical_index, body={
        "query": {"match_all": {}}, "size": 10000,
        "_source": ["document_id", "filename", "content"],
    })
    repaired: dict[str, dict] = {}
    filenames: dict[str, str] = {}
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        document_id = str(source.get("document_id", ""))
        filenames[document_id] = str(source.get("filename", ""))
        bibliography = _extract_bibliographic(
            Path(str(source.get("filename", ""))), str(source.get("content", ""))
        )
        repaired[document_id] = bibliography
        system.es_manager.es.update(
            index=record.es_physical_index, id=hit.get("_id"), doc=bibliography,
        )
    system.es_manager.es.indices.refresh(index=record.es_physical_index)

    collection = system.vector_store.client.get_collection(record.vector_collection)
    vector_payload = collection.get(include=["metadatas"])
    ids = list(vector_payload.get("ids") or [])
    metadata = [dict(item or {}) for item in vector_payload.get("metadatas") or []]
    updated_metadata = []
    for item in metadata:
        merged = dict(item)
        document_id = str(item.get("document_id", ""))
        merged.update(repaired.get(document_id, {}))
        merged["filename"] = filenames.get(document_id, str(item.get("filename", "")))
        updated_metadata.append(merged)
    for start in range(0, len(ids), 256):
        collection.update(ids=ids[start:start + 256], metadatas=updated_metadata[start:start + 256])

    documents = []
    for raw in payload.get("documents", []):
        item = dict(raw)
        bibliography = repaired.get(str(item.get("document_id", "")), item.get("bibliographic_metadata") or {})
        item["bibliographic_metadata"] = bibliography
        item["metadata_digest"] = _canonical_digest(bibliography)
        documents.append(ManifestDocument(**item))
    payload["documents"] = documents
    payload["metadata_extractor_version"] = "bibliographic-v3"
    payload["metadata_extractor_config_digest"] = _canonical_digest({
        "version": "bibliographic-v3",
        "sources": ["filename", "frontmatter"],
        "generic_heading_rejection": True,
        "bounded_year": [1900, 2026],
    })
    manifest = IngestionManifest(**payload)
    reference = manifest.write_content_addressed(registry.manifest_root)
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
        "generation_id": updated.generation_id,
        "repaired_documents": len(repaired),
        "manifest_locator": updated.manifest_locator,
        "manifest_digest": updated.manifest_hash,
        "manifest_byte_length": updated.manifest_byte_length,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
