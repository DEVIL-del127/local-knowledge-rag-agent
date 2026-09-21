from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.pipeline.generation_registry import GenerationRegistry, GenerationState
from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref
from main import PDFSearchSystem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    record = registry.get(args.generation_id)
    if record is None or record.state not in {GenerationState.PREPARED, GenerationState.RETIRED}:
        raise SystemExit("generation is missing or not PREPARED/RETIRED")
    manifest = verify_manifest_ref(registry.manifest_root, ManifestRef(
        locator=record.manifest_locator, digest=record.manifest_hash,
        byte_length=record.manifest_byte_length,
        store_id=record.manifest_store_id,
        locator_scheme=record.manifest_locator_scheme,
        digest_algorithm=record.manifest_digest_algorithm,
        schema_version=record.manifest_schema_version or "ingestion-manifest-v2"))
    system = PDFSearchSystem(db="main")
    es_count = int(system.es_manager.es.count(index=record.es_physical_index)["count"])
    collection = system.vector_store.client.get_collection(record.vector_collection)
    vector_store_id = hashlib.sha256(json.dumps({
        "client": type(system.vector_store.client).__qualname__,
        "persist_dir": str(system.vector_store.persist_dir),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )).hexdigest()
    payload = collection.get(include=["metadatas"])
    metadata = [dict(item or {}) for item in payload.get("metadatas") or []]
    paged = [item for item in metadata if isinstance(item.get("page_start"), int)
             and item["page_start"] > 0]
    invalid = [item for item in metadata if "page_start" in item and not (
        isinstance(item.get("page_start"), int) and item["page_start"] > 0)]
    report = {
        "generation_id": record.generation_id, "state": record.state.value,
        "manifest_locator": record.manifest_locator, "manifest_digest": record.manifest_hash,
        "accepted_documents": manifest.get("accepted_document_count"),
        "es_documents": es_count, "vector_chunks": len(metadata),
        "paged_chunks": len(paged),
        "page_coverage": len(paged) / len(metadata) if metadata else 0.0,
        "invalid_page_metadata": len(invalid),
        "vector_store_identity_matches": manifest.get("vector_store_id") == vector_store_id,
        "vector_collection_identity_matches": (
            manifest.get("vector_collection_id") == str(collection.id)
        ),
        "passed": bool(len(metadata) == record.chunk_count and es_count == record.document_count
                       and paged and not invalid
                       and manifest.get("vector_store_id") == vector_store_id
                       and manifest.get("vector_collection_id") == str(collection.id)),
    }
    target = (ROOT / args.output).resolve()
    if ROOT.resolve() not in target.parents: raise SystemExit("output outside workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
