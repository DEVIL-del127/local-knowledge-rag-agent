"""Read-only ACTIVE manifest versus live ES identity check."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from agent.app_settings import AppSettings
from core.contract_store import canonical_json
from ingestion.pipeline.generation_registry import GenerationRegistry
from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref

load_dotenv(ROOT / ".env")
s = AppSettings.from_env(ROOT)
r = GenerationRegistry(s.ingestion_registry_path, read_only=True)
record, revision = r.get_active()
ref = ManifestRef(locator=record.manifest_locator, digest=record.manifest_hash,
                  byte_length=record.manifest_byte_length,
                  schema_version=record.manifest_schema_version,
                  store_id=record.manifest_store_id,
                  locator_scheme=record.manifest_locator_scheme,
                  digest_algorithm=record.manifest_digest_algorithm)
manifest = verify_manifest_ref(r.manifest_root, ref)
client = Elasticsearch(f"http://{s.es_host}:{s.es_port}")
try:
    mapping_response = client.indices.get_mapping(index=record.es_physical_index)
    mapping = getattr(mapping_response, "body", mapping_response)
    settings_response = client.indices.get_settings(index=record.es_physical_index)
    settings = getattr(settings_response, "body", settings_response)
    info_response = client.info()
    info = getattr(info_response, "body", info_response)
    actual_mapping = hashlib.sha256(canonical_json(mapping)).hexdigest()
    legacy_mapping = hashlib.sha256(json.dumps(
        mapping_response, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    actual_uuid = settings[record.es_physical_index]["settings"]["index"]["uuid"]
    result = {
        "generation": record.generation_id, "revision": revision,
        "cluster_match": manifest.get("es_cluster_id") == info.get("cluster_uuid"),
        "index_uuid_match": manifest.get("es_index_uuid") == actual_uuid,
        "mapping_digest_match": manifest.get("es_mapping_digest") == actual_mapping,
        "manifest_cluster_id": manifest.get("es_cluster_id"),
        "live_cluster_id": info.get("cluster_uuid"),
        "manifest_index_uuid": manifest.get("es_index_uuid"),
        "live_index_uuid": actual_uuid,
        "manifest_mapping_digest": manifest.get("es_mapping_digest"),
        "live_mapping_digest": actual_mapping,
        "legacy_live_mapping_digest": legacy_mapping,
        "legacy_mapping_digest_match": manifest.get("es_mapping_digest") == legacy_mapping,
    }
    print(json.dumps(result, sort_keys=True))
finally:
    client.close()
