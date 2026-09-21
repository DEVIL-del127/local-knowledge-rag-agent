from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ManifestDocument:
    document_id: str
    filename: str
    source_hash: str
    parser: str
    parser_version: str
    quality: dict[str, Any]
    chunk_count: int
    status: str
    diagnostics: list[str] = field(default_factory=list)
    relative_locator: str = ""
    metadata_digest: str = ""
    chunk_set_digest: str = ""
    bibliographic_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionManifest:
    generation_id: str
    embedding_model: str
    embedding_dimension: int
    documents: list[ManifestDocument] = field(default_factory=list)
    status: str = "building"
    manifest_schema_version: str = "ingestion-manifest-v2"
    created_at_utc: str = ""
    prepared_at_utc: str = ""
    parser_name: str = ""
    parser_version: str = ""
    parser_config_digest: str = ""
    metadata_extractor_version: str = ""
    metadata_extractor_config_digest: str = ""
    chunker_name: str = "structural"
    chunker_version: str = "structural-v2"
    chunker_config_digest: str = ""
    embedding_provider: str = "ollama"
    embedding_model_revision: str = ""
    embedding_normalization: str = "unknown"
    vector_distance_metric: str = "cosine"
    es_physical_index: str = ""
    es_index_uuid: str = ""
    es_mapping_digest: str = ""
    vector_collection: str = ""
    vector_collection_id: str = ""
    catalog_version: str = ""
    catalog_digest: str = ""
    accepted_document_set_digest: str = ""
    chunk_set_digest: str = ""
    source_inventory_root_identity: str = ""
    discovered_file_count: int = 0
    accepted_document_count: int = 0
    quarantined_document_count: int = 0
    es_cluster_id: str = ""
    es_settings_digest: str = ""
    es_document_count: int = 0
    es_document_set_digest: str = ""
    vector_backend: str = "chroma"
    vector_store_id: str = ""
    vector_metadata_digest: str = ""
    vector_document_count: int = 0
    vector_chunk_count: int = 0
    vector_document_set_digest: str = ""
    catalog_schema_version: str = "catalog-contract-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return canonical_manifest_bytes(self.to_dict())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.canonical_bytes()
        fd, temporary = tempfile.mkstemp(prefix=target.name, dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write_content_addressed(self, root: str | Path) -> "ManifestRef":
        payload = self.canonical_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        locator = f"sha256/{digest[:2]}/{digest}.json"
        target = resolve_manifest_locator(root, locator, expected_digest=digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("content-addressed manifest collision")
        else:
            fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                _fsync_directory(target.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ManifestRef(
            locator=locator,
            digest=digest,
            byte_length=len(payload),
            store_id=manifest_store_identity(root),
            schema_version=self.manifest_schema_version,
        )


@dataclass(frozen=True, slots=True)
class ManifestRef:
    locator: str
    digest: str
    byte_length: int
    store_id: str = "local-manifest-store-v1"
    locator_scheme: str = "store-relative-v1"
    digest_algorithm: str = "sha256"
    schema_version: str = "ingestion-manifest-v2"


def canonical_manifest_bytes(payload: dict[str, Any]) -> bytes:
    """Return the one byte representation used for write and digest."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def manifest_store_identity(root: str | Path) -> str:
    # A manifest locator is store-relative and its payload is independently
    # protected by SHA-256.  Binding the store ID to an absolute path made the
    # same store acquire different identities on Windows and WSL.
    return "local-manifest-store-v1"


def vector_store_identity() -> str:
    """Portable identity for the local Chroma store; collection/content IDs
    provide the physical integrity checks without embedding an OS path."""
    return "local-chroma-store-v1"


def is_legacy_vector_store_identity(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _fsync_directory(directory: Path) -> None:
    """Persist a completed rename on filesystems that support directory fsync."""
    descriptor = None
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)


def resolve_manifest_locator(
    root: str | Path, locator: str, *, expected_digest: str = ""
) -> Path:
    normalized = str(locator).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) != 3 or parts[0] != "sha256" or not parts[1] or not parts[2].endswith(".json"):
        raise ValueError("invalid manifest locator")
    digest = parts[2][:-5]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("invalid manifest digest in locator")
    if parts[1] != digest[:2] or (expected_digest and digest != expected_digest.lower()):
        raise ValueError("manifest locator/digest mismatch")
    base = Path(root).resolve()
    target = (base / Path(*parts)).resolve()
    if base != target and base not in target.parents:
        raise ValueError("manifest locator escapes store root")
    return target


def verify_manifest_ref(root: str | Path, reference: ManifestRef) -> dict[str, Any]:
    if reference.locator_scheme != "store-relative-v1" or reference.digest_algorithm != "sha256":
        raise ValueError("unsupported manifest reference")
    expected_store = manifest_store_identity(root)
    legacy_prefix = f"{expected_store}:"
    legacy_digest = reference.store_id.removeprefix(legacy_prefix)
    legacy_path_identity = (
        reference.store_id.startswith(legacy_prefix)
        and len(legacy_digest) == 64
        and all(ch in "0123456789abcdef" for ch in legacy_digest)
    )
    if reference.store_id != expected_store and not legacy_path_identity:
        raise ValueError("manifest store identity mismatch")
    target = resolve_manifest_locator(root, reference.locator, expected_digest=reference.digest)
    payload = target.read_bytes()
    if len(payload) != reference.byte_length:
        raise ValueError("manifest byte length mismatch")
    if hashlib.sha256(payload).hexdigest() != reference.digest:
        raise ValueError("manifest digest mismatch")
    decoded = json.loads(payload.decode("utf-8"))
    if canonical_manifest_bytes(decoded) != payload:
        raise ValueError("manifest is not canonical")
    if decoded.get("manifest_schema_version") != reference.schema_version:
        raise ValueError("manifest schema mismatch")
    return decoded
