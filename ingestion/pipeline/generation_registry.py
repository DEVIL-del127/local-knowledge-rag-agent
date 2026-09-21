from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from .catalog_attestation import CatalogAttestationMixin


class GenerationState(str, Enum):
    BUILDING = "building"
    VALIDATED = "validated"
    PREPARED = "prepared"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"
    CORRUPT = "corrupt"


@dataclass(slots=True)
class GenerationRecord:
    generation_id: str
    state: GenerationState
    es_physical_index: str
    vector_collection: str
    manifest_hash: str = ""
    parser_version: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    document_count: int = 0
    chunk_count: int = 0
    record_revision: int = 0
    manifest_locator: str = ""
    manifest_byte_length: int = 0
    manifest_schema_version: str = ""
    manifest_store_id: str = "local-manifest-store-v1"
    manifest_locator_scheme: str = "store-relative-v1"
    manifest_digest_algorithm: str = "sha256"
    embedding_provider: str = ""
    embedding_model_revision: str = ""
    catalog_version: str = ""
    catalog_digest: str = ""
    document_set_digest: str = ""
    chunk_set_digest: str = ""
    activated_at_utc: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "GenerationRecord":
        value = dict(payload)
        value["state"] = GenerationState(value["state"])
        return cls(**value)


class GenerationConflict(RuntimeError):
    pass


class GenerationRegistry(CatalogAttestationMixin):
    """Single source of truth for ES/vector generation activation."""

    def __init__(self, path: str | Path, *, manifest_root: str | Path | None = None, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.path.parent / "manifests"
        self._lock = threading.RLock()
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError("generation registry unavailable; runtime cannot create it")
            with self._connect() as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"generations", "registry_meta"}.issubset(tables):
                    raise GenerationConflict("generation registry requires explicit maintenance")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._initialize_catalog_attestations(connection)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS generations (generation_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS generation_record_revisions ("
                "generation_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY(generation_id,revision))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS registry_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), active_generation TEXT, revision INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO registry_meta(singleton,active_generation,revision) VALUES(1,NULL,0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS build_journal ("
                "generation_id TEXT PRIMARY KEY, owner TEXT NOT NULL, stage TEXT NOT NULL, "
                "status TEXT NOT NULL, heartbeat REAL NOT NULL, lease_expires REAL NOT NULL, "
                "resources_json TEXT NOT NULL, compensations_json TEXT NOT NULL)"
            )

    def start_build(self, generation_id: str, *, lease_seconds: float = 300.0) -> str:
        owner = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO build_journal VALUES(?,?,?,?,?,?,?,?)",
                (generation_id, owner, "created", "running", now,
                 now + float(lease_seconds), "[]", "[]"),
            )
            connection.commit()
        return owner

    def update_build(
        self, generation_id: str, owner: str, *, stage: str,
        resources: list[str] | None = None,
        compensations: list[str] | None = None,
        status: str = "running", lease_seconds: float = 300.0,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner FROM build_journal WHERE generation_id=?", (generation_id,)
            ).fetchone()
            if row is None or row[0] != owner:
                connection.rollback()
                raise GenerationConflict("build journal owner changed")
            connection.execute(
                "UPDATE build_journal SET stage=?,status=?,heartbeat=?,lease_expires=?,"
                "resources_json=?,compensations_json=? WHERE generation_id=? AND owner=?",
                (stage, status, now, now + float(lease_seconds),
                 json.dumps(resources or [], sort_keys=True),
                 json.dumps(compensations or [], sort_keys=True), generation_id, owner),
            )
            connection.commit()

    def get_build(self, generation_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT generation_id,owner,stage,status,heartbeat,lease_expires,"
                "resources_json,compensations_json FROM build_journal WHERE generation_id=?",
                (generation_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "generation_id": row[0], "owner": row[1], "stage": row[2],
            "status": row[3], "heartbeat": float(row[4]),
            "lease_expires": float(row[5]), "resources": json.loads(row[6]),
            "compensations": json.loads(row[7]),
        }

    def recover_stale_builds(self, *, now: float | None = None) -> list[str]:
        """Mark expired journals for manual recovery; never delete resources."""
        cutoff = time.time() if now is None else float(now)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT generation_id FROM build_journal "
                "WHERE status='running' AND lease_expires<?", (cutoff,)
            ).fetchall()
            ids = [str(row[0]) for row in rows]
            if ids:
                connection.executemany(
                    "UPDATE build_journal SET status='needs_manual_recovery',stage='lease_expired' "
                    "WHERE generation_id=?", [(item,) for item in ids],
                )
            connection.commit()
        return ids

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro" if self.read_only else str(self.path),
            timeout=2.0, isolation_level=None, uri=self.read_only,
        )
        try:
            if self.read_only:
                connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()

    def put(self, record: GenerationRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload FROM generations WHERE generation_id=?", (record.generation_id,)
            ).fetchone()
            if existing:
                previous = GenerationRecord.from_dict(json.loads(existing[0]))
                self._archive_record(connection, previous)
                if record.record_revision != previous.record_revision:
                    connection.rollback()
                    raise GenerationConflict(
                        "generation record revision changed: "
                        f"expected={record.record_revision}, actual={previous.record_revision}"
                    )
                allowed = {
                    GenerationState.BUILDING: {GenerationState.BUILDING, GenerationState.VALIDATED, GenerationState.FAILED},
                    GenerationState.VALIDATED: {GenerationState.VALIDATED, GenerationState.PREPARED, GenerationState.FAILED},
                    GenerationState.PREPARED: {GenerationState.PREPARED, GenerationState.ACTIVE, GenerationState.FAILED, GenerationState.CORRUPT},
                    GenerationState.ACTIVE: {GenerationState.RETIRED, GenerationState.CORRUPT},
                    GenerationState.RETIRED: {GenerationState.RETIRED, GenerationState.ACTIVE, GenerationState.CORRUPT},
                    GenerationState.FAILED: {GenerationState.FAILED},
                    GenerationState.CORRUPT: {GenerationState.CORRUPT},
                }
                if record.state not in allowed[previous.state]:
                    connection.rollback()
                    raise GenerationConflict(
                        f"invalid generation transition: {previous.state.value}->{record.state.value}"
                    )
                record.record_revision = previous.record_revision + 1
            else:
                record.record_revision = max(1, record.record_revision)
            connection.execute(
                "INSERT INTO generations(generation_id,payload) VALUES(?,?) "
                "ON CONFLICT(generation_id) DO UPDATE SET payload=excluded.payload",
                (record.generation_id, json.dumps(record.to_dict(), sort_keys=True)),
            )
            self._archive_record(connection, record)
            connection.commit()

    @staticmethod
    def _archive_record(connection, record: GenerationRecord) -> str:
        payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        existing = connection.execute(
            "SELECT digest,payload FROM generation_record_revisions WHERE generation_id=? AND revision=?",
            (record.generation_id, record.record_revision),
        ).fetchone()
        if existing and existing != (digest, payload):
            raise GenerationConflict("immutable generation history conflict")
        connection.execute("INSERT OR IGNORE INTO generation_record_revisions VALUES(?,?,?,?)",
                           (record.generation_id, record.record_revision, digest, payload))
        return digest

    def get_record_revision(self, generation_id: str, revision: int, *, expected_digest: str) -> GenerationRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT digest,payload FROM generation_record_revisions WHERE generation_id=? AND revision=?",
                (generation_id, revision),
            ).fetchone()
        if row is None:
            raise GenerationConflict("historical generation record missing")
        digest, payload = row
        if digest != expected_digest or hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
            raise GenerationConflict("historical generation record digest mismatch")
        record = GenerationRecord.from_dict(json.loads(payload))
        if record.generation_id != generation_id or record.record_revision != revision:
            raise GenerationConflict("historical generation record identity mismatch")
        return record

    def get(self, generation_id: str) -> GenerationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
        return GenerationRecord.from_dict(json.loads(row[0])) if row else None

    def list_all(self) -> list[GenerationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM generations ORDER BY generation_id"
            ).fetchall()
        return [GenerationRecord.from_dict(json.loads(row[0])) for row in rows]

    def get_active(self) -> tuple[GenerationRecord | None, int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT m.active_generation,m.revision,g.payload "
                "FROM registry_meta m LEFT JOIN generations g "
                "ON g.generation_id=m.active_generation WHERE m.singleton=1"
            ).fetchone()
        active, revision, payload = row
        return (GenerationRecord.from_dict(json.loads(payload)) if active and payload else None,
                int(revision))

    def prepare(self, generation_id: str, *, manifest_hash: str,
                manifest_locator: str = "", manifest_byte_length: int = 0,
                manifest_schema_version: str = "ingestion-manifest-v2",
                manifest_store_id: str = "local-manifest-store-v1",
                manifest_locator_scheme: str = "store-relative-v1",
                manifest_digest_algorithm: str = "sha256") -> GenerationRecord:
        record = self.get(generation_id)
        if record is None or record.state not in {GenerationState.VALIDATED, GenerationState.PREPARED}:
            raise GenerationConflict("generation must be validated before prepare")
        record.state = GenerationState.PREPARED
        record.manifest_hash = manifest_hash
        record.manifest_locator = manifest_locator
        record.manifest_byte_length = int(manifest_byte_length)
        record.manifest_schema_version = manifest_schema_version
        record.manifest_store_id = manifest_store_id
        record.manifest_locator_scheme = manifest_locator_scheme
        record.manifest_digest_algorithm = manifest_digest_algorithm
        self.put(record)
        return record

    def activate(self, generation_id: str, *, expected_active: str | None, expected_revision: int,
                 expected_record_revision: int | None = None) -> int:
        return self._switch(generation_id, expected_active, expected_revision, expected_record_revision)

    def rollback(self, generation_id: str, *, expected_active: str | None, expected_revision: int,
                 expected_record_revision: int | None = None) -> int:
        return self._switch(generation_id, expected_active, expected_revision, expected_record_revision)

    def _switch(self, target: str, expected_active: str | None, expected_revision: int,
                expected_record_revision: int | None = None) -> int:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT active_generation,revision FROM registry_meta WHERE singleton=1"
            ).fetchone()
            if row[0] == target:
                connection.rollback()
                return int(row[1])
            if row != (expected_active, expected_revision):
                connection.rollback()
                raise GenerationConflict("active generation changed")
            target_row = connection.execute(
                "SELECT payload FROM generations WHERE generation_id=?", (target,)
            ).fetchone()
            if target_row is None:
                connection.rollback()
                raise GenerationConflict("target generation does not exist")
            target_record = GenerationRecord.from_dict(json.loads(target_row[0]))
            self._archive_record(connection, target_record)
            if (expected_record_revision is not None
                    and target_record.record_revision != expected_record_revision):
                connection.rollback()
                raise GenerationConflict("target generation record changed")
            if target_record.state not in {GenerationState.PREPARED, GenerationState.RETIRED, GenerationState.ACTIVE}:
                connection.rollback()
                raise GenerationConflict("target generation is not activatable")
            if expected_active and expected_active != target:
                old_row = connection.execute(
                    "SELECT payload FROM generations WHERE generation_id=?", (expected_active,)
                ).fetchone()
                if old_row:
                    old = GenerationRecord.from_dict(json.loads(old_row[0]))
                    self._archive_record(connection, old)
                    old.state = GenerationState.RETIRED
                    old.record_revision += 1
                    connection.execute(
                        "UPDATE generations SET payload=? WHERE generation_id=?",
                        (json.dumps(old.to_dict(), sort_keys=True), expected_active),
                    )
                    self._archive_record(connection, old)
            target_record.state = GenerationState.ACTIVE
            target_record.record_revision += 1
            target_record.activated_at_utc = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "UPDATE generations SET payload=? WHERE generation_id=?",
                (json.dumps(target_record.to_dict(), sort_keys=True), target),
            )
            self._archive_record(connection, target_record)
            revision = expected_revision + 1
            connection.execute(
                "UPDATE registry_meta SET active_generation=?,revision=? WHERE singleton=1",
                (target, revision),
            )
            connection.commit()
            return revision
