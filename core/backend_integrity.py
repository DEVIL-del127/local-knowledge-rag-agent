"""Live change tokens for an existing local Chroma store and Elasticsearch index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading


class IntegrityCapabilityMissing(RuntimeError):
    pass


class BackendIntegrityMonitor:
    def __init__(self, backend):
        self.backend = backend
        self._lock = threading.RLock()
        self._connection = None
        self._identity = None

    def token(self, index):
        with self._lock:
            root = Path(self.backend.vector_store.persist_dir).resolve()
            database = root / "chroma.sqlite3"
            try:
                stat = database.stat()
                identity = (str(root), stat.st_dev, stat.st_ino)
                if self._connection is None:
                    self._connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True,
                                                       check_same_thread=False, isolation_level=None)
                    self._identity = identity
                if identity != self._identity:
                    raise IntegrityCapabilityMissing("vector database was replaced")
                version = self._connection.execute("PRAGMA data_version").fetchone()[0]
                client = self.backend.es_manager.es
                stats = client.indices.stats(index=index, level="shards")
                if int(stats.get("_shards", {}).get("failed", 0)):
                    raise IntegrityCapabilityMissing("incomplete Elasticsearch shard tokens")
                indices = stats.get("indices", {})
                if index not in indices or not indices[index].get("shards"):
                    raise IntegrityCapabilityMissing("Elasticsearch shard token unavailable")
                shards = indices[index]["shards"]
                if any("seq_no" not in replica for replicas in shards.values() for replica in replicas):
                    raise IntegrityCapabilityMissing("Elasticsearch sequence token unavailable")
                token = {"vector": [*identity, version], "es": shards,
                         "mapping": self._response_body(client.indices.get_mapping(index=index)),
                         "settings": self._response_body(client.indices.get_settings(index=index))}
                # Stats contain volatile counters: retain only sequence and routing identity.
                token["es"] = {name: sorted((
                    {"seq_no": replica["seq_no"], "routing": replica.get("routing", {})}
                    for replica in replicas
                ), key=lambda item: json.dumps(item, sort_keys=True)) for name, replicas in shards.items()}
                return hashlib.sha256(json.dumps(token, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            except (OSError, sqlite3.Error, AttributeError, KeyError, TypeError) as exc:
                raise IntegrityCapabilityMissing("backend_integrity_capability_missing") from exc

    @staticmethod
    def _response_body(response):
        return getattr(response, "body", response)

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._identity = None
