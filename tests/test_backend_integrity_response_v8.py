import sqlite3
from types import SimpleNamespace

from core.backend_integrity import BackendIntegrityMonitor


class Response:
    def __init__(self, body):
        self.body = body

    def get(self, *args, **kwargs):
        return self.body.get(*args, **kwargs)


class Indices:
    def stats(self, **kwargs):
        return Response({"_shards": {"failed": 0}, "indices": {"idx": {"shards": {
            "0": [{"seq_no": {"max_seq_no": 1, "local_checkpoint": 1},
                   "routing": {"primary": True}}]
        }}}})

    def get_mapping(self, **kwargs):
        return Response({"idx": {"mappings": {"properties": {}}}})

    def get_settings(self, **kwargs):
        return Response({"idx": {"settings": {"index": {"uuid": "u"}}}})


def test_integrity_token_accepts_elasticsearch_object_response(tmp_path):
    sqlite3.connect(tmp_path / "chroma.sqlite3").close()
    backend = SimpleNamespace(
        vector_store=SimpleNamespace(persist_dir=str(tmp_path)),
        es_manager=SimpleNamespace(es=SimpleNamespace(indices=Indices())),
    )
    monitor = BackendIntegrityMonitor(backend)
    try:
        assert len(monitor.token("idx")) == 64
    finally:
        monitor.close()
