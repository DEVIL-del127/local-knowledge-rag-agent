from __future__ import annotations

from typing import Any

from elasticsearch import helpers


class ElasticsearchStagingSink:
    def __init__(self, es_client: Any, physical_index: str) -> None:
        self.es = es_client
        self.physical_index = physical_index

    def create(self) -> None:
        if self.es.indices.exists(index=self.physical_index):
            raise RuntimeError(f"staging index already exists: {self.physical_index}")
        self.es.indices.create(index=self.physical_index, body={
            "mappings": {
                "_meta": {"ingestion_contract": "v2"},
                "properties": {
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "content": {"type": "text"},
                    "content_hash": {"type": "keyword"},
                    "page_count": {"type": "integer"},
                    "ingestion_generation": {"type": "keyword"},
                    "parser_version": {"type": "keyword"},
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "publication_year": {"type": "integer"},
                    "venue": {"type": "keyword"},
                    "doi": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "topic_aliases_text": {"type": "text"},
                    "section_type": {"type": "keyword"},
                    "section_title": {"type": "text"},
                    "section_order": {"type": "integer"},
                    "is_abstract": {"type": "boolean"},
                    "is_reference": {"type": "boolean"},
                },
            }
        })

    def write(self, documents: list[dict[str, Any]]) -> int:
        actions = [
            {
                "_index": self.physical_index,
                "_id": document["document_id"],
                "_source": document,
            }
            for document in documents
        ]
        success, _ = helpers.bulk(self.es, actions, stats_only=True)
        self.es.indices.refresh(index=self.physical_index)
        return int(success)

    def count(self) -> int:
        return int(self.es.count(index=self.physical_index).get("count", 0))

    def contract_records(self) -> list[dict[str, Any]]:
        response = self.es.search(
            index=self.physical_index,
            body={
                "query": {"match_all": {}},
                "_source": ["document_id", "content_hash", "ingestion_generation"],
                "size": 10000,
            },
        )
        return [dict(hit.get("_source") or {}) for hit in response["hits"]["hits"]]


class ChromaStagingSink:
    def __init__(
        self, client: Any, collection_name: str, embedder: Any,
        *, persist_dir: str = "",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        # Public, stable storage identity used by both manifest creation and
        # runtime verification. Chroma's private `_identifier` is not a
        # portable persistence contract and may differ across client instances.
        self.persist_dir = persist_dir
        self.collection = None

    def create(self) -> None:
        existing = {item.name for item in self.client.list_collections()}
        if self.collection_name in existing:
            raise RuntimeError(f"staging collection already exists: {self.collection_name}")
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "ingestion_contract": "v2"},
        )

    def write(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        if self.collection is None:
            raise RuntimeError("staging collection was not created")
        texts = [item["text"] for item in chunks]
        embeddings = self.embedder.embed(texts)
        valid = [(item, embedding) for item, embedding in zip(chunks, embeddings) if embedding is not None]
        if not valid:
            return 0
        self.collection.upsert(
            ids=[f"{item['document_id']}::{item['chunk_index']}" for item, _ in valid],
            embeddings=[embedding for _, embedding in valid],
            documents=[item["text"] for item, _ in valid],
            metadatas=[{
                key: value for key, value in item.items()
                if key not in {"text"} and isinstance(value, (str, int, float, bool))
            } for item, _ in valid],
        )
        return len(valid)

    def count(self) -> int:
        collection = self.collection or self.client.get_collection(self.collection_name)
        return int(collection.count())

    def contract_records(self) -> tuple[list[dict[str, Any]], int]:
        collection = self.collection or self.client.get_collection(self.collection_name)
        payload = collection.get(include=["metadatas", "embeddings"])
        metadata = [dict(item or {}) for item in payload.get("metadatas") or []]
        embeddings = payload.get("embeddings")
        first = embeddings[0] if embeddings is not None and len(embeddings) else []
        return metadata, len(first)
