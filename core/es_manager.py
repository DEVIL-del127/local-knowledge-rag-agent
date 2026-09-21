# es_manager.py - 修复版 v3 (IK 自动探测 + cjk 降级)
# 修复点:
# 1. 索引/查询优先用 IK 分词器(ik_max_word 索引 + ik_smart 查询), 高亮干净、中文质量好
# 2. 自动探测: IK 插件未加载时, 自动降级用 ES 内置 cjk 分析器(中文bigram, 零依赖),
#    程序不再因 analyzer 缺失而崩溃
# 3. 高亮标签改为 ** (控制台友好), 不再输出 HTML
# 4. _safe_utf8 保持 errors='replace' 兜底, 永不抛 UnicodeDecodeError
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict
import logging
import re

from core.kb_status import (
    KnowledgeBaseStatus,
    SearchBackendError,
    SearchOutcome,
    SearchStatus,
)

logger = logging.getLogger(__name__)


class ESManager:
    def __init__(self, host='localhost', port=9200, index_name='pdf_documents'):
        self.es = Elasticsearch(f"http://{host}:{port}")
        self.index_name = index_name  # 正式库/测试库用不同索引名隔离

    def _analyzer_available(self, name: str) -> bool:
        """探测分析器是否可用(插件是否已加载)"""
        try:
            self.es.indices.analyze(body={"analyzer": name, "text": "测试中文test"})
            return True
        except Exception:
            return False

    def _pick_analyzers(self):
        """选择分析器: IK 优先, 不可用则降级 cjk(ES 内置)"""
        if self._analyzer_available('ik_max_word'):
            logger.info("IK 分词器可用 -> 使用 ik_max_word(索引) + ik_smart(查询)")
            return {
                'content_analyzer': 'ik_max_word',
                'content_search_analyzer': 'ik_smart',
                'filename_analyzer': 'ik_max_word',
            }
        logger.warning("IK 插件未加载 -> 降级使用内置 cjk 分析器(中文bigram, 效果次于IK, 但零依赖可跑)")
        return {
            'content_analyzer': 'cjk',
            'content_search_analyzer': 'cjk',
            'filename_analyzer': 'standard',
        }

    def create_index(self):
        """创建索引"""
        a = self._pick_analyzers()

        settings = {
            "settings": {
                "index": {
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "b": 0.75,
                            "k1": 1.2
                        }
                    }
                },
                "analysis": {
                    # 备用: 未装 IK 插件时的 ngram 分析器(高亮碎片多, 不推荐)
                    "analyzer": {
                        "chinese_ngram_analyzer": {
                            "type": "custom",
                            "tokenizer": "ngram_tokenizer",
                            "filter": ["lowercase"]
                        }
                    },
                    "tokenizer": {
                        "ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 1,
                            "max_gram": 2,
                            "token_chars": ["letter", "digit"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "filename": {
                        "type": "text",
                        "analyzer": a['filename_analyzer']
                    },
                    "content": {
                        "type": "text",
                        "analyzer": a['content_analyzer'],
                        "search_analyzer": a['content_search_analyzer'],
                        "similarity": "BM25"
                    },
                    "content_length": {
                        "type": "integer"
                    },
                    "page_count": {
                        "type": "integer"
                    },
                    "created_date": {
                        "type": "date"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            }
        }

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.info(f"删除已存在的索引: {self.index_name}")

        self.es.indices.create(index=self.index_name, body=settings)
        logger.info(f"创建索引成功: {self.index_name}")

    def bulk_index(self, documents: List[Dict]):
        # 安全处理文档内容
        safe_docs = []
        for doc in documents:
            safe_doc = {}
            for key, value in doc.items():
                if isinstance(value, str):
                    safe_doc[key] = self._safe_utf8(value)
                else:
                    safe_doc[key] = value
            safe_docs.append(safe_doc)

        actions = [
            {
                "_index": self.index_name,
                "_source": doc
            }
            for doc in safe_docs
        ]

        try:
            success, failed = helpers.bulk(self.es, actions, stats_only=True)
            logger.info(f"批量索引完成: {success}成功, {failed}失败")
            return success, failed
        except Exception as e:
            logger.error(f"批量索引失败: {e}")
            return 0, 0

    def search_bm25(self, query: str, size: int = 10) -> List[Dict]:
        """BM25检索 - 支持中文"""
        query = self._safe_utf8(query)

        query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()

        if not query:
            return []

        try:
            return self._search_bm25_strict(query, size=size)
        except Exception as exc:
            # Legacy callers expect a list. New runtime callers use
            # search_bm25_outcome() to retain the structured failure reason.
            logger.error("BM25检索失败: %s", exc)
            return []

    def inspect_status(
        self,
        *,
        vector_store=None,
        embedding_model: str = "",
        embedding_dimension: int | None = None,
        generation: str = "",
        index_name: str | None = None,
        vector_collection: str | None = None,
    ) -> KnowledgeBaseStatus:
        target_index = index_name or self.index_name
        try:
            self.es.info()
        except Exception as exc:
            return KnowledgeBaseStatus(
                source_id=self.index_name,
                provider_available=False,
                index_exists=False,
                health_reason=f"{type(exc).__name__}: {exc}",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                ingestion_generation=generation,
            )
        try:
            exists = bool(self.es.indices.exists(index=target_index))
        except Exception as exc:
            return KnowledgeBaseStatus(
                source_id=self.index_name,
                provider_available=True,
                index_exists=False,
                health_reason=f"index check failed: {type(exc).__name__}: {exc}",
            )
        document_count = 0
        reason = ""
        if exists:
            try:
                document_count = int(self.es.count(index=target_index).get("count", 0))
            except Exception as exc:
                reason = f"count failed: {type(exc).__name__}: {exc}"
        vector_count = 0
        vector_exists = vector_store is not None
        if vector_store is not None:
            try:
                vector_count = int(
                    vector_store.count_collection(vector_collection)
                    if vector_collection else vector_store.count()
                )
            except Exception as exc:
                vector_exists = False
                reason = reason or f"vector count failed: {type(exc).__name__}: {exc}"
        return KnowledgeBaseStatus(
            source_id=target_index,
            provider_available=True,
            index_exists=exists,
            document_count=document_count,
            vector_collection_exists=vector_exists,
            vector_chunk_count=vector_count,
            ingestion_generation=generation,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            health_reason=reason,
        )

    def find_document(
        self, identity: str, *, size: int = 10, index_name: str | None = None
    ) -> list[dict]:
        value = self._safe_utf8(identity).strip()
        if not value:
            return []
        stem = re.sub(r"\.pdf$", "", value, flags=re.I).strip()
        title = re.sub(r"[_＿][\u4e00-\u9fff]{2,4}$", "", stem).strip()
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"filename.keyword": value}},
                        {"term": {"filename.keyword": stem + ".pdf"}},
                        {"match_phrase": {"filename": value}},
                        {"match_phrase": {"title": {"query": title, "boost": 5}}},
                        {"match_phrase": {"metadata.title": {"query": title, "boost": 5}}},
                        {"term": {"_id": value}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": ["document_id", "filename", "title", "authors", "publication_year",
                        "page_count", "content_length", "ingestion_generation"],
            "size": max(1, min(int(size), 20)),
        }
        try:
            response = self.es.search(index=index_name or self.index_name, body=body)
        except Exception as exc:
            raise SearchBackendError(SearchStatus.FAILED, str(exc)) from exc
        return [
            {"id": hit.get("_id", ""), "document_id": hit.get("_id", ""),
             "generation": str((hit.get("_source") or {}).get("ingestion_generation", "")),
             **dict(hit.get("_source") or {})}
            for hit in response.get("hits", {}).get("hits", [])
        ]

    def search_bm25_outcome(self, query: str, *, size: int = 10, trace_id: str = "") -> SearchOutcome:
        status = self.inspect_status()
        if not status.provider_available:
            return SearchOutcome(
                SearchStatus.PROVIDER_UNAVAILABLE,
                diagnostics=[status.health_reason],
                trace_id=trace_id,
            )
        if not status.index_exists:
            return SearchOutcome(SearchStatus.INDEX_MISSING, trace_id=trace_id)
        if status.document_count == 0:
            return SearchOutcome(SearchStatus.EMPTY_SOURCE, trace_id=trace_id)
        try:
            evidence = self._search_bm25_strict(query, size=size)
        except Exception as exc:
            return SearchOutcome(
                SearchStatus.FAILED,
                diagnostics=[f"{type(exc).__name__}: {exc}"],
                trace_id=trace_id,
            )
        return SearchOutcome(
            SearchStatus.MATCHED if evidence else SearchStatus.NO_MATCH,
            evidence=evidence,
            executed_channels=["es"],
            trace_id=trace_id,
        )

    def _search_bm25_strict(
        self, query: str, *, size: int, index_name: str | None = None,
        temporal_range: dict[str, int] | None = None,
        document_first: bool = False,
        language: str = "", document_type: str = "",
        document_ids: list[str] | None = None,
        section_types: list[str] | None = None,
        exclude_references: bool = True,
    ) -> list[dict]:
        cleaned = self._safe_utf8(query)
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            return []
        must_query = ({
            "multi_match": {
                "query": cleaned,
                "fields": ["title^8", "metadata.title^8", "topic_aliases_text^7", "keywords^5", "metadata.keywords^5",
                           "abstract^3", "metadata.abstract^3", "content"],
                "type": "best_fields", "operator": "or",
            }
        } if document_first else {
            "multi_match": {
                "query": cleaned,
                "fields": ["title^5", "metadata.title^5", "abstract^2", "content"],
                "operator": "or",
            }
        })
        filters = []
        if temporal_range:
            bounds = {key: int(value) for key, value in temporal_range.items()
                      if key in {"gt", "gte", "lt", "lte"}}
            filters.append({"bool": {"should": [
                {"range": {"publication_year": bounds}},
                {"range": {"metadata.publication_year": bounds}},
                {"range": {"metadata.year": bounds}},
            ], "minimum_should_match": 1}})
        if language:
            filters.append({"term": {"language": language}})
        if document_type:
            filters.append({"term": {"document_type": document_type}})
        if document_ids:
            filters.append({"terms": {"document_id": [str(item) for item in document_ids]}})
        if section_types:
            filters.append({"terms": {"section_type": [str(item) for item in section_types]}})
        body = {
            "query": {"bool": {
                "must": [must_query], "filter": filters,
                "must_not": ([{"term": {"section_type": "references"}}]
                             if exclude_references and not section_types else []),
            }},
            "highlight": {"fields": {"content": {
                "fragment_size": 150, "number_of_fragments": 3,
                "pre_tags": ["**"], "post_tags": ["**"],
            }}},
            "size": max(1, min(int(size), 100)),
        }
        response = self.es.search(index=index_name or self.index_name, body=body)
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            content = self._safe_utf8(source.get("content", ""))
            results.append({
                "score": hit.get("_score"),
                "filename": self._safe_utf8(source.get("filename", "")),
                "content": content[:500] + "..." if len(content) > 500 else content,
                "highlights": [
                    self._safe_utf8(item)
                    for item in hit.get("highlight", {}).get("content", [])
                ],
                "generation": source.get("ingestion_generation", ""),
                "document_id": source.get("document_id", ""),
                "title": source.get("title") or (source.get("metadata") or {}).get("title", ""),
                "publication_year": source.get("publication_year") or (source.get("metadata") or {}).get("publication_year") or (source.get("metadata") or {}).get("year"),
                "topic_aliases_text": source.get("topic_aliases_text", ""),
                "document_type": source.get("document_type", ""),
                "language": source.get("language", ""),
            })
        return results

        search_body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "or"
                    }
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        # 控制台友好: 用 ** 标记命中词
                        "pre_tags": ["**"],
                        "post_tags": ["**"]
                    }
                }
            },
            "size": size
        }

        try:
            response = self.es.search(index=self.index_name, body=search_body)
            results = []
            for hit in response['hits']['hits']:
                source = hit.get('_source', {})
                filename = self._safe_utf8(source.get('filename', ''))
                content = self._safe_utf8(source.get('content', ''))
                highlights = hit.get('highlight', {}).get('content', [])
                highlights = [self._safe_utf8(h) for h in highlights]

                result = {
                    'score': hit['_score'],
                    'filename': filename,
                    'content': content[:500] + '...' if len(content) > 500 else content,
                    'highlights': highlights
                }
                results.append(result)
            return results
        except Exception as e:
            logger.exception(f"检索失败: {e}")
            return []

    def _safe_utf8(self, text):
        """安全处理UTF-8编码: 任何输入都不抛 UnicodeDecodeError"""
        if text is None:
            return ""
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='replace')
        return str(text)
