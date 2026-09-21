# vector_store.py - ChromaDB 向量库(与 ES 检索库分开存放)
# 数据持久化在项目根 vector_db/ 目录, 与 ES 容器数据完全独立
import os
import re
import logging

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500   # 分块字符数(中英文通用)
DEFAULT_OVERLAP = 100      # 相邻块重叠字符数, 避免切断语义


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """固定窗口切块 + 重叠"""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


class VectorStore:
    def __init__(self, persist_dir='vector_db', collection_name='pdf_chunks',
                 embedder=None, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP,
                 read_only=False):
        self.read_only = read_only
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = embedder

        if read_only:
            if not os.path.isfile(os.path.join(persist_dir, "chroma.sqlite3")):
                raise FileNotFoundError("vector store unavailable; runtime cannot initialize it")
        else:
            os.makedirs(persist_dir, exist_ok=True)
        if read_only:
            from chromadb.config import Settings
            from core.read_only_chroma import ReadOnlyChromaClient
            self.client = ReadOnlyChromaClient(chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(migrations="validate", allow_reset=False, anonymized_telemetry=False),
            ))
        else:
            self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None if read_only else self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    def count(self) -> int:
        if self.collection is None:
            raise ValueError("read-only queries require an explicit pinned collection")
        return self.collection.count()

    def get_collection(self, name: str):
        return self.client.get_collection(name=name)

    def count_collection(self, name: str) -> int:
        return int(self.get_collection(name).count())

    def collection_exists(self, name: str) -> bool:
        try:
            self.get_collection(name)
            return True
        except Exception:
            return False

    def clear(self):
        """清空重建(与 ES 删索引同步调用)"""
        if self.read_only:
            raise PermissionError("runtime vector store cannot be rebuilt")
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_pages(self, doc_id, filename, pages) -> int:
        """按页分块 -> embedding -> 入库, 返回入库块数
        pages: list[{page_num, text}] (来自 pdf_parser 的 parsed['pages'])
        """
        if self.read_only:
            raise PermissionError("runtime vector store cannot ingest documents")
        chunks, ids, metadatas = [], [], []
        for page in pages:
            page_num = page.get('page_num', 0)
            texts = chunk_text(page.get('text', ''), self.chunk_size, self.overlap)
            for idx, t in enumerate(texts):
                chunks.append(t)
                ids.append(f"{doc_id}::{page_num}::{idx}")
                metadatas.append({
                    "filename": filename,
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk": idx,
                })

        if not chunks:
            return 0

        embeddings = self.embedder.embed(chunks)
        # 过滤无法向量化的块(返回 None 的), 其余正常入库
        valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
        if not valid:
            logger.warning(f"{filename}: 所有块均无法向量化, 跳过")
            return 0
        n_dropped = len(chunks) - len(valid)
        if n_dropped:
            logger.warning(f"{filename}: {n_dropped} 个块无法向量化(特殊符号), 已跳过")
        keep_ids = [ids[i] for i, _ in valid]
        keep_emb = [e for _, e in valid]
        keep_docs = [chunks[i] for i, _ in valid]
        keep_meta = [metadatas[i] for i, _ in valid]

        self.collection.upsert(
            ids=keep_ids,
            embeddings=keep_emb,
            documents=keep_docs,
            metadatas=keep_meta,
        )
        return len(keep_ids)

    def search(self, query, top_k=10) -> list:
        """向量检索: 返回 [{filename, page, chunk, text, score}]
        text 已做上下文扩展(拼接相邻块), 减少块边界切断句子的情况
        """
        emb = self.embedder.embed_query(query)
        if emb is None:
            logger.warning("查询文本无法向量化, 请简化关键词或换一种说法")
            return []
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs = res['documents'][0]
        metas = res['metadatas'][0]
        dists = res['distances'][0]
        for doc, meta, dist in zip(docs, metas, dists):
            doc_id = meta.get('doc_id', '')
            page = meta.get('page', 0)
            chunk_idx = meta.get('chunk', 0)
            text = self._expand_context(doc_id, page, chunk_idx, doc)
            results.append({
                'filename': meta.get('filename', ''),
                'page': page,
                'chunk': chunk_idx,
                'text': text,
                'score': 1.0 - dist,  # cosine 距离 -> 相似度
            })
        return results

    def search_collection(self, collection_name: str, query: str, top_k: int = 10,
                          document_ids: list[str] | None = None,
                          section_types: list[str] | None = None,
                          exclude_references: bool = True) -> list:
        """Search one explicitly pinned physical collection without changing current state."""
        emb = self.embedder.embed_query(query)
        if emb is None:
            return []
        collection = self.get_collection(collection_name)
        clauses = []
        if document_ids:
            clauses.append({"document_id": {"$in": [str(item) for item in document_ids]}})
        if section_types:
            clauses.append({"section_type": {"$in": [str(item) for item in section_types]}})
        elif exclude_references:
            clauses.append({"section_type": {"$ne": "references"}})
        where = ({"$and": clauses} if len(clauses) > 1 else clauses[0] if clauses else None)
        query_kwargs = {
            "query_embeddings": [emb], "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        res = collection.query(**query_kwargs)
        results = []
        for doc, meta, dist in zip(
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
        ):
            metadata = dict(meta or {})
            raw_page = metadata.get("page_start")
            if not isinstance(raw_page, int) or raw_page <= 0:
                raw_page = metadata.get("page")
            page = raw_page if isinstance(raw_page, int) and raw_page > 0 else None
            results.append({
                "filename": metadata.get("filename", ""),
                "page": page,
                "chunk": metadata.get("chunk_index", metadata.get("chunk", 0)),
                "text": doc,
                "score": 1.0 - float(dist),
                "generation": metadata.get("ingestion_generation", ""),
                "document_id": metadata.get("document_id", ""),
                "content_hash": metadata.get("content_hash", ""),
                "section_type": metadata.get("section_type", "body"),
                "section_title": metadata.get("section_title", metadata.get("section_path", "")),
            })
        return results

    def _expand_context(self, doc_id: str, page: int, chunk_idx: int, hit_text: str, tail_chars: int = 200) -> str:
        """上下文扩展: 前块尾部 + 命中块 + 后块头部, 还原被块边界切断的句子
        相邻块不存在(文档首/尾块)时自动跳过
        """
        if not doc_id:
            return hit_text
        neighbor_ids = [
            f"{doc_id}::{page}::{chunk_idx - 1}",
            f"{doc_id}::{page}::{chunk_idx + 1}",
        ]
        try:
            got = self.collection.get(ids=neighbor_ids, include=["documents"])
        except Exception:
            return hit_text
        by_id = {}
        ids = got.get('ids') or []
        docs = got.get('documents') or []
        for nid, ndoc in zip(ids, docs):
            if isinstance(ndoc, str) and ndoc.strip():
                by_id[nid] = ndoc.strip()

        prefix = by_id.get(neighbor_ids[0], '')[-tail_chars:]
        suffix = by_id.get(neighbor_ids[1], '')[:tail_chars]
        if not prefix and not suffix:
            return hit_text
        return (prefix + ' ' + hit_text + ' ' + suffix).strip()
