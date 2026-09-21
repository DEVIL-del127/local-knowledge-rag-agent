# -*- coding: utf-8 -*-
"""快速验证：元数据写回 + chunks 向量检索"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from es_client import ESClient
from embedder import Embedder

es = ESClient()
emb = Embedder()

# 1) 元数据抽查
resp = es.search_all("pdf_documents", source=["id", "filename", "metadata.year", "metadata.doc_type",
                                              "metadata.venue", "metadata.language", "metadata.is_translated"], size=100)
print("=== 元数据抽查 ===")
for h in es.hits(resp):
    m = h["_source"].get("metadata", {})
    print(f"  {h['_source']['id'][:40]:<42} year={m.get('year')} type={m.get('doc_type'):<10} "
          f"venue={str(m.get('venue'))[:18]:<20} lang={m.get('language')} tr={m.get('is_translated')}")

# 2) kNN 检索验证
print("\n=== kNN 检索验证（query: 回声状态网络的训练方法） ===")
qv = emb.encode(["回声状态网络的训练方法"], query_mode=True)[0]
hits = es.hits(es.knn("pdf_chunks", qv.tolist(), k=3, source=["doc_id", "text"]))
for h in hits:
    print(f"  [{h['_score']:.3f}] {h['_source']['doc_id'][:40]}")
    print(f"      {h['_source']['text'][:80]}")

# 3) 双路召回验证（BM25 + kNN）
print("\n=== BM25 验证（query: 贝叶斯推理） ===")
r = es.search("pdf_documents", {
    "query": {"match": {"content": "贝叶斯推理"}},
    "_source": ["id"], "size": 5,
})
for h in es.hits(r):
    print(f"  [{h['_score']:.2f}] {h['_source']['id'][:50]}")
