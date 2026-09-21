# -*- coding: utf-8 -*-
"""建 intent_examples / query_logs 索引，并向量化种子示例入库"""
import sys
import json
import os

from es_client import ESClient
from embedder import Embedder
from config import EMBED_DIM, INDEX_EXAMPLES, INDEX_LOGS

sys.stdout.reconfigure(encoding="utf-8")

EXAMPLES_MAPPING = {
    "mappings": {
        "properties": {
            "intent": {"type": "keyword"},
            "text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": EMBED_DIM, "index": True, "similarity": "cosine"},
        }
    }
}

LOGS_MAPPING = {
    "mappings": {
        "properties": {
            "ts": {"type": "date"},
            "query_norm": {"type": "text"},
            "query_embedding": {"type": "dense_vector", "dims": EMBED_DIM, "index": True, "similarity": "cosine"},
            "intent": {"type": "keyword"},
            "confidence": {"type": "float"},
            "source": {"type": "keyword"},
            "entities": {"type": "object", "enabled": False},
            "hits": {"type": "integer"},
            "feedback": {"type": "keyword"},
        }
    }
}


def main():
    es = ESClient()
    emb = Embedder()

    for idx, mapping in ((INDEX_EXAMPLES, EXAMPLES_MAPPING), (INDEX_LOGS, LOGS_MAPPING)):
        if es.exists(idx):
            es.delete_index(idx)
        es.create_index(idx, mapping)
        print(f"索引已重建: {idx}")

    seed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intent_examples_seed_v2.json")
    if not os.path.exists(seed_path):
        seed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intent_examples_seed.json")
    with open(seed_path, encoding="utf-8") as f:
        seeds = json.load(f)
    print(f"种子示例: {len(seeds)} 条")

    texts = [s["text"] for s in seeds]
    embs = emb.encode(texts, query_mode=True, batch_size=32)  # 路由场景：示例与 query 同侧
    bulk = [
        {"_id": f"seed{i}", "_source": {
            "intent": s["intent"], "text": s["text"], "embedding": embs[i].tolist()}}
        for i, s in enumerate(seeds)
    ]
    for i in range(0, len(bulk), 50):
        es.bulk_index(INDEX_EXAMPLES, bulk[i:i + 50])

    print(f"intent_examples 计数: {es.count(INDEX_EXAMPLES)}")
    from collections import Counter
    print("分布:", dict(Counter(s["intent"] for s in seeds)))


if __name__ == "__main__":
    main()
