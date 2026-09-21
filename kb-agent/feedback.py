# -*- coding: utf-8 -*-
"""反馈闭环：query_logs 留痕 + 示例库回写（自进化）"""
import sys
import json
import time
from datetime import datetime, timezone

from es_client import ESClient
from embedder import Embedder
from config import INDEX_LOGS, INDEX_EXAMPLES
from router import RouteResult


def log_route(es: ESClient, emb: Embedder, query: str, route: RouteResult,
              hits: int = 0, feedback: str = None):
    """每次检索留痕。feedback: correct / wrong / None(未知)"""
    qv = emb.encode([query], query_mode=True)[0]
    doc = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_norm": query,
        "query_embedding": qv.tolist(),
        "intent": route.intent,
        "confidence": route.confidence,
        "source": route.source,
        "entities": json.dumps(route.entities.to_dict(), ensure_ascii=False),
        "hits": hits,
        "feedback": feedback or "unknown",
    }
    es.index_doc(INDEX_LOGS, doc, refresh=True)


def weekly_sync(es: ESClient, emb: Embedder, min_conf=0.8):
    """把 feedback=correct 且 conf>=min_conf 的路由样本并入意图示例库（去重）"""
    r = es.search(INDEX_LOGS, {
        "query": {"bool": {"filter": [
            {"term": {"feedback": "correct"}},
            {"range": {"confidence": {"gte": min_conf}}},
        ]}},
        "_source": ["query_norm", "intent", "confidence"], "size": 500,
    })
    candidates = [h["_source"] for h in es.hits(r)]
    # 现有示例去重
    existing = es.search(INDEX_EXAMPLES, {"query": {"match_all": {}},
                                          "_source": ["text"], "size": 2000})
    existing_texts = {h["_source"]["text"] for h in es.hits(existing)}

    added = 0
    for c in candidates:
        if c["query_norm"] in existing_texts:
            continue
        qv = emb.encode([c["query_norm"]], query_mode=True)[0]
        es.index_doc(INDEX_EXAMPLES, {
            "intent": c["intent"], "text": c["query_norm"],
            "embedding": qv.tolist(),
        }, refresh=True)
        added += 1
    return {"candidates": len(candidates), "added": added}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    es = ESClient()
    emb = Embedder()
    print("示例库同步:", weekly_sync(es, emb))
