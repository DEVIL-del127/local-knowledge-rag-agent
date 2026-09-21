# -*- coding: utf-8 -*-
"""ES REST 客户端封装（requests）"""
import json
import requests

from config import ES_URL


class ESClient:
    def __init__(self, url=ES_URL, timeout=30):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _req(self, method, path, body=None):
        r = requests.request(
            method, self.url + path,
            json=body, timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"ES {method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    # ---- 基础 ----
    def count(self, index):
        try:
            return self._req("GET", f"/{index}/_count")["count"]
        except Exception:
            return 0

    def exists(self, index):
        try:
            self._req("GET", f"/{index}")
            return True
        except Exception:
            return False

    def create_index(self, index, mapping=None):
        # mapping 为完整 body（含 "mappings" 键）；None 则创建空索引
        return self._req("PUT", f"/{index}", mapping)

    def delete_index(self, index):
        return self._req("DELETE", f"/{index}")

    def index_doc(self, index, doc, doc_id=None, refresh=False):
        suffix = "?refresh=wait_for" if refresh else ""
        path = f"/{index}/_doc" + (f"/{doc_id}" if doc_id else "") + suffix
        return self._req("POST" if not doc_id else "PUT", path, doc)

    def update_doc(self, index, doc_id, doc):
        return self._req("POST", f"/{index}/_update/{doc_id}", {"doc": doc})

    def bulk_index(self, index, docs, refresh=False):
        """docs: list[dict]，每项 {_id?, _source}（refresh=True 时 wait_for）"""
        if not docs:
            return None
        lines = []
        for d in docs:
            action = {"index": {"_index": index}}
            if d.get("_id"):
                action["index"]["_id"] = d["_id"]
            lines.append(json.dumps(action, ensure_ascii=False))
            lines.append(json.dumps(d["_source"], ensure_ascii=False))
        payload = "\n".join(lines) + "\n"
        url = f"{self.url}/_bulk" + ("?refresh=wait_for" if refresh else "")
        r = requests.post(
            url, data=payload.encode("utf-8"), timeout=60,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        return r.json()

    # ---- 查询 ----
    def search(self, index, body):
        return self._req("POST", f"/{index}/_search", body)

    def search_all(self, index, source=None, size=1000):
        body = {"query": {"match_all": {}}, "size": size}
        if source:
            body["_source"] = source
        return self.search(index, body)

    def knn(self, index, vector, k=5, field="embedding", source=None, filter_body=None):
        body = {
            "knn": {"field": field, "query_vector": vector, "k": k, "num_candidates": 100},
            "size": k,
        }
        if filter_body:
            body["knn"]["filter"] = filter_body
        if source:
            body["_source"] = source
        return self.search(index, body)

    def hits(self, resp):
        return resp.get("hits", {}).get("hits", [])

    def sim(self, hit):
        """cosine 相似度（ES knn 返回 _score 为相似度）"""
        return hit.get("_score", 0.0)
