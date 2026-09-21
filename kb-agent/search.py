# -*- coding: utf-8 -*-
"""检索执行层：按路由意图分派三路召回（BM25 / 向量 kNN / 属性过滤）+ RRF 融合 + 完整性自检"""
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from es_client import ESClient
from embedder import Embedder
from config import INDEX_DOCS, INDEX_CHUNKS, SMALL_KB
from router import RouteResult, Entities


@dataclass
class SearchResult:
    intent: str
    docs: List[Dict] = field(default_factory=list)
    total_in_kb: int = 0
    matched: int = 0
    stats: Optional[Dict] = None
    message: str = ""
    clarification: str = ""

    def to_dict(self):
        return {
            "intent": self.intent,
            "total_in_kb": self.total_in_kb,
            "matched": self.matched,
            "stats": self.stats,
            "message": self.message,
            "docs": self.docs,
        }


def rrf(lists, k=60):
    scores = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Searcher:
    def __init__(self, es: ESClient = None, emb: Embedder = None):
        self.es = es or ESClient()
        self.emb = emb or Embedder()

    # ---------- 全库枚举（属性/盘点/混合的基础） ----------
    def enumerate_docs(self, source=None):
        resp = self.es.search_all(
            INDEX_DOCS,
            source=source or ["id", "filename", "page_count", "metadata"],
            size=SMALL_KB + 100,
        )
        return [h["_source"] for h in self.es.hits(resp)]

    # ---------- 属性过滤 ----------
    def filter_docs(self, docs, ent: Entities):
        out = []
        for d in docs:
            m = d.get("metadata") or {}
            y = m.get("year")
            ok = True
            # 年份
            if ent.year_op == "lte" and (y is None or y > ent.year_to):
                ok = False
            elif ent.year_op == "lt" and (y is None or y >= ent.year_to):
                ok = False
            elif ent.year_op == "gte" and (y is None or y < ent.year_from):
                ok = False
            elif ent.year_op == "gt" and (y is None or y <= ent.year_from):
                ok = False
            elif ent.year_op == "between" and (y is None or not (ent.year_from <= y <= ent.year_to)):
                ok = False
            elif ent.year_op == "exact" and y != ent.year_exact:
                ok = False
            # 作者
            if ok and ent.author:
                hay = f"{d.get('filename','')} {m.get('Author','')}"
                if ent.author.lower() not in hay.lower():
                    ok = False
            # 期刊/会议
            if ok and ent.venue:
                v = str(m.get("venue") or "")
                if ent.venue.lower() not in v.lower():
                    ok = False
            # 语言
            if ok and ent.language and m.get("language") != ent.language:
                ok = False
            # 类型
            if ok and ent.doc_type and m.get("doc_type") != ent.doc_type:
                ok = False
            # doc_ids
            if ok and ent.doc_ids:
                if not any(did.lower() in str(d.get("id", "")).lower() or did.lower() in str(d.get("filename", "")).lower() for did in ent.doc_ids):
                    ok = False
            # 排除
            if ok:
                ex = ent.exclude
                hay = f"{d.get('filename','')} {m.get('Author','')}"
                if any(a.lower() in hay.lower() for a in ex.get("authors", [])):
                    ok = False
                if any(yd == y for yd in ex.get("years", [])):
                    ok = False
                if any(l == m.get("language") for l in ex.get("languages", [])):
                    ok = False
                if any(t == m.get("doc_type") for t in ex.get("doc_types", [])):
                    ok = False
                if ex.get("translated") and m.get("is_translated"):
                    ok = False
                if any(did.lower() in str(d.get("id", "")).lower() for did in ex.get("doc_ids", [])):
                    ok = False
            if ok:
                out.append(d)
        return out

    # ---------- 语义召回 ----------
    def bm25_docs(self, query, size=10):
        r = self.es.search(INDEX_DOCS, {
            "query": {"match": {"content": query}},
            "_source": ["id", "filename"], "size": size,
        })
        return [(h["_source"]["id"], h["_score"]) for h in self.es.hits(r)]

    def knn_chunks(self, query, k=20, doc_filter=None):
        qv = self.emb.encode([query], query_mode=True)[0]
        body_filter = None
        if doc_filter:
            body_filter = {"terms": {"doc_id": doc_filter}}
        r = self.es.knn(INDEX_CHUNKS, qv.tolist(), k=k, source=["doc_id", "text", "chunk_id"], filter_body=body_filter)
        return [(h["_source"]["doc_id"], h["_source"]["text"], h["_score"]) for h in self.es.hits(r)]

    def best_snippets(self, doc_id, query, k=3):
        """单文档内找最相关片段"""
        r = self.es.search(INDEX_CHUNKS, {
            "query": {"bool": {"filter": {"term": {"doc_id": doc_id}},
                               "must": {"match": {"text": query}}}},
            "_source": ["text", "chunk_id"], "size": k,
        })
        hits = self.es.hits(r)
        if hits:
            return [h["_source"]["text"] for h in hits]
        # 兜底：取开头片段
        r2 = self.es.search(INDEX_CHUNKS, {
            "query": {"term": {"doc_id": doc_id}},
            "_source": ["text", "chunk_id"], "size": 2, "sort": [{"chunk_id": "asc"}],
        })
        return [h["_source"]["text"] for h in self.es.hits(r2)]

    # ---------- 意图执行 ----------
    def execute(self, query: str, route: RouteResult) -> SearchResult:
        ent = route.entities
        intent = route.intent
        if intent in ("agent_workflow", "numerical_calculation"):
            return SearchResult(
                intent, total_in_kb=0, matched=0,
                message="该步骤需要 Agent Planner/工具执行，未触发知识库检索",
            )
        total = self.es.count(INDEX_DOCS)

        if intent in ("non_kb", "invalid"):
            return SearchResult(intent, total_in_kb=total, matched=0,
                                message="非知识库任务，未触发检索" if intent == "non_kb" else "无效输入，请重新描述")
        if intent == "clarification":
            return SearchResult(intent, total_in_kb=total, matched=0,
                                clarification=route.clarification_reason, message="需要澄清")

        # ---- 属性筛选 ----
        if intent == "attribute_filter":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            docs = self._sort_docs(docs, ent)
            out = [self._doc_card(d) for d in docs]
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"已核对全库 {total} 篇，符合条件 {matched_text(len(out))}")

        # ---- 盘点/统计 ----
        if intent == "inventory":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            docs = self._sort_docs(docs, ent)
            stats = self._stats(docs)
            out = [self._doc_card(d) for d in docs]
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                stats=stats, message=f"全库共 {total} 篇，符合过滤条件 {len(out)} 篇")

        # ---- 单文档问答 ----
        if intent == "doc_qa":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            if not docs:
                return SearchResult(intent, total_in_kb=total, matched=0, message="未定位到该文档")
            out = []
            for d in docs:
                did = d["id"]
                snips = self.best_snippets(did, query, k=3)
                out.append({"doc_id": did, "filename": d.get("filename"),
                            "year": (d.get("metadata") or {}).get("year"),
                            "snippets": snips})
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"已定位 {len(out)} 篇文档，片段如下")

        # ---- 跨文档对比 ----
        if intent == "cross_doc_synthesis":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            out = []
            for d in docs[:5]:
                snips = self.best_snippets(d["id"], query, k=2)
                out.append({"doc_id": d["id"], "filename": d.get("filename"),
                            "year": (d.get("metadata") or {}).get("year"),
                            "snippets": snips})
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"对比 {len(out)} 篇文档（片段级），请人工综合")

        # ---- 参考文献查询 ----
        if intent == "citation_query":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            out = []
            for d in docs[:3]:
                did = d["id"]
                refs = self._extract_refs(did)
                out.append({"doc_id": did, "filename": d.get("filename"),
                            "references": refs})
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message="引用列表提取自文档参考文献区")

        # ---- 元数据查询 ----
        if intent == "metadata_query":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            out = [self._doc_card(d) for d in docs]
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"已定位 {len(out)} 篇，元数据如下")

        # ---- 语义检索（双路召回 + RRF） ----
        if intent == "semantic_retrieval":
            bm25 = [d for d, s in self.bm25_docs(query)]
            knn_docs = {}
            for did, text, score in self.knn_chunks(query, k=20):
                if did not in knn_docs or score > knn_docs[did][0]:
                    knn_docs[did] = (score, text)
            fused = rrf([bm25[:10], list(knn_docs.keys())[:10]])
            out = []
            for did, sc in fused[:5]:
                snip = knn_docs.get(did, (0, ""))[1] or (self.best_snippets(did, query, 1) or [""])[0]
                meta = self._meta_of(did)
                out.append({"doc_id": did, "filename": meta.get("filename"),
                            "year": meta.get("year"), "doc_type": meta.get("doc_type"),
                            "venue": meta.get("venue"), "language": meta.get("language"),
                            "is_translated": meta.get("is_translated"),
                            "score": round(sc, 4), "snippet": snip[:300]})
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"双路召回融合（BM25+向量）top {len(out)}")

        # ---- 混合：属性先行缩小 → 语义排序 ----
        if intent == "hybrid":
            docs = self.filter_docs(self.enumerate_docs(), ent)
            if not docs:
                return SearchResult(intent, total_in_kb=total, matched=0,
                                    message=f"属性过滤后候选 0 篇（全库 {total} 篇）")
            cand_ids = [d["id"] for d in docs]
            bm25 = [d for d, s in self.bm25_docs(query)]
            knn_docs = {}
            for did, text, score in self.knn_chunks(query, k=20, doc_filter=cand_ids):
                if did not in knn_docs or score > knn_docs[did][0]:
                    knn_docs[did] = (score, text)
            fused = rrf([[d for d in bm25 if d in cand_ids][:10], list(knn_docs.keys())[:10]])
            out = []
            for did, sc in fused[:5]:
                snip = knn_docs.get(did, (0, ""))[1] or ""
                meta = self._meta_of(did)
                out.append({"doc_id": did, "filename": meta.get("filename"),
                            "year": meta.get("year"), "doc_type": meta.get("doc_type"),
                            "venue": meta.get("venue"), "language": meta.get("language"),
                            "is_translated": meta.get("is_translated"),
                            "score": round(sc, 4), "snippet": snip[:300]})
            return SearchResult(intent, docs=out, total_in_kb=total, matched=len(out),
                                message=f"属性候选 {len(cand_ids)} 篇内语义排序 top {len(out)}")

        return SearchResult(intent, total_in_kb=total, matched=0, message="未实现该意图执行器")

    # ---------- 工具 ----------
    def _sort_docs(self, docs, ent: Entities):
        if ent.sort == "year_desc":
            return sorted(docs, key=lambda d: (d.get("metadata") or {}).get("year") or 0, reverse=True)
        return docs

    def _stats(self, docs):
        from collections import Counter
        return {
            "by_year": dict(Counter((d.get("metadata") or {}).get("year") for d in docs)),
            "by_type": dict(Counter((d.get("metadata") or {}).get("doc_type") for d in docs)),
            "by_language": dict(Counter((d.get("metadata") or {}).get("language") for d in docs)),
        }

    def _doc_card(self, d):
        m = d.get("metadata") or {}
        return {"doc_id": d.get("id"), "filename": d.get("filename"),
                "year": m.get("year"), "doc_type": m.get("doc_type"),
                "venue": m.get("venue"), "language": m.get("language"),
                "is_translated": m.get("is_translated"), "pages": d.get("page_count")}

    def _meta_of(self, doc_id):
        r = self.es.search(INDEX_DOCS, {
            "query": {"term": {"id.keyword": doc_id}}, "_source": ["id", "filename", "metadata"], "size": 1,
        })
        hits = self.es.hits(r)
        if not hits:
            return {"filename": doc_id, "year": None, "doc_type": None, "venue": None,
                    "language": None, "is_translated": None}
        m = hits[0]["_source"].get("metadata") or {}
        return {"filename": hits[0]["_source"].get("filename"), "year": m.get("year"),
                "doc_type": m.get("doc_type"), "venue": m.get("venue"),
                "language": m.get("language"), "is_translated": m.get("is_translated")}

    def _extract_refs(self, doc_id, max_len=2500):
        r = self.es.search(INDEX_DOCS, {
            "query": {"term": {"id.keyword": doc_id}}, "_source": ["content"], "size": 1,
        })
        hits = self.es.hits(r)
        if not hits:
            return ""
        content = hits[0]["_source"].get("content", "")
        m = re.search(r"(?:参考文献|References|REFERENCES)\s*([\s\S]{0,%d})" % max_len, content)
        return m.group(1).strip()[:max_len] if m else "(未找到参考文献区)"


def matched_text(n):
    return f"{n} 篇" if n else "0 篇（空结果，已如实核对）"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from router import Router
    router = Router()
    searcher = Searcher()
    for q in ["找25年之前的文献", "2024年关于缺失值插补的论文", "讲讲回声状态网络的训练方法",
              "2021年Neurocomputing那篇引用了哪些文献", "那篇NeurIPS论文的作者是谁", "库里总共有多少篇文献"]:
        print(f"\nQ: {q}")
        for item in router.route_many(q):
            if item.route is None or not item.executable:
                continue
            route = item.route
            res = searcher.execute(item.text, route)
            print(f"  subquery: {item.text}\n  route: {route.intent}({route.source})")
            print(f"  msg: {res.message}")
            if res.stats:
                print(f"  stats: {res.stats}")
            for d in res.docs[:3]:
                if "snippets" in d:
                    print(f"    - {d['doc_id'][:36]} year={d.get('year')} snips={len(d['snippets'])}")
                elif "references" in d:
                    print(f"    - {d['doc_id'][:36]} refs_len={len(d['references'])}")
                else:
                    print(f"    - {d.get('doc_id') or d.get('filename')} year={d.get('year')} snip={str(d.get('snippet'))[:60]}")
