# -*- coding: utf-8 -*-
"""阶段0：知识库优化
1) 元数据补齐：year / doc_type / venue / language / is_translated（内置 override + 正则兜底）
2) 分块向量化：content -> pdf_chunks（embedding dense_vector 1024，Ollama bge-m3）
3) 写回 pdf_documents.metadata
"""
import sys
import re
import json

from es_client import ESClient
from embedder import Embedder
from config import CHUNK_MAX_LEN, EMBED_DIM, INDEX_DOCS, INDEX_CHUNKS

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 人工核对过的元数据（15 篇，准确优先） ----------
OVERRIDES = {
    "1-s2.0-S0925231221011309-main的全文翻译": dict(year=2021, venue="Neurocomputing", doc_type="journal", is_translated=True),
    "1-s2.0-S0952197625001290-main": dict(year=2025, venue="Engineering Applications of Artificial Intelligence", doc_type="journal", is_translated=False),
    "1-s2.0-S1877050925003059-main": dict(year=2025, venue="Procedia Computer Science", doc_type="conference", is_translated=False),
    "2210.02040v3": dict(year=2022, venue="arXiv", doc_type="preprint", is_translated=False),
    "CCE-2020-paper-25的全文翻译": dict(year=2020, venue="CCE", doc_type="conference", is_translated=True),
    "information-15-00222": dict(year=2024, venue="Information", doc_type="journal", is_translated=False),
    "Metaheuristic_Method_for_Dimensionality_Reduction_Tasks的全文翻译": dict(year=2022, venue="CCE", doc_type="conference", is_translated=True),
    "NeurIPS-2019-time-series-generative-adversarial-networks-Paper": dict(year=2019, venue="NeurIPS", doc_type="conference", is_translated=False),
    "具有双储层结构的动态误差补偿回声状态网络_张昭昭": dict(year=2024, venue="控制理论与应用", doc_type="journal", is_translated=False),
    "回声信念网络及其在时间序列预测中的应用_张昭昭": dict(year=2025, venue="控制工程", doc_type="journal", is_translated=False),
    "基于GAN的视网膜血管分割标签优化方法_王植炜": dict(year=2025, venue="华中科技大学学报（自然科学版）", doc_type="journal", is_translated=False),
    "基于改进MCMC算法和代理模型的结构仿真模型更新_缪季": dict(year=2025, venue="上海交通大学学报", doc_type="journal", is_translated=False),
    "基于条件扩散模型的卫星遥测数据缺失值插补方法_庞昭辰": dict(year=2025, venue="自动化学报", doc_type="journal", is_translated=False),
    "论文刘月": dict(year=2025, venue="西安科技大学硕士学位论文", doc_type="thesis", is_translated=False),
    "贝叶斯时空统计方法及应用进展与趋势_李俊明": dict(year=2025, venue="地球信息科学学报", doc_type="journal", is_translated=False),
}

YEAR_FALLBACK = [
    (r"arXiv:\s*(\d{4})\.\d+", 1),
    (r"第\s*\d+\s*卷.*?(20\d{2})\s*年", 1),
    (r"(20\d{2})\s*年\s*\d+\s*月", 1),
    (r"Vol\.?\s*\d+.*?\((20\d{2})\)", 1),
    (r"\((20\d{2})\)\s*\d+\s*:\s*\d+", 1),
    (r"©\s*(20\d{2})", 1),
    (r"(20\d{2})\s*,\s*\d+\s*:", 1),
]


def detect_language(text):
    zh = sum(1 for ch in text[:5000] if "\u4e00" <= ch <= "\u9fff")
    return "zh" if zh / max(len(text[:5000]), 1) > 0.05 else "en"


def fallback_year(doc_id, head, meta):
    # 正文正则兜底
    for pat, grp in YEAR_FALLBACK:
        m = re.search(pat, head, re.S)
        if m:
            y = int(m.group(grp))
            if 1990 <= y <= 2026:
                return y
    # 元数据兜底（Date/Published/Subject）
    m = meta or {}
    for key in ("Date", "Published", "Subject"):
        val = str(m.get(key, ""))
        mm = re.search(r"(20\d{2})", val)
        if mm:
            return int(mm.group(1))
    return None


def chunk_text(text, max_len=CHUNK_MAX_LEN):
    paras = [p.strip() for p in re.split(r"\n+|\r+", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > max_len:  # 超长段硬切
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), max_len):
                chunks.append(p[i:i + max_len])
            continue
        if cur and len(cur) + len(p) + 1 > max_len:
            chunks.append(cur)
            cur = p
        else:
            cur = p if not cur else cur + "\n" + p
    if cur:
        chunks.append(cur)
    return chunks


CHUNK_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "chunk_id": {"type": "integer"},
            "text": {"type": "text", "analyzer": "ik_max_word"},
            "embedding": {"type": "dense_vector", "dims": EMBED_DIM, "index": True, "similarity": "cosine"},
        }
    }
}


def main():
    es = ESClient()
    emb = Embedder()

    # 1) 拉全库
    resp = es.search_all(INDEX_DOCS, size=100)
    docs = es.hits(resp)
    print(f"全库文档数: {len(docs)}")

    # 2) 建 chunks 索引
    if es.exists(INDEX_CHUNKS):
        es.delete_index(INDEX_CHUNKS)
    es.create_index(INDEX_CHUNKS, CHUNK_MAPPING)
    print("pdf_chunks 索引已重建")

    total_chunks = 0
    # 3) 逐篇处理
    for hit in docs:
        src = hit["_source"]
        doc_id = src["id"]
        content = src.get("content", "")
        head = content[:6000]
        meta = src.get("metadata") or {}

        ov = OVERRIDES.get(doc_id)
        if ov:
            year, venue, dtype = ov["year"], ov["venue"], ov["doc_type"]
            is_tr = ov["is_translated"]
        else:
            year = fallback_year(doc_id, head, meta)
            venue = None
            dtype = "journal"
            is_tr = "全文翻译" in str(src.get("filename", ""))
            print(f"  [无override] {doc_id} year={year}")

        lang = detect_language(content)

        # 写回 metadata
        es.update_doc(INDEX_DOCS, hit["_id"], {
            "metadata": {
                "year": year, "doc_type": dtype, "venue": venue,
                "language": lang, "is_translated": is_tr,
            }
        })

        # 分块 + 向量化
        chunks = chunk_text(content)
        embs = emb.encode(chunks, query_mode=False, batch_size=32)
        bulk_docs = [
            {"_id": f"{doc_id}#{i}",
             "_source": {"doc_id": doc_id, "filename": src.get("filename"),
                         "chunk_id": i, "text": chunks[i],
                         "embedding": embs[i].tolist()}}
            for i in range(len(chunks))
        ]
        for i in range(0, len(bulk_docs), 50):
            es.bulk_index(INDEX_CHUNKS, bulk_docs[i:i + 50])
        total_chunks += len(chunks)
        print(f"  ✓ {doc_id}: year={year} type={dtype} venue={venue} lang={lang} "
              f"translated={is_tr} chunks={len(chunks)}")

    print(f"\n完成: 文档 {len(docs)} 篇, 分块 {total_chunks}")
    print(f"chunks 索引计数: {es.count(INDEX_CHUNKS)}")


if __name__ == "__main__":
    main()
