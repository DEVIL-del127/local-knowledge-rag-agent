# -*- coding: utf-8 -*-
"""回归测试：种子集 82 例（带预期意图）+ 10 例新增边界，验证路由准确率"""
import sys
import os
import json
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
# 回归默认关闭 LLM（确定性 + 不依赖 Ollama）；单独验证 L2 时设 KB_NO_LLM=0
os.environ.setdefault("KB_NO_LLM", "1")
from router import Router

ROOT = os.path.dirname(os.path.abspath(__file__))
# 回归基准固定用 v1（82 例原始种子，可复现）；扩充库 v2 只进 kNN 索引
with open(os.path.join(ROOT, "data", "intent_examples_seed_v1_bak.json"), encoding="utf-8") as f:
    seed = json.load(f)

# 新增边界用例（不在种子库中，考验泛化）
EXTRA = [
    ("把2021年那篇论文的作者和期刊告诉我", "metadata_query"),
    ("2024年发的那些文章都是关于什么的", "hybrid"),
    ("ESN和LSTM哪个好", "semantic_retrieval"),
    ("库里有几篇是2024年的", "inventory"),
    ("张昭昭2024年那篇用了什么方法", "doc_qa"),
    ("这15篇里有没有讲地震的", "hybrid"),
    ("剔除2025年的，剩下的有哪些", "attribute_filter"),
    ("给我看看information期刊那篇的摘要", "doc_qa"),
    ("神经网络的论文有哪些", "hybrid"),
    ("那篇2021年的文献是哪个期刊发的", "metadata_query"),
    ("23到25年的文献有哪些", "attribute_filter"),
    ("23年之前关于贝叶斯的文章", "hybrid"),
    ("找一下19年之后的文献", "attribute_filter"),
]

CASES = [(s["text"], s["intent"]) for s in seed] + EXTRA

# 指代类用例需要 session 上下文
SESSION_CASES = [
    ("这篇论文的方法是什么", "doc_qa"),
    ("它引用了哪些文献", "citation_query"),
    ("这篇和那篇有什么不同", "cross_doc_synthesis"),
]

def main():
    router = Router()
    ok = 0
    fail = []
    by_src = Counter()
    for q, expect in CASES:
        r = router.route(q)
        by_src[r.source] += 1
        if r.intent == expect:
            ok += 1
        else:
            fail.append((q, expect, r.intent, r.source, round(r.confidence, 3)))

    # 指代用例（带 session）
    session = {"doc_ids": ["1-s2.0-S0925231221011309"]}
    for q, expect in SESSION_CASES:
        r = router.route(q, session)
        if r.intent == expect:
            ok += 1
        else:
            fail.append((q, expect, r.intent, r.source, round(r.confidence, 3)))

    total = len(CASES) + len(SESSION_CASES)
    print(f"总计 {total} 例, 通过 {ok}, 失败 {len(fail)}, 准确率 {ok/total*100:.1f}%")
    print(f"路由来源分布: {dict(by_src)}")
    if fail:
        print("\n=== 失败用例 ===")
        for q, e, g, src, conf in fail:
            print(f"  Q: {q}\n    期望={e} 实际={g} src={src} conf={conf}")
    else:
        print("\n全部通过 ✅")

if __name__ == "__main__":
    main()
