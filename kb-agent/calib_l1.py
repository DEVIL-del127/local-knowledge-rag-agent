# -*- coding: utf-8 -*-
"""L1 阈值校准 v2：leave-one-out 相似度分布

对种子库每个样本：从索引剔除该样本 → 用其余样本检索 → 记录 top1 相似度与正确性。
反映「未见过的同义表达」的真实相似度分布，用于定 KNN_SIM_HIGH/LOW。
"""
import sys
import json
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
from es_client import ESClient
from embedder import Embedder
from nlu import analyze as nlu_analyze


def main():
    es = ESClient()
    emb = Embedder()
    seeds = json.load(open("data/intent_examples_seed.json", encoding="utf-8"))
    print(f"种子: {len(seeds)}")

    # 只校准「规则层未命中」的意图（L1 职责范围）——这些是语义层的真实服务对象
    l1_seeds = []
    for s in seeds:
        sq = nlu_analyze(s["text"])
        if not sq.intent:
            l1_seeds.append(s)
    print(f"规则层未命中（L1 职责）: {len(l1_seeds)}")
    dist = Counter(s["intent"] for s in l1_seeds)
    print("  意图分布:", dict(dist))

    rows = []
    for s in l1_seeds:
        q = s["text"]
        qv = emb.encode([q], query_mode=True)[0]
        # 剔除自身（es.knn 支持排除 doc_id 的话用 filter；否则 k=5 找 top1 非自身）
        hits = es.hits(es.knn("intent_examples", qv.tolist(), k=5, source=["intent", "text"]))
        if not hits:
            rows.append((q, s["intent"], 0.0, None))
            continue
        # 跳过自身文本
        top = None
        for h in hits:
            if h["_source"]["text"] != q:
                top = h
                break
        if top is None:
            rows.append((q, s["intent"], 0.0, None))
            continue
        rows.append((q, s["intent"], top["_score"], top["_source"]["intent"]))

    sims = sorted(r[2] for r in rows)
    n = len(sims)
    if n == 0:
        print("无样本")
        return
    print(f"\n相似度分布: min={sims[0]:.3f} p25={sims[n//4]:.3f} "
          f"median={sims[n//2]:.3f} p75={sims[3*n//4]:.3f} max={sims[-1]:.3f}")

    print("\n阈值扫描:")
    for thr in [0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.85]:
        hit = [r for r in rows if r[2] >= thr]
        correct = sum(1 for _, exp, _, got in hit if got == exp)
        cov = len(hit) / n
        acc = correct / len(hit) if hit else 1.0
        print(f"  thr={thr:.2f}: 命中 {len(hit):2d}/{n} (cov={cov*100:.0f}%) "
              f"正确 {correct}/{len(hit)} (acc={acc*100:.0f}%)")

    print("\n=== 明细（低相似度错误样本是扩种子的目标） ===")
    for q, exp, sim, got in sorted(rows, key=lambda r: r[2]):
        mark = "OK " if got == exp else "FAIL"
        print(f"  {mark} sim={sim:.3f} exp={exp} got={got} | {q}")


if __name__ == "__main__":
    main()
