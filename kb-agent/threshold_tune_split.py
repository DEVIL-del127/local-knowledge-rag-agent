# -*- coding: utf-8 -*-
"""拆分语义阈值 grid search：SIM_CUT_LOW × SIM_CUT_HIGH

数据集 data/split_grid_cases.json：[{query, expected}] expected=期望段数
判据：semantic_split 输出段数 == expected 即对。
覆盖两类边界：
  strong（qmark/conj）：sim < high → 切；否则看独立性
  non-strong（punct/guide）：sim < low → 切；否则合并
  + 规则锚点前置：事件窗口后段强制切、限定补全（"的"结尾）由 rule_split 吸收

用法：
  python threshold_tune_split.py             # 全量 grid + 输出最优
  python threshold_tune_split.py --table     # 输出混淆矩阵表
"""
import sys
import json
import os
import itertools

sys.stdout.reconfigure(encoding="utf-8")
from splitter import semantic_split

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "split_grid_cases.json")


def load_cases():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def evaluate(emb, low, high, cases):
    ok = 0
    detail = []
    for c in cases:
        q, exp = c["query"], c["expected"]
        try:
            got = len(semantic_split(q, emb, sim_low=low, sim_high=high))
        except Exception as e:
            got = -1
        good = (got == exp)
        ok += 1 if good else 0
        detail.append((q, exp, got, good))
    return ok, len(cases), detail


def main():
    cases = load_cases()
    if not cases:
        print(f"数据集为空：{DATA_PATH}")
        print("格式: [{\"query\": \"...\", \"expected\": 2}]")
        return
    from embedder import Embedder
    emb = Embedder()
    print(f"数据集: {len(cases)} 例\n")

    lows = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    highs = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []
    for low, high in itertools.product(lows, highs):
        if low >= high:
            continue
        ok, total, detail = evaluate(emb, low, high, cases)
        acc = ok / total
        results.append((acc, low, high, detail))
        print(f"low={low:.2f} high={high:.2f} -> acc={acc*100:.1f}% ({ok}/{total})")

    best = max(results, key=lambda x: x[0])
    print(f"\n最佳: low={best[1]:.2f} high={best[2]:.2f} acc={best[0]*100:.1f}%")
    print("\n=== 最佳参数下逐例结果 ===")
    for q, exp, got, good in best[3]:
        mark = "OK " if good else "FAIL"
        print(f"  {mark} 期望={exp} 实际={got} | {q[:40]}")


if __name__ == "__main__":
    main()
