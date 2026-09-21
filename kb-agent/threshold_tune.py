# -*- coding: utf-8 -*-
"""阈值校准：grid search KNN_SIM_HIGH × KNN_SIM_LOW，指标 = 准确率 + L0+L3 覆盖率"""
import sys
import itertools
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
import router as router_mod
from router import Router
from test_regression import CASES, SESSION_CASES

SESSION = {"doc_ids": ["1-s2.0-S0925231221011309"]}


def evaluate(high, low):
    router_mod.KNN_SIM_HIGH = high
    router_mod.KNN_SIM_LOW = low
    r = Router()
    ok = 0
    srcs = Counter()
    for q, exp in CASES:
        res = r.route(q)
        srcs[res.source] += 1
        if res.intent == exp:
            ok += 1
    for q, exp in SESSION_CASES:
        res = r.route(q, SESSION)
        srcs[res.source] += 1
        if res.intent == exp:
            ok += 1
    total = len(CASES) + len(SESSION_CASES)
    acc = ok / total
    l0l3 = (srcs.get("L0", 0) + srcs.get("L3", 0)) / total
    return acc, l0l3, dict(srcs)


def main():
    results = []
    for high, low in itertools.product([0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75], [0.30, 0.35, 0.40, 0.45]):
        if low >= high:
            continue
        acc, l0l3, srcs = evaluate(high, low)
        results.append((acc, l0l3, high, low, srcs))
        print(f"high={high:.2f} low={low:.2f} -> acc={acc*100:.1f}% L0+L3={l0l3*100:.1f}% {srcs}")

    best = max(results, key=lambda x: (x[0], x[1]))
    print("\n最佳组合: high=%.2f low=%.2f acc=%.1f%% L0+L3=%.1f%%" % (best[2], best[3], best[0] * 100, best[1] * 100))


if __name__ == "__main__":
    main()
