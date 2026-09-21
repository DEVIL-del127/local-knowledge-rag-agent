# -*- coding: utf-8 -*-
"""fuzzy_model 推理封装测试：降级链 / 窗口映射 / 模型预测（若已训练）"""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")
from fuzzy_model import (FuzzyWindowModel, CLASS_WINDOWS, DEFAULT_MAP, CONF_THRESHOLD,
                         get_default_model, reset_default_model)

CASES = [
    # (query, 期望 class 前缀或 None)
    ("帮我找最近半年的论文", "recent_6month"),
    ("最近一年关于贝叶斯的", "recent_1year"),
    ("那几年的文献", "fuzzy_vague"),
    ("近期GAN研究", "recent_3month"),
    ("前阵子的文章", "recent_1month"),
    ("完全没有模糊词的查询", None),
]

# 窗口映射完整性：所有类目都有 (n, unit) 或明确 None（澄清）
WINDOW_MAP_CASES = [
    ("recent_1week", (1, "week")),
    ("recent_3month", (3, "month")),
    ("recent_1year", (12, "month")),
    ("fuzzy_vague", None),
]


def main():
    fails = []
    ok = 0
    total = len(CASES) + len(WINDOW_MAP_CASES) + 3

    # 1. 窗口映射
    for cls, exp in WINDOW_MAP_CASES:
        got = CLASS_WINDOWS.get(cls)
        if got == exp:
            ok += 1
        else:
            fails.append(("WINDOW_MAP", cls, exp, got))

    # 2. 预测（模型可用时走模型；否则默认表兜底）
    m = FuzzyWindowModel()
    for q, exp_prefix in CASES:
        cls, conf = m.predict(q)
        if exp_prefix is None:
            if cls is None:
                ok += 1
            else:
                fails.append(("PREDICT_NONE", q, None, (cls, conf)))
        else:
            if cls == exp_prefix:
                ok += 1
            else:
                fails.append(("PREDICT", q, exp_prefix, (cls, conf)))

    # 3. 模型可用性一致性：available 与 _failed 互斥
    m2 = FuzzyWindowModel()
    if m2.available == (m2._failed is None):
        ok += 1
    else:
        fails.append(("AVAIL", "", "available==not failed", (m2.available, m2._failed)))

    # 4. 默认表一致性：DEFAULT_MAP 的类目必须存在于 CLASS_WINDOWS
    bad = [w for w, (cls, _) in DEFAULT_MAP.items() if cls not in CLASS_WINDOWS]
    if not bad:
        ok += 1
    else:
        fails.append(("DEFAULT_MAP", bad, "类目必须在 CLASS_WINDOWS", None))

    # 5. NLU 多子句共享同一实例，不重复加载权重
    reset_default_model()
    if get_default_model() is get_default_model():
        ok += 1
    else:
        fails.append(("SINGLETON", "", "same instance", None))

    print(f"fuzzy_model 测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] {q}\n     期望={e}\n     实际={g}")
        sys.exit(1)
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
