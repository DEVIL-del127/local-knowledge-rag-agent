# -*- coding: utf-8 -*-
"""L1 语义层种子扩充：82 → 200+（同义改写变体，覆盖表达多样性）

方法：对每条例种子，用同义改写（动词/疑问词/结构替换）生成变体，
再人工抽查去重。扩充后需重跑 seed_examples.py 重建索引。
"""
import sys
import json
import random
import re
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SRC = "data/intent_examples_seed.json"
DST = "data/intent_examples_seed_v2.json"

# 同义改写表：替换词 → 同义候选（按词长降序替换，防子串误伤）
SYNONYMS = [
    # 枚举/统计
    ("有哪些", ["有哪些", "有什么", "都是哪些", "有哪些啊", "都有哪些", "具体有哪些", "分别有哪些"]),
    ("几篇", ["几篇", "多少篇", "几篇文献", "有多少篇", "篇数", "数量"]),
    ("列出", ["列出", "列举", "罗列", "列一下", "给我列"]),
    ("盘点", ["盘点", "盘一下", "统计一下", "理一理", "整理下"]),
    # 检索动词
    ("找", ["找", "搜", "查", "检索", "搜一下", "查一下", "找找", "看看", "翻一下"]),
    ("帮我", ["帮我", "请帮我", "麻烦帮我", "帮我一下", "给我"]),
    # 时间表达
    ("之前", ["之前", "以前", "往前", "早于"]),
    ("之后", ["之后", "以后", "往后", "晚于"]),
    ("最近", ["最近", "近期", "最近一段时间", "前阵子"]),
    # 语义词
    ("关于", ["关于", "有关", "相关的", "涉及"]),
    ("讲", ["讲", "介绍", "说说", "讲讲", "讲一下", "介绍下"]),
    ("怎么样", ["怎么样", "如何", "怎么", "如何做"]),
    # 疑问词
    ("什么", ["什么", "啥", "哪些内容"]),
    ("吗", ["吗", "么", ""]),
    # 比较
    ("区别", ["区别", "差别", "不同", "差异", "有什么区别", "有何不同"]),
    # 元数据
    ("作者", ["作者", "谁写的", "哪个人写的", "作者是谁"]),
    ("期刊", ["期刊", "刊物", "哪个期刊", "什么期刊"]),
]

# 不安全的全局替换词（跳过，防语义漂移）
SKIP_REPLACE = ["吧", "呢"]

# 质量过滤：叠词/占位符/噪音
BAD_PATTERNS = [
    r"\.\.\.",           # 模板占位符残留
    r"(.)\1{2,}",          # 连续重复 3+（讲讲讲/翻一下一下）
    r"(一下|介绍|讲讲|说说)\1",  # 叠词
    r"^.{1,3}$",           # 过短
]


def _bad(q: str) -> bool:
    return any(re.search(p, q) for p in BAD_PATTERNS)


def variants(q: str, n=3, seed=42):
    """生成 q 的同义变体（每次随机替换 1-2 处）"""
    random.seed(seed + hash(q) % 10000)
    outs = set()
    for _ in range(n * 4):  # 多试几次，去重
        out = q
        # 随机选 1-2 个可替换词
        pool = [w for w, _ in SYNONYMS if w in out and w not in SKIP_REPLACE]
        if not pool:
            continue
        k = random.randint(1, min(2, len(pool)))
        chosen = random.sample(pool, k)
        for w in chosen:
            cands = [c for c in dict(SYNONYMS)[w] if c != w]
            if not cands:
                continue
            out = out.replace(w, random.choice(cands), 1)
        if out != q and len(out) >= 4:
            outs.add(out)
        if len(outs) >= n:
            break
    return list(outs)


def main():
    rows = json.load(open(SRC, encoding="utf-8"))
    print(f"原种子: {len(rows)}")

    new_rows = []
    seen = set(r["text"] for r in rows)
    per_intent = Counter(r["intent"] for r in rows)

    for r in rows:
        # 低频意图多生成变体（平衡分布）
        n = 4 if per_intent[r["intent"]] <= 6 else 2
        for v in variants(r["text"], n=n):
            if v not in seen and not _bad(v):
                seen.add(v)
                new_rows.append({"text": v, "intent": r["intent"]})

    all_rows = rows + new_rows
    dist = Counter(r["intent"] for r in all_rows)
    print(f"扩充后: {len(all_rows)}（新增 {len(new_rows)}）")
    print("分布:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")

    json.dump(all_rows, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已写入: {DST}")
    print("\n=== 抽查 15 条变体 ===")
    for r in new_rows[:15]:
        print(f"  [{r['intent']}] {r['text']}")


if __name__ == "__main__":
    main()
