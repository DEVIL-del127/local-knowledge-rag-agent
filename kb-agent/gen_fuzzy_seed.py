# -*- coding: utf-8 -*-
"""模糊时间窗口种子数据生成：模板 × 窗口类 → data/fuzzy_seed.jsonl

窗口分类任务（方案 v2 §3.5）：模糊时间词 + 上下文 → 窗口类别
- 确定性模糊（近三个月/今年/上个月）不进模型（规则已解）
- 半模糊（最近/近期/前阵子/年初…）→ 模型窗口分类
- 高度模糊（那几年/毕业后…）→ 模型判 fuzzy → 澄清

种子数据 = 规则知识转监督信号（初始版），真实分布靠 nlu_validate 标注闭环修正。
"""
import os
import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 窗口类别表（与 FUZZY_DEFAULTS 对应；value 为 (n, unit) 或 None=澄清）
WINDOW_CLASSES = {
    "none": None,            # 无模糊时间意图（模型专用类，防误触发）
    "recent_1week": (1, "week"),
    "recent_1month": (1, "month"),
    "recent_3month": (3, "month"),
    "recent_6month": (6, "month"),
    "recent_1year": (12, "month"),
    "recent_2year": (24, "month"),
    "recent_3year": (36, "month"),
    "year_start": None,      # 年初 → 规则 _h_fuzzy_period 已解；保留类目做模型兜底
    "year_mid": None,
    "year_end": None,
    "half_first": None,
    "half_second": None,
    "fuzzy_vague": None,     # 那几年/毕业后 → 澄清
}

# 触发词 → 窗口类（种子监督信号；真实语境靠模型泛化）
SEED_WORDS = {
    "最近": "recent_3month", "近期": "recent_3month", "前阵子": "recent_1month",
    "最近一段时间": "recent_3month", "最近一两个月": "recent_1month",
    "最近半年": "recent_6month", "这半年": "recent_6month",
    "最近一年": "recent_1year", "近一年": "recent_1year", "这一年来": "recent_1year",
    "最近两年": "recent_2year", "近两年": "recent_2year",
    "最近三年": "recent_3year", "近三年": "recent_3year",
    "那几年": "fuzzy_vague", "刚发布那会儿": "fuzzy_vague",
    "毕业后": "fuzzy_vague", "刚毕业那阵": "fuzzy_vague",
    "刚入行那会儿": "fuzzy_vague", "刚接触那阵子": "fuzzy_vague",
}

# 查询模板（论文检索语境）
TEMPLATES = [
    "帮我找{词}的论文",
    "{词}发表的文献有哪些",
    "{词}关于贝叶斯的文章",
    "有没有{词}的GAN论文",
    "{词}的研究进展",
    "查一下{词}的文献",
    "{词}有哪些论文",
    "我想看看{词}的文章",
    "{词}关于回声状态网络的",
    "库里{词}的论文",
    "{词}的期刊文章",
    "{词}时间序列预测的文献",
    "整理{词}的论文列表",
    "{词}的综述文章有哪些",
    "{词}相关的研究",
    "找找{词}的论文",
    "{词}有哪些研究",
    "看看{词}的文献",
    "{词}的会议论文",
    "{词}关于插补的",
]

# 附加干扰词（无模糊词 → none 类负样本，防模型对任意查询都分类）
DISTRACTORS = ["", "的", "相关的", "最新的", "中文"]

# 负样本模板（无模糊时间意图）
NEGATIVE_TEMPLATES = [
    "贝叶斯网络的研究",
    "GAN的生成对抗训练",
    "回声状态网络综述",
    "时间序列预测方法",
    "库里有几篇论文",
    "张昭昭的文章",
    "关于缺失值插补的文献",
    "ESN和LSTM的区别",
    "扩散模型最新进展",
    "元启发优化算法",
    "降维方法对比",
    "地震数据插补研究",
    "遥测数据异常检测",
    "储层计算综述",
    "贝叶斯时空模型",
    "信息期刊的论文",
    "硕士论文有哪些",
    "英文文献",
    "引用格式怎么写",
    "库里所有文献",
]


def gen_seed(seed=42, per_word=20, per_neg=20):
    random.seed(seed)
    rows = []
    for word, cls in SEED_WORDS.items():
        for _ in range(per_word):
            tpl = random.choice(TEMPLATES)
            dist = random.choice(DISTRACTORS)
            q = tpl.format(词=word + dist)
            rows.append({"query": q, "window_class": cls,
                         "word": word, "src": "seed"})
    # 负样本（none 类）
    for tpl in NEGATIVE_TEMPLATES:
        for _ in range(per_neg // len(NEGATIVE_TEMPLATES) + 1):
            q = tpl
            rows.append({"query": q, "window_class": "none",
                         "word": "", "src": "neg"})
    return rows


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    rows = gen_seed()
    path = os.path.join(out_dir, "fuzzy_seed.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 统计
    from collections import Counter
    c = Counter(r["window_class"] for r in rows)
    print(f"种子数据: {len(rows)} 条 → {path}")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    # 类目数量检查（训练需要每类 ≥10）
    low = [k for k, v in c.items() if v < 10]
    if low:
        print("警告: 类目样本不足:", low)


if __name__ == "__main__":
    main()
