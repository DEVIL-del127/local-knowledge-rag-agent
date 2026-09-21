# -*- coding: utf-8 -*-
"""拆分器测试矩阵：rule 模式（纯函数）+ semantic 模式（向量断层）+ 继承断言"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from splitter import split, rule_split, semantic_split
from nlu import analyze

# ---------- rule 模式矩阵 ----------
RULE_CASES = [
    # (query, 期望段数, 期望子句列表)
    ("找25年之前的文献", 1, ["找25年之前的文献"]),
    ("2024年关于缺失值插补的论文有哪些", 1, ["2024年关于缺失值插补的论文有哪些"]),
    ("2024年的ESN论文有哪些，还有关于GAN的", 2, ["2024年的ESN论文有哪些", "还有关于GAN的"]),
    ("我想知道2024年的论文里有没有讲贝叶斯的", 1, ["2024年的论文里有没有讲贝叶斯的"]),
    ("去年发了哪些论文，今年呢", 2, ["去年发了哪些论文", "今年呢"]),
    ("2021年以及以前的文献有哪些", 1, ["2021年以及以前的文献有哪些"]),
    ("告诉我3060什么时候发行的，现在性能最好的显卡是什么", 2,
     ["3060什么时候发行的", "现在性能最好的显卡是什么"]),
    ("2024年的论文有哪些，还有关于GAN的，张昭昭的呢", 3,
     ["2024年的论文有哪些", "还有关于GAN的", "张昭昭的呢"]),
    ("库里总共有多少篇文献", 1, ["库里总共有多少篇文献"]),
    ("不要参考文献里的，只要知识库中的", 1, ["不要参考文献里的，只要知识库中的"]),
    ("帮我写个Python脚本", 1, ["写个Python脚本"]),
    # ---- 名词并列“和/与”（2026-08-21 评审补充） ----
    ("两个时期的新能源汽车销量变化和充电桩建设数量", 2,
     ["两个时期的新能源汽车销量变化和", "充电桩建设数量"]),
    ("新能源汽车销量和充电桩数量", 2, ["新能源汽车销量和", "充电桩数量"]),
    ("2020年和2021年的论文", 1, ["2020年和2021年的论文"]),      # 时间并列不切
    ("3月和4月的文献", 1, ["3月和4月的文献"]),                  # 时间并列不切
    ("贝叶斯和GAN的论文有哪些", 1, ["贝叶斯和GAN的论文有哪些"]),  # 定语结构不切
    ("张昭昭和李俊明的论文", 1, ["张昭昭和李俊明的论文"]),        # 作者并列+共享中心词不切
    ("ESN和LSTM哪个好", 1, ["ESN和LSTM哪个好"]),                # 对比疑问不切
    # ---- 时间线案例（2026-08-21 用户案例） ----
    ("每次加息的幅度是多少，同期美国CPI同比增速从多少变化到了多少，请按月列出时间线", 3,
     ["每次加息的幅度是多少", "同期美国CPI同比增速从多少变化到了多少", "按月列出时间线"]),
    ("美联储共进行了多少次加息操作，每次加息的幅度是多少", 2,
     ["美联储共进行了多少次加息操作", "每次加息的幅度是多少"]),
    ("找一下2023年5月3号到2025年3月的文献，并告诉我25年一发了几篇文章", 2,
     ["找一下2023年5月3号到2025年3月的文献", "25年一发了几篇文章"]),
    ("提取该日期前5个交易日和后3个交易日内，股票的最高价和最低价", 1,
     ["提取该日期前5个交易日和后3个交易日内，股票的最高价和最低价"]),
    ("统计最大值和最小值", 1, ["统计最大值和最小值"]),
    ("并分别计算:环比:2026年7月 vs 2026年6月,以及2026年8月 vs 2026年7月。"
     "同比:2026年7月 vs 2025年7月,以及2026年8月 vs 2025年8月。最终输出:正向增长组合", 2,
     ["并分别计算:环比:2026年7月 vs 2026年6月,以及2026年8月 vs 2026年7月。"
      "同比:2026年7月 vs 2025年7月,以及2026年8月 vs 2025年8月",
      "最终输出:正向增长组合"]),
]

# ---------- semantic 模式矩阵（需 bge；无标点语义断层） ----------
SEMANTIC_CASES = [
    ("2024年的ESN论文有哪些 GAN的也行", 2),      # 无标点断层 → 向量补切
    ("2024年的论文里有没有讲贝叶斯的", 1),        # 嵌套限定 → 不切
]

# ---------- 继承断言（时间集合继承后，键为 time_ranges） ----------
INHERIT_CASES = [
    # (query, 子句索引(1起), 期望继承的属性集合)
    ("2024年的ESN论文有哪些，还有关于GAN的", 2, {"time_ranges"}),
    ("2024年的论文有哪些，还有关于GAN的，张昭昭的呢", 3, {"time_ranges"}),
    ("去年发了哪些论文，今年呢", 2, set()),       # 有自己的时间 → 不继承
    ("2024年的论文有哪些，张昭昭的呢", 2, {"time_ranges"}),
]


def main():
    fails = []
    ok = 0

    # 1. rule 模式
    for q, n, subs in RULE_CASES:
        got = [s for s, _ in rule_split(q)]
        if len(got) == n and got == subs:
            ok += 1
        else:
            fails.append(("RULE", q, (n, subs), got))

    # 2. semantic 模式（lazy 加载 bge）
    sem_ok = True
    try:
        from embedder import Embedder
        emb = Embedder()
        for q, n in SEMANTIC_CASES:
            got = semantic_split(q, emb)
            if len(got) == n:
                ok += 1
            else:
                fails.append(("SEMANTIC", q, n, got))
    except Exception as e:
        sem_ok = False
        fails.append(("SEMANTIC", "加载模型失败", str(e), "-"))

    # 3. 继承断言
    for q, idx, expect in INHERIT_CASES:
        subs = split(q, analyze_fn=analyze)
        if idx <= len(subs):
            sub = subs[idx - 1]
            got = set(sub.inherited.keys())
            if got == expect:
                ok += 1
            else:
                fails.append(("INHERIT", q, expect, got))
        else:
            fails.append(("INHERIT", q, f"无子句{idx}", len(subs)))

    total = len(RULE_CASES) + len(SEMANTIC_CASES) + len(INHERIT_CASES)
    print(f"拆分测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%"
          + ("" if sem_ok else "（semantic 模型未加载，仅统计 rule/继承）"))
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] Q: {q}\n     期望={e}\n     实际={g}")
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
