# -*- coding: utf-8 -*-
"""NLU 层专项测试：时间表达式矩阵 + 意图矩阵 + 实体提取

测的是 nlu.py 纯函数输出（结构化结果），不依赖 ES/模型。
新增时间/意图形态 = 矩阵加一行。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from nlu import analyze, parse_time, extract_entities

CUR = 2026

# ---------- 时间表达式矩阵（规格驱动，当前时间默认 2026-08-20 15:00） ----------
TIME_CASES = [
    # (查询, 预期 time dict)
    ("找25年之前的文献", {"op": "lte", "to": "2024"}),
    ("2025年之前", {"op": "lte", "to": "2024"}),
    ("25年以前", {"op": "lte", "to": "2024"}),
    ("2021年及以前", {"op": "lte", "to": "2021"}),
    ("2021年及之前", {"op": "lte", "to": "2021"}),
    ("19年之后的文献", {"op": "gte", "from": "2020"}),
    ("2019年以后", {"op": "gte", "from": "2020"}),
    ("2020年以来", {"op": "gte", "from": "2020"}),
    ("23到25年", {"op": "between", "from": "2023", "to": "2025"}),
    ("23至25年", {"op": "between", "from": "2023", "to": "2025"}),
    ("23-25年", {"op": "between", "from": "2023", "to": "2025"}),
    ("23~25年", {"op": "between", "from": "2023", "to": "2025"}),
    ("2019年到2024年", {"op": "between", "from": "2019", "to": "2024"}),
    ("2019-2024年的文献", {"op": "between", "from": "2019", "to": "2024"}),
    ("1990年之前的文献", {"op": "lte", "to": "1989"}),      # 19xx 不被两位模式误吞
    ("1990年代", {"op": "between", "from": "1990", "to": "1999"}),
    ("90年代的", {"op": "between", "from": "1990", "to": "1999"}),
    ("去年发表的", {"op": "exact", "exact": "2025"}),
    ("今年发表的", {"op": "exact", "exact": "2026"}),
    ("最近两年的", {"op": "gte", "from": "2024"}),
    ("最新的文献", {"sort": "year_desc"}),
    ("2021年", {"op": "exact", "exact": "2021"}),
    ("2021年的论文", {"op": "exact", "exact": "2021"}),
    ("20年左右的文献", {"fuzzy": True}),
    ("2210.02040v3.pdf", {}),                              # arXiv 号不污染年份
    ("1-s2.0-S0925231221011309", {}),                      # DOI 不污染年份
    # ---- 月/日/时/分粒度 ----
    ("2024年3月", {"op": "between", "from": "2024-03-01", "to": "2024-03-31", "granularity": "month"}),
    ("2024年3月15日", {"op": "exact", "exact": "2024-03-15", "granularity": "day"}),
    ("2024年3月15日14:30", {"op": "exact", "exact": "2024-03-15T14:30", "granularity": "minute"}),
    ("下午3点", {"op": "exact", "exact": "T15:00", "granularity": "minute"}),
    ("14:30", {"op": "exact", "exact": "T14:30", "granularity": "minute"}),
    ("15点", {"op": "exact", "exact": "T15:00", "granularity": "minute"}),
    ("近三个月", {"op": "gte", "from": "2026-05-01", "granularity": "month"}),
    ("近一个月", {"op": "gte", "from": "2026-07-01", "granularity": "month"}),
    ("近一周", {"op": "gte", "from": "2026-08-13", "granularity": "day"}),
    ("近2小时", {"op": "gte", "from": "2026-08-20T13:00", "granularity": "minute"}),
    ("上个月", {"op": "between", "from": "2026-07-01", "to": "2026-07-31", "granularity": "month"}),
    ("3月到5月", {"op": "between", "from": "2026-03-01", "to": "2026-05-31", "granularity": "month"}),
    ("今年3月", {"op": "between", "from": "2026-03-01", "to": "2026-03-31", "granularity": "month"}),
    # ---- 半模糊默认 / 高度模糊 ----
    ("最近", {"op": "gte", "from": "2026-05-01", "granularity": "month", "fuzzy_word": "最近"}),
    ("年初", {"op": "between", "from": "2026-01-01", "to": "2026-03-31", "granularity": "month", "fuzzy_word": "年初"}),
    ("下半年", {"op": "between", "from": "2026-07-01", "to": "2026-12-31", "granularity": "month", "fuzzy_word": "下半年"}),
    ("那几年", {"fuzzy": True}),
    ("毕业后", {"fuzzy": True}),
    # ---- 动态锚点 / 未来窗口 / 至今延展 ----
    ("再到现在", {"op": "exact", "exact": "2026-08-20", "granularity": "day"}),
    ("2024年至今", {"op": "between", "from": "2024-01-01", "to": "2026-08-20", "granularity": "day"}),
    ("2022年1月至今", {"op": "between", "from": "2022-01-01", "to": "2026-08-20", "granularity": "day"}),
    ("2022年3月迄今", {"op": "between", "from": "2022-03-01", "to": "2026-08-20", "granularity": "day"}),
    ("2024年3月到2025年6月", {"op": "between", "from": "2024-03-01", "to": "2025-06-30", "granularity": "month"}),
    ("2020年5月至2021年9月", {"op": "between", "from": "2020-05-01", "to": "2021-09-30", "granularity": "month"}),
    ("未来五年", {"op": "between", "from": "2026", "to": "2030"}),
    ("未来5年(到2030年)", {"op": "between", "from": "2026", "to": "2030"}),
    # ---- 事件锚点（非典=2003 / 疫情=2020~2022 / 金融危机=2008~2009） ----
    ("非典结束到现在", {"op": "between", "from": "2003-01-01", "to": "2026-08-20", "granularity": "day"}),
    ("疫情以来", {"op": "between", "from": "2020-01-01", "to": "2026-08-20", "granularity": "day"}),
    ("疫情期间", {"op": "between", "from": "2020-01-01", "to": "2022-12-31"}),
    ("金融危机前后", {"op": "between", "from": "2007-01-01", "to": "2010-12-31"}),
    ("非典前后三个月", {"op": "between", "from": "2002-10-01", "to": "2004-03-31", "granularity": "month", "fuzzy_word": "前后3月"}),
    ("非典之后", {"op": "gte", "from": "2004"}),
    ("非典以前", {"op": "lte", "to": "2002"}),
    ("非典的论文", {}),                                   # 裸事件名 → 不解析为时间
    # ---- 跨日/跨月分钟级区间（X日X时 至 X日X时） ----
    ("2024年8月15日上午10:00 至 8月20日凌晨2:00",
     {"op": "between", "from": "2024-08-15T10:00", "to": "2024-08-20T02:00", "granularity": "minute"}),
    ("2024年8月22日 09:00 至 8月25日 18:00",
     {"op": "between", "from": "2024-08-22T09:00", "to": "2024-08-25T18:00", "granularity": "minute"}),
    ("2024年8月15日10:00至8月20日2:00",
     {"op": "between", "from": "2024-08-15T10:00", "to": "2024-08-20T02:00", "granularity": "minute"}),
    ("2024年3月5日 09:00 至 2024年3月5日 11:30",
     {"op": "between", "from": "2024-03-05T09:00", "to": "2024-03-05T11:30", "granularity": "minute"}),
    ("2024年3月5日晚上10点至凌晨2点",
     {"op": "between", "from": "2024-03-05T22:00", "to": "2024-03-06T02:00", "granularity": "minute"}),
    ("2024年8月15日至8月20日",
     {"op": "between", "from": "2024-08-15", "to": "2024-08-20", "granularity": "day"}),
    ("2024年12月30日至1月2日",
     {"op": "between", "from": "2024-12-30", "to": "2025-01-02", "granularity": "day"}),
    ("2023年5月3号到2025年3月",
     {"op": "between", "from": "2023-05-03", "to": "2025-03-31", "granularity": "day"}),
    ("2023年5月到2025年3月3号",
     {"op": "between", "from": "2023-05-01", "to": "2025-03-03", "granularity": "day"}),
    ("上午9:30至下午6:00",
     {"op": "between", "from": "T09:30", "to": "T18:00", "granularity": "minute"}),
    ("北京时间2024年8月15日上午10:00至8月20日凌晨2:00",
     {"op": "between", "from": "2024-08-15T10:00", "to": "2024-08-20T02:00",
      "granularity": "minute", "timezone": "Asia/Shanghai"}),
    ("2024年8月15日 10:00 至 8月20日",                      # 终点无时刻 → 只认起点精确
     {"op": "exact", "exact": "2024-08-15T10:00", "granularity": "minute"}),
]

# ---------- 多年份/多时间点矩阵（analyze 输出 time_ranges） ----------
# 注：非多年份用例只断言 time_ranges==[]，time 由 TIME_CASES 矩阵负责
MULTI_YEAR_CASES = [
    ("2014,2018,2022年的论文有哪些", ["2014", "2018", "2022"],
     {"op": "between", "from": "2014", "to": "2022", "granularity": "year"}),
    ("（2014,2018,2022）年的文献", ["2014", "2018", "2022"],
     {"op": "between", "from": "2014", "to": "2022", "granularity": "year"}),
    ("2014、2018、2022年发表的文章", ["2014", "2018", "2022"],
     {"op": "between", "from": "2014", "to": "2022", "granularity": "year"}),
    ("2014 2018 2022 的论文", ["2014", "2018", "2022"],
     {"op": "between", "from": "2014", "to": "2022", "granularity": "year"}),
    ("2019-2024年的文献", [], None),                     # 范围写法不算多时间点
    ("2021年", [], None),                                 # 单年不算
]


# ---------- 意图矩阵 ----------
INTENT_CASES = [
    ("找25年之前的文献", "attribute_filter"),
    ("23到25年的文献有哪些", "attribute_filter"),
    ("19年之后的文献", "attribute_filter"),
    ("1990年之前的文献", "attribute_filter"),
    ("2024年发表的文献有哪些", "attribute_filter"),
    ("2024年之后的中文论文", "attribute_filter"),
    ("最新的文献有哪些", "attribute_filter"),
    ("2024年关于缺失值插补的论文", "hybrid"),
    ("23年之前关于贝叶斯的文章", "hybrid"),
    ("2024那篇GAN+贝叶斯的", "hybrid"),
    ("库里总共有多少篇文献", "inventory"),
    ("按年份统计一下库里的文献", "inventory"),
    ("库里有几篇中文的", "inventory"),
    ("不要参考文献里的，只要知识库中的", "inventory"),
    ("张昭昭的双储层网络和他那篇回声信念网络有什么区别", "cross_doc_synthesis"),
    ("对比一下2024年和2025年那两篇ESN论文", "cross_doc_synthesis"),
    ("2021年Neurocomputing那篇引用了哪些文献", "citation_query"),
    ("那篇NeurIPS论文的作者是谁", "metadata_query"),
    ("information-15-00222的doi是什么", "metadata_query"),
    ("information-15-00222 这篇论文的方法是什么", "doc_qa"),
    ("刘月的硕士论文核心内容是什么", "doc_qa"),
    ("给我看看information期刊那篇的摘要", "doc_qa"),
    ("张昭昭2024年那篇用了什么方法", "doc_qa"),
    ("帮我写个Python脚本", "non_kb"),
    ("翻译一下这篇", "non_kb"),
    ("今天几号", "non_kb"),
    ("？", "invalid"),
    ("这篇论文的方法是什么", "clarification"),            # 无上下文 → 澄清
    ("引用格式是什么", "clarification"),                   # 真歧义
    ("讲讲回声状态网络的训练方法", None),                   # 纯语义 → 下沉
    ("ESN和LSTM哪个好", "semantic_retrieval"),           # 比较意图（确定性）
    ("有没有missing data相关的", None),
    ("这15篇里有没有讲地震的", "hybrid"),
    ("上述论文中哪些讲了扩散模型", "hybrid"),
    # ---- L0 强化：同义表达（2026-08-21 新增） ----
    ("盘一下库里的文献", "inventory"),
    ("罗列一下2024年的论文", "attribute_filter"),
    ("理一理张昭昭的论文", "attribute_filter"),
    ("相比GAN，扩散模型怎么样", "semantic_retrieval"),
    ("information-15-00222 第几页", "metadata_query"),
    ("这篇论文引用了哪些文献", "clarification"),          # 无定位 → 指代不明澄清
    ("为什么扩散模型效果更好", None),                        # 低确定性 → 下沉 kNN
    ("统计一下库里有多少篇中文的", "inventory"),
    ("合计一下近三年的论文数量", "inventory"),
    ("2024年发的文章一共几篇", "inventory"),
    # ---- L0 强化：数值计算意图（2026-08-21 评审打击 semantic_retrieval 误判） ----
    ("2024年销量同比增长率是多少", "numerical_calculation"),
    ("对比这两个时期的环比增速", "numerical_calculation"),
    ("计算2020到2021年的增长率", "numerical_calculation"),
    ("两个时期的差值是多少", "numerical_calculation"),
    ("充电桩数量占比多少", "numerical_calculation"),
    ("销量翻了几倍", "numerical_calculation"),
    ("平均每年增长多少", "numerical_calculation"),
    ("哪个时期增长更快", "numerical_calculation"),
    ("2020到2021年增长了多少", "numerical_calculation"),
    ("相比2020年，2024年翻了几番", "numerical_calculation"),
    ("增速差多少", "numerical_calculation"),
    # ---- 美联储加息时间线（2026-08-21 用户案例） ----
    ("美联储共进行了多少次加息操作", "inventory"),        # 次数统计
    ("每次加息的幅度是多少", "numerical_calculation"),    # 幅度数值
    ("CPI同比增速从多少变化到了多少", "numerical_calculation"),
    ("请按月列出时间线", "inventory"),                      # 枚举输出
]

# ---------- 实体矩阵 ----------
ENTITY_CASES = [
    ("找一下张昭昭写的论文", {"author": "张昭昭"}),
    ("2021年Neurocomputing那篇", {"venue": "neurocomputing", "author": None}),
    ("库里有英文论文吗", {"language": "en"}),
    ("硕士论文有哪些", {"doc_type": "thesis"}),
    ("information-15-00222 用了什么方法", {"doc_ids": ["information-15-00222"]}),
    ("2210.02040v3.pdf", {"doc_ids": ["2210.02040"]}),
    ("除了张昭昭的，其他都有哪些", {"exclude": {"authors": ["张昭昭"]}}),
    ("不含2025年的", {"exclude": {"years": [2025]}}),
    ("别给我看翻译版的", {"exclude": {"translated": True}}),
]


def main():
    n_ok = n_fail = 0
    fails = []

    # 1. 时间矩阵（断言忽略 raw 字段）
    for q, expect in TIME_CASES:
        got = parse_time(q, CUR).to_dict()
        got.pop("raw", None)
        if got == expect:
            n_ok += 1
        else:
            n_fail += 1
            fails.append(("TIME", q, expect, got))

    # 2. 意图矩阵（规则层判定；None = 期望下沉）
    for q, expect in INTENT_CASES:
        sq = analyze(q, current_year=CUR)
        if sq.intent == expect:
            n_ok += 1
        else:
            n_fail += 1
            fails.append(("INTENT", q, expect, sq.intent))

    # 3. 实体矩阵（只断言指定键）
    for q, expect in ENTITY_CASES:
        ent = extract_entities(q)
        ok = True
        for k, v in expect.items():
            if ent.get(k) != v:
                ok = False
                break
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            fails.append(("ENTITY", q, expect, ent))

    # 4. 多年份矩阵（time_ranges + 合并 time）
    for q, exp_ranges, exp_time in MULTI_YEAR_CASES:
        sq = analyze(q, current_year=CUR)
        got_ranges = [t["exact"] for t in sq.time_ranges]
        if exp_time is None:
            ok_cond = got_ranges == exp_ranges
            got_time = sq.time  # 不做断言
        else:
            got_time = dict(sq.time)
            got_time.pop("raw", None)
            ok_cond = got_ranges == exp_ranges and got_time == exp_time
        if ok_cond:
            n_ok += 1
        else:
            n_fail += 1
            fails.append(("MULTI_YEAR", q, (exp_ranges, exp_time), (got_ranges, sq.time)))

    total = len(TIME_CASES) + len(INTENT_CASES) + len(ENTITY_CASES) + len(MULTI_YEAR_CASES)
    print(f"NLU 测试: 总计 {total}, 通过 {n_ok}, 失败 {n_fail}, 准确率 {n_ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] Q: {q}\n     期望={e}\n     实际={g}")
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
