# -*- coding: utf-8 -*-
"""复杂时间规则、L0 候选路由和 REPL 输入保护的回归测试。"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from nlu import analyze
from nlu_validate import _normalize_interactive_input
from router import Router


COMPLEX_QUERY = (
    "北京时间 2024年8月15日上午10:00 至 8月20日凌晨2:00 期间，以及 "
    "美国东部时间 2024年8月22日 09:00 至 8月25日 18:00 期间，"
    "系统发生的所有支付超时错误。"
)
MULTI_QUESTION = "找一下2023年5月3号到2025年3月的文献，并告诉我25年一发了几篇文章"
FINANCIAL_WORKFLOW = (
    "今天是 2026年8月24日。某公司通常在每季度结束后第15个工作日发布财报。"
    "找出距离今天最近的一份已发布财报的发布日期，并提取该日期前5个交易日和后3个交易日内，"
    "公司股票的最高价和最低价。最终计算这两个时间窗口的价格波动率（最高/最低 - 1）。"
)
SALES_WORKFLOW = (
    "分析某零售店 SKU(库存单位)的销售数据。请提取 2025年 和 2026年 的 6月、7月、8月 这三个月份的销售总额。"
    "并分别计算:环比增长:2026年7月 vs 2026年6月,以及 2026年8月 vs 2026年7月。"
    "同比增长:2026年7月 vs 2025年7月,以及 2026年8月 vs 2025年8月。"
    "最终输出:哪两个月份的组合(如 7月-6月 或 8月-7月)同时实现了环比和同比的正向增长?"
)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, query_mode=False):
        self.calls += 1
        return np.zeros((len(texts), 1024), dtype=np.float32)


class EmptyES:
    @staticmethod
    def knn(*args, **kwargs):
        return {}

    @staticmethod
    def hits(response):
        return []


def main():
    failures = []
    checks = 0

    sq = analyze(COMPLEX_QUERY)
    checks += 1
    expected_zones = ["Asia/Shanghai", "America/New_York"]
    if (sq.intent != "attribute_filter" or sq.confidence >= 0.95
            or len(sq.time_ranges) != 2
            or [t.get("timezone") for t in sq.time_ranges] != expected_zones
            or any(t.get("granularity") != "minute" for t in sq.time_ranges)):
        failures.append(("COMPLEX_TIME", sq.to_dict()))

    emb = FakeEmbedder()
    route = Router(es=EmptyES(), emb=emb).route(COMPLEX_QUERY)
    checks += 1
    if (route.intent != "attribute_filter" or not route.source.endswith("_candidate_fallback")
            or emb.calls != 1 or len(route.entities.time_ranges) != 2):
        failures.append(("RULE_CANDIDATE", route.to_dict(), emb.calls))

    plan = Router(es=EmptyES(), emb=FakeEmbedder()).route_many(MULTI_QUESTION)
    checks += 1
    main_items = [item for item in plan if item.route is not None]
    if ([item.text for item in main_items] != [
            "找一下2023年5月3号到2025年3月的文献", "25年一发了几篇文章"]
            or [item.route.intent for item in main_items] != ["attribute_filter", "inventory"]
            or main_items[0].route.entities.time_ranges[0].get("from") != "2023-05-03"
            or main_items[0].route.entities.time_ranges[0].get("to") != "2025-03-31"):
        failures.append(("MULTI_QUESTION_PLAN", [item.to_dict() for item in plan]))

    plan = Router(es=EmptyES(), emb=FakeEmbedder()).route_many(COMPLEX_QUERY)
    checks += 1
    routed = [item for item in plan if item.route is not None]
    if (len(plan) != 3 or len(routed) != 1 or routed[0].text != "系统发生的所有支付超时错误"
            or len(routed[0].route.entities.time_ranges) != 2):
        failures.append(("CONSTRAINT_INHERITANCE_PLAN", [item.to_dict() for item in plan]))

    emb = FakeEmbedder()
    plan = Router(es=EmptyES(), emb=emb).route_many(FINANCIAL_WORKFLOW)
    checks += 1
    mains = [item for item in plan if item.route is not None]
    expected_steps = ["date_derivation", "data_query", "calculation"]
    if (len(plan) != 5 or [item.step_type for item in mains] != expected_steps
            or [item.depends_on for item in mains] != [[1, 2], [3], [4]]
            or any(item.executable for item in mains)
            or emb.calls != 0
            or mains[0].time.get("sort") != "latest_before_anchor"
            or mains[-1].route.intent != "numerical_calculation"):
        failures.append(("DEPENDENT_WORKFLOW", [item.to_dict() for item in plan], emb.calls))

    emb = FakeEmbedder()
    plan = Router(es=EmptyES(), emb=emb).route_many(SALES_WORKFLOW)
    checks += 1
    mains = [item for item in plan if item.route is not None]
    data_step = next((item for item in mains if item.step_type == "data_query"), None)
    calc_step = next((item for item in mains if item.step_type == "calculation"), None)
    output_step = next((item for item in plan if item.role == "output"), None)
    comparisons = calc_step.parameters.get("comparisons", []) if calc_step else []
    if (len(plan) != 4 or [item.role for item in plan] != ["context", "main", "main", "output"]
            or not data_step or len(data_step.time_ranges) != 6
            or data_step.parameters.get("year_months") != [
                "2025-06", "2025-07", "2025-08", "2026-06", "2026-07", "2026-08"]
            or data_step.entities.get("exclude")
            or data_step.depends_on != [1] or calc_step.depends_on != [2]
            or [item.get("type") for item in comparisons] != ["mom", "mom", "yoy", "yoy"]
            or not output_step or output_step.depends_on != [3]
            or any(item.executable for item in plan) or emb.calls != 0):
        failures.append(("SALES_ANALYTIC_WORKFLOW", [item.to_dict() for item in plan], emb.calls))

    emb = FakeEmbedder()
    too_many = "，".join(f"{2020 + i}年的论文有哪些" for i in range(9))
    plan = Router(es=EmptyES(), emb=emb).route_many(too_many)
    checks += 1
    if (len(plan) != 1 or plan[0].route.intent != "clarification"
            or plan[0].route.source != "clarification_subquery_limit" or emb.calls != 0):
        failures.append(("SUBQUERY_FANOUT_LIMIT", [item.to_dict() for item in plan], emb.calls))

    invalid = analyze("查询2024年2月30日上午10点至下午2点的记录")
    checks += 1
    if invalid.intent != "clarification" or "无效" not in invalid.clarification:
        failures.append(("INVALID_TIME", invalid.to_dict()))

    for query in ("查询2024年8月20日至8月15日的记录", "查询25:99的记录"):
        invalid = analyze(query)
        checks += 1
        if invalid.intent != "clarification" or "无效" not in invalid.clarification:
            failures.append(("INVALID_TIME", invalid.to_dict()))

    input_cases = {
        "问题: 北京时间 2024年8月15日": "北京时间 2024年8月15日",
        "① 清洗后 : 北京时间 2024年8月15日": "",
        "--------------------------------------------------------": "",
        "最终意图 : attribute_filter": "",
        "路由输入 : 最近，最终计算": "",
        "8] 最终计算价格波动率": "",
    }
    for raw, expected in input_cases.items():
        checks += 1
        got = _normalize_interactive_input(raw)
        if got != expected:
            failures.append(("REPL_INPUT", raw, expected, got))

    print(f"规则扩展测试: 总计 {checks}, 通过 {checks - len(failures)}, 失败 {len(failures)}")
    if failures:
        for failure in failures:
            print("  ", failure)
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
