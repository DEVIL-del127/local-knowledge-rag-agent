# -*- coding: utf-8 -*-
"""LLM 解析器测试：mock 模式全链路 + 复杂度检测器 + 降级"""
import sys
import json
sys.stdout.reconfigure(encoding="utf-8")
from llm_parser import LLMParser, need_llm, ComplexStructure, EXAMPLES
from config import LLM


def _check_need_llm():
    cases = [
        # (query, 期望)
        ("找25年之前的文献", False),
        ("2024年关于缺失值插补的论文", False),
        ("库里总共有多少篇文献", False),
        # S1 子句数 ≥4
        ("2024年的论文有哪些，还有关于GAN的，张昭昭的呢，另外看看扩散模型的", True),
        # S2 输出要求
        ("2024年的论文有哪些，需要按季度展示", True),
        # S3 事件窗口
        ("对比一下从GPT-2发布到GPT-4发布期间的技术", True),
        # S4 括号时间
        ("特别是疫情放开之后（2023年初）到今年年底的政策", True),
    ]
    ok = 0
    fails = []
    for q, expect in cases:
        got = need_llm(q)
        if got == expect:
            ok += 1
        else:
            fails.append((q, expect, got))
    return ok, len(cases), fails


def test_need_llm():
    ok, total, fails = _check_need_llm()
    assert ok == total, fails


def _check_parse_mock():
    ok = 0
    fails = []
    total = 4
    MOCK_OK = json.dumps({
        "main_query": "大语言模型在推理能力和代码生成方面的主要技术突破",
        "time_range": {"from": "2023-04-01", "to": None, "granularity": "month", "note": ""},
        "constraints": [{"type": "exclude_topic", "value": "只讲应用不讲原理"}],
        "output": {"group_by": "year", "format": "milestones"},
        "confidence": 0.9,
    }, ensure_ascii=False)
    # 1. 正常解析
    cfg = dict(LLM, provider="mock", conf_threshold=0.6, mock_response=MOCK_OK)
    p = LLMParser(cfg)
    res = p.parse("复杂问题")
    if res and res.main_query == "大语言模型在推理能力和代码生成方面的主要技术突破" and res.confidence == 0.9:
        ok += 1
    else:
        fails.append(("正常解析", res))

    # 2. 低置信 → 降级 None
    low = {"main_query": "x", "time_range": {}, "constraints": [], "output": {}, "confidence": 0.3}
    cfg2 = dict(LLM, provider="mock", conf_threshold=0.6,
                mock_response=json.dumps(low, ensure_ascii=False))
    if LLMParser(cfg2).parse("x") is None:
        ok += 1
    else:
        fails.append(("低置信应降级", "返回了结果"))

    # 3. 非法 JSON → 重试后降级 None
    cfg3 = dict(LLM, provider="mock", max_retry=1, mock_response="not json")
    if LLMParser(cfg3).parse("x") is None:
        ok += 1
    else:
        fails.append(("非法JSON应降级", "返回了结果"))

    # 4. schema 缺 main_query → 降级
    bad = {"time_range": {}, "constraints": [], "output": {}, "confidence": 0.9}
    cfg4 = dict(LLM, provider="mock", conf_threshold=0.0,
                mock_response=json.dumps(bad, ensure_ascii=False))
    if LLMParser(cfg4).parse("x") is None:
        ok += 1
    else:
        fails.append(("schema缺字段应降级", "返回了结果"))
    return ok, total, fails


def test_parse_mock():
    ok, total, fails = _check_parse_mock()
    assert ok == total, fails


def main():
    ok1, t1, f1 = _check_need_llm()
    ok2, t2, f2 = _check_parse_mock()
    ok, total = ok1 + ok2, t1 + t2
    fails = [("NEED_LLM", *f) for f in f1] + [("PARSE", *f) for f in f2]
    print(f"LLM 解析器测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] {q}\n     期望={e}\n     实际={g}")
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
