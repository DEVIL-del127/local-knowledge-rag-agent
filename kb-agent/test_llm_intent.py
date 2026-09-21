# -*- coding: utf-8 -*-
"""L2 LLM 意图分类测试：prompt 组装 / JSON 校验 / 降级链（mock provider）

不实际调 LLM（用 MockParser 固定响应），测的是：
- 触发条件（哪些查询该走 LLM）
- 输出 schema 校验（非法 JSON/缺字段 → 拒绝）
- 降级链（解析失败 → None → 上层用 L3_low 结果）
"""
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
from llm_intent import (
    LLMIntentClassifier, need_llm_intent, build_intent_messages,
    IntentResult, INTENT_SCHEMA,
)


# ---------- 触发条件矩阵（什么该走 LLM） ----------
TRIGGER_CASES = [
    # (query, 是否触发 LLM 意图分类)
    ("GAN怎么处理时间序列", True),        # 语义模糊（semantic↔hybrid 边界）
    ("墨西湖城地震数据出现在哪篇论文", True),   # 低置信区案例
    ("信息量最大的文献是哪篇", True),
    ("找25年之前的文献", False),          # 规则层强信号 → 不走
    ("库里总共有多少篇文献", False),
    ("帮我写个Python脚本", False),        # non_kb 规则命中
    ("2024年的ESN论文有哪些", False),     # 规则命中
]

# ---------- 输出解析矩阵 ----------
PARSE_CASES = [
    # (LLM 返回内容, 期望 IntentResult 或 None)
    ('{"intent": "semantic_retrieval", "confidence": 0.9, "reason": "问方法"}',
     ("semantic_retrieval", 0.9)),
    ('{"intent": "hybrid", "confidence": 0.7}', ("hybrid", 0.7)),   # reason 可选
    ("不是JSON", None),                                             # 非法 → 降级
    ('{"confidence": 0.9}', None),                                  # 缺 intent → 降级
    ('{"intent": "unknown_intent", "confidence": 0.9}', None),      # 非法意图 → 降级
    ('{"intent": "semantic_retrieval", "confidence": "高"}', None), # 类型错 → 降级
]

# ---------- 降级链矩阵 ----------
DEGRADE_CASES = [
    # (LLM 不可用/失败, 期望结果)
    ("raise", None),          # provider 抛异常 → None
    ("empty", None),          # 空响应 → None
]


def main():
    fails = []
    ok = 0
    total = len(TRIGGER_CASES) + len(PARSE_CASES) + len(DEGRADE_CASES) + 2

    # 1. 触发条件
    for q, exp in TRIGGER_CASES:
        got = need_llm_intent(q)
        if got == exp:
            ok += 1
        else:
            fails.append(("TRIGGER", q, exp, got))

    # 2. 输出解析
    for content, exp in PARSE_CASES:
        clf = LLMIntentClassifier(config={"provider": "mock", "mock_response": content})
        r = clf._parse_response(content)
        if exp is None:
            if r is None:
                ok += 1
            else:
                fails.append(("PARSE", content, None, r))
        else:
            if r and r.intent == exp[0] and abs(r.confidence - exp[1]) < 1e-6:
                ok += 1
            else:
                fails.append(("PARSE", content, exp, r))

    # 3. 降级链
    for mode, exp in DEGRADE_CASES:
        if mode == "raise":
            class Boom:
                def chat(self, messages, temperature=0.1):
                    raise RuntimeError("boom")
            clf = LLMIntentClassifier(config={"provider": "mock"})
            clf._prov = Boom()
            try:
                got = clf.classify("GAN怎么处理时间序列")
            except Exception:
                got = None  # 异常被捕获 → None
            if got is None:
                ok += 1
            else:
                fails.append(("DEGRADE_RAISE", mode, exp, got))
        elif mode == "empty":
            clf = LLMIntentClassifier(config={"provider": "mock", "mock_response": ""})
            got = clf.classify("GAN怎么处理时间序列")
            if got is None:
                ok += 1
            else:
                fails.append(("DEGRADE_EMPTY", mode, exp, got))

    # 4. prompt 组装：包含意图列表 + 查询
    msgs = build_intent_messages("GAN怎么处理时间序列")
    if msgs and msgs[-1]["role"] == "user" and "GAN怎么处理时间序列" in msgs[-1]["content"]:
        ok += 1
    else:
        fails.append(("PROMPT", "build_intent_messages", "末条含查询", msgs))
    # 5. schema 完整性
    if all(k in INTENT_SCHEMA for k in ("intent", "confidence", "reason")):
        ok += 1
    else:
        fails.append(("SCHEMA", "", INTENT_SCHEMA, "缺字段"))

    print(f"LLM 意图分类测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] {q}\n     期望={e}\n     实际={g}")
        sys.exit(1)
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
