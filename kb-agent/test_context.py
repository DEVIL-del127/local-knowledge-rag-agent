# -*- coding: utf-8 -*-
"""L3 上下文意图测试：SessionState 短句/指代继承 + 重置

纯函数（不依赖 ES/模型），测 session_state.py 的意图延续逻辑。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from session_state import SessionState, continue_intent

# ---------- 延续/重置判定矩阵 ----------
# (当前句, 上一意图, 上一实体, 期望 (是否延续, 意图, 说明))
CONTINUE_CASES = [
    ("还有呢", "inventory", {"author": None}, (True, "inventory", "")),
    ("接着列", "attribute_filter", {"year_from": 2023}, (True, "attribute_filter", "")),
    ("那篇呢", "doc_qa", {"doc_ids": ["1-s2.0-S0925231221011309"]}, (True, "doc_qa", "")),
    ("继续", "semantic_retrieval", {}, (True, "semantic_retrieval", "")),
    ("还有哪些", "attribute_filter", {"year_from": 2023}, (True, "attribute_filter", "")),
    ("换个话题", "inventory", {}, (False, None, "")),
    ("不看了", "doc_qa", {}, (False, None, "")),
    ("别的", "inventory", {}, (False, None, "")),
    ("再查一下贝叶斯", "inventory", {}, (False, "hybrid", "")),   # 有实质内容 → 走正常解析
    ("2024年的论文", "inventory", {}, (False, "attribute_filter", "")),  # 完整新问题
]

# ---------- 状态对象行为 ----------
STATE_CASES = [
    # 初始无状态
    (None, "还有呢", False),
    # 更新后状态存在
    ({"last_intent": "inventory", "last_entities": {}, "has_context": True}, "还有呢", True),
]


def main():
    fails = []
    ok = 0
    total = len(CONTINUE_CASES) + len(STATE_CASES)

    # 1. 延续/重置
    for q, last_intent, last_ent, exp in CONTINUE_CASES:
        st = SessionState(last_intent=last_intent, last_entities=last_ent)
        got = continue_intent(q, st)
        # got: (continue_flag, intent, reason) 或 None
        if exp[0]:
            if got and got[0] and got[1] == exp[1]:
                ok += 1
            else:
                fails.append(("CONTINUE", q, exp, got))
        else:
            if not got or not got[0]:
                ok += 1
            else:
                fails.append(("RESET", q, exp, got))

    # 2. 状态对象
    for st_dict, q, exp in STATE_CASES:
        st = SessionState(**st_dict) if st_dict else SessionState()
        got = st.has_context and continue_intent(q, st) is not None
        if got == exp:
            ok += 1
        else:
            fails.append(("STATE", q, exp, got))

    print(f"上下文意图测试: 总计 {total}, 通过 {ok}, 失败 {len(fails)}, 准确率 {ok/total*100:.1f}%")
    if fails:
        print("\n=== 失败 ===")
        for kind, q, e, g in fails:
            print(f"  [{kind}] Q: {q}\n     期望={e}\n     实际={g}")
        sys.exit(1)
    else:
        print("全部通过 ✅")


if __name__ == "__main__":
    main()
