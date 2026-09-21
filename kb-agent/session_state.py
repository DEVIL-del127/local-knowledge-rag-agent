# -*- coding: utf-8 -*-
"""L3 上下文意图：会话状态对象 + 短句/指代意图延续

设计：
- SessionState：{last_intent, last_entities, 时间延续, 指代池}，一次路由后更新
- continue_intent(q, state)：短句且含延续信号 → 继承上一意图；含重置信号 → 重置
- 延续只在「短句 + 无实质检索内容」时触发；有实体/年份/主题词 → 走正常解析（防误继承）

纯函数，不依赖 ES/模型；router 在规则层未命中时调用。
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# 延续信号（短句 + 这些词 → 继承上一意图）
CONTINUE_WORDS = [
    "还有呢", "还有吗", "接着", "继续", "然后呢", "往下", "再多点",
    "别的呢", "其他呢", "那篇呢", "那个呢", "这篇呢", "这个呢",
    "还有哪些", "还有别的", "还有其他的",
]
# 重置信号（出现 → 清上下文，走正常解析）
RESET_WORDS = [
    "换个话题", "不看了", "不要了", "算了", "停", "重新来", "从头来",
    "别的", "另外再", "先别", "不是这个",
]
# 实质内容信号（出现 → 不延续，是新的独立问题）
SUBSTANTIVE = r"(?:19|20)\d{2}|年|月|作者|期刊|论文|文献|贝叶斯|GAN|ESN|回声|扩散|插补|降维|遥测|地震|储层|元启发|LSTM|MCMC|谁|什么方法|如何|怎么"


@dataclass
class SessionState:
    last_intent: Optional[str] = None
    last_entities: Dict = field(default_factory=dict)
    last_query: str = ""
    turn: int = 0
    has_context: bool = False

    def __post_init__(self):
        # 构造时若给了意图/查询 → 视为已有上下文（测试与真实场景一致）
        if self.last_intent or self.last_query:
            self.has_context = True

    def update(self, intent: Optional[str], entities: Dict, query: str):
        """一次路由后更新状态。clarification/invalid/non_kb 不污染上下文。"""
        if intent in ("clarification", "invalid", "non_kb"):
            return
        self.last_intent = intent
        self.last_entities = dict(entities)
        self.last_query = query
        self.turn += 1
        self.has_context = True

    def reset(self):
        self.last_intent = None
        self.last_entities = {}
        self.last_query = ""
        self.has_context = False


def continue_intent(q: str, state: SessionState):
    """短句 + 延续信号 → (True, 继承意图, 说明)；重置信号 → (False, None, 说明)；否则 None

    返回 None 表示不适用上下文（走正常解析）。
    """
    if not state or not state.has_context or not state.last_intent:
        return None
    q = q.strip()
    # 重置信号优先
    for w in RESET_WORDS:
        if q == w or q.startswith(w):
            state.reset()
            return (False, None, f"重置上下文: {w}")
    # 短句 + 延续信号 → 继承
    if len(q) <= 8:
        for w in CONTINUE_WORDS:
            if q == w or q.startswith(w):
                return (True, state.last_intent, f"延续上一意图: {w}")
    # 含实质内容 → 不延续（新问题）
    if re.search(SUBSTANTIVE, q):
        return None
    # 短句无延续词但像指代 → 不强行继承（避免误判）
    return None


def apply_context(q: str, state: SessionState):
    """router 集成：返回 (intent, entities, source, reason) 或 None"""
    r = continue_intent(q, state)
    if not r or not r[0]:
        return None
    return (r[1], dict(state.last_entities), "L3_ctx", r[2])
