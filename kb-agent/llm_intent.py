# -*- coding: utf-8 -*-
"""llm_intent.py —— L2 LLM 意图分类（低置信区决策）

铁律：
1. 只在「规则层 + 上下文层 + 语义层高置信」都未命中时触发（need_llm_intent 门控）
2. LLM 只做意图分类（I1~I9 之一），不做检索、不生成答案
3. 降级链：LLM 失败/低置信 → None → 上层回退 L3_low 结果

复用 llm_parser 的 provider 层模式（Ollama 本地 qwen2.5:3b / mock 测试）。
"""
import json
import re
import sys

from config import LLM, INTENTS
from llm_parser import ZhipuParser, MockParser

# 意图分类 prompt 用的意图清单（与 config.INTENTS 对齐，排除 invalid）
INTENT_LIST = [i for i in INTENTS if i not in ("invalid",)]

INTENT_SCHEMA = {"intent": str, "confidence": (int, float), "reason": str}

SYSTEM_PROMPT = """你是意图分类器。把用户问题归入唯一一个意图类别，输出严格 JSON：
{"intent": "<类别>", "confidence": 0.0~1.0, "reason": "一句话说明依据"}

意图类别（必须用这些英文值）：
- semantic_retrieval: 语义检索/内容了解（"讲讲/介绍/是什么/怎么用/方法/原理"）
- attribute_filter: 属性筛选（年份/作者/期刊/语言/类型 过滤）
- inventory: 全库盘点/枚举/统计（"有哪些/几篇/列出/统计/数量"）
- doc_qa: 指定文档的内容问答（定位到某篇论文 + 问内容/方法/摘要）
- cross_doc_synthesis: 多文档对比/综合（"区别/对比/有什么不同"）
- citation_query: 参考文献查询（"引用了哪些/参考文献列表"）
- metadata_query: 文档属性查询（作者/期刊/年份/页数等元数据）
- hybrid: 混合（属性筛选 + 语义检索，如"2024年关于贝叶斯的"）
- numerical_calculation: 已有数值的公式计算（增长率/波动率/差值）
- agent_workflow: 需要按依赖顺序调用日历、市场数据或 MCP 的多步任务
- non_kb: 非知识库任务（写代码/翻译/操作/闲聊）
- clarification: 信息不足需澄清（指代不明/歧义）

判定要点：
- 有明确时间/作者/期刊 + 问"有哪些/几篇" → attribute_filter 或 inventory
- 有明确时间/属性 + 问内容/方法/关于X → hybrid
- 无任何属性纯问内容 → semantic_retrieval
- 只输出 JSON，不要解释、不要 markdown 代码块"""

EXAMPLES = [
    {"user": "GAN怎么处理时间序列", "assistant": {"intent": "semantic_retrieval", "confidence": 0.9,
      "reason": "问方法，无属性过滤"}},
    {"user": "墨西湖城地震数据出现在哪篇论文", "assistant": {"intent": "semantic_retrieval", "confidence": 0.75,
      "reason": "按内容定位文档"}},
    {"user": "2024年关于贝叶斯的论文有哪些", "assistant": {"intent": "hybrid", "confidence": 0.95,
      "reason": "年份+主题"}},
]


def build_intent_messages(query: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in EXAMPLES:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["assistant"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": query})
    return msgs


def need_llm_intent(q: str) -> bool:
    """门控：什么查询值得走 LLM 意图分类？
    触发 = 语义模糊信号（无强规则意图词，句长 ≥6，非 trivial）
    由 router 在规则层/上下文层未命中后调用（本函数只做最后一道过滤）。
    """
    q = q.strip()
    if len(q) < 6:
        return False                       # 短句 → 上下文层职责
    if len(q) > 120:
        return False                       # 超长 → 结构解析走 llm_parser，不重复
    # 有强规则信号的（在 router 已被 L0 拦截，这里兜底防误触发）
    from nlu import ENUM_WORDS, STAT_WORDS, META_WORDS, COMPARE_WORDS, CITATION_WORDS, NON_KB_PATTERNS
    strong = (ENUM_WORDS + STAT_WORDS + META_WORDS + COMPARE_WORDS + CITATION_WORDS)
    if any(w in q for w in strong):
        return False
    # 年份/时间强信号（规则层能解：含 4位年份 / 两位年份+年 / 月份 / 相对时间）
    if re.search(r"(?:19|20)\d{2}\s*年|(?<!\d)\d{2}\s*年|\d{1,2}月|今天|明天|最近|去年|今年|明年|非典|疫情|金融危机", q):
        return False
    # 非知识库模式（写代码/翻译等）
    if any(re.search(p, q, re.I) for p in NON_KB_PATTERNS):
        return False
    # 纯语义表达（无属性无枚举）→ 值得 LLM 细判
    return True


class IntentResult:
    __slots__ = ("intent", "confidence", "reason")

    def __init__(self, intent: str, confidence: float, reason: str = ""):
        self.intent = intent
        self.confidence = confidence
        self.reason = reason

    def to_dict(self):
        return {"intent": self.intent, "confidence": round(self.confidence, 3),
                "reason": self.reason}


class LLMIntentClassifier:
    """意图分类器（provider 可插拔：ollama/zhipu/mock/none）"""

    def __init__(self, config: dict = None, provider=None):
        cfg = config or LLM
        self.cfg = cfg
        self.conf_threshold = cfg.get("conf_threshold", 0.6)
        p = provider or cfg.get("provider", "openai")
        if p == "mock":
            self._prov = MockParser(cfg.get("mock_response", "{}"))
        elif p in ("zhipu", "openai"):
            if not cfg.get("api_key"):
                raise RuntimeError("LLM 意图分类未配置 api_key")
            self._prov = ZhipuParser(cfg["api_key"], cfg.get("base_url"), cfg.get("model"))
        else:
            raise RuntimeError(f"LLM provider 未启用: {p}")

    def _parse_response(self, content: str):
        """严格解析：JSON + schema 校验 + 意图合法 + 置信度范围"""
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        intent = obj.get("intent")
        conf = obj.get("confidence")
        if not isinstance(intent, str) or intent not in INTENTS:
            return None
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            return None
        reason = obj.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        return IntentResult(intent, float(conf), reason)

    def classify(self, query: str):
        """→ IntentResult 或 None（降级信号）"""
        try:
            content = self._prov.chat(build_intent_messages(query), temperature=0.1)
            r = self._parse_response(content)
        except Exception:
            return None
        if r is None or r.confidence < self.conf_threshold:
            return None
        return r


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 本地模式：真实调 Ollama
    try:
        clf = LLMIntentClassifier()
        for q in ["GAN怎么处理时间序列", "墨西湖城地震数据出现在哪篇论文",
                  "信息量最大的文献是哪篇", "有没有关于降维的"]:
            r = clf.classify(q)
            print(f"Q: {q}\n  → {r.to_dict() if r else None}")
    except RuntimeError as e:
        print(f"LLM 不可用: {e}")
