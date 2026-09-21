# -*- coding: utf-8 -*-
"""llm_parser.py —— 复杂问题 LLM 结构解析（默认 Provider：智谱 GLM 免费 API）

铁律：LLM 只做问题结构解析（NLU），不做检索、不生成答案。
触发：need_llm() 复杂度检测（子句数 / 输出·约束角色 / 事件时间 / 括号时间 / 长度）
降级：解析失败 / schema 校验不过 / 置信度低 → 返回 None → 上层退回规则结果。
无 api_key 或 provider=none → 完全禁用（系统退回纯规则，现状零破坏）。

Provider 可插拔：zhipu（默认免费）/ openai兼容 / mock（测试用固定 JSON）
"""
import json
import re
import sys
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from config import LLM


@dataclass
class ComplexStructure:
    """LLM 解析结果（严格 schema，校验后使用）"""
    main_query: str = ""
    time_range: Dict = field(default_factory=dict)
    time_ranges: List[Dict] = field(default_factory=list)  # 复合时间序列（可选）
    constraints: List[Dict] = field(default_factory=list)
    output: Dict = field(default_factory=dict)
    confidence: float = 0.0
    note: str = ""

    def to_dict(self):
        return {
            "main_query": self.main_query, "time_range": self.time_range,
            "time_ranges": self.time_ranges,
            "constraints": self.constraints, "output": self.output,
            "confidence": round(self.confidence, 2), "note": self.note,
        }


# ================= 复杂度检测（何时启用 LLM） =================

def need_llm(q: str) -> bool:
    """五信号或门：命中任一 → 启用 LLM 结构解析"""
    # S1 子句数量（不依赖长度，独立检测）
    try:
        from splitter import rule_split
        segs = rule_split(q)
        if len(segs) >= LLM["trigger"]["min_subqueries"]:
            return True
    except Exception:
        pass
    # S5 长度兜底
    if len(q) > LLM["trigger"]["max_query_len"]:
        return True
    # S2 输出/约束角色词（复用 splitter 模式）
    from splitter import OUTPUT_PATTERN, CONSTRAINT_PATTERN, ROLE_PREFIX
    if re.search(OUTPUT_PATTERN, q):
        return True
    if re.search(CONSTRAINT_PATTERN, q) and re.search(ROLE_PREFIX, q):
        return True
    # S3 事件锚点时间（"从X发布到Y发布期间"）
    if re.search(r"从.{1,25}(发布|推出|上线|开始).{0,10}(到|至).{1,25}(发布|推出|上线)", q):
        return True
    # S4 括号时间 / 多时间表达未合并 / 括号多年份列表
    if re.search(r"[（(]\s*((19|20)\d{2})?\s*年?\s*[^）)]{0,8}(初|底|月|年|季)[^）)]{0,8}[）)]", q):
        return True
    if re.search(r"[（(]\s*(?:19|20)\d{2}\s*[,，]\s*(?:19|20)\d{2}", q):
        return True
    return False


# ================= Prompt =================

SYSTEM_PROMPT = """你是问题结构解析器。把用户的问题解析为 JSON，严格遵守：
1. main_query：核心查询（唯一一个，去掉时间限定/约束/输出要求后的主干）
2. time_range：主时间范围 {from, to, granularity, note}；事件锚点（如"GPT-2发布到GPT-4发布期间"）写进 note，不要硬转区间；to 可为 null（至今）
3. time_ranges：复合时间问题的多区间数组 [{"from", "to", "granularity", "note"}]，每个时间段标注事件说明（如"疫情三年期间"）；事件锚点给合理区间估计并注明"估计"；单一时间范围时可为空数组
4. constraints：过滤/限定条件数组 [{"type": "exclude_topic|include_entity|...", "value": "..."}]
5. output：输出要求 {"group_by": "year|quarter|...", "format": "..."}；没有则 {}
6. confidence：你对解析的置信度 0~1
7. 只输出 JSON 对象，不要任何解释、不要 markdown 代码块"""

EXAMPLES = [
    {
        "user": "我想研究一下十四五规划期间，特别是疫情放开之后（2023年初）到今年年底，针对中小企业的税收优惠和贷款扶持政策发生了哪些变化，需要按季度展示，并且把国资委和工信部的发文分开列",
        "assistant": {
            "main_query": "针对中小企业的税收优惠和贷款扶持政策发生了哪些变化",
            "time_range": {"from": "2023-01-01", "to": "2026-12-31", "granularity": "quarter",
                           "note": "十四五规划期间为背景；2023年初到今年年底"},
            "constraints": [{"type": "include_entity", "value": "中小企业"}],
            "output": {"group_by": "quarter", "split_by": ["国资委", "工信部"]},
            "confidence": 0.85,
        },
    },
]


def build_messages(query: str) -> List[Dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["assistant"], ensure_ascii=False)})
    messages.append({"role": "user", "content": query})
    return messages


# ================= Provider 层 =================

class ZhipuParser:
    """智谱 GLM 免费 API（OpenAI 兼容格式）"""

    def __init__(self, api_key: str, base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = (base_url or LLM["base_url"]).rstrip("/")
        self.model = model or LLM["model"]

    def chat(self, messages: List[Dict], temperature: float = 0.1, max_retries: int = 3) -> str:
        import time
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": 1024,
                   "response_format": {"type": "json_object"},
                   "think": False,  # 关闭推理模型的思考模式（Ollama，提速）
                   "options": {"num_predict": 1024}}  # Ollama 透传
        for attempt in range(max_retries):
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=180,  # 冷启动加载模型可能 >60s，给足余量
            )
            if r.status_code == 429:  # 免费模型限流 → 指数退避重试
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"GLM API {r.status_code}: {r.text[:200]}")
            content = r.json()["choices"][0]["message"]["content"]
            if not content:  # 空响应 → 重试
                time.sleep(2 ** attempt * 2)
                continue
            return content
        raise RuntimeError("GLM API 限流/空响应，重试后仍失败（免费模型高峰期，请稍后再试）")


class MockParser:
    """测试用：返回固定 JSON"""

    def __init__(self, fixed: str):
        self.fixed = fixed

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        return self.fixed


# ================= 解析器（校验 + 降级） =================

class LLMParser:
    def __init__(self, config: Dict = None, provider=None):
        cfg = config or LLM
        self.cfg = cfg
        p = provider or cfg.get("provider", "zhipu")
        if p == "mock":
            self._prov = MockParser(cfg.get("mock_response", "{}"))
        elif p in ("zhipu", "openai"):
            if not cfg.get("api_key"):
                raise RuntimeError("未配置 api_key（智谱免费 key：https://open.bigmodel.cn 注册获取）")
            self._prov = ZhipuParser(cfg["api_key"], cfg.get("base_url"), cfg.get("model"))
        else:  # none / 未知
            raise RuntimeError(f"LLM provider 未启用或未知: {p}")

    @staticmethod
    def _validate(obj) -> bool:
        if not isinstance(obj, dict):
            return False
        if not obj.get("main_query") or not isinstance(obj["main_query"], str):
            return False
        tr = obj.get("time_range") or {}
        if not isinstance(tr, dict):
            return False
        for k in ("from", "to", "granularity", "note"):
            if k in tr and tr[k] is not None and not isinstance(tr[k], str):
                return False
        if not isinstance(obj.get("constraints") or [], list):
            return False
        if not isinstance(obj.get("output") or {}, dict):
            return False
        trs = obj.get("time_ranges") or []
        if not isinstance(trs, list):
            return False
        for tr in trs:
            if not isinstance(tr, dict) or not all(isinstance(tr.get(k), (str, type(None))) for k in ("from", "to", "granularity", "note")):
                return False
        conf = obj.get("confidence", 0)
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            return False
        return True

    def parse(self, query: str) -> Optional[ComplexStructure]:
        """LLM 结构解析。失败/低置信 → None（上层降级规则）。"""
        for attempt in range(self.cfg.get("max_retry", 1) + 1):
            try:
                raw = self._prov.chat(build_messages(query))
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```\w*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                obj = json.loads(raw)
            except Exception:
                continue
            if not self._validate(obj):
                continue
            if obj.get("confidence", 0) < self.cfg.get("conf_threshold", 0.6):
                return None
            return ComplexStructure(
                main_query=obj["main_query"],
                time_range=obj.get("time_range") or {},
                time_ranges=obj.get("time_ranges") or [],
                constraints=obj.get("constraints") or [],
                output=obj.get("output") or {},
                confidence=obj.get("confidence", 0.0),
                note=obj.get("time_range", {}).get("note", ""),
            )
        return None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 冒烟：mock 模式验证链路
    cfg = dict(LLM, provider="mock", conf_threshold=0.0,
               mock_response=json.dumps(EXAMPLES[0]["assistant"], ensure_ascii=False))
    p = LLMParser(cfg)
    res = p.parse("测试复杂问题")
    print("mock 解析:", json.dumps(res.to_dict(), ensure_ascii=False, indent=1) if res else "None")
    print("need_llm('找25年之前的文献'):", need_llm("找25年之前的文献"))
    print("need_llm(复杂问题):", need_llm("对比一下从GPT-2发布到GPT-4发布期间，以及在2023年3月之后到现在，大语言模型在推理能力和代码生成方面的主要技术突破，同时过滤掉那些只讲应用不讲原理的文章，并按年份整理出每年的里程碑事件"))
