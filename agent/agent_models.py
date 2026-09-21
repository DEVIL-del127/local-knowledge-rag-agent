from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class IntentType(str, Enum):
    # ===== 知识库场景 =====
    KB_SEARCH = "kb_search"          # 知识检索问答("X是什么"/"查一下X")
    KB_TUTOR = "kb_tutor"            # 教学讲解("给我讲讲X"/"我不懂X")
    KB_ANALYSIS = "kb_analysis"      # 内容梳理/分析/总结("总结X"/"对比X和Y")
    KB_FIND = "kb_find"              # 文档定位查找("有没有X的论文"/"哪篇文档")
    KB_MANAGE = "kb_manage"          # 文档库管理("库里有哪些文档"/"收录/删除")
    # ===== 日常 =====
    CHAT_DAILY = "chat_daily"        # 日常闲聊(问候/观点/情感)
    CHAT_HELP = "chat_help"          # 能力/使用询问("你能做什么")
    UTILITY_CALCULATE = "utility_calculate"  # 本地受限算术
    # ===== 记忆 =====
    MEMORY_RECALL = "memory_recall"  # 跨会话记忆召回(用户显式)
    MEMORY_MANAGE = "memory_manage"  # 记忆治理(查看/删除/修正)
    # ===== 兜底 =====
    CLARIFY = "clarify"              # 澄清
    UNSUPPORTED = "unsupported"      # 明确不支持/不允许的请求
    UNKNOWN = "unknown"              # 未知(降级为知识检索)


@dataclass(slots=True)
class SkillCall:
    skill_name: str
    purpose: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IntentPrediction:
    intent: IntentType
    needs_retrieval: bool
    confidence: float
    user_goal: str
    rewritten_query: str = ""
    top_k: int = 5
    response_style: str = "grounded"
    clarification_question: str = ""
    skill_calls: list[SkillCall] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    router_source: str = "llm"
    reason: str = ""
    has_anaphora: bool = False  # 消息是否含指代(LLM 路由顺带判断)

    def effective_query(self, fallback: str) -> str:
        query = (self.rewritten_query or "").strip()
        return query or fallback.strip()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["intent"] = self.intent.value
        return payload


@dataclass(slots=True)
class RetrievedEvidence:
    source: str
    snippet: str
    score: float | None = None
    page: int | None = None
    channel: str | None = None
    chunk: int | None = None

    def citation_label(self) -> str:
        label = self.source
        if self.page is not None:
            label += f"#p{self.page}"
        return label

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentReply:
    answer: str
    intent: IntentPrediction
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    executed_skills: list[dict[str, Any]] = field(default_factory=list)
    semantic_trace: list[dict[str, Any]] = field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentReply":
        value = dict(payload)
        intent = dict(value["intent"])
        intent["intent"] = IntentType(intent["intent"])
        intent["skill_calls"] = [SkillCall(**call) for call in intent.get("skill_calls", [])]
        value["intent"] = IntentPrediction(**intent)
        value["evidence"] = [RetrievedEvidence(**item) for item in value.get("evidence", [])]
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "answer": self.answer,
            "intent": self.intent.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "executed_skills": self.executed_skills,
        }
        if self.semantic_trace:
            payload["semantic_trace"] = self.semantic_trace
        if self.suggested_actions:
            payload["suggested_actions"] = self.suggested_actions
        return payload


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _clamp_float(value: Any, default: float = 0.6) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(parsed, 1.0))


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_like(text: str) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    if not cleaned:
        return {}
    return json.loads(cleaned)


def build_router_tool_schema(
    skill_specs: Sequence[Mapping[str, Any]],
    *,
    strict: bool,
) -> dict[str, Any]:
    skill_names = [str(item["name"]) for item in skill_specs]
    skill_lines = []
    for item in skill_specs:
        tags = ", ".join(item.get("tags", []))
        tag_hint = f" | tags: {tags}" if tags else ""
        skill_lines.append(f"- {item['name']}: {item['description']}{tag_hint}")
    skill_help = "\n".join(skill_lines) if skill_lines else "- no user-invocable skills available"

    function: dict[str, Any] = {
        "name": "route_intent",
        "description": (
            "Classify the user's request, decide whether retrieval is required, and "
            "select the next internal skills to execute.\n"
            f"Available skills:\n{skill_help}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        IntentType.KB_SEARCH.value,
                        IntentType.KB_TUTOR.value,
                        IntentType.KB_ANALYSIS.value,
                        IntentType.KB_FIND.value,
                        IntentType.KB_MANAGE.value,
                        IntentType.CHAT_DAILY.value,
                        IntentType.CHAT_HELP.value,
                        IntentType.UTILITY_CALCULATE.value,
                        IntentType.MEMORY_RECALL.value,
                        IntentType.MEMORY_MANAGE.value,
                        IntentType.CLARIFY.value,
                        IntentType.UNSUPPORTED.value,
                        IntentType.UNKNOWN.value,
                    ],
                },
                "has_anaphora": {"type": "boolean"},
                "needs_retrieval": {"type": "boolean"},
                "confidence": {"type": "number"},
                "user_goal": {"type": "string"},
                "rewritten_query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                "response_style": {
                    "type": "string",
                    "enum": ["grounded", "concise", "analysis", "step_by_step"],
                },
                "clarification_question": {"type": "string"},
                "reason": {"type": "string"},
                "metadata_filters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                "skill_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string", "enum": skill_names},
                            "purpose": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True,
                            },
                        },
                        "required": ["skill_name", "purpose", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "intent",
                "needs_retrieval",
                "confidence",
                "user_goal",
                "rewritten_query",
                "top_k",
                "response_style",
                "clarification_question",
                "reason",
                "metadata_filters",
                "skill_calls",
                "has_anaphora",
            ],
            "additionalProperties": False,
        },
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def coerce_intent_prediction(
    payload: Mapping[str, Any] | None,
    *,
    allowed_skills: Sequence[str],
    fallback_query: str,
    router_source: str,
) -> IntentPrediction:
    payload = payload or {}
    allowed_skill_set = set(allowed_skills)

    raw_intent = str(payload.get("intent", IntentType.UNKNOWN.value))
    try:
        intent = IntentType(raw_intent)
    except ValueError:
        intent = IntentType.UNKNOWN

    rewritten_query = str(payload.get("rewritten_query", "") or "").strip()
    top_k = _clamp_int(payload.get("top_k"), default=5, minimum=1, maximum=8)

    skill_calls: list[SkillCall] = []
    raw_skill_calls = payload.get("skill_calls", [])
    if isinstance(raw_skill_calls, list):
        for item in raw_skill_calls:
            if not isinstance(item, Mapping):
                continue
            skill_name = str(item.get("skill_name", "")).strip()
            if skill_name not in allowed_skill_set:
                continue
            arguments = dict(item.get("arguments") or {})
            if "query" not in arguments:
                arguments["query"] = rewritten_query or fallback_query
            if "top_k" not in arguments:
                arguments["top_k"] = top_k
            skill_calls.append(
                SkillCall(
                    skill_name=skill_name,
                    purpose=str(item.get("purpose", "retrieve evidence")).strip()
                    or "retrieve evidence",
                    arguments=arguments,
                )
            )

    needs_retrieval = bool(payload.get("needs_retrieval"))
    if needs_retrieval and not skill_calls and allowed_skills:
        skill_calls.append(
            SkillCall(
                skill_name=allowed_skills[0],
                purpose="retrieve evidence from the private knowledge base",
                arguments={"query": rewritten_query or fallback_query, "top_k": top_k},
            )
        )

    return IntentPrediction(
        intent=intent,
        needs_retrieval=needs_retrieval,
        confidence=_clamp_float(payload.get("confidence")),
        user_goal=str(payload.get("user_goal", "") or "").strip() or fallback_query.strip(),
        rewritten_query=rewritten_query,
        top_k=top_k,
        response_style=str(payload.get("response_style", "grounded") or "grounded"),
        clarification_question=str(payload.get("clarification_question", "") or "").strip(),
        skill_calls=skill_calls,
        metadata_filters=dict(payload.get("metadata_filters") or {}),
        router_source=router_source,
        reason=str(payload.get("reason", "") or "").strip(),
        has_anaphora=bool(payload.get("has_anaphora")),
    )
