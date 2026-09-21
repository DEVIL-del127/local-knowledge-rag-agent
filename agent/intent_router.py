from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from agent.agent_models import (
    IntentPrediction,
    IntentType,
    build_router_tool_schema,
    coerce_intent_prediction,
)
from agent.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

GREETING_WORDS = {
    "hi", "hello", "你好", "您好", "在吗", "嗨", "哈喽", "早", "晚上好",
}

CHAT_HELP_HINTS = (
    "你能做什么", "怎么用你", "使用说明", "帮助", "介绍一下你自己",
    "你的能力", "你会什么", "有哪些功能", "怎么用", "你是谁",
)

ANALYSIS_HINTS = (
    "总结", "概括", "归纳", "梳理", "综述", "复习", "回顾",
    "对比", "比较", "区别", "差异", "优缺点", "分析", "评价",
    "提炼", "要点", "重点", "框架", "大纲", "结构", "知识点",
    "结论", "技术路线", "研究方法",
)

FIND_HINTS = (
    "找", "有没有", "哪篇", "哪个文档", "哪本书", "文档在哪",
    "资料在哪", "论文在哪", "文件名", "哪些文档",
)
# 文档类名词: 与 FIND_HINTS 同时出现时优先判为找文档
DOC_NOUNS = ("文档", "论文", "资料", "文件", "书", "教材", "笔记")

SEARCH_HINTS = (
    "查", "检索", "搜索", "是什么", "什么是", "解释", "介绍一下",
    "了解", "含义", "定义", "原理", "机制", "为什么",
    "怎么用", "知道", "参考",
)

TUTOR_HINTS = (
    "给我讲讲", "讲讲", "教我", "教教", "入门", "科普", "我不懂",
    "我不了解", "不了解", "从零", "怎么理解", "通俗", "简单讲",
    "初学者", "想学", "学习一下", "讲解一下", "解释一下",
)

# 文档库管理
KB_MANAGE_HINTS = (
    "收录", "添加文档", "新增文档", "导入文档", "删除文档", "移除文档",
    "清理文档", "整理文档", "重命名", "更新索引", "重建索引", "刷新索引",
    "库里有", "文档列表", "有哪些文档", "多少篇", "文档管理", "管理文档",
    "文档统计", "库里有什么", "删掉这篇", "收录这篇", "文档库", "统计一下",
)
# 记忆相关: 跨会话召回(用户显式触发)
MEMORY_RECALL_HINTS = (
    "还记得",
    "上次说的",
    "上次提",
    "之前说过",
    "之前聊过",
    "之前提过",
    "上次聊的",
    "你记得",
    "我们之前",
)
# 记忆治理
MEMORY_MANAGE_HINTS = (
    "记忆里",
    "删除记忆",
    "清除记忆",
    "管理记忆",
    "我的记忆",
    "修一下记忆",
    "改一下记忆",
)

# 否定词: 关键词前窗口内出现则抑制该关键词触发的意图
NEGATION_WORDS = ("不", "别", "没", "无需", "不用", "不要", "无须", "无", "非", "别")
NEGATION_WINDOW = 5


def _has_negation_before(message: str, index: int) -> bool:
    """检查关键词前 N 个字符内是否有否定词(如"我不需要你总结"不触发分析意图)"""
    start = max(0, index - NEGATION_WINDOW)
    segment = message[start:index]
    return any(word in segment for word in NEGATION_WORDS)


class IntentRecognizer:
    def __init__(self, llm_client: DeepSeekClient):
        self.llm_client = llm_client

    def recognize(
        self,
        user_message: str,
        available_skills: Sequence[Mapping[str, Any]],
        *,
        history: Sequence[Mapping[str, str]] | None = None,
        user_id: str = "default",
    ) -> IntentPrediction:
        compact_message = user_message.strip()
        rule_based = self._match_fast_path(compact_message, available_skills)
        if rule_based is not None:
            return rule_based

        tool_schema = build_router_tool_schema(
            available_skills,
            strict=self.llm_client.settings.router_strict,
        )
        router_messages = self._build_router_messages(compact_message, history)

        try:
            payload = self.llm_client.invoke_router(
                messages=router_messages,
                tool_schema=tool_schema,
                user_id=user_id,
            )
            return coerce_intent_prediction(
                payload,
                allowed_skills=[item["name"] for item in available_skills],
                fallback_query=compact_message,
                router_source="llm",
            )
        except Exception as exc:
            logger.warning("LLM 意图识别失败，降级为关键词路由: %s", exc)
            return self._fallback_search(compact_message, available_skills)

    def _match_fast_path(
        self,
        message: str,
        available_skills: Sequence[Mapping[str, Any]],
    ) -> IntentPrediction | None:
        if not message:
            return IntentPrediction(
                intent=IntentType.CLARIFY,
                needs_retrieval=False,
                confidence=1.0,
                user_goal="用户尚未提供有效问题",
                clarification_question="你想让我帮你查什么资料，或者解决什么问题？",
                router_source="rule",
                reason="empty_input",
            )

        lowered = message.lower()
        if lowered in GREETING_WORDS or message in GREETING_WORDS:
            return IntentPrediction(
                intent=IntentType.CHAT_DAILY,
                needs_retrieval=False,
                confidence=0.95,
                user_goal="问候或闲聊",
                router_source="rule",
                reason="greeting",
            )

        if len(message) <= 2:
            return IntentPrediction(
                intent=IntentType.CLARIFY,
                needs_retrieval=False,
                confidence=0.9,
                user_goal=message,
                clarification_question="这个问题有点短，我先帮你查资料的话可能会偏。你可以再补充一点关键词或目标吗？",
                router_source="rule",
                reason="message_too_short",
            )

        if any(token in message for token in CHAT_HELP_HINTS):
            return IntentPrediction(
                intent=IntentType.CHAT_HELP,
                needs_retrieval=False,
                confidence=0.9,
                user_goal=message,
                router_source="rule",
                reason="assistant_meta_question",
            )

        # 文档库管理(优先级高: "收录/删除/库里有哪些")
        for token in KB_MANAGE_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index):
                return self._build_kb_manage_intent(message, available_skills, reason="kb_manage_keyword")

        # 记忆意图
        for token in MEMORY_MANAGE_HINTS:
            if token in message:
                return self._build_memory_intent(
                    message, available_skills,
                    intent=IntentType.MEMORY_MANAGE,
                    reason="memory_manage_keyword",
                )
        for token in MEMORY_RECALL_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index):
                return self._build_memory_intent(
                    message, available_skills,
                    intent=IntentType.MEMORY_RECALL,
                    reason="memory_recall_keyword",
                )

        # 内容梳理/分析("总结/对比/梳理")
        for token in ANALYSIS_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index):
                return self._build_default_retrieval_intent(
                    message,
                    available_skills,
                    intent=IntentType.KB_ANALYSIS,
                    reason="analysis_keyword",
                )

        # 教学讲解("给我讲讲/我不懂/入门")
        for token in TUTOR_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index):
                return self._build_default_retrieval_intent(
                    message,
                    available_skills,
                    intent=IntentType.KB_TUTOR,
                    reason="tutor_keyword",
                )

        # 找文档(含文档类名词): "有没有X的论文/哪篇文档"
        has_doc_noun = any(n in message for n in DOC_NOUNS)
        for token in FIND_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index) and has_doc_noun:
                return self._build_default_retrieval_intent(
                    message,
                    available_skills,
                    intent=IntentType.KB_FIND,
                    reason="find_document_keyword",
                )

        # 知识检索问答("X是什么/查一下X")
        for token in SEARCH_HINTS:
            index = message.find(token)
            if index >= 0 and not _has_negation_before(message, index):
                return self._build_default_retrieval_intent(
                    message,
                    available_skills,
                    intent=IntentType.KB_SEARCH,
                    reason="search_keyword",
                )

        return None

    @staticmethod
    def _build_kb_manage_intent(
        message: str,
        available_skills: Sequence[Mapping[str, Any]],
        *,
        reason: str,
    ) -> IntentPrediction:
        """文档库管理: 由 kb_manage 技能确定性执行"""
        skill_calls = []
        for skill in available_skills:
            if skill.get("name") == "kb_manage":
                skill_calls.append({
                    "skill_name": "kb_manage",
                    "purpose": "manage knowledge base documents",
                    "arguments": {"command": "auto", "query": message},
                })
        return coerce_intent_prediction(
            {
                "intent": IntentType.KB_MANAGE.value,
                "needs_retrieval": False,
                "confidence": 0.85,
                "user_goal": message,
                "rewritten_query": message,
                "top_k": 5,
                "response_style": "concise",
                "clarification_question": "",
                "reason": reason,
                "metadata_filters": {},
                "skill_calls": skill_calls,
                "has_anaphora": False,
            },
            allowed_skills=[item["name"] for item in available_skills],
            fallback_query=message,
            router_source="rule",
        )

    @staticmethod
    def _build_memory_intent(
        message: str,
        available_skills: Sequence[Mapping[str, Any]],
        *,
        intent: IntentType,
        reason: str,
    ) -> IntentPrediction:
        """记忆类意图: 不需要知识库检索, 直接由记忆技能处理"""
        skill_calls = []
        for skill in available_skills:
            if skill.get("name") in {"memory_recall", "memory_manage"}:
                scope = "cross" if intent == IntentType.MEMORY_RECALL else None
                args = {"query": message}
                if scope:
                    args["scope"] = scope
                skill_calls.append({
                    "skill_name": skill["name"],
                    "purpose": "handle memory request",
                    "arguments": args,
                })
        return coerce_intent_prediction(
            {
                "intent": intent.value,
                "needs_retrieval": False,
                "confidence": 0.8,
                "user_goal": message,
                "rewritten_query": message,
                "top_k": 5,
                "response_style": "grounded",
                "clarification_question": "",
                "reason": reason,
                "metadata_filters": {},
                "skill_calls": skill_calls,
                "has_anaphora": True,
            },
            allowed_skills=[item["name"] for item in available_skills],
            fallback_query=message,
            router_source="rule",
        )

    def _fallback_search(
        self,
        message: str,
        available_skills: Sequence[Mapping[str, Any]],
    ) -> IntentPrediction:
        return self._build_default_retrieval_intent(
            message,
            available_skills,
            intent=IntentType.KB_SEARCH,
            reason="fallback_search",
        )

    @staticmethod
    def _build_router_messages(
        message: str,
        history: Sequence[Mapping[str, str]] | None,
    ) -> list[dict[str, str]]:
        router_prompt = (
            "你是一个私有知识库 Agent 的意图路由器。\n"
            "你的任务不是回答问题，而是判断：\n"
            "1. 当前问题是否需要检索私有知识库；\n"
            "2. 属于搜索、分析、闲聊、澄清还是未来工作流请求；\n"
            "3. 如果需要检索，应选择哪些内部 skills，并重写查询语句。\n"
            "输出必须严格遵守提供的函数参数 schema。"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": router_prompt}]
        for item in history or []:
            role = str(item.get("role", "user"))
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _build_default_retrieval_intent(
        message: str,
        available_skills: Sequence[Mapping[str, Any]],
        *,
        intent: IntentType,
        reason: str,
    ) -> IntentPrediction:
        skill_calls = []
        if available_skills:
            skill_calls.append(
                {
                    "skill_name": available_skills[0]["name"],
                    "purpose": "retrieve evidence from the private knowledge base",
                    "arguments": {"query": message, "top_k": 5},
                }
            )
        return coerce_intent_prediction(
            {
                "intent": intent.value,
                "needs_retrieval": True,
                "confidence": 0.75,
                "user_goal": message,
                "rewritten_query": message,
                "top_k": 5,
                "response_style": (
                    "tutorial" if intent == IntentType.KB_TUTOR else "grounded"
                ),
                "clarification_question": "",
                "reason": reason,
                "metadata_filters": {},
                "skill_calls": skill_calls,
            },
            allowed_skills=[item["name"] for item in available_skills],
            fallback_query=message,
            router_source="rule",
        )
