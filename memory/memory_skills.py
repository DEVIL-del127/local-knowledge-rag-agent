# memory_skills.py - 记忆系统 Skills（M2/M3，对应《任务拆分 v1.1》§3）
# memory_recall / memory_summarize / memory_manage
# 全部实现 SkillProtocol, 可注册进 SkillRegistry
from __future__ import annotations

from typing import Any, Mapping

from agent.agent_skills import SkillSpec
from memory.memory_manager import MemoryManager


class MemoryRecallSkill:
    """会话内/跨会话记忆召回(scope=current|cross)"""

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.spec = SkillSpec(
            name="memory_recall",
            description=(
                "Recall user memory: current-session facts (scope=current) or "
                "cross-session long-term memory (scope=cross)."
            ),
            kind="local",
            tags=["memory", "recall"],
            user_invocable=True,
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        scope = str(arguments.get("scope", "current")).strip()
        user_id = str(arguments.get("user_id", "default")).strip()
        session_id = str(arguments.get("session_id", "")).strip()
        if not query:
            raise ValueError("memory_recall 需要 query 参数")

        if scope == "cross":
            injection = self.memory.recall_memory(query=query, user_id=user_id)
            channel = "long-term"
        else:
            injection = self.memory.recall_for_current(
                query=query,
                window_messages=[],
                session_id=session_id or "none",
                user_id=user_id,
                has_anaphora_hint=True,
            )
            channel = "session"

        if not injection:
            return {"query": query, "channel": channel, "found": False, "injection": ""}
        return {"query": query, "channel": channel, "found": True, "injection": injection}


class MemorySummarizeSkill:
    """会话摘要(可被用户触发: "总结一下我们聊的")"""

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.spec = SkillSpec(
            name="memory_summarize",
            description=(
                "Summarize the current session into a four-part digest "
                "(tasks/conclusions/preferences/todos)."
            ),
            kind="local",
            tags=["memory", "summary"],
            user_invocable=True,
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(arguments.get("session_id", "")).strip()
        session = self.memory._load_session(session_id)
        if not session:
            return {"error": f"会话不存在: {session_id}"}
        summary = self.memory._summarize_session(session)
        if not summary:
            return {"error": "摘要生成失败(LLM 不可用?)"}
        return {"session_id": session_id, "summary": summary}


class MemoryManageSkill:
    """记忆治理(用户对话式: 查看/删除/修正/统计)"""

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.spec = SkillSpec(
            name="memory_manage",
            description=(
                "Manage user memory: list/get/delete/fix/export/clear/stats. "
                "Write operations require user confirmation."
            ),
            kind="local",
            tags=["memory", "admin"],
            user_invocable=True,
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "")).strip()
        user_id = str(arguments.get("user_id", "default")).strip()
        if command not in {"list", "get", "delete", "fix", "export", "clear", "stats", "archive_stale"}:
            return {"error": f"未知命令: {command}"}
        return self.memory.admin(command=command, user_id=user_id, args=arguments)
