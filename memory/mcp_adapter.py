# mcp_adapter.py - 记忆系统 MCP 适配层（对外暴露 2 个能力工具）
# 对应《任务拆分 v1.1》§5: 只暴露 memory.search / memory.admin
# 本模块提供工具定义(JSON Schema)与执行函数, 供 MCP server 注册;
# Agent 内部不走 MCP(直接调 MemoryManager, 避免绕路)
from __future__ import annotations

from typing import Any

from memory.memory_manager import MemoryManager

# ---------- 工具定义(JSON Schema, MCP tools 格式) ----------
MEMORY_TOOL_DEFINITIONS = [
    {
        "name": "memory.search",
        "description": "Search a user's long-term memory (facts and session summaries). "
                       "Returns matched entries with source session and timestamp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "user_id": {"type": "string", "description": "owner of the memory"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query", "user_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory.admin",
        "description": "Manage a user's memory: list / stats / delete / fix / export / clear / archive_stale. "
                       "Write operations require caller-side confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list", "stats", "delete", "fix", "export", "clear", "archive_stale"],
                },
                "user_id": {"type": "string"},
                "id": {"type": "string", "description": "memory entry id (delete/fix)"},
                "value": {"type": "string", "description": "new value (fix)"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "required": ["command", "user_id"],
            "additionalProperties": False,
        },
    },
]


class MemoryMcpAdapter:
    """MCP 适配器: 包装 MemoryManager, 提供工具执行入口"""

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory

    def tool_definitions(self) -> list[dict[str, Any]]:
        return MEMORY_TOOL_DEFINITIONS

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """按工具名分发执行(供 MCP server 的 call_tool 回调使用)"""
        if name == "memory.search":
            query = str(arguments.get("query", "")).strip()
            user_id = str(arguments.get("user_id", "")).strip()
            if not query or not user_id:
                return {"error": "query 和 user_id 必填"}
            injection = self.memory.recall_memory(query=query, user_id=user_id)
            return {"found": bool(injection), "injection": injection}

        if name == "memory.admin":
            command = str(arguments.get("command", "")).strip()
            user_id = str(arguments.get("user_id", "")).strip()
            if not user_id:
                return {"error": "user_id 必填"}
            return self.memory.admin(command=command, user_id=user_id, args=arguments)

        return {"error": f"未知工具: {name}"}


def register_to_mcp_server(memory: MemoryManager, server) -> None:
    """示例: 把记忆工具注册到 MCP server 实例(以官方 mcp SDK 为例)

    from mcp.server import Server
    server = Server("memory-server")
    adapter = MemoryMcpAdapter(memory)

    for tool in adapter.tool_definitions():
        @server.list_tools()
        async def _list(): return [tool 转换后的 Tool 对象]
        @server.call_tool()
        async def _call(name, arguments): return adapter.execute(name, arguments)

    实际接入时按所用 MCP 框架(官方 SDK / fastmcp)的注册方式适配。
    """
    adapter = MemoryMcpAdapter(memory)
    add_tool = getattr(server, "add_tool", None)
    if not callable(add_tool):
        raise TypeError("MCP server must provide add_tool(callable, name=..., description=...)")

    def search(query: str, user_id: str, top_k: int = 5):
        return adapter.execute(
            "memory.search", {"query": query, "user_id": user_id, "top_k": top_k}
        )

    def admin(command: str, user_id: str, **kwargs):
        return adapter.execute(
            "memory.admin", {"command": command, "user_id": user_id, **kwargs}
        )

    add_tool(search, name="memory.search", description=MEMORY_TOOL_DEFINITIONS[0]["description"])
    add_tool(admin, name="memory.admin", description=MEMORY_TOOL_DEFINITIONS[1]["description"])
