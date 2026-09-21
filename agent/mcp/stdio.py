from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OfficialStdioTransport:
    """Synchronous boundary around the official MCP async stdio client."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def list_tools(self) -> list[dict[str, Any]]:
        return _run(self._list_tools())

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> Any:
        return _run(self._call_tool(name, arguments, timeout))

    async def _session(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env=dict(self.env) or None,
        )
        return stdio_client(parameters), ClientSession

    async def _list_tools(self) -> list[dict[str, Any]]:
        transport, session_type = await self._session()
        async with transport as (reader, writer):
            async with session_type(reader, writer) as session:
                await session.initialize()
                response = await session.list_tools()
                return [
                    {
                        "name": item.name,
                        "description": item.description or "",
                        "inputSchema": item.inputSchema,
                    }
                    for item in response.tools
                ]

    async def _call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> Any:
        transport, session_type = await self._session()
        async with transport as (reader, writer):
            async with session_type(reader, writer) as session:
                await session.initialize()
                response = await asyncio.wait_for(
                    session.call_tool(name, arguments=arguments), timeout=timeout
                )
                return {
                    "is_error": bool(getattr(response, "isError", False)),
                    "content": [
                        item.model_dump(mode="json")
                        if hasattr(item, "model_dump") else str(item)
                        for item in response.content
                    ],
                    "structured_content": getattr(response, "structuredContent", None),
                }


def _run(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("sync MCP transport cannot run inside an active asyncio loop")
