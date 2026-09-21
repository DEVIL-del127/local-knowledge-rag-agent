from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class McpError(RuntimeError):
    pass


class McpTransport(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float) -> Any: ...


@dataclass(slots=True)
class McpObservation:
    server: str
    tool: str
    ok: bool
    result: Any = None
    error: str = ""
    elapsed_ms: float = 0.0
    diagnostics: list[str] = field(default_factory=list)


class McpClient:
    """Transport-neutral, allow-listed MCP client."""

    def __init__(self, server_name: str, transport: McpTransport, *, allowed_tools: set[str]) -> None:
        self.server_name = server_name
        self.transport = transport
        self.allowed_tools = set(allowed_tools)
        self._input_schemas: dict[str, dict[str, Any]] = {}

    def freeze_schemas(self) -> None:
        advertised = list(self.transport.list_tools())
        by_name = {str(item.get("name")): item for item in advertised if item.get("name")}
        missing = sorted(self.allowed_tools - set(by_name))
        if missing:
            raise McpError(f"allow-listed MCP tools were not advertised: {missing}")
        self._input_schemas = {
            name: self._strict_schema(
                by_name[name].get("inputSchema") or by_name[name].get("input_schema") or {}
            )
            for name in self.allowed_tools
        }

    def input_schema(self, name: str) -> dict[str, Any]:
        if not self._input_schemas:
            self.freeze_schemas()
        if name not in self._input_schemas:
            raise PermissionError(f"MCP tool is not allow-listed: {name}")
        return dict(self._input_schemas[name])

    @staticmethod
    def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
        value = dict(schema or {})
        value.setdefault("type", "object")
        value.setdefault("properties", {})
        value.setdefault("required", [])
        value["additionalProperties"] = False
        return value

    def health(self) -> dict[str, Any]:
        try:
            advertised = {item.get("name") for item in self.transport.list_tools()}
            return {
                "available": True,
                "advertised": sorted(item for item in advertised if item),
                "allowed": sorted(self.allowed_tools),
            }
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def call(self, name: str, arguments: dict[str, Any], *, timeout: float = 30.0) -> McpObservation:
        if name not in self.allowed_tools:
            raise PermissionError(f"MCP tool is not allow-listed: {name}")
        schema = self.input_schema(name)
        properties = set(dict(schema.get("properties") or {}))
        unknown = sorted(set(arguments) - properties)
        missing = sorted(set(schema.get("required") or []) - set(arguments))
        if unknown or missing:
            raise McpError(f"MCP arguments violate frozen schema: unknown={unknown}, missing={missing}")
        started = time.perf_counter()
        try:
            result = self.transport.call_tool(name, arguments, timeout)
            if isinstance(result, dict) and result.get("is_error"):
                return McpObservation(
                    server=self.server_name,
                    tool=name,
                    ok=False,
                    result=result,
                    error="MCP tool returned isError=true",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    diagnostics=["tool_protocol_error"],
                )
            return McpObservation(
                server=self.server_name,
                tool=name,
                ok=True,
                result=result,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                diagnostics = ["mcp_timeout"]
            elif isinstance(exc, (ConnectionError, BrokenPipeError)):
                diagnostics = ["mcp_transport_unavailable"]
            else:
                diagnostics = ["mcp_call_failed"]
            return McpObservation(
                server=self.server_name,
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                diagnostics=diagnostics,
            )
