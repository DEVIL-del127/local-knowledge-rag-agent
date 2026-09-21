from __future__ import annotations

from typing import Any, Mapping

from agent.agent_skills import SkillSpec


class McpInternalSkill:
    """Non-router Skill wrapper for one allow-listed MCP tool."""

    def __init__(self, server: str, tool: str, client: Any, *, read_only: bool = True):
        self.client = client
        self.tool = tool
        self.spec = SkillSpec(
            name=f"mcp.{server}.{tool}",
            description=f"Internal MCP tool {server}/{tool}",
            kind="mcp",
            user_invocable=False,
            input_schema=client.input_schema(tool),
            output_schema={"type": "object"},
            read_only=read_only,
            required_capabilities=[f"mcp:{server}:{tool}"],
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        observation = self.client.call(self.tool, dict(arguments), timeout=self.spec.timeout_seconds)
        return {
            "ok": observation.ok,
            "server": observation.server,
            "tool": observation.tool,
            "result": observation.result,
            "error": observation.error,
            "elapsed_ms": observation.elapsed_ms,
            "diagnostics": list(observation.diagnostics),
        }
