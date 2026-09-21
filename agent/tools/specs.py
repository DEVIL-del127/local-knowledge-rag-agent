from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class ToolSpec:
    name: str
    capability: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_write_class: str = "read"
    timeout_seconds: float = 10.0
    retry_policy: str = "none"
    idempotency_policy: str = "idempotent"
    provider: str = "internal"
    llm_visible: bool = False


@dataclass(slots=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    observation_validator: Callable[[Mapping[str, Any]], None] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        if tool.spec.read_write_class not in {"read", "write", "admin"}:
            raise ValueError("invalid read_write_class")
        self._tools[tool.spec.name] = tool

    def specs(self, *, llm_visible_only: bool = False) -> list[ToolSpec]:
        return [
            item.spec for item in self._tools.values()
            if not llm_visible_only or item.spec.llm_visible
        ]

    def execute(self, name: str, arguments: Mapping[str, Any], *, allow_write: bool = False) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"unregistered tool: {name}")
        tool = self._tools[name]
        if tool.spec.read_write_class != "read" and not allow_write:
            raise PermissionError(f"tool is not read-only: {name}")
        _validate(arguments, tool.spec.input_schema, "input")
        result = dict(tool.handler(arguments))
        _validate(result, tool.spec.output_schema, "output")
        if tool.observation_validator is not None:
            tool.observation_validator(result)
        return result


def _validate(value: Mapping[str, Any], schema: Mapping[str, Any], location: str) -> None:
    required = set(schema.get("required") or [])
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{location} missing required fields: {missing}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ValueError(f"{location} contains unknown fields: {unknown}")
