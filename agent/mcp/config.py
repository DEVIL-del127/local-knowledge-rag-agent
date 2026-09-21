from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .client import McpClient
from .stdio import OfficialStdioTransport


@dataclass(slots=True)
class McpServerConfig:
    name: str
    enabled: bool = False
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    allowed_tools: set[str] = field(default_factory=set)
    read_only: bool = True


def load_server_configs(path: str | Path) -> list[McpServerConfig]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    configs = []
    for name, raw in dict(payload.get("servers") or {}).items():
        item = dict(raw or {})
        env = {
            str(key): _expand_env(str(value))
            for key, value in dict(item.get("env") or {}).items()
            if _expand_env(str(value))
        }
        configs.append(McpServerConfig(
            name=str(name),
            enabled=bool(item.get("enabled", False)),
            transport=str(item.get("transport", "stdio")),
            command=_expand_env(str(item.get("command", ""))),
            args=[_expand_env(str(value)) for value in item.get("args") or []],
            env=env,
            allowed_tools={str(value) for value in item.get("allowed_tools") or []},
            read_only=bool(item.get("read_only", True)),
        ))
    return configs


def build_clients(path: str | Path) -> dict[str, McpClient]:
    clients = {}
    for config in load_server_configs(path):
        if not config.enabled:
            continue
        if config.transport != "stdio":
            raise ValueError(f"unsupported MCP transport: {config.transport}")
        if not config.command:
            raise ValueError(f"enabled MCP server has no command: {config.name}")
        client = McpClient(
            config.name,
            OfficialStdioTransport(config.command, config.args, config.env),
            allowed_tools=config.allowed_tools,
        )
        client.freeze_schemas()
        clients[config.name] = client
    return clients


def _expand_env(value: str) -> str:
    return os.path.expandvars(value).strip()
