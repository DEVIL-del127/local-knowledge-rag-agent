from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _default_ollama_url() -> str:
    """Resolve the Windows host under WSL NAT; preserve localhost elsewhere."""
    if not (os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")):
        return "http://localhost:11434"
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"], capture_output=True,
            text=True, timeout=1.0, check=False,
        )
        parts = result.stdout.split()
        if "via" in parts:
            gateway = parts[parts.index("via") + 1]
            if gateway:
                return f"http://{gateway}:11434"
    except (OSError, subprocess.SubprocessError):
        pass
    return "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class AppSettings:
    project_root: Path
    es_host: str
    es_port: int
    ollama_url: str
    embed_model: str
    agent_db: str
    user_id: str
    execution_profile: str
    state_dir: Path
    ingestion_registry_path: Path
    mcp_config_path: Path
    semantic_compiler_llm: bool
    llm_clarification: bool

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.user_id):
            raise ValueError("AGENT_USER_ID must contain 1–64 letters, digits, underscores or hyphens")
        if type(self.es_port) is not int or not 1 <= self.es_port <= 65535:
            raise ValueError("ES_PORT must be between 1 and 65535")
        if self.agent_db not in {"main", "test"}:
            raise ValueError("AGENT_DB must be main or test")
        if self.execution_profile not in {"off", "shadow", "enforce"}:
            raise ValueError("unsupported execution profile")

    @classmethod
    def from_env(cls, project_root: str | Path) -> "AppSettings":
        root = Path(project_root).resolve()
        profile = os.environ.get("AGENT_EXECUTION_PROFILE", "").strip().lower()
        if not profile:
            profile = os.environ.get("AGENT_RUNTIME_MODE", "enforce").strip().lower()
        if profile not in {"off", "shadow", "enforce"}:
            raise ValueError(f"invalid AGENT_EXECUTION_PROFILE: {profile}")

        def path_setting(name: str, default: Path) -> Path:
            raw = os.environ.get(name, "").strip()
            candidate = Path(raw) if raw else default
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        return cls(
            project_root=root,
            es_host=os.environ.get("ES_HOST", "localhost").strip() or "localhost",
            es_port=int(os.environ.get("ES_PORT", "9200")),
            ollama_url=os.environ.get("OLLAMA_URL", "").strip() or _default_ollama_url(),
            embed_model=os.environ.get("EMBED_MODEL", "bge-m3").strip() or "bge-m3",
            agent_db=os.environ.get("AGENT_DB", "main").strip() or "main",
            user_id=os.environ.get("AGENT_USER_ID", "local").strip() or "local",
            execution_profile=profile,
            state_dir=path_setting("AGENT_STATE_DIR", root / "agent_state"),
            ingestion_registry_path=path_setting(
                "INGESTION_REGISTRY_PATH", root / "data/ingestion/generation_registry.sqlite3"
            ),
            mcp_config_path=path_setting("MCP_CONFIG", root / "config/mcp/servers.yaml"),
            semantic_compiler_llm=os.environ.get("SEMANTIC_COMPILER_LLM", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            llm_clarification=os.environ.get(
                "AGENT_LLM_CLARIFICATION", "0"
            ).strip().lower() in {"1", "true", "yes", "on"},
        )
