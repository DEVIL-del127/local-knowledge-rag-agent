from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from agent.agent_models import RetrievedEvidence

logger = logging.getLogger(__name__)


class SearchBackend(Protocol):
    def hybrid_search(
        self,
        query: str,
        top_n: int = 10,
        include_chunks: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """include_chunks=True 时返回 (文档级结果, 向量块级结果)"""
        ...

    def vec_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        ...


class LegacyPrivateSearchSkill:
    """Compatibility read adapter for backends without generation metadata."""

    def __init__(self, backend: SearchBackend, *, max_top_k: int = 20) -> None:
        self.backend = backend
        self.max_top_k = max_top_k
        self.spec = SkillSpec(
            name="search_private_kb",
            description="Search the private knowledge base for grounded evidence.",
            tags=["kb", "search", "read-only"],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": max_top_k},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["evidence", "prompt_context"]},
            read_only=True,
            required_capabilities=["document_search"],
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        top_k = max(1, min(int(arguments.get("top_k", 5)), self.max_top_k))
        raw = self.backend.hybrid_search(query, top_n=top_k, include_chunks=True)
        docs, chunks = raw if isinstance(raw, tuple) else (raw, [])
        evidence = []
        for channel, items in (("document", docs), ("vector", chunks)):
            for source in list(items or []):
                item = dict(source)
                item.setdefault("channel", channel)
                item.setdefault("source", str(item.get("filename") or item.get("title") or ""))
                item.setdefault("snippet", str(
                    item.get("content") or item.get("text") or item.get("highlights")
                    or item.get("abstract") or item.get("title") or ""
                ))
                evidence.append(item)
        return {
            "query": query,
            "evidence": evidence,
            "prompt_context": "\n\n".join(str(item.get("snippet") or "") for item in evidence),
        }


@dataclass(slots=True)
class SkillSpec:
    name: str
    description: str
    kind: str = "local"
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    # 是否可被用户/意图路由触发; False = 后台自动 skill, 不进路由表
    user_invocable: bool = True
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object", "properties": {}, "additionalProperties": True,
    })
    output_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    read_only: bool = True
    timeout_seconds: float = 10.0
    retry_policy: str = "none"
    idempotency_policy: str = "idempotent"
    required_capabilities: list[str] = field(default_factory=list)

    def to_router_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "tags": list(self.tags),
        }


class SkillProtocol(Protocol):
    spec: SkillSpec

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        ...


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillProtocol] = {}

    def register(self, skill: SkillProtocol) -> None:
        self._skills[skill.spec.name] = skill

    def has(self, skill_name: str) -> bool:
        return skill_name in self._skills

    def get(self, skill_name: str) -> SkillProtocol:
        return self._skills[skill_name]

    def execute(
        self,
        skill_name: str,
        arguments: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        skill = self.get(skill_name)
        _validate_schema_object(dict(arguments), skill.spec.input_schema, location="input")
        from agent.model_execution_context import current_operation
        execution = current_operation()
        if execution is not None:
            execution[0].reserve_tool_attempt(execution[1])
        contextual = getattr(skill, "invoke_with_context", None)
        result = contextual(arguments, context or {}) if callable(contextual) else skill.invoke(arguments)
        _validate_schema_object(result, skill.spec.output_schema, location="output")
        return result

    def router_skills(self) -> list[dict[str, Any]]:
        specs = []
        for skill in self._skills.values():
            # 只暴露用户可调用的 skill, 后台 skill(user_invocable=False)不进路由表
            if skill.spec.enabled and skill.spec.user_invocable:
                specs.append(skill.spec.to_router_dict())
        return specs


class LiteratureReadSkill:
    """Typed read-only literature tool; it never interprets natural language."""

    def __init__(self, name: str, gateway: Any, *, max_top_k: int = 20) -> None:
        if name not in {"discover_documents", "resolve_document", "retrieve_document_passages"}:
            raise ValueError(f"unsupported literature tool: {name}")
        self.gateway = gateway
        self.max_top_k = max_top_k
        properties: dict[str, Any] = {
            "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": max_top_k},
        }
        required = ["query"]
        if name == "discover_documents":
            properties.update({"topic_terms": {"type": "array"}, "temporal_range": {"type": "object"},
                               "language": {"type": "string"}, "document_type": {"type": "string"}})
        elif name == "retrieve_document_passages":
            properties.update({"document_ids": {"type": "array"}, "requested_sections": {"type": "array"}})
            required.append("document_ids")
        self.spec = SkillSpec(
            name=name, description=f"Typed literature operation: {name}", kind="local",
            tags=["literature", "read-only"], user_invocable=False,
            input_schema={"type": "object", "properties": properties, "required": required,
                          "additionalProperties": False},
            output_schema={"type": "object", "required": ["query", "evidence", "prompt_context"]},
            required_capabilities=["document_search"],
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.invoke_with_context(arguments, {})

    def invoke_with_context(self, arguments: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        limit = max(1, min(int(arguments.get("limit", 8)), self.max_top_k))
        snapshot = context.get("generation_snapshot")
        if self.spec.name == "resolve_document":
            outcome = self.gateway.find_document(query, snapshot=snapshot)
        else:
            document_ids = [str(item) for item in arguments.get("document_ids", [])]
            if self.spec.name == "retrieve_document_passages" and not document_ids:
                raise ValueError("retrieve_document_passages requires document_ids")
            outcome = self.gateway.hybrid_search(
                query, top_k=limit, snapshot=snapshot,
                mode="document" if self.spec.name == "discover_documents" else "hybrid",
                topic_terms=[str(item) for item in arguments.get("topic_terms", [])],
                temporal_range=dict(arguments.get("temporal_range") or {}),
                language=str(arguments.get("language", "")),
                document_type=str(arguments.get("document_type", "")),
                document_ids=document_ids,
                section_types=[str(item) for item in arguments.get("requested_sections", [])],
            )
        evidence = []
        for raw in outcome.evidence:
            item = dict(raw)
            item.setdefault("source", str(item.get("filename") or ""))
            item.setdefault("snippet", str(
                item.get("text") or item.get("content") or item.get("highlights")
                or item.get("abstract") or item.get("title") or ""
            ))
            evidence.append(item)
        requested = {str(item) for item in arguments.get("document_ids", [])}
        if requested and any(str(item.get("document_id", "")) not in requested for item in evidence):
            raise ValueError("retrieval returned evidence outside requested document_ids")
        docs = [item for item in evidence if item.get("channel") != "vector"]
        chunks = [item for item in evidence if item.get("channel") == "vector"]
        kb_status = self.gateway.status(snapshot)
        return {
            "query": query, "search_status": outcome.status.value,
            "kb_status": kb_status.to_dict(),
            "doc_hits": docs, "chunk_hits": chunks, "evidence": evidence,
            "prompt_context": "\n\n".join(str(item.get("text") or item.get("content") or "") for item in evidence),
            "generation": outcome.generation, "trace_id": outcome.trace_id,
            "stage_latency_ms": dict(outcome.stage_latency_ms),
        }


class KnowledgeBaseDiagnoseSkill:
    """Internal read-only source inspection; never exposed to the intent router."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.spec = SkillSpec(
            name="kb_diagnose",
            description="Inspect knowledge-base health, inventory, or an explicit document identity.",
            kind="local",
            user_invocable=False,
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "identity": {"type": "string"},
                    "cursor": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["operation", "status"]},
            read_only=True,
            required_capabilities=["kb.health", "kb.inventory", "kb.find_document"],
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.invoke_with_context(arguments, {})

    def invoke_with_context(
        self, arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, Any]:
        operation = str(arguments.get("operation", "")).strip()
        snapshot = context.get("generation_snapshot")
        if operation in {"health", "inventory"}:
            status = self.gateway.status(snapshot)
            payload = {
                "operation": operation,
                "status": "available" if status.provider_available and status.index_exists else "unavailable",
                "knowledge_base": status.to_dict(),
            }
            if operation == "inventory" and payload["status"] == "available":
                payload["inventory"] = self.gateway.inventory(
                    snapshot, cursor=int(arguments.get("cursor", 0)),
                    limit=int(arguments.get("limit", 50)),
                )
            return payload
        if operation == "find_document":
            identity = str(arguments.get("identity", "")).strip()
            if not identity:
                raise ValueError("find_document requires identity")
            outcome = self.gateway.find_document(identity, snapshot=snapshot)
            return {
                "operation": operation,
                "status": outcome.status.value,
                "identity": identity,
                "evidence": outcome.evidence,
                "diagnostics": outcome.diagnostics,
                "trace_id": outcome.trace_id,
            }
        raise ValueError(f"unsupported kb_diagnose operation: {operation}")


def format_evidence_block(evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return "未检索到可用证据。"

    lines: list[str] = []
    for index, item in enumerate(evidence, start=1):
        header = f"[E{index}] source={item.source}"
        if item.page is not None:
            header += f" page={item.page}"
        if item.channel:
            header += f" channel={item.channel}"
        if item.score is not None:
            header += f" score={item.score:.4f}"
        lines.append(header)
        lines.append(item.snippet.strip())
        lines.append("")
    return "\n".join(lines).strip()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _kb_status_message(status: str, reason: str) -> str:
    if status == "provider_unavailable":
        return "当前无法连接知识库服务。" + (f" 原因：{reason}" if reason else "")
    if status == "index_missing":
        return "知识库索引尚未建立，请先完成文档入库。"
    if status == "empty_source":
        return "知识库当前为空，请先导入文档。"
    return "未检索到可用证据。"


def _document_identity(query: str) -> str:
    """Extract only explicit PDF identities; ordinary quoted topics are not documents."""
    import re

    match = re.search(r"[《\"']?([^《》\"'，。！？\s]{1,120}\.pdf)[》\"']?", query, re.IGNORECASE)
    return match.group(1) if match else ""


def _validate_schema_object(value: Any, schema: Mapping[str, Any], *, location: str) -> None:
    if schema.get("type") == "object" and not isinstance(value, dict):
        raise ValueError(f"skill {location} must be an object")
    if not isinstance(value, dict):
        return
    required = set(schema.get("required") or [])
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"skill {location} missing required fields: {missing}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ValueError(f"skill {location} contains unknown fields: {unknown}")


class KnowledgeBaseManageSkill:
    """文档库管理(最小版): list/stats; 增删改操作提示走主菜单"""

    def __init__(self, pdf_dir: str, es_index: str | None = None, es_client=None) -> None:
        self.pdf_dir = pdf_dir
        self.es_index = es_index
        self.es_client = es_client
        self.spec = SkillSpec(
            name="kb_manage",
            description=(
                "Manage the knowledge base document library: list documents, "
                "show stats (count by type), locate a document by name."
            ),
            kind="local",
            tags=["kb", "manage"],
            user_invocable=True,
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "query": {"type": "string"},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            required_capabilities=["inventory"],
        )

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "list")).strip()
        query = str(arguments.get("query", "")).strip()

        if command in ("list", "auto"):
            return self._list(query)
        if command == "stats":
            return self._stats()
        return {"error": f"未知命令: {command}"}

    def _list(self, query: str) -> dict[str, Any]:
        import os

        files = []
        if os.path.isdir(self.pdf_dir):
            for name in sorted(os.listdir(self.pdf_dir)):
                if name.lower().endswith(".pdf") and (not query or query in name):
                    files.append(name)
        return {
            "command": "list",
            "total": len(files),
            "files": files[:50],
            "note": "增删文档请使用主菜单(1/2 建库)或手动放入 pdfs/ 目录后重建索引",
        }

    def _stats(self) -> dict[str, Any]:
        import os

        total = 0
        if os.path.isdir(self.pdf_dir):
            total = sum(1 for n in os.listdir(self.pdf_dir) if n.lower().endswith(".pdf"))
        result: dict[str, Any] = {"command": "stats", "pdf_count": total}
        if self.es_client is not None and self.es_index:
            try:
                resp = self.es_client.count(index=self.es_index)
                result["indexed_count"] = resp.get("count", 0)
            except Exception:
                result["indexed_count"] = None
        return result
