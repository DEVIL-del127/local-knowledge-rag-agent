from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SearchStatus(str, Enum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    DOCUMENT_ABSENT = "document_absent"
    EMPTY_SOURCE = "empty_source"
    INDEX_MISSING = "index_missing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(slots=True)
class KnowledgeBaseStatus:
    source_id: str
    provider_available: bool
    index_exists: bool
    document_count: int = 0
    vector_collection_exists: bool = False
    vector_chunk_count: int = 0
    catalog_version: str = ""
    ingestion_generation: str = ""
    embedding_model: str = ""
    embedding_dimension: int | None = None
    last_successful_ingestion: str | None = None
    health_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchOutcome:
    status: SearchStatus
    evidence: list[dict[str, Any]] = field(default_factory=list)
    executed_channels: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    trace_id: str = ""
    generation: str = ""
    stage_latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class SearchBackendError(RuntimeError):
    def __init__(self, status: SearchStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
