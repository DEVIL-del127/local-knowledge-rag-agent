from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agent.literature_ir import RequestedSection, TemporalConstraint


class RetrievalStage(str, Enum):
    INVENTORY = "inventory"
    DOCUMENT_DISCOVERY = "document_discovery"
    DOCUMENT_RESOLUTION = "document_resolution"
    PASSAGE_RETRIEVAL = "passage_retrieval"


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    stage: RetrievalStage
    query: str
    document_ids: tuple[str, ...] = field(default_factory=tuple)
    topic_terms: tuple[str, ...] = field(default_factory=tuple)
    temporal: TemporalConstraint | None = None
    requested_sections: tuple[RequestedSection, ...] = field(default_factory=tuple)
    limit: int = 8


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    steps: tuple[RetrievalStep, ...]
    schema_version: str = "retrieval-plan-v2"
