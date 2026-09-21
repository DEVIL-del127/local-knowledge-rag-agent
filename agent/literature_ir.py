from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class LiteratureTask(str, Enum):
    ENUMERATE = "enumerate"
    LOCATE = "locate"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    QA = "qa"
    INVENTORY = "inventory"


class BoundType(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TemporalConstraint:
    lower_year: int | None = None
    lower_bound: BoundType | None = None
    upper_year: int | None = None
    upper_bound: BoundType | None = None
    relative_expression: str | None = None
    resolved_at: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if self.lower_year is not None and self.lower_bound is None:
            raise ValueError("lower_bound is required with lower_year")
        if self.upper_year is not None and self.upper_bound is None:
            raise ValueError("upper_bound is required with upper_year")
        if (self.lower_year is not None and self.upper_year is not None
                and self.lower_year > self.upper_year):
            raise ValueError("temporal lower_year exceeds upper_year")

    def to_es_range(self) -> dict[str, int]:
        result: dict[str, int] = {}
        if self.lower_year is not None:
            result["gt" if self.lower_bound == BoundType.OPEN else "gte"] = self.lower_year
        if self.upper_year is not None:
            result["lt" if self.upper_bound == BoundType.OPEN else "lte"] = self.upper_year
        return result


class ReferenceKind(str, Enum):
    NONE = "none"
    CURRENT_DOCUMENT = "current_document"
    PRIOR_RESULTS = "prior_results"
    ORDINAL = "ordinal"
    EXPLICIT_DOCUMENT = "explicit_document"


@dataclass(frozen=True, slots=True)
class DocumentReference:
    kind: ReferenceKind = ReferenceKind.NONE
    title: str | None = None
    author: str | None = None
    filename: str | None = None
    document_ids: tuple[str, ...] = field(default_factory=tuple)
    ordinals: tuple[int, ...] = field(default_factory=tuple)
    expected_count: int | None = None

    def __post_init__(self) -> None:
        if any(item < 1 for item in self.ordinals):
            raise ValueError("ordinals are one-based positive integers")


class RequestedSection(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    CONCLUSION = "conclusion"
    FULL_TEXT = "full_text"
    REFERENCES = "references"


@dataclass(frozen=True, slots=True)
class LiteratureRequestIR:
    raw_query: str
    task: LiteratureTask
    topic_terms: tuple[str, ...] = field(default_factory=tuple)
    canonical_topic: str | None = None
    sense_id: str | None = None
    temporal: TemporalConstraint | None = None
    document_reference: DocumentReference = field(default_factory=DocumentReference)
    requested_sections: tuple[RequestedSection, ...] = field(default_factory=tuple)
    language: str | None = None
    document_type: str | None = None
    result_limit: int = 8
    required_object_refs: tuple[str, ...] = field(default_factory=tuple)
    selection_policy: str | None = None
    comparison_dimensions: tuple[str, ...] = field(default_factory=tuple)
    comparison_object_queries: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "literature-request-ir-v5"

    def __post_init__(self) -> None:
        if not self.raw_query.strip():
            raise ValueError("raw_query is required")
        if not 1 <= self.result_limit <= 100:
            raise ValueError("result_limit must be between 1 and 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiteratureRequestIR":
        temporal_payload = payload.get("temporal")
        temporal = None
        if temporal_payload:
            temporal = TemporalConstraint(
                **{**temporal_payload,
                   "lower_bound": BoundType(temporal_payload["lower_bound"]) if temporal_payload.get("lower_bound") else None,
                   "upper_bound": BoundType(temporal_payload["upper_bound"]) if temporal_payload.get("upper_bound") else None}
            )
        ref_payload = dict(payload.get("document_reference") or {})
        reference = DocumentReference(
            **{**ref_payload, "kind": ReferenceKind(ref_payload.get("kind", "none")),
               "document_ids": tuple(ref_payload.get("document_ids") or ()),
               "ordinals": tuple(ref_payload.get("ordinals") or ())}
        )
        return cls(
            raw_query=str(payload["raw_query"]), task=LiteratureTask(payload["task"]),
            topic_terms=tuple(payload.get("topic_terms") or ()),
            canonical_topic=payload.get("canonical_topic"), sense_id=payload.get("sense_id"),
            temporal=temporal, document_reference=reference,
            requested_sections=tuple(RequestedSection(item) for item in payload.get("requested_sections") or ()),
            language=payload.get("language"), document_type=payload.get("document_type"),
            result_limit=int(payload.get("result_limit", 8)),
            required_object_refs=tuple(payload.get("required_object_refs") or ()),
            selection_policy=payload.get("selection_policy"),
            comparison_dimensions=tuple(payload.get("comparison_dimensions") or ()),
            comparison_object_queries=tuple(payload.get("comparison_object_queries") or ()),
            schema_version=str(payload.get("schema_version", "literature-request-ir-v5")),
        )
