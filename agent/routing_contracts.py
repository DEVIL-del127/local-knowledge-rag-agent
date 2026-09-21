from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Applicability(str, Enum):
    BUSINESS = "business"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class ContextMode(str, Enum):
    NEW = "new"
    INHERIT = "inherit"
    PATCH = "patch"


@dataclass(frozen=True, slots=True)
class ContextProposal:
    mode: ContextMode
    valid: bool
    reason_code: str
    object_ids: tuple[str, ...] = ()
    field_delta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value


@dataclass(frozen=True, slots=True)
class RoutingHint:
    applicability: Applicability
    candidate_domains: tuple[str, ...]
    reason_code: str
    context_proposals: tuple[ContextProposal, ...] = ()
    requires_interpretation: bool = False
    low_cost_latency_ms: float = 0.0
    semantic_category: str = ""
    semantic_score: float = 0.0
    semantic_margin: float = 0.0
    semantic_status: str = "not_used"
    schema_version: str = "routing-hint-v1"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "applicability": self.applicability.value,
                "context_proposals": [item.to_dict() for item in self.context_proposals]}
