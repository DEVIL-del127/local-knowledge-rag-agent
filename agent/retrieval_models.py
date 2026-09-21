from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SemanticDecision(str, Enum):
    PASS_THROUGH = "pass_through"
    STRUCTURED_RETRIEVAL = "structured_retrieval"
    CLARIFY = "clarify"
    UNSUPPORTED_ANALYTICS = "unsupported_analytics"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass(slots=True)
class SemanticCompileResult:
    decision: SemanticDecision
    effective_query: str
    catalog_version: str = ""
    ir_digest: str = ""
    diagnostics: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    model_calls: int = 0
    elapsed_ms: float = 0.0
    shadow_only: bool = True
    clarification_questions: list[dict[str, Any]] = field(default_factory=list)
    validation_status: str = ""
    executable: bool = False
    request_ir: dict[str, Any] = field(default_factory=dict)
    logical_plan: dict[str, Any] = field(default_factory=dict)
    contract_reports: dict[str, Any] = field(default_factory=dict)
    engine_revision: str = ""
    engine_tree_digest: str = ""
    engine_profile: str = ""
    engine_config_digest: str = ""
    logical_plan_digest: str = ""
    envelope_schema_version: str = "semantic-execution-envelope-v1"
    engine_source_revision: str = ""
    engine_profile_digest: str = ""
    request_ir_schema_version: str = ""
    catalog_digest: str = ""
    security_scope_digest: str = ""
    turn_context_digest: str = ""
    generation_snapshot_digest: str = ""
    admission: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    model_audit: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    resume_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload

    def stable_contract_dict(self) -> dict[str, Any]:
        """Return only deterministic semantic identity fields used by caches."""
        payload = self.to_dict()
        for key in ("elapsed_ms", "trace", "model_audit"):
            payload.pop(key, None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticCompileResult":
        value = dict(payload)
        value["decision"] = SemanticDecision(value["decision"])
        return cls(**value)
