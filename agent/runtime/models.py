from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from enum import Enum
from typing import Any

from .dialogue import PendingInteraction


class RuntimeMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @classmethod
    def parse(cls, value: str | None) -> "RuntimeMode":
        normalized = str(value or "off").strip().lower()
        return cls(normalized) if normalized in {item.value for item in cls} else cls.OFF


class RuntimeState(str, Enum):
    RECEIVE = "receive"
    LOAD_CONTEXT = "load_context"
    RESOLVE_DIALOGUE = "resolve_dialogue"
    ROUTE_DOMAIN = "route_domain"
    UNDERSTAND = "understand"
    PREFLIGHT_SOURCE = "preflight_source"
    CLARIFY = "clarify"
    WAIT_USER = "wait_user"
    RESUME = "resume"
    PLAN = "plan"
    BIND_TOOLS = "bind_tools"
    EXECUTE = "execute"
    VALIDATE_OBSERVATION = "validate_observation"
    SYNTHESIZE = "synthesize"
    VERIFY_CITATIONS = "verify_citations"
    COMPLETE = "complete"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"


class BusinessOutcome(str, Enum):
    ANSWERED = "answered"
    LISTED = "listed"
    NO_MATCH = "no_match"
    DOCUMENT_ABSENT = "document_absent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(slots=True)
class ExecutionBudget:
    clarification_rounds: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    repair_calls: int = 0
    max_clarification_rounds: int = 2
    max_model_calls: int = 2
    # A complete two-document comparison requires two resolution calls and
    # two document-scoped passage calls; one discovery call may precede them.
    max_tool_calls: int = 5
    max_repair_calls: int = 1


@dataclass(slots=True)
class PendingClarification:
    original_query: str
    question_id: str
    prompt: str
    expected_answer_type: str
    candidate_ids: list[str] = field(default_factory=list)
    ir_digest: str = ""
    catalog_version: str = ""
    gap_digest: str = ""
    source: str = "nlu_v2"
    expires_at: str | None = None
    resume_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestExecutionSnapshot:
    request_id: str
    generation_snapshot_digest: str
    nlu_engine_tree_digest: str
    nlu_profile_digest: str
    request_ir_schema_version: str
    security_scope_digest: str
    turn_context_digest: str
    semantic_envelope_digest: str
    logical_plan_digest: str
    literature_query_digest: str
    retrieval_config_digest: str
    answer_model_identity: str
    synthesis_prompt_version: str
    citation_validator_version: str
    domain_dialogue_style_digest: str = ""
    answer_model_config: dict[str, Any] = field(default_factory=dict)
    conversation_revision: int = 0
    tool_binder_version: str = "read-only-binder-v3"
    renderer_version: str = "grounded-renderer-v3"
    snapshot_schema_version: str = "request-execution-snapshot-v3"

    def cache_digest(self) -> str:
        payload = asdict(self)
        payload.pop("request_id", None)
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cache_digest": self.cache_digest()}


@dataclass(slots=True)
class AgentState:
    request_id: str
    user_id: str
    session_id: str
    thread_id: str
    state: RuntimeState = RuntimeState.RECEIVE
    raw_query: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    accepted_task_state: dict[str, Any] = field(default_factory=dict)
    dependency_router_versions: list[str] = field(default_factory=list)
    needs_revalidation: bool = False
    routing_hint: dict[str, Any] = field(default_factory=dict)
    domain_decision: dict[str, Any] = field(default_factory=dict)
    dialogue_act: dict[str, Any] = field(default_factory=dict)
    last_substantive_turn: dict[str, Any] = field(default_factory=dict)
    pending_interaction: PendingInteraction | None = None
    semantic_result: dict[str, Any] | None = None
    ir_digest: str = ""
    catalog_version: str = ""
    ingestion_generation: str = ""
    pending_clarification: PendingClarification | None = None
    logical_plan: dict[str, Any] | None = None
    bound_skill_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    budgets: ExecutionBudget = field(default_factory=ExecutionBudget)
    transition_log: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    business_outcome: str = ""
    reason_code: str = ""
    result_page: dict[str, Any] = field(default_factory=dict)
    request_execution_snapshot: dict[str, Any] | None = None
    checkpoint_revision: int = 0

    def transition(self, target: RuntimeState, *, reason: str = "") -> None:
        self.transition_log.append({
            "from": self.state.value,
            "to": target.value,
            "reason": reason,
        })
        self.state = target

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentState":
        value = dict(payload)
        value["state"] = RuntimeState(value.get("state", RuntimeState.RECEIVE.value))
        if isinstance(value.get("pending_clarification"), dict):
            value["pending_clarification"] = PendingClarification(
                **value["pending_clarification"]
            )
        if isinstance(value.get("pending_interaction"), dict):
            value["pending_interaction"] = PendingInteraction.from_dict(
                value["pending_interaction"]
            )
        if isinstance(value.get("budgets"), dict):
            value["budgets"] = ExecutionBudget(**value["budgets"])
        return cls(**value)
