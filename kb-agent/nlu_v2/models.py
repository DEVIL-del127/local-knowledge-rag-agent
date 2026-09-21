"""Typed contracts shared by the V2 understanding and planning layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Optional


SCHEMA_VERSION = "2.5"


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceSpan(Serializable):
    start: int
    end: int
    text: str


@dataclass(slots=True)
class Provenance(Serializable):
    source: str
    span: Optional[SourceSpan] = None
    rule: str = ""
    score: Optional[float] = None


@dataclass(slots=True)
class QueryEnvelope(Serializable):
    raw: str
    normalized: str
    language: str = "zh"
    context_version: str = ""
    security_scope_digest: str = ""


@dataclass(slots=True)
class ClauseNode(Serializable):
    clause_id: str
    span: SourceSpan
    text: str
    role: str = "context"
    polarity: str = "positive"
    instruction_scope: str = "active"  # active / quoted / example
    group_id: str = ""


@dataclass(slots=True)
class ClauseEdge(Serializable):
    source_id: str
    target_id: str
    relation: str
    span: Optional[SourceSpan] = None


@dataclass(slots=True)
class ClauseGraph(Serializable):
    nodes: list[ClauseNode] = dc_field(default_factory=list)
    edges: list[ClauseEdge] = dc_field(default_factory=list)


@dataclass(slots=True)
class SourceDemand(Serializable):
    """An immutable, source-text obligation used to prevent false coverage.

    ``RequirementSpec`` is the compiler-facing view of a demand.  This lower
    level record intentionally keeps every anchored source fact, including
    facts for which the current compiler has no executable operator yet.
    """

    demand_id: str
    demand_type: str  # field / quantity / unit / comparison / boolean / temporal / output / operator
    text: str
    span: SourceSpan
    clause_id: str = ""
    attributes: dict[str, Any] = dc_field(default_factory=dict)
    dependencies: list[str] = dc_field(default_factory=list)
    critical: bool = True
    consumption_status: str = "unresolved"  # mapped_to_requirement / unresolved / approved_noise
    mapped_requirement_ids: list[str] = dc_field(default_factory=list)
    approved_noise_reason: str = ""
    field_anchor_id: str = ""
    quantity_anchor_id: str = ""
    unit_anchor_id: str = ""
    operator_anchor_id: str = ""


@dataclass(slots=True)
class QuerySchemaSnapshot(Serializable):
    """Query-local schema evidence, never an implicit Catalog binding."""

    source_ids: list[str] = dc_field(default_factory=list)
    field_ids: list[str] = dc_field(default_factory=list)
    candidate_field_ids: list[str] = dc_field(default_factory=list)
    hypotheses: list[str] = dc_field(default_factory=list)
    evidence: list[SourceSpan] = dc_field(default_factory=list)
    binding_basis: str = ""  # explicit_query / source_catalog / session / candidate_only


@dataclass(slots=True)
class CandidateRef(Serializable):
    identifier: str
    score: Optional[float] = None
    source: str = "catalog"
    status: str = "resolved"  # candidate / resolved / rejected


@dataclass(slots=True)
class SymbolRef(Serializable):
    raw_name: str
    canonical_id: Optional[str] = None
    candidates: list[CandidateRef] = dc_field(default_factory=list)
    status: str = "unresolved"  # unresolved / candidate / resolved
    span: Optional[SourceSpan] = None
    ref_kind: str = "catalog"  # catalog / hypothesis


@dataclass(slots=True)
class LiteralValue(Serializable):
    value: Any
    value_type: str
    unit: Optional[str] = None
    raw_unit: Optional[str] = None


@dataclass(slots=True)
class PredicateNode(Serializable):
    kind: str = "comparison"  # comparison / boolean
    operator: str = ""
    field: Optional[SymbolRef] = None
    value: Optional[LiteralValue] = None
    right_expression: Optional["ExpressionNode"] = None
    children: list["PredicateNode"] = dc_field(default_factory=list)
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class ExpressionNode(Serializable):
    kind: str  # field / literal / reference / binary / function
    operator: str = ""
    arguments: list[Any] = dc_field(default_factory=list)
    symbol: Optional[SymbolRef] = None
    literal: Optional[LiteralValue] = None
    reference: str = ""
    result_type: str = "unknown"
    unit: Optional[str] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class SchemaHypothesis(Serializable):
    hypothesis_id: str
    raw_name: str
    normalized_name: str
    declared_type: str = "unknown"
    description: str = ""
    unit: Optional[str] = None
    aliases: list[str] = dc_field(default_factory=list)
    span: Optional[SourceSpan] = None
    executable: bool = False


@dataclass(slots=True)
class OutputRef(Serializable):
    producer_id: str
    port: str
    result_type: str
    unit: Optional[str] = None
    shape: str = "scalar"  # scalar / relation / event_set
    grain: str = "scalar"
    keys: list[str] = dc_field(default_factory=list)
    # Semantic value fields carried by this output.  ``keys`` remains reserved
    # for grouping keys; conflating it with produced values made it impossible
    # for RequirementIR to state and Coverage to prove the requested output.
    fields: list[str] = dc_field(default_factory=list)

    @property
    def ref_id(self) -> str:
        return f"{self.producer_id}.{self.port}"


@dataclass(slots=True)
class AggregateSpec(Serializable):
    aggregate_id: str
    function: str
    input: ExpressionNode
    output: OutputRef
    provenance: list[Provenance] = dc_field(default_factory=list)
    parameters: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(slots=True)
class ScopedAggregateSpec(Serializable):
    aggregate: AggregateSpec
    scope: str  # filtered / global / relation
    scope_ref: Optional[OutputRef] = None
    group_by: list[ExpressionNode] = dc_field(default_factory=list)


@dataclass(slots=True)
class GroupBySpec(Serializable):
    group_id: str
    input_ref: OutputRef
    keys: list[ExpressionNode]
    output: OutputRef
    scope: str = "grouped"
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class TopKSpec(Serializable):
    top_k_id: str
    input_ref: OutputRef
    rank_by: ExpressionNode
    limit: int
    output: OutputRef
    descending: bool = True
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class WindowAggregateSpec(Serializable):
    window_id: str
    function: str
    input: ExpressionNode
    window: WindowSpec
    output: OutputRef
    input_ref: Optional[OutputRef] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class CumulativeCountSpec(Serializable):
    count_id: str
    action: PredicateNode
    scope: WindowSpec
    output: OutputRef
    group_by: list[ExpressionNode] = dc_field(default_factory=list)
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class CumulativeDurationSpec(Serializable):
    """A duration accumulation operator with an explicit typed output.

    This is deliberately separate from a consecutive ``EventSpec``: a
    cumulative duration sums qualifying intervals in a scope and is therefore
    a numeric DAG producer rather than an event-set annotation.
    """
    duration_id: str
    condition: PredicateNode
    scope: WindowSpec
    output: OutputRef
    input_ref: Optional[OutputRef] = None
    integration_method: str = "step"  # step / linear
    group_by: list[ExpressionNode] = dc_field(default_factory=list)
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class ComparisonSpec(Serializable):
    comparison_id: str
    left: ExpressionNode
    operator: str
    right: ExpressionNode
    output: OutputRef
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class DerivedProjectionSpec(Serializable):
    projection_id: str
    function: str
    input_ref: OutputRef
    output: OutputRef
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class ArithmeticSpec(Serializable):
    arithmetic_id: str
    expression: ExpressionNode
    output: OutputRef
    provenance: list[Provenance] = dc_field(default_factory=list)
    scope: str = ""


@dataclass(slots=True)
class RelationFilterSpec(Serializable):
    """Filter one typed relation/event-set by a typed predicate."""

    filter_id: str
    input_ref: OutputRef
    predicate: PredicateNode
    output: OutputRef
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class ConversionSpec(Serializable):
    conversion_id: str
    input: ExpressionNode
    target_unit: str
    output: OutputRef
    policy: str = "explicit"
    missing_data_policy: str = "reject"
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class MetricSpec(Serializable):
    field: SymbolRef
    aggregation: str = "none"
    unit: Optional[str] = None
    alias: str = ""


@dataclass(slots=True)
class TemporalItem(Serializable):
    operator: str
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    exact: Optional[str] = None
    granularity: str = "year"
    timezone: str = ""
    raw: str = ""
    span: Optional[SourceSpan] = None
    evidence_spans: list[SourceSpan] = dc_field(default_factory=list)


@dataclass(slots=True)
class WindowSpec(Serializable):
    window_type: str  # absolute / month_to_date / calendar_day
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    reference: str = ""
    timezone: str = "Asia/Shanghai"
    granularity: str = "minute"


@dataclass(slots=True)
class SamplingPolicy(Serializable):
    order_by: Optional[SymbolRef] = None
    partition_by: list[SymbolRef] = dc_field(default_factory=list)
    expected_interval_seconds: Optional[float] = None
    max_gap_seconds: Optional[float] = None
    missing_data_policy: str = "break_segment"
    duplicate_policy: str = "keep_last"


@dataclass(slots=True)
class DurationConstraint(Serializable):
    operator: str
    value: float
    unit: str
    normalized_seconds: float
    span: Optional[SourceSpan] = None


@dataclass(slots=True)
class DerivedMetricCall(Serializable):
    metric_id: str
    arguments: dict[str, Any] = dc_field(default_factory=dict)
    result_type: str = "number"
    unit: Optional[str] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class EventSpec(Serializable):
    event_id: str
    condition: PredicateNode
    derived_metric: DerivedMetricCall
    threshold: Optional[DurationConstraint]
    window: WindowSpec
    sampling: SamplingPolicy
    group_by: list[str] = dc_field(default_factory=list)
    output_name: str = ""
    output_ref: Optional[OutputRef] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class SetOperationSpec(Serializable):
    operation: str  # intersection / union / difference
    inputs: list[str]
    output_name: str
    granularity: str = "date"
    input_refs: list[OutputRef] = dc_field(default_factory=list)
    output_ref: Optional[OutputRef] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class StateTransitionSpec(Serializable):
    transition_id: str
    partition_by: list[SymbolRef]
    order_by: SymbolRef
    initial_state: ExpressionNode
    transition_expression: ExpressionNode
    output: OutputRef
    reset_condition: Optional[PredicateNode] = None
    reset_expression: Optional[ExpressionNode] = None
    missing_data_policy: str = "reject"
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class OrderedFoldSpec(Serializable):
    fold_id: str
    transition: StateTransitionSpec
    output: OutputRef
    post_condition: Optional[PredicateNode] = None
    post_duration: Optional[DurationConstraint] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class SequenceSpec(Serializable):
    sequence_id: str
    steps: list[PredicateNode]
    partition_by: list[SymbolRef]
    order_by: Optional[SymbolRef]
    output: OutputRef
    max_duration: Optional[DurationConstraint] = None
    allow_repeated_steps: list[int] = dc_field(default_factory=list)
    missing_data_policy: str = "reject"
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class UnsatisfiableSpec(Serializable):
    check_id: str
    reason: str
    predicate_spans: list[SourceSpan] = dc_field(default_factory=list)
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class OperatorInvocation(Serializable):
    invocation_id: str
    operator_id: str
    input_refs: list[str] = dc_field(default_factory=list)
    output_ref: Optional[OutputRef] = None
    requirement_ids: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class ContentAnnotation(Serializable):
    annotation_id: str
    kind: str
    span: SourceSpan
    action: str = "treat_as_data"


@dataclass(slots=True)
class GoalSpec(Serializable):
    goal_type: str  # retrieve / aggregate / compare / compute / present
    span: Optional[SourceSpan] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class CalculationSpec(Serializable):
    calculation_type: str
    expression: str
    parameters: dict[str, Any] = dc_field(default_factory=dict)
    provenance: list[Provenance] = dc_field(default_factory=list)
    calculation_id: str = ""
    inputs: list[ExpressionNode] = dc_field(default_factory=list)
    output: Optional[OutputRef] = None
    scope: str = ""


@dataclass(slots=True)
class ReferenceSpec(Serializable):
    reference_id: str
    raw: str
    target_task_id: Optional[str] = None
    status: str = "unresolved"
    span: Optional[SourceSpan] = None


@dataclass(slots=True)
class OutputContract(Serializable):
    format: str = "structured"
    fields: list[str] = dc_field(default_factory=list)
    group_by: list[str] = dc_field(default_factory=list)
    sort: list[dict[str, str]] = dc_field(default_factory=list)
    criteria: list[str] = dc_field(default_factory=list)
    limit: Optional[int] = None
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class UnresolvedItem(Serializable):
    code: str
    message: str
    span: Optional[SourceSpan] = None
    candidates: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class Diagnostic(Serializable):
    severity: str
    code: str
    message: str
    span: Optional[SourceSpan] = None


@dataclass(slots=True)
class SemanticClaim(Serializable):
    """One auditable statement proposed by rules, retrieval, or an LLM."""

    claim_id: str
    claim_type: str
    role: str  # target / constraint / context / output / example / uncertain
    value: dict[str, Any] = dc_field(default_factory=dict)
    status: str = "accepted"  # candidate / accepted / conflict / rejected
    provenance: list[Provenance] = dc_field(default_factory=list)
    conflicts_with: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class RequirementSpec(Serializable):
    """An explicit user requirement and the claims that satisfy it."""

    requirement_id: str
    requirement_type: str
    role: str
    text: str = ""
    span: Optional[SourceSpan] = None
    expected_claim_types: list[str] = dc_field(default_factory=list)
    claim_ids: list[str] = dc_field(default_factory=list)
    status: str = "uncovered"  # satisfied / uncovered / ambiguous / ignored
    critical: bool = True
    clause_id: str = ""
    operator_family: str = ""
    cardinality: int = 1
    expected_attributes: dict[str, Any] = dc_field(default_factory=dict)
    dependencies: list[str] = dc_field(default_factory=list)
    expected_inputs: list[str] = dc_field(default_factory=list)
    expected_output_fields: list[str] = dc_field(default_factory=list)
    expected_scope: str = ""
    expected_output_shape: str = ""
    expected_output_grain: str = ""


@dataclass(slots=True)
class CoverageSpec(Serializable):
    requirement_id: str
    status: str = "uncovered"
    covered_by: list[str] = dc_field(default_factory=list)
    missing_attributes: list[str] = dc_field(default_factory=list)
    reason: str = ""
    matched_attributes: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(slots=True)
class AmbiguitySpec(Serializable):
    ambiguity_id: str
    kind: str
    message: str
    span: Optional[SourceSpan] = None
    candidates: list[str] = dc_field(default_factory=list)
    status: str = "open"  # open / resolved / accepted_default


@dataclass(slots=True)
class ContextDeltaOperation(Serializable):
    operation: str  # add / replace / remove
    slot: str
    value: Any = None


@dataclass(slots=True)
class ContextDelta(Serializable):
    owner: str  # previous_turn / new_turn
    operations: list[ContextDeltaOperation] = dc_field(default_factory=list)


@dataclass(slots=True)
class TurnDirective(Serializable):
    directive_id: str
    pre_actions: list[str] = dc_field(default_factory=list)
    primary_action: str = "new_query"
    target_turn_id: Optional[str] = None
    context_mode: str = "fresh"  # fresh / inherit
    base_context_ref: Optional[str] = None
    context_delta: Optional[ContextDelta] = None
    required_context: list[str] = dc_field(default_factory=list)
    provenance: list[Provenance] = dc_field(default_factory=list)


@dataclass(slots=True)
class TurnContextSnapshot(Serializable):
    previous_turn_id: str
    accepted_semantic_ir: Any
    context_digest: str
    schema_version: str = SCHEMA_VERSION
    catalog_version: str = ""


@dataclass(slots=True)
class PatchOperationResult(Serializable):
    operation_id: str
    operation_type: str
    gap_id: str
    status: str  # accepted / rejected
    reason: str = ""


@dataclass(slots=True)
class PatchReport(Serializable):
    protocol: str = "semantic_patch_v1"
    base_ir_digest: str = ""
    accepted: list[PatchOperationResult] = dc_field(default_factory=list)
    rejected: list[PatchOperationResult] = dc_field(default_factory=list)
    closed_gap_ids: list[str] = dc_field(default_factory=list)
    committed: bool = False
    transaction_error: str = ""
    rejection_gate: str = ""
    concrete_gain_count: int = 0
    clarification_gain_count: int = 0
    semantic_effect: Any = None

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def closed_gap_count(self) -> int:
        return len(self.closed_gap_ids)


@dataclass(slots=True)
class SemanticEffectReport(Serializable):
    """Auditable semantic outcome of a candidate-patch transaction."""

    acceptance_profile: str = "none"  # concrete / clarification / none
    closed_requirement_ids: list[str] = dc_field(default_factory=list)
    closed_gap_ids: list[str] = dc_field(default_factory=list)
    added_nodes_and_edges: list[str] = dc_field(default_factory=list)
    resolved_diagnostics: list[str] = dc_field(default_factory=list)
    new_diagnostics: list[str] = dc_field(default_factory=list)
    coverage_before: dict[str, str] = dc_field(default_factory=dict)
    coverage_after: dict[str, str] = dc_field(default_factory=dict)
    base_digest: str = ""
    candidate_digest: str = ""
    final_digest: str = ""
    concrete_gain: int = 0
    clarification_gain: int = 0
    commit_or_reject_reason: str = ""


@dataclass(slots=True)
class UnderstandingIR(Serializable):
    query: QueryEnvelope
    catalog_version: str
    schema_version: str = SCHEMA_VERSION
    clause_graph: Optional[ClauseGraph] = None
    query_schema: Optional[QuerySchemaSnapshot] = None
    source_demands: list[SourceDemand] = dc_field(default_factory=list)
    goals: list[GoalSpec] = dc_field(default_factory=list)
    source_candidates: list[CandidateRef] = dc_field(default_factory=list)
    literature_contract: dict[str, Any] = dc_field(default_factory=dict)
    source_binding: dict[str, Any] = dc_field(default_factory=dict)
    accepted_ir_digest: str = ""
    schema_hypotheses: list[SchemaHypothesis] = dc_field(default_factory=list)
    projections: list[SymbolRef] = dc_field(default_factory=list)
    metrics: list[MetricSpec] = dc_field(default_factory=list)
    filters: Optional[PredicateNode] = None
    temporal: list[TemporalItem] = dc_field(default_factory=list)
    sampling_policies: list[SamplingPolicy] = dc_field(default_factory=list)
    events: list[EventSpec] = dc_field(default_factory=list)
    derived_projections: list[DerivedProjectionSpec] = dc_field(default_factory=list)
    set_operations: list[SetOperationSpec] = dc_field(default_factory=list)
    grouping: list[SymbolRef] = dc_field(default_factory=list)
    calculations: list[CalculationSpec] = dc_field(default_factory=list)
    aggregates: list[ScopedAggregateSpec] = dc_field(default_factory=list)
    group_by_operations: list[GroupBySpec] = dc_field(default_factory=list)
    top_k_operations: list[TopKSpec] = dc_field(default_factory=list)
    window_aggregates: list[WindowAggregateSpec] = dc_field(default_factory=list)
    cumulative_counts: list[CumulativeCountSpec] = dc_field(default_factory=list)
    cumulative_durations: list[CumulativeDurationSpec] = dc_field(default_factory=list)
    comparisons: list[ComparisonSpec] = dc_field(default_factory=list)
    arithmetic: list[ArithmeticSpec] = dc_field(default_factory=list)
    relation_filters: list[RelationFilterSpec] = dc_field(default_factory=list)
    conversions: list[ConversionSpec] = dc_field(default_factory=list)
    ordered_folds: list[OrderedFoldSpec] = dc_field(default_factory=list)
    sequences: list[SequenceSpec] = dc_field(default_factory=list)
    unsatisfiable: list[UnsatisfiableSpec] = dc_field(default_factory=list)
    operator_invocations: list[OperatorInvocation] = dc_field(default_factory=list)
    content_annotations: list[ContentAnnotation] = dc_field(default_factory=list)
    references: list[ReferenceSpec] = dc_field(default_factory=list)
    output: OutputContract = dc_field(default_factory=OutputContract)
    unresolved: list[UnresolvedItem] = dc_field(default_factory=list)
    diagnostics: list[Diagnostic] = dc_field(default_factory=list)
    claims: list[SemanticClaim] = dc_field(default_factory=list)
    requirements: list[RequirementSpec] = dc_field(default_factory=list)
    coverage: list[CoverageSpec] = dc_field(default_factory=list)
    ambiguities: list[AmbiguitySpec] = dc_field(default_factory=list)
    # Answers are only ever written by ClarificationPlanner against an
    # approved resume path, after a fresh compilation.  They are never used as
    # an in-place mutation of typed semantic nodes.
    clarification_answers: dict[str, str] = dc_field(default_factory=dict)
    turn_directives: list[TurnDirective] = dc_field(default_factory=list)
    semantic_status: str = "partial"  # complete / partial / ambiguous
    understanding_status: str = "partial"  # complete / partial / ambiguous
    binding_status: str = "unbound"  # bound / partial / unbound
    capability_status: str = "unbound"  # available / unbound / unsupported
    execution_status: str = "blocked"  # ready / blocked
    compatibility_intent: str = "unknown"
    reliability: float = 0.0


# RequestIR is the preferred name. UnderstandingIR remains the compatibility name
# used by the existing validator, CLI, and legacy adapter.
RequestIR = UnderstandingIR


@dataclass(slots=True)
class ResultField(Serializable):
    name: str
    data_type: str
    unit: Optional[str] = None


@dataclass(slots=True)
class TaskNode(Serializable):
    task_id: str
    task_type: str
    depends_on: list[str] = dc_field(default_factory=list)
    source_id: Optional[str] = None
    inputs: dict[str, Any] = dc_field(default_factory=dict)
    output_schema: list[ResultField] = dc_field(default_factory=list)
    required_capabilities: list[str] = dc_field(default_factory=list)
    status: str = "planned"  # planned / blocked
    requirement_ids: list[str] = dc_field(default_factory=list)
    security_scope_digest: str = ""
    side_effect: str = "none"
    binding_status: str = "unbound"


@dataclass(slots=True)
class LogicalPlan(Serializable):
    schema_version: str
    catalog_version: str
    nodes: list[TaskNode] = dc_field(default_factory=list)
    status: str = "blocked"  # ready / blocked / unsupported
    reason: str = ""


@dataclass(slots=True)
class ToolBinding(Serializable):
    task_id: str
    tool_name: str
    arguments: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(slots=True)
class PhysicalPlan(Serializable):
    schema_version: str
    catalog_version: str
    bindings: list[ToolBinding] = dc_field(default_factory=list)
    status: str = "unbound"
    reason: str = "Physical binding belongs to the root Agent SkillRegistry"


@dataclass(slots=True)
class ValidationReport(Serializable):
    status: str  # valid / needs_clarification / unsupported
    executable: bool
    reliability: float
    errors: list[Diagnostic] = dc_field(default_factory=list)
    understanding_complete: bool = False
    binding_complete: bool = False
    authorization_status: str = "not_evaluated"
    dimensions: dict[str, dict[str, Any]] = dc_field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent(Serializable):
    stage: str
    detail: str
    duration_ms: float = 0.0


@dataclass(slots=True)
class LLMAuditRecord(Serializable):
    """Bounded, in-memory evidence for one model repair attempt."""

    provider: str = ""
    model: str = ""
    repair_unit_id: str = ""
    gap_ids: list[str] = dc_field(default_factory=list)
    atomic_operation: str = ""
    protocol: str = ""
    attempts: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    raw_response_excerpt: str = ""
    parsed_json_excerpt: str = ""
    parse_error: str = ""
    failure_kind: str = ""
    duration_ms: float = 0.0
    stop_reason: str = ""
    rejected_gate: str = ""
    rejection_reason: str = ""
    coverage_before: dict[str, str] = dc_field(default_factory=dict)
    coverage_after: dict[str, str] = dc_field(default_factory=dict)
    closed_requirement_ids: list[str] = dc_field(default_factory=list)
    source_clause_digest: str = ""
    target_contract_digest: str = ""
    base_lineage_digest: str = ""
    candidate_menu_digest: str = ""
    candidate_compiler_version: str = ""
    expression_signature_version: str = ""
    prompt_version: str = ""
    model_identity: str = ""
    decision: str = ""
    candidate_id: str = ""


@dataclass(slots=True)
class EngineResult(Serializable):
    understanding: UnderstandingIR
    logical_plan: LogicalPlan
    physical_plan: PhysicalPlan
    validation: ValidationReport
    trace: list[TraceEvent] = dc_field(default_factory=list)
    model_calls: int = 0
    patch_report: Optional[PatchReport] = None
    semantic_effect: Optional[SemanticEffectReport] = None
    clarification_plan: Any = None
    compilation_snapshot: Any = None
    llm_audit: Optional[LLMAuditRecord] = None
    event_closure_report: Any = None
    canonical_requirement_graph: Any = None
    field_binding_report: Any = None
    quantity_report: Any = None
    event_readiness_report: Any = None
    typed_lineage_report: Any = None
    m15_logical_plan: Any = None
    reference_binding_report: Any = None
