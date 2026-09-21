"""Typed, evidence-gated SemanticPatch protocol for bounded LLM enrichment."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .catalog import CatalogSnapshot
from .merge import merge_llm_candidate, walk_predicates
from .models import (
    AggregateSpec,
    AmbiguitySpec,
    CalculationSpec,
    CandidateRef,
    ComparisonSpec,
    ContextDelta,
    ContextDeltaOperation,
    DerivedMetricCall,
    DurationConstraint,
    EventSpec,
    ExpressionNode,
    LiteralValue,
    OutputRef,
    PatchOperationResult,
    PatchReport,
    Provenance,
    PredicateNode,
    SamplingPolicy,
    SequenceSpec,
    SetOperationSpec,
    ScopedAggregateSpec,
    SourceSpan,
    SymbolRef,
    TurnDirective,
    UnderstandingIR,
    WindowSpec,
)


PATCH_SCHEMA_VERSION = "semantic_patch_v1"
ATOMIC_PATCH_SCHEMA_VERSION = "semantic_patch_v2"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceSpanModel(_StrictModel):
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)

    @field_validator("end")
    @classmethod
    def _end_requires_start(cls, value: int | None, info):
        start = info.data.get("start")
        if (start is None) != (value is None):
            raise ValueError("start and end must be provided together")
        if start is not None and value is not None and value <= start:
            raise ValueError("end must be greater than start")
        return value


class GapRequest(_StrictModel):
    gap_id: str = Field(min_length=1)
    gap_type: Literal[
        "missing_event", "missing_set_inputs", "missing_formula_input",
        "missing_aggregate_scope", "unresolved_reference", "unresolved_symbol",
        "ambiguous_formula", "context_required", "missing_predicate",
        "missing_conversion", "missing_turn_directive", "missing_output",
        "missing_sequence",
    ]
    target_node_id: str = ""
    required_slots: list[str] = Field(default_factory=list)
    query_slice: str
    allowed_catalog_symbols: list[str] = Field(default_factory=list)
    allowed_hypothesis_symbols: list[str] = Field(default_factory=list)
    allowed_metrics: list[str] = Field(default_factory=list)
    available_output_refs: list[str] = Field(default_factory=list)
    allowed_operations: list[str] = Field(default_factory=list)
    allowed_anchor_ids: list[str] = Field(default_factory=list)
    existing_node_summary: dict[str, Any] = Field(default_factory=dict)
    requirement_id: str = ""
    # RequirementIR dependency edges are copied into the work order so RepairUnit
    # construction can keep one semantic DAG connected without treating the
    # entire catalogue of *available* OutputRefs as dependencies.
    requirement_dependency_ids: list[str] = Field(default_factory=list)
    clause_id: str = ""
    resolution_mode: Literal["llm", "deterministic_terminal"] = "llm"
    readiness: Literal["ready", "blocked", "deterministic"] = "ready"
    blocked_by: list[str] = Field(default_factory=list)
    choice_domain: dict[str, list[str]] = Field(default_factory=dict)
    # Recorded fixtures and third-party providers retain the original
    # node-shaped protocol. GapAnalyzer emits semantic choices for the bounded
    # local-model path; transaction metadata and AST nodes remain local-owned.
    response_mode: Literal["legacy_node", "semantic_choice"] = "legacy_node"


class PatchPrecondition(_StrictModel):
    kind: Literal["gap_open", "node_absent", "node_present"]
    value: str


class ExpressionPatch(_StrictModel):
    kind: Literal[
        "literal", "catalog_symbol", "hypothesis_symbol", "output_ref",
        "binary", "function",
    ]
    identifier: str = ""
    operator: str = ""
    arguments: list["ExpressionPatch"] = Field(default_factory=list)
    value: Any = None
    value_type: str = "unknown"
    result_type: str = "unknown"
    unit: str | None = None


class ConditionPatch(_StrictModel):
    field_id: str
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "contains"]
    value: Any = None
    unit: str | None = None
    right_expression: ExpressionPatch | None = None
    span: EvidenceSpanModel


class SequenceDurationPatch(_StrictModel):
    operator: Literal["lt", "lte", "gt", "gte"] = "lte"
    value: float = Field(gt=0)
    unit: Literal["second", "minute", "hour", "day"]
    span: EvidenceSpanModel


class DurationPatch(_StrictModel):
    operator: Literal["gt", "gte"]
    value: float = Field(gt=0)
    unit: Literal["second", "minute", "hour", "day"]
    span: EvidenceSpanModel


class EventPatch(_StrictModel):
    event_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    logic: Literal["and", "or"] = "and"
    conditions: list[ConditionPatch] = Field(min_length=1)
    duration: DurationPatch
    group_by: list[str] = Field(default_factory=lambda: ["local_date"])
    span: EvidenceSpanModel


class TurnDeltaPatch(_StrictModel):
    operation: Literal["add", "replace", "remove"]
    slot: str = Field(min_length=1)
    value: Any = None


class TurnDirectivePatch(_StrictModel):
    directive_id: str
    pre_actions: list[Literal["cancel_previous", "suspend_previous"]] = Field(default_factory=list)
    primary_action: Literal["new_query", "refine", "replace_constraints", "clarify"]
    target_turn_id: str | None = None
    context_mode: Literal["fresh", "inherit"] = "fresh"
    base_context_ref: str | None = None
    delta_owner: Literal["previous_turn", "new_turn"] = "new_turn"
    delta: list[TurnDeltaPatch] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)


class _OperationBase(_StrictModel):
    operation_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    target_node_id: str = ""
    preconditions: list[PatchPrecondition] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    dependency_group: str = ""
    evidence: list[EvidenceSpanModel] = Field(min_length=1)
    # New producers must select these stable source ids.  The empty default
    # retains import compatibility with pre-contract recorded fixtures; the
    # Engine rejects unbound concrete patches before commit.
    anchor_ids: list[str] = Field(default_factory=list)


class AddEventOperation(_OperationBase):
    operation_type: Literal["add_event"]
    event: EventPatch


class AddSetOperation(_OperationBase):
    operation_type: Literal["add_set_operation"]
    operation: Literal["intersection", "union", "difference"]
    inputs: list[str] = Field(min_length=2)
    granularity: Literal["date", "interval"] = "date"


class AddAggregateOperation(_OperationBase):
    operation_type: Literal["add_aggregate"]
    aggregate_id: str
    function: Literal["sum", "avg", "min", "max", "count", "stddev"]
    input: ExpressionPatch
    scope: Literal["filtered", "global", "relation"]
    scope_ref: str | None = None
    group_by: list[ExpressionPatch] = Field(default_factory=list)


class AddComparisonOperation(_OperationBase):
    operation_type: Literal["add_comparison"]
    comparison_id: str
    left: ExpressionPatch
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    right: ExpressionPatch


class AddCalculationOperation(_OperationBase):
    operation_type: Literal["add_calculation"]
    calculation_type: Literal[
        "mom", "yoy", "growth", "difference", "ratio", "volatility",
        "correlation", "product", "maximum",
    ]
    expression: ExpressionPatch


class AddSequenceOperation(_OperationBase):
    operation_type: Literal["add_sequence"]
    sequence_id: str = Field(min_length=1)
    steps: list[ConditionPatch] = Field(min_length=2)
    partition_by: list[str] = Field(min_length=1)
    order_by: str = Field(min_length=1)
    max_duration: SequenceDurationPatch | None = None
    missing_data_policy: Literal["reject", "break", "skip"] = "reject"


class ResolveReferenceOperation(_OperationBase):
    operation_type: Literal["resolve_reference"]
    reference_id: str
    target_ref: str


class AddAmbiguityOperation(_OperationBase):
    operation_type: Literal["add_ambiguity"]
    ambiguity_id: str
    kind: str
    message: str
    candidates: list[str] = Field(default_factory=list)


class AddTurnDirectiveOperation(_OperationBase):
    operation_type: Literal["add_turn_directive"]
    directive: TurnDirectivePatch


PatchOperation = Annotated[
    Union[
        AddEventOperation,
        AddSetOperation,
        AddAggregateOperation,
        AddComparisonOperation,
        AddCalculationOperation,
        AddSequenceOperation,
        ResolveReferenceOperation,
        AddAmbiguityOperation,
        AddTurnDirectiveOperation,
    ],
    Field(discriminator="operation_type"),
]


class SemanticPatchV1(_StrictModel):
    schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    base_ir_digest: str = Field(min_length=1)
    operations: list[PatchOperation] = Field(default_factory=list)


class _AtomicResponse(_StrictModel):
    # Source evidence and coordinates are compiler-owned transaction metadata.
    # The optional field remains accepted for migration-era providers but is
    # never trusted when constructing a Patch operation.
    evidence_text: str = ""


class AtomicEventValue(_StrictModel):
    logic: Literal["and", "or"] = "and"
    conditions: list[ConditionPatch] = Field(min_length=1)
    duration: DurationPatch
    group_by: list[str] = Field(default_factory=lambda: ["local_date"])
    span: EvidenceSpanModel


class AtomicEventResponse(_AtomicResponse):
    event: AtomicEventValue


class AtomicSetResponse(_AtomicResponse):
    operation: Literal["intersection", "union", "difference"]
    inputs: list[str] = Field(min_length=2)
    granularity: Literal["date", "interval"] = "date"


class AtomicAggregateResponse(_AtomicResponse):
    function: Literal["sum", "avg", "min", "max", "count", "stddev"]
    input: ExpressionPatch
    scope: Literal["filtered", "global", "relation"]
    scope_ref: str | None = None
    group_by: list[ExpressionPatch] = Field(default_factory=list)


class AtomicComparisonResponse(_AtomicResponse):
    left: ExpressionPatch
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    right: ExpressionPatch


class AtomicCalculationResponse(_AtomicResponse):
    calculation_type: Literal["mom", "yoy", "growth", "difference", "ratio", "volatility", "correlation"]
    expression: ExpressionPatch


class AtomicSequenceResponse(_AtomicResponse):
    steps: list[ConditionPatch] = Field(min_length=2)
    partition_by: list[str] = Field(min_length=1)
    order_by: str
    max_duration: SequenceDurationPatch | None = None
    missing_data_policy: Literal["reject", "break", "skip"] = "reject"


class AtomicReferenceResponse(_AtomicResponse):
    target_ref: str


class AtomicAmbiguityResponse(_AtomicResponse):
    kind: str
    message: str
    candidates: list[str] = Field(default_factory=list)


class AtomicTurnResponse(_AtomicResponse):
    directive: TurnDirectivePatch


class AtomicConditionChoice(_StrictModel):
    field_ref: str = Field(min_length=1)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "contains"]
    value: Any = None
    unit: str | None = None
    right_ref: str | None = None
    multiplier: float | None = None


class AtomicEventChoiceResponse(_AtomicResponse):
    logic: Literal["and", "or"] = "and"
    conditions: list[AtomicConditionChoice] = Field(min_length=1)
    duration_operator: Literal["gt", "gte"]
    duration_value: float = Field(gt=0)
    duration_unit: Literal["second", "minute", "hour", "day"]
    group_by: list[str] = Field(default_factory=lambda: ["local_date"])


class AtomicAggregateChoiceResponse(_AtomicResponse):
    function: Literal["sum", "avg", "min", "max", "count", "stddev"]
    input_ref: str = Field(min_length=1)
    scope: Literal["filtered", "global", "relation"]
    scope_ref: str | None = None
    group_by_refs: list[str] = Field(default_factory=list)


class AtomicComparisonChoiceResponse(_AtomicResponse):
    left_ref: str = Field(min_length=1)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    right_ref: str | None = None
    right_value: float | None = None
    right_unit: str | None = None


class AtomicCalculationChoiceResponse(_AtomicResponse):
    calculation_type: Literal[
        "mom", "yoy", "growth", "difference", "ratio", "volatility",
        "correlation", "product", "maximum",
    ]
    formula_template: Literal[
        "identity", "difference", "ratio", "product", "sum",
        "growth_rate", "stddev", "correlation", "max_of_product",
    ]
    operands: list[str] = Field(min_length=1)
    constants: list[float] = Field(default_factory=list)


class AtomicReferenceChoiceResponse(_AtomicResponse):
    target_ref: str = Field(min_length=1)


ATOMIC_RESPONSE_MODELS = {
    "add_event": AtomicEventResponse,
    "add_set_operation": AtomicSetResponse,
    "add_aggregate": AtomicAggregateResponse,
    "add_comparison": AtomicComparisonResponse,
    "add_calculation": AtomicCalculationResponse,
    "add_sequence": AtomicSequenceResponse,
    "resolve_reference": AtomicReferenceResponse,
    "add_ambiguity": AtomicAmbiguityResponse,
    "add_turn_directive": AtomicTurnResponse,
}

SEMANTIC_CHOICE_RESPONSE_MODELS = {
    "add_event": AtomicEventChoiceResponse,
    "add_set_operation": AtomicSetResponse,
    "add_aggregate": AtomicAggregateChoiceResponse,
    "add_comparison": AtomicComparisonChoiceResponse,
    "add_calculation": AtomicCalculationChoiceResponse,
    "resolve_reference": AtomicReferenceChoiceResponse,
}


def atomic_operation_type(gap: GapRequest) -> str:
    return next((item for item in gap.allowed_operations if item in ATOMIC_RESPONSE_MODELS), "add_ambiguity")


def atomic_response_schema(gap: GapRequest) -> dict[str, Any]:
    operation_type = atomic_operation_type(gap)
    models = (SEMANTIC_CHOICE_RESPONSE_MODELS
              if gap.response_mode == "semantic_choice" else ATOMIC_RESPONSE_MODELS)
    model = models.get(operation_type, ATOMIC_RESPONSE_MODELS[operation_type])
    schema = model.model_json_schema()
    if gap.response_mode == "semantic_choice":
        _inject_choice_enums(schema, gap)
    return schema


def parse_atomic_patch_text(text: str, *, base_ir_digest: str, gap: GapRequest
                            ) -> tuple[SemanticPatchV1 | None, str, str]:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        return None, ATOMIC_PATCH_SCHEMA_VERSION, f"invalid LLM JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, ATOMIC_PATCH_SCHEMA_VERSION, "LLM output must be one JSON object"
    # Frozen v1 providers remain accepted during the migration window.
    if "schema_version" in payload or "operations" in payload:
        return parse_patch_text(value, base_ir_digest=base_ir_digest, gaps=[gap])
    if not payload:
        return SemanticPatchV1(base_ir_digest=base_ir_digest), ATOMIC_PATCH_SCHEMA_VERSION, ""
    operation_type = atomic_operation_type(gap)
    models = (SEMANTIC_CHOICE_RESPONSE_MODELS
              if gap.response_mode == "semantic_choice" else ATOMIC_RESPONSE_MODELS)
    model = models.get(operation_type, ATOMIC_RESPONSE_MODELS[operation_type])
    try:
        # Migration compatibility: older atomic producers copied opaque IDs
        # and source-anchor IDs.  They are deliberately ignored now because
        # these values are transaction metadata owned by the local compiler.
        compact_payload = _compact_atomic_payload(payload, operation_type)
        content = model.model_validate(compact_payload)
        values = content.model_dump(mode="json")
        model_evidence_text = values.pop("evidence_text", "")
        evidence_text = (
            gap.query_slice.strip() if gap.response_mode == "semantic_choice"
            else str(model_evidence_text).strip()
        )
        if not evidence_text:
            raise ValueError("atomic GapRequest has no local evidence text")
        anchor_ids = list(gap.allowed_anchor_ids)
        operation_id = "op_" + hashlib.sha256(
            f"{gap.gap_id}|{operation_type}|{evidence_text}".encode("utf-8")
        ).hexdigest()[:16]
        common = {
            "operation_type": operation_type, "operation_id": operation_id,
            "gap_id": gap.gap_id, "target_node_id": gap.target_node_id,
            "preconditions": [{"kind": "gap_open", "value": gap.gap_id}],
            "dependencies": [], "evidence": [{"text": evidence_text}], "anchor_ids": anchor_ids,
        }
        if gap.response_mode == "semantic_choice":
            operation = _semantic_choice_operation(
                operation_type, values, gap, operation_id, common, evidence_text,
            )
        elif operation_type == "add_event":
            event_values = _expand_atomic_event_span(values["event"], gap.query_slice)
            event_values["event_id"] = _local_node_id("event", gap, operation_id)
            event_values["metric_id"] = _metric_for_gap(gap)
            operation = AddEventOperation(**common, event=event_values)
        elif operation_type == "add_set_operation":
            operation = AddSetOperation(**common, operation=values["operation"],
                                        inputs=values["inputs"], granularity=values["granularity"])
        elif operation_type == "add_aggregate":
            operation = AddAggregateOperation(
                **common, aggregate_id=_local_node_id("aggregate", gap, operation_id), **values,
            )
        elif operation_type == "add_comparison":
            operation = AddComparisonOperation(
                **common, comparison_id=_local_node_id("comparison", gap, operation_id), **values,
            )
        elif operation_type == "add_calculation":
            operation = AddCalculationOperation(**common, **values)
        elif operation_type == "add_sequence":
            operation = AddSequenceOperation(
                **common, sequence_id=_local_node_id("sequence", gap, operation_id), **values,
            )
        elif operation_type == "resolve_reference":
            reference_id = str(gap.existing_node_summary.get("reference_id") or gap.target_node_id)
            operation = ResolveReferenceOperation(**common, reference_id=reference_id, **values)
        elif operation_type == "add_turn_directive":
            operation = AddTurnDirectiveOperation(**common, **values)
        else:
            operation = AddAmbiguityOperation(
                **common, ambiguity_id=_local_node_id("ambiguity", gap, operation_id), **values,
            )
        return SemanticPatchV1(
            base_ir_digest=base_ir_digest, operations=[operation]
        ), ATOMIC_PATCH_SCHEMA_VERSION, ""
    except ValidationError as exc:
        legacy = legacy_payload_to_patch(payload, base_ir_digest=base_ir_digest, gaps=[gap])
        if legacy.operations or _looks_like_legacy_payload(payload):
            return legacy, "legacy_fragment_adapter", ""
        return None, ATOMIC_PATCH_SCHEMA_VERSION, f"invalid atomic Patch content: {exc}"


def _compact_atomic_payload(payload: Mapping[str, Any], operation_type: str) -> dict[str, Any]:
    """Remove transaction metadata emitted by migration-era providers."""
    result = dict(payload)
    result.pop("anchor_ids", None)
    if operation_type == "add_event" and isinstance(result.get("event"), Mapping):
        event = dict(result["event"])
        event.pop("event_id", None)
        event.pop("metric_id", None)
        result["event"] = event
    for key in ("aggregate_id", "comparison_id", "sequence_id", "reference_id", "ambiguity_id"):
        result.pop(key, None)
    return result


def _inject_choice_enums(schema: dict[str, Any], gap: GapRequest) -> None:
    """Restrict semantic-choice strings to compiler-owned allow-lists."""
    fields = gap.choice_domain.get("field_refs", [])
    outputs = gap.choice_domain.get("output_refs", [])
    metrics = gap.choice_domain.get("metric_refs", [])
    units = gap.choice_domain.get("unit_refs", [])
    all_inputs = sorted(dict.fromkeys([*fields, *outputs]))
    enum_by_name = {
        "field_ref": fields,
        "right_ref": all_inputs,
        "input_ref": all_inputs,
        "left_ref": all_inputs,
        "target_ref": outputs,
        "scope_ref": outputs,
        "order_by": fields,
        "unit": units,
        "right_unit": units,
        "function": gap.choice_domain.get("functions", []),
        "scope": gap.choice_domain.get("scopes", []),
        "calculation_type": gap.choice_domain.get("calculation_types", []),
        "formula_template": gap.choice_domain.get("formula_templates", []),
        "operation": gap.choice_domain.get("set_operations", []),
    }
    array_enum_by_name = {
        "inputs": outputs,
        "operands": all_inputs,
        "group_by_refs": all_inputs,
        "partition_by": fields,
    }

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                choices = enum_by_name.get(name)
                if choices:
                    nullable = isinstance(child, dict) and any(
                        item.get("type") == "null" for item in child.get("anyOf", [])
                        if isinstance(item, dict)
                    )
                    replacement: dict[str, Any] = {"type": "string", "enum": choices}
                    if nullable:
                        replacement = {
                            "anyOf": [replacement, {"type": "null"}],
                            "default": child.get("default"),
                        }
                    properties[name] = replacement
                    child = replacement
                array_choices = array_enum_by_name.get(name)
                if array_choices and isinstance(child, dict):
                    child["items"] = {"type": "string", "enum": array_choices}
                    if name == "inputs":
                        child["minItems"] = 2
                        child["maxItems"] = 2
                        child["uniqueItems"] = True
                visit(child)
        for name, child in node.items():
            if name != "properties":
                if isinstance(child, list):
                    for item in child:
                        visit(item)
                else:
                    visit(child)

    visit(schema)
    if metrics:
        schema.setdefault("x-allowed-metrics", metrics)


def _semantic_choice_operation(operation_type: str, values: dict[str, Any], gap: GapRequest,
                               operation_id: str, common: dict[str, Any],
                               evidence_text: str) -> PatchOperation:
    """Compile a small model choice into the full typed transaction locally."""
    if operation_type == "add_event":
        conditions = []
        for item in values["conditions"]:
            right = None
            if item.get("right_ref"):
                right = _choice_expression(item["right_ref"], gap)
                if item.get("multiplier") is not None:
                    right = ExpressionPatch(
                        kind="binary", operator="mul", arguments=[
                            right,
                            ExpressionPatch(
                                kind="literal", value=item["multiplier"],
                                value_type="number", result_type="number",
                            ),
                        ], result_type="number", unit=item.get("unit"),
                    )
            conditions.append(ConditionPatch(
                field_id=item["field_ref"], operator=item["operator"],
                value=None if right is not None else item.get("value"),
                unit=item.get("unit"), right_expression=right,
                span=EvidenceSpanModel(text=evidence_text),
            ))
        event = EventPatch(
            event_id=_local_node_id("event", gap, operation_id),
            metric_id=_metric_for_gap(gap), logic=values.get("logic", "and"),
            conditions=conditions,
            duration=DurationPatch(
                operator=values["duration_operator"], value=values["duration_value"],
                unit=values["duration_unit"], span=EvidenceSpanModel(text=evidence_text),
            ),
            group_by=values.get("group_by") or ["local_date"],
            span=EvidenceSpanModel(text=evidence_text),
        )
        return AddEventOperation(**common, event=event)
    if operation_type == "add_set_operation":
        return AddSetOperation(
            **common, operation=values["operation"], inputs=values["inputs"],
            granularity=values.get("granularity", "date"),
        )
    if operation_type == "add_aggregate":
        group_by = [_choice_expression(item, gap) for item in values.get("group_by_refs", [])]
        return AddAggregateOperation(
            **common, aggregate_id=_local_node_id("aggregate", gap, operation_id),
            function=values["function"], input=_choice_expression(values["input_ref"], gap),
            scope=values["scope"], scope_ref=values.get("scope_ref"), group_by=group_by,
        )
    if operation_type == "add_comparison":
        right = (_choice_expression(values["right_ref"], gap)
                 if values.get("right_ref") else ExpressionPatch(
                     kind="literal", value=values.get("right_value"), value_type="number",
                     result_type="number", unit=values.get("right_unit"),
                 ))
        return AddComparisonOperation(
            **common, comparison_id=_local_node_id("comparison", gap, operation_id),
            left=_choice_expression(values["left_ref"], gap),
            operator=values["operator"], right=right,
        )
    if operation_type == "add_calculation":
        expression = _choice_formula(
            values["formula_template"], values["operands"], values.get("constants", []), gap,
        )
        return AddCalculationOperation(
            **common, calculation_type=values["calculation_type"], expression=expression,
        )
    if operation_type == "resolve_reference":
        reference_id = str(gap.existing_node_summary.get("reference_id") or gap.target_node_id)
        return ResolveReferenceOperation(
            **common, reference_id=reference_id, target_ref=values["target_ref"],
        )
    if operation_type == "add_turn_directive":
        directive = dict(values["directive"])
        directive["directive_id"] = _local_node_id("turn", gap, operation_id)
        return AddTurnDirectiveOperation(**common, directive=directive)
    raise ValueError(f"semantic choice is unsupported for {operation_type}")


def _choice_expression(identifier: str, gap: GapRequest) -> ExpressionPatch:
    if identifier in gap.choice_domain.get("output_refs", []):
        kind = "output_ref"
    elif identifier in gap.allowed_hypothesis_symbols:
        kind = "hypothesis_symbol"
    elif identifier in gap.allowed_catalog_symbols:
        kind = "catalog_symbol"
    else:
        raise ValueError(f"choice outside allowed input domain: {identifier}")
    return ExpressionPatch(kind=kind, identifier=identifier, result_type="unknown")


def _choice_formula(template: str, operands: list[str], constants: list[float],
                    gap: GapRequest) -> ExpressionPatch:
    args = [_choice_expression(item, gap) for item in operands]
    args.extend(ExpressionPatch(
        kind="literal", value=value, value_type="number", result_type="number",
    ) for value in constants)
    arity = {
        "identity": 1, "difference": 2, "ratio": 2, "product": 2,
        "sum": 2, "growth_rate": 2, "stddev": 1, "correlation": 2,
        "max_of_product": 2,
    }[template]
    if len(args) != arity:
        raise ValueError(f"{template} requires {arity} operands, received {len(args)}")
    if template == "identity":
        return args[0]
    if template in {"difference", "ratio", "product", "sum"}:
        operator = {"difference": "sub", "ratio": "div", "product": "mul", "sum": "add"}[template]
        return ExpressionPatch(kind="binary", operator=operator, arguments=args,
                               result_type="unknown")
    if template == "growth_rate":
        delta = ExpressionPatch(kind="binary", operator="sub", arguments=args,
                                result_type="unknown")
        return ExpressionPatch(kind="binary", operator="div", arguments=[delta, args[1]],
                               result_type="unknown")
    if template == "max_of_product":
        product = ExpressionPatch(kind="binary", operator="mul", arguments=args,
                                  result_type="unknown")
        return ExpressionPatch(kind="function", operator="max", arguments=[product],
                               result_type="unknown")
    return ExpressionPatch(kind="function", operator=template, arguments=args,
                           result_type="unknown")


def _local_node_id(prefix: str, gap: GapRequest, operation_id: str) -> str:
    suffix = hashlib.sha256(f"{gap.gap_id}|{operation_id}|{prefix}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{suffix}"


def _metric_for_gap(gap: GapRequest) -> str:
    requirement_type = str(gap.existing_node_summary.get("requirement_type", ""))
    suffix = "cumulative_duration" if requirement_type == "cumulative_duration" else "consecutive_duration"
    return next(
        (metric for metric in gap.allowed_metrics if metric.endswith(suffix)),
        f"semantic.{suffix}",
    )


def _expand_atomic_event_span(event: dict[str, Any], query_slice: str) -> dict[str, Any]:
    """Derive the enclosing event phrase from model-provided atomic evidence."""
    conditions = event.get("conditions") or []
    duration = event.get("duration") or {}
    condition_text = str((conditions[0].get("span") or {}).get("text") or "") if conditions else ""
    duration_text = str((duration.get("span") or {}).get("text") or "")
    if not condition_text or not duration_text:
        return event
    condition_start = query_slice.find(condition_text)
    duration_start = query_slice.find(duration_text)
    if condition_start < 0 or duration_start < condition_start:
        return event
    end = duration_start + len(duration_text)
    expanded = dict(event)
    expanded["span"] = {"text": query_slice[condition_start:end]}
    return expanded


ExpressionPatch.model_rebuild()


def request_ir_digest(ir: UnderstandingIR) -> str:
    payload = ir.to_dict()
    for key in ("claims", "requirements", "ambiguities", "diagnostics", "reliability"):
        payload.pop(key, None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_patch_text(text: str, *, base_ir_digest: str,
                     gaps: list[GapRequest]) -> tuple[SemanticPatchV1 | None, str, str]:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        return None, "semantic_patch_v1", f"invalid LLM JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, "semantic_patch_v1", "LLM output must be one JSON object"
    if not payload:
        return SemanticPatchV1(base_ir_digest=base_ir_digest), "semantic_patch_v1", ""
    try:
        return SemanticPatchV1.model_validate(payload), "semantic_patch_v1", ""
    except ValidationError as strict_error:
        legacy = legacy_payload_to_patch(payload, base_ir_digest=base_ir_digest, gaps=gaps)
        if legacy.operations or _looks_like_legacy_payload(payload):
            return legacy, "legacy_fragment_adapter", ""
        return None, "semantic_patch_v1", f"invalid SemanticPatchV1: {strict_error}"


def legacy_payload_to_patch(payload: Mapping[str, Any], *, base_ir_digest: str,
                            gaps: list[GapRequest]) -> SemanticPatchV1:
    """Temporary adapter for the pre-2.4 fragment protocol and frozen fixtures."""
    operations: list[PatchOperation] = []
    index = 0

    def gap_for(kind: str) -> GapRequest | None:
        return next((item for item in gaps if kind in item.allowed_operations), gaps[0] if gaps else None)

    for item in _legacy_items(payload.get("events")):
        index += 1
        try:
            gap = gap_for("add_event")
            operations.append(AddEventOperation(
                operation_type="add_event", operation_id=f"legacy_{index}",
                gap_id=gap.gap_id if gap else "legacy_unbound_gap",
                target_node_id=gap.target_node_id if gap else "",
                preconditions=[PatchPrecondition(kind="gap_open", value=gap.gap_id)] if gap else [],
                anchor_ids=list(gap.allowed_anchor_ids) if gap else [],
                evidence=[item.get("span", {})],
                event=item,
            ))
        except ValidationError:
            continue
    for item in _legacy_items(payload.get("set_operations")):
        index += 1
        try:
            gap = gap_for("add_set_operation")
            operations.append(AddSetOperation(
                operation_type="add_set_operation", operation_id=f"legacy_{index}",
                gap_id=gap.gap_id if gap else "legacy_unbound_gap",
                target_node_id=gap.target_node_id if gap else "",
                preconditions=[PatchPrecondition(kind="gap_open", value=gap.gap_id)] if gap else [],
                anchor_ids=list(gap.allowed_anchor_ids) if gap else [],
                evidence=[item.get("span", {})],
                operation=item.get("operation"), inputs=item.get("inputs", []),
                granularity=item.get("granularity", "date"),
            ))
        except ValidationError:
            continue
    return SemanticPatchV1(base_ir_digest=base_ir_digest, operations=operations)


def apply_semantic_patch(base: UnderstandingIR, patch: SemanticPatchV1,
                         gaps: list[GapRequest], catalog: CatalogSnapshot,
                         *, atomic_unit: bool = False,
                         ) -> tuple[UnderstandingIR, PatchReport]:
    report = PatchReport(base_ir_digest=patch.base_ir_digest)
    expected_digest = request_ir_digest(base)
    if patch.base_ir_digest != expected_digest:
        report.transaction_error = "base_ir_digest_mismatch"
        report.rejection_gate = "patch_protocol_gate"
        return copy.deepcopy(base), report

    gap_map = {item.gap_id: item for item in gaps}
    working = copy.deepcopy(base)
    applied_operation_ids: set[str] = set()
    known_nodes = _known_node_ids(working)

    groups = [list(patch.operations)] if atomic_unit and patch.operations else _operation_groups(patch.operations)
    for group in groups:
        group_start = copy.deepcopy(working)
        group_known_nodes = set(known_nodes)
        group_applied_ids = set(applied_operation_ids)
        accepted_start = len(report.accepted)
        handled: set[str] = set()
        failed = False
        for operation in group:
            handled.add(operation.operation_id)
            result = PatchOperationResult(
                operation.operation_id, operation.operation_type, operation.gap_id, "rejected",
            )
            gap = gap_map.get(operation.gap_id)
            if gap is None:
                result.reason = "unknown_gap"
            elif operation.operation_type not in gap.allowed_operations:
                result.reason = "operation_not_allowed_for_gap"
            elif operation.operation_id in applied_operation_ids:
                result.reason = "duplicate_operation_id"
            else:
                span = _validated_operation_span(operation.evidence, working.query.normalized)
                if span is None:
                    result.reason = "invalid_operation_evidence"
                elif any(dep not in known_nodes and dep not in applied_operation_ids
                         for dep in operation.dependencies):
                    result.reason = "dependency_unresolved"
                elif not _preconditions_hold(operation.preconditions, gap_map, known_nodes):
                    result.reason = "precondition_failed"
                else:
                    handler = PATCH_APPLIERS[operation.operation_type]
                    before = _semantic_shape(working)
                    accepted, reason = handler(working, operation, span, catalog)
                    after = _semantic_shape(working)
                    if accepted and before != after:
                        result.status = "accepted"
                        report.accepted.append(result)
                        applied_operation_ids.add(operation.operation_id)
                        known_nodes.update(_known_node_ids(working))
                    else:
                        result.reason = reason or "operation_no_gain"
            if result.status != "accepted":
                report.rejected.append(result)
                failed = True
                break
        if failed and len(group) > 1:
            # A named dependency group is all-or-nothing.  Independent
            # operations use their own implicit group and keep M1.2 behavior.
            working = group_start
            known_nodes = group_known_nodes
            applied_operation_ids = group_applied_ids
            rolled_back = report.accepted[accepted_start:]
            del report.accepted[accepted_start:]
            report.rejected.extend(PatchOperationResult(
                item.operation_id, item.operation_type, item.gap_id, "rejected",
                "dependency_group_rolled_back",
            ) for item in rolled_back)
            report.rejected.extend(PatchOperationResult(
                operation.operation_id, operation.operation_type, operation.gap_id, "rejected",
                "dependency_group_rolled_back",
            ) for operation in group if operation.operation_id not in handled)

    accepted_gap_ids = {item.gap_id for item in report.accepted}
    report.closed_gap_ids = [
        gap.gap_id for gap in gaps
        if gap.gap_id in accepted_gap_ids and _gap_closed(gap, base, working)
    ]
    report.concrete_gain_count = sum(item.operation_type != "add_ambiguity" for item in report.accepted)
    report.clarification_gain_count = sum(item.operation_type == "add_ambiguity" for item in report.accepted)
    return working, report


def _operation_groups(operations: list[PatchOperation]) -> list[list[PatchOperation]]:
    """Preserve payload order while making named dependency groups atomic."""
    grouped: dict[str, list[PatchOperation]] = {}
    order: list[str] = []
    for operation in operations:
        key = operation.dependency_group or operation.operation_id
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(operation)
    return [grouped[key] for key in order]


def _apply_event(ir: UnderstandingIR, operation: AddEventOperation, span: SourceSpan,
                 catalog: CatalogSnapshot) -> tuple[bool, str]:
    return _apply_dynamic_event(ir, operation, span, catalog)


def _apply_dynamic_event(ir: UnderstandingIR, operation: AddEventOperation,
                         span: SourceSpan, catalog: CatalogSnapshot) -> tuple[bool, str]:
    """Apply a typed dynamic-threshold event without allowing executable code.

    The legacy merger only accepts literal predicate values.  Dynamic thresholds
    use the same evidence, Catalog and expression restrictions as arithmetic
    patches, then rely on the normal transaction validator for unit closure.
    """
    payload = operation.event
    event_span = _validated_operation_span([payload.span], ir.query.normalized)
    metric = catalog.derived_metric(payload.metric_id)
    source_id = catalog.source_for_derived_metric(payload.metric_id)
    source = catalog.source(source_id or "")
    if event_span is None or metric is None or source is None:
        return False, "dynamic_event_metric_or_evidence_unresolved"
    existing = [item.provenance[0].span for item in ir.events
                if item.provenance and item.provenance[0].span]
    if any(event_span.start < item.end and item.start < event_span.end for item in existing):
        return False, "event_exists"
    conditions = []
    for item in payload.conditions:
        condition_span = _validated_operation_span([item.span], ir.query.normalized)
        field = catalog.field(item.field_id)
        if (condition_span is None or field is None
                or catalog.source_for_field(item.field_id) != source_id
                or not _prove_field_in_context(ir, catalog, item.field_id, condition_span.text)):
            return False, "dynamic_event_condition_unresolved"
        right = _expression_to_ir(item.right_expression, ir, catalog, condition_span) \
            if item.right_expression is not None else None
        if item.right_expression is not None and right is None:
            return False, "dynamic_threshold_expression_unresolved"
        if item.right_expression is None and item.value is None:
            return False, "dynamic_event_value_missing"
        conditions.append(PredicateNode(
            operator=item.operator,
            field=_symbol_for_identifier(ir, catalog, item.field_id, condition_span),
            value=None if right is not None else LiteralValue(
                item.value, "range" if item.operator == "between" else _patch_value_type(item.value), item.unit,
            ),
            right_expression=right,
            provenance=[Provenance(source="llm", span=condition_span,
                                   rule="patch_dynamic_event_condition")],
        ))
    duration_span = _validated_operation_span([payload.duration.span], ir.query.normalized)
    if duration_span is None:
        return False, "dynamic_event_duration_evidence_invalid"
    if payload.duration.unit not in _PATCH_DURATION_SECONDS:
        return False, "dynamic_event_duration_unit_invalid"
    logic = payload.logic
    condition = conditions[0] if len(conditions) == 1 else PredicateNode(
        kind="boolean", operator=logic, children=conditions,
    )
    temporal = ir.temporal[0] if ir.temporal else None
    metadata = source.metadata
    expected = metadata.get("expected_sampling_interval_seconds")
    expected_seconds = int(expected) if expected is not None else None
    ir.events.append(EventSpec(
        event_id=operation.event.event_id,
        condition=condition,
        derived_metric=DerivedMetricCall(
            metric_id=payload.metric_id, arguments={"condition": "event_condition", **metric.defaults},
            result_type=metric.result_type, unit=metric.unit,
            provenance=[Provenance(source="llm", span=event_span, rule="patch_dynamic_metric")],
        ),
        threshold=DurationConstraint(
            payload.duration.operator, payload.duration.value, payload.duration.unit,
            payload.duration.value * _PATCH_DURATION_SECONDS[payload.duration.unit], duration_span,
        ),
        window=WindowSpec(
            window_type="absolute",
            from_value=(temporal.from_value or temporal.exact) if temporal else None,
            to_value=(temporal.to_value or temporal.exact) if temporal else None,
            timezone=(temporal.timezone if temporal else "")
            or str(metadata.get("default_timezone", "Asia/Shanghai")),
            granularity=temporal.granularity if temporal else "minute",
        ),
        sampling=SamplingPolicy(
            order_by=_symbol_for_identifier(ir, catalog, str(metadata.get("timestamp_field", "")), span),
            partition_by=[_symbol_for_identifier(ir, catalog, item, span)
                          for item in metadata.get("partition_fields", [])],
            expected_interval_seconds=expected_seconds,
            max_gap_seconds=expected_seconds * int(metric.defaults.get("max_gap_multiplier", 2))
            if expected_seconds else None,
            missing_data_policy=metric.missing_data_policy or "break_segment",
        ),
        group_by=list(payload.group_by), output_name=f"{operation.event.event_id}_local_dates",
        output_ref=OutputRef(
            operation.event.event_id, "intervals", "event_interval",
            shape="event_set", grain="interval", keys=["local_date"],
        ),
        provenance=[Provenance(source="llm", span=event_span, rule="patch_add_dynamic_event")],
    ))
    return True, ""


def _apply_set(ir: UnderstandingIR, operation: AddSetOperation, span: SourceSpan,
               catalog: CatalogSnapshot) -> tuple[bool, str]:
    if ir.set_operations:
        return False, "set_operation_exists"
    input_refs = [_find_output_ref(ir, value) for value in operation.inputs]
    if any(item is None for item in input_refs):
        return False, "set_input_unresolved"
    resolved_refs = [item for item in input_refs if item is not None]
    if len({item.ref_id for item in resolved_refs}) != len(resolved_refs):
        return False, "set_inputs_not_distinct"
    port = "dates" if operation.granularity == "date" else "intervals"
    result_type = "date" if operation.granularity == "date" else "event_interval"
    output = OutputRef(
        f"set_{operation.operation_id}", port, result_type,
        shape="event_set", grain=operation.granularity,
        keys=["local_date"] if operation.granularity == "date" else [],
    )
    ir.set_operations.append(SetOperationSpec(
        operation=operation.operation,
        inputs=[item.ref_id for item in resolved_refs],
        output_name=f"{operation.operation_id}_{port}",
        granularity=operation.granularity,
        input_refs=resolved_refs,
        output_ref=output,
        provenance=[Provenance(source="llm", span=span, rule="patch_add_set_operation")],
    ))
    return True, ""


def _apply_aggregate(ir: UnderstandingIR, operation: AddAggregateOperation,
                     span: SourceSpan, catalog: CatalogSnapshot) -> tuple[bool, str]:
    if any(item.aggregate.aggregate_id == operation.aggregate_id for item in ir.aggregates):
        return False, "aggregate_exists"
    expression = _expression_to_ir(operation.input, ir, catalog, span)
    if expression is None:
        return False, "aggregate_input_unresolved"
    scope_ref = _find_output_ref(ir, operation.scope_ref) if operation.scope_ref else None
    if operation.scope == "relation" and scope_ref is None:
        return False, "aggregate_scope_unresolved"
    output = OutputRef(
        operation.aggregate_id, "value", "number",
        unit=None if operation.function == "count" else expression.unit,
        shape="relation" if operation.scope == "relation" else "scalar",
        grain="group" if operation.scope == "relation" else "scalar",
        keys=list(scope_ref.keys) if scope_ref else [],
        fields=[f"{operation.function}:{expression.symbol.raw_name}"]
        if expression.symbol is not None else [operation.function],
    )
    ir.aggregates.append(ScopedAggregateSpec(
        aggregate=AggregateSpec(
            operation.aggregate_id, operation.function, expression, output,
            [Provenance(source="llm", span=span, rule="patch_add_aggregate")],
        ),
        scope=operation.scope,
        scope_ref=scope_ref,
        group_by=[item for value in operation.group_by
                  if (item := _expression_to_ir(value, ir, catalog, span)) is not None],
    ))
    return True, ""


def _apply_comparison(ir: UnderstandingIR, operation: AddComparisonOperation,
                      span: SourceSpan, catalog: CatalogSnapshot) -> tuple[bool, str]:
    if any(item.comparison_id == operation.comparison_id for item in ir.comparisons):
        return False, "comparison_exists"
    left = _expression_to_ir(operation.left, ir, catalog, span)
    right = _expression_to_ir(operation.right, ir, catalog, span)
    if left is None or right is None:
        return False, "comparison_expression_unresolved"
    ir.comparisons.append(ComparisonSpec(
        operation.comparison_id, left, operation.operator, right,
        OutputRef(operation.comparison_id, "result", "boolean"),
        [Provenance(source="llm", span=span, rule="patch_add_comparison")],
    ))
    return True, ""


def _apply_calculation(ir: UnderstandingIR, operation: AddCalculationOperation,
                       span: SourceSpan, catalog: CatalogSnapshot) -> tuple[bool, str]:
    expression = _expression_to_ir(operation.expression, ir, catalog, span)
    if expression is None:
        return False, "calculation_expression_unresolved"
    rendered = _render_expression(expression)
    identity = (operation.calculation_type, rendered)
    if any((item.calculation_type, item.expression) == identity for item in ir.calculations):
        return False, "calculation_exists"
    if (operation.calculation_type in {"ratio", "difference", "correlation"}
            or (operation.calculation_type == "volatility" and expression.operator == "stddev")):
        inputs = list(expression.arguments)
    else:
        inputs = [expression]
    parameters = {"expression_ast": expression.to_dict()}
    if operation.calculation_type == "volatility" and expression.operator == "stddev" and inputs:
        field = inputs[0].symbol if inputs[0].kind == "field" else None
        if field is not None:
            parameters.update({
                "input_field": field.canonical_id or field.raw_name,
                "derived_metric_id": "semantic.stddev",
            })
    ir.calculations.append(CalculationSpec(
        operation.calculation_type, rendered,
        parameters,
        [Provenance(source="llm", span=span, rule="patch_add_calculation")],
        calculation_id=f"calculation_{operation.operation_id}", inputs=inputs,
        output=OutputRef(
            f"calculation_{operation.operation_id}", "value", expression.result_type,
            unit="ratio" if operation.calculation_type in {"ratio", "correlation"} else expression.unit,
            fields=[operation.calculation_type],
        ),
    ))
    return True, ""


_PATCH_DURATION_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}


def _apply_sequence(ir: UnderstandingIR, operation: AddSequenceOperation,
                    span: SourceSpan, catalog: CatalogSnapshot) -> tuple[bool, str]:
    if any(item.sequence_id == operation.sequence_id for item in ir.sequences):
        return False, "sequence_exists"
    steps = []
    source_ids = set()
    for item in operation.steps:
        item_span = _validated_operation_span([item.span], ir.query.normalized)
        field = catalog.field(item.field_id)
        if (item_span is None or field is None
                or not catalog.prove_field_alias(item.field_id, item_span.text)):
            return False, "sequence_step_unresolved"
        source_id = catalog.source_for_field(item.field_id)
        if source_id:
            source_ids.add(source_id)
        steps.append(PredicateNode(
            operator=item.operator,
            field=_symbol_for_identifier(ir, catalog, item.field_id, item_span),
            value=LiteralValue(item.value, "range" if item.operator == "between" else _patch_value_type(item.value),
                               unit=item.unit),
            provenance=[Provenance(source="llm", span=item_span, rule="patch_sequence_step")],
        ))
    partition = [_symbol_for_identifier(ir, catalog, value, span) for value in operation.partition_by]
    order = _symbol_for_identifier(ir, catalog, operation.order_by, span)
    if order is None or any(item is None for item in partition):
        return False, "sequence_partition_or_order_unresolved"
    for item in [*partition, order]:
        source_id = catalog.source_for_field(item.canonical_id or "")
        if source_id:
            source_ids.add(source_id)
    if len(source_ids) > 1:
        return False, "sequence_multi_source_unsupported"
    duration = None
    if operation.max_duration is not None:
        duration_span = _validated_operation_span([operation.max_duration.span], ir.query.normalized)
        if duration_span is None:
            return False, "sequence_duration_evidence_invalid"
        duration = DurationConstraint(
            operation.max_duration.operator, operation.max_duration.value,
            operation.max_duration.unit,
            operation.max_duration.value * _PATCH_DURATION_SECONDS[operation.max_duration.unit],
            duration_span,
        )
    ir.sequences.append(SequenceSpec(
        sequence_id=operation.sequence_id, steps=steps, partition_by=list(partition), order_by=order,
        output=OutputRef(operation.sequence_id, "events", "event_interval",
                         shape="event_set", grain="interval"),
        max_duration=duration, missing_data_policy=operation.missing_data_policy,
        provenance=[Provenance(source="llm", span=span, rule="patch_add_sequence")],
    ))
    return True, ""


def _patch_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "range"
    return "string"


def _prove_field_in_context(ir: UnderstandingIR, catalog: CatalogSnapshot,
                            field_id: str, evidence_text: str) -> bool:
    """Prove a field alias, allowing a previously resolved single source context.

    A bare alias such as ``温度`` can be globally ambiguous.  It remains safe
    when the local compiler already resolved exactly one source from independent
    query evidence and that source owns the candidate field; otherwise the patch
    is rejected rather than silently selecting a source.
    """
    if catalog.prove_field_alias(field_id, evidence_text):
        return True
    source_id = catalog.source_for_field(field_id)
    resolved = {
        item.identifier for item in ir.source_candidates
        if item.status == "resolved" and catalog.source(item.identifier)
    }
    field = catalog.field(field_id)
    aliases = [field_id.rsplit(".", 1)[-1], *(field.aliases if field else [])]
    return (
        source_id is not None and resolved == {source_id}
        and any(alias and alias.lower() in evidence_text.lower() for alias in aliases)
    )


def _apply_reference(ir: UnderstandingIR, operation: ResolveReferenceOperation,
                     _span: SourceSpan, _catalog: CatalogSnapshot) -> tuple[bool, str]:
    reference = next((item for item in ir.references
                      if item.reference_id == operation.reference_id), None)
    if reference is None or _find_output_ref(ir, operation.target_ref) is None:
        return False, "reference_or_target_unresolved"
    if reference.status == "resolved" and reference.target_task_id == operation.target_ref:
        return False, "reference_already_resolved"
    reference.status = "resolved"
    reference.target_task_id = operation.target_ref
    return True, ""


def _apply_ambiguity(ir: UnderstandingIR, operation: AddAmbiguityOperation,
                     span: SourceSpan, _catalog: CatalogSnapshot) -> tuple[bool, str]:
    if any(item.ambiguity_id == operation.ambiguity_id for item in ir.ambiguities):
        return False, "ambiguity_exists"
    ir.ambiguities.append(AmbiguitySpec(
        operation.ambiguity_id, operation.kind, operation.message,
        span=span, candidates=list(operation.candidates),
    ))
    return True, ""


def _apply_turn_directive(ir: UnderstandingIR, operation: AddTurnDirectiveOperation,
                          span: SourceSpan, _catalog: CatalogSnapshot) -> tuple[bool, str]:
    payload = operation.directive
    if any(item.directive_id == payload.directive_id for item in ir.turn_directives):
        return False, "turn_directive_exists"
    if payload.context_mode == "inherit" and not payload.base_context_ref:
        return False, "base_context_required"
    if payload.context_mode == "fresh" and payload.base_context_ref:
        return False, "fresh_context_forbids_base"
    delta = ContextDelta(
        owner=payload.delta_owner,
        operations=[ContextDeltaOperation(item.operation, item.slot, item.value)
                    for item in payload.delta],
    )
    ir.turn_directives.append(TurnDirective(
        payload.directive_id, list(payload.pre_actions), payload.primary_action,
        payload.target_turn_id, payload.context_mode, payload.base_context_ref,
        delta, list(payload.required_context),
        [Provenance(source="llm", span=span, rule="patch_add_turn_directive")],
    ))
    return True, ""


PATCH_APPLIERS = {
    "add_event": _apply_event,
    "add_set_operation": _apply_set,
    "add_aggregate": _apply_aggregate,
    "add_comparison": _apply_comparison,
    "add_calculation": _apply_calculation,
    "add_sequence": _apply_sequence,
    "resolve_reference": _apply_reference,
    "add_ambiguity": _apply_ambiguity,
    "add_turn_directive": _apply_turn_directive,
}
PATCH_OPERATION_TYPES = frozenset(PATCH_APPLIERS)


def _expression_to_ir(value: ExpressionPatch, ir: UnderstandingIR,
                      catalog: CatalogSnapshot, span: SourceSpan) -> ExpressionNode | None:
    provenance = [Provenance(source="llm", span=span, rule="patch_expression")]
    if value.kind == "literal":
        return ExpressionNode(
            kind="literal", literal=LiteralValue(value.value, value.value_type, value.unit),
            result_type=value.result_type if value.result_type != "unknown" else value.value_type,
            unit=value.unit, provenance=provenance,
        )
    if value.kind in {"catalog_symbol", "hypothesis_symbol"}:
        symbol = _symbol_for_identifier(ir, catalog, value.identifier, span)
        if symbol is None:
            return None
        field = catalog.field(value.identifier)
        hypothesis = next((item for item in ir.schema_hypotheses
                           if item.hypothesis_id == value.identifier), None)
        return ExpressionNode(
            kind="field", symbol=symbol,
            result_type=(field.data_type if field else hypothesis.declared_type if hypothesis else value.result_type),
            unit=(field.unit if field else hypothesis.unit if hypothesis else value.unit),
            provenance=provenance,
        )
    if value.kind == "output_ref":
        output = _find_output_ref(ir, value.identifier)
        if output is None:
            return None
        return ExpressionNode(
            kind="output_ref", reference=output.ref_id,
            result_type=output.result_type, unit=output.unit, provenance=provenance,
        )
    arguments = []
    for argument in value.arguments:
        converted = _expression_to_ir(argument, ir, catalog, span)
        if converted is None:
            return None
        arguments.append(converted)
    if value.kind in {"binary", "function"} and arguments:
        result_type, unit = _compiled_expression_contract(value.kind, value.operator, arguments)
        return ExpressionNode(
            kind=value.kind, operator=value.operator, arguments=arguments,
            result_type=result_type, unit=unit, provenance=provenance,
        )
    return None


def _compiled_expression_contract(kind: str, operator: str,
                                  arguments: list[ExpressionNode]) -> tuple[str, str | None]:
    """Infer trusted expression metadata from already bound typed operands."""
    if kind == "function":
        if operator in {"stddev", "max"} and len(arguments) == 1:
            return arguments[0].result_type, arguments[0].unit
        if operator == "correlation" and len(arguments) == 2:
            return "number", "ratio"
        if operator in {"moving_avg"} and arguments:
            return arguments[0].result_type, arguments[0].unit
        if operator == "window_ratio":
            return "number", "ratio"
        return "unknown", None
    if kind == "binary" and len(arguments) == 2:
        left, right = arguments
        if operator in {"add", "sub"}:
            return left.result_type, left.unit
        if operator == "mul":
            if left.result_type == "number" and left.unit is None:
                return right.result_type, right.unit
            if right.result_type == "number" and right.unit is None:
                return left.result_type, left.unit
            return "unknown", None
        if operator == "div":
            if (left.result_type, left.unit) == (right.result_type, right.unit):
                return "number", "ratio"
            if right.result_type == "number" and right.unit is None:
                return left.result_type, left.unit
    return "unknown", None


def _symbol_for_identifier(ir: UnderstandingIR, catalog: CatalogSnapshot,
                           identifier: str, span: SourceSpan) -> SymbolRef | None:
    field = catalog.field(identifier)
    if field:
        # Retain the query-facing name when it already has an authoritative
        # local binding.  It lets a Patch using a catalog ID still satisfy the
        # independently extracted Requirement field contract (e.g. 温度 vs.
        # weather_observations.temperature).
        existing = next((item for item in ir.projections
                         if item.canonical_id == field.field_id), None)
        if existing is not None:
            return SymbolRef(
                existing.raw_name, field.field_id,
                [CandidateRef(field.field_id, source="catalog")], "resolved", span, "catalog",
            )
        return SymbolRef(
            field.field_id.rsplit(".", 1)[-1], field.field_id,
            [CandidateRef(field.field_id, source="catalog")], "resolved", span, "catalog",
        )
    hypothesis = next((item for item in ir.schema_hypotheses
                       if item.hypothesis_id == identifier), None)
    if hypothesis:
        existing_symbols = list(ir.projections)
        existing_symbols.extend(item.field for item in ir.metrics)
        for root in [ir.filters, *(item.condition for item in ir.events)]:
            existing_symbols.extend(
                item.field for item in walk_predicates(root) if item.field is not None
            )
        existing = next((
            item for item in existing_symbols
            if any(candidate.identifier == identifier for candidate in item.candidates)
        ), None)
        if existing is not None:
            return SymbolRef(
                existing.raw_name, existing.canonical_id, list(existing.candidates),
                existing.status, existing.span or hypothesis.span or span, existing.ref_kind,
            )
        return SymbolRef(
            hypothesis.raw_name, None,
            [CandidateRef(hypothesis.hypothesis_id, source="hypothesis", status="candidate")],
            "candidate", hypothesis.span or span, "hypothesis",
        )
    return None


def _find_output_ref(ir: UnderstandingIR, ref_id: str | None) -> OutputRef | None:
    if not ref_id:
        return None
    outputs: list[OutputRef] = []
    outputs.extend(item.output_ref for item in ir.events if item.output_ref)
    outputs.extend(item.output_ref for item in ir.set_operations if item.output_ref)
    outputs.extend(item.output for item in ir.derived_projections)
    outputs.extend(item.aggregate.output for item in ir.aggregates)
    outputs.extend(item.output for item in ir.comparisons)
    outputs.extend(item.output for item in ir.arithmetic)
    outputs.extend(item.output for item in ir.relation_filters)
    outputs.extend(item.output for item in ir.conversions)
    outputs.extend(item.output for item in ir.ordered_folds)
    outputs.extend(item.output for item in ir.sequences)
    outputs.extend(item.output for item in ir.group_by_operations)
    outputs.extend(item.output for item in ir.top_k_operations)
    outputs.extend(item.output for item in ir.window_aggregates)
    outputs.extend(item.output for item in ir.cumulative_counts)
    outputs.extend(item.output for item in ir.cumulative_durations)
    outputs.extend(item.output for item in ir.calculations if item.output)
    return next((item for item in outputs if item.ref_id == ref_id), None)


def _render_expression(node: ExpressionNode) -> str:
    if node.kind == "literal" and node.literal:
        return str(node.literal.value)
    if node.kind == "field" and node.symbol:
        return node.symbol.canonical_id or node.symbol.raw_name
    if node.kind == "output_ref":
        return node.reference
    rendered = [_render_expression(item) for item in node.arguments]
    if node.kind == "binary" and len(rendered) == 2:
        return f"({rendered[0]} {node.operator} {rendered[1]})"
    return f"{node.operator}({', '.join(rendered)})"


def _validated_operation_span(evidence: list[EvidenceSpanModel], query: str) -> SourceSpan | None:
    for item in evidence:
        if item.start is not None and item.end is not None:
            if item.end <= len(query) and query[item.start:item.end] == item.text:
                return SourceSpan(item.start, item.end, item.text)
        if query.count(item.text) == 1:
            start = query.find(item.text)
            return SourceSpan(start, start + len(item.text), item.text)
    return None


def _preconditions_hold(preconditions: list[PatchPrecondition],
                        gaps: Mapping[str, GapRequest], known_nodes: set[str]) -> bool:
    for item in preconditions:
        if item.kind == "gap_open" and item.value not in gaps:
            return False
        if item.kind == "node_absent" and item.value in known_nodes:
            return False
        if item.kind == "node_present" and item.value not in known_nodes:
            return False
    return True


def _known_node_ids(ir: UnderstandingIR) -> set[str]:
    values = {item.event_id for item in ir.events}
    values.update(item.output_name for item in ir.events)
    values.update(item.output_name for item in ir.set_operations)
    values.update(item.aggregate.aggregate_id for item in ir.aggregates)
    values.update(item.comparison_id for item in ir.comparisons)
    values.update(item.arithmetic_id for item in ir.arithmetic)
    values.update(item.output.ref_id for item in ir.arithmetic)
    values.update(item.conversion_id for item in ir.conversions)
    values.update(item.output.ref_id for item in ir.conversions)
    values.update(item.fold_id for item in ir.ordered_folds)
    values.update(item.output.ref_id for item in ir.ordered_folds)
    values.update(item.sequence_id for item in ir.sequences)
    values.update(item.output.ref_id for item in ir.sequences)
    values.update(item.reference_id for item in ir.references)
    values.update(item.directive_id for item in ir.turn_directives)
    return values


def _semantic_shape(ir: UnderstandingIR) -> tuple[int, ...]:
    return (
        len(ir.events), len(ir.set_operations), len(ir.aggregates),
        len(ir.comparisons), len(ir.calculations), len(ir.sequences),
        sum(item.status == "resolved" for item in ir.references),
        len(ir.ambiguities), len(ir.turn_directives),
    )


def _gap_closed(gap: GapRequest, before: UnderstandingIR,
                after: UnderstandingIR) -> bool:
    if gap.gap_type == "missing_event":
        return len(after.events) > len(before.events)
    if gap.gap_type == "missing_set_inputs":
        return len(after.set_operations) > len(before.set_operations)
    if gap.gap_type == "missing_formula_input":
        return len(after.calculations) > len(before.calculations)
    if gap.gap_type == "missing_sequence":
        return len(after.sequences) > len(before.sequences)
    if gap.gap_type == "missing_aggregate_scope":
        return len(after.aggregates) > len(before.aggregates)
    if gap.gap_type == "unresolved_reference":
        return sum(item.status == "resolved" for item in after.references) > sum(
            item.status == "resolved" for item in before.references
        )
    if gap.gap_type in {
        "unresolved_symbol", "ambiguous_formula", "missing_predicate", "missing_output",
    }:
        return len(after.ambiguities) > len(before.ambiguities)
    if gap.gap_type == "missing_conversion":
        return (len(after.calculations) > len(before.calculations)
                or len(after.ambiguities) > len(before.ambiguities))
    if gap.gap_type in {"context_required", "missing_turn_directive"}:
        return (len(after.turn_directives) > len(before.turn_directives)
                or len(after.ambiguities) > len(before.ambiguities))
    return False


def _looks_like_legacy_payload(payload: Mapping[str, Any]) -> bool:
    return bool(set(payload) & {
        "goals", "sources", "projections", "metrics", "filters", "events",
        "set_operations", "calculations", "aggregates", "comparisons", "unresolved",
    })


def _legacy_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]
