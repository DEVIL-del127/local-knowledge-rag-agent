"""Exact role-to-source grounding proofs derived from the existing IR."""
from __future__ import annotations

from dataclasses import dataclass

from .models import RequirementSpec, SourceDemand, SourceSpan, UnderstandingIR


@dataclass(frozen=True, slots=True)
class RoleGroundingPath:
    role: str
    status: str
    requirement_ids: tuple[str, ...] = ()
    demand_ids: tuple[str, ...] = ()
    spans: tuple[SourceSpan, ...] = ()
    values: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RoleGroundingProof:
    requirement_id: str
    roles: tuple[RoleGroundingPath, ...]
    enforceable: bool

    @property
    def complete(self) -> bool:
        return self.enforceable and all(
            item.status in {"proven", "not_required"} for item in self.roles
        )

    def role(self, name: str) -> RoleGroundingPath | None:
        return next((item for item in self.roles if item.role == name), None)


class RoleGroundingAnalyzer:
    """Build proof paths without using text proximity or model output."""

    def prove(self, requirement: RequirementSpec, ir: UnderstandingIR) -> RoleGroundingProof:
        demand_by_id = {item.demand_id: item for item in ir.source_demands}
        requirement_by_id = {item.requirement_id: item for item in ir.requirements}
        source_ids = set(requirement.expected_attributes.get("source_demand_ids", []))
        owned_ids = set(source_ids)
        pending = list(source_ids)
        while pending:
            identifier = pending.pop()
            demand = demand_by_id.get(identifier)
            if demand is None:
                continue
            for dependency in demand.dependencies:
                if dependency not in owned_ids:
                    owned_ids.add(dependency)
                    pending.append(dependency)
        mapped = [demand_by_id[item] for item in owned_ids if item in demand_by_id]
        operation_demands = [item for item in mapped if item.demand_type == "operator"]
        roles = [self._demand_role("operation", requirement, operation_demands)]

        kind = requirement.requirement_type
        if kind in {"aggregate", "scoped_aggregate"}:
            fields = self._measure_demands(requirement, demand_by_id)
            roles.append(self._unique_demand_role(
                "measure", requirement, fields,
                "aggregate measure is not grounded to exactly one field demand",
            ))
            scope_requirements = [
                requirement_by_id[item] for item in requirement.dependencies
                if item in requirement_by_id
                and requirement_by_id[item].requirement_type in {
                    "set_operation", "group_by", "predicate", "sequence_event", "event",
                }
            ]
            scope_required = (
                requirement.expected_scope == "relation" or bool(scope_requirements)
            )
            roles.append(self._requirement_role(
                "scope", scope_requirements, required=scope_required,
                missing_reason="scoped aggregate has no explicit upstream Requirement edge",
            ))
            grain_required = requirement.expected_output_grain not in {"", "scalar"}
            roles.append(RoleGroundingPath(
                role="grain",
                status="proven" if grain_required and operation_demands else
                "unproven" if grain_required else "not_required",
                requirement_ids=(requirement.requirement_id,),
                demand_ids=tuple(item.demand_id for item in operation_demands),
                spans=tuple(item.span for item in operation_demands),
                values=(requirement.expected_output_grain,) if grain_required else (),
                reason="" if not grain_required or operation_demands
                else "explicit grain has no exact operator demand",
            ))
        elif kind == "set_operation":
            producers = [
                requirement_by_id[item] for item in requirement.dependencies
                if item in requirement_by_id
                and requirement_by_id[item].requirement_type in {
                    "sequence_event", "event", "derived_projection",
                }
            ]
            roles.append(RoleGroundingPath(
                role="inputs", status="proven" if len(producers) >= 2 else "unproven",
                requirement_ids=tuple(item.requirement_id for item in producers),
                spans=tuple(item.span for item in producers if item.span),
                values=tuple(item.requirement_type for item in producers),
                reason="" if len(producers) >= 2
                else "set operation lacks two explicit producer Requirement edges",
            ))
        elif kind in {"sequence_event", "event"}:
            predicates = [item for item in mapped if item.demand_type == "comparison"]
            durations = [item for item in mapped if item.demand_type == "duration_threshold"]
            roles.extend([
                RoleGroundingPath(
                    role="predicate", status="proven" if predicates else "unproven",
                    requirement_ids=(requirement.requirement_id,),
                    demand_ids=tuple(item.demand_id for item in predicates),
                    spans=tuple(item.span for item in predicates),
                    values=tuple(str(item.attributes.get("field_text", "")) for item in predicates),
                    reason="" if predicates else "event has no exact comparison demand",
                ),
                RoleGroundingPath(
                    role="duration", status="proven" if len(durations) == 1 else "unproven",
                    requirement_ids=(requirement.requirement_id,),
                    demand_ids=tuple(item.demand_id for item in durations),
                    spans=tuple(item.span for item in durations),
                    values=tuple(str(item.attributes.get("value", "")) for item in durations),
                    reason="" if len(durations) == 1 else "event duration is missing or non-unique",
                ),
            ])
        elif kind == "calculation":
            inputs = [
                requirement_by_id[item] for item in requirement.dependencies
                if item in requirement_by_id
            ]
            fields = self._measure_demands(requirement, demand_by_id)
            roles.append(RoleGroundingPath(
                role="inputs", status="proven" if inputs or fields else "unproven",
                requirement_ids=tuple(item.requirement_id for item in inputs),
                demand_ids=tuple(item.demand_id for item in fields),
                spans=tuple([item.span for item in inputs if item.span]
                            + [item.span for item in fields]),
                values=tuple(item.attributes.get("field_text", item.text) for item in fields),
                reason="" if inputs or fields else "semantic inputs have no exact grounding edge",
            ))

        enforceable = bool(source_ids and all(item in demand_by_id for item in source_ids))
        return RoleGroundingProof(requirement.requirement_id, tuple(roles), enforceable)

    @staticmethod
    def _measure_demands(requirement: RequirementSpec,
                         demand_by_id: dict[str, SourceDemand]) -> list[SourceDemand]:
        identifiers = set(requirement.expected_attributes.get("input_demand_ids", []))
        for value in requirement.expected_attributes.get("source_demand_ids", []):
            demand = demand_by_id.get(value)
            if demand is not None:
                identifiers.update(demand.dependencies)
        return [
            demand_by_id[item] for item in identifiers
            if item in demand_by_id and demand_by_id[item].demand_type == "field"
        ]

    @staticmethod
    def _demand_role(role: str, requirement: RequirementSpec,
                     demands: list[SourceDemand]) -> RoleGroundingPath:
        return RoleGroundingPath(
            role=role, status="proven" if demands else "unproven",
            requirement_ids=(requirement.requirement_id,),
            demand_ids=tuple(item.demand_id for item in demands),
            spans=tuple(item.span for item in demands),
            values=tuple(str(item.attributes.get("operator_family", item.text)) for item in demands),
            reason="" if demands else "no exact operator SourceDemand",
        )

    @classmethod
    def _unique_demand_role(cls, role: str, requirement: RequirementSpec,
                            demands: list[SourceDemand], reason: str) -> RoleGroundingPath:
        base = cls._demand_role(role, requirement, demands)
        if len(demands) == 1:
            return base
        return RoleGroundingPath(
            role=role, status="unproven", requirement_ids=base.requirement_ids,
            demand_ids=base.demand_ids, spans=base.spans,
            values=tuple(str(item.attributes.get("field_text", item.text)) for item in demands),
            reason=reason,
        )

    @staticmethod
    def _requirement_role(role: str, requirements: list[RequirementSpec], *,
                          required: bool, missing_reason: str) -> RoleGroundingPath:
        return RoleGroundingPath(
            role=role,
            status="not_required" if not required else "proven" if requirements else "unproven",
            requirement_ids=tuple(item.requirement_id for item in requirements),
            spans=tuple(item.span for item in requirements if item.span),
            values=tuple(item.requirement_type for item in requirements),
            reason="" if not required or requirements else missing_reason,
        )
