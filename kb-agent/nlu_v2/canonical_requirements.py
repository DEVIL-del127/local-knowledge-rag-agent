"""Ephemeral canonical Requirement view for M1.5.

The graph is derived from authoritative Requirement and SourceDemand records.
It deliberately cannot mutate or replace either collection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .models import RequirementSpec, SourceSpan, UnderstandingIR


COMPILER_VERSION = "m15-requirements-v1"


@dataclass(frozen=True, slots=True)
class CanonicalRequirementNode:
    canonical_id: str
    requirement_ids: tuple[str, ...]
    source_demand_ids: tuple[str, ...]
    operator_family: str
    dependency_demand_ids: tuple[str, ...]
    output_role: str
    output_grain: str
    clause_id: str
    spans: tuple[SourceSpan, ...]
    conflict: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalRequirementGraph:
    compiler_version: str
    nodes: tuple[CanonicalRequirementNode, ...]
    edges: tuple[tuple[str, str], ...]
    conflicts: tuple[str, ...]

    @classmethod
    def build(cls, ir: UnderstandingIR) -> "CanonicalRequirementGraph":
        demands = {item.demand_id: item for item in ir.source_demands}
        groups: dict[tuple, list[RequirementSpec]] = {}
        for requirement in ir.requirements:
            source_ids = tuple(sorted(_source_demand_ids(requirement, demands)))
            dependency_demands = tuple(sorted({
                demand_id
                for dependency in requirement.dependencies
                for demand_id in _requirement_demand_ids(dependency, ir.requirements, demands)
            }))
            # Clause identity is required when evidence is clause-local. This
            # prevents identical words in separate user obligations merging.
            clause_id = requirement.clause_id if source_ids else ""
            key = (
                source_ids,
                requirement.operator_family or requirement.requirement_type,
                dependency_demands,
                requirement.role,
                requirement.expected_output_grain,
                clause_id,
            )
            groups.setdefault(key, []).append(requirement)

        nodes = []
        requirement_to_node: dict[str, str] = {}
        conflicts = []
        for key, members in groups.items():
            output_shapes = {item.expected_output_shape for item in members if item.expected_output_shape}
            operator_types = {item.requirement_type for item in members}
            conflict = ""
            if len(output_shapes) > 1 or len(operator_types) > 1:
                conflict = "canonical_group_contract_disagreement"
            canonical_id = "creq_" + hashlib.sha256(
                json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            spans = tuple(item.span for item in members if item.span is not None)
            node = CanonicalRequirementNode(
                canonical_id=canonical_id,
                requirement_ids=tuple(sorted(item.requirement_id for item in members)),
                source_demand_ids=key[0], operator_family=key[1],
                dependency_demand_ids=key[2], output_role=key[3],
                output_grain=key[4], clause_id=key[5], spans=spans,
                conflict=conflict,
            )
            nodes.append(node)
            for item in members:
                requirement_to_node[item.requirement_id] = canonical_id
            if conflict:
                conflicts.append(canonical_id)
        edges = sorted({
            (requirement_to_node[dependency], requirement_to_node[item.requirement_id])
            for item in ir.requirements for dependency in item.dependencies
            if dependency in requirement_to_node
            and requirement_to_node[dependency] != requirement_to_node[item.requirement_id]
        })
        return cls(COMPILER_VERSION, tuple(sorted(nodes, key=lambda item: item.canonical_id)),
                   tuple(edges), tuple(sorted(conflicts)))


def _source_demand_ids(requirement, demands) -> set[str]:
    found = set(requirement.expected_attributes.get("source_demand_ids", []))
    for demand in demands.values():
        if requirement.requirement_id in demand.mapped_requirement_ids:
            found.add(demand.demand_id)
    return found


def _requirement_demand_ids(requirement_id, requirements, demands) -> set[str]:
    requirement = next((item for item in requirements if item.requirement_id == requirement_id), None)
    return _source_demand_ids(requirement, demands) if requirement else set()
