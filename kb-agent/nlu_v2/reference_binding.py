"""Deterministic, output-role-aware reference binding."""
from __future__ import annotations

from dataclasses import dataclass

from .models import UnderstandingIR
from .semantic_targets import OutputLineageIndex


@dataclass(frozen=True, slots=True)
class ReferenceBinding:
    requirement_id: str
    raw: str
    expected_role: str
    candidate_refs: tuple[str, ...]
    selected_ref: str
    status: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ReferenceBindingReport:
    bindings: tuple[ReferenceBinding, ...]
    resolved_count: int
    ambiguous_count: int
    unresolved_count: int


class ReferenceBinder:
    def bind(self, ir: UnderstandingIR) -> ReferenceBindingReport:
        lineage = OutputLineageIndex.build(ir)
        coverage = {item.requirement_id: item for item in ir.coverage}
        bindings = []
        for requirement in ir.requirements:
            if requirement.requirement_type != "reference":
                continue
            role = _expected_role(requirement.text)
            dependency_refs = set()
            for dependency in requirement.dependencies:
                dependency_refs.update(coverage.get(dependency).covered_by if coverage.get(dependency) else [])
            candidates = []
            for ref_id, entry in lineage.entries.items():
                if role and entry.semantic_role != role:
                    continue
                if dependency_refs and entry.producer_id not in dependency_refs and ref_id not in dependency_refs:
                    continue
                candidates.append(ref_id)
            # If clause dependencies were not recovered, role uniqueness is
            # sufficient; proximity or producer recency is never sufficient.
            if len(candidates) == 1:
                status, selected, reason = "resolved", candidates[0], "unique_role_and_dependency"
            elif len(candidates) > 1:
                status, selected, reason = "ambiguous", "", "multiple_role_compatible_outputs"
            else:
                status, selected, reason = "unresolved", "", "no_role_compatible_output"
            bindings.append(ReferenceBinding(
                requirement.requirement_id, requirement.text, role,
                tuple(sorted(candidates)), selected, status, reason,
            ))
        return ReferenceBindingReport(
            tuple(bindings), sum(item.status == "resolved" for item in bindings),
            sum(item.status == "ambiguous" for item in bindings),
            sum(item.status == "unresolved" for item in bindings),
        )


def _expected_role(text):
    if "日期" in text:
        return "date"
    if any(value in text for value in ("时间段", "区间", "事件")):
        return "event_interval"
    if "用户" in text or "记录" in text or "文档" in text:
        return "set"
    if "结果" in text:
        return ""
    return ""
