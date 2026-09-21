"""Clarification-only recovery state; it never manufactures executable IR."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping

from .models import SemanticEffectReport, UnderstandingIR, ValidationReport


@dataclass(slots=True)
class ClarificationState:
    status: str = "none"  # none / pending / answered
    reason_kind: str = ""  # semantic / binding / capability / provider
    blocking_codes: list[str] = field(default_factory=list)
    requirement_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClarificationQuestion:
    question_id: str
    priority: int
    kind: str
    prompt: str
    candidates: list[str] = field(default_factory=list)
    requirement_ids: list[str] = field(default_factory=list)
    # These are explicit recovery bindings, not display-only metadata.
    target_demand_ids: list[str] = field(default_factory=list)
    target_requirement_ids: list[str] = field(default_factory=list)
    expected_answer_type: str = "free_text"  # candidate_id / free_text / acknowledgement
    candidate_ids: list[str] = field(default_factory=list)
    resume_write_path: str = ""


@dataclass(slots=True)
class ClarificationResumeContract:
    clarification_ids: list[str]
    source_demand_digest: str
    catalog_version: str
    allowed_answer_slots: list[str] = field(default_factory=list)
    allowed_write_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClarificationPlan:
    state: ClarificationState
    questions: list[ClarificationQuestion] = field(default_factory=list)
    resume_contract: ClarificationResumeContract | None = None


class ClarificationPlanner:
    """Translate diagnostics into a maximum of three user-facing questions."""

    def plan(self, ir: UnderstandingIR, validation: ValidationReport,
             effect: SemanticEffectReport | None = None) -> ClarificationPlan | None:
        if validation.executable:
            return None
        errors = list(validation.errors)
        if not errors and not ir.ambiguities:
            return None
        questions = self._questions(ir, errors)
        if not questions:
            return None
        kind = questions[0].kind
        state = ClarificationState(
            status="pending", reason_kind=kind,
            blocking_codes=sorted({item.code for item in errors}),
            requirement_ids=sorted({requirement_id for item in questions for requirement_id in item.requirement_ids}),
        )
        digest = hashlib.sha256(json.dumps([item.to_dict() for item in ir.source_demands],
                                           ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return ClarificationPlan(
            state=state, questions=questions[:3],
            resume_contract=ClarificationResumeContract(
                clarification_ids=[item.question_id for item in questions[:3]],
                source_demand_digest=digest, catalog_version=ir.catalog_version,
                allowed_answer_slots=[item.kind for item in questions[:3]],
                allowed_write_paths=[item.resume_write_path for item in questions[:3]],
            ),
        )

    def _questions(self, ir: UnderstandingIR, errors) -> list[ClarificationQuestion]:
        result: list[ClarificationQuestion] = []
        default_requirements = _uncovered_requirements(ir)
        for ambiguity in ir.ambiguities:
            if ambiguity.kind in {"formula_ambiguity", "temporal_century", "timezone_fold", "claim_conflict"}:
                demand_ids = _demand_ids_for_span(ir, ambiguity.span)
                requirement_ids = _requirements_for_demands(ir, demand_ids, default_requirements)
                result.append(self._question(
                    "semantic", ambiguity.message, ambiguity.candidates,
                    demand_ids, requirement_ids,
                ))
        for item in errors:
            kind = _error_class(item.code)
            # Transport/provider conditions cannot be answered by an end user.
            if kind == "provider" or _is_spurious_question(item.message):
                continue
            if kind != "binding":
                continue
            candidates = next((amb.candidates for amb in ir.ambiguities if amb.span == item.span), [])
            demand_ids = _demand_ids_for_span(ir, item.span)
            requirement_ids = _requirements_for_demands(ir, demand_ids, default_requirements)
            result.append(self._question("binding", item.message, candidates, demand_ids, requirement_ids))
        for item in errors:
            if _error_class(item.code) != "capability" or _is_spurious_question(item.message):
                continue
            demand_ids = _demand_ids_for_span(ir, item.span)
            requirement_ids = _requirements_for_demands(ir, demand_ids, default_requirements)
            result.append(self._question("capability", item.message, [], demand_ids, requirement_ids))
        return _dedupe(result, ir)

    @staticmethod
    def _question(kind: str, detail: str, candidates: list[str], demand_ids: list[str],
                  requirement_ids: list[str]) -> ClarificationQuestion:
        prefix = {
            "semantic": "为避免误解，请确认：",
            "binding": "已理解查询，但需要绑定数据：",
            "capability": "当前数据源能力不足：",
        }[kind]
        identity = f"{kind}|{detail}|{'|'.join(candidates)}"
        question_id = "clarify_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        candidate_ids = list(candidates)
        return ClarificationQuestion(
            question_id=question_id,
            priority={"semantic": 1, "binding": 2, "capability": 3}[kind],
            kind=kind, prompt=prefix + detail, candidates=list(candidates),
            requirement_ids=list(requirement_ids),
            target_demand_ids=list(demand_ids),
            target_requirement_ids=list(requirement_ids),
            expected_answer_type="candidate_id" if candidate_ids else (
                "acknowledgement" if kind == "capability" else "free_text"
            ),
            candidate_ids=candidate_ids,
            # The only writable resume namespace is deliberately dedicated to
            # user clarification.  Compiler IR nodes are reconstructed rather
            # than mutated in place.
            resume_write_path=f"clarification_answers.{question_id}",
        )

    def validate_answers(self, plan: ClarificationPlan,
                         answers: Mapping[str, str]) -> dict[str, str]:
        """Validate bounded user answers before they are made compiler input."""
        contract = plan.resume_contract
        if contract is None:
            raise ValueError("clarification plan has no resume contract")
        questions = {item.question_id: item for item in plan.questions}
        unknown = set(answers) - set(contract.clarification_ids)
        if unknown:
            raise ValueError(f"unknown clarification IDs: {sorted(unknown)}")
        normalized: dict[str, str] = {}
        for question_id, value in answers.items():
            question = questions.get(question_id)
            answer = str(value).strip()
            if question is None or not answer:
                raise ValueError(f"empty or unavailable answer for {question_id}")
            if question.resume_write_path not in contract.allowed_write_paths:
                raise ValueError(f"answer path outside resume contract: {question.resume_write_path}")
            if question.expected_answer_type == "candidate_id" and answer not in question.candidate_ids:
                raise ValueError(f"answer for {question_id} must select an advertised candidate ID")
            normalized[question_id] = answer
        if not normalized:
            raise ValueError("at least one clarification answer is required")
        return normalized

    def resume_query(self, raw_query: str, plan: ClarificationPlan,
                     answers: Mapping[str, str]) -> str:
        """Materialize answers as auditable input to a fresh compilation pass."""
        normalized = self.validate_answers(plan, answers)
        by_id = {item.question_id: item for item in plan.questions}
        lines = ["[用户澄清：以下内容仅回答本次指定缺口]"]
        for question_id in sorted(normalized):
            question = by_id[question_id]
            lines.append(
                f"澄清答案（{question.kind}）：{normalized[question_id]}"
            )
        return raw_query.rstrip() + "\n\n" + "\n".join(lines)

    def apply_answers(self, ir: UnderstandingIR, plan: ClarificationPlan,
                      answers: Mapping[str, str]) -> dict[str, str]:
        """Write only to the approved answer namespace of the new snapshot."""
        normalized = self.validate_answers(plan, answers)
        for question in plan.questions:
            if question.question_id in normalized:
                ir.clarification_answers[question.question_id] = normalized[question.question_id]
        return normalized


def _uncovered_requirements(ir: UnderstandingIR) -> list[str]:
    coverage = {item.requirement_id: item.status for item in ir.coverage}
    return [item.requirement_id for item in ir.requirements
            if coverage.get(item.requirement_id) != "satisfied"]


def _demand_ids_for_span(ir: UnderstandingIR, span) -> list[str]:
    if span is None:
        return []
    return [item.demand_id for item in ir.source_demands if not (
        item.span.end <= span.start or span.end <= item.span.start
    )]


def _requirements_for_demands(ir: UnderstandingIR, demand_ids: list[str],
                              fallback: list[str]) -> list[str]:
    matched = {
        requirement_id for demand in ir.source_demands if demand.demand_id in demand_ids
        for requirement_id in demand.mapped_requirement_ids
    }
    return sorted(matched) or list(fallback)


def _error_class(code: str) -> str:
    if code.startswith("provider_") or code.startswith("llm_"):
        return "provider"
    if code in {
        "unresolved_symbol", "unknown_catalog_id", "unknown_field", "source_unresolved", "source_missing",
        "multi_source_unsupported", "multi_source_deferred", "formula_input_unbound", "reference_unresolved",
    }:
        return "binding"
    if "unsupported" in code or "capability" in code:
        return "capability"
    return "semantic"


def _is_spurious_question(detail: str) -> bool:
    value = detail.replace(" ", "")
    return any(token in value for token in (
        "原始问题与证据文本不匹配", "字段1是什么", "的时间占比字段在哪里",
    ))


def _requirement_dependency_rank(ir: UnderstandingIR, requirement_ids: list[str]) -> int:
    by_id = {item.requirement_id: item for item in ir.requirements}

    def depth(requirement_id: str, seen: set[str]) -> int:
        if requirement_id in seen:
            return 0
        requirement = by_id.get(requirement_id)
        if requirement is None or not requirement.dependencies:
            return 0
        return 1 + max((depth(item, seen | {requirement_id})
                        for item in requirement.dependencies), default=0)

    return min((depth(item, set()) for item in requirement_ids), default=0)


def _dedupe(items: list[ClarificationQuestion], ir: UnderstandingIR) -> list[ClarificationQuestion]:
    result = []
    seen = set()
    for item in sorted(items, key=lambda value: (
        _requirement_dependency_rank(ir, value.target_requirement_ids),
        value.priority, value.question_id,
    )):
        key = (item.kind, item.prompt, tuple(item.candidates))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
