from __future__ import annotations

import re
import time
from typing import Any

from .routing_contracts import Applicability, ContextMode, ContextProposal, RoutingHint
from .runtime.domain_router import DomainDecision, DomainRouter, RequestDomain

_EXPLICIT_TOPIC_CHANGE = re.compile(r"换个话题|另外(?:一个|件)事|不说这个|new topic", re.I)
_OBJECT_REFERENCE = re.compile(
    r"它们?|那个|这篇|这(?:两|2|二)篇|二者|上述|前(?:两|2|二)篇|第[一二三四五六七八九十\d]+篇",
    re.I,
)
_PATCH_SIGNAL = re.compile(r"只看|仅看|限定为|换成|改成|年份|(?:19|20)\d{2}", re.I)
_PATCH_ONLY = re.compile(
    r"^(?:只看|仅看|限定为).+|^继续$|^(?:那|只)?\s*(?:19|20)\d{2}\s*年的?(?:呢)?[？?]?$",
    re.I,
)
_CONTINUE_SIGNAL = re.compile(r"^继续$", re.I)
_CONTEXT_ACTION = re.compile(r"^(?:总结|概括|比较|对比)(?:一下|下|呢)?[？?]?$", re.I)
_INDEPENDENT_LITERATURE_REQUEST = re.compile(
    r"(?:查|找|检索|搜索|列出|比较|对比|总结|概括).*(?:论文|文献)|"
    r"(?:论文|文献).*(?:哪些|有什么|查|找|检索|比较|总结)", re.I,
)


class RoutingService:
    """Low-cost routing orchestration; never compiles literature semantics."""

    def __init__(self, domain_router: DomainRouter | None = None, semantic_router: Any | None = None) -> None:
        self.domain_router = domain_router or DomainRouter()
        self.semantic_router = semantic_router

    def route(self, query: str, *, previous: Any | None = None,
              literature_reference: str | None = None) -> tuple[DomainDecision, RoutingHint]:
        started = time.perf_counter()
        active = self._active_context(previous)
        decision = self.domain_router.route(
            query, prior_domain=RequestDomain.KB_DOCUMENT.value if active else None,
            literature_reference=literature_reference,
        )
        proposals = self._proposals(query, active)
        object_reference = bool(_OBJECT_REFERENCE.search(query or ""))
        object_ids = tuple(str(item) for item in active.get("current_object_ids", []) if item)
        explicit_new = self._is_independent_new_request(query)
        if active and object_reference and object_ids and not explicit_new:
            decision = DomainDecision(RequestDomain.KB_DOCUMENT, 0.99, "kb_bound_object_reference")
        elif active and object_reference and not object_ids:
            decision = DomainDecision(RequestDomain.CLARIFY_DOMAIN, 1.0, "referenced_object_missing")
        elif object_reference and not active:
            decision = DomainDecision(RequestDomain.CLARIFY_DOMAIN, 1.0, "referenced_object_missing")
        elif active and _CONTINUE_SIGNAL.search(query or ""):
            decision = DomainDecision(RequestDomain.KB_DOCUMENT, 0.99, "kb_context_followup")
        elif active and any(item.valid and item.mode in {ContextMode.INHERIT, ContextMode.PATCH}
                            for item in proposals):
            decision = DomainDecision(RequestDomain.KB_DOCUMENT, 0.99, "kb_context_effect")
        valid_effects = {self._effect_key(item) for item in proposals if item.valid}
        if len(valid_effects) > 1:
            decision = DomainDecision(RequestDomain.CLARIFY_DOMAIN, 1.0, "context_candidate_conflict")
        semantic = None
        semantic_status = "not_used"
        if decision.domain == RequestDomain.CLARIFY_DOMAIN and self.semantic_router is not None and not active:
            try:
                semantic = self.semantic_router.route(query)
                semantic_status = semantic.reason_code
            except Exception as exc:
                semantic_status = str(exc)[:80]
            if semantic is not None and semantic.auto_release:
                mapped = {"literature_find": RequestDomain.KB_DOCUMENT,
                          "literature_compare": RequestDomain.KB_DOCUMENT,
                          "general_chat": RequestDomain.GENERAL_CHAT}.get(semantic.category)
                if mapped is not None:
                    decision = DomainDecision(mapped, semantic.score, "semantic_matrix_release")
        candidates = [decision.domain.value]
        if active and _OBJECT_REFERENCE.search(query or "") and RequestDomain.KB_DOCUMENT.value not in candidates:
            candidates.insert(0, RequestDomain.KB_DOCUMENT.value)
        if len(set(candidates)) > 1:
            applicability, reason = Applicability.UNKNOWN, "context_domain_conflict"
        elif decision.domain == RequestDomain.KB_DOCUMENT:
            applicability, reason = Applicability.BUSINESS, decision.reason_code
        elif decision.domain == RequestDomain.GENERAL_CHAT:
            applicability, reason = Applicability.GENERAL_CHAT, decision.reason_code
        else:
            applicability, reason = Applicability.UNKNOWN, decision.reason_code
        return decision, RoutingHint(
            applicability=applicability, candidate_domains=tuple(dict.fromkeys(candidates)),
            reason_code=reason, context_proposals=proposals,
            requires_interpretation=applicability == Applicability.UNKNOWN,
            low_cost_latency_ms=(time.perf_counter() - started) * 1000.0,
            semantic_category=semantic.category if semantic is not None else "",
            semantic_score=semantic.score if semantic is not None else 0.0,
            semantic_margin=semantic.margin if semantic is not None else 0.0,
            semantic_status=semantic_status,
        )

    @staticmethod
    def _active_context(previous: Any | None) -> dict[str, Any]:
        if previous is None or bool(getattr(previous, "needs_revalidation", False)):
            return {}
        state = dict(getattr(previous, "accepted_task_state", {}) or {})
        if not state:
            snapshot = dict(getattr(previous, "context_snapshot", {}) or {})
            if snapshot.get("literature_ir"):
                state = {"task": "literature", "current_object_ids": snapshot.get("current_document_ids", [])}
        return {} if state.get("needs_revalidation") else state

    @staticmethod
    def _proposals(query: str, active: dict[str, Any]) -> tuple[ContextProposal, ...]:
        value = str(query or "").strip()
        if _EXPLICIT_TOPIC_CHANGE.search(value) or RoutingService._is_independent_new_request(value):
            return (ContextProposal(ContextMode.NEW, True, "explicit_new_topic"),)
        proposals = [ContextProposal(ContextMode.NEW, True, "independent_candidate")]
        if active and _OBJECT_REFERENCE.search(value):
            ids = tuple(str(item) for item in active.get("current_object_ids", []) if item)
            proposals[0] = ContextProposal(ContextMode.NEW, False, "context_reference_requires_binding")
            proposals.append(ContextProposal(ContextMode.INHERIT, bool(ids),
                                              "bound_displayed_object" if ids else "referenced_object_missing",
                                              object_ids=ids))
        if active and _CONTINUE_SIGNAL.search(value):
            proposals[0] = ContextProposal(ContextMode.NEW, False, "continuation_requires_context")
            proposals.append(ContextProposal(ContextMode.INHERIT, True, "active_task_continuation"))
        if active and _CONTEXT_ACTION.search(value):
            ids = tuple(str(item) for item in active.get("current_object_ids", []) if item)
            proposals[0] = ContextProposal(ContextMode.NEW, False, "context_action_requires_objects")
            proposals.append(ContextProposal(
                ContextMode.INHERIT, bool(ids),
                "active_result_action" if ids else "referenced_object_missing",
                object_ids=ids,
            ))
        if active and _PATCH_SIGNAL.search(value):
            if _PATCH_ONLY.search(value):
                proposals[0] = ContextProposal(ContextMode.NEW, False, "patch_requires_context")
            inherited = next((item for item in proposals if item.mode == ContextMode.INHERIT and item.valid), None)
            if inherited is not None:
                proposals = [item for item in proposals if item.mode != ContextMode.INHERIT]
                proposals.append(ContextProposal(
                    ContextMode.PATCH, True, "inherited_object_with_field_patch",
                    object_ids=inherited.object_ids, field_delta={"query_fragment": value},
                ))
            else:
                proposals.append(ContextProposal(
                    ContextMode.PATCH, True, "explicit_field_patch",
                    field_delta={"query_fragment": value},
                ))
        return tuple(proposals)

    @staticmethod
    def _is_independent_new_request(query: str) -> bool:
        value = str(query or "").strip()
        if _EXPLICIT_TOPIC_CHANGE.search(value):
            return True
        if _OBJECT_REFERENCE.search(value):
            return False
        return bool(_INDEPENDENT_LITERATURE_REQUEST.search(value))

    @staticmethod
    def _effect_key(proposal: ContextProposal) -> tuple[Any, ...]:
        """Compare observable task effects, not merely candidate mode labels."""
        if proposal.mode in {ContextMode.INHERIT, ContextMode.PATCH} and proposal.object_ids:
            return ("existing_objects", proposal.object_ids, tuple(sorted(proposal.field_delta)))
        if proposal.mode == ContextMode.PATCH:
            return ("active_task_patch", tuple(sorted(proposal.field_delta)))
        return (proposal.mode.value, proposal.object_ids, tuple(sorted(proposal.field_delta)))
