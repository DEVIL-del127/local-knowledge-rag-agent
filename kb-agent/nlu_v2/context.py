"""Pure, read-only turn context resolution for the semantic compiler."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .models import (
    ContextDelta, ContextDeltaOperation, Provenance, SourceSpan, TurnContextSnapshot,
    TurnDirective, UnderstandingIR, UnresolvedItem,
)


_CONTEXT_CUE = re.compile(r"停止刚才|取消(?:旧|上一个|前一个)?|上一轮|上一步|前一步|改成|替换|只看|那.+呢|不是.+?是|先说.+?随后")


def context_snapshot_digest(snapshot_ir) -> str:
    payload = snapshot_ir.to_dict() if hasattr(snapshot_ir, "to_dict") else snapshot_ir
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TurnContextResolver:
    """Resolve explicit follow-ups without owning mutable conversation state."""

    def resolve(self, ir: UnderstandingIR,
                snapshot: TurnContextSnapshot | None) -> UnderstandingIR:
        query = ir.query.normalized
        match = _CONTEXT_CUE.search(query)
        if not match:
            return ir
        same_turn = bool(re.search(r"不是.+?是|先说.+?随后", query))
        # A same-turn correction governs the complete message, including its
        # replacement, exclusions and explicit record of the revoked value.
        # Its provenance therefore intentionally covers the whole control
        # instruction instead of only the first cue phrase.
        span = (SourceSpan(0, len(query), query) if same_turn
                else SourceSpan(match.start(), match.end(), match.group(0)))
        if same_turn and snapshot is None:
            directive = TurnDirective(
                directive_id="turn_" + hashlib.sha1(query.encode("utf-8")).hexdigest()[:12],
                primary_action="replace_constraints", context_mode="fresh",
                context_delta=ContextDelta(owner="new_turn", operations=[
                    ContextDeltaOperation("replace", "constraints", "latest_explicit_correction")
                ]),
                provenance=[Provenance(source="rule", rule="same_turn_correction", span=span)],
            )
            ir.turn_directives = [directive]
            if len(ir.temporal) > 1:
                correction_start = query.find("不是")
                corrected = [item for item in ir.temporal
                             if item.span and item.span.start >= correction_start]
                if corrected:
                    ir.temporal = corrected
            return ir
        if snapshot is None:
            ir.unresolved.append(UnresolvedItem(
                code="context_required", message="当前请求需要上一轮只读语义快照", span=span,
            ))
            return ir
        previous = snapshot.accepted_semantic_ir
        if not isinstance(previous, UnderstandingIR):
            ir.unresolved.append(UnresolvedItem(
                code="context_snapshot_incompatible",
                message="上一轮语义快照不是兼容的 RequestIR 对象", span=span,
            ))
            return ir
        if snapshot.schema_version != previous.schema_version or (
            snapshot.catalog_version and snapshot.catalog_version != previous.catalog_version
        ):
            ir.unresolved.append(UnresolvedItem(
                code="context_snapshot_incompatible", message="上一轮语义快照版本不兼容", span=span,
            ))
            return ir
        if snapshot.context_digest != context_snapshot_digest(previous):
            ir.unresolved.append(UnresolvedItem(
                code="context_snapshot_mismatch", message="上一轮语义快照摘要校验失败", span=span,
            ))
            return ir

        inherited = copy.deepcopy(previous)
        inherited.query = ir.query
        inherited.clause_graph = ir.clause_graph
        inherited.schema_version = ir.schema_version
        inherited.catalog_version = ir.catalog_version
        inherited.diagnostics = list(ir.diagnostics)
        inherited.unresolved = list(ir.unresolved)
        inherited.claims = list(ir.claims)
        inherited.requirements = list(ir.requirements)
        inherited.coverage = []
        # Current explicit structures replace inherited slots; omitted slots are retained.
        for name in (
            "goals", "source_candidates", "schema_hypotheses", "projections", "metrics",
            "temporal", "sampling_policies", "events", "set_operations", "grouping",
            "calculations", "aggregates", "comparisons", "arithmetic", "conversions",
            "ordered_folds", "unsatisfiable", "references",
        ):
            current_value = getattr(ir, name)
            if current_value:
                setattr(inherited, name, copy.deepcopy(current_value))
        # Demands belong to the current query envelope: their spans and clause
        # IDs must never leak from the previous turn.  Current requirements may
        # reference them, while inherited semantics remain accepted context.
        inherited.source_demands = copy.deepcopy(ir.source_demands)
        if ir.filters is not None:
            inherited.filters = copy.deepcopy(ir.filters)
        if ir.output.fields or ir.output.group_by or ir.output.criteria or ir.output.limit:
            inherited.output = copy.deepcopy(ir.output)

        pre_actions = ["cancel_previous"] if re.search(r"停止刚才|取消", query) else []
        delta_ops = []
        if ir.temporal:
            delta_ops.append(ContextDeltaOperation("replace", "temporal", copy.deepcopy(ir.temporal)))
        if ir.filters:
            delta_ops.append(ContextDeltaOperation("replace", "filters", copy.deepcopy(ir.filters)))
        directive = TurnDirective(
            directive_id="turn_" + hashlib.sha1(query.encode("utf-8")).hexdigest()[:12],
            pre_actions=pre_actions, primary_action="refine",
            target_turn_id=snapshot.previous_turn_id, context_mode="inherit",
            base_context_ref=f"{snapshot.previous_turn_id}:{snapshot.context_digest[:12]}",
            context_delta=ContextDelta(owner="new_turn", operations=delta_ops),
            provenance=[Provenance(source="rule", rule="turn_context", span=span)],
        )
        inherited.turn_directives = [directive]
        return inherited
