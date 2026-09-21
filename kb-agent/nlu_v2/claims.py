"""Build and reconcile auditable semantic claims from all candidate sources."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .catalog import CatalogSnapshot
from .merge import walk_predicates
from .models import (
    CandidateRef,
    Diagnostic,
    Provenance,
    SemanticClaim,
    SourceSpan,
    UnderstandingIR,
)


class ClaimReconciler:
    """Converts the merged RequestIR into claims without inventing one confidence score."""

    def reconcile(self, ir: UnderstandingIR, catalog: CatalogSnapshot,
                  vector_candidates: Iterable[CandidateRef] = ()) -> list[SemanticClaim]:
        claims = self._claims_from_ir(ir)
        claims.extend(self._claims_from_vector(vector_candidates, catalog))
        claims = self._dedupe(claims)
        self._mark_conflicts(claims)
        ir.claims = claims
        conflicts = sum(1 for claim in claims if claim.status == "conflict")
        if conflicts:
            ir.diagnostics.append(Diagnostic(
                "error", "semantic_claim_conflict",
                f"存在 {conflicts} 个无法确定性消解的语义声明冲突",
            ))
        return claims

    def _claims_from_ir(self, ir: UnderstandingIR) -> list[SemanticClaim]:
        claims: list[SemanticClaim] = []
        for goal in ir.goals:
            claims.append(self._claim(
                "goal", "target", {"goal_type": goal.goal_type}, goal.span,
                getattr(goal, "provenance", None) or self._rule_evidence(goal.span, "goal"),
            ))
        for source in ir.source_candidates:
            claims.append(self._claim(
                "source_binding", "context", {"source_id": source.identifier}, None,
                [Provenance(source=source.source, score=source.score)],
                status="accepted" if source.status == "resolved" else source.status,
            ))
        for symbol in ir.projections:
            source = symbol.candidates[0].source if symbol.candidates else "rule"
            claims.append(self._claim(
                "field", "target", {
                    "raw_name": symbol.raw_name,
                    "canonical_id": symbol.canonical_id,
                    "binding_status": symbol.status,
                }, symbol.span, [Provenance(source=source, span=symbol.span, rule="field_binding")],
                status="accepted" if symbol.status == "resolved" else "candidate",
            ))
        for metric in ir.metrics:
            span = metric.field.span
            source = metric.field.candidates[0].source if metric.field.candidates else "rule"
            claims.append(self._claim(
                "metric", "target", metric.to_dict(), span,
                [Provenance(source=source, span=span, rule="metric")],
                status="accepted" if metric.field.status == "resolved" else "candidate",
            ))
        for predicate in walk_predicates(ir.filters):
            span = self._predicate_span(predicate)
            claims.append(self._claim(
                "predicate", "constraint", predicate.to_dict(), span,
                predicate.provenance or self._rule_evidence(span, "predicate"),
            ))
        for temporal in ir.temporal:
            claims.append(self._claim(
                "temporal", "constraint", temporal.to_dict(), temporal.span,
                self._rule_evidence(temporal.span, "temporal"),
            ))
        for event in ir.events:
            span = event.provenance[0].span if event.provenance else None
            claims.append(self._claim(
                "event", "constraint", event.to_dict(), span,
                event.provenance or self._rule_evidence(span, "event"),
            ))
        for operation in ir.set_operations:
            span = operation.provenance[0].span if operation.provenance else None
            claims.append(self._claim(
                "set_operation", "target", operation.to_dict(), span,
                operation.provenance or self._rule_evidence(span, "set_operation"),
            ))
        for calculation in ir.calculations:
            span = calculation.provenance[0].span if calculation.provenance else None
            claims.append(self._claim(
                "calculation", "target", calculation.to_dict(), span,
                calculation.provenance or self._rule_evidence(span, "calculation"),
            ))
        for aggregate in ir.aggregates:
            span = aggregate.aggregate.provenance[0].span if aggregate.aggregate.provenance else None
            claims.append(self._claim(
                "aggregate", "target", aggregate.to_dict(), span,
                aggregate.aggregate.provenance or self._rule_evidence(span, "aggregate"),
                status="accepted" if aggregate.aggregate.input.symbol
                and aggregate.aggregate.input.symbol.status == "resolved" else "candidate",
            ))
        for comparison in ir.comparisons:
            span = comparison.provenance[0].span if comparison.provenance else None
            claims.append(self._claim(
                "comparison", "target", comparison.to_dict(), span,
                comparison.provenance or self._rule_evidence(span, "comparison"),
            ))
        for reference in ir.references:
            claims.append(self._claim(
                "reference", "constraint", reference.to_dict(), reference.span,
                self._rule_evidence(reference.span, "reference"),
                status="accepted" if reference.status == "resolved" else "candidate",
            ))
        if ir.output.fields or ir.output.criteria or ir.output.group_by or ir.output.limit:
            claims.append(self._claim(
                "output_contract", "output", ir.output.to_dict(), None,
                [Provenance(source="derived", rule="output_contract")],
            ))
        return claims

    def _claims_from_vector(self, candidates: Iterable[CandidateRef],
                            catalog: CatalogSnapshot) -> list[SemanticClaim]:
        result = []
        for candidate in candidates:
            if catalog.source(candidate.identifier):
                claim_type = "source_binding"
                value = {"source_id": candidate.identifier}
            elif catalog.field(candidate.identifier):
                claim_type = "field"
                value = {"canonical_id": candidate.identifier}
            else:
                continue
            result.append(self._claim(
                claim_type, "context" if claim_type == "source_binding" else "uncertain",
                value, None, [Provenance(source="vector", score=candidate.score)],
                status="candidate",
            ))
        return result

    @staticmethod
    def _predicate_span(predicate) -> SourceSpan | None:
        for evidence in predicate.provenance:
            if evidence.span:
                return evidence.span
        return predicate.field.span if predicate.field else None

    @staticmethod
    def _rule_evidence(span: SourceSpan | None, rule: str) -> list[Provenance]:
        return [Provenance(source="rule", span=span, rule=rule)]

    @classmethod
    def _claim(cls, claim_type: str, role: str, value: dict[str, Any],
               span: SourceSpan | None, provenance: list[Provenance],
               status: str = "accepted") -> SemanticClaim:
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        location = f"{span.start}:{span.end}" if span else "none"
        digest = hashlib.sha1(
            f"{claim_type}|{role}|{location}|{canonical}".encode("utf-8")
        ).hexdigest()[:12]
        return SemanticClaim(
            claim_id=f"clm_{digest}", claim_type=claim_type, role=role,
            value=value, status=status, provenance=list(provenance),
        )

    @staticmethod
    def _dedupe(claims: list[SemanticClaim]) -> list[SemanticClaim]:
        result: list[SemanticClaim] = []
        by_id: dict[str, SemanticClaim] = {}
        for claim in claims:
            existing = by_id.get(claim.claim_id)
            if existing is None:
                by_id[claim.claim_id] = claim
                result.append(claim)
                continue
            seen = {(item.source, item.rule, item.score) for item in existing.provenance}
            existing.provenance.extend(
                item for item in claim.provenance
                if (item.source, item.rule, item.score) not in seen
            )
            if existing.status == "candidate" and claim.status == "accepted":
                existing.status = "accepted"
        return result

    @staticmethod
    def _mark_conflicts(claims: list[SemanticClaim]) -> None:
        buckets: dict[tuple[str, str], list[SemanticClaim]] = {}
        for claim in claims:
            if claim.status != "accepted":
                continue
            span = next((item.span for item in claim.provenance if item.span), None)
            if claim.claim_type == "source_binding":
                key = (claim.claim_type, "single_source")
            elif span and claim.claim_type in {"field", "predicate", "calculation"}:
                key = (claim.claim_type, f"{span.start}:{span.end}")
            else:
                continue
            buckets.setdefault(key, []).append(claim)
        for bucket in buckets.values():
            values = {
                json.dumps(item.value, ensure_ascii=False, sort_keys=True, default=str)
                for item in bucket
            }
            if len(values) <= 1:
                continue
            for claim in bucket:
                claim.status = "conflict"
                claim.conflicts_with = [
                    item.claim_id for item in bucket if item.claim_id != claim.claim_id
                ]
