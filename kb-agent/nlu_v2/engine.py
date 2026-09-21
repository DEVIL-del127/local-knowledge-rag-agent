"""NLU V2 orchestration with bounded optional model enrichment."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Mapping

from .catalog import CatalogProvider, CatalogSnapshot, StaticCatalogProvider
from .clause_graph import ClauseGraphBuilder
from .claims import ClaimReconciler
from .clarification import ClarificationPlanner
from .context import TurnContextResolver
from .coverage import CoverageMatcher
from .demand_ledger import SourceDemandLedger
from .event_closure import EventClosureAnalyzer
from .canonical_requirements import CanonicalRequirementGraph
from .event_execution import EventExecutionCompiler
from .field_binding import FieldRoleBinder
from .quantities import QuantityRegistry
from .typed_lineage import TypedLineageCompiler
from .m15_planner import M15LogicalPlanner
from .reference_binding import ReferenceBinder
from .gaps import GapAnalyzer
from .llm_extractor import BoundedLLMExtractor
from .logical_planner import LogicalPlanner
from .merge import merge_llm_candidate
from .models import (
    CandidateRef, Diagnostic, EngineResult, LLMAuditRecord, PatchReport, TraceEvent, TurnContextSnapshot,
)
from .literature_semantics import analyze_literature
from .operators import GenericOperatorEnricher, OperatorRegistry
from .patch_protocol import (
    SemanticPatchV1,
    apply_semantic_patch,
    request_ir_digest,
)
from .requirements import RequirementExtractor
from .rule_extractor import AtomicRuleExtractor
from .security import CatalogAuthorizer, SecurityContext
from .requirement_normalizer import RequirementNormalizer
from .semantic_linker import DeterministicSemanticLinker
from .semantic_candidates import (
    CANDIDATE_CHOICE_PROTOCOL,
    CandidateMenu,
    SemanticCandidateCompiler,
)
from .semantic_repair import (
    CompilationSnapshot,
    RepairContract,
    RepairUnit,
    SemanticAcceptanceGate,
    contract_violation,
)
from .validator import IRValidator
from .vector_candidates import NullVectorCandidateProvider, VectorCandidateProvider


@dataclass(slots=True)
class EngineConfig:
    enable_vector: bool = False
    enable_llm: bool = False
    llm_on_complex: bool = False
    max_vector_candidates: int = 5
    cache_size: int = 256
    breaker_failures: int = 3
    breaker_cooldown_seconds: float = 60.0
    max_query_chars: int = 12000
    max_llm_catalog_fields: int = 80
    require_security_context: bool = False
    enable_patch_v1: bool = True
    enable_requirement_ir_v2: bool = True
    enable_operator_registry: bool = True
    enable_atomic_patch_v2: bool = True
    enable_semantic_repair: bool = True
    enable_requirement_normalizer: bool = True
    enable_semantic_linker: bool = True
    enable_event_closure_contract: bool = True
    enable_semantic_target_gate: bool = True
    enable_candidate_choice_v3: bool = False
    enable_m15_requirement_graph_shadow: bool = False
    enable_m15_field_binding_shadow: bool = False
    enable_m15_quantity_contract: bool = False
    enable_m15_temporal_contracts: bool = False
    enable_m15_typed_lineage: bool = False
    enable_m15_logical_plan: bool = False


class QueryUnderstandingEngine:
    def __init__(
        self,
        *,
        catalog_provider: CatalogProvider | None = None,
        rule_extractor: AtomicRuleExtractor | None = None,
        vector_provider: VectorCandidateProvider | None = None,
        llm_extractor: BoundedLLMExtractor | None = None,
        validator: IRValidator | None = None,
        planner: LogicalPlanner | None = None,
        claim_reconciler: ClaimReconciler | None = None,
        requirement_extractor: RequirementExtractor | None = None,
        catalog_authorizer: CatalogAuthorizer | None = None,
        gap_analyzer: GapAnalyzer | None = None,
        clause_graph_builder: ClauseGraphBuilder | None = None,
        coverage_matcher: CoverageMatcher | None = None,
        demand_ledger: SourceDemandLedger | None = None,
        semantic_acceptance_gate: SemanticAcceptanceGate | None = None,
        clarification_planner: ClarificationPlanner | None = None,
        context_resolver: TurnContextResolver | None = None,
        operator_registry: OperatorRegistry | None = None,
        requirement_normalizer: RequirementNormalizer | None = None,
        semantic_linker: DeterministicSemanticLinker | None = None,
        event_closure_analyzer: EventClosureAnalyzer | None = None,
        semantic_candidate_compiler: SemanticCandidateCompiler | None = None,
        field_role_binder: FieldRoleBinder | None = None,
        event_execution_compiler: EventExecutionCompiler | None = None,
        config: EngineConfig | None = None,
    ):
        self.config = config or EngineConfig()
        self.catalog_provider = catalog_provider or StaticCatalogProvider()
        self.rule_extractor = rule_extractor or AtomicRuleExtractor()
        self.vector_provider = vector_provider or NullVectorCandidateProvider()
        self.llm_extractor = llm_extractor
        self.validator = validator or IRValidator()
        self.planner = planner or LogicalPlanner()
        self.claim_reconciler = claim_reconciler or ClaimReconciler()
        self.requirement_extractor = requirement_extractor or RequirementExtractor()
        self.catalog_authorizer = catalog_authorizer or CatalogAuthorizer()
        self.gap_analyzer = gap_analyzer or GapAnalyzer()
        self.clause_graph_builder = clause_graph_builder or ClauseGraphBuilder()
        self.coverage_matcher = coverage_matcher or CoverageMatcher()
        self.coverage_matcher.enforce_semantic_target_gate = (
            self.config.enable_semantic_target_gate
        )
        self.demand_ledger = demand_ledger or SourceDemandLedger()
        self.semantic_acceptance_gate = semantic_acceptance_gate or SemanticAcceptanceGate(
            validator=self.validator, coverage_matcher=self.coverage_matcher,
        )
        self.clarification_planner = clarification_planner or ClarificationPlanner()
        self.context_resolver = context_resolver or TurnContextResolver()
        self.operator_registry = operator_registry or OperatorRegistry()
        self.operator_enricher = GenericOperatorEnricher(self.operator_registry)
        self.requirement_normalizer = requirement_normalizer or RequirementNormalizer()
        self.semantic_linker = semantic_linker or DeterministicSemanticLinker()
        self.event_closure_analyzer = event_closure_analyzer or EventClosureAnalyzer()
        self.semantic_candidate_compiler = semantic_candidate_compiler or SemanticCandidateCompiler()
        self.field_role_binder = field_role_binder or FieldRoleBinder()
        self.event_execution_compiler = event_execution_compiler or EventExecutionCompiler(
            self.field_role_binder
        )
        self.quantity_registry = QuantityRegistry()
        self.typed_lineage_compiler = TypedLineageCompiler()
        self.m15_planner = M15LogicalPlanner()
        self.reference_binder = ReferenceBinder()
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, ...], dict] = {}
        self._cache_order: list[tuple[str, ...]] = []
        self._failures = 0
        self._breaker_until = 0.0

    def analyze(self, query: str, *,
                security_context: SecurityContext | None = None,
                context_snapshot: TurnContextSnapshot | None = None) -> EngineResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > self.config.max_query_chars:
            raise ValueError(f"query exceeds {self.config.max_query_chars} characters")
        if self.config.require_security_context and security_context is None:
            raise PermissionError("trusted SecurityContext is required")
        security = security_context or SecurityContext.local_development()
        security_digest = security.digest()
        trace: list[TraceEvent] = []
        raw_catalog = self._timed(trace, "catalog", "loaded versioned snapshot",
                                  self.catalog_provider.snapshot)
        catalog = self._timed(
            trace, "authorize_catalog", "created authorized catalog view",
            lambda: self.catalog_authorizer.authorize(raw_catalog, security),
        )
        extraction = self._timed(
            trace, "rules", "extracted atomic syntax",
            lambda: self.rule_extractor.extract(query, catalog),
        )
        ir = extraction.ir
        ir.query.security_scope_digest = security_digest
        self._timed(
            trace, "literature_contract", "merged literature semantics into RequestIR",
            lambda: self._attach_literature_contract(ir, catalog, context_snapshot),
        )
        if self.config.enable_requirement_ir_v2:
            ir.clause_graph = self._timed(
                trace, "clause_graph", "built lossless clause graph",
                lambda: self.clause_graph_builder.build(ir.query.normalized),
            )
            ledger = self._timed(
                trace, "demand_ledger", "anchored source demands before semantic compilation",
                lambda: self.demand_ledger.build(ir.query.normalized, ir.clause_graph),
            )
            ir.source_demands = ledger.demands
            ir.query_schema = self._timed(
                trace, "query_schema", "built query-local schema snapshot",
                lambda: catalog.query_schema(
                    ir.query.normalized,
                    hypotheses=[item.hypothesis_id for item in ir.schema_hypotheses],
                ),
            )
            ir.requirements = self._timed(
                trace, "requirements", "extracted independent requirements from source text",
                lambda: self.requirement_extractor.extract(ir.query.normalized, ir.clause_graph, ledger),
            )
            if self.config.enable_requirement_normalizer:
                ir.requirements = self._timed(
                    trace, "requirement_normalizer", "normalized evidence into semantic obligations",
                    lambda: self.requirement_normalizer.normalize(
                        ir.query.normalized, ir.clause_graph, ir.requirements, ir.source_demands,
                    ),
                )
        if self.config.enable_operator_registry:
            self._timed(trace, "operators", "applied domain-neutral operator enrichers",
                        lambda: self.operator_enricher.enrich(ir, catalog))
        if self.config.enable_semantic_linker:
            self._timed(trace, "semantic_linker", "closed deterministic semantic dataflow",
                        lambda: self.semantic_linker.link(ir))
            if self.config.enable_operator_registry:
                self.operator_enricher.refresh(ir)
        if self.config.enable_requirement_ir_v2:
            ir = self._timed(
                trace, "context", "resolved explicit immutable turn context",
                lambda: self.context_resolver.resolve(ir, context_snapshot),
            )
            if self.config.enable_operator_registry:
                self.operator_enricher.refresh(ir)
            if self.config.enable_semantic_linker:
                self.semantic_linker.link(ir)
                if self.config.enable_operator_registry:
                    self.operator_enricher.refresh(ir)
            self.coverage_matcher.apply(ir)
            if (self.config.enable_operator_registry
                    and self.operator_enricher.finalize_coverage_lineage(ir)):
                self.coverage_matcher.apply(ir)

        vector_candidates = []
        if self.config.enable_vector:
            vector_candidates = self._timed(
                trace, "vector", "resolved optional candidates",
                lambda: self.vector_provider.candidates(
                    ir.query.normalized, catalog, self.config.max_vector_candidates
                ),
            )
            existing_sources = {item.identifier for item in ir.source_candidates}
            for candidate in vector_candidates:
                if catalog.source(candidate.identifier) and candidate.identifier not in existing_sources:
                    candidate.status = "candidate"
                    ir.source_candidates.append(candidate)
                    existing_sources.add(candidate.identifier)
            for unresolved in ir.unresolved:
                unresolved.candidates.extend(
                    item.identifier for item in vector_candidates if catalog.field(item.identifier)
                )

        model_calls = 0
        patch_report: PatchReport | None = None
        llm_audit: LLMAuditRecord | None = None
        all_gaps = self.gap_analyzer.analyze(ir, catalog) if self.config.enable_patch_v1 else []
        ready_gaps = [
            item for item in all_gaps
            if item.readiness == "ready" and item.resolution_mode == "llm"
        ]
        unit_gaps = all_gaps if self.config.enable_candidate_choice_v3 else ready_gaps
        repair_units = self.gap_analyzer.repair_units(unit_gaps)
        # The 7B protocol deliberately has one atomic response schema.  Select
        # the highest-priority connected unit; independent units remain open for
        # a later user turn instead of being mixed in one unconstrained request.
        ready_gap_ids = {item.gap_id for item in ready_gaps}
        active_unit = next((item for item in repair_units if any(
            gap.gap_id in ready_gap_ids
            and any(operation != "add_ambiguity" for operation in gap.allowed_operations)
            for gap in item.gaps
        )), repair_units[0] if repair_units else None)
        if (self.config.enable_atomic_patch_v2 and not self.config.enable_candidate_choice_v3
                and active_unit and len(active_unit.gaps) > 1):
            # Small local models are substantially more reliable when asked to
            # close one typed gap.  Dependency-connected gaps remain visible in
            # CoverageDiff, but only the highest-priority atomic work order is
            # submitted in this bounded model call.
            atomic_gap = next((
                gap for gap in active_unit.gaps
                if any(operation != "add_ambiguity" for operation in gap.allowed_operations)
            ), active_unit.gaps[0])
            active_unit = RepairUnit.from_gaps([atomic_gap])[0]
        gaps = [item for item in active_unit.gaps if item.gap_id in ready_gap_ids] \
            if active_unit else []
        if all_gaps:
            trace.append(TraceEvent("gaps", f"identified {len(all_gaps)} deterministic semantic gaps"))
            blocked = len(all_gaps) - len(ready_gaps)
            if blocked:
                trace.append(TraceEvent(
                    "gaps", f"deferred {blocked} non-ready gaps before model dispatch",
                ))
        candidate_choice_handled = self.config.enable_candidate_choice_v3
        if self.config.enable_candidate_choice_v3 and active_unit and gaps:
            ir, patch_report, llm_audit, model_calls = self._execute_candidate_bundle_v3(
                ir, gaps[0], active_unit, catalog, security_digest, trace,
            )
        if not candidate_choice_handled and self._should_call_llm(ir, gaps):
            cache_key = self._cache_key(
                ir.query.normalized, catalog.version, security_digest
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                if self.config.enable_patch_v1:
                    try:
                        cached_patch = SemanticPatchV1.model_validate(cached)
                        candidate_ir, patch_report = self._apply_patch_transaction(
                            ir, cached_patch, gaps, catalog, repair_unit=active_unit,
                        )
                        if patch_report.committed:
                            ir = candidate_ir
                            trace.append(TraceEvent("llm", "used committed semantic Patch cache"))
                        else:
                            trace.append(TraceEvent("llm", "ignored stale semantic Patch cache"))
                    except Exception as exc:
                        patch_report = PatchReport(
                            transaction_error="invalid_cached_patch",
                            rejection_gate="cache_validation_gate",
                        )
                        ir.diagnostics.append(Diagnostic(
                            "warning", "llm_cache_invalid", f"忽略无效 Patch 缓存：{exc}",
                        ))
                else:
                    merge_llm_candidate(ir, cached, catalog)
                    trace.append(TraceEvent("llm", "used validated candidate cache"))
            elif self._breaker_open():
                ir.diagnostics.append(Diagnostic(
                    "warning", "llm_breaker_open", "结构化 LLM 熔断中，本次仅使用本地解析",
                ))
                trace.append(TraceEvent("llm", "skipped because breaker is open"))
            else:
                llm_started = time.perf_counter()
                candidate = self._timed(
                    trace, "llm", "requested one bounded IR candidate",
                    lambda: self.llm_extractor.extract(
                        ir.query.normalized, self._catalog_payload(catalog, ir),
                        self._existing_ir_summary(ir),
                        gaps=gaps,
                        base_ir_digest=request_ir_digest(ir),
                        enable_patch_v1=self.config.enable_patch_v1,
                        enable_atomic_patch_v2=self.config.enable_atomic_patch_v2,
                    ),
                )
                llm_duration_ms = round((time.perf_counter() - llm_started) * 1000, 3)
                model_calls = min(2, candidate.attempts)
                provider = getattr(self.llm_extractor, "provider", None)
                llm_audit = LLMAuditRecord(
                    provider=type(provider).__name__ if provider is not None else "",
                    model=str(getattr(provider, "model", "")),
                    repair_unit_id=active_unit.unit_id if active_unit else "",
                    gap_ids=list(candidate.selected_gap_ids),
                    atomic_operation=candidate.atomic_operation,
                    protocol=candidate.protocol,
                    attempts=candidate.attempts,
                    prompt_chars=candidate.prompt_chars,
                    response_chars=candidate.response_chars,
                    raw_response_excerpt=candidate.raw_response_excerpt,
                    parsed_json_excerpt=candidate.parsed_json_excerpt,
                    parse_error=candidate.error,
                    failure_kind=candidate.failure_kind,
                    duration_ms=llm_duration_ms,
                    stop_reason=candidate.stop_reason,
                )
                if candidate.valid_json:
                    if self.config.enable_patch_v1 and candidate.patch is not None:
                        candidate_ir, patch_report = self._apply_patch_transaction(
                            ir, candidate.patch, gaps, catalog, repair_unit=active_unit,
                        )
                        patch_report.protocol = candidate.protocol or "semantic_patch_v1"
                        if patch_report.committed:
                            ir = candidate_ir
                            self._cache_put(cache_key, candidate.patch.model_dump(mode="json"))
                            self._record_llm_success()
                        elif candidate.patch.operations:
                            ir.diagnostics.append(Diagnostic(
                                "warning", "llm_nodes_rejected",
                                "模型 Patch 未关闭目标缺口或未通过事务校验，已整体回滚",
                            ))
                        else:
                            ir.diagnostics.append(Diagnostic(
                                "info", "llm_no_gain", "模型返回空 Patch，未改变本地语义结果",
                            ))
                    else:
                        merge_llm_candidate(ir, candidate.payload, catalog)
                        self._cache_put(cache_key, candidate.payload)
                        self._record_llm_success()
                else:
                    if self.config.enable_patch_v1 and candidate.failure_kind == "schema":
                        patch_report = PatchReport(
                            protocol=candidate.protocol or (
                                "semantic_patch_v2" if candidate.atomic_operation else "semantic_patch_v1"
                            ),
                            base_ir_digest=request_ir_digest(ir),
                            transaction_error="llm_patch_parse_failed",
                            rejection_gate="schema_parse_gate",
                        )
                    if candidate.failure_kind == "provider" or self._is_provider_failure(candidate.error):
                        self._record_llm_failure()
                    if candidate.error:
                        diagnostic_code = (
                            self._provider_diagnostic_code(candidate.error)
                            if candidate.failure_kind == "provider" or self._is_provider_failure(candidate.error)
                            else "llm_candidate_failed"
                        )
                        ir.diagnostics.append(Diagnostic(
                            "warning", diagnostic_code, candidate.error,
                        ))
                if llm_audit is not None and patch_report is not None:
                    llm_audit.rejected_gate = patch_report.rejection_gate
                    llm_audit.rejection_reason = patch_report.transaction_error
                    if patch_report.semantic_effect is not None:
                        llm_audit.coverage_before = dict(
                            patch_report.semantic_effect.coverage_before
                        )
                        llm_audit.coverage_after = dict(
                            patch_report.semantic_effect.coverage_after
                        )
                        llm_audit.closed_requirement_ids = list(
                            patch_report.semantic_effect.closed_requirement_ids
                        )

        self._timed(
            trace, "claims", "reconciled rule, vector, and model evidence",
            lambda: self.claim_reconciler.reconcile(ir, catalog, vector_candidates),
        )
        if self.config.enable_requirement_ir_v2:
            self._timed(
                trace, "coverage", "matched independent requirements to typed SemanticIR",
                lambda: self.coverage_matcher.apply(ir),
            )
            if (self.config.enable_operator_registry
                    and self.operator_enricher.finalize_coverage_lineage(ir)):
                self._timed(
                    trace, "coverage_rebind", "rebound typed lineage from proven coverage",
                    lambda: self.coverage_matcher.apply(ir),
                )

        validation = self._timed(
            trace, "validate", "validated binding and phase-one safety",
            lambda: self.validator.validate(ir, catalog),
        )
        validation.dimensions["llm_effect"] = self._llm_effect_dimension(
            patch_report, model_calls, ir.diagnostics,
        )
        logical = self._timed(
            trace, "logical_plan", "built read-only single-source DAG",
            lambda: self.planner.plan(ir, validation, catalog),
        )
        physical = self.planner.physical_shell(ir)
        trace.append(TraceEvent("physical_plan", "left unbound for root Agent SkillRegistry"))
        clarification_plan = self._timed(
            trace, "clarification", "prepared clarification-only recovery when execution is blocked",
            lambda: self.clarification_planner.plan(
                ir, validation, patch_report.semantic_effect if patch_report else None,
            ),
        )
        event_closure_report = (
            self.event_closure_analyzer.analyze(ir)
            if self.config.enable_event_closure_contract else None
        )
        canonical_requirement_graph = (
            CanonicalRequirementGraph.build(ir)
            if self.config.enable_m15_requirement_graph_shadow else None
        )
        field_binding_report = (
            self.field_role_binder.bind(ir, catalog)
            if self.config.enable_m15_field_binding_shadow else None
        )
        quantity_report = (
            {"registry_version": "m15-quantity-v1", "status": "shadow"}
            if self.config.enable_m15_quantity_contract else None
        )
        event_readiness_report = (
            self.event_execution_compiler.compile(ir, catalog)
            if self.config.enable_m15_temporal_contracts else None
        )
        typed_lineage_report = (
            self.typed_lineage_compiler.compile(ir)
            if self.config.enable_m15_typed_lineage else None
        )
        m15_logical_plan = (
            self.m15_planner.plan(ir, catalog, event_readiness_report, typed_lineage_report)
            if self.config.enable_m15_logical_plan else None
        )
        reference_binding_report = (
            self.reference_binder.bind(ir)
            if self.config.enable_m15_typed_lineage else None
        )
        return EngineResult(
            ir, logical, physical, validation, trace, model_calls,
            patch_report=patch_report,
            semantic_effect=patch_report.semantic_effect if patch_report else None,
            clarification_plan=clarification_plan,
            compilation_snapshot=CompilationSnapshot.capture(ir),
            llm_audit=llm_audit,
            event_closure_report=event_closure_report,
            canonical_requirement_graph=canonical_requirement_graph,
            field_binding_report=field_binding_report,
            quantity_report=quantity_report,
            event_readiness_report=event_readiness_report,
            typed_lineage_report=typed_lineage_report,
            m15_logical_plan=m15_logical_plan,
            reference_binding_report=reference_binding_report,
        )

    @staticmethod
    def _attach_literature_contract(ir, catalog, context_snapshot=None) -> None:
        request = analyze_literature(ir.query.normalized).request
        text = ir.query.normalized
        is_literature = bool(
            request.canonical_topic
            or request.document_reference.kind.value != "none"
            or re.search(r"论文|文献|知识库|论文库|文献库|库里|库内|paper\b|doi\b", text, re.I)
            or (context_snapshot is not None and bool(getattr(
                context_snapshot.accepted_semantic_ir, "literature_contract", {}
            )))
        )
        if not is_literature:
            return
        previous_contract = {}
        if context_snapshot is not None:
            previous_contract = dict(
                getattr(context_snapshot.accepted_semantic_ir, "literature_contract", {}) or {}
            )
        if previous_contract:
            from dataclasses import replace
            from agent.literature_ir import DocumentReference, LiteratureRequestIR, ReferenceKind
            previous_request = LiteratureRequestIR.from_dict(previous_contract)
            fragment = ir.query.normalized.strip()
            if re.fullmatch(r"(?:那|只)?\s*(?:19|20)\d{2}\s*年的?(?:呢)?[？?]?", fragment, re.I):
                request = replace(
                    previous_request, raw_query=ir.query.normalized,
                    temporal=request.temporal or previous_request.temporal,
                )
            elif re.fullmatch(r"(?:总结|概括)(?:一下|下|呢)?[？?]?", fragment, re.I):
                request = replace(
                    previous_request, raw_query=ir.query.normalized,
                    task=type(previous_request.task).SUMMARIZE,
                    document_reference=DocumentReference(ReferenceKind.PRIOR_RESULTS),
                )
            elif re.fullmatch(r"(?:比较|对比)(?:一下|下|呢)?[？?]?", fragment, re.I):
                request = replace(
                    previous_request, raw_query=ir.query.normalized,
                    task=type(previous_request.task).COMPARE,
                    document_reference=DocumentReference(ReferenceKind.PRIOR_RESULTS, expected_count=2),
                )
            ir.unresolved = [item for item in ir.unresolved if item.code != "context_required"]
        required = "inventory" if request.task.value == "inventory" else "retrieve"
        eligible = [
            source for source in catalog.sources
            if source.read_only and required in set(source.capabilities)
        ]
        ir.literature_contract = request.to_dict()
        if len(eligible) == 1:
            source = eligible[0]
            ir.source_candidates = [
                item for item in ir.source_candidates if item.identifier != source.source_id
            ]
            ir.source_candidates.append(CandidateRef(
                identifier=source.source_id, score=1.0,
                source="current_authorized_catalog", status="resolved",
            ))
            ir.source_binding = {
                "source_id": source.source_id,
                "binding_origin": "current_authorized_catalog",
                "required_capability": required,
                "available_capabilities": list(source.capabilities),
            }
        elif not eligible:
            ir.source_binding = {
                "binding_origin": "authorized_catalog",
                "required_capability": required,
                "error_code": "unsupported_capability",
            }
        else:
            ir.source_binding = {
                "binding_origin": "authorized_catalog",
                "required_capability": required,
                "candidate_source_ids": [item.source_id for item in eligible],
                "error_code": "catalog_source_ambiguous",
            }
        identity = {
            "literature_contract": ir.literature_contract,
            "source_binding": ir.source_binding,
            "catalog_version": ir.catalog_version,
        }
        ir.accepted_ir_digest = hashlib.sha256(json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()

    def _execute_candidate_choice_v3(self, ir, gap, active_unit, catalog,
                                     security_digest, trace):
        menu = self.semantic_candidate_compiler.compile(ir, gap, catalog)
        trace.append(TraceEvent(
            "candidate_compile",
            f"dispatch={menu.dispatch}, candidates={len(menu.candidates)}, "
            f"blocked={','.join(menu.blocked_by) or '-'}",
        ))
        if menu.dispatch == "blocked_by_upstream":
            ir.diagnostics.append(Diagnostic(
                "info", "candidate_blocked_by_upstream",
                "候选编译被上游语义缺口阻断：" + ", ".join(menu.blocked_by),
            ))
            return ir, None, None, 0
        if menu.dispatch == "clarify":
            ir.diagnostics.append(Diagnostic(
                "warning", "candidate_menu_ambiguous",
                "存在过多合法语义候选，需要用户澄清",
            ))
            return ir, None, None, 0
        if menu.dispatch == "local_compile":
            patch, error = self.semantic_candidate_compiler.compile_patch(
                menu.candidates[0], gap, ir,
            )
            if patch is None:
                report = PatchReport(
                    protocol=CANDIDATE_CHOICE_PROTOCOL,
                    base_ir_digest=request_ir_digest(ir),
                    transaction_error=error or "bundle_compile_failure",
                    rejection_gate="bundle_compile_gate",
                )
                return ir, report, None, 0
            candidate_ir, report = self._apply_patch_transaction(
                ir, patch, [gap], catalog, repair_unit=active_unit,
            )
            report.protocol = CANDIDATE_CHOICE_PROTOCOL
            if report.committed:
                trace.append(TraceEvent(
                    "candidate_compile", "committed unique local semantic candidate",
                ))
                return candidate_ir, report, None, 0
            return ir, report, None, 0

        if not self._should_call_llm(ir, [gap]):
            ir.diagnostics.append(Diagnostic(
                "warning", "candidate_choice_requires_llm",
                "存在多个合法语义候选，但本次未启用 LLM，只能请求澄清",
            ))
            return ir, None, None, 0
        if self._breaker_open():
            ir.diagnostics.append(Diagnostic(
                "warning", "llm_breaker_open", "结构化 LLM 熔断中，本次候选未选择",
            ))
            return ir, None, None, 0

        cache_key = self._candidate_choice_cache_key(menu, catalog.version, security_digest)
        cached = self._cache_get(cache_key)
        provider = getattr(self.llm_extractor, "provider", None)
        model = str(getattr(provider, "model", ""))
        if isinstance(cached, dict) and cached.get("candidate_id"):
            candidate_id = str(cached["candidate_id"])
            selected = menu.candidate(candidate_id)
            llm_candidate = None
            duration_ms = 0.0
            decision = "select" if selected else "invalid"
        else:
            started = time.perf_counter()
            llm_candidate = self.llm_extractor.choose_candidate(menu)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            decision = str(llm_candidate.payload.get("decision", "invalid"))
            candidate_id = str(llm_candidate.payload.get("candidate_id", ""))
            selected = menu.candidate(candidate_id) if decision == "select" else None

        audit = LLMAuditRecord(
            provider=type(provider).__name__ if provider is not None else "",
            model=model, repair_unit_id=active_unit.unit_id,
            gap_ids=[gap.gap_id], atomic_operation="candidate_select",
            protocol=CANDIDATE_CHOICE_PROTOCOL,
            attempts=llm_candidate.attempts if llm_candidate else 0,
            prompt_chars=llm_candidate.prompt_chars if llm_candidate else 0,
            response_chars=llm_candidate.response_chars if llm_candidate else 0,
            raw_response_excerpt=llm_candidate.raw_response_excerpt if llm_candidate else "",
            parsed_json_excerpt=llm_candidate.parsed_json_excerpt if llm_candidate else "",
            parse_error=llm_candidate.error if llm_candidate else "",
            failure_kind=llm_candidate.failure_kind if llm_candidate else "",
            duration_ms=duration_ms,
            stop_reason=llm_candidate.stop_reason if llm_candidate else "cache_hit",
            source_clause_digest=menu.source_clause_digest,
            target_contract_digest=menu.target_contract_digest,
            base_lineage_digest=menu.base_lineage_digest,
            candidate_menu_digest=menu.candidate_menu_digest,
            candidate_compiler_version=menu.candidate_compiler_version,
            expression_signature_version=menu.expression_signature_version,
            prompt_version=CANDIDATE_CHOICE_PROTOCOL,
            model_identity=model,
            decision=decision, candidate_id=candidate_id,
        )
        model_calls = min(1, llm_candidate.attempts) if llm_candidate else 0
        if llm_candidate and not llm_candidate.valid_json:
            if llm_candidate.failure_kind == "provider":
                self._record_llm_failure()
            report = PatchReport(
                protocol=CANDIDATE_CHOICE_PROTOCOL,
                base_ir_digest=request_ir_digest(ir),
                transaction_error=llm_candidate.error or "choice_schema_failure",
                rejection_gate="candidate_choice_parse_gate",
            )
            audit.rejected_gate = report.rejection_gate
            audit.rejection_reason = report.transaction_error
            return ir, report, audit, model_calls
        if decision == "none_of_above":
            ir.diagnostics.append(Diagnostic(
                "warning", "candidate_none_of_above",
                "本地候选均不符合原文，未生成语义 Patch",
            ))
            return ir, None, audit, model_calls
        if decision == "ambiguous":
            ir.diagnostics.append(Diagnostic(
                "warning", "candidate_choice_ambiguous",
                "多个本地候选均可能成立，需要用户澄清",
            ))
            return ir, None, audit, model_calls
        if selected is None:
            report = PatchReport(
                protocol=CANDIDATE_CHOICE_PROTOCOL,
                base_ir_digest=request_ir_digest(ir),
                transaction_error="candidate_not_allowed",
                rejection_gate="candidate_allowlist_gate",
            )
            audit.rejected_gate = report.rejection_gate
            audit.rejection_reason = report.transaction_error
            return ir, report, audit, model_calls
        patch, error = self.semantic_candidate_compiler.compile_patch(selected, gap, ir)
        if patch is None:
            report = PatchReport(
                protocol=CANDIDATE_CHOICE_PROTOCOL,
                base_ir_digest=request_ir_digest(ir),
                transaction_error=error or "bundle_compile_failure",
                rejection_gate="bundle_compile_gate",
            )
            audit.rejected_gate = report.rejection_gate
            audit.rejection_reason = report.transaction_error
            return ir, report, audit, model_calls
        candidate_ir, report = self._apply_patch_transaction(
            ir, patch, [gap], catalog, repair_unit=active_unit,
        )
        report.protocol = CANDIDATE_CHOICE_PROTOCOL
        audit.rejected_gate = report.rejection_gate
        audit.rejection_reason = report.transaction_error
        if report.semantic_effect:
            audit.coverage_before = dict(report.semantic_effect.coverage_before)
            audit.coverage_after = dict(report.semantic_effect.coverage_after)
            audit.closed_requirement_ids = list(report.semantic_effect.closed_requirement_ids)
        if report.committed:
            self._cache_put(cache_key, {"candidate_id": selected.candidate_id})
            self._record_llm_success()
            return candidate_ir, report, audit, model_calls
        return ir, report, audit, model_calls

    def _execute_candidate_bundle_v3(self, ir, gap, active_unit, catalog,
                                     security_digest, trace):
        """Close deterministic downstream gaps after at most one model choice."""
        baseline = ir
        current, report, audit, model_calls = self._execute_candidate_choice_v3(
            ir, gap, RepairUnit.from_gaps([gap])[0], catalog, security_digest, trace,
        )
        reports = [report] if report is not None else []
        if report is None or not report.committed:
            return current, report, audit, model_calls

        allowed_requirements = self._downstream_requirement_ids(
            current, gap.requirement_id, set(active_unit.requirement_ids),
        )
        handled = {gap.gap_id}
        while True:
            remaining = [
                item for item in self.gap_analyzer.analyze(current, catalog)
                if item.requirement_id in allowed_requirements
                and item.gap_id not in handled
                and item.readiness == "ready"
                and item.resolution_mode == "llm"
            ]
            if not remaining:
                break
            next_gap = remaining[0]
            handled.add(next_gap.gap_id)
            menu = self.semantic_candidate_compiler.compile(current, next_gap, catalog)
            if menu.dispatch != "local_compile":
                trace.append(TraceEvent(
                    "candidate_bundle",
                    f"stopped deterministic closure at {next_gap.gap_id}: {menu.dispatch}",
                ))
                break
            patch, error = self.semantic_candidate_compiler.compile_patch(
                menu.candidates[0], next_gap, current,
            )
            if patch is None:
                reports.append(PatchReport(
                    protocol=CANDIDATE_CHOICE_PROTOCOL,
                    base_ir_digest=request_ir_digest(current),
                    transaction_error=error or "bundle_compile_failure",
                    rejection_gate="bundle_compile_gate",
                ))
                trace.append(TraceEvent(
                    "candidate_bundle", f"rolled back at compile gap {next_gap.gap_id}",
                ))
                return baseline, self._combine_patch_reports(reports), audit, model_calls
            candidate, step_report = self._apply_patch_transaction(
                current, patch, [next_gap], catalog,
                repair_unit=RepairUnit.from_gaps([next_gap])[0],
            )
            step_report.protocol = CANDIDATE_CHOICE_PROTOCOL
            reports.append(step_report)
            if not step_report.committed:
                trace.append(TraceEvent(
                    "candidate_bundle", f"rolled back at rejected gap {next_gap.gap_id}",
                ))
                return baseline, self._combine_patch_reports(reports), audit, model_calls
            current = candidate
            trace.append(TraceEvent(
                "candidate_bundle", f"committed deterministic gap {next_gap.gap_id}",
            ))
        return current, self._combine_patch_reports(reports), audit, model_calls

    @staticmethod
    def _downstream_requirement_ids(ir, root_id: str,
                                    unit_requirement_ids: set[str]) -> set[str]:
        """Authorize the selected root and its descendants, never sibling roots."""
        allowed = {root_id} if root_id else set()
        changed = True
        while changed:
            changed = False
            for requirement in ir.requirements:
                if requirement.requirement_id not in unit_requirement_ids:
                    continue
                if requirement.requirement_id in allowed:
                    continue
                if any(item in allowed for item in requirement.dependencies):
                    allowed.add(requirement.requirement_id)
                    changed = True
        return allowed

    @staticmethod
    def _combine_patch_reports(reports: list[PatchReport]) -> PatchReport | None:
        if not reports:
            return None
        combined = PatchReport(
            protocol=CANDIDATE_CHOICE_PROTOCOL,
            base_ir_digest=reports[0].base_ir_digest,
            committed=all(item.committed for item in reports),
        )
        for item in reports:
            combined.accepted.extend(item.accepted)
            combined.rejected.extend(item.rejected)
            combined.closed_gap_ids.extend(item.closed_gap_ids)
            combined.concrete_gain_count += item.concrete_gain_count
            combined.clarification_gain_count += item.clarification_gain_count
            if item.transaction_error:
                combined.transaction_error = item.transaction_error
                combined.rejection_gate = item.rejection_gate
            if item.semantic_effect is not None:
                combined.semantic_effect = item.semantic_effect
        combined.closed_gap_ids = list(dict.fromkeys(combined.closed_gap_ids))
        return combined

    def _candidate_choice_cache_key(self, menu: CandidateMenu, catalog_version: str,
                                    security_digest: str) -> tuple[str, ...]:
        provider = getattr(self.llm_extractor, "provider", None)
        model = str(getattr(provider, "model", ""))
        return (
            "candidate-choice-v3", menu.source_clause_digest,
            menu.target_contract_digest, menu.base_lineage_digest,
            menu.candidate_menu_digest, menu.candidate_compiler_version,
            menu.expression_signature_version, CANDIDATE_CHOICE_PROTOCOL,
            model, catalog_version, security_digest,
        )

    def resume_clarification(self, previous: EngineResult, answers: Mapping[str, str], *,
                             security_context: SecurityContext | None = None,
                             context_snapshot: TurnContextSnapshot | None = None) -> EngineResult:
        """Recompile after bounded user clarification; never patch the old IR.

        The resume contract ties answers to the exact source-demand digest and
        catalog version that produced the question.  Each accepted answer is
        persisted only in the dedicated answer namespace of the *new*
        compilation snapshot.
        """
        plan = previous.clarification_plan
        if plan is None or plan.resume_contract is None:
            raise ValueError("previous result has no resumable clarification plan")
        contract = plan.resume_contract
        snapshot = CompilationSnapshot.capture(previous.understanding)
        if snapshot.source_demand_digest != contract.source_demand_digest:
            raise ValueError("clarification source-demand snapshot is stale")
        if snapshot.catalog_version != contract.catalog_version:
            raise ValueError("clarification catalog snapshot is stale")
        security = security_context or SecurityContext.local_development()
        current_catalog = self.catalog_authorizer.authorize(
            self.catalog_provider.snapshot(), security,
        )
        if current_catalog.version != contract.catalog_version:
            raise ValueError("catalog version changed; request fresh clarification")

        normalized = self.clarification_planner.validate_answers(plan, answers)
        resumed_query = self.clarification_planner.resume_query(
            previous.understanding.query.raw, plan, normalized,
        )
        result = self.analyze(
            resumed_query, security_context=security_context,
            context_snapshot=context_snapshot,
        )
        self.clarification_planner.apply_answers(result.understanding, plan, normalized)
        result.compilation_snapshot = CompilationSnapshot.capture(result.understanding)
        result.trace.append(TraceEvent(
            "clarification_resume",
            f"recompiled after {len(normalized)} approved clarification answer(s)",
        ))
        return result

    def _apply_patch_transaction(self, ir, patch, gaps, catalog, *,
                                 repair_unit: RepairUnit | None = None):
        """Commit an effective Patch only through the semantic acceptance gate."""
        snapshot = CompilationSnapshot.capture(ir)
        if repair_unit is None:
            units = RepairUnit.from_gaps(gaps)
            if len(units) != 1:
                report = PatchReport(
                    base_ir_digest=patch.base_ir_digest,
                    transaction_error="repair_unit_not_explicit",
                    rejection_gate="repair_unit_gate",
                )
                report.semantic_effect = self._semantic_effect_rejection(
                    snapshot, report.transaction_error,
                )
                return ir, report
            unit = units[0]
        else:
            unit = repair_unit
            if {item.gap_id for item in unit.gaps} != {item.gap_id for item in gaps}:
                report = PatchReport(
                    base_ir_digest=patch.base_ir_digest,
                    transaction_error="repair_unit_gap_mismatch",
                    rejection_gate="repair_unit_gate",
                )
                report.semantic_effect = self._semantic_effect_rejection(
                    snapshot, report.transaction_error,
                )
                return ir, report
        profile = "clarification" if all(
            set(item.allowed_operations) <= {"add_ambiguity"} for item in unit.gaps
        ) else "concrete"
        contract = RepairContract.for_unit(snapshot, unit, profile, ir, catalog)
        # A candidate must stay inside its RepairContract *before* any Patch
        # applier is allowed to construct an in-memory IR.  This makes the
        # whitelist a true authorization boundary rather than a post-hoc
        # commit filter.
        violation = contract_violation(contract, patch, ir)
        if violation:
            report = PatchReport(
                base_ir_digest=patch.base_ir_digest, transaction_error=violation,
                rejection_gate="repair_contract_gate",
            )
            report.semantic_effect = self._semantic_effect_rejection(snapshot, violation)
            return ir, report
        candidate_ir, report = apply_semantic_patch(
            ir, patch, list(unit.gaps), catalog, atomic_unit=len(unit.gaps) > 1,
        )
        if report.transaction_error:
            report.rejection_gate = report.rejection_gate or "patch_protocol_gate"
            report.semantic_effect = self._semantic_effect_rejection(snapshot, report.transaction_error)
            return ir, report
        if report.accepted_count == 0:
            report.transaction_error = "no_accepted_operations" if patch.operations else ""
            report.rejection_gate = "patch_applier_gate" if patch.operations else "semantic_gain_gate"
            report.semantic_effect = self._semantic_effect_rejection(
                snapshot, report.transaction_error or "no_gain",
            )
            return ir, report
        if report.closed_gap_count == 0:
            report.transaction_error = "no_target_gap_closed"
            report.rejection_gate = "gap_closure_gate"
            report.semantic_effect = self._semantic_effect_rejection(snapshot, report.transaction_error)
            return ir, report

        def refresh(candidate):
            if self.config.enable_operator_registry:
                self.operator_enricher.refresh(candidate)

        if self.config.enable_semantic_repair:
            decision = self.semantic_acceptance_gate.evaluate(
                ir, candidate_ir, report, contract, catalog, refresh=refresh,
            )
            report.semantic_effect = decision.effect
            if not decision.accepted:
                report.transaction_error = decision.effect.commit_or_reject_reason
                report.rejection_gate = "semantic_acceptance_gate"
                return ir, report
            if request_ir_digest(ir) != snapshot.ir_digest:
                report.transaction_error = "snapshot_changed_before_commit"
                report.rejection_gate = "snapshot_commit_gate"
                report.semantic_effect.commit_or_reject_reason = report.transaction_error
                return ir, report
            report.committed = True
            return decision.candidate, report

        refresh(candidate_ir)
        report.committed = True
        return candidate_ir, report

    @staticmethod
    def _semantic_effect_rejection(snapshot, reason):
        from .models import SemanticEffectReport
        return SemanticEffectReport(
            base_digest=snapshot.ir_digest, candidate_digest=snapshot.ir_digest,
            commit_or_reject_reason=reason,
        )

    def _should_call_llm(self, ir, gaps=None) -> bool:
        if not self.config.enable_llm or self.llm_extractor is None:
            return False
        if ir.unsatisfiable:
            return False
        if self.config.enable_requirement_ir_v2 and gaps:
            # Binding ambiguities need a user/catalog decision; asking an LLM
            # cannot authorize that binding and only burns tokens.
            return any(
                operation != "add_ambiguity"
                for gap in gaps for operation in gap.allowed_operations
            )
        if (self.config.enable_requirement_ir_v2 and self.config.enable_patch_v1
                and not self.config.llm_on_complex):
            return False
        if ir.unresolved or not any(
            item.status == "resolved" for item in ir.source_candidates
        ):
            return True
        if any(item.status != "resolved" for item in ir.projections):
            return True
        if any(item.status != "resolved" for item in ir.references):
            return True
        if any(
            item.calculation_type == "volatility"
            and not item.parameters.get("input_field")
            for item in ir.calculations
        ):
            return True
        if not self.config.llm_on_complex:
            return False
        goal_types = {item.goal_type for item in ir.goals}
        return bool(ir.references or ir.calculations or len(goal_types) >= 4)

    @staticmethod
    def _is_provider_failure(error: str) -> bool:
        return bool(error and any(token in error for token in (
            "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout",
            "HTTPError", "Ollama unavailable", "deadline exceeded",
        )))

    @staticmethod
    def _provider_diagnostic_code(error: str) -> str:
        lowered = error.lower()
        return "provider_timeout" if "timeout" in lowered or "deadline exceeded" in lowered else "provider_error"

    @staticmethod
    def _llm_effect_dimension(patch_report, model_calls: int, diagnostics) -> dict[str, object]:
        if any(item.code.startswith("provider_") for item in diagnostics):
            return {"status": "provider_failed", "score": 0.0}
        if patch_report is not None and patch_report.committed:
            gain = patch_report.concrete_gain_count + patch_report.clarification_gain_count
            return {
                "status": "committed" if gain else "no_gain",
                "score": 1.0 if patch_report.concrete_gain_count else 0.6 if gain else 0.25,
                "concrete_gain": patch_report.concrete_gain_count,
                "clarification_gain": patch_report.clarification_gain_count,
            }
        if patch_report is not None and patch_report.transaction_error:
            return {"status": "rejected", "score": 0.0, "reason": patch_report.transaction_error}
        if model_calls:
            return {"status": "no_gain", "score": 0.25}
        return {"status": "not_attempted", "score": 0.5}

    @staticmethod
    def _existing_ir_summary(ir) -> dict:
        return {
            "schema_hypotheses": [item.to_dict() for item in ir.schema_hypotheses],
            "sources": [item.identifier for item in ir.source_candidates],
            "projections": [
                item.canonical_id or item.raw_name for item in ir.projections
            ],
            "filters": ir.filters.to_dict() if ir.filters else None,
            "temporal": [item.to_dict() for item in ir.temporal],
            "sampling": [item.to_dict() for item in ir.sampling_policies],
            "events": [
                {"event_id": item.event_id, "output_name": item.output_name,
                 "metric_id": item.derived_metric.metric_id}
                for item in ir.events
            ],
            "set_operations": [item.to_dict() for item in ir.set_operations],
            "calculations": [
                {"type": item.calculation_type, "parameters": item.parameters}
                for item in ir.calculations
            ],
            "aggregates": [item.to_dict() for item in ir.aggregates],
            "comparisons": [item.to_dict() for item in ir.comparisons],
            "requirements": [item.to_dict() for item in ir.requirements],
            "coverage": [item.to_dict() for item in ir.coverage],
            "unresolved": [
                {"code": item.code, "message": item.message}
                for item in ir.unresolved
            ],
        }

    def _catalog_payload(self, catalog: CatalogSnapshot, ir) -> list[dict]:
        preferred = {
            item.identifier for item in ir.source_candidates if catalog.source(item.identifier)
        }
        sources = [item for item in catalog.sources if not preferred or item.source_id in preferred]
        remaining = max(1, self.config.max_llm_catalog_fields)
        payload = []
        for source in sources:
            item = asdict(source)
            item["fields"] = item["fields"][:remaining]
            remaining -= len(item["fields"])
            payload.append(item)
            if remaining <= 0:
                break
        return payload

    @staticmethod
    def _timed(trace: list[TraceEvent], stage: str, detail: str, fn):
        started = time.perf_counter()
        value = fn()
        trace.append(TraceEvent(stage, detail, round((time.perf_counter() - started) * 1000, 3)))
        return value

    def _cache_key(self, query: str, version: str,
                   security_digest: str) -> tuple[str, ...]:
        model_identity = "no-llm"
        if self.llm_extractor is not None:
            model_identity = self.llm_extractor.cache_identity()
        return query, version, security_digest, model_identity

    def _cache_get(self, key: tuple[str, ...]) -> dict | None:
        with self._lock:
            value = self._cache.get(key)
            return copy.deepcopy(value) if value is not None else None

    def _cache_put(self, key: tuple[str, ...], value: dict) -> None:
        with self._lock:
            if key not in self._cache:
                self._cache_order.append(key)
            self._cache[key] = copy.deepcopy(value)
            while len(self._cache_order) > max(1, self.config.cache_size):
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)

    def _breaker_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._breaker_until

    def _record_llm_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._breaker_until = 0.0

    def _record_llm_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= max(1, self.config.breaker_failures):
                self._breaker_until = time.monotonic() + self.config.breaker_cooldown_seconds
