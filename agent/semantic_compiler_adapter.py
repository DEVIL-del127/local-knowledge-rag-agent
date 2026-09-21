from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import types
from dataclasses import fields, is_dataclass
from typing import Union, get_args, get_origin, get_type_hints
from pathlib import Path
from typing import Any

from agent.retrieval_models import SemanticCompileResult, SemanticDecision
from agent.nlu_profile import NluExecutionProfile
from agent.semantic_policy import SemanticAdmissionPolicy
from agent.source_span_audit import audit_source_spans
from core.catalog_provider import CatalogContractSnapshot, MainProjectCatalogReader

logger = logging.getLogger(__name__)

from agent.runtime.errors import ContextSnapshotMismatch


class SemanticCompilerAdapter:
    """Fail-silent shadow adapter. It cannot mutate or execute a SkillCall."""

    def __init__(
        self, *, search_backend: Any, nlu_root: str | Path | None = None,
        generation_registry: Any | None = None, enable_llm: bool = True,
        require_active_generation: bool = False,
        contract_store: Any | None = None,
    ) -> None:
        self.search_backend = search_backend
        project_root = Path(__file__).resolve().parent.parent
        self.nlu_root = Path(nlu_root or project_root / "kb-agent").resolve()
        if nlu_root is None and not (self.nlu_root / "nlu_v2" / "__init__.py").is_file():
            from importlib.util import find_spec
            installed = find_spec("nlu_v2")
            if installed is not None and installed.origin:
                self.nlu_root = Path(installed.origin).resolve().parent.parent
        self._engine = None
        self._engine_generation = ""
        self._engine_catalog_digest = ""
        self.generation_registry = generation_registry
        self.enable_llm = bool(enable_llm)
        self.require_active_generation = bool(require_active_generation)
        self.contract_store = contract_store
        self._engine_tree_digest = _tree_digest(self.nlu_root / "nlu_v2")
        self._profile = NluExecutionProfile.latest(enable_llm=self.enable_llm)
        self._engine_profile = self._profile.profile_id
        self._engine_source_revision = _source_revision(project_root)
        self._engine_results: dict[str, Any] = {}

    def compile(
        self, query: str, *, generation_snapshot: Any | None = None,
    ) -> SemanticCompileResult:
        started = time.perf_counter()
        try:
            engine = self._get_engine(generation_snapshot)
            result = engine.analyze(query)
            ir = result.understanding
            payload = result.to_dict()
            serialized = ir.to_dict()
            digest = hashlib.sha256(
                json.dumps(serialized, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            diagnostics = [item.code for item in ir.diagnostics]
            diagnostics.extend(item.code for item in ir.unresolved)
            source_ids = [
                item.identifier for item in ir.source_candidates
                if getattr(item, "status", "") == "resolved"
                and getattr(item, "identifier", "")
            ]
            questions = _clarification_questions(result)
            logical_plan = _authoritative_plan(result, ir)
            decision = _admission(ir, result, logical_plan, questions, query)
            executable = bool(
                getattr(result.validation, "executable", False)
                and logical_plan.get("status") == "ready"
            )
            validation_status = str(getattr(result.validation, "status", ""))
            contract_reports = _contract_reports(payload)
            trace = list(payload.get("trace") or [])
            span_audit = audit_source_spans(
                query, serialized.get("requirements") or [],
                serialized.get("requirements") or [], logical_plan,
                dict(getattr(ir, "literature_contract", {}) or {}),
            )
            if not span_audit.complete:
                decision = SemanticDecision.BLOCKED
                executable = False
                validation_status = "source_span_incomplete"
                diagnostics.append("source_span_unassigned_action")
            if getattr(ir, "literature_contract", None):
                contract_reports["literature_semantics"] = dict(ir.literature_contract)
                contract_reports["accepted_literature_ir"] = {
                    "digest": str(getattr(ir, "accepted_ir_digest", "")),
                    "source_binding": dict(getattr(ir, "source_binding", {}) or {}),
                    "plan_digest": _dict_digest(logical_plan),
                }
                if decision == SemanticDecision.PASS_THROUGH and executable:
                    contract_reports["literature_contract_bridge"] = _literature_compatibility_report(
                        ir, validation_status
                    )
                    trace.append({
                        "stage": "literature_contract_bridge",
                        "detail": "derived read-only view of engine-owned AcceptedIR",
                        "duration_ms": 0.0,
                    })
            contract_reports["source_span_audit"] = span_audit.to_dict()
            config_digest = _config_digest(engine.config)
            query_payload = serialized.get("query") or {}
            self._engine_results[digest] = result
            return SemanticCompileResult(
                decision=decision,
                effective_query=query,
                catalog_version=ir.catalog_version,
                ir_digest=digest,
                diagnostics=list(dict.fromkeys(diagnostics))[:32],
                source_ids=list(dict.fromkeys(source_ids)),
                model_calls=int(getattr(result, "model_calls", 0) or 0),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                clarification_questions=questions,
                validation_status=validation_status,
                executable=executable,
                request_ir=serialized,
                logical_plan=logical_plan,
                contract_reports=contract_reports,
                engine_revision="request-ir-2.5",
                engine_tree_digest=self._engine_tree_digest,
                engine_profile=self._engine_profile,
                engine_config_digest=config_digest,
                logical_plan_digest=_dict_digest(logical_plan),
                shadow_only=False,
                engine_source_revision=self._engine_source_revision,
                engine_profile_digest=self._profile.digest(),
                request_ir_schema_version=str(serialized.get("schema_version", "")),
                catalog_digest=self._engine_catalog_digest,
                security_scope_digest=str(query_payload.get("security_scope_digest", "")),
                admission=decision.value,
                validation=(payload.get("validation") or {}),
                model_audit=payload.get("llm_audit"),
                trace=trace,
                resume_snapshot={
                    "understanding": serialized,
                    "clarification_plan": payload.get("clarification_plan"),
                } if questions else {},
            )
        except Exception as exc:
            logger.warning("NLU V2 shadow compile failed; legacy retrieval is unchanged: %s", exc)
            return SemanticCompileResult(
                decision=SemanticDecision.ERROR,
                effective_query=query,
                diagnostics=[f"{type(exc).__name__}: {exc}"],
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )

    def compile_pinned(self, query: str, generation_snapshot: Any) -> SemanticCompileResult:
        """Compile against the exact generation already pinned for this request."""
        return self.compile(query, generation_snapshot=generation_snapshot)

    def resume(
        self,
        query: str,
        answers: dict[str, str],
        *,
        expected_ir_digest: str = "",
        expected_catalog_version: str = "",
        previous_snapshot: dict[str, Any] | None = None,
        generation_snapshot: Any | None = None,
    ) -> SemanticCompileResult:
        """Resume from the exact retained or persisted snapshot; never re-analyze it first."""
        started = time.perf_counter()
        engine = self._get_engine(generation_snapshot)
        previous = self._engine_results.get(expected_ir_digest)
        if previous is None and previous_snapshot:
            previous = _rehydrate_resume_snapshot(previous_snapshot)
        if previous is None:
            raise ContextSnapshotMismatch("clarification snapshot is unavailable; ask a fresh question")
        previous_digest = _ir_digest(previous.understanding)
        if previous_digest != expected_ir_digest:
            raise ContextSnapshotMismatch("clarification IR snapshot changed; ask a fresh question")
        if (
            expected_catalog_version
            and previous.understanding.catalog_version != expected_catalog_version
        ):
            raise ContextSnapshotMismatch("catalog version changed; ask a fresh question")
        result = engine.resume_clarification(previous, answers)
        ir = result.understanding
        questions = _clarification_questions(result)
        payload = result.to_dict()
        logical_plan = _authoritative_plan(result, ir)
        self._engine_results[_ir_digest(ir)] = result
        decision = _admission(ir, result, logical_plan, questions, query)
        diagnostics = list(dict.fromkeys(
            [item.code for item in ir.diagnostics]
            + [item.code for item in ir.unresolved]
        ))[:32]
        return SemanticCompileResult(
            decision=decision,
            effective_query=ir.query.normalized,
            catalog_version=ir.catalog_version,
            ir_digest=_ir_digest(ir),
            diagnostics=diagnostics,
            source_ids=[
                item.identifier for item in ir.source_candidates
                if getattr(item, "status", "") == "resolved"
                and getattr(item, "identifier", "")
            ],
            model_calls=int(getattr(result, "model_calls", 0) or 0),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            clarification_questions=questions,
            validation_status=str(getattr(result.validation, "status", "")),
            executable=bool(
                getattr(result.validation, "executable", False)
                and logical_plan.get("status") == "ready"
            ),
            request_ir=ir.to_dict(),
            logical_plan=logical_plan,
            contract_reports=_contract_reports(payload),
            engine_revision="request-ir-2.5",
            engine_tree_digest=self._engine_tree_digest,
            engine_profile=self._engine_profile,
            engine_config_digest=_config_digest(engine.config),
            logical_plan_digest=_dict_digest(logical_plan),
            shadow_only=False, engine_source_revision=self._engine_source_revision,
            engine_profile_digest=self._profile.digest(),
            request_ir_schema_version=str(ir.schema_version),
            catalog_digest=self._engine_catalog_digest,
            security_scope_digest=str(getattr(ir.query, "security_scope_digest", "")),
            admission=decision.value, validation=(payload.get("validation") or {}),
            model_audit=payload.get("llm_audit"), trace=list(payload.get("trace") or []),
            resume_snapshot={
                "understanding": ir.to_dict(),
                "clarification_plan": payload.get("clarification_plan"),
            } if questions else {},
        )

    def compile_with_context(
        self, query: str, previous_semantic_result: dict[str, Any], *, previous_turn_id: str,
        generation_snapshot: Any | None = None,
    ) -> SemanticCompileResult:
        """Compile a follow-up against a verified, persisted RequestIR snapshot."""
        started = time.perf_counter()
        engine = self._get_engine(generation_snapshot)
        stored_ir = dict(previous_semantic_result.get("request_ir") or {})
        expected_digest = str(previous_semantic_result.get("ir_digest") or "")
        previous = self._engine_results.get(expected_digest)
        if previous is None and stored_ir:
            previous = types.SimpleNamespace(
                understanding=_rehydrate_understanding_ir(stored_ir)
            )
        if previous is None:
            raise ContextSnapshotMismatch(
                "persisted semantic object is unavailable; compile without stale context"
            )
        actual_digest = _ir_digest(previous.understanding)
        if expected_digest and actual_digest != expected_digest:
            raise ContextSnapshotMismatch("persisted RequestIR digest mismatch")
        expected_catalog = str(previous_semantic_result.get("catalog_version") or "")
        if expected_catalog and previous.understanding.catalog_version != expected_catalog:
            raise ContextSnapshotMismatch("persisted RequestIR catalog mismatch")
        from nlu_v2.context import context_snapshot_digest
        from nlu_v2.models import TurnContextSnapshot
        snapshot = TurnContextSnapshot(
            previous_turn_id=previous_turn_id,
            accepted_semantic_ir=previous.understanding,
            context_digest=context_snapshot_digest(previous.understanding),
            schema_version=previous.understanding.schema_version,
            catalog_version=previous.understanding.catalog_version,
        )
        result = engine.analyze(query, context_snapshot=snapshot)
        ir = result.understanding
        payload = result.to_dict()
        questions = _clarification_questions(result)
        logical_plan = _authoritative_plan(result, ir)
        decision = _admission(ir, result, logical_plan, questions, query)
        executable = bool(getattr(result.validation, "executable", False)
                          and logical_plan.get("status") == "ready")
        validation_status = str(getattr(result.validation, "status", ""))
        diagnostics = list(dict.fromkeys(
            [item.code for item in ir.diagnostics] + [item.code for item in ir.unresolved]
        ))[:32]
        contract_reports = _contract_reports(payload)
        trace = list(payload.get("trace") or [])
        serialized = ir.to_dict()
        span_audit = audit_source_spans(
            query, serialized.get("requirements") or [],
            serialized.get("requirements") or [], logical_plan,
            dict(getattr(ir, "literature_contract", {}) or {}),
        )
        if not span_audit.complete:
            decision = SemanticDecision.BLOCKED
            executable = False
            validation_status = "source_span_incomplete"
            diagnostics.append("source_span_unassigned_action")
        if getattr(ir, "literature_contract", None):
            contract_reports["literature_semantics"] = dict(ir.literature_contract)
            contract_reports["accepted_literature_ir"] = {
                "digest": str(getattr(ir, "accepted_ir_digest", "")),
                "source_binding": dict(getattr(ir, "source_binding", {}) or {}),
                "plan_digest": _dict_digest(logical_plan),
            }
            if decision == SemanticDecision.PASS_THROUGH and executable:
                contract_reports["literature_contract_bridge"] = _literature_compatibility_report(
                    ir, validation_status, contextual=True
                )
                trace.append({
                    "stage": "literature_contract_bridge",
                    "detail": "derived read-only view of contextual engine-owned AcceptedIR",
                    "duration_ms": 0.0,
                })
        contract_reports["source_span_audit"] = span_audit.to_dict()
        # Every accepted turn can become the immutable base for the next
        # follow-up.  Omitting contextual results here made turn 2 work but
        # forced turn 3 to fall back to a context-free compile.
        self._engine_results[_ir_digest(ir)] = result
        return SemanticCompileResult(
            decision=decision, effective_query=ir.query.normalized,
            catalog_version=ir.catalog_version, ir_digest=_ir_digest(ir),
            diagnostics=diagnostics,
            source_ids=[item.identifier for item in ir.source_candidates
                        if getattr(item, "status", "") == "resolved"],
            model_calls=int(getattr(result, "model_calls", 0) or 0),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            clarification_questions=questions,
            validation_status=validation_status,
            executable=executable,
            request_ir=ir.to_dict(), logical_plan=logical_plan,
            contract_reports=contract_reports, engine_revision="request-ir-2.5",
            engine_tree_digest=self._engine_tree_digest, engine_profile=self._engine_profile,
            engine_config_digest=_config_digest(engine.config),
            logical_plan_digest=_dict_digest(logical_plan),
            shadow_only=False, engine_source_revision=self._engine_source_revision,
            engine_profile_digest=self._profile.digest(),
            request_ir_schema_version=str(ir.schema_version),
            catalog_digest=self._engine_catalog_digest,
            security_scope_digest=str(getattr(ir.query, "security_scope_digest", "")),
            admission=decision.value, validation=(payload.get("validation") or {}),
            model_audit=payload.get("llm_audit"), trace=trace,
            resume_snapshot={
                "understanding": ir.to_dict(),
                "clarification_plan": payload.get("clarification_plan"),
            } if questions else {},
        )

    def compile_with_pinned_context(
        self,
        query: str,
        previous_semantic_result: dict[str, Any],
        generation_snapshot: Any,
        *,
        previous_turn_id: str,
    ) -> SemanticCompileResult:
        """Compile a contextual turn without consulting the mutable ACTIVE pointer."""
        return self.compile_with_context(
            query,
            previous_semantic_result,
            previous_turn_id=previous_turn_id,
            generation_snapshot=generation_snapshot,
        )

    def _get_engine(self, generation_snapshot: Any | None = None):
        self._verify_loaded_source()
        active = generation_snapshot
        if active is None and self.generation_registry is not None:
            active, _ = self.generation_registry.get_active()
        if self.require_active_generation and active is None:
            raise RuntimeError("no ACTIVE knowledge-base generation")
        generation = active.generation_id if active else ""
        engine_identity = (
            str(getattr(active, "snapshot_digest", ""))
            or f"generation:{generation}:record:{getattr(active, 'record_revision', '')}"
        )
        if self._engine is not None and engine_identity == self._engine_generation:
            if getattr(generation_snapshot, "catalog_ref", None) is not None:
                from core.contract_store import ContractRef
                if self.contract_store is None:
                    raise RuntimeError("catalog object store unavailable")
                self.contract_store.resolve(ContractRef(**generation_snapshot.catalog_ref))
            return self._engine
        if not (self.nlu_root / "nlu_v2" / "__init__.py").is_file():
            raise RuntimeError(f"NLU V2 package not found: {self.nlu_root}")
        root_text = str(self.nlu_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        from nlu_v2.catalog import CatalogSnapshot, DataSourceSpec, FieldSpec
        from nlu_v2.engine import EngineConfig, QueryUnderstandingEngine
        self._verify_loaded_source()

        llm_extractor = None
        if self.enable_llm:
            from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider
            from agent.budgeted_nlu_provider import BudgetedNluProvider
            llm_extractor = BoundedLLMExtractor(BudgetedNluProvider(
                OllamaOpenAIProvider(), require_operation=self.require_active_generation,
                session_limit=int(os.environ.get("AGENT_SESSION_TOKEN_BUDGET", "100000")),
                daily_limit=int(os.environ.get("AGENT_DAILY_TOKEN_BUDGET", "500000")),
            ))

        catalog_ref = getattr(generation_snapshot, "catalog_ref", None)
        if catalog_ref is not None:
            from core.contract_store import ContractRef
            if self.contract_store is None:
                raise RuntimeError("catalog object store unavailable")
            ref = ContractRef(**catalog_ref)
            contract = CatalogContractSnapshot.from_identity_payload(
                self.contract_store.resolve(ref), version=generation_snapshot.catalog_version,
            )
            if contract.digest() != generation_snapshot.catalog_digest:
                raise RuntimeError("pinned catalog object mismatch")
        else:
            reader = MainProjectCatalogReader(
                self.search_backend.es_manager.es,
                active.es_physical_index if active else self.search_backend.es_manager.index_name,
            )
            contract = reader.snapshot()
        self._engine_catalog_digest = contract.digest()
        if (
            generation_snapshot is not None
            and str(getattr(generation_snapshot, "catalog_schema_version", ""))
            == "catalog-contract-v2"
        ):
            expected_catalog_digest = str(
                getattr(generation_snapshot, "catalog_digest", "")
            )
            if expected_catalog_digest != self._engine_catalog_digest:
                raise RuntimeError("pinned catalog digest mismatch")

        class BoundCatalogProvider:
            def snapshot(inner_self):
                return _to_nlu_catalog(contract, CatalogSnapshot, DataSourceSpec, FieldSpec)

        self._engine = QueryUnderstandingEngine(
            catalog_provider=BoundCatalogProvider(),
            llm_extractor=llm_extractor,
            config=EngineConfig(**self._profile.engine_kwargs()),
        )
        self._engine_generation = engine_identity
        self._engine_results.clear()
        return self._engine

    def _verify_loaded_source(self) -> None:
        expected = (self.nlu_root / "nlu_v2").resolve()
        for name, module in tuple(sys.modules.items()):
            if name != "nlu_v2" and not name.startswith("nlu_v2."):
                continue
            source = getattr(module, "__file__", None)
            if source is None or expected not in Path(source).resolve().parents:
                raise RuntimeError("NLU source locator mismatch")
        if _tree_digest(expected) != self._engine_tree_digest:
            raise RuntimeError("NLU source tree changed after initialization")


def _to_nlu_catalog(contract: CatalogContractSnapshot, snapshot_cls, source_cls, field_cls):
    sources = []
    for source in contract.sources:
        fields = [
            field_cls(
                field_id=item.canonical_id,
                aliases=list(item.aliases),
                data_type=item.data_type,
                allowed_operators=list(item.allowed_operators),
            )
            for item in source.fields
        ]
        sources.append(source_cls(
            source_id=source.source_id,
            aliases=list(source.aliases),
            kind="elasticsearch",
            capabilities=list(source.capabilities),
            fields=fields,
            read_only=True,
            version=source.version,
        ))
    return snapshot_cls(contract.version, sources)


def _ir_digest(ir: Any) -> str:
    return hashlib.sha256(
        json.dumps(ir.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _clarification_questions(result: Any) -> list[dict[str, Any]]:
    plan = getattr(result, "clarification_plan", None)
    questions = getattr(plan, "questions", None) or []
    return [
        {
            "question_id": item.question_id,
            "kind": item.kind,
            "prompt": item.prompt,
            "expected_answer_type": item.expected_answer_type,
            "candidate_ids": list(item.candidate_ids),
            "candidates": list(item.candidates),
        }
        for item in questions[:3]
    ]


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        entries.append(f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _source_revision(project_root: Path) -> str:
    """Resolve the checked-out git revision without invoking a shell."""
    git_dir = project_root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            return (git_dir / head[5:]).read_text(encoding="utf-8").strip()
        return head
    except OSError:
        return "unavailable"


def _rehydrate_understanding_ir(payload: dict[str, Any]) -> Any:
    """Rebuild the typed RequestIR from its canonical persisted dictionary."""
    try:
        import nlu_v2.models as model_types
        return _coerce_typed_value(model_types.UnderstandingIR, payload, model_types)
    except ContextSnapshotMismatch:
        raise
    except Exception as exc:
        raise ContextSnapshotMismatch(
            f"persisted RequestIR cannot be rehydrated: {type(exc).__name__}"
        ) from exc


def _coerce_typed_value(annotation: Any, value: Any, model_types: Any) -> Any:
    if annotation is Any or annotation is object:
        return value
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        failures = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _coerce_typed_value(candidate, value, model_types)
            except Exception as exc:
                failures.append(exc)
        raise ContextSnapshotMismatch("persisted union field is incompatible") from (
            failures[-1] if failures else None
        )
    if origin is list:
        return [_coerce_typed_value(arguments[0], item, model_types) for item in (value or [])]
    if origin is tuple:
        element_type = arguments[0] if arguments else Any
        return tuple(_coerce_typed_value(element_type, item, model_types) for item in (value or []))
    if origin is dict:
        key_type, value_type = arguments if len(arguments) == 2 else (Any, Any)
        return {
            _coerce_typed_value(key_type, key, model_types):
            _coerce_typed_value(value_type, item, model_types)
            for key, item in dict(value or {}).items()
        }
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ContextSnapshotMismatch(
                f"persisted {getattr(annotation, '__name__', 'dataclass')} field is not an object"
            )
        hints = get_type_hints(
            annotation, globalns=vars(model_types), localns=vars(model_types)
        )
        kwargs = {
            item.name: _coerce_typed_value(hints.get(item.name, Any), value[item.name], model_types)
            for item in fields(annotation) if item.name in value
        }
        return annotation(**kwargs)
    return value


def _rehydrate_resume_snapshot(snapshot: dict[str, Any]) -> Any:
    """Rehydrate only the immutable surface consumed by resume_clarification."""
    from types import SimpleNamespace
    from nlu_v2.clarification import (
        ClarificationPlan, ClarificationQuestion, ClarificationResumeContract,
        ClarificationState,
    )

    ir_payload = dict(snapshot.get("understanding") or {})
    plan_payload = dict(snapshot.get("clarification_plan") or {})
    if not ir_payload or not plan_payload:
        raise ContextSnapshotMismatch("incomplete persisted clarification snapshot")

    class FrozenPayload:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = dict(payload)

        def to_dict(self) -> dict[str, Any]:
            return dict(self._payload)

    class ResumeUnderstanding(FrozenPayload):
        def __init__(self, payload: dict[str, Any]) -> None:
            super().__init__(payload)
            self.catalog_version = str(payload.get("catalog_version", ""))
            self.source_demands = [FrozenPayload(item) for item in payload.get("source_demands") or []]
            query = dict(payload.get("query") or {})
            self.query = SimpleNamespace(raw=str(query.get("raw", "")))

    state_payload = dict(plan_payload.get("state") or {})
    contract_payload = dict(plan_payload.get("resume_contract") or {})
    plan = ClarificationPlan(
        state=ClarificationState(**state_payload),
        questions=[ClarificationQuestion(**item) for item in plan_payload.get("questions") or []],
        resume_contract=ClarificationResumeContract(**contract_payload) if contract_payload else None,
    )
    return SimpleNamespace(
        understanding=ResumeUnderstanding(ir_payload), clarification_plan=plan,
    )


def _dict_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _config_digest(config: Any) -> str:
    from dataclasses import asdict
    return _dict_digest(asdict(config))


def _has_m15_work(ir: Any) -> bool:
    return any((
        ir.events, ir.ordered_folds, ir.sequences, ir.window_aggregates,
        ir.cumulative_counts, ir.cumulative_durations, ir.set_operations,
    ))


def _authoritative_plan(result: Any, ir: Any) -> dict[str, Any]:
    base = result.logical_plan.to_dict()
    m15 = getattr(result, "m15_logical_plan", None)
    if _has_m15_work(ir):
        return m15.to_dict() if m15 is not None else {
            "schema_version": ir.schema_version,
            "catalog_version": ir.catalog_version,
            "nodes": [], "status": "blocked", "reason": "M1.5 plan unavailable",
        }
    return base


def _literature_nlu_plan(query: str, catalog_version: str) -> dict[str, Any]:
    return {
        "schema_version": "literature-nlu-plan-v2", "catalog_version": catalog_version,
        "nodes": [{"task_id": "literature_semantics_1", "task_type": "retrieve",
                   "depends_on": [], "inputs": {"query": query}, "status": "planned",
                   "side_effect": "none", "binding_status": "unbound"}],
        "status": "ready", "reason": "nlu_v2 literature semantics accepted",
    }


def _admission(ir: Any, result: Any, logical_plan: dict[str, Any],
               questions: list[dict[str, Any]], query: str) -> SemanticDecision:
    if questions:
        return SemanticDecision.CLARIFY
    status = str(logical_plan.get("status", "blocked"))
    if status == "unsupported":
        return SemanticDecision.UNSUPPORTED_ANALYTICS
    if status != "ready" or not bool(getattr(result.validation, "executable", False)):
        return SemanticDecision.BLOCKED
    return SemanticAdmissionPolicy.decide(ir, query)


def _contract_reports(payload: dict[str, Any]) -> dict[str, Any]:
    names = (
        "canonical_requirement_graph", "field_binding_report", "quantity_report",
        "event_readiness_report", "typed_lineage_report", "reference_binding_report",
        "event_closure_report", "compilation_snapshot", "llm_audit", "semantic_effect",
    )
    return {name: payload.get(name) for name in names}


def _literature_compatibility_report(
    ir: Any, validation_status: str, *, contextual: bool = False,
) -> dict[str, Any]:
    """Expose legacy diagnostics without reparsing or mutating AcceptedIR."""
    contract = dict(getattr(ir, "literature_contract", {}) or {})
    task = contract.get("task")
    task_value = getattr(task, "value", task)
    return {
        "task": str(task_value or ""),
        "sense_id": str(contract.get("sense_id") or ""),
        "contextual_topic_inherited": bool(contextual),
        "request_ir_mutated": False,
        "validation_overridden": False,
        "underlying_nlu": {"validation_status": str(validation_status or "")},
        "source": "engine_accepted_ir",
    }
