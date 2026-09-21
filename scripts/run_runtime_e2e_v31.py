from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.agent_models import AgentReply
from agent.agent_service import AgentSettings, PrivateKnowledgeAgent
from agent.agent_skills import SkillRegistry
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.literature_ir import ReferenceKind
from nlu_v2.literature_semantics import analyze_literature
from agent.runtime.facade import RuntimeAgentFacade
from agent.runtime.graph import RuntimeSemanticGraph
from agent.semantic_compiler_adapter import SemanticCompilerAdapter
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from core.retrieval_gateway import RetrievalGateway
from core.contract_store import ContractStore
from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem
from scripts.run_release_acceptance_v3 import CASES


REPORT = ROOT / "kb-agent/docs/reports/2026-09-03_literature-agent-v31-runtime-e2e-deterministic-bridge-v2.json"
STATE = ROOT / "agent_state/release_v31_e2e/runtime.sqlite3"
NORMAL_CATEGORIES = {"enumerate", "locate", "qa", "summarize_compare", "multi_turn"}


class _NoExternalModel:
    settings = SimpleNamespace(model="deterministic-grounded-eval-v1")

    def invoke_text(self, **kwargs):  # pragma: no cover - a release assertion
        raise AssertionError("deterministic E2E must not call an external answer model")


class _DeterministicGroundedAgent(PrivateKnowledgeAgent):
    def answer_from_observations(self, query, *, intent, observations, **kwargs):
        raw = [
            item
            for observation in observations
            for item in list((observation.get("result") or {}).get("evidence") or [])
        ]
        evidence = self._coerce_evidence(raw)
        lines = ["基于当前固定 generation 的检索证据："]
        for index, item in enumerate(evidence, 1):
            snippet = " ".join(item.snippet.split())[:180]
            lines.append(f"{index}. {snippet} [E{index}]")
        return AgentReply(
            answer="\n".join(lines), intent=intent, evidence=evidence,
            executed_skills=[{
                "skill_name": item.get("skill_name", ""),
                "status": item.get("status", "unknown"),
                "arguments": dict(item.get("arguments") or {}),
            } for item in observations],
            semantic_trace=[{
                "runtime": "enforce",
                "synthesis": "deterministic-grounded-eval-v1",
                "external_model": False,
            }],
        )


def _percentiles(values: list[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"sample_count": 0, "p50_ms": None, "p95_ms": None}
    pick = lambda q: ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {
        "sample_count": len(ordered),
        "p50_ms": round(pick(0.50), 3),
        "p95_ms": round(pick(0.95), 3),
    }


def _citation_numbers(answer: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[E(\d+)\]", answer)]


def _make_runtime(system, registry, *, real_understanding_model: bool = False,
                  real_answer_model: bool = False):
    compiler = SemanticCompilerAdapter(
        search_backend=system,
        generation_registry=registry,
        enable_llm=real_understanding_model,
        require_active_generation=True,
        contract_store=ContractStore(
            registry.path.parent / "contracts", store_id="local-contracts-v1",
        ),
    )
    agent_class = PrivateKnowledgeAgent if real_answer_model else _DeterministicGroundedAgent
    answer_client = DeepSeekClient(DeepSeekSettings.from_env()) if real_answer_model else _NoExternalModel()
    legacy = agent_class(
        search_backend=system,
        deepseek_client=answer_client,
        registry=SkillRegistry(),
        settings=AgentSettings(
            state_dir=str(STATE.parent),
            semantic_compiler_mode="off",
        ),
        generation_registry=registry,
    )
    from agent.routing_service import RoutingService
    from agent.semantic_matrix import SemanticMatrixRouter
    from core.embedder import OllamaEmbedder
    routing_package = ROOT / "data/routing/v1a/package.json"
    semantic_router = SemanticMatrixRouter(
        routing_package,
        OllamaEmbedder(model=os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")).embed_query,
        timeout_seconds=2.0,
    ) if routing_package.is_file() else None
    coordinator = RuntimeCoordinator(
        compiler=compiler,
        checkpoint_store=SQLiteCheckpointStore(STATE),
        graph=RuntimeSemanticGraph(compiler),
        routing_service=RoutingService(semantic_router=semantic_router),
    )
    coordinator._v1a_assembly_profile = {
        "runtime": "production-equivalent-explicit-assembly",
        "understanding_model": real_understanding_model,
        "answer_model": real_answer_model,
        "semantic_matrix": semantic_router is not None,
        "routing_package": str(routing_package.relative_to(ROOT)) if routing_package.is_file() else "",
        "production_parity": semantic_router is not None,
    }
    return RuntimeAgentFacade(legacy=legacy, mode="enforce", coordinator=coordinator), coordinator


def _seed_context(agent, category: str, query: str, identity: dict) -> list[dict]:
    seeds = []
    if category == "multi_turn" or (
        category == "summarize_compare"
        and analyze_literature(query).request.document_reference.kind
        in {ReferenceKind.PRIOR_RESULTS, ReferenceKind.ORDINAL, ReferenceKind.CURRENT_DOCUMENT}
    ):
        seeds.append({
            "query": "2019年到2025年之间关于ESN的论文有哪些",
            "reply": agent.chat(
                "2019年到2025年之间关于ESN的论文有哪些", **identity
            ).to_dict(),
        })
    if category == "multi_turn" and query == "只看英文期刊论文":
        seeds.append({
            "query": "那2024年的呢",
            "reply": agent.chat("那2024年的呢", **identity).to_dict(),
        })
    return seeds


def _case_passed(category: str, query: str, state, reply,
                 disallowed_skills: list[str]) -> tuple[bool, list[str]]:
    reasons = []
    terminal = state.state.value if state is not None else "missing"
    if terminal not in {"complete", "wait_user", "unsupported", "failed"}:
        reasons.append(f"non_terminal:{terminal}")
    if disallowed_skills:
        reasons.append("non_read_only_binding:" + ",".join(disallowed_skills))
    if category in NORMAL_CATEGORIES:
        if terminal != "complete":
            reasons.append(f"expected_complete:{terminal}")
        # This chained filter has a labelled empty result in the ACTIVE corpus.
        # A successful, grounded zero-result is a valid retrieval outcome.
        expected_empty = category == "multi_turn" and query == "只看英文期刊论文"
        if not reply.evidence and not expected_empty:
            reasons.append("expected_grounded_evidence")
    elif category == "inventory":
        page = dict(getattr(state, "result_page", {}) or {}) if state is not None else {}
        if (terminal != "complete"
                or getattr(state, "business_outcome", "") != "listed"
                or not page.get("total_documents")
                or page.get("item_count", 0) > page.get("total_documents", 0)
                or (page.get("complete") and page.get("item_count") != page.get("total_documents"))):
            reasons.append("inventory_not_completed")
    return not reasons, reasons


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the v3.1 literature runtime E2E corpus")
    parser.add_argument("--output", default=str(REPORT.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--real-deepseek", action="store_true")
    parser.add_argument("--real-understanding-model", action="store_true")
    parser.add_argument("--real-answer-model", action="store_true")
    parser.add_argument("--case-indices", default="", help="comma-separated 1-based case indices")
    args = parser.parse_args()
    report_path = (ROOT / args.output).resolve()
    if ROOT.resolve() not in report_path.parents:
        raise SystemExit("output must stay inside the project workspace")
    logging.getLogger().setLevel(logging.WARNING)
    if len(CASES) != 80:
        raise SystemExit(f"expected 80 cases, got {len(CASES)}")
    selected_indices = list(range(1, len(CASES) + 1))
    if args.case_indices:
        try:
            selected_indices = sorted({int(item) for item in args.case_indices.split(",")})
        except ValueError:
            raise SystemExit("invalid case indices") from None
        if not selected_indices or selected_indices[0] < 1 or selected_indices[-1] > len(CASES):
            raise SystemExit("case indices outside corpus")
    system = PDFSearchSystem(db="main")
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    gateway = RetrievalGateway(system, generation_registry=registry)
    snapshot = gateway.pin()
    if snapshot is None:
        raise SystemExit("no ACTIVE generation")
    gateway.verify_snapshot(snapshot)
    manifest = gateway.inventory(snapshot)
    accepted_sources = set(manifest["manifest_accepted"])
    collection = system.vector_store.client.get_collection(snapshot.vector_collection)
    vector_metadata = [dict(item or {}) for item in collection.get(include=["metadatas"]).get("metadatas") or []]
    known_pages = {
        (str(item.get("filename", "")), int(item["page_start"]))
        for item in vector_metadata
        if isinstance(item.get("page_start"), int) and item["page_start"] > 0
    }
    real_understanding = bool(args.real_deepseek or args.real_understanding_model)
    real_answer = bool(args.real_deepseek or args.real_answer_model)
    if real_answer and os.environ.get("AUTHORIZE_PRIVATE_DEEPSEEK_E2E") != "1":
        raise SystemExit("real DeepSeek E2E requires explicit authorization flag")
    agent, coordinator = _make_runtime(
        system, registry, real_understanding_model=real_understanding,
        real_answer_model=real_answer,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    rows = []
    latencies = {
        "deterministic_nlu": [], "snapshot_verify": [], "preflight": [],
        "document_search": [], "passage_search": [], "fusion": [],
        "synthesis": [], "total": [],
    }
    citation_checks = []
    known_page_checks = []
    unknown_page_fabrications = 0
    for index in selected_indices:
        case = CASES[index - 1]
        identity = {
            "user_id": f"release-v31-{run_id}",
            "session_id": f"{run_id}-e2e-{index:03d}",
            "thread_id": f"{run_id}-e2e-{index:03d}",
            "history": [],
        }
        seeds = _seed_context(agent, case["category"], case["query"], identity)
        started = time.perf_counter()
        error = ""
        try:
            reply = agent.chat(case["query"], **identity)
        except Exception as exc:  # keep a complete machine-readable failure row
            reply = AgentReply(answer="", intent=SimpleNamespace(to_dict=lambda: {}))
            error = f"{type(exc).__name__}: {exc}"
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        state = coordinator.store.load(
            identity["user_id"], identity["session_id"], identity["thread_id"]
        )
        bound = list(getattr(state, "bound_skill_calls", []) or []) if state else []
        disallowed = [
            str(item.get("skill_name", "")) for item in bound
            if str(item.get("skill_name", "")) not in {
                "discover_documents", "resolve_document", "retrieve_document_passages", "kb_diagnose"
            }
        ]
        passed, reasons = _case_passed(
            case["category"], case["query"], state, reply, disallowed
        )
        if error:
            reasons.append(error)
            passed = False
        semantic = dict(getattr(state, "semantic_result", None) or {}) if state else {}
        nlu_ms = semantic.get("elapsed_ms")
        if isinstance(nlu_ms, (int, float)):
            latencies["deterministic_nlu"].append(float(nlu_ms))
        for observation in list(getattr(state, "observations", []) or []) if state else []:
            stage = dict((observation.get("result") or {}).get("stage_latency_ms") or {})
            for name in (
                "snapshot_verify", "preflight", "document_search",
                "passage_search", "fusion",
            ):
                if isinstance(stage.get(name), (int, float)):
                    latencies[name].append(float(stage[name]))
        synthesis_ms = dict(getattr(state, "context_snapshot", {}) or {}).get(
            "stage_latency_ms", {}
        ).get("synthesis") if state else None
        if isinstance(synthesis_ms, (int, float)):
            latencies["synthesis"].append(float(synthesis_ms))
        latencies["total"].append(total_ms)
        numbers = _citation_numbers(reply.answer)
        if reply.evidence:
            citation_checks.append(bool(numbers) and all(
                1 <= number <= len(reply.evidence) for number in numbers
            ) and all(item.source in accepted_sources for item in reply.evidence))
        for item in reply.evidence:
            if item.page is None:
                if "#p" in item.citation_label():
                    unknown_page_fabrications += 1
            else:
                known_page_checks.append((item.source, item.page) in known_pages)
        rows.append({
            "id": f"E2E{index:03d}", "category": case["category"],
            "query": case["query"], "seed_turns": seeds,
            "terminal_state": state.state.value if state else "missing",
            "generation_id": str(getattr(state, "ingestion_generation", "") or ""),
            "bound_skills": [str(item.get("skill_name", "")) for item in bound],
            "executed_skills": [str(item.get("skill_name", "")) for item in
                                list(getattr(state, "observations", []) or [])] if state else [],
            "business_outcome": str(getattr(state, "business_outcome", "") or "") if state else "",
            "reason_code": str(getattr(state, "reason_code", "") or "") if state else "",
            "result_page": dict(getattr(state, "result_page", {}) or {}) if state else {},
            "request_id": str(getattr(state, "request_id", "") or "") if state else "",
            "router_package_version": str((getattr(state, "routing_hint", {}) or {}).get("schema_version", "")) if state else "",
            "domain": str((getattr(state, "domain_decision", {}) or {}).get("domain", "")) if state else "",
            "accepted_ir": {
                "digest": semantic.get("ir_digest", ""),
                "summary": dict((semantic.get("contract_reports") or {}).get("accepted_literature_ir") or {}),
            },
            "resolved_sources": list(semantic.get("source_ids") or []),
            "blocking_gap_codes": list(semantic.get("diagnostics") or []),
            "logical_plan": dict(semantic.get("logical_plan") or {}),
            "search_statuses": [str((item.get("result") or {}).get("search_status", "")) for item in
                                list(getattr(state, "observations", []) or [])] if state else [],
            "context_compile_fallback": str((getattr(state, "context_snapshot", {}) or {}).get(
                "context_compile_fallback", ""
            )) if state else "",
            "first_breakpoint": (reasons[0] if reasons else "passed"),
            "evidence_count": len(reply.evidence), "citation_numbers": numbers,
            "answer_excerpt": reply.answer[:500], "total_ms": total_ms,
            "passed": passed, "failure_reasons": reasons,
        })
    passed_count = sum(bool(item["passed"]) for item in rows)
    critical_integrity = all(
        item["passed"] for item in rows if item["category"] == "integrity"
    )
    citation_accuracy = sum(citation_checks) / len(citation_checks) if citation_checks else 0.0
    # A corpus without page-level provenance is not a perfect page result.  Keep
    # it explicitly unevaluated so release tooling cannot pass on a vacuous 1.0.
    known_page_accuracy = (
        sum(known_page_checks) / len(known_page_checks) if known_page_checks else None
    )
    report = {
        "schema_version": "literature-runtime-e2e-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "scope": "authoritative NLU -> typed binder -> real ES/Chroma -> deterministic grounded synthesis -> citation validation",
        "external_answer_model": real_answer,
        "assembly_profile": dict(coordinator._v1a_assembly_profile),
        "generation": {
            "id": snapshot.generation_id,
            "snapshot_digest": snapshot.snapshot_digest,
            "manifest_locator": snapshot.manifest_locator,
            "manifest_digest": snapshot.manifest_hash,
        },
        "metrics": {
            "total": len(rows), "passed": passed_count,
            "failed": len(rows) - passed_count,
            "success_rate": passed_count / len(rows),
            "critical_integrity_passed": critical_integrity,
            "citation_visibility_and_source_accuracy": citation_accuracy,
            "citation_checks": len(citation_checks),
            "known_page_accuracy": known_page_accuracy,
            "known_page_checks": len(known_page_checks),
            "unknown_page_fabrications": unknown_page_fabrications,
        },
        "latency": {name: _percentiles(values) for name, values in latencies.items()},
        "llm_repair_latency": {
            "sample_count": 0, "p50_ms": None, "p95_ms": None,
            "status": "not_invoked_in_deterministic_no-external-model-corpus",
        },
        "passed": (
            passed_count / len(rows) >= 0.90 and critical_integrity
            and citation_accuracy == 1.0
            and len(known_page_checks) > 0 and known_page_accuracy == 1.0
            and unknown_page_fabrications == 0
        ),
        "cases": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "generation_id": snapshot.generation_id,
        **report["metrics"], "latency": report["latency"], "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
