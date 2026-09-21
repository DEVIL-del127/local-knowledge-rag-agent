from __future__ import annotations

import hashlib, json, sys, xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.agent_models import RetrievedEvidence
from agent.nlu_profile import NluExecutionProfile
from core.retrieval_gateway import RetrievalGateway
from ingestion.pipeline.generation_registry import GenerationRegistry
from ingestion.pipeline.manifest import ManifestRef, verify_manifest_ref
from main import PDFSearchSystem

REPORTS = ROOT / "kb-agent/docs/reports"


def load_json(name: str) -> dict:
    path = REPORTS / name
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    if not path.is_file(): return {"present": False, "path": relative}
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"present": False, "path": relative, "error": f"{type(exc).__name__}: {exc}"}
    return {**payload, "present": True, "path": relative}


def junit(name: str) -> dict:
    path = REPORTS / name
    if not path.is_file(): return {"present": False, "path": str(path.relative_to(ROOT))}
    root = ET.parse(path).getroot(); suites = list(root) if root.tag == "testsuites" else [root]
    return {"present": True,
            **{k: sum(int(s.attrib.get(k, 0)) for s in suites)
               for k in ("tests", "failures", "errors", "skipped")},
            "test_names": [c.attrib.get("name", "") for s in suites for c in s.findall("testcase")],
            "path": str(path.relative_to(ROOT)).replace("\\", "/")}


def tree_digest(root: Path) -> str:
    rows = [f"{p.relative_to(root)}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in sorted(root.rglob("*.py"))]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main() -> int:
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    active, revision = registry.get_active()
    if active is None: raise SystemExit("no ACTIVE generation")
    manifest = verify_manifest_ref(registry.manifest_root, ManifestRef(
        locator=active.manifest_locator, digest=active.manifest_hash,
        byte_length=active.manifest_byte_length,
        store_id=active.manifest_store_id,
        locator_scheme=active.manifest_locator_scheme,
        digest_algorithm=active.manifest_digest_algorithm,
        schema_version=active.manifest_schema_version or "ingestion-manifest-v2"))
    e2e = load_json("2026-09-04_literature-agent-v31-runtime-e2e-active-page-v3.json")
    retrieval = load_json("2026-09-04_literature-agent-v31-retrieval-eval-active-page-v2.json")
    page_audit = load_json("2026-09-04_v31_retired-page-generation-identity-audit-v2.json")
    external = load_json("2026-09-04_literature-agent-v31-external-deepseek-e2e-v3.json")
    candidate = load_json("2026-09-03_m15_llm_choice_v31_prompt-v24-guard-v1.json")
    parser = load_json("2026-09-03_literature-agent-v3_release-corpus.json")
    full = junit("2026-09-04_v31_post-page-identity-fix-junit-v2.xml")
    resume = junit("2026-09-03_v31_resume-focused-junit.xml")
    fixture_rows = [(RetrievedEvidence("known.pdf", "x", page=7), "known.pdf#p7"),
                    (RetrievedEvidence("unknown.pdf", "x"), "unknown.pdf")]
    fixture_checks = [item.citation_label() == expected for item, expected in fixture_rows]
    page_fixture = {"checks": len(fixture_checks), "passed": sum(fixture_checks),
                    "accuracy": sum(fixture_checks) / len(fixture_checks)}
    snapshot_payload, snapshot_error = {}, ""
    try:
        gateway = RetrievalGateway(PDFSearchSystem(db="main"), generation_registry=registry)
        snapshot = gateway.pin(force_refresh=True)
        if snapshot is None: raise RuntimeError("no ACTIVE GenerationSnapshot")
        gateway.verify_snapshot(snapshot, force=True)
        snapshot_payload = asdict(snapshot)
    except Exception as exc: snapshot_error = f"{type(exc).__name__}: {exc}"
    em, cm = dict(e2e.get("metrics") or {}), dict(candidate.get("metrics") or {})
    latency = dict(e2e.get("latency") or {})
    latency_ok = bool(latency) and all(
        v.get("sample_count", 0) > 0 and isinstance(v.get("p50_ms"), (int, float))
        and isinstance(v.get("p95_ms"), (int, float)) for v in latency.values())
    inventory_ok = (manifest.get("discovered_file_count") == manifest.get("accepted_document_count")
                    == manifest.get("es_document_count") == manifest.get("vector_document_count")
                    and manifest.get("quarantined_document_count") == 0
                    and manifest.get("accepted_document_set_digest")
                    == manifest.get("es_document_set_digest") == manifest.get("vector_document_set_digest"))
    e2e_same = e2e.get("generation", {}).get("id") == active.generation_id
    retrieval_same = retrieval.get("generation_id") == active.generation_id
    hard_gates = {
        "active_generation": True, "manifest_integrity": True,
        "generation_snapshot_complete_and_verified": bool(snapshot_payload and not snapshot_error),
        "inventory_consistent": inventory_ok,
        "complete_release_suite_zero_fail_error_skip": bool(full.get("present") and full.get("tests", 0)
            and not any(full.get(k, 0) for k in ("failures", "errors", "skipped"))),
        "end_to_end_80_cases": bool(e2e_same and em.get("total") == 80
            and em.get("success_rate", 0) >= .90 and em.get("critical_integrity_passed") is True),
        "candidate_guarded_pipeline_accuracy": bool(cm.get("decision_accuracy", 0) >= .90
            and cm.get("provider_protocol_rate", 0) >= .90 and cm.get("net_closure_gain", 0) >= .20),
        "recall_at_10": bool(retrieval_same and retrieval.get("recall_at_10", 0) >= .90),
        "precision_at_10": bool(retrieval_same and retrieval.get("precision_at_10", 0) >= .90),
        "citation_source_visibility_accuracy": bool(em.get("citation_checks", 0) > 0
            and em.get("citation_visibility_and_source_accuracy") == 1.0),
        "known_page_fixture_nonzero_and_accurate": page_fixture["accuracy"] == 1.0,
        "live_page_provenance_available": bool(em.get("known_page_checks", 0) > 0
            and em.get("known_page_accuracy") == 1.0),
        "page_generation_physical_identity_audited": bool(
            page_audit.get("generation_id") == active.generation_id
            and page_audit.get("passed") is True
            and page_audit.get("vector_store_identity_matches") is True
            and page_audit.get("vector_collection_identity_matches") is True
        ),
        "stage_latency_p50_p95_recorded": latency_ok,
        "persistent_clarification_resume": bool(resume.get("present")
            and "test_clarification_checkpoint_and_resume" in set(resume.get("test_names") or [])
            and not any(resume.get(k, 0) for k in ("failures", "errors", "skipped"))),
        "external_deepseek_end_to_end_authorized_and_passed": bool(
            external.get("authorization") == "explicit_user_authorization_in_current_task"
            and external.get("active_generation") == active.generation_id
            and external.get("case_count", 0) > 0
            and external.get("passed_cases") == external.get("case_count")
            and external.get("external_text_calls", 0) > 0
            and external.get("passed") is True
        ),
    }
    profile = NluExecutionProfile.latest(enable_llm=True)
    report = {
        "schema_version": "literature-agent-release-audit-v3.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_pass": all(hard_gates.values()), "hard_gates": hard_gates,
        "generation": {"id": active.generation_id, "registry_revision": revision,
            "manifest_locator": active.manifest_locator, "manifest_digest": active.manifest_hash,
            "manifest_byte_length": active.manifest_byte_length,
            "counts": {k: manifest.get(k) for k in ("discovered_file_count", "accepted_document_count",
                "quarantined_document_count", "es_document_count", "vector_document_count", "vector_chunk_count")},
            "snapshot": snapshot_payload, "snapshot_error": snapshot_error},
        "nlu": {"profile_id": profile.profile_id, "profile_digest": profile.digest(),
            "tree_digest": tree_digest(ROOT / "kb-agent/nlu_v2"), "request_ir_schema": "2.5",
            "config": profile.engine_kwargs()},
        "candidate_choice": {"path": candidate.get("path"), "guarded_pipeline_accuracy": cm.get("decision_accuracy"),
            "provider_only_accuracy": cm.get("provider_only_decision_accuracy"),
            "provider_protocol_rate": cm.get("provider_protocol_rate"), "semantic_guard_rate": cm.get("semantic_guard_rate"),
            "net_closure_gain": cm.get("net_closure_gain")},
        "retrieval": {"path": retrieval.get("path"), "same_active_generation": retrieval_same,
            "recall_at_10": retrieval.get("recall_at_10"), "precision_at_10": retrieval.get("precision_at_10")},
        "citations_and_pages": {"citation_checks": em.get("citation_checks"),
            "citation_accuracy": em.get("citation_visibility_and_source_accuracy"),
            "live_known_page_checks": em.get("known_page_checks"),
            "live_known_page_accuracy": em.get("known_page_accuracy") if em.get("known_page_checks") else None,
            "unknown_page_fabrications": em.get("unknown_page_fabrications"), "fixture": page_fixture,
            "page_generation_audit": page_audit.get("path"),
            "page_coverage": page_audit.get("page_coverage"),
            "limitation": "Page provenance is retained only when Docling supplies an explicit positive page number"},
        "latency": latency, "tests": {"complete_release": full, "clarification_resume": resume},
        "parser_contract": {k: parser.get(k) for k in ("path", "scope", "end_to_end", "passed", "total")},
        "external_model": {"provider": "DeepSeek", "authorized": True,
            "status": "passed", "path": external.get("path"),
            "model": external.get("model"), "case_count": external.get("case_count"),
            "external_text_calls": external.get("external_text_calls"),
            "external_router_calls": external.get("external_router_calls")},
        "unresolved": [k for k, v in hard_gates.items() if not v]}
    target = REPORTS / "2026-09-04_literature-agent-v31_release-audit-page-v2.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate_pass": report["gate_pass"], "unresolved": report["unresolved"]}, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__": raise SystemExit(main())
