from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# These contracts require dedicated acceptance evidence. Functional corpus results
# alone must never promote them to implemented or verified.
PENDING_ARCHITECTURE_GATES = (
    "single_graph_execution_ownership",
    "pinned_clarification_resume",
    "resolvable_catalog_snapshot_contract",
    "stable_public_request_replay_identity",
    "transactional_per_attempt_budget",
    "unified_persistence_privacy_and_retention",
    "local_ollama_acceptance",
    "bounded_external_model_acceptance",
)
GATE_TEST_IDS = {
    "single_graph_execution_ownership": {"tests.test_execution_graph_v8::test_all_domains_use_single_graph"},
    "pinned_clarification_resume": {"tests.test_generation_resume_v8::test_retired_generation_resume_after_restart"},
    "resolvable_catalog_snapshot_contract": {"tests.test_catalog_ref_v8::test_legacy_catalog_ref_resolves_and_detects_tamper"},
    "stable_public_request_replay_identity": {"tests.test_runtime_operations_v8::test_v8_public_request_replays_after_new_coordinator"},
    "transactional_per_attempt_budget": {"tests.test_token_reservations_v8::test_multiprocess_reservations_share_ceiling"},
    "unified_persistence_privacy_and_retention": {"tests.test_privacy_acceptance_v8::test_all_persistence_boundaries_canary_and_ttl"},
    "local_ollama_acceptance": {"tests.test_local_model_acceptance_v8::test_full_nlu_and_embedding_live"},
    "bounded_external_model_acceptance": {"tests.test_external_acceptance_v8::test_persisted_two_attempt_budget_smoke"},
}

from agent.privacy import PersistencePolicy
from ingestion.pipeline.generation_registry import GenerationRecord, GenerationState
from agent.release_evidence import code_tree_digest, verify_gate_evidence
from agent.app_settings import AppSettings


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _ledger_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    # Audit must not create a missing database or change its schema/journal mode.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_calls'"
        ).fetchone()
        if exists is None:
            return {}
        rows = connection.execute(
            "SELECT state,COUNT(*) FROM model_calls GROUP BY state ORDER BY state"
        ).fetchall()
    return {str(state): int(count) for state, count in rows}


def _read_active(path: Path) -> tuple[GenerationRecord | None, int]:
    """Read pointer and record in one SQLite snapshot without registry setup."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        pointer = connection.execute(
            "SELECT active_generation, revision FROM registry_meta WHERE singleton=1"
        ).fetchone()
        if pointer is None:
            raise ValueError("registry pointer missing")
        generation_id, revision = pointer
        if generation_id is None:
            return None, int(revision)
        row = connection.execute("SELECT payload FROM generations WHERE generation_id=?",
                                 (generation_id,)).fetchone()
        if row is None:
            raise ValueError("active record missing")
        record = GenerationRecord.from_dict(json.loads(row[0]))
        if record.generation_id != generation_id or record.state != GenerationState.ACTIVE:
            raise ValueError("active record identity mismatch")
        return record, int(revision)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the content-minimized Agent architecture release audit"
    )
    parser.add_argument("--junit", required=True)
    parser.add_argument("--e2e", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture-evidence")
    args = parser.parse_args()
    paths = {name: (ROOT / value).resolve() for name, value in {
        "junit": args.junit, "e2e": args.e2e,
        "retrieval": args.retrieval, "output": args.output,
    }.items()}
    if any(ROOT.resolve() not in path.parents for path in paths.values()):
        raise SystemExit("all audit paths must stay inside the project")

    tests = _junit(paths["junit"])
    e2e = _json(paths["e2e"])
    retrieval = _json(paths["retrieval"])
    settings = AppSettings.from_env(ROOT)
    active, revision = _read_active(settings.ingestion_registry_path)
    if active is None:
        raise SystemExit("no ACTIVE generation")
    e2e_metrics = dict(e2e.get("metrics") or {})
    expected_manifest = "57cfe0b850b7715fc360db059f16d4a1ce20af0f1ca515bb696a0a8c3180ac3f"
    gates = {
        "offline_suite_zero_failure_error_skip": bool(
            tests["tests"] and not tests["failures"]
            and not tests["errors"] and not tests["skipped"]
        ),
        "runtime_e2e_80_of_80": bool(
            e2e_metrics.get("total") == 80 and e2e_metrics.get("failed") == 0
            and e2e.get("passed") is True
        ),
        "citation_and_page_integrity": bool(
            e2e_metrics.get("citation_visibility_and_source_accuracy") == 1.0
            and e2e_metrics.get("known_page_accuracy") == 1.0
            and e2e_metrics.get("unknown_page_fabrications") == 0
        ),
        "retrieval_quality": bool(
            retrieval.get("passed") is True
            and retrieval.get("recall_at_10") == 1.0
            and retrieval.get("precision_at_10") == 1.0
        ),
        "active_generation_frozen": bool(
            active.generation_id == "g20260904t091206-14daf132"
            and revision == 11 and active.manifest_hash == expected_manifest
        ),
        "reports_match_active_generation": bool(
            (e2e.get("generation") or {}).get("id") == active.generation_id
            and (e2e.get("generation") or {}).get("manifest_digest") == active.manifest_hash
            and retrieval.get("generation_id") == active.generation_id
        ),
    }
    functional_pass = all(gates.values())
    evidence = {}
    if args.architecture_evidence:
        evidence_path = (ROOT / args.architecture_evidence).resolve()
        if ROOT.resolve() not in evidence_path.parents:
            raise SystemExit("architecture evidence must stay inside the project")
        evidence = _json(evidence_path)
    current_code = code_tree_digest(ROOT)
    architecture_status = {
        name: verify_gate_evidence(ROOT, evidence.get(name), code_digest=current_code,
                                   required_test_ids=GATE_TEST_IDS[name])
        for name in PENDING_ARCHITECTURE_GATES
    }
    gates.update({name: result["status"] == "pass" for name, result in architecture_status.items()})
    report = PersistencePolicy.prepare("release-audit-v1", {
        "schema_version": "agent-architecture-release-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_pass": all(gates.values()),
        "generation": {
            "id": active.generation_id, "registry_revision": revision,
            "manifest_digest": active.manifest_hash,
            "document_count": active.document_count, "chunk_count": active.chunk_count,
        },
        "metrics": {
            "functional_gates_pass": functional_pass,
            "architecture_completion_verified": all(item["status"] == "pass" for item in architecture_status.values()),
            "architecture_gate_evidence": architecture_status,
            "code_tree_digest": current_code,
            "evidence_sources": {
                name: {
                    "path": str(paths[name].relative_to(ROOT)),
                    "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                } for name in ("junit", "e2e", "retrieval")
            },
            "e2e_created_at_utc": e2e.get("created_at_utc"),
            "retrieval_created_at_utc": retrieval.get("created_at_utc"),
            "gates": gates,
            "runtime_e2e": {key: e2e_metrics.get(key) for key in (
                "total", "passed", "failed", "success_rate",
                "citation_visibility_and_source_accuracy", "known_page_accuracy",
                "unknown_page_fabrications",
            )},
            "retrieval": {key: retrieval.get(key) for key in (
                "case_count", "recall_at_10", "precision_at_10", "passed",
            )},
        },
        "tests": tests,
        "privacy": {
            "content_minimized": True,
            "contains_query_answer_or_evidence_fields": False,
            "report_sha256_algorithm": "sha256",
        },
        "model_calls": {
            "ledger_state_counts": _ledger_counts(settings.state_dir / "checkpoints/runtime.sqlite3"),
            "new_real_deepseek_attempts_in_this_audit": 0,
        },
        "unresolved": [name for name, passed in gates.items() if not passed],
    })
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    paths["output"].write_bytes(encoded)
    print(json.dumps({
        "gate_pass": report["gate_pass"], "unresolved": report["unresolved"],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
