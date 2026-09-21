from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem
from scripts.run_runtime_e2e_v31 import _make_runtime


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    system = PDFSearchSystem(db="main")
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    rows = []
    for query in ("总结ESN文献", "对比ESN文献"):
        agent, coordinator = _make_runtime(system, registry)
        run_id = uuid.uuid4().hex
        identity = {"user_id": f"topic-v3-{run_id}", "session_id": run_id,
                    "thread_id": run_id, "history": []}
        reply = agent.chat(query, **identity)
        state = coordinator.store.load(identity["user_id"], identity["session_id"], identity["thread_id"])
        bindings = [item.get("skill_name") for item in state.bound_skill_calls]
        passage_scopes = [
            list(item.get("arguments", {}).get("document_ids") or [])
            for item in state.bound_skill_calls
            if item.get("skill_name") == "retrieve_document_passages"
        ]
        rows.append({
            "query": query, "state": state.state.value, "answer": reply.answer,
            "bound_skills": bindings, "passage_document_scopes": passage_scopes,
            "literature_task": state.context_snapshot.get("literature_ir", {}).get("task"),
            "observation_statuses": [
                {"skill": item.get("skill_name"), "status": item.get("status"),
                 "error": item.get("error"),
                 "document_ids": sorted({str(ev.get("document_id") or "")
                                         for ev in (item.get("result") or {}).get("evidence", [])
                                         if ev.get("document_id")})}
                for item in state.observations
            ],
        })
    passed = all(
        row["state"] == "complete"
        and row["bound_skills"][:1] == ["discover_documents"]
        and len(row["passage_document_scopes"]) == 2
        and all(len(scope) == 1 for scope in row["passage_document_scopes"])
        and row["passage_document_scopes"][0] != row["passage_document_scopes"][1]
        for row in rows
    )
    report = ROOT / "kb-agent/docs/reports/2026-09-09_literature-ir-v3-topic-e2e.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "cases": [
        {key: row[key] for key in ("query", "state", "bound_skills", "passage_document_scopes")}
        for row in rows
    ]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
