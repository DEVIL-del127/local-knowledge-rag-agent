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
    agent, coordinator = _make_runtime(system, registry)
    run_id = uuid.uuid4().hex
    identity = {"user_id": f"exact-v3-{run_id}", "session_id": run_id,
                "thread_id": run_id, "history": []}
    questions = [
        "一种基于多目标优化的ESN网络架构搜索方法_张昭昭主要讲什么",
        "它用了哪些优化目标",
        "实验结果怎么样",
    ]
    rows = []
    for question in questions:
        reply = agent.chat(question, **identity)
        state = coordinator.store.load(identity["user_id"], identity["session_id"], identity["thread_id"])
        rows.append({
            "query": question, "state": state.state.value, "answer": reply.answer,
            "bound_skills": [item.get("skill_name") for item in state.bound_skill_calls],
            "evidence_document_ids": sorted({str(item.get("document_id") or "") for item in state.evidence}),
            "requested_document_ids": state.context_snapshot["literature_ir"]["document_reference"]["document_ids"],
        })
    scoped_ids = [
        {item for item in row["evidence_document_ids"] if item}
        for row in rows
    ]
    passed = (
        all(row["state"] == "complete" for row in rows)
        and all(row["bound_skills"] for row in rows)
        and all(ids and len(ids) == 1 for ids in scoped_ids)
        and len(set.union(*scoped_ids)) == 1
    )
    report = ROOT / "kb-agent/docs/reports/2026-09-09_literature-ir-v3-exact-e2e.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "turns": [{key: row[key] for key in (
        "query", "state", "bound_skills", "requested_document_ids", "evidence_document_ids"
    )} for row in rows]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
