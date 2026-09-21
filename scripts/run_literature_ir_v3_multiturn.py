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
    identity = {"user_id": f"literature-v3-{run_id}", "session_id": run_id,
                "thread_id": run_id, "history": []}
    questions = [
        "帮我查一下2023年之后的ESN文献",
        "这两篇文献分别讲什么",
        "第二篇用了什么方法",
        "对比它们的实验结果",
    ]
    rows = []
    for question in questions:
        reply = agent.chat(question, **identity)
        state = coordinator.store.load(identity["user_id"], identity["session_id"], identity["thread_id"])
        rows.append({
            "query": question, "answer": reply.answer, "state": state.state.value,
            "bound_skills": [item.get("skill_name") for item in state.bound_skill_calls],
            "evidence_count": len(state.evidence),
            "literature_ir": state.context_snapshot.get("literature_ir"),
            "artifact_ref": state.context_snapshot.get("last_result_artifact_ref"),
        })
    report = ROOT / "kb-agent/docs/reports/2026-09-09_literature-ir-v3-multiturn-e2e.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = all(row["state"] == "complete" and row["evidence_count"] > 0 for row in rows)
    print(json.dumps({"passed": passed, "turns": [{
        "query": row["query"], "state": row["state"], "bound_skills": row["bound_skills"],
        "evidence_count": row["evidence_count"],
        "document_ids": row["literature_ir"]["document_reference"]["document_ids"],
    } for row in rows]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
