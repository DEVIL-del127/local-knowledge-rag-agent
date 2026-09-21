from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.app_settings import AppSettings
from agent.application import build_application


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs/v1a-production-real-answer-smoke.json"


CASES = (
    {
        "id": "P01", "kind": "topic_qa", "query": "ESN是什么？",
        "expected": {"business_outcome": "answered", "evidence": True},
    },
    {
        "id": "P02", "kind": "explicit_document_qa",
        "query": "《Adaptive model based on ESN for anomaly detection in industrial systems.pdf》的结论是什么？",
        "expected": {"business_outcome": "answered", "evidence": True},
    },
    {
        "id": "P03", "kind": "summary",
        "query": "总结《1-s2.0-S0925231221011309-main的全文翻译.pdf》的方法和结论",
        "expected": {"business_outcome": "answered", "evidence": True},
    },
    {
        "id": "P04", "kind": "explicit_compare",
        "query": (
            "比较《Adaptive model based on ESN for anomaly detection in industrial systems.pdf》"
            "和《1-s2.0-S0925231221011309-main的全文翻译.pdf》的方法和结论"
        ),
        "expected": {"business_outcome": "answered", "evidence": True,
                     "comparison_complete": True},
    },
    {
        "id": "P05", "kind": "multiturn_reference", "seed": "查找ESN论文",
        "query": "第二篇讲什么？",
        "expected": {"business_outcome": "answered", "evidence": True},
    },
    {
        "id": "P06", "kind": "empty_result", "query": "查找1990年到1991年的ESN论文",
        "expected": {"business_outcome": "no_match", "evidence": False},
    },
    {
        "id": "P07", "kind": "necessary_clarification", "query": "比较这两篇",
        "expected": {"business_outcome": "needs_clarification", "evidence": False},
    },
    {
        "id": "P08", "kind": "inventory", "query": "库里有多少篇，哪些还没入索引",
        "expected": {"business_outcome": "listed", "evidence": False},
    },
)


def _case_checks(case, state, reply) -> dict[str, bool]:
    expected = case["expected"]
    coverage = dict(state.context_snapshot.get("comparison_coverage") or {})
    checks = {
        "business_outcome": state.business_outcome == expected["business_outcome"],
        "evidence_presence": bool(reply.evidence) == bool(expected["evidence"]),
        "terminal_state": state.state.value in {"complete", "wait_user"},
        "citation_sources": all(bool(item.source) for item in reply.evidence),
    }
    if "comparison_complete" in expected:
        checks["comparison_complete"] = (
            bool(coverage.get("complete")) == bool(expected["comparison_complete"])
        )
    if expected["business_outcome"] == "listed":
        page = dict(state.result_page or {})
        checks["inventory_page"] = (
            int(page.get("total_documents", 0)) >= int(page.get("item_count", 0)) > 0
        )
    if expected["business_outcome"] == "needs_clarification":
        checks["pending_clarification"] = state.pending_clarification is not None
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V1-A real-answer production assembly smoke")
    parser.add_argument("--output", default=str(DEFAULT_REPORT.relative_to(ROOT)).replace("\\", "/"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if ROOT.resolve() not in output.parents:
        raise SystemExit("output must stay inside project workspace")

    load_dotenv(ROOT / ".env")
    if os.environ.get("AUTHORIZE_PRIVATE_DEEPSEEK_E2E") != "1":
        raise SystemExit("real-answer production smoke requires AUTHORIZE_PRIVATE_DEEPSEEK_E2E=1")
    os.environ["AGENT_EXECUTION_PROFILE"] = "enforce"
    os.environ["AGENT_DB"] = "main"
    os.environ["AGENT_USER_ID"] = "v1a-production-smoke"
    os.environ["SEMANTIC_COMPILER_LLM"] = "0"
    os.environ["AGENT_LLM_CLARIFICATION"] = "0"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    # A fresh cache/state directory proves real answer-model calls instead of
    # accepting answers cached by an earlier smoke run.
    os.environ["AGENT_STATE_DIR"] = str(ROOT / "agent_state/v1a_production_smoke" / run_id)
    app = build_application(AppSettings.from_env(ROOT))
    rows = []
    try:
        for case in CASES:
            session = f"{run_id}-{case['id'].lower()}"
            common = {
                "user_id": "v1a-production-smoke",
                "session_id": session,
                "thread_id": session,
            }
            if case.get("seed"):
                app.agent.chat(case["seed"], request_id=f"{case['id']}-seed-{run_id}", **common)
            started = time.perf_counter()
            reply = app.agent.chat(case["query"], request_id=f"{case['id']}-main-{run_id}", **common)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            state = app.agent.coordinator.store.load(common["user_id"], session, session)
            checks = _case_checks(case, state, reply)
            rows.append({
                "id": case["id"], "kind": case["kind"], "query": case["query"],
                "state": state.state.value, "business_outcome": state.business_outcome,
                "reason_code": state.reason_code, "evidence_count": len(reply.evidence),
                "model_calls": state.budgets.model_calls,
                "comparison_coverage": dict(state.context_snapshot.get("comparison_coverage") or {}),
                "result_page": dict(state.result_page or {}), "elapsed_ms": elapsed_ms,
                "checks": checks, "passed": all(checks.values()),
                "answer_excerpt": reply.answer[:500],
            })
    finally:
        app.close()

    report = {
        "schema_version": "v1a-production-real-answer-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "assembly": "agent.application.build_application",
        "execution_profile": "enforce",
        "external_answer_model": True,
        "understanding_model": False,
        "fresh_state_directory": True,
        "total": len(rows), "passed_count": sum(row["passed"] for row in rows),
        "passed": len(rows) == len(CASES) and all(row["passed"] for row in rows),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output), "passed": report["passed"],
        "passed_count": report["passed_count"], "total": report["total"],
        "cases": [{"id": row["id"], "passed": row["passed"],
                   "outcome": row["business_outcome"], "model_calls": row["model_calls"],
                   "elapsed_ms": row["elapsed_ms"]} for row in rows],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
