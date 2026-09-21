from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from agent.agent_service import AgentSettings, PrivateKnowledgeAgent
from agent.agent_skills import SkillRegistry
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.facade import RuntimeAgentFacade
from agent.runtime.graph import RuntimeSemanticGraph
from agent.semantic_compiler_adapter import SemanticCompilerAdapter
from core.contract_store import ContractStore
from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem


CASES = [
    "ESN的谱半径有什么作用",
]


class CountingDeepSeekClient:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client
        self.settings = client.settings
        self.text_calls = 0
        self.router_calls = 0

    def invoke_text(self, **kwargs):
        self.text_calls += 1
        return self.client.invoke_text(**kwargs)

    def invoke_router(self, **kwargs):
        self.router_calls += 1
        return self.client.invoke_router(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("AUTHORIZE_PRIVATE_DEEPSEEK_E2E") != "1":
        raise SystemExit("set AUTHORIZE_PRIVATE_DEEPSEEK_E2E=1 after explicit user authorization")
    load_dotenv(ROOT / ".env")
    client = CountingDeepSeekClient(DeepSeekClient(DeepSeekSettings.from_env()))
    client.client.require_api_key()
    system = PDFSearchSystem(db="main")
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    active, revision = registry.get_active()
    if active is None:
        raise SystemExit("no ACTIVE generation")
    compiler = SemanticCompilerAdapter(
        search_backend=system,
        generation_registry=registry,
        enable_llm=True,
        require_active_generation=True,
        contract_store=ContractStore(
            registry.path.parent / "contracts", store_id="local-contracts-v1",
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="deepseek-e2e-v31-") as tmp:
        legacy = PrivateKnowledgeAgent(
            search_backend=system,
            deepseek_client=client,
            registry=SkillRegistry(),
            settings=AgentSettings(state_dir=tmp, semantic_compiler_mode="off"),
            semantic_compiler=compiler,
            generation_registry=registry,
        )
        coordinator = RuntimeCoordinator(
            compiler=compiler,
            checkpoint_store=SQLiteCheckpointStore(Path(tmp) / "runtime.sqlite3"),
            graph=RuntimeSemanticGraph(compiler),
        )
        agent = RuntimeAgentFacade(legacy=legacy, mode="enforce", coordinator=coordinator)
        for index, query in enumerate(CASES, 1):
            identity = {
                "user_id": "authorized-deepseek-release-v31",
                "session_id": f"external-e2e-{index}",
                "thread_id": f"external-e2e-{index}",
                "history": [],
            }
            started = time.perf_counter()
            error = ""
            try:
                reply = agent.chat(query, **identity)
            except Exception as exc:
                reply = None
                error = f"{type(exc).__name__}: {exc}"
            state = coordinator.store.load(
                identity["user_id"], identity["session_id"], identity["thread_id"]
            )
            evidence = list(getattr(reply, "evidence", []) or []) if reply else []
            answer = str(getattr(reply, "answer", "") or "") if reply else ""
            labels = [item.citation_label() for item in evidence]
            passed = bool(
                not error and answer.strip() and evidence and labels
                and any(label in answer for label in labels)
                and getattr(state, "ingestion_generation", "") == active.generation_id
                and getattr(getattr(state, "state", None), "value", "") == "complete"
            )
            rows.append({
                "id": f"DEEPSEEK-E2E-{index}",
                "query": query,
                "passed": passed,
                "error": error,
                "terminal_state": getattr(getattr(state, "state", None), "value", "missing"),
                "generation_id": getattr(state, "ingestion_generation", ""),
                "evidence_count": len(evidence),
                "citation_count": len(labels),
                "answer_nonempty": bool(answer.strip()),
                "citation_visible": any(label in answer for label in labels),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            })
    passed = bool(rows and all(row["passed"] for row in rows) and client.text_calls >= len(rows))
    report = {
        "schema_version": "literature-agent-external-deepseek-e2e-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "explicit_user_authorization_in_current_task",
        "provider": "DeepSeek",
        "model": client.settings.model,
        "base_url": client.settings.base_url,
        "active_generation": active.generation_id,
        "registry_revision": revision,
        "case_count": len(rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "external_text_calls": client.text_calls,
        "external_router_calls": client.router_calls,
        "passed": passed,
        "cases": rows,
    }
    target = (ROOT / args.output).resolve()
    if ROOT.resolve() not in target.parents:
        raise SystemExit("output outside workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "model", "active_generation", "case_count", "passed_cases",
        "external_text_calls", "external_router_calls", "passed",
    )}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
