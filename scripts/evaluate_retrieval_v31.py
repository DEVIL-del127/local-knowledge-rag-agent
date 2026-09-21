from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval_gateway import RetrievalGateway
from ingestion.pipeline.generation_registry import GenerationRegistry
from main import PDFSearchSystem


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate labelled literature retrieval qrels")
    parser.add_argument(
        "--output",
        default="kb-agent/docs/reports/2026-09-03_literature-agent-v31-retrieval-eval.json",
    )
    args = parser.parse_args()
    qrels_path = ROOT / "kb-agent/eval/literature_qrels_v31.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    system = PDFSearchSystem(db="main")
    registry = GenerationRegistry(ROOT / "data/ingestion/generation_registry.sqlite3")
    gateway = RetrievalGateway(system, generation_registry=registry)
    snapshot = gateway.pin()
    if snapshot is None:
        raise SystemExit("no ACTIVE generation")
    gateway.verify_snapshot(snapshot)
    rows = []
    total_relevant = total_found = total_returned = 0
    for case in qrels["cases"]:
        topic_terms = (
            ["MCMC", "Markov Chain Monte Carlo", "马尔科夫链蒙特卡洛"]
            if case["sense_id"] == "markov-chain-monte-carlo" else
            ["Echo State Network", "ESN", "回声状态网络", "reservoir computing"]
        )
        outcome = gateway.hybrid_search(
            case["query"], top_k=10, snapshot=snapshot, mode="document",
            topic_terms=topic_terms, sense_id=case["sense_id"],
            year_from=case["year_from"], year_to=case["year_to"],
        )
        returned = [str(item.get("filename", "")) for item in outcome.evidence
                    if item.get("channel") == "bm25"][:10]
        relevant = set(case["relevant"])
        found = relevant.intersection(returned)
        total_relevant += len(relevant)
        total_found += len(found)
        total_returned += len(returned)
        rows.append({
            "id": case["id"], "relevant": sorted(relevant), "returned": returned,
            "missed": sorted(relevant - found),
            "false_positive": sorted(set(returned) - relevant),
            "recall_at_10": len(found) / len(relevant) if relevant else 1.0,
            "precision_at_10": len(found) / len(returned) if returned else (1.0 if not relevant else 0.0),
        })
    recall = total_found / total_relevant if total_relevant else 1.0
    precision = total_found / total_returned if total_returned else 0.0
    report = {
        "schema_version": "literature-retrieval-eval-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_id": snapshot.generation_id if snapshot else "",
        "qrels": str(qrels_path.relative_to(ROOT)).replace("\\", "/"),
        "qrels_sha256": hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
        "case_count": len(rows), "relevant_judgements": total_relevant,
        "recall_at_10": recall, "precision_at_10": precision,
        "threshold": 0.90, "passed": recall >= 0.90 and precision >= 0.90,
        "cases": rows,
    }
    target = (ROOT / args.output).resolve()
    if ROOT.resolve() not in target.parents:
        raise SystemExit("output must stay inside the project workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "case_count", "relevant_judgements", "recall_at_10", "precision_at_10", "passed"
    )}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
