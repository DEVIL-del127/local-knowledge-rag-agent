from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from agent.routing_service import RoutingService
from agent.runtime.domain_router import DomainRouter, RequestDomain
from agent.v1a_acceptance import load_jsonl, validate_core_corpus


def _previous(payload):
    if not payload:
        return None
    return SimpleNamespace(**payload)


def _outcome(domain):
    if domain == RequestDomain.CLARIFY_DOMAIN:
        return "clarify"
    if domain == RequestDomain.UNSUPPORTED:
        return "reject"
    return "auto_release"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    corpus = load_jsonl(args.corpus)
    validate_core_corpus(corpus)
    baseline = DomainRouter()
    candidate = RoutingService(baseline)
    rows = []
    for case in corpus:
        previous = _previous(case.get("precondition"))
        started = time.perf_counter()
        if args.mode == "candidate":
            decision, _ = candidate.route(case["query"], previous=previous)
        else:
            prior = RequestDomain.KB_DOCUMENT.value if previous else None
            decision = baseline.route(case["query"], prior_domain=prior)
        latency = (time.perf_counter() - started) * 1000.0
        expected_domain = case["expected"]["domain"]
        actual_outcome = _outcome(decision.domain)
        correct = decision.domain.value == expected_domain
        rows.append({
            "case_id": case["id"], "domain": decision.domain.value,
            "reason_code": decision.reason_code, "outcome": actual_outcome,
            "correct": correct,
            "critical_error": not correct and case["expected"]["outcome"] in {"clarify", "reject"}
                              and actual_outcome == "auto_release",
            "unnecessary_clarification": actual_outcome == "clarify"
                                          and case["expected"]["outcome"] == "auto_release",
            "incorrect_rejection": actual_outcome == "reject"
                                   and case["expected"]["outcome"] != "reject",
            "domain_correct": correct,
            "full_task_completed": None,
            "task_completion_status": "not_measured_by_domain_router",
            "low_cost_latency_ms": latency,
        })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(target, len(rows))


if __name__ == "__main__":
    main()
