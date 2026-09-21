"""Evaluate the frozen M1.5 selection-only LLM matrix with resumable rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider
from nlu_v2.semantic_candidates import CandidateMenu, SemanticCandidate


ORACLE_PATH = ROOT / "tests" / "fixtures" / "m15_llm_choice_oracle.json"
BASELINE_PATH = ROOT / "tests" / "fixtures" / "m15_deterministic_baseline.json"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3, choices=[3])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))
        _verify_resume(state)
    else:
        state = _empty_report(args.model)
    completed = {item["oracle_id"] for item in state["rows"]}
    oracle = _load(ORACLE_PATH)
    rows = oracle["rows"][:args.limit or None]
    provider = OllamaOpenAIProvider(model=args.model)
    health = provider.health()
    state["provider_health"] = health
    if not health.get("available"):
        state["gate_result"] = "not_evaluated"
        _write(args.output, state)
        return 2
    extractor = BoundedLLMExtractor(provider, max_remote_attempts=1)
    for index, row in enumerate(rows, start=1):
        if row["oracle_id"] in completed:
            continue
        attempts = []
        for attempt in range(1, args.attempts + 1):
            started = time.perf_counter()
            result = extractor.choose_candidate(_menu(row))
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            selected_digest = ""
            if result.payload.get("candidate_id"):
                selected = next((item for item in row["candidates"]
                                 if item["payload_digest"] == result.payload["candidate_id"]), None)
                selected_digest = selected["payload_digest"] if selected else ""
            provider_decision, provider_digest = _provider_outcome(
                result.raw_response_excerpt, row,
            )
            attempts.append({
                "attempt": attempt, "decision": result.payload.get("decision", "provider_failure"),
                "selected_payload_digest": selected_digest, "valid_protocol": result.valid_json,
                "decision_source": (
                    "semantic_guard" if "candidate-semantic-guard" in result.stop_reason
                    else "provider"
                ),
                "provider_decision": provider_decision,
                "provider_selected_payload_digest": provider_digest,
                "failure_kind": result.failure_kind, "error": result.error,
                "stop_reason": result.stop_reason, "prompt_chars": result.prompt_chars,
                "response_chars": result.response_chars,
                "duration_ms": duration_ms,
                "raw_response_excerpt": result.raw_response_excerpt,
            })
        outcome = _majority(attempts)
        provider_outcome = _majority(attempts, prefix="provider_")
        expected = (row["expected_decision"], row.get("expected_selected_payload_digest", ""))
        correct = outcome == expected
        state["rows"].append({
            "oracle_id": row["oracle_id"], "family": row["family"], "attempts": attempts,
            "majority_decision": outcome[0], "majority_selected_payload_digest": outcome[1],
            "provider_majority_decision": provider_outcome[0],
            "provider_majority_selected_payload_digest": provider_outcome[1],
            "provider_oracle_correct": provider_outcome == expected,
            "oracle_correct": correct,
            "concrete_closure": int(correct and outcome[0] == "select"
                                    and row.get("expected_coverage_delta", 0) > 0),
        })
        state["updated_at"] = datetime.now().astimezone().isoformat()
        _summarize(state, oracle)
        _write(args.output, state)
        print(f"[{index}/{len(rows)}] {row['oracle_id']}: {outcome[0]} correct={correct}", flush=True)
    _summarize(state, oracle)
    _write(args.output, state)
    return 0 if state["gate_result"] == "pass" else 1


def _menu(row):
    candidates = tuple(SemanticCandidate(
        item["payload_digest"], "candidate_select", item["semantic_payload"], item["summary"]
    ) for item in row["candidates"])
    return CandidateMenu(
        row["oracle_id"], row["oracle_id"], row["source_clause"],
        {"family": row["family"]}, candidates, "llm_choice", (),
        _digest(row["source_clause"]), _digest(row["family"]), "frozen",
        row["menu_digest"],
    )


def _majority(attempts, *, prefix=""):
    votes = Counter((item[f"{prefix}decision"], item[f"{prefix}selected_payload_digest"])
                    for item in attempts if item["valid_protocol"])
    if not votes:
        return "provider_failure", ""
    outcome, count = votes.most_common(1)[0]
    return outcome if count >= 2 else ("no_majority", "")


def _summarize(state, oracle):
    rows = state["rows"]
    total = len(oracle["rows"])
    correct = sum(item["oracle_correct"] for item in rows)
    valid = sum(attempt["valid_protocol"] for item in rows for attempt in item["attempts"])
    attempts = sum(len(item["attempts"]) for item in rows)
    llm_closure = sum(item["concrete_closure"] for item in rows)
    provider_correct = sum(bool(item.get("provider_oracle_correct")) for item in rows)
    guarded_attempts = sum(
        item.get("decision_source") == "semantic_guard"
        for row in rows for item in row["attempts"]
    )
    baseline = _load(BASELINE_PATH)
    abstain = sum(item["deterministic_abstain"]["oracle_correct"]
                  and item["deterministic_abstain"]["expected_coverage_delta"] > 0
                  for item in baseline["rows"])
    ranker = sum(item["deterministic_ranker"]["oracle_correct"]
                 and item["deterministic_ranker"]["expected_coverage_delta"] > 0
                 for item in baseline["rows"])
    selectable = sum(item.get("expected_coverage_delta", 0) > 0 for item in oracle["rows"])
    state["metrics"] = {
        "completed_rows": len(rows), "frozen_rows": total,
        "decision_accuracy": correct / total,
        "provider_only_decision_accuracy": provider_correct / total,
        "provider_protocol_rate": valid / attempts if attempts else 0.0,
        "semantic_guard_attempts": guarded_attempts,
        "semantic_guard_rate": guarded_attempts / attempts if attempts else 0.0,
        "llm_oracle_correct_closure": llm_closure,
        "deterministic_abstain_closure": abstain,
        "deterministic_ranker_closure": ranker,
        "best_deterministic_baseline_closure": max(abstain, ranker),
        "selectable_obligations": selectable,
        "net_closure_gain": (llm_closure - max(abstain, ranker)) / selectable if selectable else 0.0,
    }
    complete = len(rows) == total
    state["gate_result"] = "pass" if complete and state["metrics"]["decision_accuracy"] >= .9 \
        and state["metrics"]["net_closure_gain"] >= .2 else "fail" if complete else "in_progress"


def _empty_report(model):
    return {
        "schema_version": "m15-llm-evaluation-v1", "model": model,
        "oracle_sha256": _file_digest(ORACLE_PATH),
        "baseline_sha256": _file_digest(BASELINE_PATH), "rows": [],
        "gate_result": "in_progress",
    }


def _verify_resume(state):
    if state.get("oracle_sha256") != _file_digest(ORACLE_PATH):
        raise ValueError("frozen oracle digest mismatch")
    if state.get("baseline_sha256") != _file_digest(BASELINE_PATH):
        raise ValueError("frozen baseline digest mismatch")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _provider_outcome(raw: str, row: dict) -> tuple[str, str]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return "provider_failure", ""
    decision = str(payload.get("decision", "provider_failure"))
    if decision != "select":
        return decision, ""
    alias = str(payload.get("candidate_id", ""))
    match = re.fullmatch(r"choice_(\d+)", alias)
    index = int(match.group(1)) - 1 if match else -1
    candidates = list(row.get("candidates") or [])
    if not 0 <= index < len(candidates):
        return "provider_failure", ""
    return decision, str(candidates[index].get("payload_digest", ""))


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
