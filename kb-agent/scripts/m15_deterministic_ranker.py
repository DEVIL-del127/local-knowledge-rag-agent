"""Frozen, oracle-blind lexical baseline for M1.5 candidate-choice evaluation."""
from __future__ import annotations

import re
from typing import Any


BASELINE_VERSION = "m15-lexical-ranker-v1"
MIN_SCORE = 0.35
MIN_MARGIN = 0.10


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    ascii_tokens = set(re.findall(r"[a-z_][a-z0-9_]*|[-+]?\d+(?:\.\d+)?", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    grams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    return ascii_tokens | grams


def _score(source_clause: str, summary: str) -> float:
    source = _tokens(source_clause)
    candidate = _tokens(summary)
    if not source or not candidate:
        return 0.0
    return len(source & candidate) / len(source | candidate)


def rank(source_clause: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only a unique lexical winner above frozen score and margin thresholds."""
    scored = [
        {
            "alias": str(item["alias"]),
            "payload_digest": str(item["payload_digest"]),
            "score": round(_score(source_clause, str(item["summary"])), 6),
        }
        for item in candidates
    ]
    scored.sort(key=lambda item: (-item["score"], item["alias"]))
    if not scored:
        return {"decision": "abstain", "reason": "empty_menu", "scores": []}
    best = scored[0]
    runner_up = scored[1]["score"] if len(scored) > 1 else 0.0
    if best["score"] < MIN_SCORE:
        return {"decision": "abstain", "reason": "score_below_threshold", "scores": scored}
    if best["score"] - runner_up < MIN_MARGIN:
        return {"decision": "abstain", "reason": "margin_below_threshold", "scores": scored}
    return {
        "decision": "select",
        "candidate_alias": best["alias"],
        "payload_digest": best["payload_digest"],
        "reason": "unique_lexical_winner",
        "scores": scored,
    }
