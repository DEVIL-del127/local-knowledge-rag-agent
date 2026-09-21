from __future__ import annotations

import json
from pathlib import Path

from nlu_v2.llm_extractor import _guard_candidate_semantics
from nlu_v2.semantic_candidates import CandidateMenu, SemanticCandidate


ORACLE = Path(__file__).parent / "fixtures" / "m15_llm_choice_oracle.json"


def _menu(row: dict, *, reverse: bool = False) -> CandidateMenu:
    values = list(reversed(row["candidates"])) if reverse else list(row["candidates"])
    candidates = tuple(
        SemanticCandidate(
            candidate_id=item["payload_digest"],
            operation_type="candidate_select",
            semantic_payload=item["semantic_payload"],
            evidence_summary=item["summary"],
        )
        for item in values
    )
    return CandidateMenu(
        gap_id="guard-test",
        requirement_id="guard-test",
        source_clause=row["source_clause"],
        target_summary={"family": row["family"]},
        candidates=candidates,
        dispatch="llm_choice",
        blocked_by=(),
        source_clause_digest="source",
        target_contract_digest="target",
        base_lineage_digest="lineage",
        candidate_menu_digest="menu",
    )


def test_semantic_guard_matches_frozen_cross_family_contract():
    rows = json.loads(ORACLE.read_text(encoding="utf-8"))["rows"]
    failures = []
    for row in rows:
        guarded = _guard_candidate_semantics(_menu(row))
        if guarded is None:
            failures.append((row["oracle_id"], "not_recognized"))
            continue
        decision, _ = guarded
        actual = (decision.decision, decision.candidate_id)
        expected = (
            row["expected_decision"],
            row.get("expected_selected_payload_digest", ""),
        )
        if actual != expected:
            failures.append((row["oracle_id"], expected, actual))
    assert not failures, failures


def test_semantic_guard_is_invariant_to_candidate_order():
    rows = json.loads(ORACLE.read_text(encoding="utf-8"))["rows"]
    for row in rows:
        first, _ = _guard_candidate_semantics(_menu(row))
        reversed_result, _ = _guard_candidate_semantics(_menu(row, reverse=True))
        assert (first.decision, first.candidate_id) == (
            reversed_result.decision, reversed_result.candidate_id,
        ), row["oracle_id"]
