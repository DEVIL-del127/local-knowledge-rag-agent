import copy

import pytest

from agent.v1a_acceptance import (
    AcceptanceDataError,
    CORE_STRATA,
    compare_candidate,
    evaluate_category_evidence,
    summarize_run,
    validate_core_corpus,
)


def _corpus():
    rows = []
    index = 0
    for stratum, count in CORE_STRATA.items():
        for number in range(count):
            index += 1
            row = {
                "id": f"A{index:03d}",
                "split": "acceptance",
                "stratum": stratum,
                "template_group": f"{stratum}-{number // 2}",
                "query": f"case {index}",
                "expected": {"outcome": "auto_release"},
            }
            if stratum == "multi_turn":
                row.update({"conversation_id": f"c{number // 2}",
                            "precondition": {"current_object_ids": ["d1"]}})
            rows.append(row)
    return rows


def _predictions(corpus, *, correct_releases=0, latency=100.0):
    values = []
    for index, item in enumerate(corpus):
        released = index < correct_releases
        values.append({
            "case_id": item["id"],
            "outcome": "auto_release" if released else "clarify",
            "correct": True,
            "critical_error": False,
            "unnecessary_clarification": False,
            "incorrect_rejection": False,
            "domain_correct": True,
            "full_task_completed": None,
            "low_cost_latency_ms": latency,
        })
    return values


def test_core_corpus_requires_exact_frozen_strata():
    corpus = _corpus()
    assert len(validate_core_corpus(corpus)) == 64
    with pytest.raises(AcceptanceDataError, match="exactly 200"):
        validate_core_corpus(corpus[:-1])
    invalid = copy.deepcopy(corpus)
    invalid[0]["split"] = "development"
    with pytest.raises(AcceptanceDataError, match="split must be acceptance"):
        validate_core_corpus(invalid)


def test_candidate_gate_requires_coverage_accuracy_integrity_and_latency():
    corpus = _corpus()
    baseline = summarize_run(corpus, _predictions(corpus))
    candidate = summarize_run(corpus, _predictions(corpus, correct_releases=10, latency=999.0))
    report = compare_candidate(baseline, candidate)
    assert report["passed"] and all(report["checks"].values())

    broken_rows = _predictions(corpus, correct_releases=10, latency=1001.0)
    broken_rows[0]["correct"] = False
    broken_rows[0]["critical_error"] = True
    broken = compare_candidate(baseline, summarize_run(corpus, broken_rows))
    assert not broken["passed"]
    assert not broken["checks"]["critical_errors_zero"]
    assert not broken["checks"]["warm_low_cost_p95_lte_1000ms"]


def test_category_evidence_enforces_sample_precision_and_critical_error_gates():
    rows = [{"category": "literature_find", "released": True, "correct": True,
             "critical_error": False} for _ in range(20)]
    assert evaluate_category_evidence(rows, minimum_releases=20)["passed"]
    rows[0]["correct"] = False
    assert not evaluate_category_evidence(rows, minimum_releases=20)["passed"]
    assert not evaluate_category_evidence(rows[:19], minimum_releases=20)["passed"]
