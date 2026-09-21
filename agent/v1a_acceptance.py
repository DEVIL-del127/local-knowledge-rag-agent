from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CORE_STRATA = {
    "single_turn": 100,
    "multi_turn": 50,
    "multitask_negation": 25,
    "out_of_scope_ambiguous": 25,
}
ALLOWED_OUTCOMES = {"auto_release", "clarify", "upgrade", "reject"}


class AcceptanceDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AcceptanceSummary:
    corpus_digest: str
    total: int
    strata: dict[str, int]
    automatic_releases: int
    automatic_release_errors: int
    automatic_release_error_rate: float
    critical_errors: int
    unnecessary_clarifications: int
    incorrect_rejections: int
    full_multiturn_completions: int
    correct_automatic_releases: int
    low_cost_p95_ms: float | None
    upgrade_p50_ms: float | None
    upgrade_p95_ms: float | None
    end_to_end_p50_ms: float | None
    end_to_end_p95_ms: float | None
    automatic_release_error_wilson95: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AcceptanceDataError(f"invalid JSON at line {number}") from exc
        if not isinstance(value, dict):
            raise AcceptanceDataError(f"line {number} must be an object")
        rows.append(value)
    return rows


def validate_core_corpus(rows: Iterable[dict[str, Any]]) -> str:
    values = list(rows)
    if len(values) != 200:
        raise AcceptanceDataError(f"core corpus must contain exactly 200 cases, got {len(values)}")
    ids = [str(item.get("id") or "") for item in values]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise AcceptanceDataError("case ids must be non-empty and unique")
    strata = Counter(str(item.get("stratum") or "") for item in values)
    if dict(strata) != CORE_STRATA:
        raise AcceptanceDataError(f"invalid stratum counts: {dict(strata)}")
    for item in values:
        if item.get("split") != "acceptance":
            raise AcceptanceDataError(f"{item['id']}: split must be acceptance")
        if not item.get("template_group"):
            raise AcceptanceDataError(f"{item['id']}: template_group is required")
        if not str(item.get("query") or "").strip():
            raise AcceptanceDataError(f"{item['id']}: query is required")
        expected = item.get("expected")
        if not isinstance(expected, dict) or expected.get("outcome") not in ALLOWED_OUTCOMES:
            raise AcceptanceDataError(f"{item['id']}: expected outcome is invalid")
        if item["stratum"] == "multi_turn":
            if not item.get("conversation_id") or not isinstance(item.get("precondition"), dict):
                raise AcceptanceDataError(f"{item['id']}: multi-turn case needs conversation and precondition")
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def wilson_interval(errors: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = errors / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_run(corpus: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> AcceptanceSummary:
    cases = list(corpus)
    digest = validate_core_corpus(cases)
    expected = {str(item["id"]): item for item in cases}
    actual = list(predictions)
    by_id = {str(item.get("case_id") or ""): item for item in actual}
    if len(by_id) != len(actual) or set(by_id) != set(expected):
        raise AcceptanceDataError("predictions must contain every case exactly once")
    automatic_releases = automatic_errors = critical = unnecessary = rejected = completed = correct_release = 0
    latencies: list[float] = []
    upgrade_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    for case_id, case in expected.items():
        row = by_id[case_id]
        outcome = str(row.get("outcome") or "")
        if outcome not in ALLOWED_OUTCOMES:
            raise AcceptanceDataError(f"{case_id}: prediction outcome is invalid")
        wanted = str(case["expected"]["outcome"])
        released = outcome == "auto_release"
        automatic_releases += int(released)
        correct = bool(row.get("correct", outcome == wanted))
        automatic_errors += int(released and not correct)
        correct_release += int(released and correct)
        critical += int(bool(row.get("critical_error")))
        unnecessary += int(bool(row.get("unnecessary_clarification")))
        rejected += int(bool(row.get("incorrect_rejection")))
        completed += int(
            case["stratum"] == "multi_turn"
            and row.get("full_task_completed") is True
        )
        if row.get("low_cost_latency_ms") is not None:
            latencies.append(float(row["low_cost_latency_ms"]))
        if row.get("upgrade_latency_ms") is not None:
            upgrade_latencies.append(float(row["upgrade_latency_ms"]))
        if row.get("end_to_end_latency_ms") is not None:
            end_to_end_latencies.append(float(row["end_to_end_latency_ms"]))
    return AcceptanceSummary(
        corpus_digest=digest,
        total=len(cases),
        strata=dict(Counter(str(item["stratum"]) for item in cases)),
        automatic_releases=automatic_releases,
        automatic_release_errors=automatic_errors,
        automatic_release_error_rate=(automatic_errors / automatic_releases
                                      if automatic_releases else 0.0),
        critical_errors=critical,
        unnecessary_clarifications=unnecessary,
        incorrect_rejections=rejected,
        full_multiturn_completions=completed,
        correct_automatic_releases=correct_release,
        low_cost_p95_ms=_percentile(latencies, 0.95),
        upgrade_p50_ms=_percentile(upgrade_latencies, 0.50),
        upgrade_p95_ms=_percentile(upgrade_latencies, 0.95),
        end_to_end_p50_ms=_percentile(end_to_end_latencies, 0.50),
        end_to_end_p95_ms=_percentile(end_to_end_latencies, 0.95),
        automatic_release_error_wilson95=wilson_interval(automatic_errors, automatic_releases),
    )


def evaluate_category_evidence(rows: Iterable[dict[str, Any]], *, minimum_releases: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category") or "")
        if not category:
            raise AcceptanceDataError("category evidence requires category")
        grouped.setdefault(category, []).append(row)
    categories = {}
    for category, values in grouped.items():
        released = [item for item in values if item.get("released")]
        correct = sum(bool(item.get("correct")) for item in released)
        critical = sum(bool(item.get("critical_error")) for item in values)
        precision = correct / len(released) if released else 0.0
        checks = {
            "minimum_releases": len(released) >= minimum_releases,
            "precision_gte_98pct": precision >= 0.98,
            "critical_errors_zero": critical == 0,
        }
        categories[category] = {
            "passed": all(checks.values()), "checks": checks,
            "samples": len(values), "released": len(released), "correct": correct,
            "precision": precision,
            "precision_wilson95": wilson_interval(correct, len(released)),
            "critical_errors": critical,
        }
    return {"passed": bool(categories) and all(item["passed"] for item in categories.values()),
            "minimum_releases": minimum_releases, "categories": categories}


def compare_candidate(baseline: AcceptanceSummary, candidate: AcceptanceSummary) -> dict[str, Any]:
    checks = {
        "same_corpus": baseline.corpus_digest == candidate.corpus_digest,
        "automatic_release_error_rate_lte_2pct": candidate.automatic_release_error_rate <= 0.02,
        "critical_errors_zero": candidate.critical_errors == 0,
        "unnecessary_clarifications_not_worse":
            candidate.unnecessary_clarifications <= baseline.unnecessary_clarifications,
        "incorrect_rejections_not_worse": candidate.incorrect_rejections <= baseline.incorrect_rejections,
        "multiturn_completions_not_worse":
            candidate.full_multiturn_completions >= baseline.full_multiturn_completions,
        "added_correct_coverage":
            candidate.correct_automatic_releases > baseline.correct_automatic_releases,
        "warm_low_cost_p95_lte_1000ms":
            candidate.low_cost_p95_ms is not None and candidate.low_cost_p95_ms <= 1000.0,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "baseline": baseline.to_dict(), "candidate": candidate.to_dict()}
