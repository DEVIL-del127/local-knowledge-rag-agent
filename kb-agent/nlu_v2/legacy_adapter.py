"""Acceptance-only compatibility bridge; no production entrypoint imports this module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import EngineResult


@dataclass(slots=True)
class DualRunResult:
    query: str
    v2: dict[str, Any]
    legacy: dict[str, Any] | None
    comparison: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "v2": self.v2,
            "legacy": self.legacy,
            "comparison": self.comparison,
        }


def to_legacy_route(result: EngineResult):
    """Convert only executable private-KB plans to the old RouteResult contract."""
    if not result.validation.executable or result.logical_plan.status != "ready":
        raise ValueError("only validated, ready V2 plans can be converted")
    sources = {node.source_id for node in result.logical_plan.nodes if node.source_id}
    if sources != {"private_kb"}:
        raise ValueError("legacy RouteResult only supports private_kb")

    from router import Entities, RouteResult

    temporal = result.understanding.temporal
    years = sorted({
        int(value[:4])
        for item in temporal
        for value in (item.from_value, item.to_value, item.exact)
        if value and len(value) >= 4 and value[:4].isdigit()
    })
    entities = Entities(
        year_from=years[0] if len(years) > 1 else None,
        year_to=years[-1] if len(years) > 1 else None,
        year_exact=years[0] if len(years) == 1 else None,
        year_op="between" if len(years) > 1 else ("exact" if years else None),
        granularity=temporal[0].granularity if temporal else None,
        time_ranges=[{
            "op": item.operator,
            **({"from": item.from_value} if item.from_value else {}),
            **({"to": item.to_value} if item.to_value else {}),
            **({"exact": item.exact} if item.exact else {}),
            "granularity": item.granularity,
            **({"timezone": item.timezone} if item.timezone else {}),
            "raw": item.raw,
        } for item in temporal],
    )
    return RouteResult(
        intent=result.understanding.compatibility_intent,
        confidence=result.validation.reliability,
        source="nlu_v2_legacy_adapter",
        entities=entities,
        raw_query=result.understanding.query.raw,
    )


def dual_run(query: str, engine, legacy_router=None) -> DualRunResult:
    """Run V2 and, only when explicitly supplied, the old Router."""
    v2_result = engine.analyze(query)
    legacy_result = legacy_router.route(query) if legacy_router is not None else None
    v2_summary = {
        "status": v2_result.validation.status,
        "intent": v2_result.understanding.compatibility_intent,
        "sources": sorted({
            node.source_id for node in v2_result.logical_plan.nodes if node.source_id
        }),
        "model_calls": v2_result.model_calls,
    }
    old_dict = legacy_result.to_dict() if legacy_result is not None else None
    comparison = {
        "legacy_executed": legacy_result is not None,
        "intent_equal": (
            legacy_result.intent == v2_summary["intent"] if legacy_result is not None else None
        ),
        "v2_executable": v2_result.validation.executable,
    }
    return DualRunResult(query, v2_result.to_dict(), old_dict, comparison)
