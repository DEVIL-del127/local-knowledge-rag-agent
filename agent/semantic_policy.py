from __future__ import annotations

import re
from typing import Any

from agent.retrieval_models import SemanticDecision


_DOCUMENT_GOAL = re.compile(r"(?:找|查|检索|搜索|哪篇|哪些).{0,12}(?:文档|论文|资料|文件|知识库)")


class SemanticAdmissionPolicy:
    """Classify semantics only; this policy never builds executable filters."""

    @staticmethod
    def decide(ir: Any, query: str) -> SemanticDecision:
        has_analytics = any((
            ir.events,
            ir.window_aggregates,
            ir.cumulative_durations,
            ir.cumulative_counts,
            ir.aggregates,
            ir.calculations,
            ir.set_operations,
        ))
        if has_analytics and not _DOCUMENT_GOAL.search(query):
            return SemanticDecision.UNSUPPORTED_ANALYTICS
        if ir.unsatisfiable:
            return SemanticDecision.BLOCKED
        if ir.ambiguities or any(
            getattr(item, "status", "unresolved") != "resolved"
            for item in ir.references
        ):
            return SemanticDecision.CLARIFY
        return SemanticDecision.PASS_THROUGH
