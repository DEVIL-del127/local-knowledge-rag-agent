"""Optional vector candidate interface; never required for deterministic operation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .catalog import CatalogSnapshot
from .models import CandidateRef


class VectorCandidateProvider(Protocol):
    def candidates(self, query: str, catalog: CatalogSnapshot,
                   limit: int = 5) -> list[CandidateRef]:
        ...


@dataclass(slots=True)
class NullVectorCandidateProvider:
    def candidates(self, query: str, catalog: CatalogSnapshot,
                   limit: int = 5) -> list[CandidateRef]:
        return []


class CallableVectorCandidateProvider:
    """Adapter for the existing Embedder/ES stack without importing it at startup."""

    def __init__(self, resolver):
        self.resolver = resolver

    def candidates(self, query: str, catalog: CatalogSnapshot,
                   limit: int = 5) -> list[CandidateRef]:
        rows = self.resolver(query=query, catalog=catalog, limit=limit)
        result = []
        for row in rows or []:
            identifier = str(row.get("identifier", ""))
            if identifier and (catalog.field(identifier) or catalog.source(identifier)):
                result.append(CandidateRef(
                    identifier=identifier,
                    score=float(row["score"]) if row.get("score") is not None else None,
                    source="vector",
                    status="candidate",
                ))
        return result[:limit]
