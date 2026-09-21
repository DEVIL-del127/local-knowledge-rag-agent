"""Role-aware, proof-carrying field binding views."""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import CatalogSnapshot
from .field_roles import FieldPhraseParser
from .merge import walk_predicates
from .models import SourceSpan, SymbolRef, UnderstandingIR


@dataclass(frozen=True, slots=True)
class FieldBindingProof:
    raw_span: SourceSpan | None
    cleaned_name: str
    role: str
    catalog_id: str = ""
    hypothesis_id: str = ""
    data_type: str = "unknown"
    quantity_kind: str = "unknown"
    unit_dimension: str = "unknown"
    binding_state: str = "unresolved"
    evidence_source: str = ""

    @property
    def executable(self) -> bool:
        return self.binding_state == "catalog_bound" and bool(self.catalog_id)


@dataclass(frozen=True, slots=True)
class FieldBindingReport:
    proofs: tuple[FieldBindingProof, ...]
    unresolved_count: int
    hypothesis_count: int
    executable_count: int


class FieldRoleBinder:
    """Bind only explicit or unique Catalog evidence; never promote hypotheses."""

    def bind(self, ir: UnderstandingIR, catalog: CatalogSnapshot) -> FieldBindingReport:
        symbols: list[tuple[SymbolRef, str]] = []
        for predicate in walk_predicates(ir.filters):
            if predicate.field:
                symbols.append((predicate.field, "measure"))
        for event in ir.events:
            for predicate in walk_predicates(event.condition):
                if predicate.field:
                    symbols.append((predicate.field, "measure"))
            if event.sampling.order_by:
                symbols.append((event.sampling.order_by, "order"))
            symbols.extend((item, "partition") for item in event.sampling.partition_by)
        symbols.extend((item, "output") for item in ir.projections)
        symbols.extend((item.field, "measure") for item in ir.metrics)

        proofs = []
        seen = set()
        hypotheses = {item.hypothesis_id: item for item in ir.schema_hypotheses}
        for symbol, role in symbols:
            key = (symbol.raw_name, symbol.canonical_id, role, symbol.span.start if symbol.span else -1)
            if key in seen:
                continue
            seen.add(key)
            proofs.append(self._prove(symbol, role, catalog, hypotheses))
        return FieldBindingReport(
            tuple(proofs),
            sum(item.binding_state == "unresolved" for item in proofs),
            sum(item.binding_state == "hypothesis_only" for item in proofs),
            sum(item.executable for item in proofs),
        )

    def _prove(self, symbol, role, catalog, hypotheses):
        cleaned = FieldPhraseParser.clean_field(symbol.raw_name) or symbol.raw_name.strip()
        candidate = symbol.canonical_id or ""
        if candidate and catalog.field(candidate):
            return _catalog_proof(symbol, cleaned, role, candidate, catalog, "symbol_canonical_id")
        exact = catalog.resolve_field(cleaned)
        if len(exact) == 1:
            return _catalog_proof(symbol, cleaned, role, exact[0], catalog, "unique_catalog_alias")
        hypothesis_id = candidate if candidate in hypotheses else next((
            item.hypothesis_id for item in hypotheses.values()
            if cleaned in {item.raw_name, item.normalized_name, *item.aliases}
        ), "")
        if hypothesis_id:
            hypothesis = hypotheses[hypothesis_id]
            return FieldBindingProof(symbol.span, cleaned, role, hypothesis_id=hypothesis_id,
                                     data_type=hypothesis.declared_type,
                                     binding_state="hypothesis_only", evidence_source="query_schema")
        return FieldBindingProof(symbol.span, cleaned, role, binding_state="unresolved",
                                 evidence_source="ambiguous_catalog_alias" if exact else "none")


def _catalog_proof(symbol, cleaned, role, field_id, catalog, evidence):
    field = catalog.field(field_id)
    return FieldBindingProof(
        symbol.span, cleaned, role, catalog_id=field_id,
        data_type=field.data_type, quantity_kind=_quantity_kind(field.unit),
        unit_dimension=_dimension(field.unit), binding_state="catalog_bound",
        evidence_source=evidence,
    )


def _quantity_kind(unit):
    return "absolute_temperature" if unit in {"celsius", "fahrenheit", "kelvin"} else "measure"


def _dimension(unit):
    return {
        "celsius": "temperature", "fahrenheit": "temperature", "kelvin": "temperature",
        "second": "time", "minute": "time", "hour": "time", "dB": "sound_level",
    }.get(unit, "unknown" if not unit else unit)
