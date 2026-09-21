"""Read-only event producer/consumer closure analysis."""
from __future__ import annotations

from dataclasses import dataclass

from .models import UnderstandingIR
from .semantic_targets import OutputLineageIndex


_CLOSURE_REQUIREMENTS = {
    "sequence_event", "event", "derived_projection", "set_operation",
    "aggregate", "scoped_aggregate", "calculation", "output",
}


@dataclass(frozen=True, slots=True)
class EventClosureReport:
    required_requirement_ids: tuple[str, ...]
    covered_requirement_ids: tuple[str, ...]
    producer_consumer_edges: tuple[tuple[str, str], ...]
    unresolved_dependencies: tuple[str, ...]
    structure_status: str
    binding_status: str
    execution_status: str
    terminal_output_refs: tuple[str, ...]


class EventClosureAnalyzer:
    """Prove typed event paths without adding or mutating IR nodes."""

    def analyze(self, ir: UnderstandingIR) -> EventClosureReport:
        lineage = OutputLineageIndex.build(ir)
        coverage = {item.requirement_id: item.status for item in ir.coverage}
        required = tuple(
            item.requirement_id for item in ir.requirements
            if item.requirement_type in _CLOSURE_REQUIREMENTS
        )
        covered = tuple(item for item in required if coverage.get(item) == "satisfied")
        edges = tuple(sorted({
            (parent, entry.ref_id)
            for entry in lineage.entries.values() for parent in entry.input_refs
        }))
        known = set(lineage.entries)
        unresolved = {
            f"output_ref:{parent}"
            for entry in lineage.entries.values()
            for parent in entry.input_refs
            if parent not in known and parent != "source_relation.rows"
        }
        unresolved.update(
            f"requirement:{item}" for item in required if coverage.get(item) != "satisfied"
        )

        consumed = {source for source, _ in edges}
        terminal = tuple(sorted(
            ref_id for ref_id in known
            if ref_id not in consumed and lineage.entries[ref_id].semantic_role in {
                "event_interval", "date", "set", "aggregate", "calculation", "comparison",
            }
        ))
        event_roots = {
            item.output_ref.ref_id for item in ir.events if item.output_ref is not None
        }
        terminal_set = set(terminal)
        requires_downstream = any(
            item.requirement_type in {
                "derived_projection", "set_operation", "aggregate",
                "scoped_aggregate", "calculation",
            }
            for item in ir.requirements
        )
        disconnected = []
        for root in sorted(event_roots):
            reachable_terminals = _reachable({root}, edges) & terminal_set
            # An event may itself be the requested terminal for a simple
            # interval query.  Once the request explicitly requires a
            # projection, set or calculation, an unconsumed event is not a
            # closed producer path.
            if requires_downstream:
                reachable_terminals.discard(root)
            if not reachable_terminals:
                disconnected.append(root)
        unresolved.update(f"disconnected_event:{item}" for item in disconnected)
        if required and not unresolved and terminal:
            structure_status = "complete"
        elif required or event_roots:
            structure_status = "partial"
        else:
            structure_status = "unsupported"

        unresolved_symbols = any(
            item.code in {"unknown_field", "source_unresolved", "unresolved_symbol"}
            for item in ir.unresolved
        ) or bool(ir.schema_hypotheses)
        if structure_status == "complete" and not unresolved_symbols and ir.binding_status == "bound":
            binding_status = "complete"
        elif structure_status == "complete":
            binding_status = "pending"
        else:
            binding_status = "blocked"
        execution_status = "ready" if (
            structure_status == "complete" and ir.execution_status == "ready"
        ) else "blocked"
        return EventClosureReport(
            required, covered, edges, tuple(sorted(unresolved)), structure_status,
            binding_status, execution_status, terminal,
        )


def _reachable(roots: set[str], edges: tuple[tuple[str, str], ...]) -> set[str]:
    if not roots:
        return set()
    forward: dict[str, set[str]] = {}
    for source, target in edges:
        forward.setdefault(source, set()).add(target)
    seen = set(roots)
    pending = list(roots)
    while pending:
        current = pending.pop()
        for target in forward.get(current, ()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen
