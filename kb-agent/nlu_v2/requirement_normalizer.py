"""Conservative RequirementIR normalization with lossless source ownership."""
from __future__ import annotations

import re
from collections import defaultdict

from .models import ClauseGraph, RequirementSpec, SourceDemand, SourceSpan


_SEMANTIC_OWNERS = {
    # Only node families that carry an explicit Boolean condition or set edge
    # may absorb a connector.  Analytic/output connectors remain standalone
    # until their dataflow edge is materialized; absorbing them early would
    # make a calculation responsible for proving unrelated source punctuation.
    "predicate", "event", "sequence_event", "cumulative_duration",
    "set_operation", "comparison",
}


class RequirementNormalizer:
    """Collapse evidence-level cues without discarding their provenance.

    SourceDemand remains lossless.  This pass only changes which semantic
    Requirement owns a demand and is deliberately narrower than extraction.
    """

    def normalize(self, query: str, graph: ClauseGraph,
                  requirements: list[RequirementSpec],
                  demands: list[SourceDemand]) -> list[RequirementSpec]:
        items = list(requirements)
        aliases: dict[str, str] = {}
        nodes = {item.clause_id: item for item in graph.nodes}

        items = self._merge_duration_fragments(query, items, demands, nodes, aliases)
        items = self._merge_result_references(items, nodes, aliases)
        items = self._absorb_boolean_cues(items, aliases)
        items = self._merge_same_clause_analytics(items, aliases)
        items = self._merge_generic_calculation_cues(items, aliases)
        items = self._merge_same_clause_outputs(items, aliases)
        self._normalize_analytic_bindings(items, demands)
        self._link_event_predicates(items, demands)
        self._link_set_producers(query, items)
        self._link_duration_producers(query, items)
        self._rewrite_requirement_edges(items, aliases)
        self._reduce_transitive_dependencies(items)
        self._remap_demands(demands, aliases)
        return items

    @staticmethod
    def _link_event_predicates(items: list[RequirementSpec],
                               demands: list[SourceDemand]) -> None:
        """Attach only predicates fully contained by one event source span."""
        demand_by_id = {item.demand_id: item for item in demands}
        predicates = [item for item in items if item.requirement_type == "predicate"]
        for event in items:
            if event.requirement_type not in {"sequence_event", "event"} or event.span is None:
                continue
            contained = [
                item for item in predicates if item.span is not None
                and event.span.start <= item.span.start
                and item.span.end <= event.span.end
            ]
            if len(contained) != 1:
                continue
            predicate = contained[0]
            event.dependencies = sorted(set(event.dependencies) | {predicate.requirement_id})
            source_ids = set(event.expected_attributes.get("source_demand_ids", []))
            predicate_ids = set(predicate.expected_attributes.get("source_demand_ids", []))
            event.expected_attributes["source_demand_ids"] = sorted(source_ids | predicate_ids)
            for demand_id in predicate_ids:
                demand = demand_by_id.get(demand_id)
                if demand is not None:
                    demand.mapped_requirement_ids = sorted(
                        set(demand.mapped_requirement_ids) | {event.requirement_id}
                    )

    @staticmethod
    def _link_set_producers(query: str, items: list[RequirementSpec]) -> None:
        """Bind an explicit two-party set cue to exactly two prior event producers."""
        outputs = [item for item in items if item.requirement_type == "output"]
        by_id = {item.requirement_id: item for item in items}
        for operation in items:
            if operation.requirement_type != "set_operation" or operation.span is None:
                continue
            prior_outputs = [
                item for item in outputs if item.span is not None
                and item.span.end <= operation.span.start
            ]
            if not prior_outputs:
                continue
            owner = max(prior_outputs, key=_end)
            context = query[owner.span.start:operation.span.end]
            if not re.search(r"(?:两类|两个|二者|双方).{0,24}(?:交集|并集|差集|重合)", context):
                continue
            producers = [
                dependency for dependency in owner.dependencies
                if dependency in by_id and by_id[dependency].requirement_type in {
                    "sequence_event", "event", "derived_projection",
                }
            ]
            if len(set(producers)) != 2:
                continue
            operation.dependencies = sorted(set(operation.dependencies) | set(producers))
            operation.expected_inputs = sorted(
                set(operation.expected_inputs)
                | {f"requirement:{item}" for item in producers}
            )

    @staticmethod
    def _link_duration_producers(query: str, items: list[RequirementSpec]) -> None:
        """Lower an explicit set-duration phrase to relation<day,duration>."""
        sets = [item for item in items if item.requirement_type == "set_operation" and item.span]
        by_id = {item.requirement_id: item for item in items}
        for duration in items:
            if duration.requirement_type != "cumulative_duration" or duration.span is None:
                continue
            producer = next((
                by_id[item] for item in duration.dependencies
                if item in by_id and by_id[item].requirement_type == "set_operation"
            ), None)
            if producer is None:
                candidates = [item for item in sets if item.span.end <= duration.span.start]
                if not candidates:
                    continue
                producer = max(candidates, key=_end)
                context = query[producer.span.start:duration.span.end]
                if not re.search(r"(?:交集|并集|差集|重合).{0,16}(?:总|累计|合计)时长", context):
                    continue
            # The set result is the sole executable input. Presentation and
            # anaphoric dependencies remain independently validated but must
            # not become duration dataflow parents.
            duration.dependencies = [producer.requirement_id]
            duration.expected_inputs = sorted(
                set(duration.expected_inputs) | {f"requirement:{producer.requirement_id}"}
            )
            duration.expected_attributes["expected_output_type"] = "number"
            duration.expected_output_shape = "relation"
            duration.expected_output_grain = "day"
            duration.expected_output_fields = ["cumulative_duration"]
            duration.expected_scope = "calendar_day"

    @staticmethod
    def _normalize_analytic_bindings(items: list[RequirementSpec],
                                     demands: list[SourceDemand]) -> None:
        """Restore aggregate roles from the operator's exact dependency anchors."""
        demand_by_id = {item.demand_id: item for item in demands}
        requirement_by_id = {item.requirement_id: item for item in items}
        for requirement in items:
            if requirement.requirement_type not in {"aggregate", "scoped_aggregate"}:
                continue
            source_ids = set(requirement.expected_attributes.get("source_demand_ids", []))
            operators = [
                demand_by_id[item] for item in source_ids if item in demand_by_id
                and demand_by_id[item].demand_type == "operator"
                and requirement.span is not None
                and demand_by_id[item].span.start == requirement.span.start
                and demand_by_id[item].span.end == requirement.span.end
            ]
            field_demands = []
            if len(operators) == 1:
                field_demands = [
                    demand_by_id[item] for item in operators[0].dependencies
                    if item in demand_by_id and demand_by_id[item].demand_type == "field"
                ]
            if len(field_demands) == 1:
                field = field_demands[0]
                field_text = str(field.attributes.get("field_text", field.text.strip("`")))
                requirement.expected_attributes["input_demand_ids"] = [field.demand_id]
                requirement.expected_inputs = [f"field:{field_text}"]
                function = next(iter(
                    requirement.expected_attributes.get("expected_functions", [])
                ), str(requirement.expected_attributes.get("function", "aggregate")))
                requirement.expected_output_fields = [f"{function}:{field_text}"]
                if re.match(r"(?:每日|日均|当日|当天)", field.text):
                    requirement.expected_output_grain = "day"
            if any(
                dependency in requirement_by_id
                and requirement_by_id[dependency].requirement_type == "set_operation"
                for dependency in requirement.dependencies
            ):
                requirement.expected_scope = "relation"
                requirement.expected_output_shape = "relation"

    @staticmethod
    def _merge_duration_fragments(query: str, items: list[RequirementSpec],
                                  demands: list[SourceDemand], nodes: dict,
                                  aliases: dict[str, str]) -> list[RequirementSpec]:
        demand_by_id = {item.demand_id: item for item in demands}
        ordered = sorted(items, key=_start)
        removed: set[str] = set()
        for fragment in ordered:
            if fragment.requirement_type != "sequence_event" or "持续" not in fragment.text:
                continue
            node = nodes.get(fragment.clause_id)
            if node is None or not re.match(r"\s*(?:且|并且)?\s*持续", node.text):
                continue
            owners = [
                item for item in ordered
                if item.requirement_id not in removed
                and item.requirement_type == "sequence_event"
                and item.requirement_id != fragment.requirement_id
                and _end(item) <= _start(fragment)
                and "持续" not in item.text
            ]
            if not owners:
                continue
            owner = max(owners, key=_end)
            if _start(fragment) - _end(owner) > 24:
                continue
            _merge_requirement(owner, fragment)
            owned_demands = [
                demand_by_id.get(value)
                for value in owner.expected_attributes.get("source_demand_ids", [])
            ]
            span_end = max(
                [item.span.end for item in owned_demands if item is not None]
                + [_end(fragment)],
            )
            if owner.span:
                owner.span = SourceSpan(owner.span.start, span_end, query[owner.span.start:span_end])
                owner.text = owner.span.text
            owner.expected_attributes["duration_slot_attached"] = True
            aliases[fragment.requirement_id] = owner.requirement_id
            removed.add(fragment.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _merge_result_references(items: list[RequirementSpec], nodes: dict,
                                 aliases: dict[str, str]) -> list[RequirementSpec]:
        """Merge phrases such as '并集日期中' into the producing set requirement."""
        ordered = sorted(items, key=_start)
        removed: set[str] = set()
        for item in ordered:
            if item.requirement_type != "set_operation" or item.requirement_id in removed:
                continue
            node = nodes.get(item.clause_id)
            if node is None or not re.search(r"(?:交集|并集|差集)(?:日期|时间段|区间)?中", node.text):
                continue
            prior = [
                candidate for candidate in ordered
                if candidate.requirement_type == "set_operation"
                and candidate.requirement_id not in removed
                and candidate.requirement_id != item.requirement_id
                and candidate.text == item.text and _end(candidate) <= _start(item)
            ]
            if not prior:
                continue
            owner = max(prior, key=_end)
            _merge_requirement(owner, item)
            aliases[item.requirement_id] = owner.requirement_id
            removed.add(item.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _absorb_boolean_cues(items: list[RequirementSpec],
                             aliases: dict[str, str]) -> list[RequirementSpec]:
        ordered = sorted(items, key=_start)
        removed: set[str] = set()
        for cue in ordered:
            if cue.requirement_type != "boolean_logic":
                continue
            candidates = [
                item for item in ordered
                if item.requirement_id != cue.requirement_id
                and item.requirement_id not in removed
                and item.requirement_type in _SEMANTIC_OWNERS
                and (item.clause_id == cue.clause_id or _distance(item, cue) <= 32)
            ]
            if not candidates:
                continue
            owner = min(
                candidates,
                key=lambda item: (
                    0 if item.clause_id == cue.clause_id else 1,
                    _distance(item, cue),
                    0 if item.role == "constraint" else 1,
                ),
            )
            relation = str(cue.expected_attributes.get("relation", "and"))
            relations = list(owner.expected_attributes.get("boolean_relations", []))
            relations.append({"relation": relation, "span": cue.span.to_dict() if cue.span else None})
            owner.expected_attributes["boolean_relations"] = relations
            _merge_requirement(owner, cue)
            aliases[cue.requirement_id] = owner.requirement_id
            removed.add(cue.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _merge_same_clause_outputs(items: list[RequirementSpec],
                                   aliases: dict[str, str]) -> list[RequirementSpec]:
        groups: dict[str, list[RequirementSpec]] = defaultdict(list)
        for item in items:
            if item.requirement_type == "output":
                groups[item.clause_id].append(item)
        removed: set[str] = set()
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort(key=_start)
            owner = group[0]
            for duplicate in group[1:]:
                _merge_requirement(owner, duplicate)
                aliases[duplicate.requirement_id] = owner.requirement_id
                removed.add(duplicate.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _merge_generic_calculation_cues(items: list[RequirementSpec],
                                        aliases: dict[str, str]) -> list[RequirementSpec]:
        """Fold a bare '计算' cue into one exact same-clause analytic target."""
        removed: set[str] = set()
        for cue in items:
            if (cue.requirement_type != "calculation"
                    or cue.operator_family not in {"", "calculation"}
                    or not re.fullmatch(r"\s*(?:计算|求|统计)\s*", cue.text)):
                continue
            targets = [
                item for item in items
                if item.requirement_id != cue.requirement_id
                and item.requirement_type in {"aggregate", "scoped_aggregate", "calculation"}
                and item.operator_family not in {"", "calculation"}
                and item.clause_id == cue.clause_id
            ]
            if len(targets) != 1:
                continue
            target = targets[0]
            _merge_requirement(target, cue)
            aliases[cue.requirement_id] = target.requirement_id
            removed.add(cue.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _merge_same_clause_analytics(items: list[RequirementSpec],
                                     aliases: dict[str, str]) -> list[RequirementSpec]:
        """Unify a broad scoped cue with its exact analytic operator demand.

        A clause such as ``这些日期中,效率的日平均值`` creates one broad
        scoped obligation and one exact ``avg`` demand.  They are merged only
        when the exact operator is inside the same ClauseGraph node and source
        span, avoiding any nearest-field inference.
        """
        ordered = sorted(items, key=_start)
        by_id = {item.requirement_id: item for item in ordered}
        removed: set[str] = set()
        for owner in ordered:
            if owner.requirement_type != "scoped_aggregate" or owner.span is None:
                continue
            candidates = [
                item for item in ordered
                if item.requirement_type == "aggregate"
                and item.requirement_id not in removed
                and item.clause_id == owner.clause_id
                and item.span is not None
                and owner.span.start <= item.span.start < owner.span.end + 4
            ]
            if len(candidates) != 1:
                continue
            exact = candidates[0]
            _merge_requirement(owner, exact)
            if any(
                by_id.get(dependency) is not None
                and by_id[dependency].requirement_type == "set_operation"
                for dependency in owner.dependencies
            ):
                owner.expected_scope = "relation"
                owner.expected_output_shape = "relation"
            aliases[exact.requirement_id] = owner.requirement_id
            removed.add(exact.requirement_id)
        return [item for item in items if item.requirement_id not in removed]

    @staticmethod
    def _rewrite_requirement_edges(items: list[RequirementSpec], aliases: dict[str, str]) -> None:
        def canonical(value: str) -> str:
            seen = set()
            while value in aliases and value not in seen:
                seen.add(value)
                value = aliases[value]
            return value

        valid = {item.requirement_id for item in items}
        for item in items:
            item.dependencies = sorted({
                canonical(value) for value in item.dependencies
                if canonical(value) in valid and canonical(value) != item.requirement_id
            })
            plain = {value for value in item.expected_inputs if not value.startswith("requirement:")}
            refs = {
                "requirement:" + canonical(value.removeprefix("requirement:"))
                for value in item.expected_inputs if value.startswith("requirement:")
                and canonical(value.removeprefix("requirement:")) in valid
                and canonical(value.removeprefix("requirement:")) != item.requirement_id
            }
            item.expected_inputs = sorted(plain | refs)

    @staticmethod
    def _reduce_transitive_dependencies(items: list[RequirementSpec]) -> None:
        """Keep direct Requirement DAG edges while preserving reachability."""
        graph = {item.requirement_id: set(item.dependencies) for item in items}

        def reaches(start: str, target: str) -> bool:
            pending = list(graph.get(start, ()))
            seen = set()
            while pending:
                current = pending.pop()
                if current == target:
                    return True
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(graph.get(current, ()))
            return False

        for item in items:
            direct = set(item.dependencies)
            redundant = {
                candidate for candidate in direct
                if any(other != candidate and reaches(other, candidate) for other in direct)
            }
            item.dependencies = sorted(direct - redundant)
            item.expected_inputs = sorted({
                value for value in item.expected_inputs
                if not value.startswith("requirement:")
                or value.removeprefix("requirement:") not in redundant
            })

    @staticmethod
    def _remap_demands(demands: list[SourceDemand], aliases: dict[str, str]) -> None:
        for demand in demands:
            mapped = []
            for requirement_id in demand.mapped_requirement_ids:
                seen = set()
                while requirement_id in aliases and requirement_id not in seen:
                    seen.add(requirement_id)
                    requirement_id = aliases[requirement_id]
                mapped.append(requirement_id)
            demand.mapped_requirement_ids = sorted(set(mapped))
            if mapped:
                demand.consumption_status = "mapped_to_requirement"


def _merge_requirement(owner: RequirementSpec, other: RequirementSpec) -> None:
    owner.dependencies = sorted(set(owner.dependencies) | set(other.dependencies))
    owner.expected_inputs = sorted(set(owner.expected_inputs) | set(other.expected_inputs))
    owner.expected_output_fields = sorted(
        set(owner.expected_output_fields) | set(other.expected_output_fields)
    )
    owner.cardinality = max(owner.cardinality, other.cardinality)
    if not owner.operator_family or owner.operator_family in {"aggregate", "calculation"}:
        owner.operator_family = other.operator_family or owner.operator_family
    if not owner.expected_scope:
        owner.expected_scope = other.expected_scope
    if not owner.expected_output_shape:
        owner.expected_output_shape = other.expected_output_shape
    if not owner.expected_output_grain:
        owner.expected_output_grain = other.expected_output_grain
    attrs = dict(owner.expected_attributes)
    for key, value in other.expected_attributes.items():
        if key == "source_demand_ids":
            continue
        if key not in attrs or attrs[key] in (None, "", []):
            attrs[key] = value
    attrs["source_demand_ids"] = sorted(set(
        owner.expected_attributes.get("source_demand_ids", [])
    ) | set(other.expected_attributes.get("source_demand_ids", [])))
    owner.expected_attributes = attrs


def _start(item: RequirementSpec) -> int:
    return item.span.start if item.span else -1


def _end(item: RequirementSpec) -> int:
    return item.span.end if item.span else -1


def _distance(left: RequirementSpec, right: RequirementSpec) -> int:
    if _end(left) <= _start(right):
        return _start(right) - _end(left)
    if _end(right) <= _start(left):
        return _start(left) - _end(right)
    return 0
