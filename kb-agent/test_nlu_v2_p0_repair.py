"""P0-9~12: RepairContract boundaries and atomic Patch behaviour."""
from __future__ import annotations

import unittest

from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.models import QueryEnvelope, SourceDemand, SourceSpan, UnderstandingIR
from nlu_v2.patch_protocol import AddAggregateOperation, AddAmbiguityOperation, GapRequest, SemanticPatchV1, apply_semantic_patch, request_ir_digest
from nlu_v2.semantic_repair import CompilationSnapshot, RepairContract, RepairUnit, contract_violation


class P0RepairContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = QueryUnderstandingEngine()
        self.catalog = self.engine.catalog_provider.snapshot()
        demand = SourceDemand("anchor_ok", "field", "证据", SourceSpan(0, 2, "证据"))
        self.ir = UnderstandingIR(QueryEnvelope("证据", "证据"), self.catalog.version,
                                  source_demands=[demand])
        self.gap = GapRequest(
            gap_id="gap_one", gap_type="missing_predicate", query_slice="证据",
            allowed_operations=["add_ambiguity"], allowed_anchor_ids=["anchor_ok"],
        )
        self.unit = RepairUnit.from_gaps([self.gap])[0]
        self.contract = RepairContract.for_unit(
            CompilationSnapshot.capture(self.ir), self.unit, "clarification", self.ir, self.catalog,
        )

    def _operation(self, anchors):
        return AddAmbiguityOperation(
            operation_type="add_ambiguity", operation_id="op_one", gap_id="gap_one",
            evidence=[{"text": "证据"}], anchor_ids=anchors,
            ambiguity_id="amb_one", kind="field", message="需要澄清",
        )

    def test_contract_rejects_missing_or_foreign_anchor(self):
        missing = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir),
                                  operations=[self._operation([])])
        foreign = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir),
                                  operations=[self._operation(["anchor_bad"])])
        self.assertEqual(contract_violation(self.contract, missing, self.ir),
                         "repair_contract_anchor_required")
        self.assertEqual(contract_violation(self.contract, foreign, self.ir),
                         "repair_contract_anchor_out_of_scope")

    def test_multi_operation_unit_rolls_back_as_a_whole(self):
        first = self._operation(["anchor_ok"])
        second = self._operation(["anchor_ok"])
        second.operation_id = "op_two"
        patch = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir), operations=[first, second])
        candidate, report = apply_semantic_patch(self.ir, patch, [self.gap], self.catalog, atomic_unit=True)
        self.assertEqual(candidate.ambiguities, [])
        self.assertEqual(report.accepted_count, 0)

    def test_repair_unit_uses_requirement_edges_not_shared_allow_lists(self):
        shared_outputs = ["source_relation.rows", "event_existing.intervals"]
        upstream = GapRequest(
            gap_id="gap_upstream", gap_type="missing_event", query_slice="子句一",
            allowed_operations=["add_event"], available_output_refs=shared_outputs,
            requirement_id="req_event", clause_id="clause_one",
        )
        downstream = GapRequest(
            gap_id="gap_downstream", gap_type="missing_set_inputs", query_slice="子句二",
            allowed_operations=["add_set_operation"], available_output_refs=shared_outputs,
            requirement_id="req_output", requirement_dependency_ids=["req_event"],
            clause_id="clause_two",
        )
        unrelated = GapRequest(
            gap_id="gap_unrelated", gap_type="missing_formula_input", query_slice="独立子句",
            allowed_operations=["add_calculation"], available_output_refs=shared_outputs,
            requirement_id="req_other", clause_id="clause_three",
        )
        units = RepairUnit.from_gaps([upstream, downstream, unrelated])
        grouped = {frozenset(item.gap_id for item in unit.gaps) for unit in units}
        self.assertIn(frozenset({"gap_upstream", "gap_downstream"}), grouped)
        self.assertIn(frozenset({"gap_unrelated"}), grouped)

    def test_transaction_rejects_multiple_unrelated_units_without_selection(self):
        first = GapRequest(
            gap_id="gap_first", gap_type="missing_predicate", query_slice="证据",
            allowed_operations=["add_ambiguity"], allowed_anchor_ids=["anchor_ok"],
            clause_id="clause_one",
        )
        second = GapRequest(
            gap_id="gap_second", gap_type="missing_predicate", query_slice="另一证据",
            allowed_operations=["add_ambiguity"], allowed_anchor_ids=["anchor_ok"],
            clause_id="clause_two",
        )
        patch = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir), operations=[])
        candidate, report = self.engine._apply_patch_transaction(
            self.ir, patch, [first, second], self.catalog,
        )
        self.assertIs(candidate, self.ir)
        self.assertEqual(report.transaction_error, "repair_unit_not_explicit")

    def test_patch_replay_is_idempotently_a_noop(self):
        patch = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir),
                               operations=[self._operation(["anchor_ok"])])
        candidate, first = apply_semantic_patch(self.ir, patch, [self.gap], self.catalog)
        _, replay = apply_semantic_patch(candidate, patch, [self.gap], self.catalog)
        self.assertEqual(first.accepted_count, 1)
        self.assertEqual(replay.accepted_count, 0)
        self.assertEqual(len(candidate.ambiguities), 1)

    def _aggregate_contract(self, *, expected_output_shape="", protected=()):
        gap = GapRequest(
            gap_id="gap_aggregate", gap_type="missing_aggregate_scope", query_slice="证据",
            allowed_catalog_symbols=["device_telemetry.temperature"],
            allowed_operations=["add_aggregate"], allowed_anchor_ids=["anchor_ok"],
        )
        unit = RepairUnit.from_gaps([gap])[0]
        return RepairContract(
            CompilationSnapshot.capture(self.ir), unit, "concrete",
            allowed_anchor_ids=("anchor_ok",),
            allowed_field_ids=("device_telemetry.temperature",),
            allowed_operator_ids=("add_aggregate",),
            allowed_unit_ids=("celsius",),
            allowed_input_refs=("source_relation.rows",),
            allowed_output_types=("number",),
            allowed_write_paths=("aggregates",),
            expected_output_shape=expected_output_shape,
            protected_rule_node_ids=tuple(protected),
        )

    @staticmethod
    def _aggregate_operation(aggregate_id="new_aggregate", *, field_id="device_telemetry.temperature",
                             unit="celsius", scope="global", scope_ref=None):
        return AddAggregateOperation(
            operation_type="add_aggregate", operation_id="op_aggregate", gap_id="gap_aggregate",
            evidence=[{"text": "证据"}], anchor_ids=["anchor_ok"], aggregate_id=aggregate_id,
            function="avg", input={"kind": "catalog_symbol", "identifier": field_id,
                                   "result_type": "number", "unit": unit}, scope=scope, scope_ref=scope_ref,
        )

    def test_contract_enforces_expected_shape_and_protected_nodes(self):
        shape_patch = SemanticPatchV1(base_ir_digest=request_ir_digest(self.ir),
                                      operations=[self._aggregate_operation()])
        self.assertEqual(contract_violation(
            self._aggregate_contract(expected_output_shape="relation"), shape_patch, self.ir,
        ), "repair_contract_output_shape_out_of_scope")

        protected_patch = SemanticPatchV1(
            base_ir_digest=request_ir_digest(self.ir),
            operations=[self._aggregate_operation("protected_aggregate")],
        )
        self.assertEqual(contract_violation(
            self._aggregate_contract(protected=("protected_aggregate",)), protected_patch, self.ir,
        ), "repair_contract_protected_node_write")

    def test_contract_rejects_foreign_field_unit_and_input_ref(self):
        foreign_field = SemanticPatchV1(
            base_ir_digest=request_ir_digest(self.ir),
            operations=[self._aggregate_operation(field_id="device_telemetry.voltage")],
        )
        self.assertEqual(contract_violation(self._aggregate_contract(), foreign_field, self.ir),
                         "repair_contract_input_out_of_scope")

        foreign_unit = SemanticPatchV1(
            base_ir_digest=request_ir_digest(self.ir),
            operations=[self._aggregate_operation(unit="volt")],
        )
        self.assertEqual(contract_violation(self._aggregate_contract(), foreign_unit, self.ir),
                         "repair_contract_unit_out_of_scope")

        foreign_ref = SemanticPatchV1(
            base_ir_digest=request_ir_digest(self.ir),
            operations=[self._aggregate_operation(scope="relation", scope_ref="foreign.output")],
        )
        self.assertEqual(contract_violation(self._aggregate_contract(), foreign_ref, self.ir),
                         "repair_contract_input_out_of_scope")


if __name__ == "__main__":
    unittest.main()
