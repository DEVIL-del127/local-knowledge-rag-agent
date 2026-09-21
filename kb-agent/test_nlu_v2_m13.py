from __future__ import annotations

import copy
import unittest

from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.clarification import ClarificationPlanner
from nlu_v2.gaps import GapAnalyzer
from nlu_v2.models import (
    AmbiguitySpec,
    CoverageSpec,
    PatchOperationResult,
    PatchReport,
    QueryEnvelope,
    RequirementSpec,
    SourceSpan,
    UnderstandingIR,
)
from nlu_v2.patch_protocol import (
    AddAmbiguityOperation,
    GapRequest,
    SemanticPatchV1,
    apply_semantic_patch,
    parse_atomic_patch_text,
    request_ir_digest,
)
from nlu_v2.semantic_repair import (
    CompilationSnapshot,
    RepairContract,
    RepairUnit,
    SemanticAcceptanceGate,
)


class M13SemanticRecoveryTests(unittest.TestCase):
    def test_requirement_normalizer_absorbs_connector_and_duration_fragments(self):
        result = QueryUnderstandingEngine().analyze(
            "设备中温度连续超过85℃且持续超过10分钟，以及压力连续超过2MPa且持续超过5分钟"
        )
        requirements = result.understanding.requirements
        self.assertFalse([item for item in requirements if item.requirement_type == "boolean_logic"])
        events = [item for item in requirements if item.requirement_type == "sequence_event"]
        self.assertEqual(len(events), 2)
        self.assertTrue(all(item.expected_attributes.get("duration_slot_attached") for item in events))
        self.assertTrue(all(
            item.consumption_status == "mapped_to_requirement"
            for item in result.understanding.source_demands if item.critical
        ))

    def test_semantic_linker_builds_union_from_two_existing_events(self):
        result = QueryUnderstandingEngine().analyze(
            "设备中温度连续超过85℃且持续超过10分钟，以及电压连续超过220V且持续超过5分钟，输出并集日期"
        )
        self.assertEqual(len(result.understanding.events), 2)
        self.assertEqual(len(result.understanding.set_operations), 1)
        operation = result.understanding.set_operations[0]
        self.assertEqual(operation.operation, "union")
        self.assertEqual(operation.granularity, "date")
        self.assertEqual(len(operation.input_refs), 2)
        self.assertTrue(all(item.grain == "date" for item in operation.input_refs))

    def test_query_records_source_demands_before_requirement_compilation(self):
        result = QueryUnderstandingEngine().analyze(
            "设备传感器中温度连续超过85℃并持续10分钟，输出异常区间"
        )
        demands = result.understanding.source_demands
        types = {item.demand_type for item in demands}
        self.assertTrue({"field", "comparison", "quantity", "unit", "operator", "output"} <= types)
        comparison = next(item for item in demands if item.demand_type == "comparison")
        self.assertEqual(comparison.attributes["operator"], "gt")
        self.assertEqual(comparison.attributes["value"], 85.0)
        self.assertEqual(result.understanding.query_schema.source_ids, ["device_telemetry"])
        self.assertTrue(all(
            result.understanding.query.normalized[item.span.start:item.span.end] == item.text
            for item in demands
        ))

    def test_turn_directive_gap_has_source_anchor_and_is_ready(self):
        engine = QueryUnderstandingEngine()
        result = engine.analyze("请将上一轮的论文查询改为英文，并且只保留有代码的论文。")
        ir = result.understanding
        ir.turn_directives = []
        engine.coverage_matcher.apply(ir)
        gap = next(
            item for item in engine.gap_analyzer.analyze(
                ir, engine.catalog_provider.snapshot(),
            )
            if item.gap_type == "missing_turn_directive"
        )
        self.assertEqual(gap.readiness, "ready")
        self.assertTrue(gap.allowed_anchor_ids)
        self.assertTrue(all(
            anchor_id in {item.demand_id for item in ir.source_demands}
            for anchor_id in gap.allowed_anchor_ids
        ))

    def test_concrete_gate_rejects_patch_without_requirement_coverage_gain(self):
        span = SourceSpan(0, 2, "温度")
        baseline = UnderstandingIR(
            query=QueryEnvelope("温度", "温度"), catalog_version="test",
            requirements=[RequirementSpec(
                "req_temperature", "predicate", "constraint", "温度", span,
                expected_attributes={"field_text": "温度", "operator": "gt", "value": 85},
            )],
            coverage=[CoverageSpec("req_temperature", "uncovered")],
        )
        catalog = QueryUnderstandingEngine().catalog_provider.snapshot()
        gap = GapRequest(
            gap_id="gap_temperature", gap_type="missing_predicate", query_slice="温度",
            allowed_operations=["add_ambiguity"], requirement_id="req_temperature",
        )
        report = PatchReport(
            accepted=[PatchOperationResult("op", "add_event", gap.gap_id, "accepted")],
            closed_gap_ids=[gap.gap_id], concrete_gain_count=1,
        )
        contract = RepairContract(
            CompilationSnapshot.capture(baseline), RepairUnit.from_gaps([gap])[0], "concrete",
        )
        decision = SemanticAcceptanceGate().evaluate(
            baseline, baseline, report, contract, catalog,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.effect.commit_or_reject_reason, "no_requirement_coverage_gain")

    def test_named_dependency_group_rolls_back_all_its_operations(self):
        engine = QueryUnderstandingEngine()
        catalog = engine.catalog_provider.snapshot()
        ir = engine.rule_extractor.extract("需要澄清", catalog).ir
        gap = GapRequest(
            gap_id="gap_ambiguity", gap_type="unresolved_symbol", query_slice="需要澄清",
            allowed_operations=["add_ambiguity"],
        )
        first = AddAmbiguityOperation(
            operation_type="add_ambiguity", operation_id="first", gap_id=gap.gap_id,
            dependency_group="group_a", evidence=[{"text": "需要澄清"}],
            ambiguity_id="same", kind="field", message="需要字段", candidates=[],
        )
        second = AddAmbiguityOperation(
            operation_type="add_ambiguity", operation_id="second", gap_id=gap.gap_id,
            dependency_group="group_a", evidence=[{"text": "需要澄清"}],
            ambiguity_id="same", kind="field", message="需要字段", candidates=[],
        )
        patch = SemanticPatchV1(base_ir_digest=request_ir_digest(ir), operations=[first, second])
        candidate, report = apply_semantic_patch(ir, patch, [gap], catalog)
        self.assertEqual(candidate.ambiguities, ir.ambiguities)
        self.assertFalse(report.accepted)
        self.assertIn("dependency_group_rolled_back", [item.reason for item in report.rejected])

    def test_clarification_gate_commits_only_non_executable_ambiguity(self):
        baseline = UnderstandingIR(query=QueryEnvelope("字段", "字段"), catalog_version="test")
        candidate = copy.deepcopy(baseline)
        candidate.ambiguities.append(AmbiguitySpec("amb", "field", "请选择字段"))
        catalog = QueryUnderstandingEngine().catalog_provider.snapshot()
        gap = GapRequest(
            gap_id="gap_field", gap_type="unresolved_symbol", query_slice="字段",
            allowed_operations=["add_ambiguity"],
        )
        report = PatchReport(
            accepted=[PatchOperationResult("op", "add_ambiguity", gap.gap_id, "accepted")],
            closed_gap_ids=[gap.gap_id], clarification_gain_count=1,
        )
        contract = RepairContract(
            CompilationSnapshot.capture(baseline), RepairUnit.from_gaps([gap])[0], "clarification",
        )
        decision = SemanticAcceptanceGate().evaluate(
            baseline, candidate, report, contract, catalog,
        )
        self.assertTrue(decision.accepted, decision.effect.commit_or_reject_reason)
        self.assertEqual(decision.effect.acceptance_profile, "clarification")

    def test_binding_clarification_has_a_resume_contract(self):
        result = QueryUnderstandingEngine().analyze("查询体感舒适指数大于5的数据")
        plan = ClarificationPlanner().plan(result.understanding, result.validation)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.state.reason_kind, "binding")
        self.assertTrue(plan.resume_contract)
        self.assertIn("binding", plan.resume_contract.allowed_answer_slots)

    def test_set_gap_waits_for_two_typed_upstream_outputs(self):
        engine = QueryUnderstandingEngine()
        catalog = engine.catalog_provider.snapshot()
        ir = engine.rule_extractor.extract("输出两类异常的交集日期", catalog).ir
        gaps = GapAnalyzer().analyze(ir, catalog)
        set_gap = next(item for item in gaps if item.gap_type == "missing_set_inputs")
        self.assertEqual(set_gap.readiness, "blocked")
        self.assertIn("two_typed_output_refs", set_gap.blocked_by)
        self.assertEqual(set_gap.response_mode, "semantic_choice")

    def test_catalog_symbol_gap_is_deterministic_terminal(self):
        engine = QueryUnderstandingEngine()
        catalog = engine.catalog_provider.snapshot()
        ir = engine.rule_extractor.extract("查询完全未知指标超过10的记录", catalog).ir
        gaps = GapAnalyzer().analyze(ir, catalog)
        symbol_gaps = [item for item in gaps if item.gap_type == "unresolved_symbol"]
        self.assertTrue(symbol_gaps)
        self.assertTrue(all(item.readiness == "deterministic" for item in symbol_gaps))
        self.assertTrue(all(item.resolution_mode == "deterministic_terminal" for item in symbol_gaps))

    def test_product_maximum_uses_two_distinct_prefix_aggregate_fields(self):
        result = QueryUnderstandingEngine().analyze(
            "设备每10秒记录一次压力（`pressure`）和功率（`power`）。找出 2026年1月1日 至今，压力连续超过 5 MPa 且持续超过 3 分钟"
            "的所有时间段，以及功率连续超过 7 W 且持续超过 5 分钟的所有时间段。"
            "输出并集日期，并计算并集日期中当日最高压力与当日最高功率的乘积的最大值。"
        )
        ir = result.understanding
        fields = [item.aggregate.input.symbol.raw_name for item in ir.aggregates]
        self.assertEqual(fields, ["压力", "功率"])
        self.assertEqual([item.aggregate.function for item in ir.aggregates], ["max", "max"])
        self.assertTrue(all(item.scope_ref == ir.set_operations[0].output_ref for item in ir.aggregates))
        self.assertEqual([item.expression.operator for item in ir.arithmetic], ["mul"])
        self.assertEqual([item.calculation_type for item in ir.calculations], ["maximum"])
        self.assertNotIn("unsafe_calculation", [item.code for item in result.validation.errors])

    def test_filtered_and_global_sum_feed_one_ratio_without_catalog_guessing(self):
        result = QueryUnderstandingEngine().analyze(
            "库存数据包含sku_id、stock_count和unit_price。找出库存量超过50且单价超过500的SKU，"
            "计算这些SKU的总库存金额占全站总库存金额的比例。"
        )
        aggregates = result.understanding.aggregates
        self.assertEqual([item.aggregate.function for item in aggregates], ["sum", "sum"])
        self.assertEqual([item.scope for item in aggregates], ["filtered", "global"])
        self.assertEqual(
            [item.aggregate.input.symbol.raw_name for item in aggregates],
            ["库存金额", "库存金额"],
        )
        ratio = next(item for item in result.understanding.calculations
                     if item.calculation_type == "ratio")
        self.assertEqual(
            [item.reference for item in ratio.inputs],
            [item.aggregate.output.ref_id for item in aggregates],
        )

    def test_scoped_aggregate_atomic_choice_closes_its_requirement(self):
        engine = QueryUnderstandingEngine()
        result = engine.analyze("设备传感器，计算这些记录的温度平均值。")
        ir = result.understanding
        catalog = engine.catalog_provider.snapshot()
        gap = next(
            item for item in engine.gap_analyzer.analyze(ir, catalog)
            if item.gap_type == "missing_aggregate_scope" and item.readiness == "ready"
        )
        patch, _, error = parse_atomic_patch_text(
            '{"function":"avg","input_ref":"device_telemetry.temperature",'
            '"scope":"filtered","scope_ref":null,"group_by_refs":[]}',
            base_ir_digest=request_ir_digest(ir), gap=gap,
        )
        self.assertFalse(error)
        candidate, report = engine._apply_patch_transaction(ir, patch, [gap], catalog)
        self.assertTrue(report.committed, report.transaction_error)
        self.assertEqual(candidate.aggregates[0].scope, "filtered")
        self.assertIn(gap.requirement_id, report.semantic_effect.closed_requirement_ids)

    def test_window_qualified_event_is_field_neutral_and_fully_wired(self):
        result = QueryUnderstandingEngine().analyze(
            "设备每100毫秒记录压力(`pressure`,MPa)和功率(`power`,W)。找出2026年3月1日至今，"
            "压力连续超过5MPa且超过7MPa的时间占比大于20%的所有时间段，以及功率连续超过75W"
            "且持续超过30秒的所有时间段。输出交集日期，并计算这些日期中压力超过7MPa的总时长"
            "占当日总时长的比例。"
        )
        ir = result.understanding
        self.assertEqual(len(ir.events), 2)
        self.assertEqual(len(ir.relation_filters), 1)
        self.assertEqual(len(ir.set_operations), 1)
        self.assertEqual(ir.set_operations[0].operation, "intersection")
        self.assertTrue(any(
            item.input_ref.ref_id == ir.relation_filters[0].output.ref_id
            for item in ir.derived_projections
        ))
        self.assertEqual(
            sorted(item.scope.granularity for item in ir.cumulative_durations),
            ["day", "interval"],
        )
        self.assertEqual([item.expression.operator for item in ir.arithmetic], ["div", "div"])
        self.assertEqual(ir.events[0].sampling.expected_interval_seconds, 0.1)
        errors = [item.code for item in result.validation.errors]
        self.assertNotIn("source_demand_slot_anchor_missing", errors)
        self.assertNotIn("event_structure_incomplete", errors)
        self.assertNotIn("set_inputs_unbound", errors)
        self.assertNotIn("formula_input_unbound", errors)
        self.assertNotIn("output_ref_unbound", errors)


if __name__ == "__main__":
    unittest.main()
