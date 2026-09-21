"""P0 regressions for demand contracts, local schema, and strict coverage."""
from __future__ import annotations

import unittest
import copy

from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.field_roles import FieldPhraseParser
from nlu_v2.models import QueryEnvelope, SourceDemand, SourceSpan, UnderstandingIR
from nlu_v2.validator import IRValidator


class P0RequirementContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = QueryUnderstandingEngine()
        self.catalog = self.engine.catalog_provider.snapshot()

    def test_unconsumed_high_value_demand_blocks_semantic_complete(self):
        demand = SourceDemand("d_quantity", "quantity", "5", SourceSpan(0, 1, "5"))
        ir = UnderstandingIR(QueryEnvelope("5", "5"), self.catalog.version, source_demands=[demand])
        report = IRValidator().validate(ir, self.catalog)
        self.assertIn("source_demand_unconsumed", [item.code for item in report.errors])
        self.assertNotEqual(ir.semantic_status, "complete")

    def test_field_property_must_be_entailed_by_its_own_anchor(self):
        forged = SourceDemand(
            "d_field", "field", "连续", SourceSpan(0, 2, "连续"),
            attributes={"field_text": "温度"},
        )
        ir = UnderstandingIR(QueryEnvelope("连续", "连续"), self.catalog.version, source_demands=[forged])
        report = IRValidator().validate(ir, self.catalog)
        self.assertIn("source_demand_field_anchor_mismatch", [item.code for item in report.errors])

    def test_comparison_has_four_slot_anchors_and_is_consumed(self):
        result = self.engine.analyze("设备传感器中温度超过85℃，输出异常区间")
        comparison = next(item for item in result.understanding.source_demands
                          if item.demand_type == "comparison")
        self.assertTrue(comparison.field_anchor_id)
        self.assertTrue(comparison.operator_anchor_id)
        self.assertTrue(comparison.quantity_anchor_id)
        self.assertTrue(comparison.unit_anchor_id)
        self.assertEqual(comparison.consumption_status, "mapped_to_requirement")
        self.assertTrue(set((comparison.field_anchor_id, comparison.operator_anchor_id,
                             comparison.quantity_anchor_id, comparison.unit_anchor_id))
                        <= set(comparison.dependencies))
        self.assertNotIn("source_demand_slot_anchor_invalid",
                         [item.code for item in result.validation.errors])
        self.assertTrue(all(item.consumption_status == "mapped_to_requirement"
                            for item in result.understanding.source_demands if item.critical))

    def test_spaced_duration_unit_is_anchored_without_fake_field(self):
        result = self.engine.analyze(
            "设备传感器中温度连续超过 85℃并持续超过 10 分钟，输出异常区间"
        )
        duration = next(
            item for item in result.understanding.source_demands
            if item.demand_type == "duration_threshold" and item.attributes.get("unit") == "分钟"
        )
        self.assertTrue(duration.unit_anchor_id)
        self.assertFalse(duration.field_anchor_id)
        self.assertNotIn(
            "source_demand_slot_anchor_missing",
            [item.code for item in result.validation.errors
             if item.span == duration.span],
        )

    def test_unspecified_source_keeps_cross_domain_city_as_candidate(self):
        result = self.engine.analyze("查询物流城市等于北京的订单")
        city = next(item for item in result.understanding.projections if item.raw_name == "城市")
        self.assertEqual(city.status, "candidate")
        self.assertIsNone(city.canonical_id)
        self.assertNotEqual(city.canonical_id, "weather_observations.city")

        scoped = self.engine.analyze("天气数据中温度大于30摄氏度")
        temperature = next(item for item in scoped.understanding.projections if item.raw_name == "温度")
        self.assertEqual(temperature.status, "resolved")
        self.assertEqual(temperature.canonical_id, "weather_observations.temperature")

    def test_role_parser_rejects_connectors_actions_and_time_ratio_as_fields(self):
        self.assertEqual(FieldPhraseParser.field_before("温度超过85和主轴转速", len("温度超过85和主轴转速")).text,
                         "主轴转速")
        self.assertEqual(FieldPhraseParser.classify("内连续浏览").role, "action_value")
        self.assertEqual(FieldPhraseParser.classify("的时间占比").role, "derived_operator")
        self.assertEqual(FieldPhraseParser.classify("当天累计加购次数").role, "action_value")
        self.assertNotEqual(FieldPhraseParser.classify("click").role, "field")
        self.assertNotEqual(FieldPhraseParser.classify("view").role, "field")
        self.assertNotEqual(FieldPhraseParser.classify("purchase").role, "field")

    def test_missing_aggregate_correlation_and_topk_are_never_covered_by_node_type(self):
        correlation = self.engine.analyze("能源数据中计算平均功率与平均风速的相关系数")
        correlation.understanding.aggregates.clear()
        correlation.understanding.calculations.clear()
        self.engine.coverage_matcher.apply(correlation.understanding)
        correlation_reqs = [item for item in correlation.understanding.requirements
                            if item.operator_family in {"aggregate", "correlation"}]
        coverage = {item.requirement_id: item.status for item in correlation.understanding.coverage}
        self.assertTrue(correlation_reqs)
        self.assertTrue(all(coverage[item.requirement_id] == "uncovered" for item in correlation_reqs))

        topk = self.engine.analyze("电池数据按电池组分组，求温度峰值并取Top3")
        topk.understanding.group_by_operations.clear()
        topk.understanding.aggregates.clear()
        topk.understanding.top_k_operations.clear()
        self.engine.coverage_matcher.apply(topk.understanding)
        topk_reqs = [item for item in topk.understanding.requirements
                     if item.requirement_type in {"group_by", "aggregate", "top_k"}]
        coverage = {item.requirement_id: item.status for item in topk.understanding.coverage}
        self.assertTrue(topk_reqs)
        self.assertTrue(all(coverage[item.requirement_id] == "uncovered" for item in topk_reqs))

    def test_requirement_ir_declares_inputs_outputs_scope_and_dependency_edges(self):
        result = self.engine.analyze("能源数据中计算平均功率与平均风速的相关系数")
        aggregates = [item for item in result.understanding.requirements if item.requirement_type == "aggregate"]
        correlation = next(item for item in result.understanding.requirements
                           if item.operator_family == "correlation")
        self.assertGreaterEqual(len(aggregates), 2)
        self.assertTrue(all(item.expected_inputs for item in aggregates))
        self.assertEqual(correlation.expected_output_shape, "scalar")
        self.assertTrue(correlation.dependencies)
        self.assertTrue(all(value.startswith("requirement:") for value in correlation.expected_inputs
                            if value.startswith("requirement:")))

        grouped = self.engine.analyze("电池数据按电池组分组，求温度峰值并取Top3")
        group = next(item for item in grouped.understanding.requirements
                     if item.requirement_type == "group_by")
        topk = next(item for item in grouped.understanding.requirements
                    if item.requirement_type == "top_k")
        self.assertEqual(group.expected_output_shape, "relation")
        self.assertEqual(topk.expected_attributes["expected_limit"], 3)
        self.assertTrue(topk.dependencies)

        self.assertTrue(all(item.expected_output_fields for item in aggregates))
        self.assertEqual(correlation.expected_output_fields, ["correlation"])
        self.assertEqual(correlation.expected_attributes["expected_output_type"], "number")

    def test_mutated_comparison_field_and_unit_slots_are_blocked(self):
        baseline = self.engine.analyze("设备传感器中温度超过85℃，输出异常区间").understanding
        forged_field = copy.deepcopy(baseline)
        comparison = next(item for item in forged_field.source_demands if item.demand_type == "comparison")
        comparison.attributes["field_text"] = "风速"
        report = IRValidator().validate(forged_field, self.catalog)
        self.assertIn("source_demand_comparison_field_mismatch", [item.code for item in report.errors])

        forged_unit = copy.deepcopy(baseline)
        comparison = next(item for item in forged_unit.source_demands if item.demand_type == "comparison")
        comparison.attributes["unit"] = "kPa"
        report = IRValidator().validate(forged_unit, self.catalog)
        self.assertIn("source_demand_comparison_unit_mismatch", [item.code for item in report.errors])


if __name__ == "__main__":
    unittest.main()
