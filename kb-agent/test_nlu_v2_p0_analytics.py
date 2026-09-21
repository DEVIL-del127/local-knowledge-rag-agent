"""P0-7/8 regressions: explicit analytic DAGs and formula lineage."""
from __future__ import annotations

import copy
import unittest

from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.validator import IRValidator


class P0AnalyticDagTests(unittest.TestCase):
    def setUp(self):
        self.engine = QueryUnderstandingEngine()
        self.catalog = self.engine.catalog_provider.snapshot()

    def test_correlation_consumes_two_named_aggregate_outputs(self):
        result = self.engine.analyze("能源数据中计算平均功率与平均风速的相关系数")
        ir = result.understanding
        self.assertEqual({item.aggregate.input.symbol.raw_name for item in ir.aggregates}, {"功率", "风速"})
        correlation = next(item for item in ir.calculations if item.calculation_type == "correlation")
        self.assertEqual(
            {item.reference for item in correlation.inputs},
            {item.aggregate.output.ref_id for item in ir.aggregates},
        )
        self.assertEqual(correlation.output.unit, "ratio")

    def test_ratio_consumes_two_daily_peak_outputs(self):
        result = self.engine.analyze("天气数据中每日温度峰值与风速峰值的比值")
        ir = result.understanding
        self.assertEqual([item.aggregate.function for item in ir.aggregates], ["max", "max"])
        ratio = next(item for item in ir.calculations
                     if item.calculation_type == "ratio" and item.output is not None)
        self.assertEqual({item.reference for item in ratio.inputs},
                         {item.aggregate.output.ref_id for item in ir.aggregates})

    def test_group_peak_topk_is_a_three_stage_dag(self):
        result = self.engine.analyze("电池数据按电池组分组，求温度峰值并取Top3")
        ir = result.understanding
        self.assertEqual(len(ir.group_by_operations), 1)
        self.assertEqual(ir.group_by_operations[0].keys[0].symbol.raw_name, "电池组")
        self.assertEqual(len(ir.aggregates), 1)
        self.assertEqual(ir.aggregates[0].aggregate.function, "max")
        self.assertEqual(ir.aggregates[0].aggregate.input.symbol.raw_name, "温度")
        self.assertEqual(ir.aggregates[0].scope_ref.ref_id, ir.group_by_operations[0].output.ref_id)
        self.assertEqual(len(ir.top_k_operations), 1)
        self.assertEqual(ir.top_k_operations[0].limit, 3)
        self.assertEqual(ir.top_k_operations[0].input_ref.ref_id, ir.aggregates[0].aggregate.output.ref_id)

    def test_volatility_needs_an_explicitly_attached_field(self):
        explicit = self.engine.analyze("市场数据中成交量和价格，计算价格波动率").understanding
        volatility = next(item for item in explicit.calculations if item.calculation_type == "volatility")
        self.assertEqual(volatility.parameters["input_field"], "market_data.price")
        self.assertEqual(len(volatility.inputs), 1)
        self.assertEqual(volatility.inputs[0].symbol.canonical_id, "market_data.price")
        self.assertIsNotNone(volatility.output)

        gap = self.engine.analyze("市场数据中价格和成交量，计算波动率").understanding
        volatility_gap = next(item for item in gap.calculations if item.calculation_type == "volatility")
        self.assertEqual(volatility_gap.parameters["formula_gap"], "input_field_unbound")
        self.assertIn("formula_input_unbound", [item.code for item in gap.unresolved])

    def test_missing_calculation_output_ref_is_rejected(self):
        ir = copy.deepcopy(self.engine.analyze("能源数据中计算平均功率与平均风速的相关系数").understanding)
        calculation = next(item for item in ir.calculations if item.calculation_type == "correlation")
        calculation.inputs[0].reference = "missing.output"
        report = IRValidator().validate(ir, self.catalog)
        self.assertTrue({"expression_reference_unbound", "output_ref_unbound"}
                        & {item.code for item in report.errors})

        ir = copy.deepcopy(self.engine.analyze("能源数据中计算平均功率与平均风速的相关系数").understanding)
        calculation = next(item for item in ir.calculations if item.calculation_type == "correlation")
        calculation.output = None
        report = IRValidator().validate(ir, self.catalog)
        self.assertIn("calculation_output_unbound", [item.code for item in report.errors])

    def test_cumulative_duration_is_a_typed_numeric_dag_node(self):
        result = self.engine.analyze("行为数据中当天累计浏览时长")
        ir = result.understanding
        self.assertEqual(len(ir.cumulative_durations), 1)
        duration = ir.cumulative_durations[0]
        self.assertEqual(duration.condition.value.value, "view")
        self.assertEqual((duration.output.result_type, duration.output.unit,
                          duration.output.shape, duration.output.grain),
                         ("number", "second", "scalar", "day"))
        self.assertIn("CUMULATIVE_DURATION", [item.operator_id for item in ir.operator_invocations])
        self.assertTrue(any(item.status == "satisfied" and item.requirement_id
                            for item in ir.coverage))

    def test_cumulative_duration_supports_an_explicit_numeric_condition(self):
        result = self.engine.analyze("设备数据中当天温度超过85℃的累计时长")
        ir = result.understanding
        self.assertEqual(len(ir.cumulative_durations), 1)
        duration = ir.cumulative_durations[0]
        self.assertEqual(duration.condition.operator, "gt")
        self.assertEqual(duration.condition.field.raw_name, "温度")
        self.assertEqual((duration.condition.value.value, duration.condition.value.unit), (85.0, "℃"))
        self.assertEqual((duration.output.result_type, duration.output.unit,
                          duration.output.shape, duration.output.grain),
                         ("number", "second", "scalar", "day"))
        self.assertFalse(any(item.code == "cumulative_duration_condition_unbound"
                             for item in result.validation.errors))

    def test_coverage_requires_the_exact_requirement_output_ref(self):
        ir = copy.deepcopy(self.engine.analyze("电池数据按电池组分组，求温度峰值并取Top3").understanding)
        top_k = ir.top_k_operations[0]
        group_ref = ir.group_by_operations[0].output
        top_k.input_ref = group_ref
        top_k.rank_by.reference = group_ref.ref_id
        self.engine.coverage_matcher.apply(ir)
        top_requirement = next(item for item in ir.requirements if item.requirement_type == "top_k")
        coverage = {item.requirement_id: item for item in ir.coverage}
        self.assertEqual(coverage[top_requirement.requirement_id].status, "uncovered")
        self.assertTrue(any(value.startswith("input_output_ref:")
                            for value in coverage[top_requirement.requirement_id].missing_attributes))

        ir = copy.deepcopy(self.engine.analyze("电池数据按电池组分组，求温度峰值并取Top3").understanding)
        ir.top_k_operations[0].output.fields.clear()
        self.engine.coverage_matcher.apply(ir)
        coverage = {item.requirement_id: item for item in ir.coverage}
        self.assertEqual(coverage[top_requirement.requirement_id].status, "uncovered")
        self.assertIn("output_fields", coverage[top_requirement.requirement_id].missing_attributes)

    def test_field_substitutions_preserve_sequence_set_topology(self):
        template = (
            "每2秒记录一次{field}（`metric`）。找出{field}连续超过2.5且持续超过5秒的时间段，"
            "以及{field}连续低于-2.5且持续超过5秒的时间段，输出交集日期。"
        )
        topologies = []
        for field in ("温度", "压力", "速度", "功率"):
            ir = self.engine.analyze(template.format(field=field)).understanding
            topologies.append((len(ir.events), len(ir.set_operations),
                               tuple(item.operator_id for item in ir.operator_invocations)))
        self.assertEqual(topologies, [topologies[0]] * len(topologies))


if __name__ == "__main__":
    unittest.main()
