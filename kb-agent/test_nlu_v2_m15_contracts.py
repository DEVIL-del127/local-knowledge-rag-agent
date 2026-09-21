import unittest
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from nlu_v2.canonical_requirements import CanonicalRequirementGraph
from nlu_v2.catalog import StaticCatalogProvider
from nlu_v2.engine import EngineConfig, QueryUnderstandingEngine
from nlu_v2.event_execution import EventExecutionCompiler
from nlu_v2.models import (
    AggregateSpec, CumulativeDurationSpec, ExpressionNode, LiteralValue,
    OutputRef, PredicateNode, QueryEnvelope, RequirementSpec, SchemaHypothesis,
    ScopedAggregateSpec, SourceDemand, SourceSpan, SymbolRef, UnderstandingIR,
    WindowSpec,
)
from nlu_v2.quantities import QuantityRegistry
from nlu_v2.typed_lineage import TypedLineageCompiler, project_local_dates
from nlu_v2.reference_binding import ReferenceBinder
from nlu_v2.catalog_contracts import CatalogContractPreflight


class CanonicalRequirementTests(unittest.TestCase):
    def test_same_source_demand_duplicates_merge(self):
        ir = _empty_ir()
        demand = SourceDemand("d1", "operator", "交集", SourceSpan(0, 2, "交集"),
                              clause_id="c1", mapped_requirement_ids=["r1", "r2"])
        ir.source_demands = [demand]
        ir.requirements = [
            RequirementSpec("r1", "set_operation", "output", clause_id="c1",
                            operator_family="set", expected_output_grain="date"),
            RequirementSpec("r2", "set_operation", "output", clause_id="c1",
                            operator_family="set", expected_output_grain="date"),
        ]
        graph = CanonicalRequirementGraph.build(ir)
        self.assertEqual(1, len(graph.nodes))
        self.assertEqual(("r1", "r2"), graph.nodes[0].requirement_ids)

    def test_source_distinct_obligations_do_not_merge(self):
        ir = _empty_ir()
        ir.source_demands = [
            SourceDemand("d1", "operator", "交集", SourceSpan(0, 2, "交集"),
                         clause_id="c1", mapped_requirement_ids=["r1"]),
            SourceDemand("d2", "operator", "交集", SourceSpan(10, 12, "交集"),
                         clause_id="c2", mapped_requirement_ids=["r2"]),
        ]
        ir.requirements = [
            RequirementSpec("r1", "set_operation", "output", clause_id="c1", operator_family="set"),
            RequirementSpec("r2", "set_operation", "output", clause_id="c2", operator_family="set"),
        ]
        self.assertEqual(2, len(CanonicalRequirementGraph.build(ir).nodes))


class QuantityContractTests(unittest.TestCase):
    def test_absolute_and_delta_temperature_are_not_interchangeable(self):
        registry = QuantityRegistry()
        conversion = registry.conversion("℃", "fahrenheit", quantity_kind="temperature_delta")
        self.assertAlmostEqual(18.0, conversion.convert(10.0), places=6)
        absolute = registry.conversion("℃", "fahrenheit", quantity_kind="absolute_temperature")
        self.assertAlmostEqual(50.0, absolute.convert(10.0), places=5)

    def test_logarithmic_unit_never_uses_scalar_conversion(self):
        registry = QuantityRegistry()
        same = registry.conversion("dB", "dB", quantity_kind="log_level")
        self.assertTrue(same.allowed)
        unknown = registry.conversion("dB", "ratio", quantity_kind="log_level")
        self.assertFalse(unknown.allowed)

    def test_initial_industrial_dimensions_convert_linearly(self):
        registry = QuantityRegistry()
        self.assertAlmostEqual(10.0, registry.conversion("km/h", "m/s", quantity_kind="speed").convert(36))
        self.assertAlmostEqual(100000.0, registry.conversion("kpa", "pa", quantity_kind="pressure").convert(100))
        self.assertAlmostEqual(1000.0, registry.conversion("km", "meter", quantity_kind="distance").convert(1))


class TemporalContractTests(unittest.TestCase):
    def test_quantiles_bind_exact_parameters_and_feed_difference(self):
        result = QueryUnderstandingEngine(config=EngineConfig(enable_llm=False)).analyze(
            "交易数据包含信用利差（`credit_spread`）。计算信用利差的90分位数和10分位数的差值。"
        )
        ir = result.understanding
        coverage = {item.requirement_id: item for item in ir.coverage}
        quantiles = [item for item in ir.requirements if item.operator_family == "quantile"]
        self.assertEqual(2, len(quantiles))
        self.assertTrue(all(coverage[item.requirement_id].status == "satisfied"
                            for item in quantiles))
        self.assertEqual(1, len(coverage[quantiles[0].requirement_id].covered_by))
        self.assertEqual(1, len(coverage[quantiles[1].requirement_id].covered_by))
        self.assertNotEqual(coverage[quantiles[0].requirement_id].covered_by,
                            coverage[quantiles[1].requirement_id].covered_by)
        difference = next(item for item in ir.requirements
                          if item.operator_family == "difference")
        self.assertEqual("satisfied", coverage[difference.requirement_id].status)

    def test_unique_set_output_restatement_and_scoped_scalar_aggregate_close(self):
        result = QueryUnderstandingEngine(config=EngineConfig(enable_llm=False)).analyze(
            "服务器每分钟记录 CPU 使用率（`cpu_usage`）和内存使用率（`memory_usage`）。"
            "找出 CPU 使用率连续超过80%且持续超过10分钟的区间，"
            "以及内存使用率连续超过90%且持续超过10分钟的区间。"
            "输出重合时间段，并计算重合时间段内 CPU 使用率的平均值。"
        )
        ir = result.understanding
        coverage = {item.requirement_id: item for item in ir.coverage}
        relevant = [item for item in ir.requirements
                    if item.requirement_type in {"set_operation", "aggregate"}
                    and item.span and item.span.start > 100]
        self.assertTrue(relevant)
        self.assertTrue(all(coverage[item.requirement_id].status == "satisfied"
                            for item in relevant))
        aggregate = next(item for item in ir.aggregates
                         if item.aggregate.function == "avg")
        self.assertEqual(("filtered", "scalar", "scalar"), (
            aggregate.scope, aggregate.aggregate.output.shape,
            aggregate.aggregate.output.grain,
        ))

    def test_event_readiness_false_ready_counter_accepts_tuple_blockers(self):
        result = QueryUnderstandingEngine(config=EngineConfig(
            enable_llm=False, enable_m15_temporal_contracts=True,
        )).analyze(
            "制造质量数据，良率低于95%连续10分钟，输出异常区间"
        )
        self.assertIsNotNone(result.event_readiness_report)
        self.assertIsInstance(result.event_readiness_report.false_ready_count, int)

    def test_duration_rebinds_to_covered_set_at_coverage_fixed_point(self):
        result = QueryUnderstandingEngine(config=EngineConfig(enable_llm=False)).analyze(
            "产线每分钟记录运行状态（`running`）和转速（`rpm`）。"
            "找出运行状态为0连续超过10分钟的时间段，以及转速低于100连续超过5分钟的时间段。"
            "输出两类事件日期的并集，并计算这些日期内异常总时长。"
        )
        ir = result.understanding
        coverage = {item.requirement_id: item for item in ir.coverage}
        requirement = next(item for item in ir.requirements
                           if item.requirement_type == "cumulative_duration")
        duration = next(item for item in ir.cumulative_durations
                        if item.duration_id in coverage[requirement.requirement_id].covered_by)
        set_requirement = next(item for item in ir.requirements
                               if item.requirement_id in requirement.dependencies)
        set_node_id = coverage[set_requirement.requirement_id].covered_by[0]
        set_operation = next(item for item in ir.set_operations if item.output_name == set_node_id)
        self.assertEqual("satisfied", coverage[requirement.requirement_id].status)
        self.assertEqual(set_operation.output_ref.ref_id, duration.input_ref.ref_id)
        self.assertEqual(("relation", "day", ["local_date"]),
                         (duration.output.shape, duration.output.grain, duration.output.keys))

    def test_reference_requirement_is_extracted_for_these_dates(self):
        result = QueryUnderstandingEngine(config=EngineConfig(enable_llm=False)).analyze(
            "设备温度连续超过85℃持续10分钟，输出这些日期的交集结果"
        )
        references = [item for item in result.understanding.requirements if item.requirement_type == "reference"]
        self.assertTrue(any(item.text == "这些日期" for item in references))

    def test_moving_average_thresholds_compile_to_typed_window_outputs(self):
        result = QueryUnderstandingEngine(config=EngineConfig(enable_llm=False)).analyze(
            "股票数据包含价格(`price`)和成交量(`volume`)。"
            "价格连续高于20日均线且持续超过60分钟，以及成交量连续超过10日均量且持续超过30分钟。"
        )
        ir = result.understanding
        self.assertEqual(["20 日", "10 日"], [item.window.reference for item in ir.window_aggregates])
        self.assertEqual(["价格", "成交量"], [item.input.symbol.raw_name for item in ir.window_aggregates])
        window_refs = {item.output.ref_id for item in ir.window_aggregates}
        event_refs = {item.condition.right_expression.reference for item in ir.events
                      if item.condition.right_expression}
        self.assertEqual(window_refs, event_refs)
    def test_half_open_date_projection_handles_midnight_and_deduplicates(self):
        zone = ZoneInfo("Asia/Shanghai")
        intervals = [
            (datetime(2026, 8, 1, 23, 0, tzinfo=zone), datetime(2026, 8, 2, 0, 0, tzinfo=zone)),
            (datetime(2026, 8, 1, 1, 0, tzinfo=zone), datetime(2026, 8, 1, 2, 0, tzinfo=zone)),
        ]
        self.assertEqual(("2026-08-01",), project_local_dates(intervals, "Asia/Shanghai"))

    def test_naive_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone_ambiguity"):
            project_local_dates([(datetime(2026, 11, 1, 1), datetime(2026, 11, 1, 2))],
                                "America/New_York")

    def test_typed_lineage_rejects_same_producer_and_mixed_grain_sets(self):
        from nlu_v2.models import SetOperationSpec
        ir = _empty_ir()
        date_ref = OutputRef("producer", "dates", "date", shape="event_set", grain="date")
        interval_ref = OutputRef("producer", "intervals", "event_interval",
                                 shape="event_set", grain="interval")
        ir.set_operations = [SetOperationSpec(
            "intersection", [date_ref.ref_id, interval_ref.ref_id], "bad", input_refs=[date_ref, interval_ref],
            output_ref=OutputRef("bad", "rows", "date", shape="event_set", grain="date"),
        )]
        report = TypedLineageCompiler().compile(ir)
        self.assertIn("bad.rows:same_producer_set", report.blockers)
        self.assertIn("bad.rows:mixed_set_type_or_grain", report.blockers)

    def test_schema_hypothesis_never_reaches_ready(self):
        ir = _empty_ir()
        ir.schema_hypotheses = [SchemaHypothesis("hyp.x", "未知温度", "未知温度")]
        ir.filters = PredicateNode(field=SymbolRef("未知温度", "hyp.x", ref_kind="hypothesis"),
                                   operator="gt", value=LiteralValue(1, "number"))
        report = EventExecutionCompiler().compile(ir, StaticCatalogProvider().snapshot())
        self.assertEqual(0, report.false_ready_count)

    def test_duration_and_value_contracts_reject_swapped_types(self):
        ir = _empty_ir()
        ir.cumulative_durations = [CumulativeDurationSpec(
            "duration1", PredicateNode(), WindowSpec("absolute"),
            OutputRef("duration1", "value", "distance", shape="relation", grain="date"),
        )]
        symbol = SymbolRef("里程", "vehicle_telemetry.distance", status="resolved")
        ir.aggregates = [ScopedAggregateSpec(AggregateSpec(
            "sum1", "sum", ExpressionNode("field", symbol=symbol),
            OutputRef("sum1", "value", "duration", shape="relation", grain="date"),
        ), "filtered")]
        report = EventExecutionCompiler().compile(ir, StaticCatalogProvider().snapshot())
        self.assertIn("duration_output_type_invalid", report.duration_accumulations[0].blockers)
        self.assertIn("value_output_cannot_be_duration", report.cumulative_values[0].blockers)

    def test_shadow_flags_leave_default_behavior_unchanged(self):
        result = QueryUnderstandingEngine(config=EngineConfig()).analyze("查找知识库中的论文")
        self.assertIsNone(result.canonical_requirement_graph)
        self.assertIsNone(result.field_binding_report)
        self.assertIsNone(result.event_readiness_report)


class CatalogContractTests(unittest.TestCase):
    def test_frozen_positive_and_negative_contracts_stop_at_exact_stage(self):
        fixture = Path(__file__).parent / "tests" / "fixtures" / "m15_catalog_contract_matrix.json"
        contracts = json.loads(fixture.read_text(encoding="utf-8"))["contracts"]
        preflight = CatalogContractPreflight()
        results = {item["contract_id"]: preflight.evaluate(item) for item in contracts}
        self.assertEqual(len(contracts), len(results))
        for item in contracts:
            self.assertEqual(item["expected_readiness_level"],
                             results[item["contract_id"]].readiness_level)


def _empty_ir():
    return UnderstandingIR(QueryEnvelope("q", "q"), "catalog")


if __name__ == "__main__":
    unittest.main()
