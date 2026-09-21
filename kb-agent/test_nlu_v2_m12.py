from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmark_nlu_v2_100 import parse_questions
from nlu_v2 import EngineConfig, QueryUnderstandingEngine, TurnContextSnapshot
from nlu_v2.context import context_snapshot_digest
from nlu_v2.llm_extractor import BoundedLLMExtractor
from nlu_v2.merge import walk_predicates
from nlu_v2.coverage import CoverageMatcher
from nlu_v2.catalog import StaticCatalogProvider
from nlu_v2.logical_planner import LogicalPlanner
from nlu_v2.validator import IRValidator
from nlu_v2.models import (
    ArithmeticSpec, CandidateRef, ComparisonSpec, DerivedMetricCall, DurationConstraint,
    EventSpec, ExpressionNode, LiteralValue, OrderedFoldSpec, OutputRef, PredicateNode,
    Provenance, QueryEnvelope, RequirementSpec, SamplingPolicy, SequenceSpec, SourceSpan,
    StateTransitionSpec, SymbolRef, UnderstandingIR, WindowSpec,
)


class CaptureProvider:
    def __init__(self, payload="{}"):
        self.payload = payload
        self.calls = 0
        self.schemas = []
        self.prompts = []

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        self.schemas.append(response_schema)
        self.prompts.append(prompt)
        return self.payload


class M12SemanticCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {
            item["id"]: item["query"] for item in parse_questions(
                Path(__file__).parent / "docs" / "NLU_V2_复杂测试题_51-100.md"
            )
        }

    def test_five_historical_false_valid_cases_are_not_executable(self):
        engine = QueryUnderstandingEngine()
        for case_id in (77, 78, 83, 89, 95):
            with self.subTest(case_id=case_id):
                result = engine.analyze(self.cases[case_id])
                self.assertFalse(result.validation.executable)

    def test_unsatisfiable_is_local_terminal_and_does_not_call_llm(self):
        provider = CaptureProvider()
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True, llm_on_complex=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(self.cases[95])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(result.understanding.unsatisfiable), 1)
        self.assertIn("unsatisfiable_constraints", [x.code for x in result.validation.errors])

    def test_quoted_prompt_injection_is_not_a_write_instruction(self):
        result = QueryUnderstandingEngine().analyze(self.cases[92])
        self.assertNotIn("write_operation", [x.code for x in result.validation.errors])
        self.assertEqual(result.understanding.content_annotations[0].kind,
                         "prompt_injection_evidence")
        self.assertEqual(result.understanding.content_annotations[0].action, "treat_as_data")

    def test_turn_context_is_explicit_immutable_input(self):
        engine = QueryUnderstandingEngine()
        previous = engine.analyze("查找2025年中文RAG论文").understanding
        before = json.dumps(previous.to_dict(), ensure_ascii=False, sort_keys=True)
        snapshot = TurnContextSnapshot(
            previous_turn_id="turn_1", accepted_semantic_ir=previous,
            context_digest=context_snapshot_digest(previous),
            catalog_version=previous.catalog_version,
        )
        result = engine.analyze("那英文的呢？只要有代码的", context_snapshot=snapshot)
        predicates = list(walk_predicates(result.understanding.filters))
        self.assertTrue(any(x.value and x.value.value == "英文" for x in predicates))
        self.assertFalse(any(x.value and x.value.value == "中文" for x in predicates))
        self.assertTrue(any(x.field and x.field.raw_name == "code_url" for x in predicates))
        self.assertEqual(result.understanding.turn_directives[0].target_turn_id, "turn_1")
        self.assertEqual(json.dumps(previous.to_dict(), ensure_ascii=False, sort_keys=True), before)

    def test_follow_up_without_snapshot_is_blocked(self):
        result = QueryUnderstandingEngine().analyze("那英文的呢？只要有代码的")
        self.assertIn("context_required", [x.code for x in result.validation.errors])
        self.assertFalse(result.validation.executable)

    def test_temporal_follow_up_preserves_its_source_demand(self):
        engine = QueryUnderstandingEngine()
        previous = engine.analyze("查找2025年中文RAG论文").understanding
        snapshot = TurnContextSnapshot(
            previous_turn_id="turn_temporal", accepted_semantic_ir=previous,
            context_digest=context_snapshot_digest(previous),
            catalog_version=previous.catalog_version,
        )
        result = engine.analyze("那2024年的呢", context_snapshot=snapshot)
        codes = [item.code for item in result.validation.errors]
        self.assertNotIn("requirement_demand_missing", codes)
        demand_ids = {item.demand_id for item in result.understanding.source_demands}
        referenced = {
            demand_id for req in result.understanding.requirements
            for demand_id in req.expected_attributes.get("source_demand_ids", [])
        }
        self.assertTrue(referenced.issubset(demand_ids))

    def test_resettable_state_operator_is_domain_neutral(self):
        result = QueryUnderstandingEngine().analyze(self.cases[69])
        self.assertEqual(len(result.understanding.ordered_folds), 1)
        fold = result.understanding.ordered_folds[0]
        self.assertIsNotNone(fold.transition.reset_condition)
        self.assertIsNotNone(fold.transition.reset_expression)
        self.assertEqual(fold.post_duration.normalized_seconds, 7200)
        self.assertIn("RESETTABLE_SCAN", [x.operator_id for x in result.understanding.operator_invocations])

    def test_schema_keeps_local_envelope_private_only_for_atomic_single_gap_units(self):
        provider = CaptureProvider()
        QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze("将订单金额换算为美元")
        self.assertEqual(provider.calls, 1)
        properties = provider.schemas[0].get("properties", {})
        if "base_ir_digest" not in properties:
            # A one-gap atomic response carries these envelope values locally.
            self.assertNotIn("gap_id", properties)
            self.assertNotIn("operation_id", properties)
        else:
            # A connected multi-gap RepairUnit intentionally uses the typed V1
            # patch transaction, whose public schema must carry its digest and
            # operation list so the whole unit can be validated atomically.
            self.assertIn("operations", properties)

    def test_semantic_schema_failure_does_not_open_provider_breaker(self):
        provider = CaptureProvider("{")
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True, breaker_failures=1),
            llm_extractor=BoundedLLMExtractor(provider, allow_json_repair=False),
        )
        engine.analyze("将订单金额换算为美元")
        engine.analyze("将基金金额换算为欧元")
        self.assertEqual(provider.calls, 2)
        self.assertFalse(engine._breaker_open())

    def test_coverage_is_separate_from_requirement_source_evidence(self):
        result = QueryUnderstandingEngine().analyze("查询温度大于30摄氏度的数据")
        self.assertTrue(result.understanding.coverage)
        self.assertTrue(all(item.status == "uncovered"
                            for item in result.understanding.requirements))
        self.assertTrue(any(item.status == "satisfied"
                            for item in result.understanding.coverage))

    def test_timezone_fold_ambiguity_covers_its_temporal_requirement(self):
        result = QueryUnderstandingEngine().analyze(self.cases[71])
        coverage = {item.requirement_id: item for item in result.understanding.coverage}
        temporal_requirement = next(
            item for item in result.understanding.requirements
            if item.requirement_type == "temporal_ambiguity"
        )
        self.assertEqual(coverage[temporal_requirement.requirement_id].status, "satisfied")
        self.assertNotIn("requirement_uncovered", [item.code for item in result.validation.errors])

    def test_event_cannot_cover_a_different_requirement_clause(self):
        first = SourceSpan(0, 5, "温度连续")
        second = SourceSpan(6, 11, "湿度连续")
        event = EventSpec(
            event_id="event_temperature", condition=PredicateNode(),
            derived_metric=DerivedMetricCall("semantic.consecutive_duration"),
            threshold=DurationConstraint("gte", 10, "minute", 600),
            window=WindowSpec("absolute"), sampling=SamplingPolicy(),
            provenance=[Provenance(source="rule", span=first)],
        )
        ir = UnderstandingIR(QueryEnvelope("温度连续 湿度连续", "温度连续 湿度连续"), "test", events=[event])
        ir.requirements = [
            RequirementSpec("req_temperature", "sequence_event", "constraint", span=first),
            RequirementSpec("req_humidity", "sequence_event", "constraint", span=second),
        ]
        coverage = {item.requirement_id: item for item in CoverageMatcher().apply(ir)}
        self.assertEqual(coverage["req_temperature"].status, "satisfied")
        self.assertEqual(coverage["req_humidity"].status, "uncovered")

    def test_logical_plan_serializes_sequence_and_arithmetic_capabilities(self):
        catalog = StaticCatalogProvider().snapshot()
        source = catalog.sources[0]
        partition = SymbolRef("device_id", source.fields[0].field_id, status="resolved")
        order = SymbolRef("timestamp", source.fields[1].field_id, status="resolved")
        sequence = SequenceSpec(
            "sequence_1", [PredicateNode(), PredicateNode()], [partition], order,
            OutputRef("sequence_1", "events", "event_interval", shape="event_set", grain="interval"),
        )
        arithmetic = ArithmeticSpec(
            "arithmetic_1", ExpressionNode("literal", literal=None, result_type="number"),
            OutputRef("arithmetic_1", "value", "number"),
        )
        ir = UnderstandingIR(QueryEnvelope("测试", "测试"), catalog.version,
                             sequences=[sequence], arithmetic=[arithmetic])
        plan = LogicalPlanner()._plan_structured_analytics(ir, source.source_id, source)
        analytics = plan.nodes[1]
        self.assertEqual(plan.status, "ready")
        self.assertIn("sequences", analytics.inputs)
        self.assertIn("arithmetic", analytics.inputs)
        self.assertIn("timeseries.sequence", analytics.required_capabilities)
        self.assertIn("math.expression", analytics.required_capabilities)

    def test_trading_day_duration_requires_market_calendar(self):
        catalog = StaticCatalogProvider().snapshot()

        def sequence_for(source_id, partition_index, order_index):
            source = catalog.source(source_id)
            partition_field = source.fields[partition_index]
            order_field = source.fields[order_index]
            value_field = source.fields[1]
            partition = SymbolRef(partition_field.field_id.rsplit(".", 1)[-1],
                                  partition_field.field_id, status="resolved")
            order = SymbolRef(order_field.field_id.rsplit(".", 1)[-1],
                              order_field.field_id, status="resolved")
            value = SymbolRef(value_field.field_id.rsplit(".", 1)[-1],
                              value_field.field_id, status="resolved")
            return SequenceSpec(
                f"{source_id}_sequence",
                [PredicateNode(operator="gt", field=value, value=LiteralValue(1, "number")),
                 PredicateNode(operator="gt", field=value, value=LiteralValue(2, "number"))],
                [partition], order,
                OutputRef(f"{source_id}_sequence", "events", "event_interval",
                          shape="event_set", grain="interval"),
                max_duration=DurationConstraint("lte", 5, "交易日", 432000),
            )

        device_ir = UnderstandingIR(
            QueryEnvelope("设备连续状态", "设备连续状态"), catalog.version,
            source_candidates=[CandidateRef("device_telemetry", status="resolved")],
            sequences=[sequence_for("device_telemetry", 0, 1)],
        )
        device_codes = {item.code for item in IRValidator().validate(device_ir, catalog).errors}
        self.assertIn("trading_calendar_unbound", device_codes)

        market_ir = UnderstandingIR(
            QueryEnvelope("行情连续状态", "行情连续状态"), catalog.version,
            source_candidates=[CandidateRef("market_data", status="resolved")],
            sequences=[sequence_for("market_data", 0, 4)],
        )
        market_codes = {item.code for item in IRValidator().validate(market_ir, catalog).errors}
        self.assertNotIn("trading_calendar_unbound", market_codes)

    def test_previous_state_is_rejected_outside_transition_expression(self):
        catalog = StaticCatalogProvider().snapshot()
        source = catalog.source("device_telemetry")
        partition = SymbolRef("device_id", source.fields[0].field_id, status="resolved")
        order = SymbolRef("timestamp", source.fields[1].field_id, status="resolved")
        previous = ExpressionNode("output_ref", reference="previous_state", result_type="number")
        transition = StateTransitionSpec(
            "fold_transition", [partition], order, previous,
            ExpressionNode("literal", literal=LiteralValue(0, "number"), result_type="number"),
            OutputRef("fold_1", "state", "number", shape="relation", grain="partition_order"),
            reset_condition=PredicateNode(operator="eq", field=partition,
                                          value=LiteralValue("reset", "string")),
            reset_expression=previous,
        )
        fold = OrderedFoldSpec("fold_1", transition, transition.output)
        ir = UnderstandingIR(
            QueryEnvelope("状态递推", "状态递推"), catalog.version,
            source_candidates=[CandidateRef("device_telemetry", status="resolved")],
            ordered_folds=[fold],
        )
        codes = [item.code for item in IRValidator().validate(ir, catalog).errors]
        self.assertGreaterEqual(codes.count("state_previous_state_scope_invalid"), 2)

    def test_window_expression_shapes_and_sequence_output_reference_are_validated(self):
        catalog = StaticCatalogProvider().snapshot()
        source = catalog.source("device_telemetry")
        device = SymbolRef("device_id", source.fields[0].field_id, status="resolved")
        timestamp = SymbolRef("timestamp", source.fields[1].field_id, status="resolved")
        temperature = SymbolRef("temperature", source.fields[2].field_id, status="resolved")
        voltage = SymbolRef("voltage", source.fields[3].field_id, status="resolved")
        duration = ExpressionNode(
            "literal", literal=LiteralValue(600, "duration", "second"),
            result_type="duration", unit="second",
        )
        valid_average = ExpressionNode(
            "function", operator="moving_avg",
            arguments=[ExpressionNode("field", symbol=temperature, result_type="number", unit="celsius"),
                       duration],
            result_type="number", unit="celsius",
        )
        invalid_average = ExpressionNode(
            "function", operator="moving_avg",
            arguments=[ExpressionNode("field", symbol=device, result_type="string"), duration],
            result_type="string",
        )
        invalid_ratio = ExpressionNode(
            "function", operator="window_ratio",
            arguments=[ExpressionNode("field", symbol=temperature, result_type="number", unit="celsius"),
                       ExpressionNode("field", symbol=voltage, result_type="number", unit="volt")],
            result_type="number", unit="ratio",
        )
        sequence = SequenceSpec(
            "sequence_1",
            [PredicateNode(operator="gt", field=temperature, value=LiteralValue(30, "number", "celsius")),
             PredicateNode(operator="gt", field=temperature, value=LiteralValue(35, "number", "celsius"))],
            [device], timestamp,
            OutputRef("sequence_1", "events", "event_interval", shape="event_set", grain="interval"),
        )
        sequence_ref = ExpressionNode("output_ref", reference=sequence.output.ref_id,
                                      result_type="event_interval")
        ir = UnderstandingIR(
            QueryEnvelope("窗口表达式", "窗口表达式"), catalog.version,
            source_candidates=[CandidateRef("device_telemetry", status="resolved")],
            arithmetic=[
                ArithmeticSpec("valid_average", valid_average,
                               OutputRef("valid_average", "value", "number", "celsius")),
                ArithmeticSpec("invalid_average", invalid_average,
                               OutputRef("invalid_average", "value", "string")),
                ArithmeticSpec("invalid_ratio", invalid_ratio,
                               OutputRef("invalid_ratio", "value", "number", "ratio")),
            ],
            comparisons=[ComparisonSpec(
                "sequence_comparison", sequence_ref, "eq", sequence_ref,
                OutputRef("sequence_comparison", "result", "boolean"),
            )],
            sequences=[sequence],
        )
        codes = [item.code for item in IRValidator().validate(ir, catalog).errors]
        self.assertEqual(codes.count("window_function_shape_invalid"), 2)
        self.assertNotIn("expression_reference_unbound", codes)


if __name__ == "__main__":
    unittest.main()
