import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from nlu_v2 import EngineConfig, QueryUnderstandingEngine
from nlu_v2.expression_signatures import (
    ExpressionSignatureRegistry,
    ValueSignature,
)
from nlu_v2.llm_extractor import BoundedLLMExtractor, ProviderResponse
from nlu_v2.coverage import CoverageMatcher
from nlu_v2.event_closure import EventClosureAnalyzer
from nlu_v2.models import (
    AggregateSpec,
    ExpressionNode,
    OutputRef,
    PatchReport,
    Provenance,
    ScopedAggregateSpec,
)
from nlu_v2.role_grounding import RoleGroundingAnalyzer
from nlu_v2.semantic_targets import OutputLineageIndex, SemanticTargetBuilder
from nlu_v2.semantic_candidates import (
    CandidateMenu,
    SemanticCandidate,
    SemanticCandidateCompiler,
    classify_candidate_dispatch,
    parse_candidate_decision,
)


FIXTURES = Path(__file__).parent / "tests" / "fixtures"
CASE_018_QUERY = (
    "光伏电站的逆变器日志，每 30 秒记录一次辐照度（`irradiance`）和转换效率"
    "（`efficiency`）。找出 2026年4月1日 至 2026年6月30日 期间，辐照度"
    "连续超过 800 W/m² 且持续超过 15 分钟的所有时间段，以及转换效率连续低于 "
    "18% 且持续超过 10 分钟的所有时间段。输出这两类异常时间段的交集日期"
    "（仅日期），并计算这些日期中，转换效率的日平均值。"
)
CASE_008_QUERY = (
    "光伏电站的环境监测系统，每 10 秒记录一次辐照度（`irradiance`）和面板温度"
    "（`panel_temp`）。找出 2026年6月1日 至 2026年8月31日 期间，辐照度"
    "连续低于 200 W/m² 且持续超过 20 分钟的所有时间段，以及面板温度连续超过 "
    "55℃ 且持续超过 15 分钟的所有时间段。输出这两类异常时间段的交集日期"
    "（仅日期），并计算这些日期中，日均辐照度与日均面板温度的比值。"
)
CASE_001_QUERY = (
    "一条自动化产线的设备日志，每 15 秒记录一次设备温度（`temp`）和主轴转速"
    "（`speed`）。找出 **2026年1月1日 至今**，温度连续超过 85℃ 且持续超过 8 分钟"
    "的所有时间段，以及转速连续低于额定转速 70% 且持续超过 5 分钟的所有时间段。"
    "输出这两类异常时间段的**交集日期**（仅日期），并计算这些日期中，每日温度峰值"
    "与转速峰值的比值。"
)
CASE_017_QUERY = (
    "一条 PCB 检测线，每 1 分钟记录一次缺陷密度（`defect_density`，个/平方米）"
    "和良率（`yield_rate`）。找出 **2026年8月1日 至今**，缺陷密度连续超过 "
    "0.5 个/平方米且持续超过 3 分钟的所有时间段，以及良率连续低于 97% 且持续"
    "超过 4 分钟的所有时间段。输出这两类异常时间段的**交集时间段**（起止时间），"
    "并计算交集时间段内缺陷密度与良率的**相关系数**。"
)
CASE_014_QUERY = (
    "电商平台的用户画像数据，包含 `user_id`、`total_spend`（累计消费金额）、"
    "`login_days`（登录天数）。找出截至今日，累计消费金额超过 10000 元的所有用户，"
    "以及登录天数超过 200 天的所有用户。输出同时满足这两个条件的用户 ID，并计算"
    "这些用户的平均累计消费金额是否高于全站平均值的 1.5 倍。"
)


class ChoiceProvider:
    supports_semantic_choice = True
    model = "test-choice-model"

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        return ProviderResponse(self.response, attempts=1, stop_reason="stop")


def _choice_menu() -> CandidateMenu:
    first = SemanticCandidate(
        "cand:sha256:" + "1" * 64, "add_calculation",
        {"formula_template": "ratio", "operands": ["a", "b"]}, "a / b",
    )
    second = SemanticCandidate(
        "cand:sha256:" + "2" * 64, "add_calculation",
        {"formula_template": "ratio", "operands": ["b", "a"]}, "b / a",
    )
    return CandidateMenu(
        "gap_choice", "req_choice", "计算 A 和 B 两者的比值", {"operation": "ratio"},
        (first, second), "llm_choice", (), "source", "target", "lineage", "menu",
    )


class M14FrozenFixtureTests(unittest.TestCase):
    @staticmethod
    def _gap_and_menu(engine, ir, gap_type, requirement_id):
        engine.coverage_matcher.apply(ir)
        catalog = engine.catalog_provider.snapshot()
        gap = next(
            item for item in engine.gap_analyzer.analyze(ir, catalog)
            if item.gap_type == gap_type and item.requirement_id == requirement_id
        )
        return gap, engine.semantic_candidate_compiler.compile(ir, gap, catalog)

    @classmethod
    def _missing_event_menu(cls, query, clause_token):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(query).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "sequence_event" and clause_token in item.text
        )
        coverage = next(
            item for item in ir.coverage if item.requirement_id == requirement.requirement_id
        )
        event_id = coverage.covered_by[0]
        event = next(item for item in ir.events if item.event_id == event_id)
        event_ref = event.output_ref.ref_id
        ir.events.remove(event)
        ir.derived_projections = [
            item for item in ir.derived_projections if item.input_ref.ref_id != event_ref
        ]
        ir.set_operations.clear()
        ir.aggregates.clear()
        ir.calculations.clear()
        return cls._gap_and_menu(
            engine, ir, "missing_event", requirement.requirement_id,
        )

    @classmethod
    def _missing_set_menu(cls, query):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(query).understanding)
        requirement = next(
            item for item in ir.requirements if item.requirement_type == "set_operation"
        )
        ir.set_operations.clear()
        ir.aggregates.clear()
        ir.calculations.clear()
        return cls._gap_and_menu(
            engine, ir, "missing_set_inputs", requirement.requirement_id,
        )

    @classmethod
    def _missing_calculation_menu(cls, query, family):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(query).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "calculation" and item.operator_family == family
        )
        ir.calculations.clear()
        return cls._gap_and_menu(
            engine, ir, "missing_formula_input", requirement.requirement_id,
        )

    @classmethod
    def _missing_aggregate_menu(cls, query):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(query).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type in {"aggregate", "scoped_aggregate"}
            and "avg" in item.expected_attributes.get("expected_functions", [])
        )
        ir.aggregates.clear()
        ir.calculations.clear()
        return cls._gap_and_menu(
            engine, ir, "missing_aggregate_scope", requirement.requirement_id,
        )

    @staticmethod
    def _wrong_menu(base, family):
        candidates = tuple(
            SemanticCandidate(
                "cand:sha256:" + marker * 64,
                base.candidates[0].operation_type,
                {"fixture_family": family, "wrong_variant": index},
                f"wrong {family} candidate {index}",
            )
            for index, marker in enumerate(("a", "b"), start=1)
        )
        dispatch, blockers = classify_candidate_dispatch(len(candidates), ())
        return replace(
            base, candidates=candidates, dispatch=dispatch,
            blocked_by=blockers, candidate_menu_digest="wrong-menu-fixture",
        )

    def test_decision_oracle_has_dispatch_controls_for_each_family(self):
        payload = json.loads((FIXTURES / "m14_decision_oracle.json").read_text("utf-8"))
        cases = payload["cases"]
        self.assertEqual(16, len(cases))
        families = {"event", "set", "aggregate", "calculation"}
        self.assertEqual(families, {item["family"] for item in cases})
        for family in families:
            values = [item for item in cases if item["family"] == family]
            self.assertTrue(any(item["expected_dispatch"] == "local_compile" for item in values))
            self.assertTrue(any(item["expected_dispatch"] == "blocked_by_upstream" for item in values))
            self.assertTrue(any(item["expected_dispatch"] == "llm_choice" for item in values))
        for item in cases:
            dispatch = item["expected_dispatch"]
            self.assertIn(dispatch, {"blocked_by_upstream", "local_compile", "llm_choice"})
            if dispatch != "llm_choice":
                self.assertNotIn(item["expected_decision"], {"none_of_above", "ambiguous"})

    def test_case_018_oracle_rejects_wrong_measure_roles(self):
        payload = json.loads((FIXTURES / "m14_decision_oracle.json").read_text("utf-8"))
        item = next(x for x in payload["cases"] if x["oracle_id"] == "aggregate_case_018_efficiency_avg")
        self.assertEqual("efficiency", item["allowed_semantics"][0]["measure"])
        self.assertIn({"measure": "irradiance"}, item["forbidden_semantics"])
        self.assertIn({"measure_role": "date"}, item["forbidden_semantics"])

    def test_decision_oracle_local_event_dispatches_use_exact_anchors(self):
        cases = (
            (CASE_001_QUERY, "超过 85", "gt", 85.0, 8.0),
            (CASE_018_QUERY, "低于 18", "lt", 18.0, 10.0),
        )
        for query, token, operator, value, minutes in cases:
            with self.subTest(token=token):
                _, menu = self._missing_event_menu(query, token)
                self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
                self.assertEqual(1, len(menu.candidates))
                payload = menu.candidates[0].semantic_payload
                self.assertEqual(operator, payload["conditions"][0]["operator"])
                self.assertEqual(value, payload["conditions"][0]["value"])
                self.assertEqual(minutes, payload["duration_value"])
                self.assertEqual("minute", payload["duration_unit"])

    def test_decision_oracle_local_set_dispatches_preserve_requested_grain(self):
        for query, grain in ((CASE_008_QUERY, "date"), (CASE_017_QUERY, "interval")):
            with self.subTest(grain=grain):
                _, menu = self._missing_set_menu(query)
                self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
                payload = menu.candidates[0].semantic_payload
                self.assertEqual("intersection", payload["operation"])
                self.assertEqual(grain, payload["granularity"])
                self.assertEqual(2, len(set(payload["inputs"])))

    def test_decision_oracle_case_014_scoped_averages_compile_without_llm(self):
        result = QueryUnderstandingEngine().analyze(CASE_014_QUERY)
        aggregates = result.understanding.aggregates
        self.assertEqual(0, result.model_calls)
        self.assertEqual(["filtered", "global"], [item.scope for item in aggregates])
        self.assertTrue(all(item.aggregate.function == "avg" for item in aggregates))
        self.assertTrue(all(
            item.aggregate.input.symbol
            and item.aggregate.input.symbol.raw_name == "累计消费金额"
            for item in aggregates
        ))

    def test_decision_oracle_local_calculation_dispatches_use_typed_inputs(self):
        cases = (
            (CASE_001_QUERY, "ratio", "ratio"),
            (CASE_017_QUERY, "correlation", "correlation"),
        )
        for query, family, template in cases:
            with self.subTest(family=family):
                _, menu = self._missing_calculation_menu(query, family)
                self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
                payload = menu.candidates[0].semantic_payload
                self.assertEqual(template, payload["formula_template"])
                self.assertEqual(2, len(set(payload["operands"])))

    def test_decision_oracle_blocked_event_has_no_duration_candidate(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_008_QUERY).understanding)
        requirement = min(
            (item for item in ir.requirements if item.requirement_type == "sequence_event"),
            key=lambda item: item.span.start,
        )
        event_id = next(
            item.covered_by[0] for item in ir.coverage
            if item.requirement_id == requirement.requirement_id
        )
        event = next(item for item in ir.events if item.event_id == event_id)
        ir.events.remove(event)
        duration_ids = {
            item.demand_id for item in ir.source_demands
            if item.demand_type == "duration_threshold"
            and requirement.span.start <= item.span.start < requirement.span.end
        }
        ir.source_demands = [
            item for item in ir.source_demands if item.demand_id not in duration_ids
        ]
        requirement.expected_attributes["source_demand_ids"] = [
            item for item in requirement.expected_attributes.get("source_demand_ids", [])
            if item not in duration_ids
        ]
        _, menu = self._gap_and_menu(
            engine, ir, "missing_event", requirement.requirement_id,
        )
        self.assertEqual("blocked_by_upstream", menu.dispatch)
        self.assertEqual(0, len(menu.candidates))
        self.assertIn("role:duration", menu.blocked_by)

    def test_decision_oracle_blocked_set_has_no_event_outputs(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_008_QUERY).understanding)
        requirement = next(
            item for item in ir.requirements if item.requirement_type == "set_operation"
        )
        ir.events.clear()
        ir.derived_projections.clear()
        ir.set_operations.clear()
        ir.aggregates.clear()
        ir.calculations.clear()
        _, menu = self._gap_and_menu(
            engine, ir, "missing_set_inputs", requirement.requirement_id,
        )
        self.assertEqual("blocked_by_upstream", menu.dispatch)
        self.assertEqual(0, len(menu.candidates))
        self.assertIn("two_typed_output_refs", menu.blocked_by)

    def test_decision_oracle_blocked_aggregate_has_no_named_measure(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_018_QUERY).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type in {"aggregate", "scoped_aggregate"}
        )
        source_ids = set(requirement.expected_attributes.get("source_demand_ids", []))
        for demand in ir.source_demands:
            if demand.demand_id in source_ids and demand.demand_type == "operator":
                demand.dependencies.clear()
        requirement.expected_attributes["input_demand_ids"] = []
        requirement.expected_inputs = []
        ir.aggregates.clear()
        _, menu = self._gap_and_menu(
            engine, ir, "missing_aggregate_scope", requirement.requirement_id,
        )
        self.assertEqual("blocked_by_upstream", menu.dispatch)
        self.assertEqual(0, len(menu.candidates))
        self.assertIn("role:measure", menu.blocked_by)

    def test_decision_oracle_blocked_calculation_has_one_unbound_input(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_017_QUERY).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "calculation"
            and item.operator_family == "correlation"
        )
        ir.calculations.clear()
        ir.schema_hypotheses = [
            item for item in ir.schema_hypotheses if item.raw_name != "yield_rate"
        ]
        _, menu = self._gap_and_menu(
            engine, ir, "missing_formula_input", requirement.requirement_id,
        )
        self.assertEqual("blocked_by_upstream", menu.dispatch)
        self.assertEqual(0, len(menu.candidates))
        self.assertIn("calculation_inputs_not_unique", menu.blocked_by)

    def test_decision_oracle_wrong_menus_are_one_call_none_of_above(self):
        bases = (
            ("event", self._missing_event_menu(CASE_001_QUERY, "超过 85")[1]),
            ("set", self._missing_set_menu(CASE_008_QUERY)[1]),
            ("aggregate", self._missing_aggregate_menu(CASE_018_QUERY)[1]),
        )
        for family, base in bases:
            with self.subTest(family=family):
                self.assertEqual("local_compile", base.dispatch, base.blocked_by)
                menu = self._wrong_menu(base, family)
                self.assertEqual("llm_choice", menu.dispatch)
                provider = ChoiceProvider('{"decision":"none_of_above"}')
                result = BoundedLLMExtractor(provider).choose_candidate(menu)
                self.assertTrue(result.valid_json, result.error)
                self.assertEqual("none_of_above", result.payload["decision"])
                self.assertEqual(1, provider.calls)

    def test_decision_oracle_unspecified_ratio_order_is_one_call_ambiguous(self):
        menu = _choice_menu()
        dispatch, blockers = classify_candidate_dispatch(len(menu.candidates), ())
        self.assertEqual("llm_choice", dispatch)
        self.assertFalse(blockers)
        provider = ChoiceProvider('{"decision":"ambiguous"}')
        result = BoundedLLMExtractor(provider).choose_candidate(menu)
        self.assertTrue(result.valid_json, result.error)
        self.assertEqual("ambiguous", result.payload["decision"])
        self.assertEqual(1, provider.calls)

    def test_bundle_authorizes_descendants_but_not_a_sibling_event(self):
        ir = QueryUnderstandingEngine().analyze(CASE_008_QUERY).understanding
        events = sorted(
            (item for item in ir.requirements if item.requirement_type == "sequence_event"),
            key=lambda item: item.span.start,
        )
        set_requirement = next(
            item for item in ir.requirements if item.requirement_type == "set_operation"
        )
        self.assertIn(events[0].requirement_id, set_requirement.dependencies)
        self.assertIn(events[1].requirement_id, set_requirement.dependencies)
        unit_ids = {item.requirement_id for item in ir.requirements}
        allowed = QueryUnderstandingEngine._downstream_requirement_ids(
            ir, events[0].requirement_id, unit_ids,
        )
        self.assertIn(events[0].requirement_id, allowed)
        self.assertIn(set_requirement.requirement_id, allowed)
        self.assertNotIn(events[1].requirement_id, allowed)

    def test_bundle_report_aggregates_each_local_transaction(self):
        reports = [
            PatchReport(
                base_ir_digest="base", committed=True,
                closed_gap_ids=["gap_event"], concrete_gain_count=1,
            ),
            PatchReport(
                base_ir_digest="next", committed=True,
                closed_gap_ids=["gap_set"], concrete_gain_count=1,
            ),
        ]
        combined = QueryUnderstandingEngine._combine_patch_reports(reports)
        self.assertTrue(combined.committed)
        self.assertEqual("base", combined.base_ir_digest)
        self.assertEqual(["gap_event", "gap_set"], combined.closed_gap_ids)
        self.assertEqual(2, combined.concrete_gain_count)

    def test_bundle_rejection_rolls_back_every_prior_step(self):
        engine = QueryUnderstandingEngine()
        baseline = SimpleNamespace(requirements=[
            SimpleNamespace(requirement_id="req_event", dependencies=[]),
            SimpleNamespace(requirement_id="req_set", dependencies=["req_event"]),
        ])
        after_event = SimpleNamespace(requirements=baseline.requirements, marker="event_added")
        event_gap = SimpleNamespace(
            gap_id="gap_event", requirement_id="req_event",
            allowed_operations=["add_event"],
        )
        set_gap = SimpleNamespace(
            gap_id="gap_set", requirement_id="req_set",
            allowed_operations=["add_set_operation"], readiness="ready",
            resolution_mode="llm",
        )
        active_unit = SimpleNamespace(
            requirement_ids=("req_event", "req_set"),
        )
        first_report = PatchReport(
            base_ir_digest="base", committed=True,
            closed_gap_ids=["gap_event"], concrete_gain_count=1,
        )
        rejected_report = PatchReport(
            base_ir_digest="after_event", committed=False,
            transaction_error="semantic_contract_rejection",
            rejection_gate="semantic_acceptance_gate",
        )
        engine._execute_candidate_choice_v3 = lambda *args: (
            after_event, first_report, None, 1,
        )
        engine.gap_analyzer = SimpleNamespace(analyze=lambda *_: [set_gap])
        engine.semantic_candidate_compiler = SimpleNamespace(
            compile=lambda *_: SimpleNamespace(
                dispatch="local_compile", candidates=[SimpleNamespace()],
            ),
            compile_patch=lambda *_: (SimpleNamespace(), ""),
        )
        engine._apply_patch_transaction = lambda *args, **kwargs: (
            SimpleNamespace(marker="set_added"), rejected_report,
        )
        result, report, _, calls = engine._execute_candidate_bundle_v3(
            baseline, event_gap, active_unit, SimpleNamespace(), "security", [],
        )
        self.assertIs(result, baseline)
        self.assertFalse(report.committed)
        self.assertEqual("semantic_contract_rejection", report.transaction_error)
        self.assertEqual(1, calls)

    def test_eligible_manifest_freezes_all_fifty_cases(self):
        path = FIXTURES / "m14_eligible_manifest.json"
        payload = json.loads(path.read_text("utf-8"))
        cases = payload["cases"]
        self.assertEqual(list(range(1, 51)), [item["case_id"] for item in cases])
        self.assertTrue(all(item["reason"] for item in cases))
        self.assertGreater(sum(1 for item in cases if item["included"]), 20)
        self.assertRegex(hashlib.sha256(path.read_bytes()).hexdigest(), r"^[0-9a-f]{64}$")

    def test_role_grounding_baseline_covers_fifty_cases(self):
        path = FIXTURES / "m14_role_grounding_baseline.json"
        payload = json.loads(path.read_text("utf-8"))
        self.assertEqual(list(range(1, 51)), [item["case_id"] for item in payload["cases"]])
        self.assertTrue(any(item["rows"] for item in payload["cases"]))

    def test_case_018_baseline_exposes_unproven_aggregate_roles(self):
        payload = json.loads(
            (FIXTURES / "m14_role_grounding_baseline.json").read_text("utf-8")
        )
        case = next(item for item in payload["cases"] if item["case_id"] == 18)
        row = next(
            item for item in case["rows"]
            if item["requirement_type"] == "scoped_aggregate"
        )
        self.assertEqual("unproven", row["roles"]["measure"]["status"])
        self.assertEqual("unproven", row["roles"]["scope"]["status"])

    def test_expression_signature_rejects_date_as_average_measure(self):
        result = ExpressionSignatureRegistry().validate(
            "avg", [ValueSignature("date", shape="event_set")], semantic_role="measure",
        )
        self.assertFalse(result.valid)
        self.assertEqual("expression_input_type_invalid", result.diagnostic_code)

    def test_case_018_requirement_grounds_measure_scope_and_grain(self):
        result = QueryUnderstandingEngine().analyze(CASE_018_QUERY)
        ir = result.understanding
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "scoped_aggregate"
        )
        self.assertEqual(["field:转换效率"], requirement.expected_inputs)
        self.assertEqual("relation", requirement.expected_scope)
        self.assertEqual("day", requirement.expected_output_grain)
        self.assertTrue(any(
            next(value for value in ir.requirements if value.requirement_id == dependency)
            .requirement_type == "set_operation"
            for dependency in requirement.dependencies
        ))
        proof = RoleGroundingAnalyzer().prove(requirement, ir)
        self.assertTrue(proof.complete)
        self.assertEqual(("转换效率",), proof.role("measure").values)
        target = SemanticTargetBuilder().build(requirement, ir)
        self.assertTrue(target.ready)
        self.assertEqual("avg", target.operation)
        self.assertEqual("day", target.group_by_grain)

    def test_output_lineage_keeps_date_role_separate_from_measure_fields(self):
        ir = QueryUnderstandingEngine().analyze(CASE_018_QUERY).understanding
        lineage = OutputLineageIndex.build(ir)
        date_entries = [lineage.get(item.output.ref_id) for item in ir.derived_projections]
        self.assertTrue(date_entries)
        self.assertTrue(all(item.semantic_role == "date" for item in date_entries))
        self.assertTrue(any("efficiency" in item.source_fields for item in date_entries))

    def test_event_closure_report_exposes_typed_producer_consumer_edges(self):
        result = QueryUnderstandingEngine().analyze(CASE_018_QUERY)
        report = result.event_closure_report
        self.assertIsNotNone(report)
        edges = set(report.producer_consumer_edges)
        event_refs = {
            item.output_ref.ref_id for item in result.understanding.events if item.output_ref
        }
        projection_refs = {
            item.output.ref_id for item in result.understanding.derived_projections
        }
        self.assertTrue(any(source in event_refs and target in projection_refs
                            for source, target in edges))
        self.assertEqual("partial", report.structure_status)
        self.assertTrue(any(item.startswith("requirement:")
                            for item in report.unresolved_dependencies))

    def test_event_closure_is_field_neutral_for_supported_predicates(self):
        fields = [
            ("温度", "temperature", "85℃"),
            ("压力", "pressure", "5MPa"),
            ("速度", "speed", "80km/h"),
            ("库存", "inventory", "50"),
            ("功率", "power", "7W"),
        ]
        for raw_name, identifier, threshold in fields:
            with self.subTest(field=identifier):
                query = (
                    f"设备数据包含{raw_name}(`{identifier}`)。找出{raw_name}连续超过"
                    f"{threshold}且持续超过5分钟的所有时间段，输出异常日期。"
                )
                result = QueryUnderstandingEngine().analyze(query)
                self.assertEqual(1, len(result.understanding.events))
                event_ref = result.understanding.events[0].output_ref.ref_id
                self.assertTrue(any(
                    source == event_ref for source, _
                    in result.event_closure_report.producer_consumer_edges
                ))

    def test_event_closure_marks_an_unconsumed_event_as_disconnected(self):
        ir = copy.deepcopy(QueryUnderstandingEngine().analyze(CASE_018_QUERY).understanding)
        isolated = ir.events[0].output_ref.ref_id
        ir.derived_projections = [
            item for item in ir.derived_projections
            if item.input_ref.ref_id != isolated
        ]
        report = EventClosureAnalyzer().analyze(ir)
        self.assertIn(f"disconnected_event:{isolated}", report.unresolved_dependencies)
        self.assertEqual("partial", report.structure_status)

    def test_event_closure_accepts_a_simple_event_as_terminal(self):
        query = (
            "设备数据包含温度(`temperature`)。找出温度连续超过85℃"
            "且持续超过5分钟的所有时间段。"
        )
        result = QueryUnderstandingEngine().analyze(query)
        event_ref = result.understanding.events[0].output_ref.ref_id
        self.assertNotIn(
            f"disconnected_event:{event_ref}",
            result.event_closure_report.unresolved_dependencies,
        )

    def test_semantic_target_gate_flag_controls_aggregate_role_proof(self):
        baseline = QueryUnderstandingEngine(config=EngineConfig(
            enable_candidate_choice_v3=True,
        )).analyze(CASE_018_QUERY).understanding
        requirement = next(
            item for item in baseline.requirements
            if item.requirement_type == "scoped_aggregate"
        )
        current_inputs = set(requirement.expected_attributes.get("input_demand_ids", []))
        unrelated_field = next(
            item.demand_id for item in baseline.source_demands
            if item.demand_type == "field" and item.demand_id not in current_inputs
        )
        requirement.expected_attributes.setdefault("input_demand_ids", []).append(
            unrelated_field
        )
        strict = copy.deepcopy(baseline)
        compatibility = copy.deepcopy(baseline)
        CoverageMatcher(enforce_semantic_target_gate=True).apply(strict)
        CoverageMatcher(enforce_semantic_target_gate=False).apply(compatibility)
        requirement_id = next(
            item.requirement_id for item in strict.requirements
            if item.requirement_type == "scoped_aggregate"
        )
        strict_status = next(
            item.status for item in strict.coverage
            if item.requirement_id == requirement_id
        )
        compatibility_status = next(
            item.status for item in compatibility.coverage
            if item.requirement_id == requirement_id
        )
        self.assertEqual("uncovered", strict_status)
        self.assertEqual("satisfied", compatibility_status)

    def test_case_018_unique_aggregate_candidate_compiles_locally(self):
        engine = QueryUnderstandingEngine(config=EngineConfig(
            enable_candidate_choice_v3=True,
        ))
        result = engine.analyze(CASE_018_QUERY)
        self.assertEqual(0, result.model_calls)
        self.assertIsNotNone(result.patch_report)
        self.assertTrue(result.patch_report.committed, result.patch_report.transaction_error)
        aggregate = next(item for item in result.understanding.aggregates
                         if item.aggregate.function == "avg")
        self.assertEqual("转换效率", aggregate.aggregate.input.symbol.raw_name)
        self.assertEqual("relation", aggregate.scope)
        self.assertEqual("date", aggregate.group_by[0].result_type)

    def test_dynamic_baseline_does_not_shadow_measured_speed_aggregate(self):
        ir = QueryUnderstandingEngine().analyze(CASE_001_QUERY).understanding
        speed_hypothesis = next(
            item for item in ir.schema_hypotheses if item.raw_name == "speed"
        )
        baseline_hypothesis = next(
            item for item in ir.schema_hypotheses if item.raw_name == "额定转速"
        )
        aggregate = next(
            item for item in ir.aggregates
            if item.aggregate.input.symbol
            and item.aggregate.input.symbol.raw_name == "主轴转速"
        )
        identifiers = {
            item.identifier for item in aggregate.aggregate.input.symbol.candidates
        }
        self.assertIn(speed_hypothesis.hypothesis_id, identifiers)
        self.assertNotIn(baseline_hypothesis.hypothesis_id, identifiers)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "aggregate"
            and item.expected_inputs == ["field:转速"]
        )
        coverage = next(
            item for item in ir.coverage
            if item.requirement_id == requirement.requirement_id
        )
        self.assertNotIn("input_fields", coverage.missing_attributes)
        self.assertTrue(
            all(item.startswith("dependency:") for item in coverage.missing_attributes),
            coverage.missing_attributes,
        )

    def test_event_candidate_compiles_from_exact_predicate_and_duration_anchors(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_008_QUERY).understanding)
        requirement = min(
            (item for item in ir.requirements
             if item.requirement_type == "sequence_event"),
            key=lambda item: item.span.start,
        )
        event = next(
            item for item in ir.events
            if any(prov.span and requirement.span
                   and prov.span.start < requirement.span.end
                   and requirement.span.start < prov.span.end
                   for prov in item.provenance)
        )
        event_ref = event.output_ref.ref_id
        ir.events.remove(event)
        ir.derived_projections = [
            item for item in ir.derived_projections if item.input_ref.ref_id != event_ref
        ]
        ir.set_operations.clear()
        ir.aggregates.clear()
        ir.calculations.clear()
        gap, menu = self._gap_and_menu(
            engine, ir, "missing_event", requirement.requirement_id,
        )
        self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
        payload = menu.candidates[0].semantic_payload
        self.assertEqual("lt", payload["conditions"][0]["operator"])
        self.assertEqual(200.0, payload["conditions"][0]["value"])
        self.assertEqual(20.0, payload["duration_value"])
        self.assertEqual("minute", payload["duration_unit"])
        patch, error = engine.semantic_candidate_compiler.compile_patch(
            menu.candidates[0], gap, ir,
        )
        self.assertIsNotNone(patch, error)
        self.assertEqual("add_event", patch.operations[0].operation_type)

    def test_set_candidate_compiles_two_distinct_date_producers(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_008_QUERY).understanding)
        requirement = next(
            item for item in ir.requirements if item.requirement_type == "set_operation"
        )
        ir.set_operations.clear()
        ir.aggregates.clear()
        ir.calculations.clear()
        gap, menu = self._gap_and_menu(
            engine, ir, "missing_set_inputs", requirement.requirement_id,
        )
        self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
        payload = menu.candidates[0].semantic_payload
        self.assertEqual("intersection", payload["operation"])
        self.assertEqual("date", payload["granularity"])
        self.assertEqual(2, len(set(payload["inputs"])))
        patch, error = engine.semantic_candidate_compiler.compile_patch(
            menu.candidates[0], gap, ir,
        )
        self.assertIsNotNone(patch, error)
        self.assertEqual("add_set_operation", patch.operations[0].operation_type)

    def test_calculation_candidate_preserves_explicit_source_operand_order(self):
        engine = QueryUnderstandingEngine()
        ir = copy.deepcopy(engine.analyze(CASE_008_QUERY).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "calculation" and item.operator_family == "ratio"
        )
        ir.calculations.clear()
        gap, menu = self._gap_and_menu(
            engine, ir, "missing_formula_input", requirement.requirement_id,
        )
        self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
        operands = menu.candidates[0].semantic_payload["operands"]
        expected_operands = [
            item.aggregate.output.ref_id for item in sorted(
                ir.aggregates,
                key=lambda item: item.aggregate.provenance[0].span.start,
            )
        ]
        self.assertEqual(
            expected_operands, operands,
        )
        patch, error = engine.semantic_candidate_compiler.compile_patch(
            menu.candidates[0], gap, ir,
        )
        self.assertIsNotNone(patch, error)
        self.assertEqual("add_calculation", patch.operations[0].operation_type)

    def test_reference_candidate_uses_the_only_date_role_output(self):
        query = (
            "设备温度数据。找出温度连续超过85℃且持续超过5分钟的所有时间段，"
            "输出异常日期，并计算该日期的温度平均值。"
        )
        engine = QueryUnderstandingEngine()
        ir = engine.analyze(query).understanding
        requirement = next(
            item for item in ir.requirements if item.requirement_type == "reference"
        )
        gap, menu = self._gap_and_menu(
            engine, ir, "unresolved_reference", requirement.requirement_id,
        )
        self.assertEqual("local_compile", menu.dispatch, menu.blocked_by)
        self.assertEqual(
            "project_event_1_consecutive_date.dates",
            menu.candidates[0].semantic_payload["target_ref"],
        )
        patch, error = engine.semantic_candidate_compiler.compile_patch(
            menu.candidates[0], gap, ir,
        )
        self.assertIsNotNone(patch, error)
        self.assertEqual("resolve_reference", patch.operations[0].operation_type)

    def test_candidate_decision_rejects_unknown_content_addressed_id(self):
        engine = QueryUnderstandingEngine()
        result = engine.analyze(CASE_018_QUERY)
        gap = next(item for item in engine.gap_analyzer.analyze(
            result.understanding, engine.catalog_provider.snapshot(),
        ) if item.gap_type == "missing_aggregate_scope")
        menu = SemanticCandidateCompiler().compile(
            result.understanding, gap, engine.catalog_provider.snapshot(),
        )
        decision = parse_candidate_decision(
            '{"decision":"select","candidate_id":"cand:sha256:' + '0' * 64 + '"}',
            menu,
        )
        self.assertEqual("candidate_not_allowed", decision.error)

    def test_candidate_choice_llm_returns_only_allowed_id_in_one_call(self):
        menu = _choice_menu()
        provider = ChoiceProvider(
            '{"decision":"select","candidate_id":"choice_1"}'
        )
        result = BoundedLLMExtractor(provider).choose_candidate(menu)
        self.assertTrue(result.valid_json, result.error)
        self.assertEqual(1, result.attempts)
        self.assertEqual(1, provider.calls)
        # The source does not define numerator/denominator, so the semantic
        # guard must not accept an arbitrary provider selection.
        self.assertEqual("ambiguous", result.payload["decision"])
        self.assertNotIn("candidate_id", result.payload)

    def test_candidate_choice_does_not_repair_or_retry_truncated_output(self):
        provider = ChoiceProvider('{"decision":"select","candidate_id":"cand:sha256:1')
        result = BoundedLLMExtractor(
            provider, max_remote_attempts=2, allow_json_repair=True,
        ).choose_candidate(_choice_menu())
        self.assertFalse(result.valid_json)
        self.assertEqual("json_syntax_failure", result.error)
        self.assertEqual(1, provider.calls)

    def test_case_018_wrong_average_replay_is_rejected_without_coverage(self):
        engine = QueryUnderstandingEngine()
        result = engine.analyze(CASE_018_QUERY)
        ir = result.understanding
        date_output = next(item.output for item in ir.derived_projections
                           if item.output.result_type == "date")
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "scoped_aggregate"
        )
        wrong_id = "aggregate_wrong_date_measure"
        ir.aggregates.append(ScopedAggregateSpec(
            aggregate=AggregateSpec(
                aggregate_id=wrong_id,
                function="avg",
                input=ExpressionNode(
                    "output_ref", reference=date_output.ref_id, result_type="date",
                ),
                output=OutputRef(
                    wrong_id, "value", "number", fields=["avg"],
                ),
                provenance=[Provenance(
                    source="llm", span=requirement.span, rule="case_018_wrong_replay",
                )],
            ),
            scope="filtered",
        ))
        engine.coverage_matcher.apply(ir)
        coverage = next(item for item in ir.coverage
                        if item.requirement_id == requirement.requirement_id)
        self.assertEqual("uncovered", coverage.status)
        validation = engine.validator.validate(ir, engine.catalog_provider.snapshot())
        self.assertIn("aggregate_signature_invalid", {item.code for item in validation.errors})


if __name__ == "__main__":
    unittest.main()
