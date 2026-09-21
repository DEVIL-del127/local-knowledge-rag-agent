from __future__ import annotations

import copy
import json
import os
import unittest
import uuid
from dataclasses import asdict
from datetime import datetime
from unittest.mock import patch

from nlu_v2 import EngineConfig, QueryUnderstandingEngine, SecurityContext
from nlu_v2.catalog import StaticCatalogProvider, default_sources
from nlu_v2.legacy_adapter import to_legacy_route
from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider
from nlu_v2.merge import _recover_duration_span, walk_predicates
from nlu_v2.models import Diagnostic, ExpressionNode, LiteralValue, OutputRef, SourceSpan
from nlu_v2.validator import _infer_expression
from nlu_v2.vector_candidates import CallableVectorCandidateProvider
from nlu_v2_validate import build_parser


SKU_QUERY = (
    "分析某零售店 SKU(库存单位)的销售数据。请提取 2025年 和 2026年 的 "
    "6月、7月、8月 这三个月份的销售总额。并分别计算:环比增长:"
    "2026年7月 vs 2026年6月,以及 2026年8月 vs 2026年7月。"
    "同比增长:2026年7月 vs 2025年7月,以及 2026年8月 vs 2025年8月。"
    "最终输出:哪两个月份的组合同时实现了环比和同比的正向增长?"
)

TELEMETRY_QUERY = (
    "监测设备的温度传感器。找出 2026年1月1日至今，温度连续超过 85℃ "
    "超过 10分钟 的所有时间段，以及 累计超过 80℃ 但未超过 85℃ 的总时长"
    "超过 6小时 的所有日期。输出这两类异常时间段的交集日期（仅日期，忽略 "
    "具体分钟），并计算该日期当月截至目前的日均电压波动率。"
)


class FakeProvider:
    def __init__(self, response="{}"):
        self.response = response
        self.calls = 0
        self.budgets = []

    def complete(self, prompt, *, deadline, attempt_budget):
        self.calls += 1
        self.budgets.append(attempt_budget)
        return self.response


class FailingProvider(FakeProvider):
    def complete(self, prompt, *, deadline, attempt_budget):
        self.calls += 1
        self.budgets.append(attempt_budget)
        raise TimeoutError("simulated timeout")


class NluV2Tests(unittest.TestCase):
    def test_cli_defaults_to_full_llm_chain(self):
        parser = build_parser()
        self.assertTrue(parser.parse_args([]).llm)
        self.assertFalse(parser.parse_args(["--no-llm"]).llm)
        self.assertFalse(parser.parse_args([]).candidate_choice_v3)
        self.assertTrue(
            parser.parse_args(["--candidate-choice-v3"]).candidate_choice_v3
        )

    def test_wsl_default_ollama_url_uses_host_gateway(self):
        completed = type("Completed", (), {"stdout": "default via 192.168.144.1 dev eth0"})()
        with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True):
            with patch("nlu_v2.llm_extractor.subprocess.run", return_value=completed):
                provider = OllamaOpenAIProvider()
        self.assertEqual(provider.base_url, "http://192.168.144.1:11434")

    def test_time_series_query_is_scoped_to_structured_analytics_skill(self):
        result = QueryUnderstandingEngine().analyze(TELEMETRY_QUERY)
        ir = result.understanding
        self.assertEqual(
            [item.identifier for item in ir.source_candidates], ["device_telemetry"]
        )
        self.assertEqual(len(ir.events), 2)
        self.assertTrue(ir.events[0].derived_metric.metric_id.endswith("consecutive_duration"))
        self.assertEqual(ir.events[0].threshold.normalized_seconds, 600.0)
        self.assertEqual(ir.events[0].sampling.max_gap_seconds, 120)
        self.assertTrue(ir.events[1].derived_metric.metric_id.endswith("cumulative_duration"))
        self.assertEqual(ir.events[1].threshold.normalized_seconds, 21600.0)
        cumulative_ops = [
            item.operator for item in walk_predicates(ir.events[1].condition)
        ]
        self.assertEqual(cumulative_ops, ["gt", "lte"])
        self.assertEqual(ir.temporal[0].to_value, datetime.now().strftime("%Y-%m-%d"))
        self.assertEqual(ir.set_operations[0].operation, "intersection")
        self.assertEqual(ir.references[0].status, "resolved")
        self.assertEqual(
            ir.calculations[0].parameters["input_field"], "device_telemetry.voltage"
        )
        self.assertEqual(ir.calculations[0].parameters["window"], "month_to_date")
        self.assertEqual(ir.output.fields, [
            "abnormal_intersection_dates", "daily_voltage_volatility",
        ])
        task_types = [item.task_type for item in result.logical_plan.nodes]
        self.assertEqual(task_types, ["retrieve", "calculate", "compose"])
        analytics = result.logical_plan.nodes[1]
        self.assertEqual(analytics.inputs["capability"], "structured_analytics")
        self.assertEqual(len(analytics.inputs["events"]), 2)
        self.assertEqual(len(analytics.inputs["set_operations"]), 1)
        self.assertEqual(len(analytics.inputs["calculations"]), 1)
        self.assertEqual(result.validation.status, "valid")
        self.assertEqual(result.logical_plan.status, "ready")
        self.assertEqual(result.model_calls, 0)

    def test_incomplete_time_series_semantics_are_blocked(self):
        result = QueryUnderstandingEngine().analyze(
            "监测设备2026年1月以来温度连续超过85℃的时间段"
        )
        self.assertFalse(result.validation.executable)
        self.assertTrue(any(
            item.code == "event_structure_incomplete" for item in result.understanding.unresolved
        ))

    def test_missing_time_series_capability_is_unsupported(self):
        sources = default_sources()
        device = next(item for item in sources if item.source_id == "device_telemetry")
        device.capabilities.remove("event_segmentation")
        result = QueryUnderstandingEngine(
            catalog_provider=StaticCatalogProvider(sources)
        ).analyze(TELEMETRY_QUERY)
        self.assertEqual(result.validation.status, "unsupported")
        self.assertTrue(any(
            item.code == "derived_capability_missing" for item in result.validation.errors
        ))

    def test_weather_predicates_are_typed_and_single_source(self):
        result = QueryUnderstandingEngine().analyze(
            "查询温度大于30摄氏度且风速3到8米每秒的数据"
        )
        predicates = list(walk_predicates(result.understanding.filters))
        self.assertEqual([item.operator for item in predicates], ["gt", "between"])
        self.assertEqual([item.value.unit for item in predicates], ["celsius", "m/s"])
        self.assertEqual(result.logical_plan.nodes[0].source_id, "weather_observations")
        self.assertTrue(result.validation.executable)

    def test_units_boolean_and_repeated_field_conditions_are_normalized(self):
        engine = QueryUnderstandingEngine()
        converted = engine.analyze("查询温度大于86华氏度且风速36公里每小时的数据")
        values = list(walk_predicates(converted.understanding.filters))
        self.assertEqual([item.value.value for item in values], [30.0, 10.0])
        self.assertEqual([item.value.unit for item in values], ["celsius", "m/s"])

        bounded = engine.analyze("查询温度大于20且小于30摄氏度的数据")
        self.assertEqual(
            [item.operator for item in walk_predicates(bounded.understanding.filters)],
            ["gt", "lt"],
        )

        alternatives = engine.analyze("查询温度大于30或风速小于2米每秒的数据")
        self.assertEqual(alternatives.understanding.filters.operator, "or")

        precedence = engine.analyze("查询温度大于30或风速大于2且小于5米每秒的数据")
        root = precedence.understanding.filters
        self.assertEqual(root.operator, "or")
        self.assertEqual(root.children[1].operator, "and")

    def test_categorical_predicate_is_structured(self):
        result = QueryUnderstandingEngine().analyze("查询语言为中文的论文")
        predicates = list(walk_predicates(result.understanding.filters))
        self.assertEqual(len(predicates), 1)
        self.assertEqual(predicates[0].operator, "eq")
        self.assertEqual(predicates[0].value.value, "中文")

    def test_sku_workflow_expands_periods_and_comparisons(self):
        result = QueryUnderstandingEngine().analyze(SKU_QUERY)
        periods = [(item.from_value or item.exact)[:7] for item in result.understanding.temporal]
        self.assertEqual(periods, [
            "2025-06", "2025-07", "2025-08", "2026-06", "2026-07", "2026-08",
        ])
        pairs = [
            (item.calculation_type, item.parameters["left"], item.parameters["right"])
            for item in result.understanding.calculations
        ]
        self.assertEqual(pairs, [
            ("mom", "2026-07", "2026-06"),
            ("mom", "2026-08", "2026-07"),
            ("yoy", "2026-07", "2025-07"),
            ("yoy", "2026-08", "2025-08"),
        ])
        self.assertEqual({node.source_id for node in result.logical_plan.nodes}, {"sales_data"})
        compute = [node for node in result.logical_plan.nodes if node.task_type == "calculate"]
        self.assertEqual(len(compute), 4)
        self.assertTrue(all(node.depends_on == ["t2_aggregate"] for node in compute))

    def test_unknown_field_blocks_instead_of_semantic_fallback(self):
        result = QueryUnderstandingEngine().analyze("查询体感舒适指数大于5的数据")
        self.assertEqual(result.validation.status, "needs_clarification")
        self.assertFalse(result.validation.executable)
        self.assertTrue(any(item.code == "unknown_field" for item in result.understanding.unresolved))
        self.assertEqual(result.logical_plan.status, "blocked")

    def test_mixed_sources_are_unsupported(self):
        result = QueryUnderstandingEngine().analyze("查询温度大于30度时的销售总额")
        self.assertEqual(result.validation.status, "unsupported")
        self.assertTrue(any(item.code == "multi_source_unsupported" for item in result.validation.errors))

    def test_write_requests_are_hard_blocked(self):
        result = QueryUnderstandingEngine().analyze("删除知识库里2024年的论文")
        self.assertEqual(result.validation.status, "unsupported")
        self.assertFalse(result.validation.executable)
        self.assertTrue(any(item.code == "write_operation" for item in result.validation.errors))

    def test_simple_rules_do_not_call_enabled_llm(self):
        provider = FakeProvider()
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        result = engine.analyze("查询温度大于30摄氏度的数据")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(result.model_calls, 0)
        self.assertFalse(result.validation.executable)

    def test_complex_query_calls_llm_at_most_once_by_default(self):
        provider = FakeProvider("{}")
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True, llm_on_complex=True),
            llm_extractor=BoundedLLMExtractor(provider, allow_json_repair=False),
        )
        result = engine.analyze(SKU_QUERY)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.model_calls, 1)
        self.assertLessEqual(max(provider.budgets), 2)
        self.assertTrue(result.validation.executable)

    def test_malformed_llm_cannot_overwrite_rule_nodes(self):
        provider = FakeProvider('''{
          "metrics": [{"field_id":"market_data.price","aggregation":"sum",
                       "span":{"start":0,"end":2,"text":"伪造"}}],
          "filters": [{"field_id":"weather_observations.temperature","operator":"gt",
                       "value":999,"span":{"start":0,"end":2,"text":"伪造"}}]
        }''')
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        result = engine.analyze(SKU_QUERY)
        metric_ids = {item.field.canonical_id for item in result.understanding.metrics}
        self.assertNotIn("market_data.price", metric_ids)
        self.assertEqual(len(result.understanding.calculations), 4)
        self.assertTrue(result.validation.executable)

    def test_catalog_version_is_stable_and_schema_sensitive(self):
        first = StaticCatalogProvider().snapshot()
        second = StaticCatalogProvider().snapshot()
        self.assertEqual(first.version, second.version)
        changed = copy.deepcopy(default_sources())
        changed[0].fields[0].aliases.append("新标题别名")
        third = StaticCatalogProvider(changed).snapshot()
        self.assertNotEqual(first.version, third.version)

    def test_physical_plan_is_never_bound(self):
        result = QueryUnderstandingEngine().analyze("查询2024年的论文")
        self.assertEqual(result.physical_plan.status, "unbound")
        self.assertEqual(result.physical_plan.bindings, [])

    def test_legacy_conversion_accepts_only_valid_private_kb(self):
        private_result = QueryUnderstandingEngine().analyze("查询2024年的论文")
        route = to_legacy_route(private_result)
        self.assertEqual(route.source, "nlu_v2_legacy_adapter")
        self.assertEqual(route.entities.year_exact, 2024)

        weather_result = QueryUnderstandingEngine().analyze("查询温度大于30摄氏度的数据")
        with self.assertRaises(ValueError):
            to_legacy_route(weather_result)

    def test_json_repair_still_has_hard_two_attempt_cap(self):
        provider = FakeProvider("not json")
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider, allow_json_repair=True),
        )
        result = engine.analyze(
            "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(provider.budgets, [1, 1])
        self.assertFalse(result.validation.executable)

    def test_transport_failure_is_counted_without_retry(self):
        provider = FailingProvider()
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider, allow_json_repair=True),
        ).analyze("设备数据中查询温度一直高于85℃并持续10分钟的区间")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.budgets, [1])
        self.assertEqual(result.model_calls, 1)
        self.assertTrue(any(
            item.code in {"provider_error", "provider_timeout"}
            for item in result.understanding.diagnostics
        ))

    def test_valid_empty_llm_candidate_does_not_open_breaker(self):
        provider = FakeProvider("{}")
        engine = QueryUnderstandingEngine(
            config=EngineConfig(
                enable_llm=True, llm_on_complex=True, breaker_failures=1
            ),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        first = engine.analyze(SKU_QUERY)
        second = engine.analyze(SKU_QUERY + " 请用表格展示")
        self.assertEqual(first.model_calls, 1)
        self.assertEqual(second.model_calls, 1)
        self.assertEqual(provider.calls, 2)

    def test_llm_can_fill_evidence_backed_consecutive_event(self):
        query = (
            "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
        )

        def span(text):
            start = query.index(text)
            return {"start": start, "end": start + len(text), "text": text}

        payload = {
            "events": [{
                "event_id": "high_temperature",
                "metric_id": "device_telemetry.consecutive_duration",
                "logic": "and",
                "conditions": [{
                    "field_id": "device_telemetry.temperature",
                    "operator": "gt", "value": 85, "unit": "celsius",
                    "span": span("温度一直高于85℃"),
                }],
                "duration": {
                    "operator": "gte", "value": 10, "unit": "minute",
                    "span": span("10分钟"),
                },
                "group_by": ["local_date"],
                "span": span("温度一直高于85℃并持续10分钟以上"),
            }]
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(query)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(result.understanding.events), 1)
        self.assertEqual(
            result.understanding.events[0].threshold.normalized_seconds, 600.0
        )
        self.assertEqual(
            result.understanding.events[0].provenance[0].source, "llm"
        )
        self.assertEqual(result.validation.status, "valid")
        self.assertEqual(
            [item.task_type for item in result.logical_plan.nodes],
            ["retrieve", "calculate", "compose"],
        )
        self.assertEqual(
            result.logical_plan.nodes[1].inputs["capability"], "structured_analytics"
        )

    def test_llm_event_with_forged_span_is_rejected(self):
        query = "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
        payload = {
            "events": [{
                "event_id": "forged",
                "metric_id": "device_telemetry.consecutive_duration",
                "conditions": [{
                    "field_id": "device_telemetry.temperature",
                    "operator": "gt", "value": 85,
                    "span": {"start": 0, "end": 2, "text": "伪造"},
                }],
                "duration": {
                    "operator": "gt", "value": 10, "unit": "minute",
                    "span": {"start": 0, "end": 2, "text": "伪造"},
                },
                "span": {"start": 0, "end": 2, "text": "伪造"},
            }]
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(query)
        self.assertEqual(result.understanding.events, [])
        self.assertFalse(result.validation.executable)
        self.assertTrue(any(
            item.code == "llm_nodes_rejected" for item in result.understanding.diagnostics
        ))

    def test_unique_text_only_evidence_is_resolved_locally(self):
        query = "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
        payload = {
            "events": [{
                "event_id": "text_evidence",
                "metric_id": "device_telemetry.consecutive_duration",
                "conditions": [{
                    "field_id": "device_telemetry.temperature",
                    "operator": "gt", "value": 85, "unit": "celsius",
                    "span": {"text": "温度一直高于85℃"},
                }],
                "duration": {
                    "operator": "gte", "value": 10, "unit": "minute",
                    "span": {"text": "10分钟"},
                },
                "span": {"text": "温度一直高于85℃并持续10分钟以上"},
            }]
        }
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(
                FakeProvider(json.dumps(payload, ensure_ascii=False))
            ),
        ).analyze(query)
        self.assertEqual(result.validation.status, "valid")
        self.assertEqual(result.understanding.events[0].provenance[0].span.text,
                         "温度一直高于85℃并持续10分钟以上")

    def test_oversized_query_is_rejected_before_any_model_call(self):
        provider = FakeProvider("{}")
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True, max_query_chars=20),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        with self.assertRaises(ValueError):
            engine.analyze("查询" + "很长" * 20)
        self.assertEqual(provider.calls, 0)

    def test_vector_recall_is_evidence_not_an_executable_binding(self):
        vector = CallableVectorCandidateProvider(
            lambda **_: [{"identifier": "private_kb", "score": 0.99}]
        )
        result = QueryUnderstandingEngine(
            vector_provider=vector,
            config=EngineConfig(enable_vector=True),
        ).analyze("解释一个未登记的内部概念")
        candidate = next(
            item for item in result.understanding.source_candidates
            if item.identifier == "private_kb"
        )
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(result.validation.executable)
        self.assertTrue(any(
            item.claim_type == "source_binding" and item.status == "candidate"
            for item in result.understanding.claims
        ))

    def test_security_context_is_required_in_strict_mode(self):
        engine = QueryUnderstandingEngine(
            config=EngineConfig(require_security_context=True)
        )
        with self.assertRaises(PermissionError):
            engine.analyze("查询2024年的论文")

    def test_authorized_catalog_hides_forbidden_fields(self):
        security = SecurityContext(
            tenant_id="tenant-a", principal_id="user-a",
            allowed_sources=("weather_observations",),
            allowed_fields=("weather_observations.wind_speed",),
        )
        result = QueryUnderstandingEngine().analyze(
            "查询天气温度大于30摄氏度的数据", security_context=security
        )
        serialized = json.dumps(result.understanding.to_dict(), ensure_ascii=False)
        self.assertNotIn("weather_observations.temperature", serialized)
        self.assertFalse(result.validation.executable)

    def test_authorized_catalog_does_not_leak_derived_schema_metadata(self):
        security = SecurityContext(
            tenant_id="tenant-a", principal_id="user-a",
            allowed_sources=("device_telemetry",),
            allowed_fields=("device_telemetry.temperature",),
        )
        engine = QueryUnderstandingEngine()
        authorized = engine.catalog_authorizer.authorize(
            engine.catalog_provider.snapshot(), security
        )
        serialized = json.dumps(
            [asdict(item) for item in authorized.sources],
            ensure_ascii=False,
        )
        device = authorized.source("device_telemetry")
        self.assertEqual(device.derived_metrics, [])
        self.assertNotIn("timestamp_field", device.metadata)
        self.assertNotIn("device_telemetry.voltage", serialized)

    def test_llm_cache_is_isolated_by_security_scope(self):
        query = "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
        payload = {
            "events": [{
                "event_id": "high_temperature",
                "metric_id": "device_telemetry.consecutive_duration",
                "conditions": [{
                    "field_id": "device_telemetry.temperature",
                    "operator": "gt", "value": 85, "unit": "celsius",
                    "span": {"text": "温度一直高于85℃"},
                }],
                "duration": {
                    "operator": "gte", "value": 10, "unit": "minute",
                    "span": {"text": "10分钟"},
                },
                "span": {"text": "温度一直高于85℃并持续10分钟以上"},
            }]
        }
        provider = FakeProvider(json.dumps(payload, ensure_ascii=False))
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        first = SecurityContext("tenant-a", "user-a")
        second = SecurityContext("tenant-b", "user-b")
        first_result = engine.analyze(query, security_context=first)
        second_result = engine.analyze(query, security_context=second)
        cached_result = engine.analyze(query, security_context=first)
        self.assertEqual(provider.calls, 2)
        self.assertTrue(first_result.patch_report.committed)
        self.assertTrue(second_result.patch_report.committed)
        self.assertEqual(cached_result.model_calls, 0)

    def test_request_ir_reports_multidimensional_validation(self):
        result = QueryUnderstandingEngine().analyze(
            "查询温度大于30摄氏度且风速3到8米每秒的数据"
        )
        self.assertEqual(result.understanding.understanding_status, "complete")
        self.assertEqual(result.understanding.binding_status, "bound")
        self.assertTrue(result.validation.understanding_complete)
        self.assertTrue(result.validation.binding_complete)
        self.assertEqual(
            result.validation.dimensions["requirement_coverage"]["status"], "pass"
        )
        self.assertTrue(all(
            node.security_scope_digest for node in result.logical_plan.nodes
        ))

    def test_duration_evidence_recovery_requires_one_exact_match(self):
        query = "温度一直高于85℃并持续10分钟以上"
        event_span = SourceSpan(0, len(query), query)
        recovered = _recover_duration_span(event_span, query, 600.0)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.text, "10分钟以上")

        ambiguous = "温度一直高于85℃并持续10分钟或600秒以上"
        self.assertIsNone(_recover_duration_span(
            SourceSpan(0, len(ambiguous), ambiguous), ambiguous, 600.0
        ))

    def test_inline_schema_supports_scoped_aggregate_comparison(self):
        query = (
            "用户画像包含 `user_id`、`total_spend`（累计消费金额）、"
            "`login_days`（登录天数）。找出累计消费金额超过10000元以及登录天数"
            "超过200天的用户。输出同时满足条件的用户 ID，并计算这些用户的平均"
            "累计消费金额是否高于全站平均值的1.5倍。"
        )
        result = QueryUnderstandingEngine().analyze(query)
        ir = result.understanding
        self.assertEqual({item.raw_name for item in ir.schema_hypotheses},
                         {"user_id", "total_spend", "login_days"})
        self.assertEqual(len(list(walk_predicates(ir.filters))), 2)
        self.assertTrue(any(item.raw_name == "user_id" for item in ir.projections))
        self.assertEqual([item.scope for item in ir.aggregates], ["filtered", "global"])
        self.assertEqual(ir.aggregates[0].aggregate.input.symbol.raw_name, "累计消费金额")
        self.assertEqual(ir.comparisons[0].right.operator, "mul")
        self.assertFalse(result.validation.executable)
        self.assertEqual(result.logical_plan.status, "blocked")

    def test_generic_sequence_sampling_set_and_volatility(self):
        query = (
            "驾驶数据每2秒记录一次加速度（`acceleration`）。找出2026年6月1日至"
            "2026年8月24日期间，加速度连续超过2.5 m/s²且持续超过5秒的时间段，"
            "以及加速度连续低于-2.5 m/s²且持续超过5秒的时间段。输出交集日期，"
            "并计算这些日期中加速度的波动率。"
        )
        result = QueryUnderstandingEngine().analyze(query)
        ir = result.understanding
        self.assertEqual(ir.sampling_policies[0].expected_interval_seconds, 2)
        self.assertEqual(ir.temporal[0].to_value, "2026-08-24")
        self.assertEqual([item.condition.operator for item in ir.events], ["gt", "lt"])
        self.assertEqual(len(ir.derived_projections), 2)
        self.assertEqual(len(ir.set_operations), 1)
        self.assertEqual(len(set(ir.set_operations[0].inputs)), 2)
        self.assertEqual(ir.calculations[0].expression, "stddev(acceleration)")
        self.assertNotIn("voltage", ir.calculations[0].expression)
        self.assertFalse(result.validation.executable)

    def test_sequence_topology_is_invariant_for_generated_field_name(self):
        field_name = "metric_" + uuid.uuid4().hex[:10]
        query = (
            f"每2秒记录一次指标（`{field_name}`）。找出指标连续超过2.5且持续超过5秒"
            "的时间段，以及指标连续低于-2.5且持续超过5秒的时间段，输出交集日期并"
            "计算指标的波动率。"
        )
        ir = QueryUnderstandingEngine().analyze(query).understanding
        self.assertEqual(len(ir.events), 2)
        self.assertEqual(len(ir.set_operations), 1)
        self.assertEqual(ir.calculations[0].parameters["input_field"],
                         ir.schema_hypotheses[0].hypothesis_id)

    def test_llm_unrelated_source_span_cannot_authorize_source(self):
        payload = {"sources": [{
            "source_id": "private_kb", "span": {"text": "结果"}
        }]}
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(
                FakeProvider(json.dumps(payload, ensure_ascii=False))
            ),
        ).analyze("列出结果")
        self.assertFalse(any(
            item.identifier == "private_kb" and item.status == "resolved"
            for item in result.understanding.source_candidates
        ))
        self.assertFalse(result.validation.executable)

    def test_planner_revalidates_ir_instead_of_trusting_stale_report(self):
        engine = QueryUnderstandingEngine()
        result = engine.analyze("天气数据中查询温度大于30摄氏度的数据")
        self.assertTrue(result.validation.executable)
        predicate = next(iter(walk_predicates(result.understanding.filters)))
        predicate.field.canonical_id = None
        predicate.field.status = "candidate"
        stale = engine.planner.plan(
            result.understanding, result.validation,
            engine.catalog_provider.snapshot(),
        )
        self.assertEqual(stale.status, "blocked")

    def test_typed_expression_unit_algebra_rejects_currency_squared(self):
        catalog = StaticCatalogProvider().snapshot()
        outputs = {
            "left.value": OutputRef("left", "value", "number", unit="currency"),
            "right.value": OutputRef("right", "value", "number", unit="currency"),
        }
        errors: list[Diagnostic] = []
        invalid = ExpressionNode(
            kind="binary", operator="mul", arguments=[
                ExpressionNode(kind="output_ref", reference="left.value",
                               result_type="number", unit="currency"),
                ExpressionNode(kind="output_ref", reference="right.value",
                               result_type="number", unit="currency"),
            ], result_type="number", unit="currency",
        )
        _infer_expression(invalid, outputs, catalog, errors)
        self.assertTrue(any(item.code == "compound_unit_unsupported" for item in errors))

        errors = []
        valid = ExpressionNode(
            kind="binary", operator="mul", arguments=[
                ExpressionNode(kind="output_ref", reference="left.value",
                               result_type="number", unit="currency"),
                ExpressionNode(kind="literal", literal=LiteralValue(1.5, "number"),
                               result_type="number"),
            ], result_type="number", unit="currency",
        )
        self.assertEqual(_infer_expression(valid, outputs, catalog, errors),
                         ("number", "currency"))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
