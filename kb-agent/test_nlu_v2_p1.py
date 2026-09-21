from __future__ import annotations

import time
import unittest
from unittest.mock import Mock, patch

from benchmark_nlu_v2_100 import DEFAULT_BENCHMARK_INPUT, parse_questions
from nlu_v2 import QueryUnderstandingEngine
from nlu_v2.clarification import ClarificationPlanner
from nlu_v2.llm_extractor import (
    BoundedLLMExtractor,
    DEFAULT_LLM_TOTAL_DEADLINE_SECONDS,
    OllamaOpenAIProvider,
)
from nlu_v2.models import Diagnostic, QueryEnvelope, UnderstandingIR, ValidationReport
from nlu_v2.patch_protocol import GapRequest


class TimeoutProvider:
    def __init__(self):
        self.calls = 0
        self.budgets = []

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        self.budgets.append(attempt_budget)
        raise TimeoutError("simulated provider timeout")


class StaticProvider:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        return self.text


class P1ReliabilityAndRecoveryTests(unittest.TestCase):
    def test_default_deadline_is_warmed_model_budget_and_provider_splits_timeouts(self):
        self.assertGreaterEqual(DEFAULT_LLM_TOTAL_DEADLINE_SECONDS, 60.0)
        self.assertLessEqual(DEFAULT_LLM_TOTAL_DEADLINE_SECONDS, 90.0)
        extractor = BoundedLLMExtractor(TimeoutProvider())
        self.assertGreaterEqual(extractor.timeout, 60.0)
        self.assertLessEqual(extractor.timeout, 90.0)

        response = Mock()
        response.json.return_value = {"message": {"content": "{}"}}
        provider = OllamaOpenAIProvider(connect_timeout=2.0, read_timeout=8.0)
        with patch("requests.post", return_value=response) as post:
            provider.complete("{}", deadline=time.monotonic() + 4.0, attempt_budget=1)
        timeout = post.call_args.kwargs["timeout"]
        self.assertEqual(timeout[0], 2.0)
        self.assertGreater(timeout[1], 3.0)
        self.assertLessEqual(timeout[1], 4.0)

    def test_provider_timeout_has_one_transport_attempt_and_is_not_clarification(self):
        provider = TimeoutProvider()
        candidate = BoundedLLMExtractor(provider, allow_json_repair=True).extract("查询温度", [], {})
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.budgets, [1])
        self.assertEqual(candidate.failure_kind, "provider")

        validation = ValidationReport(
            status="needs_clarification", executable=False, reliability=0.0,
            errors=[Diagnostic("error", "provider_timeout", "provider timeout")],
        )
        plan = ClarificationPlanner().plan(
            UnderstandingIR(QueryEnvelope("查询", "查询"), "catalog"), validation,
        )
        self.assertIsNone(plan)

        fake_prompt_validation = ValidationReport(
            status="needs_clarification", executable=False, reliability=0.0,
            errors=[Diagnostic("error", "unknown_field", "原始问题与证据文本不匹配")],
        )
        self.assertIsNone(ClarificationPlanner().plan(
            UnderstandingIR(QueryEnvelope("查询", "查询"), "catalog"), fake_prompt_validation,
        ))

    def test_valid_but_semantically_invalid_json_is_not_repaired(self):
        provider = StaticProvider('{"unexpected": true}')
        gap = GapRequest(
            gap_id="gap_event", gap_type="missing_event", query_slice="温度连续超过85℃",
            allowed_operations=["add_event"],
        )
        candidate = BoundedLLMExtractor(provider, allow_json_repair=True).extract(
            "温度连续超过85℃", [], {}, gaps=[gap], base_ir_digest="digest",
            enable_patch_v1=True,
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(candidate.failure_kind, "schema")

    def test_clarification_answers_are_bounded_written_and_recompiled(self):
        engine = QueryUnderstandingEngine()
        initial = engine.analyze("查询体感舒适指数大于5的数据")
        plan = initial.clarification_plan
        self.assertIsNotNone(plan)
        question = plan.questions[0]
        answer = question.candidate_ids[0] if question.candidate_ids else "体感舒适指数"
        resumed = engine.resume_clarification(initial, {question.question_id: answer})
        self.assertEqual(resumed.understanding.clarification_answers[question.question_id], answer)
        self.assertNotEqual(initial.compilation_snapshot.ir_digest,
                            resumed.compilation_snapshot.ir_digest)
        self.assertIn("clarification_resume", [item.stage for item in resumed.trace])

    def test_reliability_is_decomposed_and_llm_effect_is_explicit(self):
        result = QueryUnderstandingEngine().analyze("查询温度大于30摄氏度的数据")
        dimensions = result.validation.dimensions
        self.assertTrue({
            "requirement_coverage", "semantic_consistency", "schema_binding",
            "operator_completeness", "executability", "llm_effect",
        } <= set(dimensions))
        self.assertEqual(dimensions["llm_effect"]["status"], "not_attempted")
        self.assertTrue(all("score" in dimensions[name] for name in dimensions))

    def test_benchmark_default_is_a_project_local_complete_fixture(self):
        self.assertTrue(DEFAULT_BENCHMARK_INPUT.is_file())
        self.assertEqual([item["id"] for item in parse_questions(DEFAULT_BENCHMARK_INPUT)],
                         list(range(1, 101)))


if __name__ == "__main__":
    unittest.main()
