"""Optional real Ollama acceptance test.

Set KB_RUN_OLLAMA_TESTS=1 to enable it. Normal CI remains offline and deterministic.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from nlu_v2_validate import build_engine
from nlu_v2.llm_extractor import BoundedLLMExtractor, OllamaOpenAIProvider
from nlu_v2.semantic_candidates import CandidateMenu, SemanticCandidate
from nlu_v2.semantic_repair import RepairUnit


@unittest.skipUnless(
    os.environ.get("KB_RUN_OLLAMA_TESTS") == "1",
    "set KB_RUN_OLLAMA_TESTS=1 to run local Ollama integration tests",
)
class OllamaIntegrationTests(unittest.TestCase):
    def test_unique_consecutive_event_does_not_spend_a_model_call(self):
        model = os.environ.get("KB_NLU_LLM_MODEL", "qwen2.5:7b")
        health = OllamaOpenAIProvider(model=model).health()
        self.assertTrue(health["available"], f"Ollama model not found: {model}")

        result = build_engine(True, model=model).analyze(
            "设备日志包含温度(`temperature`)。查询2026年1月1日至今，"
            "温度连续超过85℃且持续超过10分钟的所有时间段。"
        )
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(result.validation.status, "needs_clarification")
        self.assertEqual(len(result.understanding.events), 1)
        self.assertEqual(
            result.understanding.events[0].derived_metric.metric_id,
            "semantic.consecutive_duration",
        )
        self.assertEqual(
            result.understanding.events[0].threshold.normalized_seconds, 600.0
        )
        self.assertEqual(result.physical_plan.status, "unbound")

    def test_qwen_decision_matrix_is_bounded_and_semantically_correct(self):
        model = os.environ.get("KB_NLU_LLM_MODEL", "qwen2.5:7b")
        provider = OllamaOpenAIProvider(model=model)
        fixture = json.loads(
            (Path(__file__).parent / "tests" / "fixtures" /
             "m14_real_llm_decision_matrix.json").read_text("utf-8")
        )
        for row_index, row in enumerate(fixture["rows"]):
            name = row["name"]
            family = row["family"]
            with self.subTest(name=name):
                candidates = tuple(
                    SemanticCandidate(
                        "cand:sha256:" + "123456789abc"[row_index * 2 + index] * 64,
                        "candidate_select",
                        (row.get("candidate_payloads") or [{}] * len(row["candidate_summaries"]))[index],
                        summary,
                    )
                    for index, summary in enumerate(row["candidate_summaries"])
                )
                menu = CandidateMenu(
                    f"gap_{name}", f"req_{name}", row["source_clause"],
                    {
                        "family": family,
                        **row.get("target", {}),
                    }, candidates, "llm_choice", (),
                    "source", "target", "lineage", f"menu_{name}",
                )
                result = BoundedLLMExtractor(provider).choose_candidate(menu)
                self.assertTrue(
                    result.valid_json,
                    f"{result.error}; chars={result.response_chars}; "
                    f"stop={result.stop_reason}; raw={result.raw_response_excerpt}",
                )
                self.assertEqual(1, result.attempts)
                self.assertEqual(row["expected_decision"], result.payload["decision"])
                selected_index = row.get("expected_candidate_index")
                if selected_index is None:
                    self.assertNotIn("candidate_id", result.payload)
                else:
                    self.assertEqual(
                        candidates[selected_index].candidate_id,
                        result.payload["candidate_id"],
                    )
                self.assertGreater(result.prompt_chars, 0)
                self.assertGreater(result.response_chars, 0)
                self.assertTrue(result.raw_response_excerpt)
                self.assertTrue(result.parsed_json_excerpt)

    def test_qwen_select_commits_concrete_event_gain_end_to_end(self):
        model = os.environ.get("KB_NLU_LLM_MODEL", "qwen2.5:7b")
        engine = build_engine(True, model=model)
        query = (
            "设备传感器，查询2026年1月1日至今，温度连续超过85℃"
            "且持续超过10分钟的所有时间段。"
        )
        ir = copy.deepcopy(engine.analyze(query).understanding)
        requirement = next(
            item for item in ir.requirements
            if item.requirement_type == "sequence_event"
        )
        coverage = next(
            item for item in ir.coverage if item.requirement_id == requirement.requirement_id
        )
        event = next(item for item in ir.events if item.event_id == coverage.covered_by[0])
        event_ref = event.output_ref.ref_id
        ir.events.remove(event)
        ir.derived_projections = [
            item for item in ir.derived_projections if item.input_ref.ref_id != event_ref
        ]
        engine.coverage_matcher.apply(ir)
        catalog = engine.catalog_provider.snapshot()
        gap = next(
            item for item in engine.gap_analyzer.analyze(ir, catalog)
            if item.gap_type == "missing_event"
            and item.requirement_id == requirement.requirement_id
        )
        delegate = engine.semantic_candidate_compiler
        local_menu = delegate.compile(ir, gap, catalog)
        self.assertEqual("local_compile", local_menu.dispatch, local_menu.blocked_by)
        correct = local_menu.candidates[0]
        wrong_payload = copy.deepcopy(correct.semantic_payload)
        wrong_payload["conditions"][0]["operator"] = "lt"
        wrong = delegate._candidate(
            correct.operation_type, wrong_payload,
            "温度连续低于85℃且持续超过10分钟",
        )
        choice_menu = replace(
            local_menu,
            source_clause="温度连续超过85℃且持续超过10分钟",
            candidates=(replace(
                correct, evidence_summary="温度连续超过85℃且持续超过10分钟",
            ), wrong),
            dispatch="llm_choice",
            candidate_menu_digest="real-select-e2e-v1",
        )

        class FrozenChoiceCompiler:
            def compile(self, *_):
                return choice_menu

            def compile_patch(self, candidate, selected_gap, selected_ir):
                return delegate.compile_patch(candidate, selected_gap, selected_ir)

        engine.semantic_candidate_compiler = FrozenChoiceCompiler()
        result_ir, report, audit, calls = engine._execute_candidate_bundle_v3(
            ir, gap, RepairUnit.from_gaps([gap])[0], catalog,
            "real-select-security", [],
        )
        self.assertEqual(1, calls)
        self.assertIsNotNone(audit)
        self.assertEqual("select", audit.decision)
        self.assertEqual(correct.candidate_id, audit.candidate_id)
        self.assertTrue(
            report.committed,
            f"{report.transaction_error}; payload={correct.semantic_payload}; "
            f"effect={report.semantic_effect.to_dict() if report.semantic_effect else None}",
        )
        self.assertGreaterEqual(report.concrete_gain_count, 1)
        self.assertIn(requirement.requirement_id, report.semantic_effect.closed_requirement_ids)
        self.assertTrue(any(
            item.condition.operator == "gt" for item in result_ir.events
        ))


if __name__ == "__main__":
    unittest.main()
