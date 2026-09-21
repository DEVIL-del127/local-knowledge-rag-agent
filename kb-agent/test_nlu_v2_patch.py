from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from nlu_v2 import EngineConfig, QueryUnderstandingEngine, SecurityContext
from nlu_v2.gaps import GapAnalyzer
from nlu_v2.llm_extractor import (
    BoundedLLMExtractor,
    OllamaOpenAIProvider,
    ProviderResponse,
)
from nlu_v2.patch_protocol import (
    PATCH_OPERATION_TYPES,
    AddCalculationOperation,
    AddAmbiguityOperation,
    GapRequest,
    SemanticPatchV1,
    apply_semantic_patch,
    atomic_response_schema,
    parse_atomic_patch_text,
    request_ir_digest,
)


QUERY = "设备传感器，查询2026年1月1日至今温度一直高于85℃并持续10分钟以上的区间"
NO_WINDOW_QUERY = "设备传感器，查询温度一直高于85℃并持续10分钟以上的区间"
SET_QUERY = (
    "监测设备的温度传感器。找出 2026年1月1日至今，温度连续超过 85℃ "
    "超过 10分钟 的所有时间段，以及 累计超过 80℃ 但未超过 85℃ 的总时长"
    "超过 6小时 的所有日期。输出这两类异常时间段的交集日期。"
)


class DynamicPatchProvider:
    def __init__(self, *, forged: bool = False, wrong_offsets: bool = False,
                 second_operation: dict | None = None):
        self.calls = 0
        self.forged = forged
        self.wrong_offsets = wrong_offsets
        self.second_operation = second_operation
        self.schemas = []

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        self.schemas.append(response_schema)
        digest = re.search(r"BASE_IR_DIGEST=([0-9a-f]+)", prompt).group(1)
        gaps = json.loads(re.search(r"GAP_REQUESTS=(\[.*\])\nRELEVANT_CATALOG=", prompt).group(1))
        event_gap = next(item for item in gaps if item["gap_type"] == "missing_event")
        evidence = "伪造" if self.forged else "温度一直高于85℃并持续10分钟以上"
        operation_evidence = {"text": evidence}
        condition_evidence = {"text": "温度一直高于85℃"}
        duration_evidence = {"text": "10分钟"}
        event_evidence = {"text": "温度一直高于85℃并持续10分钟以上"}
        if self.wrong_offsets:
            for item in (operation_evidence, condition_evidence,
                         duration_evidence, event_evidence):
                item.update({"start": 0, "end": 1})
        operation = {
            "operation_type": "add_event",
            "operation_id": "op_event_1",
            "gap_id": event_gap["gap_id"],
            "preconditions": [{"kind": "gap_open", "value": event_gap["gap_id"]}],
            "dependencies": [],
            "anchor_ids": event_gap["allowed_anchor_ids"][:1],
            "evidence": [operation_evidence],
            "event": {
                "event_id": "high_temperature",
                "metric_id": "device_telemetry.consecutive_duration",
                "conditions": [{
                    "field_id": "device_telemetry.temperature",
                    "operator": "gt",
                    "value": 85,
                    "unit": "celsius",
                    "span": condition_evidence
                }],
                "duration": {
                    "operator": "gte", "value": 10, "unit": "minute",
                    "span": duration_evidence
                },
                "span": event_evidence
            }
        }
        operations = [operation]
        if self.second_operation:
            extra = dict(self.second_operation)
            extra.setdefault("evidence", [{"text": "温度一直高于85℃"}])
            operations.append(extra)
        return json.dumps({
            "schema_version": "semantic_patch_v1",
            "base_ir_digest": digest,
            "operations": operations,
        }, ensure_ascii=False)


class EmptyPatchProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        digest = re.search(r"BASE_IR_DIGEST=([0-9a-f]+)", prompt).group(1)
        return json.dumps({
            "schema_version": "semantic_patch_v1",
            "base_ir_digest": digest,
            "operations": [],
        })


class LegacyEventProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        return json.dumps({"events": [{
            "event_id": "legacy_temperature",
            "metric_id": "device_telemetry.consecutive_duration",
            "conditions": [{
                "field_id": "device_telemetry.temperature", "operator": "gt",
                "value": 85, "unit": "celsius", "span": {"text": "温度一直高于85℃"},
            }],
            "duration": {
                "operator": "gte", "value": 10, "unit": "minute",
                "span": {"text": "10分钟"},
            },
            "span": {"text": "温度一直高于85℃并持续10分钟以上"},
        }]}, ensure_ascii=False)


class SemanticChoiceEventProvider:
    supports_semantic_choice = True

    def __init__(self):
        self.calls = 0
        self.prompt = ""
        self.schema = None

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None):
        self.calls += 1
        self.prompt = prompt
        self.schema = response_schema
        return json.dumps({
            "evidence_text": "温度一直高于85℃并持续10分钟以上",
            "logic": "and",
            "conditions": [{
                "field_ref": "device_telemetry.temperature",
                "operator": "gt", "value": 85, "unit": "celsius",
            }],
            "duration_operator": "gte",
            "duration_value": 10,
            "duration_unit": "minute",
            "group_by": ["local_date"],
        }, ensure_ascii=False)


class SemanticPatchProtocolTests(unittest.TestCase):
    def _base(self, query=QUERY):
        engine = QueryUnderstandingEngine()
        catalog = engine.catalog_provider.snapshot()
        ir = engine.rule_extractor.extract(query, catalog).ir
        ir.query.security_scope_digest = "test-scope"
        gaps = GapAnalyzer().analyze(ir, catalog)
        return engine, catalog, ir, gaps

    def test_gold_operation_union_matches_schema_and_applier_registry(self):
        fixture = json.loads(
            (Path(__file__).parent / "data" / "nlu_v2_protocol_gold_v1.json").read_text(
                encoding="utf-8"
            )
        )
        schema = SemanticPatchV1.model_json_schema()
        schema_operations = {
            value["const"]
            for definition in schema.get("$defs", {}).values()
            for value in definition.get("properties", {}).values()
            if isinstance(value, dict) and "const" in value
            and value["const"] in PATCH_OPERATION_TYPES
        }
        self.assertEqual(set(fixture["operation_types"]), PATCH_OPERATION_TYPES)
        self.assertEqual(schema_operations, PATCH_OPERATION_TYPES)

    def test_strict_schema_rejects_unknown_and_malformed_operations(self):
        with self.assertRaises(ValidationError):
            SemanticPatchV1.model_validate({
                "schema_version": "semantic_patch_v1",
                "base_ir_digest": "abc",
                "operations": [{
                    "operation_type": "delete_node", "operation_id": "x",
                    "gap_id": "g", "evidence": [{"text": "x"}],
                }],
            })
        with self.assertRaises(ValidationError):
            AddAmbiguityOperation.model_validate({
                "operation_type": "add_ambiguity", "operation_id": "x", "gap_id": "g",
                "evidence": [], "ambiguity_id": "a", "kind": "field", "message": "m",
                "unexpected": True,
            })

    def test_atomic_sequence_schema_and_evidence_gated_application(self):
        engine, catalog, ir, _ = self._base()
        catalog = engine.catalog_authorizer.authorize(
            catalog, SecurityContext("test", "sequence-test", allowed_sources=("device_telemetry",)),
        )
        source = catalog.source("device_telemetry")
        self.assertIsNotNone(source)
        gap = GapRequest(
            gap_id="gap_sequence", gap_type="missing_sequence", query_slice=QUERY,
            allowed_catalog_symbols=[field.field_id for field in source.fields],
            allowed_operations=["add_sequence"],
        )
        schema = atomic_response_schema(gap)
        self.assertIn("steps", schema["properties"])
        payload = json.dumps({
            "evidence_text": "温度一直高于85℃并持续10分钟以上",
            "sequence_id": "temperature_sequence",
            "steps": [
                {"field_id": "device_telemetry.temperature", "operator": "gt", "value": 85,
                 "unit": "celsius", "span": {"text": "温度一直高于85℃"}},
                {"field_id": "device_telemetry.temperature", "operator": "gt", "value": 85,
                 "unit": "celsius", "span": {"text": "温度一直高于85℃"}},
            ],
            "partition_by": list(source.metadata["partition_fields"]),
            "order_by": source.metadata["timestamp_field"],
            "max_duration": {"operator": "lte", "value": 10, "unit": "minute",
                             "span": {"text": "10分钟"}},
        }, ensure_ascii=False)
        patch_value, protocol, error = parse_atomic_patch_text(
            payload, base_ir_digest=request_ir_digest(ir), gap=gap,
        )
        self.assertEqual(protocol, "semantic_patch_v2")
        self.assertFalse(error)
        candidate, report = apply_semantic_patch(ir, patch_value, [gap], catalog)
        self.assertEqual(report.accepted_count, 1, report.rejected)
        self.assertEqual(report.closed_gap_ids, [gap.gap_id])
        self.assertEqual(candidate.sequences[0].max_duration.normalized_seconds, 600)

    def test_atomic_set_patch_binds_typed_output_refs(self):
        engine = QueryUnderstandingEngine()
        baseline = engine.analyze(SET_QUERY).understanding
        catalog = engine.catalog_provider.snapshot()
        self.assertGreaterEqual(len(baseline.derived_projections), 2)
        baseline.set_operations = []
        inputs = [item.output.ref_id for item in baseline.derived_projections[:2]]
        gap = GapRequest(
            gap_id="gap_set", gap_type="missing_set_inputs", query_slice=SET_QUERY,
            available_output_refs=inputs, allowed_operations=["add_set_operation"],
        )
        patch = SemanticPatchV1.model_validate({
            "base_ir_digest": request_ir_digest(baseline),
            "operations": [{
                "operation_type": "add_set_operation", "operation_id": "set_dates",
                "gap_id": gap.gap_id,
                "preconditions": [{"kind": "gap_open", "value": gap.gap_id}],
                "evidence": [{"text": "交集日期"}],
                "operation": "intersection", "inputs": inputs, "granularity": "date",
            }],
        })
        candidate, report = apply_semantic_patch(baseline, patch, [gap], catalog)
        self.assertEqual(report.accepted_count, 1, report.rejected)
        self.assertEqual(report.closed_gap_ids, [gap.gap_id])
        self.assertEqual(candidate.set_operations[0].inputs, inputs)
        self.assertEqual(candidate.set_operations[0].output_ref.ref_id, "set_set_dates.dates")

    def test_atomic_event_schema_accepts_typed_dynamic_threshold_expression(self):
        gap = GapRequest(
            gap_id="gap_dynamic", gap_type="missing_event", query_slice=QUERY,
            allowed_catalog_symbols=["device_telemetry.temperature"],
            allowed_metrics=["device_telemetry.consecutive_duration"],
            allowed_operations=["add_event"],
        )
        payload = json.dumps({
            "evidence_text": "温度一直高于85℃并持续10分钟以上",
            "event": {
                "event_id": "dynamic_temperature", "metric_id": "device_telemetry.consecutive_duration",
                "conditions": [{
                    "field_id": "device_telemetry.temperature", "operator": "gt",
                    "right_expression": {
                        "kind": "binary", "operator": "mul", "result_type": "number", "unit": "celsius",
                        "arguments": [
                            {"kind": "catalog_symbol", "identifier": "device_telemetry.temperature",
                             "result_type": "number", "unit": "celsius"},
                            {"kind": "literal", "value": 0.9, "value_type": "number",
                             "result_type": "number"},
                        ],
                    },
                    "span": {"text": "温度一直高于85℃"},
                }],
                "duration": {"operator": "gte", "value": 10, "unit": "minute",
                             "span": {"text": "10分钟"}},
                "span": {"text": "温度一直高于85℃并持续10分钟以上"},
            },
        }, ensure_ascii=False)
        patch_value, protocol, error = parse_atomic_patch_text(
            payload, base_ir_digest="a" * 64, gap=gap,
        )
        self.assertEqual(protocol, "semantic_patch_v2")
        self.assertFalse(error)
        self.assertEqual(patch_value.operations[0].event.conditions[0].right_expression.operator, "mul")

    def test_atomic_prompt_exposes_a_copyable_canonical_evidence_anchor(self):
        gap = GapRequest(
            gap_id="gap_anchor", gap_type="missing_aggregate_scope",
            query_slice="计算温度的平均值", allowed_operations=["add_aggregate"],
        )
        prompt = BoundedLLMExtractor._atomic_prompt(
            gap.query_slice, [], {}, gap, "a" * 64,
        )
        self.assertIn("CANONICAL_EVIDENCE_TEXT=\"计算温度的平均值\"", prompt)

    def test_semantic_choice_schema_injects_local_allow_lists(self):
        gap = GapRequest(
            gap_id="gap_choice", gap_type="missing_event", query_slice=QUERY,
            allowed_catalog_symbols=["device_telemetry.temperature"],
            allowed_metrics=["device_telemetry.consecutive_duration"],
            allowed_operations=["add_event"], response_mode="semantic_choice",
            choice_domain={
                "field_refs": ["device_telemetry.temperature"],
                "metric_refs": ["device_telemetry.consecutive_duration"],
                "output_refs": [], "unit_refs": ["celsius", "minute"],
            },
        )
        schema = atomic_response_schema(gap)
        condition = schema["$defs"]["AtomicConditionChoice"]
        self.assertEqual(
            condition["properties"]["field_ref"]["enum"],
            ["device_telemetry.temperature"],
        )
        self.assertNotIn("span", condition["properties"])

    def test_semantic_choice_event_commits_with_short_local_prompt(self):
        provider = SemanticChoiceEventProvider()
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(QUERY)
        self.assertTrue(result.patch_report.committed, result.patch_report.transaction_error)
        self.assertEqual(len(result.understanding.events), 1)
        self.assertEqual(result.understanding.events[0].condition.value.value, 85)
        self.assertLess(result.llm_audit.prompt_chars, 6000)
        self.assertNotIn("BASE_IR_DIGEST", provider.prompt)
        self.assertNotIn("EXISTING_IR", provider.prompt)
        self.assertIn("ALLOWED_CHOICES", provider.prompt)
        self.assertTrue(result.llm_audit.coverage_before)
        self.assertTrue(result.llm_audit.coverage_after)

    def test_dynamic_event_patch_commits_after_local_type_validation(self):
        engine, catalog, ir, _ = self._base()
        gap = GapRequest(
            gap_id="gap_dynamic_transaction", gap_type="missing_event", query_slice=QUERY,
            allowed_catalog_symbols=["device_telemetry.temperature"],
            allowed_metrics=["device_telemetry.consecutive_duration"],
            allowed_operations=["add_event"],
        )
        patch = SemanticPatchV1.model_validate({
            "base_ir_digest": request_ir_digest(ir),
            "operations": [{
                "operation_type": "add_event", "operation_id": "dynamic_event",
                "gap_id": gap.gap_id,
                "preconditions": [{"kind": "gap_open", "value": gap.gap_id}],
                "evidence": [{"text": "温度一直高于85℃并持续10分钟以上"}],
                "event": {
                    "event_id": "dynamic_temperature",
                    "metric_id": "device_telemetry.consecutive_duration",
                    "conditions": [{
                        "field_id": "device_telemetry.temperature", "operator": "gt",
                        "right_expression": {
                            "kind": "binary", "operator": "mul", "result_type": "number",
                            "unit": "celsius", "arguments": [
                                {"kind": "catalog_symbol", "identifier": "device_telemetry.temperature",
                                 "result_type": "number", "unit": "celsius"},
                                {"kind": "literal", "value": 0.9, "value_type": "number",
                                 "result_type": "number"},
                            ],
                        },
                        "span": {"text": "温度一直高于85℃"},
                    }],
                    "duration": {"operator": "gte", "value": 10, "unit": "minute",
                                 "span": {"text": "10分钟"}},
                    "span": {"text": "温度一直高于85℃并持续10分钟以上"},
                },
            }],
        })
        candidate, report = engine._apply_patch_transaction(ir, patch, [gap], catalog)
        self.assertTrue(report.committed, report.transaction_error)
        self.assertEqual(len(candidate.events), 1)
        self.assertEqual(candidate.events[0].condition.right_expression.operator, "mul")

    def test_atomic_calculation_window_expression_is_transactionally_validated(self):
        engine, catalog, ir, _ = self._base()
        gap = GapRequest(
            gap_id="gap_calculation", gap_type="missing_formula_input", query_slice=QUERY,
            allowed_catalog_symbols=["device_telemetry.temperature", "device_telemetry.device_id"],
            allowed_operations=["add_calculation"],
        )

        def operation(operation_id, evidence_text, field_id, result_type, unit=None):
            return AddCalculationOperation(
                operation_type="add_calculation", operation_id=operation_id, gap_id=gap.gap_id,
                preconditions=[{"kind": "gap_open", "value": gap.gap_id}],
                evidence=[{"text": evidence_text}], calculation_type="growth",
                expression={
                    "kind": "function", "operator": "moving_avg",
                    "result_type": result_type, "unit": unit,
                    "arguments": [
                        {"kind": "catalog_symbol", "identifier": field_id,
                         "result_type": result_type, "unit": unit},
                        {"kind": "literal", "value": 600, "value_type": "duration",
                         "result_type": "duration", "unit": "second"},
                    ],
                },
            )

        valid = SemanticPatchV1(
            base_ir_digest=request_ir_digest(ir),
            operations=[operation("valid_calculation", "温度一直高于85℃", "device_telemetry.temperature",
                                  "number", "celsius")],
        )
        candidate, report = engine._apply_patch_transaction(ir, valid, [gap], catalog)
        self.assertTrue(report.committed, report.transaction_error)
        self.assertEqual(candidate.calculations[0].parameters["expression_ast"]["operator"], "moving_avg")

        invalid = SemanticPatchV1(
            base_ir_digest=request_ir_digest(ir),
            operations=[operation("invalid_calculation", "温度一直高于85℃", "device_telemetry.device_id",
                                  "string")],
        )
        candidate, report = engine._apply_patch_transaction(ir, invalid, [gap], catalog)
        self.assertFalse(report.committed)
        self.assertIn("window_function_shape_invalid", report.transaction_error)
        self.assertEqual(candidate.to_dict(), ir.to_dict())

        forged = SemanticPatchV1(
            base_ir_digest=request_ir_digest(ir),
            operations=[operation("forged_calculation", "伪造证据", "device_telemetry.temperature",
                                  "number", "celsius")],
        )
        candidate, report = apply_semantic_patch(ir, forged, [gap], catalog)
        self.assertEqual(candidate.to_dict(), ir.to_dict())
        self.assertEqual(report.rejected[0].reason, "invalid_operation_evidence")

    def test_base_digest_mismatch_rejects_entire_patch(self):
        _, catalog, ir, gaps = self._base()
        patch_value = SemanticPatchV1(base_ir_digest="stale", operations=[])
        candidate, report = apply_semantic_patch(ir, patch_value, gaps, catalog)
        self.assertEqual(candidate.to_dict(), ir.to_dict())
        self.assertEqual(report.transaction_error, "base_ir_digest_mismatch")

    def test_unknown_gap_dependency_and_precondition_are_rejected_independently(self):
        _, catalog, ir, gaps = self._base()
        gap = gaps[0]
        common = {
            "operation_type": "add_ambiguity", "kind": "field", "message": "需要澄清",
            "candidates": [], "evidence": [{"text": "温度一直高于85℃"}],
        }
        operations = [
            AddAmbiguityOperation(
                **common, operation_id="unknown_gap", gap_id="missing",
                ambiguity_id="a1",
            ),
            AddAmbiguityOperation(
                **common, operation_id="unknown_dep", gap_id=gap.gap_id,
                ambiguity_id="a2", dependencies=["missing_node"],
            ),
            AddAmbiguityOperation(
                **common, operation_id="bad_precondition", gap_id=gap.gap_id,
                ambiguity_id="a3",
                preconditions=[{"kind": "node_present", "value": "missing_node"}],
            ),
        ]
        # Allow ambiguity in this isolated protocol test so dependency checks are reached.
        widened = [gap.model_copy(update={"allowed_operations": ["add_ambiguity"]})]
        patch_value = SemanticPatchV1(
            base_ir_digest=request_ir_digest(ir), operations=operations,
        )
        _, report = apply_semantic_patch(ir, patch_value, widened, catalog)
        self.assertEqual(
            [item.reason for item in report.rejected],
            ["unknown_gap", "dependency_unresolved", "precondition_failed"],
        )

    def test_forged_evidence_is_rejected(self):
        provider = DynamicPatchProvider(forged=True)
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(QUERY)
        self.assertFalse(result.patch_report.committed)
        self.assertEqual(result.patch_report.rejected[0].reason, "invalid_operation_evidence")
        self.assertEqual(result.patch_report.rejection_gate, "patch_applier_gate")
        self.assertEqual(result.llm_audit.rejected_gate, "patch_applier_gate")
        self.assertTrue(result.llm_audit.parsed_json_excerpt)
        self.assertGreaterEqual(result.llm_audit.duration_ms, 0.0)
        self.assertEqual(result.understanding.events, [])

    def test_unique_evidence_text_is_canonicalized_when_model_offsets_are_wrong(self):
        provider = DynamicPatchProvider(wrong_offsets=True)
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(QUERY)
        self.assertTrue(result.patch_report.committed)
        event_span = result.understanding.events[0].provenance[0].span
        self.assertEqual(
            result.understanding.query.normalized[event_span.start:event_span.end],
            event_span.text,
        )

    def test_partial_operation_rejection_can_commit_valid_gap_closure(self):
        provider = DynamicPatchProvider(second_operation={
            "operation_type": "add_ambiguity", "operation_id": "op_unknown",
            "gap_id": "unknown_gap", "ambiguity_id": "amb", "kind": "field",
            "message": "unknown",
        })
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(QUERY)
        self.assertFalse(result.patch_report.committed)
        self.assertEqual(result.patch_report.transaction_error, "repair_contract_operator_out_of_scope")

    def test_new_fatal_validation_error_rolls_back_atomically(self):
        provider = DynamicPatchProvider()
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        ).analyze(NO_WINDOW_QUERY)
        self.assertFalse(result.patch_report.committed)
        self.assertIn("event_window_unbound", result.patch_report.transaction_error)
        self.assertEqual(result.understanding.events, [])

    def test_empty_patch_is_neutral_and_not_cached(self):
        provider = EmptyPatchProvider()
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        engine.analyze(QUERY)
        engine.analyze(QUERY)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(engine._cache, {})

    def test_committed_patch_is_cached(self):
        provider = DynamicPatchProvider()
        engine = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(provider),
        )
        first = engine.analyze(QUERY)
        second = engine.analyze(QUERY)
        self.assertTrue(first.patch_report.committed)
        self.assertTrue(second.patch_report.committed)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(second.model_calls, 0)

    def test_legacy_fragment_fixture_uses_adapter_and_commits(self):
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(LegacyEventProvider()),
        ).analyze(QUERY)
        self.assertTrue(result.patch_report.committed)
        self.assertEqual(result.patch_report.transaction_error, "")
        self.assertEqual(result.patch_report.protocol, "legacy_fragment_adapter")
        self.assertEqual(len(result.understanding.events), 1)
        self.assertEqual(result.validation.status, "valid")

    def test_patch_feature_off_preserves_legacy_merge_path(self):
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True, enable_patch_v1=False),
            llm_extractor=BoundedLLMExtractor(LegacyEventProvider()),
        ).analyze(QUERY)
        self.assertIsNone(result.patch_report)
        self.assertEqual(result.validation.status, "valid")
        self.assertEqual(len(result.understanding.events), 1)

    def test_ollama_receives_json_schema_object(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": "{}"}, "done": True, "done_reason": "stop",
        }
        provider = OllamaOpenAIProvider(base_url="http://127.0.0.1:11434")
        with patch("requests.post", return_value=response) as post:
            result = provider.complete(
                "prompt", deadline=10**12, attempt_budget=1,
                response_schema={"type": "object", "additionalProperties": False},
            )
        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(post.call_args.kwargs["json"]["format"]["type"], "object")

    def test_schema_failure_audit_names_parse_gate_and_retains_parsed_json(self):
        result = QueryUnderstandingEngine(
            config=EngineConfig(enable_llm=True),
            llm_extractor=BoundedLLMExtractor(
                type("InvalidProvider", (), {
                    "complete": lambda self, prompt, **kwargs: (
                        '{"unexpected": true, "token": "audit-secret"}'
                    )
                })()
            ),
        ).analyze(QUERY)
        self.assertEqual(result.patch_report.rejection_gate, "schema_parse_gate")
        self.assertEqual(result.llm_audit.rejected_gate, "schema_parse_gate")
        self.assertIn('"unexpected": true', result.llm_audit.parsed_json_excerpt)
        self.assertIn('"token": "[REDACTED]"', result.llm_audit.parsed_json_excerpt)
        self.assertNotIn("audit-secret", result.llm_audit.raw_response_excerpt)
        self.assertTrue(result.llm_audit.parse_error)


if __name__ == "__main__":
    unittest.main()
