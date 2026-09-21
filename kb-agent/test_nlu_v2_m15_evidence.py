"""Independent contract tests for the frozen M1.5 pre-production evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures"
REPORT = ROOT / "docs" / "reports" / "2026-08-31_m15_evidence_freeze.json"
RANKER = ROOT / "scripts" / "m15_deterministic_ranker.py"
SOURCE_BATCH = ROOT / "test_m14_acceptance_2026-08-31"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text("utf-8"))


class M15EvidenceFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text("utf-8"))

    def test_report_passes_and_every_artifact_blob_digest_matches(self):
        self.assertEqual("pass", self.report["gate_result"])
        self.assertTrue(self.report["checks"]["all_passed"])
        self.assertEqual(6, len(self.report["artifacts"]))
        for item in self.report["artifacts"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(item["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(item["valid"])

    def test_manifest_freezes_exact_source_artifacts_and_obligation_ids(self):
        payload = _load("m15_obligation_manifest.json")
        self.assertEqual(list(range(1, 51)), [item["case_id"] for item in payload["source_cases"]])
        for item in payload["source_cases"]:
            source = SOURCE_BATCH / item["artifact"]
            self.assertEqual(item["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        obligations = payload["obligations"]
        identifiers = [item["obligation_id"] for item in obligations]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(re.fullmatch(r"obl:sha256:[0-9a-f]{64}", item) for item in identifiers))
        self.assertTrue(all(item["source_spans"] for item in obligations))
        self.assertTrue(all(item["metric_membership"] or not item["in_scope"] for item in obligations))

    def test_case_031_keeps_supported_events_but_excludes_unsupported_formula(self):
        obligations = [
            item for item in _load("m15_obligation_manifest.json")["obligations"]
            if item["case_id"] == 31
        ]
        events = [item for item in obligations if item["operator_family"] == "sequence_event"]
        calculations = [item for item in obligations if item["operator_family"] == "calculation"]
        self.assertTrue(events)
        self.assertTrue(all(item["in_scope"] for item in events))
        self.assertTrue(calculations)
        self.assertTrue(all(not item["in_scope"] for item in calculations))

    def test_catalog_positive_rows_have_matched_capability_and_policy_negatives(self):
        groups = defaultdict(set)
        for item in _load("m15_catalog_contract_matrix.json")["contracts"]:
            self.assertTrue(item["read_only"])
            groups[item["matched_group"]].add(item["variant"])
        self.assertGreaterEqual(len(groups), 3)
        expected = {"positive", "missing_capability", "missing_policy"}
        self.assertTrue(all(expected.issubset(values) for values in groups.values()))

    def test_temporal_operator_outputs_are_incompatible_by_contract(self):
        payload = _load("m15_operator_output_oracle.json")
        types = {item["operator"]: item["output"] for item in payload["operator_type_matrix"]}
        self.assertEqual("relation<interval>", types["consecutive"])
        self.assertEqual("relation<group_key,duration>", types["cumulative_duration"])
        self.assertEqual("relation<group_key,quantity>", types["cumulative_value"])
        self.assertTrue({24, 27, 47}.issubset({item["case_id"] for item in payload["case_dags"]}))

    def test_choice_oracle_has_frozen_family_and_outcome_denominators(self):
        rows = _load("m15_llm_choice_oracle.json")["rows"]
        counts = Counter(item["family"] for item in rows)
        self.assertEqual(40, len(rows))
        self.assertEqual({
            "event": 8, "set": 8, "aggregate": 8, "calculation": 8, "reference": 8,
        }, dict(counts))
        for family in counts:
            outcomes = {item["expected_decision"] for item in rows if item["family"] == family}
            self.assertTrue({"select", "none_of_above", "ambiguous"}.issubset(outcomes))
        for row in rows:
            self.assertGreaterEqual(len(row["candidates"]), 2)
            self.assertLessEqual(len(row["candidates"]), 5)
            for candidate in row["candidates"]:
                canonical = json.dumps(
                    candidate["semantic_payload"], ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual(candidate["payload_digest"], hashlib.sha256(canonical).hexdigest())

    def test_deterministic_baseline_binds_ranker_and_choice_oracle_blobs(self):
        payload = _load("m15_deterministic_baseline.json")
        choice_path = FIXTURES / "m15_llm_choice_oracle.json"
        self.assertEqual(hashlib.sha256(RANKER.read_bytes()).hexdigest(), payload["ranker_source_blob_sha256"])
        self.assertEqual(hashlib.sha256(choice_path.read_bytes()).hexdigest(), payload["choice_oracle_sha256"])
        oracle_ids = {item["oracle_id"] for item in _load("m15_llm_choice_oracle.json")["rows"]}
        baseline_ids = {item["oracle_id"] for item in payload["rows"]}
        self.assertEqual(oracle_ids, baseline_ids)
        self.assertTrue(all(
            "deterministic_abstain" in item and "deterministic_ranker" in item
            for item in payload["rows"]
        ))


if __name__ == "__main__":
    unittest.main()
