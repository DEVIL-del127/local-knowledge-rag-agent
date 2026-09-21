import unittest

from scripts.evaluate_m15_llm_choice import _majority


class M15EvaluationTests(unittest.TestCase):
    def test_majority_uses_two_matching_protocol_votes(self):
        attempts = [
            {"decision": "select", "selected_payload_digest": "a", "valid_protocol": True},
            {"decision": "select", "selected_payload_digest": "a", "valid_protocol": True},
            {"decision": "ambiguous", "selected_payload_digest": "", "valid_protocol": True},
        ]
        self.assertEqual(("select", "a"), _majority(attempts))

    def test_provider_failure_has_no_vote(self):
        attempts = [
            {"decision": "provider_failure", "selected_payload_digest": "", "valid_protocol": False},
            {"decision": "select", "selected_payload_digest": "a", "valid_protocol": True},
            {"decision": "ambiguous", "selected_payload_digest": "", "valid_protocol": True},
        ]
        self.assertEqual(("no_majority", ""), _majority(attempts))


if __name__ == "__main__":
    unittest.main()
