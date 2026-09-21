import unittest

from agent.agent_models import IntentType
from agent.intent_router import IntentRecognizer


SKILLS = [
    {
        "name": "search_private_kb",
        "description": "search the private knowledge base",
        "kind": "local",
        "tags": ["retrieval"],
    }
]


class FakeRouterClient:
    def __init__(self, payload):
        self.payload = payload
        self.settings = type("Settings", (), {"router_strict": True})()

    def invoke_router(self, *, messages, tool_schema):
        return self.payload


class IntentRouterTests(unittest.TestCase):
    def test_blank_input_returns_clarify(self):
        router = IntentRecognizer(FakeRouterClient({}))
        result = router.recognize("   ", SKILLS)
        self.assertEqual(result.intent, IntentType.CLARIFY)
        self.assertFalse(result.needs_retrieval)

    def test_meta_question_routes_to_chat_help(self):
        router = IntentRecognizer(FakeRouterClient({}))
        result = router.recognize("你能做什么？", SKILLS)
        self.assertEqual(result.intent, IntentType.CHAT_HELP)
        self.assertFalse(result.needs_retrieval)

    def test_search_keyword_routes_to_kb(self):
        router = IntentRecognizer(FakeRouterClient({}))
        result = router.recognize("帮我查一下 ESN 的误差补偿方法", SKILLS)
        self.assertEqual(result.intent, IntentType.KB_SEARCH)
        self.assertTrue(result.needs_retrieval)
        self.assertEqual(result.skill_calls[0].skill_name, "search_private_kb")

    def test_llm_payload_filters_unknown_skills(self):
        payload = {
            "intent": "kb_search",
            "needs_retrieval": True,
            "confidence": 0.93,
            "user_goal": "查找 ESN 资料",
            "rewritten_query": "ESN 误差补偿",
            "top_k": 3,
            "response_style": "grounded",
            "clarification_question": "",
            "reason": "llm",
            "metadata_filters": {},
            "skill_calls": [
                {
                    "skill_name": "unknown_skill",
                    "purpose": "ignore",
                    "arguments": {},
                },
                {
                    "skill_name": "search_private_kb",
                    "purpose": "retrieve evidence",
                    "arguments": {"query": "ESN 误差补偿", "top_k": 3},
                },
            ],
        }
        router = IntentRecognizer(FakeRouterClient(payload))
        result = router.recognize("帮我查一下 ESN", SKILLS)
        self.assertEqual(len(result.skill_calls), 1)
        self.assertEqual(result.skill_calls[0].skill_name, "search_private_kb")


if __name__ == "__main__":
    unittest.main()
