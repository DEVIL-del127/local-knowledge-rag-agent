import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_models import IntentType
import tempfile
from agent.agent_service import AgentSettings, PrivateKnowledgeAgent


class FakeSearchBackend:
    def __init__(self):
        self.last_query = ""

    def hybrid_search(self, query: str, top_n: int = 10, include_chunks: bool = False):
        self.last_query = query
        if include_chunks:
            return [
                {
                    "filename": "esn-paper.pdf",
                    "rrf": 1.0,
                    "highlights": ["ESN 可用于时间序列建模与误差补偿。"],
                    "content": "ESN 可用于时间序列建模与误差补偿。",
                }
            ], []
        return [
            {
                "filename": "esn-paper.pdf",
                "rrf": 1.0,
                "highlights": ["ESN 可用于时间序列建模与误差补偿。"],
                "content": "ESN 可用于时间序列建模与误差补偿。",
            }
        ]

    def vec_search(self, query: str, top_k: int = 10):
        return [
            {
                "filename": "esn-paper.pdf",
                "page": 3,
                "chunk": 0,
                "text": "ESN 是一种递归神经网络，常用于时间序列建模。",
                "score": 0.94,
            }
        ]


class EmptySearchBackend:
    def hybrid_search(self, query: str, top_n: int = 10, include_chunks: bool = False):
        if include_chunks:
            return [], []
        return []

    def vec_search(self, query: str, top_k: int = 10):
        return []


class FakeDeepSeekClient:
    def __init__(self):
        self.settings = type("Settings", (), {"router_strict": True})()

    def invoke_router(self, *, messages, tool_schema, user_id: str = "default"):
        # 用当前用户消息作为查询(模拟真实路由的 rewritten_query)
        last_user = [m for m in messages if m.get("role") == "user"][-1]["content"]
        return {
            "intent": "kb_search",
            "needs_retrieval": True,
            "confidence": 0.88,
            "user_goal": last_user,
            "rewritten_query": last_user,
            "top_k": 3,
            "response_style": "grounded",
            "clarification_question": "",
            "reason": "fake_router",
            "metadata_filters": {},
            "has_anaphora": False,
            "skill_calls": [
                {
                    "skill_name": "search_private_kb",
                    "purpose": "retrieve evidence",
                    "arguments": {"query": last_user, "top_k": 3},
                }
            ],
        }

    def invoke_text(self, *, system_prompt, user_prompt, history=None, temperature=None, max_tokens=None, user_id: str = "default"):
        return "ESN 是一种常用于时间序列建模的递归神经网络。[E1]"


class CountingDeepSeekClient:
    def __init__(self):
        self.settings = type("Settings", (), {"router_strict": True})()
        self.router_calls = 0
        self.answer_calls = 0

    def invoke_router(self, *, messages, tool_schema, user_id: str = "default"):
        self.router_calls += 1
        return {
            "intent": "kb_search",
            "needs_retrieval": True,
            "confidence": 0.88,
            "user_goal": "look for evidence",
            "rewritten_query": "look for evidence",
            "top_k": 3,
            "response_style": "grounded",
            "clarification_question": "",
            "reason": "fake_router",
            "metadata_filters": {},
            "has_anaphora": False,
            "skill_calls": [
                {
                    "skill_name": "search_private_kb",
                    "purpose": "retrieve evidence",
                    "arguments": {"query": "look for evidence", "top_k": 3},
                }
            ],
        }

    def invoke_text(self, *, system_prompt, user_prompt, history=None, temperature=None, max_tokens=None, user_id: str = "default"):
        self.answer_calls += 1
        return "This should not be called when no evidence is found."


class PrivateKnowledgeAgentTests(unittest.TestCase):
    def test_agent_runs_retrieval_and_returns_evidence(self):
        agent = PrivateKnowledgeAgent(
            search_backend=FakeSearchBackend(),
            deepseek_client=FakeDeepSeekClient(),
            settings=AgentSettings(state_dir=tempfile.mkdtemp()),
        )
        reply = agent.chat("ESN 是什么？")
        self.assertEqual(reply.intent.intent, IntentType.KB_SEARCH)
        self.assertTrue(reply.evidence)
        self.assertIn("ESN", reply.answer)
        self.assertEqual(reply.evidence[0].source, "esn-paper.pdf")

    def test_agent_handles_clarify_path(self):
        agent = PrivateKnowledgeAgent(
            search_backend=FakeSearchBackend(),
            deepseek_client=FakeDeepSeekClient(),
            settings=AgentSettings(state_dir=tempfile.mkdtemp()),
        )
        reply = agent.chat("啊？")
        self.assertEqual(reply.intent.intent, IntentType.CLARIFY)
        self.assertIn("补充", reply.answer)

    def test_repeated_no_result_query_does_not_reenter_llm(self):
        client = CountingDeepSeekClient()
        agent = PrivateKnowledgeAgent(
            search_backend=EmptySearchBackend(),
            deepseek_client=client,
            settings=AgentSettings(state_dir=tempfile.mkdtemp()),
        )

        first_reply = agent.chat("这个方案和前文的联系在哪里？")
        second_reply = agent.chat("这个方案和前文的联系在哪里？")

        self.assertEqual(client.router_calls, 1)
        self.assertEqual(client.answer_calls, 0)
        self.assertIn("证据", first_reply.answer)
        self.assertEqual(second_reply.intent.intent, IntentType.CLARIFY)

    def test_repeated_success_query_hits_session_cache(self):
        client = CountingDeepSeekClient()
        agent = PrivateKnowledgeAgent(
            search_backend=FakeSearchBackend(),
            deepseek_client=client,
            settings=AgentSettings(state_dir=tempfile.mkdtemp()),
        )

        first_reply = agent.chat("这个方法的核心思路是什么？")
        second_reply = agent.chat("这个方法的核心思路是什么？")

        self.assertEqual(client.router_calls, 0)
        self.assertEqual(client.answer_calls, 1)
        self.assertEqual(first_reply.answer, second_reply.answer)
        self.assertEqual(second_reply.intent.intent, IntentType.KB_SEARCH)


    def test_followup_query_rewritten_with_context(self):
        """多轮指代: 第二轮省略主语, 查询应被上文实体补全"""
        backend = FakeSearchBackend()
        agent = PrivateKnowledgeAgent(
            search_backend=backend,
            deepseek_client=FakeDeepSeekClient(),
            settings=AgentSettings(state_dir=tempfile.mkdtemp()),
        )
        history: list[dict] = []
        agent.chat("刘月的论文写了什么", history=history)
        history = [
            {"role": "user", "content": "刘月的论文写了什么"},
            {"role": "assistant", "content": "刘月的论文是研究误差补偿的"},
        ]
        agent.chat("用了哪些技术和模型", history=history)
        self.assertIn("刘月的论文", backend.last_query)
        self.assertIn("技术", backend.last_query)


if __name__ == "__main__":
    unittest.main()
