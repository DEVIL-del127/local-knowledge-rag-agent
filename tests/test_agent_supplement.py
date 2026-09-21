# -*- coding: utf-8 -*-
"""Agent 补充测试(2026-08-20): 技能注册表 / Service 扩展路径 / CLI 循环 / MCP 适配
盲区补测: 原套件未覆盖 CHAT_DAILY/CHAT_HELP 直答、熔断/预算降级、KB_MANAGE、
复合问题多路检索、证据粗筛、blocked TTL、SkillRegistry 过滤、agent_cli、mcp_adapter。
运行: venv/bin/python tests/test_agent_supplement.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_cli import MAX_HISTORY_ROUNDS, run_agent_loop
from agent.agent_limits import BudgetExceededError, CircuitOpenError
from agent.agent_models import AgentReply, IntentPrediction, IntentType
from agent.agent_service import AgentSettings, PrivateKnowledgeAgent
from agent.agent_skills import (
    KnowledgeBaseManageSkill,
    SkillRegistry,
    SkillSpec,
    format_evidence_block,
)


# ============ 测试替身 ============
class FakeSearchBackend:
    """带 pdf_dir 的检索后端"""

    def __init__(self, pdf_dir):
        self.pdf_dir = pdf_dir
        self.last_query = ""

    def hybrid_search(self, query, top_n=10, include_chunks=False):
        self.last_query = query
        doc = {
            "filename": "esn-paper.pdf",
            "rrf": 1.0,
            "highlights": ["ESN 可用于时间序列建模与误差补偿。"],
            "content": "ESN 可用于时间序列建模与误差补偿。",
        }
        if include_chunks:
            return [doc], []
        return [doc]

    def vec_search(self, query, top_k=10):
        return [{
            "filename": "esn-paper.pdf", "page": 3, "chunk": 0,
            "text": "ESN 是一种递归神经网络。", "score": 0.94,
        }]


class EmptySearchBackend(FakeSearchBackend):
    def hybrid_search(self, query, top_n=10, include_chunks=False):
        self.last_query = query
        if include_chunks:
            return [], []
        return []

    def vec_search(self, query, top_k=10):
        return []


class FakeDeepSeekClient:
    def __init__(self):
        self.settings = type("Settings", (), {"router_strict": True})()
        self.text_calls = 0
        self.router_calls = 0

    def invoke_router(self, *, messages, tool_schema, user_id="default"):
        self.router_calls += 1
        last_user = [m for m in messages if m.get("role") == "user"][-1]["content"]
        return {
            "intent": "kb_search", "needs_retrieval": True, "confidence": 0.88,
            "user_goal": last_user, "rewritten_query": last_user, "top_k": 3,
            "response_style": "grounded", "clarification_question": "",
            "reason": "fake", "metadata_filters": {}, "has_anaphora": False,
            "skill_calls": [{
                "skill_name": "search_private_kb", "purpose": "retrieve evidence",
                "arguments": {"query": last_user, "top_k": 3},
            }],
        }

    def invoke_text(self, *, system_prompt, user_prompt, history=None,
                    temperature=None, max_tokens=None, user_id="default"):
        self.text_calls += 1
        return "基于证据的回答。[E1]"


class FailingDeepSeekClient(FakeDeepSeekClient):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def invoke_text(self, **kwargs):
        raise self.exc


class FakeMemory:
    def __init__(self):
        self.ingested = []

    def recall_memory(self, *, query, user_id):
        return "长期记忆: 用户在研究 ESN 误差补偿。" if "ESN" in query else ""

    def recall_for_current(self, *, query, window_messages, session_id, user_id, has_anaphora_hint):
        return ""

    def admin(self, *, command, user_id, args=None):
        if command == "stats":
            return {"total": 3, "by_type": {"fact": 2, "summary": 1}, "avg_confidence": 0.8}
        if command == "list":
            return {"total": 1, "items": [{"id": "abc123", "value": "测试记忆内容"}]}
        return {"ok": True}

    def ingest_message(self, **kwargs):
        self.ingested.append(kwargs)


def _make_agent(backend=None, client=None, memory=None, tmp=None):
    tmp = tmp or tempfile.mkdtemp()
    return PrivateKnowledgeAgent(
        search_backend=backend or FakeSearchBackend(tmp),
        deepseek_client=client or FakeDeepSeekClient(),
        settings=AgentSettings(state_dir=tmp),
        memory=memory,
    )


# ============ 1. SkillRegistry ============
class _DummySkill:
    def __init__(self, name, enabled=True, user_invocable=True):
        self.spec = SkillSpec(
            name=name, description=f"skill {name}",
            enabled=enabled, user_invocable=user_invocable,
        )

    def invoke(self, arguments):
        return {"called": self.spec.name, "args": dict(arguments)}


class SkillRegistryTests(unittest.TestCase):
    def test_register_and_execute(self):
        reg = SkillRegistry()
        reg.register(_DummySkill("s1"))
        self.assertTrue(reg.has("s1"))
        result = reg.execute("s1", {"a": 1})
        self.assertEqual(result["called"], "s1")

    def test_get_missing_raises_keyerror(self):
        reg = SkillRegistry()
        with self.assertRaises(KeyError):
            reg.get("nope")

    def test_router_skills_filters_disabled_and_background(self):
        reg = SkillRegistry()
        reg.register(_DummySkill("normal"))
        reg.register(_DummySkill("disabled", enabled=False))
        reg.register(_DummySkill("background", user_invocable=False))
        names = [s["name"] for s in reg.router_skills()]
        self.assertEqual(names, ["normal"])

    def test_router_dict_shape(self):
        reg = SkillRegistry()
        reg.register(_DummySkill("s1"))
        spec = reg.router_skills()[0]
        self.assertEqual(
            set(spec.keys()), {"name", "description", "kind", "tags"},
        )


class KnowledgeBaseManageSkillTests(unittest.TestCase):
    def test_list_filters_pdf_and_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("刘月论文.pdf", "ESN笔记.pdf", "readme.txt"):
                open(os.path.join(tmp, name), "w").close()
            skill = KnowledgeBaseManageSkill(pdf_dir=tmp)
            result = skill.invoke({"command": "list", "query": "刘月"})
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["files"], ["刘月论文.pdf"])

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.pdf"), "w").close()
            open(os.path.join(tmp, "b.pdf"), "w").close()
            skill = KnowledgeBaseManageSkill(pdf_dir=tmp)
            result = skill.invoke({"command": "stats"})
            self.assertEqual(result["pdf_count"], 2)

    def test_unknown_command(self):
        skill = KnowledgeBaseManageSkill(pdf_dir=tempfile.mkdtemp())
        result = skill.invoke({"command": "explode"})
        self.assertIn("error", result)

    def test_format_evidence_block_empty(self):
        self.assertEqual(format_evidence_block([]), "未检索到可用证据。")


# ============ 2. Service 扩展路径 ============
class ServiceExtraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.backend = FakeSearchBackend(self.tmp)
        self.client = FakeDeepSeekClient()
        self.agent = _make_agent(self.backend, self.client, tmp=self.tmp)

    def test_greeting_direct_answer_no_retrieval(self):
        """CHAT_DAILY: 直接回答, 不检索"""
        reply = self.agent.chat("你好")
        self.assertEqual(reply.intent.intent, IntentType.CHAT_DAILY)
        self.assertEqual(reply.intent.router_source, "rule")
        self.assertEqual(reply.evidence, [])
        self.assertEqual(self.client.text_calls, 1)
        self.assertEqual(self.backend.last_query, "")  # 未触发检索

    def test_help_question_direct_answer(self):
        reply = self.agent.chat("你能做什么")
        self.assertEqual(reply.intent.intent, IntentType.CHAT_HELP)
        self.assertEqual(reply.evidence, [])
        self.assertEqual(self.client.text_calls, 1)

    def test_circuit_open_returns_error_reply(self):
        agent = _make_agent(
            self.backend, FailingDeepSeekClient(CircuitOpenError("熔断中")), tmp=self.tmp,
        )
        reply = agent.chat("查一下 ESN 误差补偿")
        self.assertEqual(reply.intent.reason, "circuit_open")
        self.assertIn("熔断", reply.answer)

    def test_budget_exceeded_returns_error_reply(self):
        agent = _make_agent(
            self.backend, FailingDeepSeekClient(BudgetExceededError("预算超限")), tmp=self.tmp,
        )
        reply = agent.chat("查一下 ESN 误差补偿")
        self.assertEqual(reply.intent.reason, "budget_exceeded")
        self.assertIn("预算", reply.answer)

    def test_kb_manage_lists_documents(self):
        open(os.path.join(self.tmp, "刘月论文.pdf"), "w").close()
        open(os.path.join(self.tmp, "ESN笔记.pdf"), "w").close()
        reply = self.agent.chat("库里有哪些文档")
        self.assertEqual(reply.intent.intent, IntentType.KB_MANAGE)
        self.assertIn("刘月论文.pdf", reply.answer)
        self.assertEqual(reply.executed_skills[0]["skill_name"], "kb_manage")

    def test_compound_question_multi_retrieval(self):
        """复合问题: 主问题 + 附加问题各一路检索"""
        reply = self.agent.chat("什么是GAN？顺便查一下贝叶斯论文")
        self.assertEqual(reply.intent.intent, IntentType.KB_SEARCH)
        executed = [s["skill_name"] for s in reply.executed_skills]
        self.assertEqual(executed.count("search_private_kb"), 2)

    def test_evidence_filtered_by_entity_relevance(self):
        """证据粗筛: 实体出现在 snippet 中则保留"""
        class SnippetBackend(FakeSearchBackend):
            def hybrid_search(self, query, top_n=10, include_chunks=False):
                self.last_query = query
                doc = {
                    "filename": "esn-paper.pdf", "rrf": 1.0,
                    "highlights": ["ESN 用于时间序列建模。"],
                    "content": "ESN 用于时间序列建模。",
                }
                if include_chunks:
                    return [doc], []
                return [doc]

        agent = _make_agent(SnippetBackend(self.tmp), tmp=self.tmp)
        reply = agent.chat("查一下 ESN 误差补偿")
        self.assertEqual(reply.intent.intent, IntentType.KB_SEARCH)
        self.assertTrue(reply.evidence)

    def test_blocked_query_repeat_returns_clarify(self):
        """低置信度/证据不足被 blocked: TTL 内重复问返回澄清"""
        agent = _make_agent(EmptySearchBackend(self.tmp), tmp=self.tmp)
        first = agent.chat("这个方案和前文的联系在哪里？")
        self.assertIn("证据", first.answer)
        second = agent.chat("这个方案和前文的联系在哪里？")
        self.assertEqual(second.intent.intent, IntentType.CLARIFY)
        self.assertEqual(second.intent.reason, "repeat_blocked_query")

    def test_memory_recall_with_injection(self):
        agent = _make_agent(self.backend, client=FakeDeepSeekClient(), memory=FakeMemory(), tmp=self.tmp)
        reply = agent.chat("还记得我之前说的ESN吗")
        # rule 路由命中 MEMORY_RECALL("还记得")
        self.assertEqual(reply.intent.intent, IntentType.MEMORY_RECALL)
        self.assertTrue(reply.executed_skills)
        self.assertEqual(reply.executed_skills[0]["skill_name"], "memory_recall")

    def test_evidence_signature_stability(self):
        from agent.agent_service import PrivateKnowledgeAgent as PKA

        e1 = [type("E", (), {"source": "a.pdf", "page": 1, "chunk": 0, "snippet": "内容内容"})()]
        e2 = [type("E", (), {"source": "a.pdf", "page": 1, "chunk": 0, "snippet": "内容内容"})()]
        self.assertEqual(PKA._evidence_signature(e1), PKA._evidence_signature(e2))
        e3 = [type("E", (), {"source": "b.pdf", "page": 1, "chunk": 0, "snippet": "内容内容"})()]
        self.assertNotEqual(PKA._evidence_signature(e1), PKA._evidence_signature(e3))

    def test_coerce_evidence_drops_dirty_items(self):
        from agent.agent_service import PrivateKnowledgeAgent as PKA

        evidence = PKA._coerce_evidence([
            {"source": "a.pdf", "snippet": "好内容"},
            {"source": "", "snippet": "无来源丢弃"},
            {"source": "b.pdf", "snippet": "  "},
            {"source": "c.pdf", "snippet": "OK", "page": "3", "chunk": "x"},  # 非 int 归一为 None
        ])
        self.assertEqual(len(evidence), 2)
        self.assertIsNone(evidence[1].page)
        self.assertIsNone(evidence[1].chunk)


# ============ 3. CLI 循环 ============
class _CliAgent:
    def __init__(self):
        self.calls = 0
        self.history_lens = []

    def chat(self, user_message, history=None, user_id="default", session_id=""):
        self.calls += 1
        self.history_lens.append(len(history or []))
        if self.calls == 2:
            raise RuntimeError("模拟内部故障")
        return AgentReply(
            answer=f"回答{self.calls}",
            intent=IntentPrediction(
                intent=IntentType.KB_SEARCH, needs_retrieval=True, confidence=0.9,
                user_goal=user_message, router_source="rule",
            ),
            evidence=[],
        )


class _CliAgentNoFail:
    """不抛异常的 CLI 替身(测历史裁剪专用)"""

    def __init__(self):
        self.calls = 0
        self.history_lens = []

    def chat(self, user_message, history=None, user_id="default", session_id=""):
        self.calls += 1
        self.history_lens.append(len(history or []))
        return AgentReply(
            answer=f"回答{self.calls}",
            intent=IntentPrediction(
                intent=IntentType.KB_SEARCH, needs_retrieval=True, confidence=0.9,
                user_goal=user_message, router_source="rule",
            ),
            evidence=[],
        )


class CliLoopTests(unittest.TestCase):
    def test_request_ids_are_allocated_per_input(self):
        class RequestAgent(_CliAgentNoFail):
            def __init__(self):
                super().__init__()
                self.request_ids = []

            def chat(self, user_message, *, request_id, **kwargs):
                self.request_ids.append(request_id)
                return super().chat(user_message, **kwargs)

        agent = RequestAgent()
        with mock.patch("builtins.input", side_effect=["same", "same", "q"]):
            run_agent_loop(agent)
        self.assertEqual(len(set(agent.request_ids)), 2)
        self.assertTrue(all(agent.request_ids))

    def test_type_error_after_dispatch_is_not_retried(self):
        class FailingAgent:
            calls = 0

            def chat(self, user_message, **kwargs):
                self.calls += 1
                raise TypeError("synthetic post-dispatch error")

        agent = FailingAgent()
        with mock.patch("builtins.input", side_effect=["question", "q"]):
            run_agent_loop(agent)
        self.assertEqual(agent.calls, 1)

    def test_quit_exits(self):
        agent = _CliAgent()
        with mock.patch("builtins.input", side_effect=["quit"]):
            run_agent_loop(agent, user_id="u1")
        self.assertEqual(agent.calls, 0)

    def test_internal_error_does_not_crash(self):
        """agent.chat 抛异常: 打印错误继续, 会话不崩"""
        agent = _CliAgent()
        with mock.patch("builtins.input", side_effect=["问题一", "问题二", "问题三", "q"]):
            run_agent_loop(agent, user_id="u1")
        self.assertEqual(agent.calls, 3)  # 第三次仍正常

    def test_history_cropped_at_max_rounds(self):
        agent = _CliAgentNoFail()
        questions = [f"问题{i}" for i in range(15)] + ["q"]
        with mock.patch("builtins.input", side_effect=questions):
            run_agent_loop(agent, user_id="u1")
        # 第 13 轮起 history 应为 24 条(裁剪后), 不再增长
        self.assertEqual(agent.history_lens[12], MAX_HISTORY_ROUNDS * 2)
        self.assertEqual(agent.history_lens[13], MAX_HISTORY_ROUNDS * 2)
        self.assertEqual(agent.history_lens[14], MAX_HISTORY_ROUNDS * 2)

    def test_memory_ingest_failure_ignored(self):
        class FailingMemory:
            def ingest_message(self, **kwargs):
                raise RuntimeError("记忆写入故障")

        agent = _CliAgent()
        mem = FailingMemory()
        with mock.patch("builtins.input", side_effect=["问题", "q"]):
            run_agent_loop(agent, user_id="u1", memory=mem, session_id="s1")
        self.assertEqual(agent.calls, 1)  # 记忆写入失败不影响主流程

    def test_blank_input_skipped(self):
        agent = _CliAgent()
        with mock.patch("builtins.input", side_effect=["", "  ", "问题", "q"]):
            run_agent_loop(agent, user_id="u1")
        self.assertEqual(agent.calls, 1)  # 空输入不计数

    def test_eof_exits(self):
        agent = _CliAgent()
        with mock.patch("builtins.input", side_effect=EOFError):
            run_agent_loop(agent, user_id="u1")
        self.assertEqual(agent.calls, 0)


# ============ 4. MCP 适配 ============
class McpAdapterTests(unittest.TestCase):
    def setUp(self):
        from memory.mcp_adapter import MEMORY_TOOL_DEFINITIONS, MemoryMcpAdapter

        self.definitions = MEMORY_TOOL_DEFINITIONS
        self.adapter = MemoryMcpAdapter(FakeMemory())

    def test_two_tool_definitions(self):
        names = [d["name"] for d in self.definitions]
        self.assertEqual(names, ["memory.search", "memory.admin"])

    def test_schema_required_fields(self):
        search_schema = self.definitions[0]["inputSchema"]
        self.assertEqual(search_schema["required"], ["query", "user_id"])
        self.assertEqual(search_schema["additionalProperties"], False)
        admin_schema = self.definitions[1]["inputSchema"]
        self.assertEqual(admin_schema["required"], ["command", "user_id"])

    def test_search_missing_args(self):
        result = self.adapter.execute("memory.search", {})
        self.assertIn("error", result)

    def test_search_hit(self):
        result = self.adapter.execute("memory.search", {"query": "ESN", "user_id": "u1"})
        self.assertTrue(result["found"])
        self.assertIn("ESN", result["injection"])

    def test_search_miss(self):
        result = self.adapter.execute("memory.search", {"query": "不存在的词XYZ", "user_id": "u1"})
        self.assertFalse(result["found"])

    def test_admin_missing_user_id(self):
        result = self.adapter.execute("memory.admin", {"command": "list"})
        self.assertIn("error", result)

    def test_unknown_tool(self):
        result = self.adapter.execute("no_such_tool", {})
        self.assertIn("error", result)

    def test_tool_definitions_consistency(self):
        """定义与 execute 分发一一对应"""
        result = self.adapter.tool_definitions()
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
