from __future__ import annotations

import tempfile
import time
import unittest

from agent.agent_service import AgentSettings, PrivateKnowledgeAgent
from agent.retrieval_models import SemanticCompileResult, SemanticDecision
from core.catalog_provider import MainProjectCatalogReader
from tests.test_agent_service import FakeDeepSeekClient, FakeSearchBackend


class RecordingCompiler:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.queries: list[str] = []

    def compile(self, query: str) -> SemanticCompileResult:
        self.queries.append(query)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("synthetic compiler failure")
        return SemanticCompileResult(
            decision=SemanticDecision.UNSUPPORTED_ANALYTICS,
            effective_query=query,
            model_calls=0,
        )


class FakeIndices:
    def get_mapping(self, *, index: str):
        return {
            index: {
                "mappings": {
                    "_meta": {"version": "7"},
                    "properties": {
                        "filename": {"type": "keyword"},
                        "content": {"type": "text"},
                        "created_date": {"type": "date"},
                    },
                }
            }
        }


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()


class SemanticIntegrationTests(unittest.TestCase):
    def _agent(self, mode: str, compiler=None, *, timeout: float = 2.0):
        return PrivateKnowledgeAgent(
            search_backend=FakeSearchBackend(),
            deepseek_client=FakeDeepSeekClient(),
            settings=AgentSettings(
                state_dir=tempfile.mkdtemp(),
                semantic_compiler_mode=mode,
                semantic_compiler_timeout_seconds=timeout,
            ),
            semantic_compiler=compiler,
        )

    def test_off_never_calls_compiler_and_preserves_reply_shape(self):
        compiler = RecordingCompiler()
        reply = self._agent("off", compiler).chat("ESN 是什么？")
        self.assertEqual(compiler.queries, [])
        self.assertNotIn("semantic_trace", reply.to_dict())

    def test_shadow_records_trace_without_blocking_retrieval(self):
        compiler = RecordingCompiler()
        agent = self._agent("shadow", compiler)
        reply = agent.chat("查找温度连续异常的论文")
        self.assertEqual(compiler.queries, [agent.search_backend.last_query])
        self.assertTrue(reply.evidence)
        self.assertEqual(reply.semantic_trace[0]["decision"], "unsupported_analytics")
        self.assertEqual(reply.semantic_trace[0]["model_calls"], 0)

    def test_shadow_trace_survives_reply_cache_round_trip(self):
        compiler = RecordingCompiler()
        agent = self._agent("shadow", compiler)
        first = agent.chat("查找温度连续异常的论文")
        second = agent.chat("查找温度连续异常的论文")
        self.assertEqual(second.semantic_trace, first.semantic_trace)
        self.assertEqual(len(compiler.queries), 1)

    def test_shadow_failure_keeps_legacy_retrieval_running(self):
        compiler = RecordingCompiler(fail=True)
        agent = self._agent("shadow", compiler)
        reply = agent.chat("ESN 是什么？")
        self.assertTrue(reply.evidence)
        self.assertEqual(reply.semantic_trace[0]["decision"], "error")
        self.assertEqual(agent.search_backend.last_query, compiler.queries[0])

    def test_shadow_timeout_does_not_delay_or_mutate_legacy_retrieval(self):
        compiler = RecordingCompiler(delay=0.2)
        agent = self._agent("shadow", compiler, timeout=0.01)
        started = time.perf_counter()
        reply = agent.chat("ESN 是什么？")
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.15)
        self.assertTrue(reply.evidence)
        self.assertEqual(agent.search_backend.last_query, compiler.queries[0])
        self.assertEqual(reply.semantic_trace[0]["decision"], "error")
        self.assertIn("shadow_timeout", reply.semantic_trace[0]["diagnostics"][0])

    def test_enforce_is_downgraded_until_execution_contract_is_ready(self):
        compiler = RecordingCompiler()
        agent = self._agent("enforce", compiler)
        reply = agent.chat("ESN 是什么？")
        self.assertEqual(agent._semantic_compiler_mode, "shadow")
        self.assertTrue(reply.evidence)
        self.assertEqual(len(compiler.queries), 1)

    def test_catalog_reader_exposes_only_document_retrieval_capabilities(self):
        snapshot = MainProjectCatalogReader(FakeElasticsearch(), "pdf_documents").snapshot()
        source = snapshot.sources[0]
        self.assertEqual(source.version, "7")
        self.assertIn("论文", source.aliases)
        self.assertIn("aggregate", source.capabilities)
        self.assertIn("inventory", source.capabilities)
        self.assertNotIn("sequence", source.capabilities)
        created = next(item for item in source.fields if item.backend_field == "created_date")
        self.assertIn("收录时间", created.aliases)
        self.assertNotIn("发布时间", created.aliases)


if __name__ == "__main__":
    unittest.main()
