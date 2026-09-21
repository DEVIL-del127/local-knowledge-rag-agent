from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.runtime.checkpoint_store import CheckpointConflict, SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.runtime.dialogue import PendingInteraction
from agent.runtime.domain_router import DomainRouter, RequestDomain
from agent.runtime.expression_service import ExpressionError, ExpressionService
from agent.runtime.models import AgentState
from agent.semantic_compiler_adapter import SemanticCompilerAdapter
from agent.retrieval_models import SemanticCompileResult, SemanticDecision
from core.kb_status import KnowledgeBaseStatus
from core.retrieval_gateway import GenerationSnapshot


class DomainRouterTests(unittest.TestCase):
    def test_generic_document_and_technical_expressions(self):
        router = DomainRouter()
        for term in ("Echo State Network", "Random Forest", "Transformer"):
            for template in ("{} papers", "哪篇文档讲{}", "{}如何训练",
                             "{}有哪些参数", "{}适合什么任务", "{}如何做多步预测",
                             "综述{}应用", "{}如何使用"):
                with self.subTest(term=term, template=template):
                    self.assertEqual(router.route(template.format(term)).domain,
                                     RequestDomain.KB_DOCUMENT)

    def test_routes_daily_calculation_help_and_kb_without_model(self):
        router = DomainRouter()
        self.assertEqual(router.route("今天吃啥呀").domain, RequestDomain.GENERAL_CHAT)
        self.assertEqual(router.route("1+1等于几").domain, RequestDomain.UTILITY_CALCULATE)
        self.assertEqual(router.route("你能做什么").domain, RequestDomain.HELP)
        self.assertEqual(router.route("mcmc是什么").domain, RequestDomain.KB_DOCUMENT)
        self.assertEqual(router.route("FFT是什么").domain, RequestDomain.KB_DOCUMENT)
        self.assertEqual(router.route("Transformer的作用是什么").domain, RequestDomain.KB_DOCUMENT)
        self.assertEqual(
            router.route("2019年到2025年之间关于esn的论文有哪些").domain,
            RequestDomain.KB_DOCUMENT,
        )

    def test_routes_elliptical_year_followups_only_with_kb_context(self):
        router = DomainRouter()
        for query in ("2023年的呢", "2023的呢", "那2024年的呢"):
            with self.subTest(query=query):
                decision = router.route(query, prior_domain=RequestDomain.KB_DOCUMENT.value)
                self.assertEqual(decision.domain, RequestDomain.KB_DOCUMENT)
                self.assertEqual(decision.reason_code, "kb_context_followup")
        self.assertEqual(router.route("2023年的呢").domain, RequestDomain.GENERAL_CHAT)


class ExpressionServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ExpressionService()

    def test_evaluates_small_expression_exactly(self):
        self.assertEqual(self.service.evaluate("1+1等于几").value, "2")
        self.assertEqual(self.service.evaluate("请计算 (3+2)*4").value, "20")
        self.assertEqual(self.service.evaluate("1/2").value, "0.5")

    def test_rejects_resource_exhaustion_before_expensive_work(self):
        attacks = [
            "9**999999999",
            "1" * 65 + "+1",
            "(" * 17 + "1+1" + ")" * 17,
        ]
        for expression in attacks:
            with self.subTest(expression=expression[:30]):
                with self.assertRaises(ExpressionError) as caught:
                    self.service.evaluate(expression)
                self.assertEqual(caught.exception.code, "expression_limit_exceeded")

    def test_rejects_code_and_zero_division(self):
        with self.assertRaisesRegex(ExpressionError, "没有识别"):
            self.service.evaluate("__import__('os').system('x')")
        with self.assertRaises(ExpressionError) as caught:
            self.service.evaluate("1/0")
        self.assertEqual(caught.exception.code, "expression_division_by_zero")


class _NeverCompiler:
    def __init__(self):
        self.calls = 0

    def compile(self, query):
        self.calls += 1
        raise AssertionError("non-KB request must not reach NLU")


class _NoLegacyCalls:
    pass


class InteractionRuntimeTests(unittest.TestCase):
    def test_non_kb_requests_never_compile_or_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            compiler = _NeverCompiler()
            coordinator = RuntimeCoordinator(
                compiler=compiler,
                checkpoint_store=SQLiteCheckpointStore(Path(tmp) / "state.sqlite3"),
            )
            chat = coordinator.handle(
                "今天吃啥呀", legacy=_NoLegacyCalls(), user_id="u", session_id="s",
            )
            calculation = coordinator.handle(
                "1+1等于几", legacy=_NoLegacyCalls(), user_id="u", session_id="s",
            )
            self.assertIn("主食", chat.answer)
            self.assertEqual(calculation.answer, "1+1 = 2")
            self.assertEqual(compiler.calls, 0)
            state = coordinator.store.load("u", "s", "s")
            self.assertEqual(state.domain_decision["domain"], "utility_calculate")

    def test_confirmation_consumes_structured_pending_action_without_nlu(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteCheckpointStore(Path(tmp) / "state.sqlite3")
            state = AgentState("r1", "u", "s", "s")
            state.last_substantive_turn = {
                "turn_id": "r1", "answer": "MCMC会反复抽样来逼近目标分布。",
                "domain": "kb_document",
            }
            state.pending_interaction = PendingInteraction(
                action_id="a1", action_type="example", payload={"style": "example"},
                source_turn_id="r1", expires_at_epoch=4102444800,
            )
            store.save(state)
            compiler = _NeverCompiler()
            coordinator = RuntimeCoordinator(compiler=compiler, checkpoint_store=store)
            reply = coordinator.handle("行", legacy=_NoLegacyCalls(), user_id="u", session_id="s")
            self.assertIn("生成模型", reply.answer)
            self.assertEqual(reply.intent.reason, "dialogue_generation_unavailable")
            self.assertEqual(compiler.calls, 0)
            loaded = store.load("u", "s", "s")
            self.assertEqual(loaded.pending_interaction.state, "consumed")
            self.assertEqual(loaded.last_substantive_turn["turn_id"], "r1")

    def test_confirmation_without_pending_is_clarified(self):
        with tempfile.TemporaryDirectory() as tmp:
            compiler = _NeverCompiler()
            coordinator = RuntimeCoordinator(
                compiler=compiler,
                checkpoint_store=SQLiteCheckpointStore(Path(tmp) / "state.sqlite3"),
            )
            reply = coordinator.handle("行", legacy=_NoLegacyCalls(), user_id="u", session_id="s")
            self.assertEqual(reply.intent.reason, "confirmation_without_pending")
            self.assertEqual(compiler.calls, 0)

    def test_kb_compile_and_execution_share_one_pinned_generation(self):
        class Compiler:
            def __init__(self):
                self.snapshots = []

            def compile_pinned(self, query, snapshot):
                self.snapshots.append(snapshot)
                return SemanticCompileResult(
                    decision=SemanticDecision.CLARIFY,
                    effective_query=query,
                    clarification_questions=[{
                        "question_id": "q1", "prompt": "请补充范围",
                        "expected_answer_type": "text", "candidate_ids": [],
                    }],
                )

        pinned = GenerationSnapshot(
            generation_id="g1", revision=7, registry_revision=7,
            es_physical_index="idx_g1", vector_collection="vec_g1",
            embedding_model="bge-m3", embedding_dimension=1024,
            snapshot_digest="snapshot-g1",
        )

        class Gateway:
            registry = object()

            def __init__(self):
                self.pin_calls = 0

            def pin(self):
                self.pin_calls += 1
                return pinned

            def status(self, snapshot):
                assert snapshot is pinned
                return KnowledgeBaseStatus(
                    source_id="idx_g1", provider_available=True,
                    index_exists=True, document_count=1,
                    ingestion_generation="g1",
                )

        gateway = Gateway()
        skill = type("Skill", (), {"gateway": gateway})()

        class Registry:
            def has(self, name):
                return name == "search_private_kb"

            def get(self, name):
                return skill

        legacy = type("Legacy", (), {
            "registry": Registry(),
            "answer_from_observations": lambda *args, **kwargs: None,
        })()
        with tempfile.TemporaryDirectory() as tmp:
            compiler = Compiler()
            coordinator = RuntimeCoordinator(
                compiler=compiler,
                checkpoint_store=SQLiteCheckpointStore(Path(tmp) / "state.sqlite3"),
            )
            reply = coordinator.handle(
                "关于ESN的论文", legacy=legacy, user_id="u", session_id="s",
            )
        self.assertEqual(reply.intent.reason, "runtime_clarification")
        self.assertEqual(gateway.pin_calls, 1)
        self.assertEqual(compiler.snapshots, [pinned])


class CheckpointCASTests(unittest.TestCase):
    def test_stale_concurrent_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteCheckpointStore(Path(tmp) / "state.sqlite3")
            original = AgentState("r1", "u", "s", "s")
            store.save(original)
            first = store.load("u", "s", "s")
            second = store.load("u", "s", "s")
            first.raw_query = "first"
            store.save(first)
            second.raw_query = "second"
            with self.assertRaises(CheckpointConflict):
                store.save(second)
            self.assertEqual(store.load("u", "s", "s").raw_query, "first")


def _semantic_backend():
    class Indices:
        def get_mapping(self, *, index):
            return {index: {"mappings": {"_meta": {"version": "test-v1"}, "properties": {
                "filename": {"type": "keyword"}, "content": {"type": "text"},
                "publication_year": {"type": "integer"},
            }}}}

    es = type("ES", (), {"indices": Indices()})()
    manager = type("Manager", (), {"es": es, "index_name": "documents"})()
    return type("Backend", (), {"es_manager": manager})()


class PersistedSemanticContextTests(unittest.TestCase):
    def test_ordinary_followup_rehydrates_request_ir_in_fresh_adapter(self):
        first_adapter = SemanticCompilerAdapter(search_backend=_semantic_backend(), enable_llm=False)
        first = first_adapter.compile("2019年到2025年之间关于ESN的论文有哪些")

        second_adapter = SemanticCompilerAdapter(search_backend=_semantic_backend(), enable_llm=False)
        second = second_adapter.compile_with_context(
            "那2024年的呢", first.to_dict(), previous_turn_id="turn-1",
        )

        self.assertTrue(second.executable)
        bridge = second.contract_reports["literature_contract_bridge"]
        self.assertTrue(bridge["contextual_topic_inherited"])
        self.assertEqual(bridge["sense_id"], "echo-state-network")


if __name__ == "__main__":
    unittest.main()
