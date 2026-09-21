from agent.runtime.models import AgentState
from agent.runtime.checkpoint_store import SQLiteCheckpointStore
from agent.runtime.coordinator import RuntimeCoordinator
from agent.retrieval_models import SemanticCompileResult, SemanticDecision


def test_new_subject_does_not_append_old_topic_or_reuse_ir(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite3")
    state = AgentState("prior", "u", "s", "s")
    state.last_substantive_turn = {"domain": "kb_document", "answer": "old"}
    state.semantic_result = {"old_ir": True}
    state.context_snapshot = {"literature": {"canonical_topic": "ESN", "sense_id": "echo-state-network",
                                              "year_from": 2024, "year_to": 2024}}
    store.save(state)
    seen = []
    class Compiler:
        def compile(self, query):
            seen.append(query)
            return SemanticCompileResult(decision=SemanticDecision.CLARIFY, effective_query=query,
                clarification_questions=[{"question_id": "q", "prompt": "选择范围", "expected_answer_type": "text"}])
        def compile_with_context(self, *args, **kwargs):
            raise AssertionError("new topic reused old IR")
    runtime = RuntimeCoordinator(compiler=Compiler(), checkpoint_store=store)
    runtime.handle("关于Transformer的论文", legacy=object(), user_id="u", session_id="s")
    assert seen == ["关于Transformer的论文"]
    saved = store.load("u", "s", "s")
    assert saved.context_snapshot.get("literature", {}).get("year_from") is None
    assert "ESN" not in str(saved.context_snapshot.get("literature", {}))
