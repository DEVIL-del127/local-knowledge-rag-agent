from __future__ import annotations

from typing import Any, TypedDict

from agent.retrieval_models import SemanticCompileResult


class RuntimeGraphState(TypedDict, total=False):
    query: str
    runtime_state: str
    previous_semantic_result: dict[str, Any]
    previous_turn_id: str
    generation_snapshot: Any
    generation_snapshot_digest: str
    semantic_result: dict[str, Any]
    route: str
    logical_plan: dict[str, Any]
    node_trace: list[dict[str, str]]


class RuntimeSemanticGraph:
    """Executable LangGraph for pinned semantic compilation and admission."""

    RETRY_CLASS = {
        "load_context": "pure", "pin_generation": "pure", "understand": "budgeted_model",
        "admission": "pure", "plan": "pure", "commit_compile": "idempotent",
    }

    def __init__(self, compiler: Any) -> None:
        from langgraph.graph import END, START, StateGraph

        self.compiler = compiler
        builder = StateGraph(RuntimeGraphState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("pin_generation", self._pin_generation)
        builder.add_node("understand", self._understand)
        builder.add_node("admission", self._admission)
        builder.add_node("plan", self._plan)
        builder.add_node("commit_compile", self._commit_compile)
        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "pin_generation")
        builder.add_edge("pin_generation", "understand")
        builder.add_edge("understand", "admission")
        builder.add_conditional_edges(
            "admission", self._continue_after_admission,
            {"continue": "plan", "stop": "commit_compile"},
        )
        builder.add_edge("plan", "commit_compile")
        builder.add_edge("commit_compile", END)
        self._graph = builder.compile()

    @classmethod
    def _trace(cls, state: RuntimeGraphState, node: str) -> list[dict[str, str]]:
        return [*list(state.get("node_trace") or []), {
            "node": node, "retry_class": cls.RETRY_CLASS[node],
        }]

    @classmethod
    def _load_context(cls, state: RuntimeGraphState) -> RuntimeGraphState:
        previous = state.get("previous_semantic_result")
        if previous is not None and not isinstance(previous, dict):
            raise TypeError("previous semantic context must be a mapping")
        return {"runtime_state": "load_context", "node_trace": cls._trace(state, "load_context")}

    @classmethod
    def _pin_generation(cls, state: RuntimeGraphState) -> RuntimeGraphState:
        snapshot = state.get("generation_snapshot")
        digest = str(getattr(snapshot, "snapshot_digest", "")) if snapshot is not None else ""
        if snapshot is not None and not digest:
            raise RuntimeError("pinned generation snapshot has no digest")
        return {"runtime_state": "pin_generation", "generation_snapshot_digest": digest,
                "node_trace": cls._trace(state, "pin_generation")}

    def _understand(self, state: RuntimeGraphState) -> RuntimeGraphState:
        query = state["query"]
        snapshot = state.get("generation_snapshot")
        previous = state.get("previous_semantic_result")
        if previous and snapshot is not None and hasattr(self.compiler, "compile_with_pinned_context"):
            result = self.compiler.compile_with_pinned_context(
                query, previous, snapshot,
                previous_turn_id=str(state.get("previous_turn_id") or ""),
            )
        elif previous and hasattr(self.compiler, "compile_with_context"):
            result = self.compiler.compile_with_context(
                query, previous, previous_turn_id=str(state.get("previous_turn_id") or ""),
            )
        elif snapshot is not None and hasattr(self.compiler, "compile_pinned"):
            result = self.compiler.compile_pinned(query, snapshot)
        else:
            result = self.compiler.compile(query)
        return {"runtime_state": "understand", "semantic_result": result.to_dict(),
                "node_trace": self._trace(state, "understand")}

    @classmethod
    def _admission(cls, state: RuntimeGraphState) -> RuntimeGraphState:
        result = dict(state.get("semantic_result") or {})
        return {"runtime_state": "admission", "route": str(result.get("decision") or "error"),
                "node_trace": cls._trace(state, "admission")}

    @staticmethod
    def _continue_after_admission(state: RuntimeGraphState) -> str:
        return "stop" if state.get("route") in {
            "clarify", "blocked", "unsupported_analytics", "error",
        } else "continue"

    @classmethod
    def _plan(cls, state: RuntimeGraphState) -> RuntimeGraphState:
        result = dict(state.get("semantic_result") or {})
        plan = dict(result.get("logical_plan") or {})
        if result.get("executable") and not plan:
            raise RuntimeError("executable semantic result has no logical plan")
        return {"runtime_state": "plan", "logical_plan": plan,
                "node_trace": cls._trace(state, "plan")}

    @classmethod
    def _commit_compile(cls, state: RuntimeGraphState) -> RuntimeGraphState:
        if not state.get("semantic_result"):
            raise RuntimeError("semantic result missing at graph commit")
        return {"runtime_state": "commit_compile",
                "node_trace": cls._trace(state, "commit_compile")}

    def _invoke(self, query: str, *, generation_snapshot: Any | None = None,
                previous_semantic_result: dict[str, Any] | None = None,
                previous_turn_id: str = "") -> SemanticCompileResult:
        state = self._graph.invoke({
            "query": query, "runtime_state": "receive", "generation_snapshot": generation_snapshot,
            "previous_semantic_result": previous_semantic_result,
            "previous_turn_id": previous_turn_id, "node_trace": [],
        })
        return SemanticCompileResult.from_dict(state["semantic_result"])

    def compile(self, query: str) -> SemanticCompileResult:
        return self._invoke(query)

    def compile_pinned(self, query: str, generation_snapshot: Any) -> SemanticCompileResult:
        return self._invoke(query, generation_snapshot=generation_snapshot)

    def compile_with_pinned_context(self, query: str, previous_semantic_result: dict[str, Any],
                                    generation_snapshot: Any, *, previous_turn_id: str) -> SemanticCompileResult:
        return self._invoke(query, generation_snapshot=generation_snapshot,
                            previous_semantic_result=previous_semantic_result,
                            previous_turn_id=previous_turn_id)
