"""Request-level execution graph with one durable admission/commit boundary.

Retrieval, validation, synthesis and citations are separate execution nodes.
Resume validates its retained snapshot before compiling and rejoins binding.
No graph retry policy is installed on nodes which may perform model/tool calls.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, TypedDict

from agent.agent_models import AgentReply
from agent.agent_limits import BudgetExceededError


class RequestGraphState(TypedDict, total=False):
    query: str
    kwargs: dict[str, Any]
    legacy: Any
    claim: Any
    branch: str
    previous: Any
    act: Any
    domain: Any
    routing_hint: Any
    reply: AgentReply
    trace: list[str]
    prepared: dict[str, Any]


class RequestExecutionGraph:
    def __init__(self, services):
        from langgraph.graph import START, END, StateGraph
        self.services = services
        self._active_claim = ContextVar("request_graph_claim", default=None)
        builder = StateGraph(RequestGraphState)
        builder.add_node("admit", self._admit)
        builder.add_node("route", self._route)
        for branch in ("output_action", "non_kb"):
            builder.add_node(branch, self._execute)
            builder.add_edge(branch, "commit")
        builder.add_node("resume", self._execute)
        builder.add_node("kb", self._prepare_kb)
        builder.add_node("understand", self._understand)
        builder.add_node("bind", self._read_only)
        pipeline = (("retrieve", "_retrieve_bound"), ("validate_evidence", "_validate_retrieved"),
                    ("synthesize", "_synthesize_bound"), ("verify_citations", "_verify_generated"))
        for node, method in pipeline:
            builder.add_node(node, self._pipeline_node(node, method))
        builder.add_conditional_edges("kb", lambda state: "stop" if "reply" in state else "continue",
                                      {"stop": "commit", "continue": "understand"})
        builder.add_conditional_edges("understand", lambda state: "stop" if "reply" in state else "continue",
                                      {"stop": "commit", "continue": "bind"})
        stages = ["bind", *[name for name, _ in pipeline], "commit"]
        for node, successor in zip(stages, stages[1:]):
            builder.add_conditional_edges(node, lambda state: "stop" if "reply" in state else "continue",
                                          {"stop": "commit", "continue": successor})
        builder.add_conditional_edges("resume", lambda state: "stop" if "reply" in state else "continue",
                                      {"stop": "commit", "continue": "bind"})
        builder.add_node("commit", self._commit)
        builder.add_edge(START, "admit")
        builder.add_conditional_edges("admit", lambda state: "done" if "reply" in state else "route",
                                      {"done": END, "route": "route"})
        builder.add_conditional_edges("route", lambda state: state["branch"],
                                      {name: name for name in ("output_action", "non_kb", "resume", "kb")})
        builder.add_edge("commit", END)
        self.graph = builder.compile()

    def _admit(self, state):
        result = self.services._admit_request(state["query"], legacy=state["legacy"], **state["kwargs"])
        if isinstance(result, AgentReply):
            return {"reply": result, "trace": ["admit"]}
        # Scope-local exception cleanup; never stored in an instance shared by requests.
        holder = self._active_claim.get()
        holder.append(result)
        return {"claim": result, "trace": ["admit"]}

    def _route(self, state):
        with self.services.store.node_scope(state["claim"], "route"):
            result = self.services._route_request(state["query"], legacy=state["legacy"], **state["kwargs"])
        return {**result, "trace": [*state["trace"], "route"]}

    def _execute(self, state):
        service = self.services
        user, session, thread = service._identity(state["kwargs"])
        common = dict(previous=state.get("previous"), legacy=state["legacy"],
                      user_id=user, session_id=session, thread_id=thread,
                      history=state["kwargs"].get("history"))
        with service.store.node_scope(state["claim"], state["branch"]):
            if state["branch"] == "output_action":
                common.pop("history", None)
                reply = service._handle_dialogue_control(state["query"], state["act"], **common)
            elif state["branch"] == "non_kb":
                reply = service._handle_non_kb(state["query"], state["domain"], state["act"], **common)
            elif state["branch"] == "resume":
                reply = service._resume(state["previous"], state["query"], legacy=state["legacy"], kwargs=state["kwargs"])
        if not isinstance(reply, AgentReply):
            if state["branch"] == "resume" and isinstance(reply, dict):
                return self._step(state, reply, "resume")
            raise RuntimeError("graph_branch_missing_terminal_reply")
        return {"reply": reply, "trace": [*state["trace"], state["branch"]]}

    @staticmethod
    def _step(state, result, node):
        return {("reply" if isinstance(result, AgentReply) else "prepared"): result,
                "trace": [*state["trace"], node]}

    def _prepare_kb(self, state):
        with self.services.store.node_scope(state["claim"], "pin_generation"):
            result = self.services._prepare_kb_request(state["query"], previous=state["previous"],
                domain=state["domain"], dialogue_act=state["act"], legacy=state["legacy"],
                kwargs=state["kwargs"], routing_hint=state.get("routing_hint"))
        return self._step(state, result, "pin_generation")

    def _understand(self, state):
        with self.services.store.node_scope(state["claim"], "understand"):
            result = self.services._compile_kb_request(state["prepared"], legacy=state["legacy"], kwargs=state["kwargs"])
        return self._step(state, result, "understand")

    def _read_only(self, state):
        prepared = state["prepared"]
        with self.services.store.node_scope(state["claim"], "bind"):
            reply = self.services._bind_read_only(prepared["state"], prepared["query"], result=prepared["result"],
                generation_snapshot=prepared["generation_snapshot"], legacy=state["legacy"], kwargs=state["kwargs"])
        return self._step(state, reply, "bind")

    def _pipeline_node(self, node, method):
        def execute(state):
            with self.services.store.node_scope(state["claim"], node):
                result = getattr(self.services, method)(state["prepared"], legacy=state["legacy"], kwargs=state["kwargs"])
            return self._step(state, result, node)
        return execute

    def _commit(self, state):
        trace = [*state["trace"], "commit"]
        reply = state["reply"]
        reply.semantic_trace.append({"runtime_execution_graph": trace})
        self.services.store.complete_operation(state["claim"], reply.to_dict())
        return {"reply": reply, "trace": trace}

    def invoke(self, query, *, legacy, kwargs):
        holder = []
        token = self._active_claim.set(holder)
        try:
            return self.graph.invoke({"query": query, "kwargs": kwargs, "legacy": legacy})["reply"]
        except BudgetExceededError:
            from .node_services import _runtime_reply
            reply = _runtime_reply(
                query, "可用调用或令牌预算不足，本次操作已停止，不会自动重试。",
                reason="runtime_budget_exhausted",
            )
            if holder:
                claim = holder[0]
                claim.memory_events.clear()
                from .models import RuntimeState
                pending = claim.pending_state
                if pending is not None:
                    pending.transition(RuntimeState.FAILED, reason="runtime_budget_exhausted")
                    pending.error = {"type": "runtime_budget_exhausted", "message": "shared budget denied operation"}
                # Persist a terminal result so a repeated request cannot obtain a
                # fresh budget merely because a downstream node was denied.
                try:
                    self.services.store.complete_operation(claim, reply.to_dict())
                except Exception:
                    self.services.store.abandon_operation(claim)
                    raise
            return reply
        except Exception:
            if holder:
                claim = holder[0]
                self.services.store.abandon_operation(claim)
                if self.services.store.operation_status(claim) == "cancelled":
                    from .node_services import _runtime_reply
                    return _runtime_reply(query, "该操作已取消，迟到的结果没有提交。", reason="runtime_operation_cancelled")
            raise
        finally:
            self._active_claim.reset(token)
