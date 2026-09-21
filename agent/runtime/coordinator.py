"""Compatibility entry and dependency injection; no request execution branches."""
import re
from threading import Lock

from .node_services import RuntimeNodeServices, _runtime_reply


def _requires_prior_comparison(query: str) -> bool:
    """Compatibility predicate for context-bound comparison expressions."""
    value = str(query or "").strip()
    return bool(re.search(r"前(?:两|2|二)篇|这(?:两|2|二)篇|对比一下|比较一下|二者|它们", value, re.I))


class RuntimeCoordinator:
    def __init__(self, *, compiler, checkpoint_store, clarification_rewriter=None,
                 graph=None, domain_router=None, dialogue_resolver=None, expression_service=None,
                 routing_service=None):
        self._services = RuntimeNodeServices(
            compiler=compiler, checkpoint_store=checkpoint_store,
            clarification_rewriter=clarification_rewriter, graph=graph,
            domain_router=domain_router, dialogue_resolver=dialogue_resolver,
            expression_service=expression_service,
            routing_service=routing_service,
        )
        self._execution_graph = None
        self._graph_lock = Lock()

    def handle(self, user_message, *, legacy, **kwargs):
        with self._graph_lock:
            if self._execution_graph is None:
                from .execution_graph import RequestExecutionGraph
                self._execution_graph = RequestExecutionGraph(self._services)
        return self._execution_graph.invoke(user_message, legacy=legacy, kwargs=kwargs)

    def observe(self, user_message, *, reply, **kwargs):
        return self._services.observe(user_message, reply=reply, **kwargs)

    def __getattr__(self, name):
        return getattr(self._services, name)
