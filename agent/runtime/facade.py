from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RuntimeAgentFacade:
    """Compatibility facade; shadow failures never affect the legacy response."""

    def __init__(self, *, legacy: Any, mode: str, coordinator: Any | None = None) -> None:
        self.legacy = legacy
        self.mode = mode
        self.coordinator = coordinator

    def chat(self, user_message: str, **kwargs):
        if self.mode == "enforce" and self.coordinator is not None:
            memory = getattr(self.legacy, "memory", None)
            if memory is not None:
                try:
                    self.coordinator.store.project_memory_events(memory)
                except Exception:
                    from .domain_router import DomainRouter, RequestDomain
                    memory_request = DomainRouter().route(user_message).domain == RequestDomain.MEMORY
                    if not memory_request:
                        logger.warning("Memory recovery pending; unrelated request remains available")
                    else:
                        return self._memory_pending_reply(user_message)
            user_id = str(kwargs.get("user_id", "default"))
            limiter = getattr(self.legacy, "_rate_limiter", None)
            if limiter is not None:
                allowed, retry_after = limiter.allow(user_id)
                if not allowed:
                    from agent.agent_models import AgentReply, IntentPrediction, IntentType
                    return AgentReply(
                        answer=f"请求太频繁了，请 {max(1, int(retry_after))} 秒后再试。",
                        intent=IntentPrediction(intent=IntentType.CLARIFY, needs_retrieval=False,
                                                confidence=1.0, user_goal=user_message,
                                                router_source="rate_limit", reason="rate_limited"),
                    )
            started = time.time()
            handled = self.coordinator.handle(user_message, legacy=self.legacy, **kwargs)
            if handled is not None:
                if memory is not None:
                    try:
                        self.coordinator.store.project_memory_events(memory)
                    except Exception:
                        # The reply and intent already committed; do not retry the
                        # request or misreport a durable operation as rolled back.
                        logger.warning("Committed memory projection pending; retained for recovery")
                self._emit_enforce_telemetry(user_message, handled, started, **kwargs)
                return handled
            raise RuntimeError("enforce runtime returned without a terminal reply")
        reply = self.legacy.chat(user_message, **kwargs)
        if self.mode == "shadow" and self.coordinator is not None:
            try:
                self.coordinator.observe(user_message, reply=reply, **kwargs)
            except Exception as exc:
                logger.warning("runtime shadow failed; legacy reply preserved: %s", exc)
        return reply

    def _emit_enforce_telemetry(self, query: str, reply: Any, started: float, **kwargs) -> None:
        bus = getattr(self.legacy, "_telemetry_bus", None)
        if bus is None:
            return
        try:
            from telemetry.context import RequestContext
            user_id, session_id, thread_id = self.coordinator._identity(kwargs)
            state = self.coordinator.store.load(user_id, session_id, thread_id)
            execution_snapshot = dict(getattr(state, "request_execution_snapshot", None) or {})
            observations = list(getattr(state, "observations", []) or [])
            channels = [channel for item in observations
                        for channel in ((item.get("result") or {}).get("executed_channels") or [])]
            bus.emit(RequestContext(
                request_id=str(getattr(state, "request_id", "")), user_id=user_id,
                session_id=session_id, node_id=getattr(bus, "node_id", "node-a"),
                query_raw=query, query_cleaned=query, query_primary=query, query_rewritten=query,
                execution={
                    "snapshot_digest": execution_snapshot.get("generation_snapshot_digest", ""),
                    "request_ir_digest": getattr(state, "ir_digest", ""),
                    "logical_plan_digest": execution_snapshot.get("logical_plan_digest", ""),
                    "literature_query_digest": execution_snapshot.get("literature_query_digest", ""),
                    "execution_snapshot_digest": execution_snapshot.get("cache_digest", ""),
                    "admission": getattr(getattr(state, "state", None), "value", ""),
                    "skill_bindings": [item.get("skill_name", "") for item in
                                       (getattr(state, "bound_skill_calls", []) or [])],
                    "executed_channels": channels,
                    "result_type": getattr(getattr(state, "state", None), "value", ""),
                    "result_count": len(getattr(reply, "evidence", []) or []),
                    "latency_ms": int((time.time() - started) * 1000),
                    "citation_valid": bool(getattr(reply, "evidence", []) or []),
                    "error_type": str((getattr(state, "error", None) or {}).get("type", "")),
                },
            ))
        except Exception as exc:
            logger.warning("enforce telemetry failed: %s", exc, exc_info=True)

    @staticmethod
    def _memory_pending_reply(query):
        from agent.agent_models import AgentReply, IntentPrediction, IntentType
        return AgentReply(
            answer="上次已提交的记忆操作尚未恢复完成，本次记忆请求未执行。请稍后重试。",
            intent=IntentPrediction(intent=IntentType.CLARIFY, needs_retrieval=False,
                                    confidence=1.0, user_goal=query,
                                    router_source="runtime", reason="memory_projection_pending"),
        )

    def __getattr__(self, name: str):
        return getattr(self.legacy, name)
