# bus.py - TelemetryBus: 从 RequestContext 组装检索词事件, 交给 recorder 落盘
# 只记问题侧(question_only): 意图/结果/耗时/回答/trace 均不采集
# 解耦点: 主链路只调 emit(ctx) 一次; 事件字段/存储细节全部隔离在这里
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any

from telemetry.context import RequestContext
from telemetry.search_log.config import SearchLogConfig
from telemetry.search_log.models import SearchLogEvent, now_iso
from telemetry.search_log.recorder import SearchRecorder
from telemetry.search_log.storage import LogStorage, build_storage

logger = logging.getLogger(__name__)


class TelemetryBus:
    """组装层: 只产 search_log 事件(问题侧), 幂等防重, fail-silent"""

    def __init__(
        self,
        node_id: str = "node-a",
        config: SearchLogConfig | None = None,
        storage: LogStorage | None = None,
        recorder: SearchRecorder | None = None,
    ) -> None:
        config = config or SearchLogConfig()
        self.node_id = node_id
        self._config = config
        # 未显式传 storage 时按配置自动装配(json/sqlite/forwarding), 保证 forward_url 可达
        if recorder is not None and storage is not None:
            raise ValueError("pass recorder or storage, not both")
        self._search_recorder = recorder or SearchRecorder(
            storage=storage or build_storage(config), config=config,
        )
        self._owns_recorder = recorder is None
        # 幂等集有界: 超出容量淘汰最旧 request_id, 防长跑内存泄漏(BUG-002)
        self._emitted: "OrderedDict[str, None]" = OrderedDict()
        self._dedup_capacity = max(1, config.dedup_capacity)
        self._lock = threading.Lock()
        self._last_error = ""

    # ---------- 主链路唯一入口 ----------
    def emit(self, ctx: RequestContext) -> None:
        try:
            with self._lock:
                if ctx.request_id in self._emitted:
                    return
                self._emitted[ctx.request_id] = None
                if len(self._emitted) > self._dedup_capacity:
                    self._emitted.popitem(last=False)  # 淘汰最旧
            self._search_recorder.record(self._build_search_event(ctx))
        except Exception as exc:  # fail-silent
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("TelemetryBus emit 失败(问答结果不受影响): %s", exc, exc_info=True)

    def health(self) -> dict[str, Any]:
        metrics = self._search_recorder.metrics.snapshot()
        return {"healthy": not self._last_error and int(metrics.get("failed", 0)) == 0,
                "last_error": self._last_error, "metrics": metrics}

    # ---------- 事件组装(只问题侧) ----------
    @staticmethod
    def _build_search_event(ctx: RequestContext) -> SearchLogEvent:
        execution = dict(ctx.execution or {})
        return SearchLogEvent(
            query=ctx.query_raw,
            query_cleaned=ctx.query_cleaned or ctx.query_raw,
            query_primary=ctx.query_primary or ctx.query_raw,
            query_subs=ctx.query_subs,
            query_rewritten=ctx.query_rewritten or ctx.query_primary or ctx.query_raw,
            session_id=ctx.session_id,
            node_id=ctx.node_id,
            request_id=ctx.request_id,
            user_id=ctx.user_id or None,
            channel="agent",
            ts=now_iso(),  # 事件产生时间打点, 队列积压不影响 ts(BUG-009)
            snapshot_digest=str(execution.get("snapshot_digest", "")),
            request_ir_digest=str(execution.get("request_ir_digest", "")),
            logical_plan_digest=str(execution.get("logical_plan_digest", "")),
            literature_query_digest=str(execution.get("literature_query_digest", "")),
            execution_snapshot_digest=str(execution.get("execution_snapshot_digest", "")),
            admission=str(execution.get("admission", "")),
            skill_bindings=list(execution.get("skill_bindings") or []),
            executed_channels=list(execution.get("executed_channels") or []),
            result_type=str(execution.get("result_type", "")),
            result_count=int(execution.get("result_count", 0)),
            latency_ms=int(execution.get("latency_ms", 0)),
            citation_valid=execution.get("citation_valid"),
            error_type=str(execution.get("error_type", "")),
        )

    # ---------- 生命周期 ----------
    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._owns_recorder:
            self._search_recorder.shutdown(timeout_s=timeout_s)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"
