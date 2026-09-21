"""检索词记录功能 - 核心记录器（对应方案文档 §4/§6）

旁路拦截器: 主链路只调 record(event) 一次
  - 有界队列(背压): 满则丢弃并计数, 绝不阻塞/抛给搜索线程
  - 异步批量 flush: 攒 batch_size 条或 flush_interval 秒
  - fail-silent: 存储失败仅计数, 不回传错误、不重试阻塞
  - 开关 + 采样 + 压力自动降采样
  - 自监控计数(失败/丢弃/积压/慢写)
"""
from __future__ import annotations

import itertools
import logging
import queue
import random
import threading
import time
from typing import Any

from telemetry.search_log.config import SearchLogConfig
from telemetry.search_log.metrics import SearchLogMetrics
from telemetry.search_log.models import SearchLogEvent, SearchLogRecord, now_iso
from telemetry.search_log.storage import LogStorage, build_storage
from telemetry.search_log.utils import hash_user, mask_ip, normalize_query

logger = logging.getLogger(__name__)


class SearchRecorder:
    def __init__(
        self,
        storage: LogStorage,
        config: SearchLogConfig | None = None,
        metrics: SearchLogMetrics | None = None,
        auto_start: bool = True,
    ) -> None:
        self.config = config or SearchLogConfig()
        self.storage = storage
        self.metrics = metrics or SearchLogMetrics()
        self._queue: "queue.Queue[SearchLogEvent | None]" = queue.Queue(
            maxsize=self.config.queue_size
        )
        self._enabled = self.config.enabled
        self._seq = itertools.count(1)
        self._worker = threading.Thread(
            target=self._flush_loop, name="search-log-flusher", daemon=True
        )
        if auto_start:
            self._worker.start()

    @classmethod
    def from_config(cls, config: SearchLogConfig | None = None) -> "SearchRecorder":
        """便捷构造: 按配置选择存储(sqlite/forwarding/json, 见 build_storage)"""
        config = config or SearchLogConfig()
        return cls(storage=build_storage(config), config=config)

    # ---------- 主链路唯一入口(旁路) ----------
    def record(self, event: SearchLogEvent) -> None:
        """搜索主链路调用; 任何异常都不抛给调用方(fail-silent)"""
        if not self._enabled:
            return
        try:
            if not self._should_sample():
                self.metrics.incr("sampled_skipped")
                return
            event.sampled = True  # 标记采样命中(事件为旁路临时对象, 调用方不再复用)
            self._queue.put_nowait(event)  # 满则抛 Full -> 丢弃计数
        except queue.Full:
            self.metrics.incr("dropped")
        except Exception as exc:  # pragma: no cover - 最后一道防线
            logger.warning("检索词记录旁路异常(已忽略): %s", exc)

    # ---------- 采样 ----------
    def _should_sample(self) -> bool:
        rate = self._effective_sample_rate()
        return random.random() < rate

    def _effective_sample_rate(self) -> float:
        """压力自动降采样: 队列水位超阈值时降到 sample_rate*auto_sample_reduce"""
        rate = self.config.sample_rate
        size = self.config.queue_size
        if size > 0:
            water = self._queue.qsize() / size
            if water >= self.config.high_watermark:
                self.metrics.incr("high_water_events")
                rate = rate * self.config.auto_sample_reduce
        return max(0.0, min(1.0, rate))

    # ---------- 后台批量写入 ----------
    def _flush_loop(self) -> None:
        batch: list[SearchLogEvent] = []
        while True:
            try:
                item = self._queue.get(timeout=self.config.flush_interval)
            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []
                continue
            if item is None:  # shutdown 哨兵
                if batch:
                    self._flush(batch)
                return
            batch.append(item)
            if len(batch) >= self.config.batch_size:
                self._flush(batch)
                batch = []

    def _flush(self, events: list[SearchLogEvent]) -> None:
        try:
            records = [self._build_record(event) for event in events]
            start = time.monotonic()
            self.storage.write(records, timeout_s=self.config.write_timeout)
            elapsed = time.monotonic() - start
            self.metrics.incr("written", len(records))
            if elapsed > self.config.write_timeout:
                self.metrics.incr("slow_writes")
                logger.warning("搜索日志写入慢: %.2fs (> %.1fs)", elapsed, self.config.write_timeout)
        except Exception as exc:  # fail-silent: 丢弃并计数, 不重试
            self.metrics.incr("failed", len(events))
            logger.warning("搜索日志写入失败(已丢弃 %d 条): %s", len(events), exc)

    def _build_record(self, event: SearchLogEvent) -> SearchLogRecord:
        raw = (
            str(event.query or "")[: self.config.query_max_len]
            if self.config.persist_query_text else ""
        )
        cleaned = event.query_cleaned or event.query
        primary = event.query_primary or event.query
        rewritten = event.query_rewritten or event.query
        return SearchLogRecord(
            event_id=f"{event.request_id or 'anon'}:{next(self._seq):04d}",
            ts=event.ts or now_iso(),  # 事件产生时间优先; 旧调用方未打点则落库时补齐
            user_id=hash_user(event.user_id, self.config.hash_salt),
            dept_id=event.dept_id,
            query_raw=raw,
            query_norm=(normalize_query(raw, self.config.query_max_len).lower()
                        if self.config.persist_query_text else ""),
            query_cleaned=(normalize_query(cleaned, self.config.query_max_len)
                           if self.config.persist_query_text else ""),
            query_primary=(normalize_query(primary, self.config.query_max_len)
                           if self.config.persist_query_text else ""),
            query_subs=([normalize_query(s, self.config.query_max_len) for s in event.query_subs]
                        if self.config.persist_query_text else []),
            query_rewritten=(normalize_query(rewritten, self.config.query_max_len)
                             if self.config.persist_query_text else ""),
            channel=event.channel,
            request_id=event.request_id,
            sampled=event.sampled,
            ip=mask_ip(event.client_ip),
            session_id=event.session_id,
            node_id=event.node_id or self.config.node_id,
            execution={
                "snapshot_digest": event.snapshot_digest,
                "request_ir_digest": event.request_ir_digest,
                "logical_plan_digest": event.logical_plan_digest,
                "literature_query_digest": event.literature_query_digest,
                "execution_snapshot_digest": event.execution_snapshot_digest,
                "admission": event.admission,
                "skill_bindings": list(event.skill_bindings),
                "executed_channels": list(event.executed_channels),
                "result_type": event.result_type,
                "result_count": int(event.result_count),
                "latency_ms": int(event.latency_ms),
                "citation_valid": event.citation_valid,
                "error_type": event.error_type,
            },
        )

    # ---------- 生命周期 ----------
    def shutdown(self, timeout_s: float = 5.0) -> None:
        """优雅关闭: 置开关 → 同步强刷队列剩余(不依赖 worker 是否来得及) → 哨兵 → 等 worker

        超时后仍残留的数据计数 dropped(不再静默丢弃); storage 照常关闭。
        """
        self._enabled = False
        # 1) 主线程直接 drain 队列剩余并落库, 不依赖 worker 线程调度
        remaining: list[SearchLogEvent] = []
        try:
            while True:
                remaining.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if remaining:
            self._flush(remaining)
        # 2) 投递哨兵让 worker 退出(队列满则等其超时退出)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=timeout_s)
        # 3) 超时未退出: 剩余数据随 daemon 线程丢弃, 必须计数
        if self._worker.is_alive():
            left = self._queue.qsize()
            if left:
                self.metrics.incr("dropped", left)
                logger.warning("shutdown 超时, %d 条检索词记录随 worker 丢弃", left)
        if hasattr(self.storage, "close"):
            self.storage.close()  # type: ignore[attr-defined]

    def snapshot(self) -> dict[str, Any]:
        """自监控快照: 计数 + 队列水位"""
        snap = self.metrics.summary()
        snap["queue_size"] = self._queue.qsize()
        snap["queue_max"] = self.config.queue_size
        snap["enabled"] = self._enabled
        snap["effective_sample_rate"] = round(self._effective_sample_rate(), 4)
        return snap
