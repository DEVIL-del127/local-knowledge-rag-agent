"""检索词记录功能 - 自监控计数（对应方案文档 §8 四指标）

指标:
    written            成功写入条数
    failed             写入失败条数(弃写, fail-silent)
    dropped            队列满丢弃条数
    sampled_skipped    采样跳过条数
    slow_writes        软超时写入次数
    high_water_events  触发自动降采样次数

接入 Prometheus/Micrometer 时, 把这些计数暴露为 gauge/counter 即可。
"""
from __future__ import annotations

import threading
from typing import Any


class SearchLogMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters = {
            "written": 0,
            "failed": 0,
            "dropped": 0,
            "sampled_skipped": 0,
            "slow_writes": 0,
            "high_water_events": 0,
        }

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def summary(self) -> dict[str, Any]:
        snap = self.snapshot()
        total = snap["written"] + snap["failed"] + snap["dropped"] + snap["sampled_skipped"]
        snap["failure_rate"] = round(snap["failed"] / total, 4) if total else 0.0
        snap["drop_rate"] = round(snap["dropped"] / total, 4) if total else 0.0
        return snap
