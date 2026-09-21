from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    category: str
    score: float
    margin: float
    auto_release: bool
    reason_code: str
    latency_ms: float


class EncodingUnavailable(RuntimeError):
    pass


class EncodingStatus(str, Enum):
    IDLE = "idle"
    INFLIGHT = "inflight"
    TIMEOUT_PENDING = "timeout_pending"
    FINISHED = "finished"
    CANCEL_CONFIRMED = "cancel_confirmed"


class SemanticMatrixRouter:
    """Immutable matrix lookup with one query encoding and fail-closed timeout state."""

    def __init__(self, package_path: str | Path, encoder: Callable[[str], Sequence[float]],
                 *, timeout_seconds: float = 2.0, circuit_seconds: float = 30.0) -> None:
        payload = json.loads(Path(package_path).read_text(encoding="utf-8"))
        self.package = payload
        self.encoder = encoder
        self.timeout_seconds = timeout_seconds
        self.circuit_seconds = circuit_seconds
        self.matrix = np.load(Path(package_path).with_name(payload["matrix_file"]), allow_pickle=False)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(payload["examples"]):
            raise ValueError("semantic matrix shape does not match package examples")
        norms = np.linalg.norm(self.matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("semantic matrix rows must be normalized")
        digest = hashlib.sha256(self.matrix.astype(np.float32).tobytes()).hexdigest()
        if digest != payload["matrix_digest"]:
            raise ValueError("semantic matrix digest mismatch")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="routing-embed")
        self._lock = threading.Lock()
        self._inflight = False
        self._timeout_pending = False
        self._consecutive_timeouts = 0
        self._opened_at = 0.0
        self._timed_out: set[int] = set()
        self._status = EncodingStatus.IDLE
        self._late_completions = 0
        self._last_completion_late = False
        self._probe_inflight = False

    def route(self, query: str) -> SemanticCandidate:
        started = time.perf_counter()
        with self._lock:
            if self._timeout_pending or self._inflight:
                raise EncodingUnavailable("encoding_slot_unavailable")
            circuit_elapsed = time.monotonic() - self._opened_at
            if self._consecutive_timeouts >= 3 and circuit_elapsed < self.circuit_seconds:
                raise EncodingUnavailable("encoding_circuit_open")
            self._probe_inflight = self._consecutive_timeouts >= 3
            self._inflight = True
            self._status = EncodingStatus.INFLIGHT
        future = self._executor.submit(self.encoder, query)
        future.add_done_callback(self._finished)
        try:
            vector = np.asarray(future.result(timeout=self.timeout_seconds), dtype=np.float32)
        except TimeoutError as exc:
            with self._lock:
                callback_already_finished = not self._inflight
                if callback_already_finished:
                    self._late_completions += 1
                    self._last_completion_late = True
                    self._status = EncodingStatus.FINISHED
                else:
                    self._timed_out.add(id(future))
                    self._timeout_pending = True
                    self._status = EncodingStatus.TIMEOUT_PENDING
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= 3:
                    self._opened_at = time.monotonic()
            future.cancel()
            raise EncodingUnavailable("encoding_timeout") from exc
        vector /= np.linalg.norm(vector) + 1e-12
        scores = self.matrix @ vector
        grouped: dict[str, dict[str, float]] = {}
        for example, score in zip(self.package["examples"], scores.tolist()):
            clusters = grouped.setdefault(example["category"], {})
            cluster_id = str(example["cluster_id"])
            clusters[cluster_id] = max(clusters.get(cluster_id, -1.0), float(score))
        ranked = sorted(((category, float(np.mean(sorted(clusters.values(), reverse=True)[:3])), len(clusters))
                         for category, clusters in grouped.items()), key=lambda item: item[1], reverse=True)
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else ("", -1.0, 0)
        margin = top[1] - second[1]
        policy = self.package["categories"].get(top[0], {})
        enabled = bool(policy.get("enabled", False))
        release = enabled and top[2] >= 3 and top[1] >= float(policy.get("threshold", 1.0)) \
            and margin >= float(policy.get("margin", 1.0))
        reason = "released" if release else ("insufficient_clusters" if top[2] < 3 else
                 "category_disabled" if not enabled else "threshold_not_met")
        return SemanticCandidate(top[0], top[1], margin, release, reason,
                                 (time.perf_counter() - started) * 1000.0)

    def _finished(self, future) -> None:
        with self._lock:
            timed_out = id(future) in self._timed_out
            was_probe = self._probe_inflight
            self._timed_out.discard(id(future))
            self._inflight = False
            self._timeout_pending = False
            self._probe_inflight = False
            self._last_completion_late = timed_out
            self._status = EncodingStatus.CANCEL_CONFIRMED if future.cancelled() else EncodingStatus.FINISHED
            if timed_out:
                self._late_completions += 1
            succeeded = not timed_out and not future.cancelled() and future.exception() is None
            if succeeded:
                self._consecutive_timeouts = 0
                self._opened_at = 0.0
            elif was_probe:
                self._opened_at = time.monotonic()

    def confirm_service_recovery(self) -> None:
        """Clear a tripped circuit only after an external health/restart confirmation."""
        with self._lock:
            if self._inflight or self._timeout_pending:
                raise EncodingUnavailable("encoding_recovery_pending")
            self._consecutive_timeouts = 0
            self._opened_at = 0.0
            self._status = EncodingStatus.IDLE

    def state(self) -> dict[str, object]:
        with self._lock:
            elapsed = time.monotonic() - self._opened_at
            return {"status": self._status.value,
                    "inflight": self._inflight, "timeout_pending": self._timeout_pending,
                    "consecutive_timeouts": self._consecutive_timeouts,
                    "circuit_open": self._consecutive_timeouts >= 3 and elapsed < self.circuit_seconds,
                    "probe_ready": (self._consecutive_timeouts >= 3
                                    and elapsed >= self.circuit_seconds
                                    and not self._probe_inflight),
                    "probe_inflight": self._probe_inflight,
                    "late_completions": self._late_completions,
                    "last_completion_late": self._last_completion_late}
