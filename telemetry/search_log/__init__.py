# search_log - 检索词记录功能（最终方案：旁路拦截器 + 有界队列异步批量落库）
# 零第三方依赖(仅标准库), 可直接复制进现有服务
from telemetry.search_log.config import SearchLogConfig
from telemetry.search_log.metrics import SearchLogMetrics
from telemetry.search_log.models import SearchLogEvent, SearchLogRecord
from telemetry.search_log.recorder import SearchRecorder
from telemetry.search_log.storage import (
    ForwardingStorage,
    JsonFileStorage,
    LogStorage,
    SqliteStorage,
)

__all__ = [
    "SearchLogConfig",
    "SearchLogMetrics",
    "SearchLogEvent",
    "SearchLogRecord",
    "SearchRecorder",
    "LogStorage",
    "JsonFileStorage",
    "ForwardingStorage",
    "SqliteStorage",
]
