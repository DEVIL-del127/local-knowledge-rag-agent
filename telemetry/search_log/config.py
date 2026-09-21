"""检索词记录功能 - 配置（对应方案文档 §6 配置结构）"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SearchLogConfig:
    # 功能开关: 一键关闭后 record() 直接返回, 主链路零开销
    enabled: bool = True
    # 采样率 0.0~1.0 (灰度: 0.1 → 0.5 → 1.0)
    sample_rate: float = 1.0
    # 有界队列容量(背压: 满则丢弃并计数, 绝不阻塞搜索线程)
    queue_size: int = 10_000
    # 批量攒批: 条数或时间(秒)先到先刷
    batch_size: int = 100
    flush_interval: float = 2.0
    # 写入软超时(秒): 超时仅计 slow 指标, 不中断写入(避免半写状态)
    write_timeout: float = 2.0
    # 超长检索词截断
    query_max_len: int = 200
    # Production defaults to operational digests only; tests/dev may opt in.
    persist_query_text: bool = True
    # 用户标识 HMAC 盐(生产必配; 不配则明文, 合规要求下必须配)
    hash_salt: str = ""
    # 存储: json(默认, JSON lines 按天轮转) | sqlite(零依赖 DB 演示)
    storage: str = "json"
    # json 存储目录
    json_dir: str = "logs/search_log"
    # sqlite 存储文件
    sqlite_path: str = "logs/search_log.sqlite3"
    # 保留天数: 过期文件自动清理(0=不清理)
    retention_days: int = 90
    # 节点标识(配置下发, 非 hostname; 多服务器归并用)
    node_id: str = "node-a"
    # 转发到中心(留空=不转发)
    forward_url: str = ""
    forward_token: str = ""
    forward_interval_s: float = 3600.0
    # 转发游标文件
    forward_cursor_file: str = "logs/search_log/.forward_cursor.json"
    # 幂等集容量: TelemetryBus 去重集合上限, 超出淘汰最旧 request_id(防长跑内存泄漏)
    dedup_capacity: int = 100_000
    # 压力自动降采样: 队列水位超过 high_watermark 时, 有效采样率降为 sample_rate*auto_sample_reduce
    auto_sample_reduce: float = 0.1
    high_watermark: float = 0.8

    @classmethod
    def from_env(cls) -> "SearchLogConfig":
        """从环境变量加载(前缀 SEARCH_LOG_), 便于容器化/多环境配置"""
        def _bool(v: str | None, default: bool) -> bool:
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "on"}

        def _float(v: str | None, default: float) -> float:
            try:
                return float(v) if v is not None else default
            except ValueError:
                return default

        def _int(v: str | None, default: int) -> int:
            try:
                return int(v) if v is not None else default
            except ValueError:
                return default

        return cls(
            enabled=_bool(os.environ.get("SEARCH_LOG_ENABLED"), True),
            sample_rate=_float(os.environ.get("SEARCH_LOG_SAMPLE_RATE"), 1.0),
            queue_size=_int(os.environ.get("SEARCH_LOG_QUEUE_SIZE"), 10_000),
            batch_size=_int(os.environ.get("SEARCH_LOG_BATCH_SIZE"), 100),
            flush_interval=_float(os.environ.get("SEARCH_LOG_FLUSH_INTERVAL"), 2.0),
            write_timeout=_float(os.environ.get("SEARCH_LOG_WRITE_TIMEOUT"), 2.0),
            query_max_len=_int(os.environ.get("SEARCH_LOG_QUERY_MAX_LEN"), 200),
            persist_query_text=_bool(os.environ.get("SEARCH_LOG_PERSIST_QUERY_TEXT"), False),
            hash_salt=os.environ.get("SEARCH_LOG_HASH_SALT", ""),
            storage=os.environ.get("SEARCH_LOG_STORAGE", "json"),
            json_dir=os.environ.get("SEARCH_LOG_JSON_DIR", "logs/search_log"),
            sqlite_path=os.environ.get("SEARCH_LOG_SQLITE_PATH", "logs/search_log.sqlite3"),
            auto_sample_reduce=_float(os.environ.get("SEARCH_LOG_AUTO_SAMPLE_REDUCE"), 0.1),
            high_watermark=_float(os.environ.get("SEARCH_LOG_HIGH_WATERMARK"), 0.8),
            retention_days=_int(os.environ.get("SEARCH_LOG_RETENTION_DAYS"), 30),
            node_id=os.environ.get("NODE_ID", "node-a"),
            forward_url=os.environ.get("SEARCH_LOG_FORWARD_URL", ""),
            forward_token=os.environ.get("SEARCH_LOG_FORWARD_TOKEN", ""),
            forward_interval_s=_float(os.environ.get("SEARCH_LOG_FORWARD_INTERVAL_S"), 3600.0),
            forward_cursor_file=os.environ.get("SEARCH_LOG_FORWARD_CURSOR", "logs/search_log/.forward_cursor.json"),
            dedup_capacity=_int(os.environ.get("SEARCH_LOG_DEDUP_CAPACITY"), 100_000),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchLogConfig":
        """从 dict 加载(便于接入配置中心)"""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**kwargs)
