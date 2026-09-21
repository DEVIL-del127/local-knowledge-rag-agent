"""检索词记录功能 - 数据模型(问题侧 only)
只记用户问题: 原始/清洗/拆分/重写 + 关联字段; 不记意图/结果/耗时/回答
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchLogEvent:
    """主链路旁路构造的原始事件(问题侧)"""

    query: str  # 原始问题
    query_cleaned: str = ""        # 清洗后
    query_primary: str = ""        # 拆分主问题
    query_subs: list[str] = field(default_factory=list)  # 附加问题
    query_rewritten: str = ""      # 实际进检索的词(指代重写后)
    request_id: str = ""
    user_id: str | None = None
    dept_id: str | None = None
    channel: str | None = None
    client_ip: str | None = None
    session_id: str = ""
    node_id: str = ""
    ts: str = ""  # 事件产生时间(ISO8601, 由组装层打点); 空则由 recorder 落库时补齐
    sampled: bool = False  # 是否采样样本(由 recorder 采样命中后置位)
    snapshot_digest: str = ""
    request_ir_digest: str = ""
    logical_plan_digest: str = ""
    literature_query_digest: str = ""
    execution_snapshot_digest: str = ""
    admission: str = ""
    skill_bindings: list[str] = field(default_factory=list)
    executed_channels: list[str] = field(default_factory=list)
    result_type: str = ""
    result_count: int = 0
    latency_ms: int = 0
    citation_valid: bool | None = None
    error_type: str = ""


@dataclass
class SearchLogRecord:
    """落库记录(问题侧; 时间/脱敏/规范化由 recorder 在后台完成)"""

    event_id: str
    ts: str  # ISO8601 带时区
    user_id: str | None  # HMAC 哈希后
    dept_id: str | None
    query_raw: str  # 原始问题(只截断)
    query_norm: str  # 规范化 + 小写(便于检索)
    query_cleaned: str = ""
    query_primary: str = ""
    query_subs: list[str] = field(default_factory=list)
    query_rewritten: str = ""
    channel: str | None = None
    request_id: str = ""
    sampled: bool = False
    ip: str | None = None
    session_id: str = ""
    node_id: str = ""
    execution: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        import json

        return json.dumps(asdict(self), ensure_ascii=False)


def now_iso() -> str:
    """带时区的 ISO8601 时间戳(统一时区, 便于跨环境分析)"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
