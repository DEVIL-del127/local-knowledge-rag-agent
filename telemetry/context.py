# context.py - 请求上下文(主链路与日志的唯一契约)
# 只记录问题侧: 原始/清洗/拆分/重写 + 关联字段(意图/结果/耗时等不采集)
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestContext:
    """一次用户提问的问题上下文(只含问题侧字段)"""

    # 关联字段
    request_id: str
    user_id: str
    session_id: str = ""
    node_id: str = "node-a"

    # 问题侧
    query_raw: str = ""            # 用户原始输入
    query_cleaned: str = ""        # 清洗后
    query_primary: str = ""        # 拆分主问题
    query_subs: list[str] = field(default_factory=list)  # 附加问题
    query_rewritten: str = ""      # 实际进检索的词(指代重写后, 首个检索路)
    execution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
