from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone as datetime_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.literature_ir import BoundType, TemporalConstraint

_YEAR = r"((?:19|20)\d{2})"


def parse_temporal_constraint(
    text: str, *, now: datetime | None = None, timezone: str = "Asia/Shanghai",
) -> TemporalConstraint | None:
    raw = str(text)
    try:
        tzinfo = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        if timezone != "Asia/Shanghai":
            raise
        tzinfo = datetime_timezone(timedelta(hours=8), name="Asia/Shanghai")
    current = now or datetime.now(tzinfo)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tzinfo)
    stamp = current.isoformat()

    match = re.search(_YEAR + r"\s*年?\s*(?:到|至|[-–—~～]|through|to)\s*" + _YEAR + r"\s*年?", raw, re.I)
    if match:
        lower, upper = sorted((int(match.group(1)), int(match.group(2))))
        return TemporalConstraint(lower, BoundType.CLOSED, upper, BoundType.CLOSED)
    match = re.search(_YEAR + r"\s*年?\s*(?:之后|以后|后|later than|after)", raw, re.I)
    if match:
        return TemporalConstraint(int(match.group(1)), BoundType.OPEN)
    match = re.search(_YEAR + r"\s*年?\s*(?:以来|起|及以后|since)", raw, re.I)
    if match:
        return TemporalConstraint(int(match.group(1)), BoundType.CLOSED)
    match = re.search(r"(?:截至|截止到?|不晚于|through)\s*" + _YEAR + r"\s*年?", raw, re.I)
    if match:
        return TemporalConstraint(None, None, int(match.group(1)), BoundType.CLOSED)
    match = re.search(_YEAR + r"\s*年?\s*(?:以前|之前|前|before)", raw, re.I)
    if match:
        return TemporalConstraint(None, None, int(match.group(1)), BoundType.OPEN)
    match = re.search(r"(?:近|最近)\s*([一二两三四五六七八九十\d]+)\s*年", raw)
    if match:
        count = _number(match.group(1))
        if count > 0:
            return TemporalConstraint(
                current.year - count + 1, BoundType.CLOSED, current.year, BoundType.CLOSED,
                relative_expression=match.group(0), resolved_at=stamp, timezone=timezone,
            )
    match = re.search(_YEAR + r"\s*年", raw)
    if match:
        year = int(match.group(1))
        return TemporalConstraint(year, BoundType.CLOSED, year, BoundType.CLOSED)
    return None


def _number(value: str) -> int:
    if value.isdigit():
        return int(value)
    values = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return values.get(value, 0)
