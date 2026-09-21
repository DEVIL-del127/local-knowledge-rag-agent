"""Deterministic, explicit memory commands; never infer permission from a fact."""
from __future__ import annotations

import re


def parse_memory_command(text: str) -> tuple[str, dict]:
    value = text.strip()
    save = re.fullmatch(r"(?:请)?(?:帮我)?(?:记住|保存记忆)[：:\s]*(.+)", value, re.S)
    if save:
        content = save.group(1).strip()
        if not content or len(content) > 500:
            raise ValueError("请提供 1–500 字的具体记忆内容。")
        return "save", {"value": content, "explicit_consent": True}
    if re.fullmatch(r"(?:请)?(?:清空|清除)(?:我的|全部|所有)?记忆[。！!]?", value):
        return "clear", {}
    delete = re.fullmatch(r"(?:请)?(?:删除|删掉|忘掉|忘记)(?:记忆)?\s*([A-Za-z0-9_-]+)[。！!]?", value)
    if delete:
        return "delete", {"id": delete.group(1)}
    fix = re.fullmatch(r"(?:请)?修正(?:记忆)?\s*([A-Za-z0-9_-]+)\s*(?:为|：|:)\s*(.+)", value, re.S)
    if fix:
        return "fix", {"id": fix.group(1), "value": fix.group(2).strip()}
    if any(token in value for token in ("删除", "删掉", "忘掉", "忘记", "清空", "清除", "修正")):
        raise ValueError("请明确指定操作，例如“删除记忆 完整ID”“修正记忆 完整ID：新内容”或“清空我的记忆”。")
    return ("stats", {}) if "统计" in value or "多少" in value else ("list", {})
