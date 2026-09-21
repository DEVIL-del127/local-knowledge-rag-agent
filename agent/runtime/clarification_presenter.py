from __future__ import annotations

import re
from typing import Any


class LLMClarificationPresenter:
    """Use an LLM only to phrase a deterministic clarification question."""

    _SYSTEM_PROMPT = """你只负责把系统给出的澄清问题改写成简洁、友好的中文问句。
严格要求：
1. 不回答问题，不猜测值，不新增事实、字段、数字、单位或候选项。
2. 保留原问题中的所有数字、单位、字段名和候选 ID。
3. 只输出一段问句，不输出解释、JSON、编号列表或内部术语。
4. 如果无法安全改写，原样输出。
"""

    def __init__(self, client: Any, *, max_chars: int = 500) -> None:
        self.client = client
        self.max_chars = max_chars

    def rewrite(self, deterministic_prompt: str) -> str:
        source = deterministic_prompt.strip()
        if not source:
            return source
        try:
            output = self.client.invoke_text(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=source,
                temperature=0.0,
                max_tokens=160,
            ).strip()
        except Exception:
            return source
        if not output or len(output) > self.max_chars:
            return source
        if _numbers(source) != _numbers(output):
            return source
        if any(token in output for token in ("```", "{", "}", "答案是", "建议值为")):
            return source
        return output


def _numbers(value: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", value)
