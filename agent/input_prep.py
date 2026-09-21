# input_prep.py - 用户输入预处理: 清洗 + 复合问题拆分
# 位置: 入口 _chat_inner, 缓存检查之后、意图识别之前
# 纯函数, 可单测
from __future__ import annotations

import re

# ---------- 清洗 ----------
_FULLWIDTH_ALNUM = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)
_FULLWIDTH_PUNCT = str.maketrans(
    "，。！？：；（）【】“”‘’～",
    ",.!?:;()[]\"\"''~",
)

_CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def _fullwidth_to_halfwidth(text: str) -> str:
    text = text.translate(_FULLWIDTH_ALNUM)
    text = text.translate(_FULLWIDTH_PUNCT)
    return text


def clean_user_input(text: str) -> str:
    """用户输入清洗:
    1. 去非法替换字符 U+FFFD / 控制字符 / 零宽字符
    2. 全角数字字母标点转半角
    3. 压缩重复标点(!!!→!  ???→?)
    4. 空白整理(多空格/换行→单空格, 去首尾)
    """
    if not text:
        return ""
    s = text
    s = s.replace("\ufffd", "")  # 输入法/终端坏字节
    s = _CONTROL_RE.sub("", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _fullwidth_to_halfwidth(s)
    s = re.sub(r"[!！]{2,}", "！", s)
    s = re.sub(r"[?？]{2,}", "？", s)
    s = re.sub(r"[.。]{2,}", "。", s)
    s = re.sub(r"[,，]{2,}", "，", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------- 复合问题拆分 ----------
CONNECTOR_MARKERS = (
    "顺便", "另外", "还有", "同时", "再帮我", "以及", "还有没有",
)


def split_compound_question(text: str) -> tuple[str, list[str]]:
    """复合问题拆分: 返回 (主问题, 附加问题列表)

    规则:
    - 多个问号 → 第一段为主问题, 其余为附加
    - 单个问句含连接词(顺便/另外/还有...) → 连接词前为主问题, 后为附加
    - 无法拆分 → 返回 (原文, [])
    例: "什么是GAN？顺便查一下贝叶斯论文"
        → ("什么是GAN", ["顺便查一下贝叶斯论文"])
    """
    text = (text or "").strip()
    if not text:
        return text, []

    # 多问句: 按问号拆分
    parts = [p.strip() for p in re.split(r"[?？]+", text) if p.strip()]
    if len(parts) >= 2:
        primary = parts[0].rstrip("，,。.;；:： ")
        extras = [p for p in parts[1:] if len(p) >= 2]
        # 主段至少 2 字才拆(防"好？不好？"这类二选一碎问句)
        if len(primary) >= 2 and primary and extras:
            return primary, extras[:2]

    # 单问句含连接词
    for marker in CONNECTOR_MARKERS:
        idx = text.find(marker)
        if idx > 2:
            primary = text[:idx].rstrip("，,。.;；:： ")
            extra = text[idx:].strip()
            if primary and len(extra) >= 2 and len(primary) >= 4:
                return primary, [extra]

    return text, []
