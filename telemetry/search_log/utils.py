"""检索词记录功能 - 规范化/脱敏工具（对应方案文档 §5/§9）"""
from __future__ import annotations

import hashlib
import hmac
import re


def normalize_query(query: str, max_len: int = 200) -> str:
    """入库前规范化: trim + 全角转半角 + 压缩空白 + 截断超长

    全角转半角(数字/字母/常用符号), 中文不受影响。
    非字符串输入兜底: None/空 → "", 其他类型 str() 转换,
    防止单个非字符串元素(如 query_subs=[None, 123])拖垮整批写入。
    """
    if not isinstance(query, str):
        query = str(query) if query is not None else ""
    if not query:
        return ""
    text = _fullwidth_to_halfwidth(query.strip())
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def _fullwidth_to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:  # 全角符号/字母/数字
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def hash_user(user_id: str | None, salt: str) -> str | None:
    """用户标识去标识化: HMAC-SHA256 + 盐(生产必须配盐; 不用 MD5)

    未登录/无 user_id 时返回 None(由 traceId + 脱敏 IP 兜底可追踪)。
    """
    if not user_id:
        return None
    if not salt:
        # 无盐时退化为明文(仅限非生产; 合规要求下必须配盐)
        return user_id
    digest = hmac.new(salt.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def mask_ip(ip: str | None) -> str | None:
    """IP 脱敏: 截断到网段(不存完整 IP)"""
    if not ip:
        return None
    parts = ip.strip().split(".")
    if len(parts) == 4:
        return ".".join(parts[:3]) + ".*"
    return "***"
