# memory_cleaning.py - 记忆文本清洗与校验（L1 代码层 + L3 保真校验）
# 对应《Agent 记忆管理设计 v2.0》§2.2.2 文本清洗规范
# 纯函数、无状态、可单测; 确定性规则优先, 语义规则留给 LLM(L2)
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------- 词表 ----------
COLLOQUIAL_MAP = {
    "俺": "我", "咱": "我们", "咋": "怎么", "啥": "什么", "整": "弄",
    "行吧": "好", "中不中": "可以吗", "妥了": "好了",
}
FILLER_WORDS = ("嗯嗯", "嗯", "哦哦", "哦", "啊啊", "哎", "哎呀", "哈", "呃", "emmm", "emm")
EMPHASIS_REPEAT = {"对": "对", "好": "好", "是": "是", "行": "行"}
NEGATION_WORDS = (
    "不", "没", "别", "无", "非", "未", "莫", "勿",
    "难以", "无法", "避免", "拒绝", "不要", "不用", "不是", "不能", "不想",
)
SENTENCE_END = "。！？；!?;"
PRONOUN_USER = ("我", "俺", "咱")
PRONOUN_AGENT = ("你", "您")

URL_RE = re.compile(r"https?://[^\s，。；！？]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PATH_RE = re.compile(r"(?:[A-Za-z]:)?[\\/][\w\-.\\/ ]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
PHONE_MASKED_RE = re.compile(r"1[3-9]\d(?:\*{4}|\d{4})\d{4}")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
BANKCARD_RE = re.compile(r"(?<!\d)(?:62|60)\d{14,18}(?!\d)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff\u0000-\u001f\u007f-\u009f]")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F1E6-\U0001F1FF]"
)
WS_RE = re.compile(r"\s+")


# ---------- S1 字符规范化 ----------
def _fullwidth_to_halfwidth(text: str) -> str:
    """只转全角数字/字母和全角空格, 保留中文标点(，。；：！？等)"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF10 <= code <= 0xFF19:  # 全角数字 0-9
            out.append(chr(code - 0xFEE0))
        elif 0xFF21 <= code <= 0xFF3A:  # 全角大写 A-Z
            out.append(chr(code - 0xFEE0))
        elif 0xFF41 <= code <= 0xFF5A:  # 全角小写 a-z
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def s1_char_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _fullwidth_to_halfwidth(text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = EMOJI_RE.sub("[emoji]", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


# ---------- S2 格式归一 ----------
def s2_format_normalize(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PATH_RE.sub("[PATH]", text)
    return text


# ---------- S3 口语归一 ----------
def s3_colloquial_normalize(text: str) -> str:
    for src, dst in COLLOQUIAL_MAP.items():
        text = text.replace(src, dst)
    # 句首填充词删除
    for filler in FILLER_WORDS:
        if text.startswith(filler):
            text = text[len(filler):].lstrip()
            break
    # 重复强调折叠: "对对对"->"对", "好好好"->"好"
    for word in EMPHASIS_REPEAT:
        pattern = re.compile(rf"({re.escape(word)}){{2,}}")
        text = pattern.sub(word, text)
    return text


# ---------- S4 否定保护(检测+返回作用域, 供 L2/verify 使用) ----------
def s4_negation_spans(text: str) -> list[tuple[int, int]]:
    """返回否定词作用域(到句读为止)的字符区间列表"""
    spans: list[tuple[int, int]] = []
    for match in re.finditer("|".join(sorted(NEGATION_WORDS, key=len, reverse=True)), text):
        start = match.start()
        end = start
        while end < len(text) and text[end] not in SENTENCE_END:
            end += 1
        spans.append((start, end))
    return spans


# ---------- S5 人称预替换 ----------
def s5_pronoun_replace(text: str) -> str:
    for word in PRONOUN_USER:
        text = text.replace(word, "用户")
    for word in PRONOUN_AGENT:
        text = text.replace(word, "Agent")
    return text


# ---------- S6 敏感脱敏 ----------
def s6_pii_mask(text: str) -> str:
    text = PHONE_RE.sub("[PII]", text)
    text = PHONE_MASKED_RE.sub("[PII]", text)
    text = IDCARD_RE.sub("[PII]", text)
    text = BANKCARD_RE.sub("[PII]", text)
    return text


# ---------- 主清洗管线 ----------
@dataclass
class CleanResult:
    text: str
    negation_spans: list[tuple[int, int]] = field(default_factory=list)


def clean_text(text: str) -> CleanResult:
    """L1 代码层清洗六步(S1-S6)"""
    if not text:
        return CleanResult(text="")
    text = s1_char_normalize(text)
    text = s2_format_normalize(text)
    text = s3_colloquial_normalize(text)
    spans = s4_negation_spans(text)
    text = s5_pronoun_replace(text)
    text = s6_pii_mask(text)
    return CleanResult(text=text, negation_spans=spans)


# ---------- L3 保真校验(V5) ----------
def verify_fidelity(original: str, cleaned: str) -> tuple[bool, list[str]]:
    """断言: 原文中的数字/否定词/占位标记必须存在于清洗后文本
    注意: 被脱敏(PII/URL 等)有意移除的数字不检查——脱敏优先级高于保真
    """
    missing: list[str] = []

    # 被脱敏模式覆盖的数字(允许丢失)
    protected_numbers: set[str] = set()
    for pattern in (PHONE_RE, PHONE_MASKED_RE, IDCARD_RE, BANKCARD_RE, URL_RE, EMAIL_RE):
        for match in pattern.finditer(original):
            protected_numbers.update(NUMBER_RE.findall(match.group(0)))

    # 数字
    orig_numbers = set(NUMBER_RE.findall(original))
    clean_numbers = set(NUMBER_RE.findall(cleaned))
    for number in orig_numbers:
        if number in protected_numbers:
            continue
        if number not in clean_numbers:
            missing.append(f"数字[{number}]")

    # 否定词(只查原文含否定词的)
    for word in NEGATION_WORDS:
        if word in original and word not in cleaned:
            missing.append(f"否定词[{word}]")

    # 占位标记
    for marker in ("[URL]", "[EMAIL]", "[PATH]", "[PII]"):
        if marker in original and marker not in cleaned:
            missing.append(marker)

    return (len(missing) == 0), missing


# ---------- D1 字面去重 ----------
def literal_dedup(facts: list[dict], existing: list[dict]) -> list[dict]:
    """规范化后字符串相等即重复: 跳过新值, 保留旧值(不更新)"""
    existing_keys = {f.get("content", "").strip() for f in existing}
    result = []
    for fact in facts:
        key = fact.get("content", "").strip()
        if not key:
            continue
        if key in existing_keys:
            continue
        result.append(fact)
    return result


# ---------- C1 锚词冲突标记 ----------
CONFLICT_ANCHORS = ("改为", "不要", "其实", "换成", "不是", "改成", "换")


def anchor_conflict(facts: list[dict], existing: list[dict]) -> list[dict]:
    """锚词触发的确定性冲突标记: 新事实含锚词且与旧事实同 type 且语义相近(共现实体)"""
    for fact in facts:
        content = fact.get("content", "")
        if not any(anchor in content for anchor in CONFLICT_ANCHORS):
            continue
        for old in existing:
            if old.get("type") != fact.get("type"):
                continue
            # 共现实体启发: 提取两边共同的中文名词片段(≥2字)判断是否同主题
            shared = _shared_topic(content, old.get("content", ""))
            if shared:
                fact["conflict_with"] = old.get("id")
                break
    return facts


def _shared_topic(a: str, b: str) -> str | None:
    """取两边共现的 2-4 字中文片段作为主题启发"""
    tokens_a = {a[i:i + n] for n in (2, 3, 4) for i in range(len(a) - n + 1)}
    for n in (4, 3, 2):
        for i in range(len(b) - n + 1):
            seg = b[i:i + n]
            if seg in tokens_a and all("\u4e00" <= c <= "\u9fff" for c in seg):
                return seg
    return None
