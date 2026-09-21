# -*- coding: utf-8 -*-
"""cleaner.py —— 文本清洗层（P0：引导壳剥离 / 填充词 / 全角半角 / 时间语境中文数字）

设计原则：
1. 表驱动：CLEAN_RULES 声明式（名称/处理器），与 TIME_RULES 同构
2. 保守优先：只做语义安全的清洗——剥离壳/噪声/格式统一；
   不做有歧义的改写（如"有没有→检索"会破坏嵌套限定语义，明确不做）
3. 可观测：clean() 返回 (cleaned, steps)，每步记录改了什么（nlu_validate 可显示）
4. 不删语义词："呢/吧/吗"保留（splitter 的疑问/独立判定依赖它们）
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------- 引导壳（句首剥离，按长度降序优先） ----------
SHELLS = [
    "我想看看", "我想了解", "我想知道", "我想查一下", "我需要", "我想要",
    "回头看", "我们来看", "来看一下", "我们来看一下", "帮我看看", "帮我查一下",
    "帮我找一下", "我想", "我要", "请问", "麻烦", "帮我",
]

# ---------- 口语规范化（P1：语义安全映射，不做有歧义改写） ----------
# 注意："有没有→检索" "搞→做" 这类会破坏嵌套限定/引入歧义，明确不做
COLLOQUIAL_MAP = [
    ("咋样", "如何"),
    ("咋", "怎么"),
    ("啥", "什么"),
    ("瞅瞅", "看看"),
    ("瞧瞧", "看看"),
    ("看下", "查询"),
    ("查查", "查询"),
]

# ---------- 填充词（全删） ----------
# 注意："呢/吧"不能删——splitter 疑问/独立判定依赖它们
FILLERS = ["就是说", "那个", "然后", "嗯嗯", "哦哦", "嗯", "哦", "呃", "emmm", "emm",
           "啊", "呀"]

# ---------- 中文数字 ----------
CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn2arab(s: str) -> int:
    """中文数字 → 阿拉伯数字。纯数字位（二零二四→2024）按十进制拼接；带单位（二十四→24）按位权。"""
    if not s:
        return None
    if all(ch in CN_DIGITS for ch in s):
        return int("".join(str(CN_DIGITS[ch]) for ch in s))
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            u = CN_UNITS[ch]
            section += (num or 1) * u
            num = 0
        else:
            return None
    return total + section + num


# ---------- 清洗规则（顺序执行，每步留痕） ----------

def _fullwidth(q: str) -> Tuple[str, str]:
    """全角→半角（字母/数字/常用标点）"""
    out = []
    for ch in q:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out), "全角→半角"


def _fillers(q: str) -> Tuple[str, str]:
    out, n = q, 0
    for f in FILLERS:
        if f in out:
            out = out.replace(f, "")
            n += 1
    return out, f"删除填充词×{n}" if n else ""


def _shells(q: str) -> Tuple[str, str]:
    """引导壳剥离：整句句首 + 问号/句号后的分句句首（"…修订？我需要…"→剥"我需要"）"""
    def strip_shell(s: str) -> str:
        out, n = s, 0
        changed = True
        while changed and out:
            changed = False
            for sh in SHELLS:
                if out.startswith(sh):
                    out = out[len(sh):]
                    n += 1
                    changed = True
                    break
        out = re.sub(r"^[，,。;；、！?！\s]+", "", out)
        return out, n

    parts = re.split(r"([？?。！!])", q)  # 保留分隔符
    total = 0
    out_parts = []
    for i, part in enumerate(parts):
        if part in ("？", "?", "。", "！", "!"):
            out_parts.append(part)
            continue
        s, n = strip_shell(part)
        total += n
        out_parts.append(s)
    out = "".join(out_parts)
    return out, f"剥离引导壳×{total}" if total else ""


def _punct_compress(q: str) -> Tuple[str, str]:
    out = re.sub(r"([，。！？；；!?])\1+", r"\1", q)
    return out, "压缩连续标点" if out != q else ""


def _time_cn_digits(q: str) -> Tuple[str, str]:
    """时间语境中文数字→阿拉伯：X年/X月/X号/X日/X点/X时 前的数字"""
    def repl(m):
        v = cn2arab(m.group(1))
        return f"{v}{m.group(2)}" if v is not None else m.group(0)
    out = re.sub(r"([零一二两三四五六七八九十百]+)(年|月|号|日|点|时)", repl, q)
    return out, "中文数字→阿拉伯" if out != q else ""


def _colloquial(q: str) -> Tuple[str, str]:
    """口语规范化：咋→怎么 啥→什么 瞅瞅→看看（表驱动，按长度降序）"""
    out, n = q, 0
    for a, b in sorted(COLLOQUIAL_MAP, key=lambda x: -len(x[0])):
        if a in out:
            out = out.replace(a, b)
            n += 1
    return out, f"口语规范化×{n}" if n else ""


def _edit_distance(a: str, b: str) -> int:
    """编辑距离（DP，用于实体词典容错匹配）"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


# 实体容错词典（P1：pypinyin + 编辑距离 ≤1，长度≥3 防误伤）
# 只对库里已有实体（作者/期刊/主题词）做纠正，绝不发明新词
_TYPO_SOURCES = None  # 惰性加载：nlu 的 AUTHORS/VENUES/TOPIC_WORDS


def _load_typo_sources():
    global _TYPO_SOURCES
    if _TYPO_SOURCES is None:
        try:
            from nlu import AUTHORS, VENUES, TOPIC_WORDS
            _TYPO_SOURCES = [w for w in AUTHORS + VENUES + TOPIC_WORDS
                             if len(w) >= 3 and re.fullmatch(r"[\u4e00-\u9fff]+", w)]
        except Exception:
            _TYPO_SOURCES = []
    return _TYPO_SOURCES


def _typo_fix(q: str) -> Tuple[str, str]:
    """实体词典容错：同音/近音 + 编辑距离≤1 → 纠正为库内实体（如 张昭招→张昭昭）
    滑窗方案：对每个库内实体词，在原文按同长度滑窗找近似写法，避免整段误配。"""
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return q, ""
    sources = _load_typo_sources()
    if not sources:
        return q, ""
    src_py = {w: "".join(lazy_pinyin(w)) for w in sources}
    out, n = q, 0
    for w, wpy in src_py.items():
        L = len(w)
        i = 0
        while i <= len(out) - L:
            seg = out[i:i + L]
            if seg == w:
                i += L
                continue
            if not re.fullmatch(r"[\u4e00-\u9fff]+", seg):
                i += 1
                continue
            seg_py = "".join(lazy_pinyin(seg))
            # 只纠同音（错别字本质是同音字）；形近但不同音不纠（"回归"≠"回声"）
            if seg_py == wpy:
                out = out[:i] + w + out[i + L:]
                n += 1
                i += L
            else:
                i += 1
    return out, f"错别字纠正×{n}" if n else ""


def _garbage_chars(q: str) -> Tuple[str, str]:
    """乱码清洗：替换符 U+FFFD / 控制字符（除空白）→ 删除（"23�年"→"23年"）"""
    out = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", "", q)
    return out, "删除乱码字符" if out != q else ""


CLEAN_RULES = [
    ("fullwidth", _fullwidth),
    ("garbage_chars", _garbage_chars),
    ("fillers", _fillers),
    ("colloquial", _colloquial),
    ("shells", _shells),
    ("typo_fix", _typo_fix),
    ("punct_compress", _punct_compress),
    ("time_cn_digits", _time_cn_digits),
]


@dataclass
class CleanResult:
    cleaned: str
    steps: List[Tuple[str, str, str]] = field(default_factory=list)  # (规则, 改前, 改后)

    def to_dict(self):
        return {"cleaned": self.cleaned,
                "steps": [{"rule": r, "detail": f"{a} → {b}" if a != b else "-"}
                          for r, a, b in self.steps]}


def clean(q: str) -> CleanResult:
    out = q.strip()
    res = CleanResult(cleaned=out)
    for name, fn in CLEAN_RULES:
        before = out
        try:
            out, detail = fn(out)
        except Exception:
            continue
        if out != before:
            res.steps.append((name, before, out))
    res.cleaned = out
    return res


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "我想看看在小米和理想汽车上市前后的两个季度里的专利布局",
        "回头看，从非典结束到现在，中国在公共卫生应急体系上做了哪些修订？",
        "帮我查一下2022年美联储加息前后三个月的北向资金",
        "嗯那个，二零二四年三月的论文有哪些啊",
        "请问库里总共有多少篇文献",
        "我需要2018年机构改革前后的对比数据",
        "再三天（再三考虑不变）",
    ]
    for t in tests:
        r = clean(t)
        print(f"Q: {t}")
        print(f"  → {r.cleaned}")
        for name, a, b in r.steps:
            print(f"    [{name}] {a[:24]} → {b[:24]}")
        print()
