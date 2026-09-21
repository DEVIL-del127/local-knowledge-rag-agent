# memory_retrieval.py - 记忆检索与指代检测（M2）
# 对应《任务拆分 v1.1》: memory_search 能力 + M2 指代检测三级设计
# 确定性部分纯函数可单测; 向量检索依赖外部 embedder(复用 bge-m3)
from __future__ import annotations

import re
from typing import Any, Sequence

# ---------- M2① 规则级指代检测 ----------
ANAPHORA_WORDS = (
    "它", "他", "她", "那个", "这个", "刚才", "之前", "上次", "前面",
    "上面", "那些", "这种", "那篇", "那本书", "那条", "那件事",
)
# 疑问指代/主语省略标记(触发查询重写)
QUERY_ANAPHORA = (
    "哪些", "什么", "如何", "咋样", "怎么样", "为啥", "为什么", "怎么", "多少",
)
ANAPHORA_PHRASES = (
    "刚才说的", "上次提的", "上次说的", "之前那个", "之前说的",
    "前面说的", "那个关于", "之前提过", "上次聊的",
)


def detect_anaphora(message: str, window_messages: Sequence[str], window_size: int = 3) -> bool:
    """规则级指代检测: 含指代词/锚点 且 前 N 轮无对应先行词"""
    if not message:
        return False
    # 短消息才走规则(长消息自带上下文, 误触发率高)
    if len(message) > 50:
        return False
    has_anchor = any(p in message for p in ANAPHORA_PHRASES) or any(
        w in message for w in ANAPHORA_WORDS
    )
    if not has_anchor:
        return False
    # 先行词检查: 窗口内最近消息是否有实体词(非指代的名词)
    recent = " ".join(window_messages[-window_size:])
    return not bool(extract_entities(recent))


# ---------- 实体提取(指代召回用) ----------
ENTITY_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-]{1,24}"           # 英文/数字词 GAN, ESN, B100
    r"|[\u4e00-\u9fa5]{2,6}(?:论文|项目|方案|模型|系统|方法|实验|结果|数据|网络|算法|技术|资料|工作|事)?"  # 中文名词
)
# 停用字符集: 全部由这些字符组成的中文 token 视为非实体(指代/口语)
STOP_CHARS = set(
    "那个什么怎么这这些那些是吗的好继续嗯可以知道明白没问题稍等是的对啊行吧谢谢"
    "咋俺咱啥哎哎呀哦哦嗯嗯呀呢吧嘛啦"
)
# 领域后缀(强实体标志)
DOMAIN_SUFFIXES = (
    "论文", "项目", "方案", "模型", "系统", "方法", "实验", "结果",
    "数据", "网络", "算法", "技术", "资料", "工作", "研究", "原理",
    "结构", "框架", "应用", "预测",
)
# 弱前缀(连词/介词开头的碎片不是实体)
WEAK_PREFIXES = ("和", "与", "及", "或", "并", "在", "对", "把", "从", "为", "用")
# 尾部动词(贪婪匹配吃多的修正: "刘月的论文写" → "刘月的论文")
VERB_TAILS = (
    "写", "用", "讲", "做", "说", "是", "有", "着", "考", "看",
    "查", "完", "改", "提", "读", "问", "算", "学", "了", "过",
)
# 指示后缀(截断: "刘月那篇" → "刘月")
DEICTIC_TAILS = (
    "那篇", "这篇", "那本", "这本", "那几篇", "那个", "这个",
    "那些", "这种", "那篇论文", "这篇论文",
)
# 指示前缀(触发替换: "上次查的论文" → 窗口实体)
DEICTIC_PREFIXES = ("上次", "刚才", "之前", "前面", "先前", "上回")


def _strip_verb_tail(token: str) -> str:
    for _ in range(2):
        if len(token) > 2 and token[-1] in VERB_TAILS:
            token = token[:-1]
        else:
            break
    return token


def is_strong_entity(token: str) -> bool:
    """实体强度: 英文词 / ≤4字 / 以领域后缀结尾"""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,24}", token):
        return True
    if len(token) <= 4:
        return True
    return any(token.endswith(s) for s in DOMAIN_SUFFIXES)


def _strip_deictic_tail(token: str) -> str:
    """指示后缀截断: "刘月那篇"→"刘月"; "GAN那个"→"GAN"""
    for suffix in sorted(DEICTIC_TAILS, key=len, reverse=True):
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[: -len(suffix)]
    return token


def extract_entities(text: str) -> list[str]:
    """提取候选实体: 英文词 + 中文名词片段; 过滤指代/停用/疑问/弱碎片/动词尾/指示后缀"""
    STOP = {"这个", "那个", "这些", "那些", "什么", "怎么", "为什么", "是不是", "没有", "不是"}
    found = []
    for match in ENTITY_RE.finditer(text or ""):
        token = match.group(0)
        if token in STOP:
            continue
        if token in ANAPHORA_WORDS:
            continue
        # 纯停用字符组成的中文 token(如"那个什么怎么")跳过
        if all(c in STOP_CHARS for c in token):
            continue
        # 含疑问指代的 token("写了什么"/"用了哪些")不是实体
        if any(q in token for q in QUERY_ANAPHORA):
            continue
        # 尾部动词截断(贪婪匹配吃多): "刘月的论文写"→"刘月的论文"
        token = _strip_verb_tail(token)
        # 指示后缀截断: "刘月那篇"→"刘月"
        token = _strip_deictic_tail(token)
        # 弱前缀碎片("和模型")与弱实体("今天天气不错")剔除
        if any(token.startswith(p) for p in WEAK_PREFIXES):
            continue
        if not is_strong_entity(token):
            continue
        if token not in found:
            found.append(token)
    return found[:5]


# ---------- 查询重写(多轮指代补全) ----------
def rewrite_query_with_context(message: str, window_messages: Sequence[str]) -> str:
    """查询重写: 消息含疑问指代/主语省略 → 用上文实体补全查询
    返回重写后的查询(无实体可补则返回原消息)
    例: "用了哪些技术和模型" + 上文["刘月的论文写了什么"]
        → "刘月的论文 用了哪些技术和模型"
    """
    msg = (message or "").strip()
    if not msg:
        return msg

    # 指示短语(上次/刚才/之前 + 名词)触发替换, 即使消息提取到实体
    has_deictic_prefix = any(msg.startswith(p) for p in DEICTIC_PREFIXES)
    # 消息尾部指示后缀("刘月那篇"→"刘月")清理
    cleaned_msg = _strip_deictic_tail(msg) if msg.endswith(DEICTIC_TAILS) else msg

    # 消息本身含实体 且 无指示前缀 → 无需重写
    if extract_entities(cleaned_msg) and not has_deictic_prefix:
        return cleaned_msg

    # 含疑问指代 或 消息过短 或 含指示前缀 → 需要上下文补全
    needs = (
        any(q in cleaned_msg for q in QUERY_ANAPHORA)
        or len(cleaned_msg) <= 12
        or has_deictic_prefix
    )
    if not needs:
        return cleaned_msg

    # 从窗口最近消息提取实体(优先最近的 user 消息)
    entities: list[str] = []
    for text in reversed(list(window_messages)[-2:]):
        entities = extract_entities(text)
        if entities:
            break
    if not entities:
        return cleaned_msg
    entity = entities[0]
    if entity in cleaned_msg:
        return cleaned_msg

    # 指示短语替换: "上次查的论文用了哪些模型技术" → "刘月的论文用了哪些模型技术"
    if has_deictic_prefix:
        import re as _re

        pattern = _re.compile(
            r"(?:上次|刚才|之前|前面|先前|上回)"
            r"(?:(?:查|找|看|说|提|讲|问|写|用)?的?|那个|这个|那篇|这篇|那本)"
            r"(?:那篇论文|论文|方案|模型|那篇|文章|资料|实验|结果|东西)"
        )
        rewritten = pattern.sub(entity, cleaned_msg, count=1)
        if rewritten != cleaned_msg:
            return rewritten

    return f"{entity} {cleaned_msg}"


# ---------- 关键词检索(L1 < 500 条) ----------
def keyword_search(facts: Sequence[dict], entities: Sequence[str], top_k: int = 5) -> list[dict]:
    """简单打分: 实体命中数 > 置信度 > 时间新; 兼容 L1(content)与 L2(value)结构"""
    scored = []
    for fact in facts:
        content = str(fact.get("content") or fact.get("value") or "")
        hits = sum(1 for e in entities if e and e.lower() in content.lower())
        if hits == 0:
            continue
        scored.append((hits, float(fact.get("confidence", 0)), fact))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item[2] for item in scored[:top_k]]


# ---------- 注入格式化(对应 §4 模板) ----------
def format_injection(
    kind: str,
    items: Sequence[dict],
    *,
    budget_tokens: int = 400,
    source_label: str | None = None,
) -> str:
    """格式化记忆注入块; 按预算裁剪(超预算先砍画像型, 再砍 summary 型)"""
    if not items:
        return ""
    lines: list[str] = []
    if kind == "会话记忆":
        lines.append("[会话记忆]")
        for item in items[:5]:
            ftype = item.get("type", "fact")
            lines.append(f"- {ftype}: {str(item.get('content', ''))[:80]}（置信度 {item.get('confidence', '')}）")
        if source_label:
            lines.append(f"（来源: {source_label}）")
    elif kind == "长期记忆":
        lines.append("[长期记忆]")
        for item in items[:5]:
            entity = item.get("entity", "")
            relation = item.get("relation", "")
            value = str(item.get("value", ""))[:100]
            lines.append(f"- {entity} {relation} {value}")
        if source_label:
            lines.append(f"（来源: {source_label}）")
    else:  # 画像
        lines.append("[画像]")
        for item in items[:3]:
            field = item.get("field", "")
            value = str(item.get("value", ""))[:80]
            lines.append(f"- {field}: {value}")

    text = "\n".join(lines)
    # 预算裁剪: 估算 token(中英混合粗略: 字符数/2), 超预算逐行砍尾部
    while len(text) // 2 > budget_tokens and len(lines) > 1:
        lines.pop()
        text = "\n".join(lines)
    return text.strip()


# ---------- 向量检索(≥500 条 / L2) ----------
def vector_search(
    collection,
    embedder,
    query: str,
    *,
    top_k: int = 5,
    threshold: float = 0.65,
    where: dict | None = None,
) -> list[dict]:
    """ChromaDB 向量检索 + 相似度阈值过滤"""
    if embedder is None or collection is None:
        return []
    emb = embedder.embed_query(query)
    if emb is None:
        return []
    kwargs = {"query_embeddings": [emb], "n_results": top_k,
              "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    results = []
    ids = (res.get("ids") or [[]])[0]
    for item_id, doc, meta, dist in zip(
        ids, res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1.0 - dist
        if score < threshold:
            continue
        results.append({**dict(meta), "id": item_id, "content": doc, "score": score})
    return results
