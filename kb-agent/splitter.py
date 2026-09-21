# -*- coding: utf-8 -*-
"""splitter.py —— 子问题拆分 v3：规则锚点 + 向量语义断层 + 属性继承

层次：
  rule_split()      规则粗切（疑问词/并列词/引导词/标点）→ 候选片段
  semantic_split()  向量验证（bge）：相邻片段相似度 → 切/合并决策
  apply_inheritance() 属性继承（时间/作者/期刊/语言/类型，就近+传递）
  split()           总入口（mode="rule" 纯函数秒开 / mode="semantic" 懒加载向量）

不依赖项目本体（search/ES），只做用户问题的结构化。
"""
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ---------- 规则表 ----------

QUESTION_MARKS = [
    "什么时候", "是什么", "哪款", "哪个", "哪些", "多少", "几篇",
    "是不是", "有没有", "是否", "是谁", "谁", "怎么样", "如何", "什么",
]
CONJ_MARKS = ["另外还有", "还有", "另外", "顺便", "以及", "同期", "同时期"]
GUIDE_WORDS = ["告诉我", "我想知道", "想知道", "我想", "帮我", "查一下", "看看", "请问", "请"]
PUNCTS = ["，", ",", "。", "；", ";", "？", "?", "！", "!"]  # 含半角（清洗层全角转半角后兼容）

# 继承链重置词（子句以这些开头 → 不继承前文属性）
RESET_WORDS = ["另外再看看", "顺便问下", "再查一下", "先查", "换个", "另外查"]

# 语义决策阈值（grid search 校准 2026-08-21：14 例数据集 100%，low 0.40→0.50）
# 规则锚点前置：事件窗口/并列词开头强制切；“的/的那篇”限定补全由 rule_split 吸收
SIM_CUT_LOW = 0.50    # 无锚点边界：低于此 → 语义断层，补切
SIM_CUT_HIGH = 0.55   # 强锚点边界：高于此且无独立疑问 → 合并（修误切）


@dataclass
class SubQuery:
    text: str
    intent: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    time: Dict = field(default_factory=dict)
    time_ranges: List[Dict] = field(default_factory=list)  # 多时间段（复合问题继承集合）
    entities: Dict = field(default_factory=dict)
    role: str = "main"              # main / context / time / constraint / output
    inherit_from: Optional[str] = None       # "sub_1" / None
    inherited: Dict = field(default_factory=dict)  # 继承了哪些属性
    step_type: str = "retrieval"     # retrieval / date_derivation / data_query / calculation
    depends_on: List[int] = field(default_factory=list)
    requires_tools: List[str] = field(default_factory=list)
    executable: bool = True           # 当前 Searcher 是否可直接执行
    blocked_reason: str = ""
    parameters: Dict = field(default_factory=dict)  # Planner/MCP 可直接消费的步骤参数

    def to_dict(self):
        return {
            "text": self.text, "intent": self.intent, "source": self.source,
            "confidence": self.confidence, "time": self.time,
            "time_ranges": self.time_ranges,
            "entities": self.entities, "role": self.role,
            "inherit_from": self.inherit_from, "inherited": self.inherited,
            "step_type": self.step_type, "depends_on": self.depends_on,
            "requires_tools": self.requires_tools, "executable": self.executable,
            "blocked_reason": self.blocked_reason,
            "parameters": self.parameters,
        }


# ---------- 规则粗切 ----------

def _find_anchors(q: str) -> List[tuple]:
    """收集锚点：(位置, 结束位置, 类型) 类型: qmark/conj/guide/punct/noun_conj"""
    anchors = []
    # 计算列表内的逗号、句号和“以及”都是公式项分隔，不是子问题边界。
    calc_spans = [m.span() for m in re.finditer(
        r"(?:并)?(?:分别)?计算.*?(?=[。；;]?\s*最终(?:输出|给出|判断)|$)", q)]

    def in_calc_span(position: int) -> bool:
        return any(start <= position < end for start, end in calc_spans)
    for m in re.finditer("|".join(re.escape(x) for x in QUESTION_MARKS), q):
        # 排除：数值区间语境（"从多少变化到了多少"里的"多少"不是疑问边界）
        before = q[max(0, m.start() - 2):m.start()]
        after = q[m.end():m.end() + 2]
        if before.endswith("从") or after.startswith("到了") or after.startswith("到"):
            continue
        anchors.append((m.start(), m.end(), "qmark"))
    for mk in CONJ_MARKS:
        for m in re.finditer(re.escape(mk), q):
            if in_calc_span(m.start()):
                continue
            after = q[m.end():m.end() + 4]
            if mk == "以及" and re.match(r"以前|之前|以后", after):
                continue
            anchors.append((m.start(), m.end(), "conj"))
    # 名词并列“和/与”（两个独立名词短语）→ noun_conj 锚点
    # 排除：时间并列（2020年和2021年）/ 定语共享（贝叶斯和GAN的论文）/ 对比疑问（ESN和LSTM哪个好）
    for m in re.finditer(r"[和与]", q):
        before = q[max(0, m.start() - 16):m.start()]
        after = q[m.end():m.end() + 14]
        # 对称或共享字段表达是一个条件组，不是两个可独立执行的问题。
        paired_window = (
            re.search(r"(?:前|之前)\s*\d+\s*个?(?:交易日|工作日|天|周|月|年)\s*$", before)
            and re.match(r"\s*(?:后|之后)\s*\d+\s*个?(?:交易日|工作日|天|周|月|年)", after)
        )
        paired_metric = (
            re.search(r"(?:最高(?:价|值)?|最大(?:值)?|上限|起点|开始时间)\s*$", before)
            and re.match(r"\s*(?:最低(?:价|值)?|最小(?:值)?|下限|终点|结束时间)", after)
        ) or (
            re.search(r"(?:最低(?:价|值)?|最小(?:值)?|下限|终点|结束时间)\s*$", before)
            and re.match(r"\s*(?:最高(?:价|值)?|最大(?:值)?|上限|起点|开始时间)", after)
        )
        if paired_window or paired_metric:
            continue
        # 时间并列排除：前是年份/月份/日期数字
        if re.search(r"(?:19|20)\d{2}年|(?<!\d)\d{1,2}月|(?<!\d)\d{1,2}[日号]", before):
            continue
        # 定语共享排除：后段含“的”（贝叶斯和GAN的论文）
        if "的" in after:
            continue
        # 对比疑问排除：后段含 哪个/吗/呢/是不是
        if re.search(r"哪个|吗|呢|是不是", after):
            continue
        # 后段过短（单字）不切（“我和他”人名并列）
        if len(after.strip()) < 2:
            continue
        anchors.append((m.start(), m.end(), "noun_conj"))
    for mk in GUIDE_WORDS:
        for m in re.finditer(re.escape(mk), q):
            anchors.append((m.start(), m.end(), "guide"))
    for mk in PUNCTS:
        for m in re.finditer(re.escape(mk), q):
            if in_calc_span(m.start()):
                continue
            anchors.append((m.start(), m.end(), "punct"))
    # 排序 + 重叠处理（同位置取最长；qmark 优先于 punct）
    anchors.sort(key=lambda x: (x[0], -len(q[x[0]:x[1]])))
    dedup, seen = [], set()
    for a in anchors:
        if a[0] not in seen:
            dedup.append(a)
            seen.add(a[0])
    return dedup


def rule_split(q: str) -> List[tuple]:
    """规则粗切：返回 [(片段文本, 该片段与下一片段之间的边界类型)]"""
    q = q.strip()
    if not q:
        return []
    # 内部礼貌/操作壳只表达“再问一个问题”，不应粘在前一子句尾部。
    # 例：A，并告诉我B → A，B；句首“告诉我B”仍交给原 GUIDE_WORDS 逻辑。
    q = re.sub(
        r"\s*[,，]?\s*(?:并|再|然后|同时)\s*(?:请)?(?:告诉我|帮我|查一下|看看)\s*",
        "，", q,
    )
    anchors = _find_anchors(q)
    segments = []
    prev = 0
    for i, (start, end, kind) in enumerate(anchors):
        # 边界切分点：qmark/guide/punct/noun_conj 切在词后；conj 切在词前（词归下段）
        if kind == "conj":
            cut = start
        else:
            cut = end
        seg = q[prev:cut].strip("，,。；;？！?! ")
        if seg:
            segments.append((seg, kind))
        prev = cut
    last = q[prev:].strip("，,。；;？！?! ")
    if last:
        segments.append((last, None))
    # 单字尾段（语气词 好/吗/呢/吧/啊/呀）→ 并回前段（"ESN和LSTM哪个好" → 完整保留）
    if len(segments) >= 2 and len(segments[-1][0]) == 1 \
            and segments[-1][0] in "好吗呢吧啊呀哦":
        t, _ = segments[-1]
        segments[-2] = (segments[-2][0] + t, segments[-2][1])
        segments.pop()
    # 过滤单字符/纯标点段
    segments = [(s, k) for s, k in segments if len(s) >= 2]
    if not segments:
        return [(q, None)]
    # ---- 段吸收（修碎段）：引导词并入后段；限定补全尾并入前段 ----
    # 吸收判定：短段且符合以下任一 → 独立（不吸收）：
    #   以 呢/吗 结尾（疑问语气）| 含疑问/枚举词 | 以 也行/也可以/吧 结尾（完整子句）
    # 以 "的" 结尾 → 限定补全 → 吸收
    def _absorbable(text):
        # 限定补全（"的/的那篇"结尾且无独立疑问词、非并列省略句）→ 吸收
        # （"2024年的论文，讲贝叶斯的那篇" 吸收；"还有关于GAN的" 是并列省略，不吸收）
        # 排除条件式配对段（"只要知识库中的"，与"不要…"配对，final 阶段合并）
        if (re.search(r"(的那篇|的)$", text) and not _is_independent(text)
                and not text.startswith(tuple(CONJ_MARKS))
                and not re.match(r"只要|就要|只", text)):
            return True
        if len(text) > 6:
            return False
        if any(text.startswith(w) for w in CONJ_MARKS + GUIDE_WORDS):
            return False
        if re.search(r"[呢吗]$", text):
            return False
        if re.search(r"哪些|多少|几篇|什么时候|谁|有没有", text):
            return False
        if re.search(r"也行|也可以|吧$|就好$|就行$|随便", text):
            return False
        return True

    merged = []
    i = 0
    while i < len(segments):
        text, kind = segments[i]
        # 引导词段（"我想知道/告诉我/帮我"等）→ 丢弃（引导语不构成独立段，后段继续处理）
        if text in GUIDE_WORDS and i + 1 < len(segments):
            i += 1
            continue
        # 名词并列后段（前段边界是 noun_conj）→ 不吸收，独立成段
        if merged and merged[-1][1] == "noun_conj":
            merged.append((text, kind))
            i += 1
            continue
        # 短段吸收
        if _absorbable(text) and merged:
            merged[-1] = (merged[-1][0] + text, merged[-1][1])
            i += 1
            continue
        # 并列引导段（以 CONJ 开头且自身不完整）→ 与后段合并（"顺便看看"+"ESN的"）
        # 完整省略句（"还有关于GAN的"以"的"收尾）不合并
        # 含事件窗口的时间限定（"以及…至…期间"）不合并——它是独立 time 子句
        if (i + 1 < len(segments) and text.startswith(tuple(CONJ_MARKS))
                and not re.search(r"[呢吗的]$|哪些|多少|几篇|什么时候|谁|有没有", text)
                and not re.search(EVENT_WINDOW_PATTERN, text)):
            merged.append((text + segments[i + 1][0], segments[i + 1][1]))
            i += 2
            continue
        # 宾语补全吸收：前段以疑问词结尾（"做了哪些"）→ 后段是名词短语宾语，无论长度
        # 排除：后段本身含疑问词（"…是多少，…从多少变化到了多少" 是并列疑问句，不吸收）
        # 排除：后段是输出指令（"按月列出/按年统计"，动词开头）→ 独立 output 子句
        if (merged and re.search(r"(哪些|什么|多少|几篇|有没有|谁|哪款|哪个)$", merged[-1][0])
                and not re.search(r"也行|也可以|吧$|呢$|吗$|就好$|就行$|随便|，|,|以及|还有", text)
                and not re.search(r"哪些|什么|多少|几篇|有没有|谁|哪款|哪个|吗|呢", text)
                and not re.match(r"按.{0,4}(月|年|季度|日|周|时间|天)|请|分别|分开|汇总|整理|列出|统计", text)
                and not text.startswith(tuple(CONJ_MARKS))):
            merged[-1] = (merged[-1][0] + text, merged[-1][1])
            i += 1
            continue
        merged.append((text, kind))
        i += 1
    # ---- 条件式配对合并："不要X，只要Y" 是一个整体表达 ----
    final = []
    j = 0
    while j < len(merged):
        t1, k1 = merged[j]
        if (j + 1 < len(merged) and re.search(r"不要|除了|排除|不含", t1)
                and re.match(r"只要|就要|只", merged[j + 1][0])):
            final.append((t1 + "，" + merged[j + 1][0], k1))
            j += 2
            continue
        final.append((t1, k1))
        j += 1
    # 相对窗口与其共享指标是同一个数据查询，不应被中间逗号拆开。
    collapsed = []
    for text, kind in final:
        if (collapsed
                and re.search(r"前\s*\d+\s*个?(?:交易日|工作日|天).*后\s*\d+\s*个?(?:交易日|工作日|天)(?:内)?$", collapsed[-1][0])
                and re.search(r"股票|股价|最高(?:价|值)?|最低(?:价|值)?", text)):
            collapsed[-1] = (collapsed[-1][0] + "，" + text, kind)
        else:
            collapsed.append((text, kind))
    return collapsed if collapsed else [(q, None)]


def _is_independent(text: str) -> bool:
    """弱独立判断：段内含疑问/枚举/统计词 → 可能是独立问题"""
    return bool(re.search(
        r"什么时候|什么|哪款|哪个|哪些|多少|几篇|是不是|有没有|是否|是谁|谁|怎么样|如何|呢|吗$",
        text,
    ))


# ---------- 向量语义验证 ----------

def semantic_split(q: str, emb, sim_low=SIM_CUT_LOW, sim_high=SIM_CUT_HIGH) -> List[str]:
    """规则粗切 + 向量验证。返回最终子句文本列表。"""
    segments = rule_split(q)
    if len(segments) <= 1:
        return [segments[0][0]] if segments else [q]

    texts = [s for s, _ in segments]
    embs = emb.encode(texts, query_mode=True, batch_size=16)

    # 决策：从后向前合并（保证合并后片段连续性）
    keep = [True] * len(segments)          # keep[i] = segments[i] 是否保留为子句起点
    for i in range(len(segments) - 1):
        sim = float(emb.cosine(embs[i], embs[i + 1]))
        kind = segments[i][1]  # 边界类型（i 与 i+1 之间）
        strong = kind in ("qmark", "conj", "noun_conj")
        # 时间限定后段（事件窗口结构）→ 强制切（不受阈值影响）
        if re.search(EVENT_WINDOW_PATTERN, texts[i + 1]):
            keep[i + 1] = True
            continue
        # 并列省略后段（还有/另外/顺便/以及 开头）→ 强制切（强独立信号）
        if texts[i + 1].startswith(tuple(CONJ_MARKS)):
            keep[i + 1] = True
            continue
        if strong:
            if sim < sim_high:
                keep[i + 1] = True      # 切
            else:
                # 高相似：两段都独立 → 切；否则合并
                keep[i + 1] = _is_independent(texts[i]) and _is_independent(texts[i + 1])
        else:
            if sim < sim_low:
                keep[i + 1] = True      # 语义断层补切
            else:
                keep[i + 1] = False     # 合并

    subs = []
    cur = texts[0]
    for i in range(1, len(texts)):
        if keep[i]:
            subs.append(cur)
            cur = texts[i]
        else:
            cur = cur + "，" + texts[i]
    subs.append(cur)
    return subs


# ---------- 附属成分识别（约束/输出/时间限定，非独立子问题） ----------

CONSTRAINT_PATTERN = r"过滤|排除|只要|不要|剔除|只讲|只选|不看|仅限|删掉|不包含|去掉"
OUTPUT_PATTERN = r"^(?:最终)?输出|^(?:最终)?(?:判断|给出)|按.{0,8}(展示|整理|列出|统计|分开|分组|汇总|排列)|整理出|需要.{0,6}(展示|列出|整理)|分开列|分别列|细分到|按.{0,6}分类"
ROLE_PREFIX = r"^(同时|并且|还要|另外|以及|顺便|需要|再|剔除掉?|要|只|把|重点|同时要)"
# 事件窗口结构（时间解析不出年份时也判为时间限定）
EVENT_WINDOW_PATTERN = r"从.{1,20}(到|至)|前后|期间|以来|至今|到现在|之前到现在"
CONTEXT_TIME_PATTERN = r"^(?:今天|当前日期|现在时间|基准日期)是"
BUSINESS_RULE_PATTERN = (
    r"(?:通常|一般|规定|约定|固定)?.{0,8}"
    r"(?:每年|每月|每季度|季度|年度).{0,20}"
    r"(?:第\s*\d+\s*个?(?:工作日|交易日)|月末|季末|年末).{0,12}"
    r"(?:发布|公布|披露|执行|生效)"
)
DATA_SCOPE_PATTERN = r"^(?:分析|处理|查看).{0,40}(?:SKU|销售数据|业务数据|库存数据|数据集)$"


def classify_role(text: str, time: Dict, intent) -> str:
    """子句角色：main=独立问题 / time=时间限定 / constraint=约束 / output=输出要求"""
    if re.search(CONTEXT_TIME_PATTERN, text):
        return "time"
    if re.search(BUSINESS_RULE_PATTERN, text):
        return "constraint"
    if re.search(DATA_SCOPE_PATTERN, text, re.I):
        return "context"
    if time.get("sort") == "latest_before_anchor":
        return "main"
    if re.search(OUTPUT_PATTERN, text):
        return "output"
    if re.search(CONSTRAINT_PATTERN, text) and re.match(ROLE_PREFIX, text):
        return "constraint"
    # 时间限定：连接/介词开头含时间；或事件窗口结构（非典结束到现在/上市前后/被制裁至今）
    time_lead = bool(re.match(r"^(以及|特别是|尤其|在|从|自|以及从|从.{0,15}到)", text))
    event_window = bool(re.search(EVENT_WINDOW_PATTERN, text))
    if (time_lead and _has_time(time)) or (event_window and not _is_independent(text)):
        return "time"
    if _has_time(time) and not _is_independent(text) and not intent:
        return "time"
    return "main"


def _annotate_dependencies(results: List[SubQuery]) -> None:
    """为需要 Planner/MCP 的依赖链标注执行元数据，不在 NLU 层擅自执行。"""
    for sub in results:
        if sub.role == "main" and re.search(
                r"计算|波动率|收益率|方差|标准差|最高\s*/\s*最低", sub.text):
            sub.step_type = "calculation"
            sub.requires_tools = ["calculator"]
            sub.executable = False
            sub.blocked_reason = "数值计算需要 Agent Planner 先准备输入，当前 Searcher 不直接执行"
    reference_chain = any(re.search(r"该日期|这个日期|上述日期|前述日期|这两个时间窗口", s.text) for s in results)
    analytic_chain = (
        any(re.search(r"计算|环比|同比|波动率|收益率", s.text) for s in results)
        and any(re.search(r"提取|查询|获取|汇总|销售总额", s.text) for s in results)
    )
    dependent = reference_chain or analytic_chain
    if not dependent:
        return

    support_indices = []
    previous_main = None
    for index, sub in enumerate(results, 1):
        if sub.role in ("context", "time", "constraint"):
            support_indices.append(index)
            sub.step_type = "context" if sub.role in ("context", "time") else "constraint"
            sub.executable = False
            continue
        if sub.role == "output":
            sub.step_type = "output"
            if previous_main is not None:
                sub.depends_on = [previous_main]
            pairs = re.findall(r"(\d{1,2})月\s*[-~]\s*(\d{1,2})月", sub.text)
            sub.parameters = {
                "criterion": "mom > 0 and yoy > 0",
                "candidate_month_pairs": [f"{left.zfill(2)}-{right.zfill(2)}" for left, right in pairs],
            }
            if pairs and "组合" in sub.text:
                # “如 7月-6月”是结果组合示例，不是倒序时间区间。
                sub.time = {}
                sub.time_ranges = []
            sub.executable = False
            sub.blocked_reason = "等待上游计算结果后由 Agent 生成最终输出"
            continue
        if sub.role != "main":
            sub.executable = False
            continue

        if re.search(r"计算|波动率|收益率|方差|标准差|最高\s*/\s*最低", sub.text):
            sub.step_type = "calculation"
            sub.requires_tools = ["calculator"]
        elif re.search(r"股票|股价|交易日|最高价|最低价", sub.text):
            sub.step_type = "data_query"
            sub.requires_tools = ["market_calendar", "market_data"]
        elif re.search(r"销售数据|销售总额|SKU|库存单位", sub.text, re.I):
            sub.step_type = "data_query"
            sub.requires_tools = ["sales_data"]
            sub.parameters = {
                "metric": "sales_total",
                "year_months": [t.get("from", "")[:7] for t in sub.time_ranges if t.get("granularity") == "month"],
            }
        elif re.search(r"发布日期|披露日期|财报|工作日", sub.text):
            sub.step_type = "date_derivation"
            sub.requires_tools = ["business_calendar"]

        if previous_main is not None and (
                (analytic_chain and sub.step_type == "calculation")
                or re.search(r"该日期|这个日期|上述|前述|这两个|这些|其", sub.text)):
            sub.depends_on = [previous_main]
        elif support_indices:
            sub.depends_on = list(support_indices)
        if sub.requires_tools or sub.depends_on:
            sub.executable = False
            sub.blocked_reason = "需要 Agent Planner 按依赖顺序调用外部工具，当前 Searcher 不直接执行"
        previous_main = index

        if sub.step_type == "calculation":
            comparisons = []
            for match in re.finditer(
                    r"((?:19|20)\d{2})年(\d{1,2})月\s*(?:vs|VS|对比)\s*((?:19|20)\d{2})年(\d{1,2})月", sub.text):
                before = sub.text[:match.start()]
                kind = "yoy" if before.rfind("同比") > before.rfind("环比") else "mom"
                comparisons.append({
                    "type": kind,
                    "left": f"{match.group(1)}-{int(match.group(2)):02d}",
                    "right": f"{match.group(3)}-{int(match.group(4)):02d}",
                    "formula": "left / right - 1",
                })
            sub.parameters = {"comparisons": comparisons}


# ---------- 属性继承 ----------

_ATTR_KEYS = ["author", "venue", "language", "doc_type"]

# 时间意图词：子句含这些词但解析不出时间 → 事件时间未解析（阻断继承，防止错误时间传播）
TIME_INTENT_WORDS = r"期间|以来|到现在|前后|未来|近几年|这几年|那几年|当年|至今"


def _has_time(t: Dict) -> bool:
    return bool(t and (t.get("op") or t.get("sort") or t.get("fuzzy")))


def _time_key(t: Dict) -> str:
    """时间去重键（区间/精确值）"""
    return json.dumps({k: t.get(k) for k in ("op", "from", "to", "exact") if t.get(k)}, sort_keys=True)


def apply_inheritance(subs_text: List[str], analyze_fn) -> List[SubQuery]:
    """对每个子句 analyze + 角色标注 + 时间分组继承。
    时间组作用域：main 继承【自上一个 main 以来收集的时间组】；
    main 之后出现新的 time 子句 → 开新组（"T1, main1, T2, main2" → main2 只继承 T2）。
    constraint/output 是附属成分，不参与继承；事件时间未解析的不进组。"""
    results = []
    pending_times = []      # 当前时间组（仅含已解析的时间子句）
    group_consumed = False  # 当前组是否已被某个 main 消费过
    _attr_ctx = {}          # 属性上下文（作者/期刊等，就近传递）
    for i, text in enumerate(subs_text):
        sq = analyze_fn(text)
        sub = SubQuery(
            text=text, intent=sq.intent, source=sq.source,
            confidence=sq.confidence, time=dict(sq.time),
            time_ranges=[dict(t) for t in sq.time_ranges],
            entities=dict(sq.entities),
        )
        sub.role = classify_role(text, sub.time, sub.intent)
        unresolved_time = (sub.role == "time" and not _has_time(sub.time)
                           and bool(re.search(TIME_INTENT_WORDS, text)))

        if sub.role == "time":
            # main 之后出现新时间 → 开新组（不跨组传递）
            if group_consumed:
                pending_times = []
                group_consumed = False
            if _has_time(sub.time):
                keys = [_time_key(t) for t in pending_times]
                if _time_key(sub.time) not in keys:
                    pending_times.append(dict(sub.time))
        elif sub.role == "main":
            # 主问题自带时间 → 新组起点（自己的时间优先，不算继承）
            if sub.time.get("sort") == "latest_before_anchor" and pending_times:
                sub.time_ranges = [dict(t) for t in pending_times] + [dict(sub.time)]
                sub.inherited["time_ranges"] = [dict(t) for t in pending_times]
                sub.inherit_from = "time_group"
                pending_times = [dict(t) for t in sub.time_ranges]
                group_consumed = False
            elif _has_time(sub.time):
                own_times = [dict(t) for t in sub.time_ranges] or [dict(sub.time)]
                pending_times = own_times
                group_consumed = False
                sub.time_ranges = own_times
            # 无自带时间 → 继承当前时间组（快照）
            elif pending_times:
                sub.time_ranges = [dict(t) for t in pending_times]
                sub.time = dict(sub.time_ranges[0])
                sub.inherited["time_ranges"] = sub.time_ranges
                sub.inherit_from = "time_group"
            group_consumed = True
            # 实体继承（属性仍就近传递）
            if i > 0:
                for key in _ATTR_KEYS:
                    if not sub.entities.get(key) and _attr_ctx.get(key):
                        sub.inherited[key] = _attr_ctx[key]
                        sub.entities[key] = _attr_ctx[key]
            for key in _ATTR_KEYS:
                if sub.entities.get(key):
                    _attr_ctx[key] = sub.entities[key]
        # 属性上下文（作者/期刊等）
        if sub.role == "time" and _has_time(sub.time):
            pass
        results.append(sub)
    # 修正 inherit_from 为实际来源序号（时间组来源）
    for idx, sub in enumerate(results):
        if sub.inherited and sub.inherit_from == "time_group":
            # 向前找最近的 time 子句作为来源标注
            for back in range(idx - 1, -1, -1):
                if results[back].role == "time" and _has_time(results[back].time):
                    sub.inherit_from = f"sub_{back + 1}"
                    break
    _annotate_dependencies(results)
    return results


# ---------- 总入口 ----------

def split(q: str, emb=None, mode: str = "rule", analyze_fn=None) -> List[SubQuery]:
    """mode="rule"：纯规则切分（秒开，无需模型）
       mode="semantic"：+ 向量语义断层验证（需传 emb）
       analyze_fn：子句分析函数（默认 nlu.analyze），可注入以便测试"""
    if analyze_fn is None:
        from nlu import analyze as analyze_fn_default
        analyze_fn = analyze_fn_default

    if mode == "semantic" and emb is not None:
        subs_text = semantic_split(q, emb)
    else:
        segments = rule_split(q)
        subs_text = [s for s, _ in segments] if len(segments) > 1 else [q]

    return apply_inheritance(subs_text, analyze_fn)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from nlu import analyze
    tests = [
        "2024年的ESN论文有哪些，还有关于GAN的",
        "我想知道2024年的论文里有没有讲贝叶斯的",
        "去年发了哪些论文，今年呢",
        "2021年以及以前的文献有哪些",
        "告诉我3060什么时候发行的，现在性能最好的显卡是什么",
        "2024年的论文有哪些，还有关于GAN的，张昭昭的呢",
    ]
    for t in tests:
        print(f"Q: {t}")
        for i, sub in enumerate(split(t, analyze_fn=analyze), 1):
            inh = f" [继承自 {sub.inherit_from}: {sub.inherited}]" if sub.inherited else ""
            print(f"  sub{i}: {sub.text}  time={sub.time}{inh}")
        print()
