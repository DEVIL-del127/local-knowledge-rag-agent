# -*- coding: utf-8 -*-
"""nlu.py —— 意图识别 + 问题结构化（NLU 层）

设计原则：
1. 纯函数：不依赖 ES / 向量模型，输入自然语言 → 输出结构化对象（StructuredQuery）
2. 表驱动：时间解析（TIME_RULES）、意图判定（INTENT_RULES）都是声明式规则表
   —— 新增形态 = 表加一行 + 测试矩阵加一行，禁止堆 if-else
3. 可观测：analyze(debug=True) 返回每一步解析轨迹 steps，可单独查看
4. 可单测：test_nlu.py 直接断言结构化输出

用法：
    from nlu import analyze
    sq = analyze("找25年之前的文献", current_year=2026)   # 纯函数
    print(sq.to_dict())   # 查看结构化结果
"""
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ================= 数据结构 =================

@dataclass
class TimeExpr:
    op: Optional[str] = None          # lt/lte/gt/gte/between/exact
    from_: Optional[str] = None       # ISO 片段："2024" / "2024-03" / "2024-03-15" / "2024-03-15T14:30"
    to: Optional[str] = None
    exact: Optional[str] = None
    granularity: str = "year"         # year / month / day / hour / minute
    sort: Optional[str] = None        # year_desc
    fuzzy: bool = False               # 高度模糊 → 需澄清
    fuzzy_word: str = ""             # 半模糊默认值的来源词（"最近"等）
    timezone: str = ""               # IANA 时区；只保留语义，不在 NLU 层擅自换算
    invalid_reason: str = ""         # 日期/时间越界等确定性错误
    raw: str = ""

    def to_dict(self):
        d = {}
        if self.op: d["op"] = self.op
        if self.from_ is not None: d["from"] = self.from_
        if self.to is not None: d["to"] = self.to
        if self.exact is not None: d["exact"] = self.exact
        if self.granularity != "year": d["granularity"] = self.granularity
        if self.sort: d["sort"] = self.sort
        if self.fuzzy: d["fuzzy"] = True
        if self.fuzzy_word: d["fuzzy_word"] = self.fuzzy_word
        if self.timezone: d["timezone"] = self.timezone
        if self.invalid_reason: d["invalid_reason"] = self.invalid_reason
        if self.raw: d["raw"] = self.raw
        return d


@dataclass
class ParseStep:
    name: str
    detail: str

    def to_dict(self):
        return {"step": self.name, "detail": self.detail}


@dataclass
class StructuredQuery:
    query: str                        # 清洗后的问题
    intent: Optional[str] = None      # None = 规则层未命中（下沉 kNN/LLM）
    confidence: float = 0.0
    source: str = ""                  # rule_<规则名> / invalid / clarification / non_kb / unclassified
    entities: Dict = field(default_factory=dict)   # author/venue/language/doc_type/doc_ids/exclude
    time: Dict = field(default_factory=dict)       # TimeExpr.to_dict()
    time_ranges: List[Dict] = field(default_factory=list)  # 多年份/多时间点（如 2014,2018,2022）
    clarification: str = ""
    steps: List[ParseStep] = field(default_factory=list)

    def to_dict(self):
        d = {
            "query": self.query, "intent": self.intent,
            "confidence": self.confidence, "source": self.source,
            "entities": self.entities, "time": self.time,
            "clarification": self.clarification,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.time_ranges:
            d["time_ranges"] = self.time_ranges
        return d


# ================= 词典 =================

AUTHORS = ["张昭昭", "朱应钦", "余文", "李俊明", "刘月", "王植炜", "缪季", "庞昭辰",
           "胡雅璇", "段广仁", "曹喜滨", "Xiaoou Li", "Jorge Morales", "Wen Yu",
           "Luciano Sánchez", "Jinsung Yoon"]

VENUES = ["neurocomputing", "information", "控制理论与应用", "控制工程", "自动化学报",
          "地球信息科学学报", "上海交通大学学报", "华中科技大学学报", "procedia", "neurips",
          "cce", "arxiv", "engineering applications", "eaa"]

DOC_IDS = [
    "1-s2.0-S0925231221011309", "1-s2.0-S0952197625001290", "1-s2.0-S1877050925003059",
    "2210.02040", "CCE-2020", "information-15-00222",
    "Metaheuristic_Method_for_Dimensionality_Reduction_Tasks", "NeurIPS-2019",
    "具有双储层结构的动态误差补偿回声状态网络", "回声信念网络及其在时间序列预测中的应用",
    "基于GAN的视网膜血管分割标签优化方法", "基于改进MCMC算法和代理模型的结构仿真模型更新",
    "基于条件扩散模型的卫星遥测数据缺失值插补方法", "论文刘月", "贝叶斯时空统计方法及应用进展与趋势",
]

NON_KB_PATTERNS = [
    r"写.{0,8}(代码|脚本|程序|函数)",
    r"改.{0,8}(代码|脚本|程序|bug)",
    r"(写|做|出).{0,4}(方案|设计|计划|报告)(?!.*(论文|文献))",
    r"翻译.{0,8}(这篇|文档|pdf|文件|文章)",
    r"(删除|移除|改名|重命名|排序|整理).{0,6}(文档|文件|论文|文献)",
    r"(今天|现在).{0,3}(几号|几点|星期|时间|日期)",
    r"天气|你叫什么|你是谁|谢谢|再见|你好呀?$|在吗",
    r"分析.{0,6}(代码|pdf|文件|数据|报错|乱码|方案)(?!.*(论文|文献))",
]

EXCLUDE_WORDS = ["不要", "除了", "排除", "不含", "别给", "别", "剔除", "去掉", "不看"]
CITATION_WORDS = ["引用", "参考文献", "reference", "references", "引用了", "引用列表", "参考文献列表", "citation"]
COMPARE_WORDS = ["区别", "对比", "比较", "异同", "有什么不同", "差别", "相比", "相较", "vs", "哪个好", "谁更好", "区别在哪", "不同之处"]
ENUM_WORDS = ["列出", "所有", "哪些", "几篇", "盘点", "统计", "分别", "都有哪些", "全部", "总数", "多少篇",
              "盘一下", "罗列", "理一理", "列一下", "整理下", "列举", "枚举", "一共", "合计", "多少次"]
STAT_WORDS = ["几篇", "多少", "总数", "统计", "占比", "数量", "篇数", "合计", "一共"]
META_WORDS = ["作者", "哪年", "年份", "doi", "期刊", "页数", "标题", "谁写的", "出版社",
              "第几页", "页码", "出版时间", "刊名", "卷号", "期号", "哪个期刊", "哪里发的"]
TOPIC_WORDS = ["贝叶斯", "GAN", "生成对抗", "ESN", "回声状态", "LSTM", "MCMC", "神经网络",
               "扩散模型", "插补", "降维", "地震", "遥测", "时间序列", "储层", "元启发"]
LANG_WORDS = {"中文": "zh", "英文": "en", "英语": "en"}
TYPE_WORDS = {"硕士论文": "thesis", "学位论文": "thesis", "会议": "conference",
              "期刊": "journal", "预印本": "preprint", "arxiv": "preprint"}
SEMANTIC_PATTERN = r"关于|相关|方法|主题|怎么样|如何|内容|讲|介绍|了解|说说|知道|用了|使用|哪个好"
STRONG_REF_PATTERN = r"这篇|该文|那篇|那份|本文|上面|刚才|那个|这个"
FIND_VERB_PATTERN = r"找找|找一下|帮我找|搜一下|搜索|查一下"
CONTENT_WORD_PATTERN = r"方法|内容|讲了|讲什么|如何|怎么样|用了|摘要|结论"

# 数值计算强信号（增长率/差值/占比/倍数/翻番等）——优先于语义/枚举/比较
CALC_WORDS = ["增长率", "增速", "环比", "同比", "差值", "占比", "比例", "百分比",
              "翻了几倍", "翻几倍", "翻了几番", "翻几番", "倍数", "涨了多少", "增长了多少",
              "增长多少", "增长更快", "增速差", "差多少", "平均每年增长", "年均增长",
              "增长率是多少", "增长速率", "速率差异", "幅度是多少", "变化到了多少",
              "波动率", "收益率", "方差", "标准差", "极差", "复合增长率"]


# ================= 清洗 =================

def clean(query: str) -> str:
    q = query.strip()
    if len(q) > 500:
        q = q[:500]
    return q


def is_invalid(q: str) -> bool:
    return not q or bool(re.fullmatch(r"[\s\W_]+", q))


def is_non_kb(q: str) -> bool:
    return any(re.search(p, q, re.I) for p in NON_KB_PATTERNS)


# ================= 时间表达式解析（表驱动 v3：年/月/日/时/分粒度） =================

# 事件锚点词典：事件名 → (起始年, 结束年)。新增事件 = 加一行（规则层，无需模型）
# 只匹配带时间后缀的用法（以来/期间/前后/之后/之前）；裸事件名当主题词不解析
EVENT_ANCHORS = {
    "非典": (2003, 2003),
    "sars": (2003, 2003),
    "疫情": (2020, 2022),
    "新冠": (2020, 2022),
    "甲流": (2009, 2009),
    "金融危机": (2008, 2009),
    "次贷危机": (2007, 2008),
    "汶川地震": (2008, 2008),
    "北京奥运": (2008, 2008),
    "改革开放": (1978, 1978),
    "入世": (2001, 2001),
    "加入wto": (2001, 2001),
    "互联网泡沫": (2000, 2001),
    "贸易战": (2018, 2019),
}

_EVENT_NAMES = "|".join(sorted(EVENT_ANCHORS, key=len, reverse=True))
_CN_DIGIT = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _event_anchor_year(raw: str) -> tuple:
    """取事件名（大小写不敏感）对应的 (起始年, 结束年)"""
    for key in EVENT_ANCHORS:
        if raw[:len(key)].lower() == key.lower():
            return EVENT_ANCHORS[key]
    return EVENT_ANCHORS[list(EVENT_ANCHORS)[0]]


def _h_event(m, cur):
    """事件锚点：非典以来/疫情期间/金融危机前后/非典之后/非典以前"""
    raw = m.group(0)
    y1, y2 = _event_anchor_year(raw)
    name = next(k for k in EVENT_ANCHORS if raw[:len(k)].lower() == k.lower())
    tail = raw[len(name):]
    # 前后N年/月/季度
    mm = re.search(r"前后([\d一二两三四五六七八九十]+)(?:个)?(年|月|季度)", tail)
    if mm:
        n = int(mm.group(1)) if mm.group(1).isdigit() else _CN_DIGIT.get(mm.group(1), 1)
        unit = mm.group(2)
        if unit == "年":
            return TimeExpr(op="between", from_=f"{y1 - n}-01-01", to=f"{y2 + n}-12-31",
                            granularity="year", fuzzy_word=f"前后{n}年", raw=raw)
        dm = n * 3 if unit == "季度" else n
        fy, fmo = y1, 1 - dm
        while fmo <= 0:
            fmo += 12
            fy -= 1
        ty, tmo = y2, 12 + dm
        while tmo > 12:
            tmo -= 12
            ty += 1
        from calendar import monthrange
        last = monthrange(ty, tmo)[1]
        return TimeExpr(op="between", from_=f"{fy:04d}-{fmo:02d}-01",
                        to=f"{ty:04d}-{tmo:02d}-{last:02d}",
                        granularity="month", fuzzy_word=f"前后{n}{unit}", raw=raw)
    # 前后（无数字）→ 前后各1年
    if "前后" in tail:
        return TimeExpr(op="between", from_=f"{y1 - 1}-01-01", to=f"{y2 + 1}-12-31", raw=raw)
    # 期间 → 事件持续区间
    if "期间" in tail:
        return TimeExpr(op="between", from_=f"{y1}-01-01", to=f"{y2}-12-31", raw=raw)
    # 以来/至今/到现在 → 事件起点至今
    if any(w in tail for w in ("以来", "至今", "到现在")):
        return TimeExpr(op="between", from_=f"{y1}-01-01",
                        to=f"{cur[0]:04d}-{cur[1]:02d}-{cur[2]:02d}",
                        granularity="day", raw=raw)
    # 之后/以后 → 事件结束后
    if any(w in tail for w in ("之后", "以后")):
        return TimeExpr(op="gte", from_=str(y2 + 1), raw=raw)
    # 之前/以前 → 事件开始前
    if any(w in tail for w in ("之前", "以前")):
        return TimeExpr(op="lte", to=str(y1 - 1), raw=raw)
    return None


# 半模糊词默认窗口（论文检索语境；模型接口预留，见方案 v2 §3.5）
FUZZY_DEFAULTS = {
    "最近": (3, "month"), "近期": (3, "month"), "前阵子": (1, "month"),
    "最近一段时间": (3, "month"), "年初": None, "年中": None, "年底": None,
    "上半年": None, "下半年": None,
}


def _month_range(y, m):
    import calendar
    last = calendar.monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"


def _h_range_2digit(m, cur):
    return TimeExpr(op="between", from_=str(2000 + int(m.group(1))), to=str(2000 + int(m.group(2))), raw=m.group(0))

def _h_range_4digit(m, cur):
    return TimeExpr(op="between", from_=m.group(1), to=m.group(2), raw=m.group(0))

def _h_before_2digit(m, cur):
    return TimeExpr(op="lte", to=str(2000 + int(m.group(1)) - 1), raw=m.group(0))

def _h_after_2digit(m, cur):
    return TimeExpr(op="gte", from_=str(2000 + int(m.group(1)) + 1), raw=m.group(0))

def _h_before_4digit(m, cur):
    return TimeExpr(op="lte", to=str(int(m.group(1)) - 1), raw=m.group(0))

def _h_until_4digit(m, cur):
    return TimeExpr(op="lte", to=m.group(1), raw=m.group(0))

def _h_after_4digit(m, cur):
    return TimeExpr(op="gte", from_=str(int(m.group(1)) + 1), raw=m.group(0))

def _h_since_4digit(m, cur):
    return TimeExpr(op="gte", from_=m.group(1), raw=m.group(0))

def _h_decade(m, cur):
    return TimeExpr(op="between", from_=m.group(1), to=str(int(m.group(1)) + 9), raw=m.group(0))

def _h_decade_2digit(m, cur):
    y = 2000 + int(m.group(1))
    if y > cur[0]:
        y -= 100
    return TimeExpr(op="between", from_=str(y), to=str(y + 9), raw=m.group(0))

def _h_exact(m, cur):
    return TimeExpr(op="exact", exact=m.group(1), raw=m.group(0))

def _h_exact_2digit(m, cur):
    return TimeExpr(op="exact", exact=str(2000 + int(m.group(1))), raw=m.group(0))

def _h_month_full(m, cur):
    y, mo = int(m.group(1)), int(m.group(2))
    f, t = _month_range(y, mo)
    return TimeExpr(op="between", from_=f, to=t, granularity="month", raw=m.group(0))

def _h_month_short(m, cur):
    mo = int(m.group(1))
    f, t = _month_range(cur[0], mo)
    return TimeExpr(op="between", from_=f, to=t, granularity="month", raw=m.group(0))

def _h_day_full(m, cur):
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    reason = _validate_datetime(y, mo, d)
    return TimeExpr(op="exact", exact=f"{y:04d}-{mo:02d}-{d:02d}", granularity="day",
                    invalid_reason=reason, raw=m.group(0))


def _to_24_hour(hour: int, period: str = "") -> int:
    """把中文时段转换为 24 小时制；无时段时保持原值。"""
    period = period or ""
    if period in ("下午", "傍晚", "晚上") and 1 <= hour < 12:
        return hour + 12
    if period == "中午" and 1 <= hour < 11:
        return hour + 12
    if period in ("凌晨", "清晨", "早上", "上午") and hour == 12:
        return 0
    return hour


def _validate_datetime(year: int, month: int, day: int,
                       hour: int = 0, minute: int = 0) -> str:
    try:
        datetime(year, month, day, hour, minute)
        return ""
    except ValueError as exc:
        return f"无效日期或时间：{exc}"


def _clock_parts(m, suffix: str):
    hour = int(m.group(f"h{suffix}"))
    minute_raw = m.group(f"min{suffix}") or m.group(f"min{suffix}_cn")
    minute = int(minute_raw) if minute_raw is not None else 0
    return _to_24_hour(hour, m.group(f"ap{suffix}") or ""), minute

def _h_datetime_range(m, cur):
    """跨日/跨月分钟级区间：2024年8月15日上午10:00 至 8月20日凌晨2:00
    支持：同日（2024年3月5日 09:00 至 11:30）/ 同月跨日 / 跨月；终点可省略年份。
    时区词（北京时间/美国东部时间）出现在前后文中不影响（正则不消费时区词）。
    """
    y1, mo1, d1 = int(m.group("y1")), int(m.group("mo1")), int(m.group("d1"))
    hh1, mm1 = _clock_parts(m, "1")
    explicit_end_date = m.group("d2") is not None
    y2 = int(m.group("y2")) if m.group("y2") else y1
    mo2 = int(m.group("mo2")) if m.group("mo2") else mo1
    d2 = int(m.group("d2")) if m.group("d2") else d1
    hh2, mm2 = _clock_parts(m, "2")

    reason = _validate_datetime(y1, mo1, d1, hh1, mm1)
    reason = reason or _validate_datetime(y2, mo2, d2, hh2, mm2)
    if reason:
        return TimeExpr(granularity="minute", invalid_reason=reason, raw=m.group(0))

    start = datetime(y1, mo1, d1, hh1, mm1)
    end = datetime(y2, mo2, d2, hh2, mm2)
    # 终点未显式给年时允许自然跨年；只写终点时刻且更早则视为跨午夜。
    if end < start and not m.group("y2"):
        if explicit_end_date and mo2 < mo1:
            end = end.replace(year=end.year + 1)
        elif not explicit_end_date:
            end += timedelta(days=1)
    if end < start:
        return TimeExpr(granularity="minute", invalid_reason="结束时间早于开始时间",
                        raw=m.group(0))
    return TimeExpr(op="between",
                    from_=start.isoformat(timespec="minutes"),
                    to=end.isoformat(timespec="minutes"),
                    granularity="minute", raw=m.group(0))


def _h_date_range(m, cur):
    y1, mo1, d1 = int(m.group("dy1")), int(m.group("dmo1")), int(m.group("dd1"))
    y2 = int(m.group("dy2")) if m.group("dy2") else y1
    mo2 = int(m.group("dmo2")) if m.group("dmo2") else mo1
    d2 = int(m.group("dd2"))
    reason = _validate_datetime(y1, mo1, d1) or _validate_datetime(y2, mo2, d2)
    if reason:
        return TimeExpr(granularity="day", invalid_reason=reason, raw=m.group(0))
    start, end = datetime(y1, mo1, d1), datetime(y2, mo2, d2)
    if end < start and not m.group("dy2") and mo2 < mo1:
        end = end.replace(year=end.year + 1)
    if end < start:
        return TimeExpr(granularity="day", invalid_reason="结束日期早于开始日期", raw=m.group(0))
    return TimeExpr(op="between", from_=start.date().isoformat(), to=end.date().isoformat(),
                    granularity="day", raw=m.group(0))


def _h_day_to_month_range(m, cur):
    """完整日期到月份：终点按该月最后一天闭区间处理。"""
    import calendar
    y1, mo1, d1 = int(m.group("dmy1")), int(m.group("dmmo1")), int(m.group("dmd1"))
    y2, mo2 = int(m.group("dmy2")), int(m.group("dmmo2"))
    reason = _validate_datetime(y1, mo1, d1)
    try:
        d2 = calendar.monthrange(y2, mo2)[1]
    except (ValueError, calendar.IllegalMonthError):
        return TimeExpr(granularity="day", invalid_reason="终点月份超出有效范围", raw=m.group(0))
    if reason:
        return TimeExpr(granularity="day", invalid_reason=reason, raw=m.group(0))
    start, end = datetime(y1, mo1, d1), datetime(y2, mo2, d2)
    if end < start:
        return TimeExpr(granularity="day", invalid_reason="结束月份早于开始日期", raw=m.group(0))
    return TimeExpr(op="between", from_=start.date().isoformat(), to=end.date().isoformat(),
                    granularity="day", raw=m.group(0))


def _h_month_to_day_range(m, cur):
    """月份到完整日期：起点按该月第一天闭区间处理。"""
    y1, mo1 = int(m.group("mdy1")), int(m.group("mdmo1"))
    y2, mo2, d2 = int(m.group("mdy2")), int(m.group("mdmo2")), int(m.group("mdd2"))
    reason = _validate_datetime(y1, mo1, 1) or _validate_datetime(y2, mo2, d2)
    if reason:
        return TimeExpr(granularity="day", invalid_reason=reason, raw=m.group(0))
    start, end = datetime(y1, mo1, 1), datetime(y2, mo2, d2)
    if end < start:
        return TimeExpr(granularity="day", invalid_reason="结束日期早于开始月份", raw=m.group(0))
    return TimeExpr(op="between", from_=start.date().isoformat(), to=end.date().isoformat(),
                    granularity="day", raw=m.group(0))


def _h_clock_range(m, cur):
    hh1, mm1 = _clock_parts(m, "c1")
    hh2, mm2 = _clock_parts(m, "c2")
    if not 0 <= hh1 <= 23 or not 0 <= hh2 <= 23 or not 0 <= mm1 <= 59 or not 0 <= mm2 <= 59:
        return TimeExpr(granularity="minute", invalid_reason="小时或分钟超出有效范围", raw=m.group(0))
    return TimeExpr(op="between", from_=f"T{hh1:02d}:{mm1:02d}",
                    to=f"T{hh2:02d}:{mm2:02d}", granularity="minute", raw=m.group(0))


def _h_minute_full(m, cur):
    y, mo, d, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    reason = _validate_datetime(y, mo, d, hh, mm)
    return TimeExpr(op="exact", exact=f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}",
                    granularity="minute", invalid_reason=reason, raw=m.group(0))

def _h_hhmm(m, cur):
    hh, mm = int(m.group(1)), int(m.group(2))
    reason = "小时或分钟超出有效范围" if not 0 <= hh <= 23 or not 0 <= mm <= 59 else ""
    return TimeExpr(op="exact", exact=f"T{hh:02d}:{mm:02d}", granularity="minute",
                    invalid_reason=reason, raw=m.group(0))

def _h_hour_cn(m, cur):
    hh = int(m.group(2))
    if m.group(1) and "下" in m.group(1) and hh < 12:
        hh += 12
    if m.group(1) and "晚" in m.group(1) and hh < 12:
        hh += 12
    if m.group(1) and "上" in m.group(1) and hh >= 12:
        hh -= 12
    reason = "小时超出有效范围" if not 0 <= hh <= 23 else ""
    return TimeExpr(op="exact", exact=f"T{hh:02d}:00", granularity="minute",
                    invalid_reason=reason, raw=m.group(0))

def _h_month_range(m, cur):
    m1, m2 = int(m.group(1)), int(m.group(2))
    f, _ = _month_range(cur[0], m1)
    _, t = _month_range(cur[0], m2)
    return TimeExpr(op="between", from_=f, to=t, granularity="month", raw=m.group(0))

def _h_relative_ym(m, cur):
    y = cur[0] + {"去年": -1, "今年": 0, "明年": 1}.get(m.group(1), 0)
    mo = int(m.group(2))
    f, t = _month_range(y, mo)
    return TimeExpr(op="between", from_=f, to=t, granularity="month", raw=m.group(0))

def _h_rel_months(m, cur):
    num = m.group(1) or m.group(2)
    cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    n = int(num) if num.isdigit() else cn.get(num, 3)
    y, mo = cur[0], cur[1] - n
    while mo <= 0:
        mo += 12
        y -= 1
    f, _ = _month_range(y, mo)
    return TimeExpr(op="gte", from_=f, granularity="month", raw=m.group(0))

def _h_rel_days(m, cur):
    from datetime import date, timedelta
    if "周" in m.group(0) or "星期" in m.group(0):
        n = 7
    else:
        num = m.group(1) or m.group(2)
        cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        n = int(num) if num.isdigit() else cn.get(num, 3)
    start = date(cur[0], cur[1], cur[2]) - timedelta(days=n)
    return TimeExpr(op="gte", from_=start.isoformat(), granularity="day", raw=m.group(0))

def _h_rel_hours(m, cur):
    n = int(m.group(1))
    from datetime import datetime, timedelta
    now = datetime(cur[0], cur[1], cur[2], cur[3], cur[4])
    start = now - timedelta(hours=n)
    return TimeExpr(op="gte", from_=start.isoformat(timespec="minutes"), granularity="minute", raw=m.group(0))

def _h_this_month(m, cur):
    t = m.group(0)
    y, mo = cur[0], cur[1]
    if "上" in t:
        mo -= 1
    elif "下" in t:
        mo += 1
    if mo <= 0:
        mo += 12
        y -= 1
    elif mo > 12:
        mo -= 12
        y += 1
    f, tt = _month_range(y, mo)
    return TimeExpr(op="between", from_=f, to=tt, granularity="month", raw=m.group(0))

def _h_fuzzy_default(m, cur):
    word = m.group(0)
    # 模型优先（若已训练）：模糊词 + 上下文 → 窗口类；低置信/不可用 → 默认表
    try:
        from fuzzy_model import get_default_model
        win = get_default_model().window_for(m.string)  # 完整 query，模型看上下文
        if win:
            n, unit = win
            if unit == "month":
                y, mo = cur[0], cur[1] - n
                while mo <= 0:
                    mo += 12
                    y -= 1
                f, _ = _month_range(y, mo)
                return TimeExpr(op="gte", from_=f, granularity="month", fuzzy_word=word, raw=word)
            if unit == "week":
                from datetime import date, timedelta
                start = date(cur[0], cur[1], cur[2]) - timedelta(weeks=n)
                return TimeExpr(op="gte", from_=start.isoformat(), granularity="day", fuzzy_word=word, raw=word)
            if unit == "year":
                return TimeExpr(op="gte", from_=str(cur[0] - n), granularity="year", fuzzy_word=word, raw=word)
    except Exception:
        pass
    spec = FUZZY_DEFAULTS.get(word)
    if not spec:
        return TimeExpr(fuzzy=True, raw=word)
    n, unit = spec
    if unit == "month":
        y, mo = cur[0], cur[1] - n
        while mo <= 0:
            mo += 12
            y -= 1
        f, _ = _month_range(y, mo)
        return TimeExpr(op="gte", from_=f, granularity="month", fuzzy_word=word, raw=word)
    return TimeExpr(fuzzy=True, raw=word)


def _h_nearest_item(m, cur):
    """“距离锚点最近的一份”是排序/选取语义，不是“最近三个月”时间窗口。"""
    return TimeExpr(sort="latest_before_anchor", raw=m.group(0))

def _h_fuzzy_period(m, cur):
    word = m.group(0)
    y = cur[0]
    # 支持年份组合："2023年初" / "2021年上半年"
    ym = re.search(r"((?:19|20)\d{2})", word)
    if ym:
        y = int(ym.group(1))
    if "初" in word:
        return TimeExpr(op="between", from_=f"{y}-01-01", to=f"{y}-03-31", granularity="month", fuzzy_word=word, raw=word)
    if "年中" in word:
        return TimeExpr(op="between", from_=f"{y}-04-01", to=f"{y}-09-30", granularity="month", fuzzy_word=word, raw=word)
    if "年底" in word:
        return TimeExpr(op="between", from_=f"{y}-10-01", to=f"{y}-12-31", granularity="month", fuzzy_word=word, raw=word)
    if "上半年" in word:
        return TimeExpr(op="between", from_=f"{y}-01-01", to=f"{y}-06-30", granularity="month", fuzzy_word=word, raw=word)
    if "下半年" in word:
        return TimeExpr(op="between", from_=f"{y}-07-01", to=f"{y}-12-31", granularity="month", fuzzy_word=word, raw=word)
    return TimeExpr(fuzzy=True, raw=word)

def _h_fuzzy_vague(m, cur):
    return TimeExpr(fuzzy=True, raw=m.group(0))

def _h_relative(m, cur):
    t = m.group(0)
    if "去年" in t:
        return TimeExpr(op="exact", exact=str(cur[0] - 1), raw=t)
    if "今年" in t:
        return TimeExpr(op="exact", exact=str(cur[0]), raw=t)
    if "最近两年" in t or "最近2年" in t:
        return TimeExpr(op="gte", from_=str(cur[0] - 2), raw=t)
    if "近三年" in t:
        return TimeExpr(op="gte", from_=str(cur[0] - 3), raw=t)
    if "最新" in t:
        return TimeExpr(sort="year_desc", raw=t)
    return None


def _h_fuzzy(m, cur):
    return TimeExpr(fuzzy=True, raw=m.group(0))

def _h_month_after(m, cur):
    y, mo = int(m.group(1)), int(m.group(2))
    if mo == 12:
        y, mo = y + 1, 1
    else:
        mo += 1
    return TimeExpr(op="gte", from_=f"{y:04d}-{mo:02d}-01", granularity="month", raw=m.group(0))

def _h_month_before(m, cur):
    import calendar
    y, mo = int(m.group(1)), int(m.group(2))
    if mo == 1:
        y, mo = y - 1, 12
    else:
        mo -= 1
    last = calendar.monthrange(y, mo)[1]
    return TimeExpr(op="lte", to=f"{y:04d}-{mo:02d}-{last:02d}", granularity="month", raw=m.group(0))


def _h_around(m, cur):
    """事件前后窗口（近似）：2022年…前后三个月 → 以该年为中心前后扩展 N 月/年"""
    from datetime import date
    y, n = int(m.group(1)), int(m.group(2))
    unit = m.group(3)
    if unit == "年":
        return TimeExpr(op="between", from_=f"{y - n}-01-01", to=f"{y + n}-12-31",
                        granularity="year", fuzzy_word=f"前后{n}年", raw=m.group(0))
    dm = n * 3 if unit == "季度" else n
    # 以该年首尾向前后扩展 dm 个月
    fy, fmo = y, 1 - dm
    while fmo <= 0:
        fmo += 12
        fy -= 1
    ty, tmo = y, 12 + dm
    while tmo > 12:
        tmo -= 12
        ty += 1
    from calendar import monthrange
    last = monthrange(ty, tmo)[1]
    return TimeExpr(op="between", from_=f"{fy:04d}-{fmo:02d}-01", to=f"{ty:04d}-{tmo:02d}-{last:02d}",
                    granularity="month", fuzzy_word=f"前后{n}{unit}", raw=m.group(0))


def _h_month_range_full(m, cur):
    """X年A月到Y年B月（跨年）→ between y1-m1-01 ~ y2-m2-月末"""
    import calendar
    y1, m1, y2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    last = calendar.monthrange(y2, m2)[1]
    return TimeExpr(op="between", from_=f"{y1:04d}-{m1:02d}-01", to=f"{y2:04d}-{m2:02d}-{last:02d}",
                    granularity="month", raw=m.group(0))

def _h_now(m, cur):
    """动态锚点：现在/当前/如今 → 当前日期（日粒度）"""
    return TimeExpr(op="exact", exact=f"{cur[0]:04d}-{cur[1]:02d}-{cur[2]:02d}",
                    granularity="day", raw=m.group(0))

def _h_since_year_now(m, cur):
    """X年至今 → between X-01-01 ~ 当前日期（日粒度终点）"""
    return TimeExpr(op="between", from_=f"{m.group(1)}-01-01",
                    to=f"{cur[0]:04d}-{cur[1]:02d}-{cur[2]:02d}",
                    granularity="day", raw=m.group(0))

def _h_month_to_now(m, cur):
    """X年X月至今（含同义词 迄今/为止/到现在）→ between X年X月 ~ 当前日期"""
    y, mo = int(m.group(1)), int(m.group(2))
    return TimeExpr(op="between", from_=f"{y:04d}-{mo:02d}-01",
                    to=f"{cur[0]:04d}-{cur[1]:02d}-{cur[2]:02d}",
                    granularity="day", raw=m.group(0))


def _h_day_to_now(m, cur):
    """X年X月X日至今 → between X年X月X日 ~ 当前日期（日粒度）"""
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return TimeExpr(op="between", from_=f"{y:04d}-{mo:02d}-{d:02d}",
                    to=f"{cur[0]:04d}-{cur[1]:02d}-{cur[2]:02d}",
                    granularity="day", raw=m.group(0))


def _h_future(m, cur):
    """未来N年：当前年 ~ 当前年+N-1（未来5年=2026~2030）"""
    raw = m.group(0)
    num = m.group(1)
    if num and num.isdigit():
        n = int(num)
    else:
        cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        mm = re.search(r"[一二两三四五六七八九十]+", raw)
        n = cn.get(mm.group(0), 5) if mm else 5
    return TimeExpr(op="between", from_=str(cur[0]), to=str(cur[0] + n - 1),
                    granularity="year", raw=raw)


_DAY_PERIOD = r"凌晨|清晨|早上|上午|中午|下午|傍晚|晚上"
_DATETIME_RANGE_PATTERN = (
    rf"(?P<y1>(?:19|20)\d{{2}})年(?P<mo1>\d{{1,2}})月(?P<d1>\d{{1,2}})[日号]?\s*"
    rf"(?:(?P<ap1>{_DAY_PERIOD})\s*)?(?P<h1>\d{{1,2}})"
    rf"(?:(?:[:：](?P<min1>\d{{2}}))|(?:[点时](?P<min1_cn>\d{{1,2}})?分?))\s*"
    rf"(?:至|到|~|-)\s*"
    rf"(?:(?P<y2>(?:19|20)\d{{2}})年)?"
    rf"(?:(?P<mo2>\d{{1,2}})月(?P<d2>\d{{1,2}})[日号]?)?\s*"
    rf"(?:(?P<ap2>{_DAY_PERIOD})\s*)?(?P<h2>\d{{1,2}})"
    rf"(?:(?:[:：](?P<min2>\d{{2}}))|(?:[点时](?P<min2_cn>\d{{1,2}})?分?))"
)
_DATE_RANGE_PATTERN = (
    r"(?P<dy1>(?:19|20)\d{2})年(?P<dmo1>\d{1,2})月(?P<dd1>\d{1,2})[日号]?\s*"
    r"(?:至|到|~|-)\s*(?:(?P<dy2>(?:19|20)\d{2})年)?"
    r"(?:(?P<dmo2>\d{1,2})月)?(?P<dd2>\d{1,2})[日号]"
)
_DAY_TO_MONTH_RANGE_PATTERN = (
    r"(?P<dmy1>(?:19|20)\d{2})年(?P<dmmo1>\d{1,2})月(?P<dmd1>\d{1,2})[日号]?\s*"
    r"(?:至|到|~|-)\s*(?P<dmy2>(?:19|20)\d{2})年(?P<dmmo2>\d{1,2})月(?:份)?"
)
_MONTH_TO_DAY_RANGE_PATTERN = (
    r"(?P<mdy1>(?:19|20)\d{2})年(?P<mdmo1>\d{1,2})月(?:份)?\s*"
    r"(?:至|到|~|-)\s*(?P<mdy2>(?:19|20)\d{2})年(?P<mdmo2>\d{1,2})月"
    r"(?P<mdd2>\d{1,2})[日号]"
)
_CLOCK_RANGE_PATTERN = (
    rf"(?:(?P<apc1>{_DAY_PERIOD})\s*)?(?P<hc1>\d{{1,2}})"
    rf"(?:(?:[:：](?P<minc1>\d{{2}}))|(?:[点时](?P<minc1_cn>\d{{1,2}})?分?))\s*"
    rf"(?:至|到|~|-)\s*(?:(?P<apc2>{_DAY_PERIOD})\s*)?(?P<hc2>\d{{1,2}})"
    rf"(?:(?:[:：](?P<minc2>\d{{2}}))|(?:[点时](?P<minc2_cn>\d{{1,2}})?分?))"
)


# 规则表：(名称, 正则, 处理器) —— 按优先级排列；宽区间必须先于单点规则
TIME_RULES = [
    ("event_anchor", rf"({_EVENT_NAMES})(?:结束)?(?:以来|至今|到现在|爆发以来|期间|前后(?:[\d一二两三四五六七八九十]+(?:个)?(?:年|月|季度))?|之后|以后|之前|以前)", _h_event),
    ("month_to_now", r"((?:19|20)\d{2})年(\d{1,2})月(?:份)?(?:至今|迄今|为止|到现在)", _h_month_to_now),
    ("since_year_now", r"((?:19|20)\d{2})年(?:至今|迄今|为止|到现在)", _h_since_year_now),
    ("day_to_month_range", _DAY_TO_MONTH_RANGE_PATTERN, _h_day_to_month_range),
    ("month_to_day_range", _MONTH_TO_DAY_RANGE_PATTERN, _h_month_to_day_range),
    ("month_range_full", r"((?:19|20)\d{2})年(\d{1,2})月(?:份)?(?:到|至|~|-)((?:19|20)\d{2})年(\d{1,2})月", _h_month_range_full),
    ("now_anchor", r"(?<!出)(?:到)?现在|当前|如今|当下", _h_now),
    ("future_n", r"未来(?:(\d{1,2})|[一二两三四五六七八九十]+)年", _h_future),
    ("around_window", r"((?:19|20)\d{2})年.{0,24}前后(\d{1,2})(?:个)?(年|月|季度)", _h_around),
    ("datetime_range", _DATETIME_RANGE_PATTERN, _h_datetime_range),
    ("date_range", _DATE_RANGE_PATTERN, _h_date_range),
    ("clock_range", _CLOCK_RANGE_PATTERN, _h_clock_range),
    ("minute_full", r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})[日号]?\s*(\d{1,2})[点时:：](\d{1,2})分?", _h_minute_full),
    ("month_after", r"((?:19|20)\d{2})年(\d{1,2})月(?:份)?(?:之后|以后|以来)", _h_month_after),
    ("month_before", r"((?:19|20)\d{2})年(\d{1,2})月(?:份)?(?:之前|以前)", _h_month_before),
    ("day_to_now", r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})[日号]?\s*(?:至今|迄今|为止|到现在)", _h_day_to_now),
    ("day_full", r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})[日号]", _h_day_full),
    ("month_full", r"((?:19|20)\d{2})年(\d{1,2})月", _h_month_full),
    ("time_hhmm", r"(?<!\d)(\d{1,2})[:：](\d{2})\s*(?:分)?(?!\d)", _h_hhmm),
    ("time_cn", r"([上中下晚]午)?\s*(\d{1,2})[点时]", _h_hour_cn),
    ("month_range", r"(?<!\d)(\d{1,2})月(?:份)?(?:到|至|~|-)(\d{1,2})月", _h_month_range),
    ("year_month_rel", r"(去年|今年|明年)(\d{1,2})月", _h_relative_ym),
    ("rel_months", r"近(?:(\d{1,2})|([一二两三四五六七八九十]+))个?月", _h_rel_months),
    ("rel_days", r"近(?:(\d{1,2})|([一二两三四五六七八九十]+))天|近一周|近一个星期", _h_rel_days),
    ("rel_hours", r"近(\d{1,2})个?小时", _h_rel_hours),
    ("this_month", r"上个月|这个月|本月|下个月", _h_this_month),
    ("fuzzy_vague", r"那几年|那阵子|刚发布那会儿|毕业后|刚.{0,4}那会儿", _h_fuzzy_vague),
    ("fuzzy_period", r"((?:19|20)\d{2})?\s*(?:年初|年中|年底|上半年|下半年)", _h_fuzzy_period),
    ("nearest_item", r"(?:距离|离)(?:今天|当前|现在|指定日期|该日期).{0,10}?最近的?(?:一|1|某)?(?:份|篇|次|条|个)|最近的?(?:一|1)(?:份|篇|次|条|个)", _h_nearest_item),
    ("relative", r"去年|今年|最近两年|最近2年|最新|近三年", _h_relative),
    ("fuzzy_default", r"最近|近期|前阵子|最近一段时间", _h_fuzzy_default),
    ("range_2digit", r"(?<!\d)(\d{2})\s*年?\s*(?:到|至|~|-)\s*(\d{2})\s*年?(?!\d)", _h_range_2digit),
    ("range_4digit", r"((?:19|20)\d{2})\s*(?:年)?\s*(?:到|至|~|-)\s*((?:19|20)\d{2})", _h_range_4digit),
    ("until_4digit", r"((?:19|20)\d{2})\s*年?\s*及\s*(?:以前|之前)", _h_until_4digit),
    ("before_2digit", r"(?<!\d)(\d{2})\s*年\s*(?:之前|以前)", _h_before_2digit),
    ("after_2digit", r"(?<!\d)(\d{2})\s*年\s*(?:之后|以后)", _h_after_2digit),
    ("before_4digit", r"(?<!\d)((?:19|20)\d{2})\s*年?\s*(?:之前|以前)", _h_before_4digit),
    ("after_4digit", r"(?<!\d)((?:19|20)\d{2})\s*年?\s*(?:之后|以后)", _h_after_4digit),
    ("since_4digit", r"(?<!\d)((?:19|20)\d{2})\s*年?\s*(?:以来|起)", _h_since_4digit),
    ("decade_4digit", r"((?:19|20)\d{2})\s*年代", _h_decade),
    ("decade_2digit", r"(?<!\d)(\d{2})\s*年代", _h_decade_2digit),
    ("fuzzy", r"(?<!\d)(\d{2}|\d{4})\s*年\s*左右", _h_fuzzy),
    ("exact_4digit", r"(?<!\d)((?:19|20)\d{2})\s*年?(?!\d)", _h_exact),
    ("exact_2digit", r"(?<!\d)(\d{2})\s*年(?!\d)", _h_exact_2digit),
]

_TIME_RE = [(name, re.compile(pat), fn) for name, pat, fn in TIME_RULES]

_TIMEZONE_ALIASES = {
    "北京时间": "Asia/Shanghai", "中国标准时间": "Asia/Shanghai", "中国时间": "Asia/Shanghai",
    "美国东部时间": "America/New_York", "美东时间": "America/New_York",
    "美国西部时间": "America/Los_Angeles", "美西时间": "America/Los_Angeles",
    "英国时间": "Europe/London", "伦敦时间": "Europe/London",
    "日本时间": "Asia/Tokyo", "东京时间": "Asia/Tokyo",
    "UTC": "UTC", "GMT": "UTC",
}
_TIMEZONE_RE = re.compile("|".join(sorted(map(re.escape, _TIMEZONE_ALIASES), key=len, reverse=True)), re.I)


def _prepare_time_query(query: str) -> str:
    """剔除会伪装成年份/时间的文档标识符，同时保持字符串位置不变。"""
    patterns = (r"\d{4}\.\d{5}[v.]?\d*", r"S\d{11,}", r"10\.\d{4,9}/[\w.\-]+")
    q = query
    for pattern in patterns:
        q = re.sub(pattern, lambda m: " " * len(m.group(0)), q)
    return q


def _timezone_before(query: str, position: int) -> str:
    matches = list(_TIMEZONE_RE.finditer(query, 0, position))
    if not matches:
        return ""
    # 时区作用域延续到下一个时区词；跨句时不继承，避免污染后续独立问题。
    last = matches[-1]
    if re.search(r"[。！？!?]", query[last.end():position]):
        return ""
    return _TIMEZONE_ALIASES.get(last.group(0), _TIMEZONE_ALIASES.get(last.group(0).upper(), ""))


def parse_time_ranges(query: str, current_year: int = 2026, current_month: int = 8,
                      current_day: int = 20, current_hour: int = 15,
                      current_minute: int = 0) -> List[TimeExpr]:
    """解析全部互不重叠的时间表达式。

    规则优先级仍由 TIME_RULES 决定：先选完整区间，再屏蔽区间内部的单日期、
    单年份和单时刻候选。最终结果按原文顺序返回。
    """
    cur = (current_year, current_month, current_day, current_hour, current_minute)
    q = _prepare_time_query(query)
    selected = []
    for priority, (name, pattern, fn) in enumerate(_TIME_RE):
        for match in pattern.finditer(q):
            start, end = match.span()
            if any(start < old_end and end > old_start for old_start, old_end, _, _ in selected):
                continue
            try:
                expr = fn(match, cur)
            except (ValueError, OverflowError):
                expr = TimeExpr(invalid_reason="时间表达式数值无效", raw=match.group(0))
            if not expr:
                continue
            expr.raw = match.group(0)
            expr.timezone = _timezone_before(query, start)
            selected.append((start, end, priority, expr))
    selected.sort(key=lambda item: item[0])
    return [item[3] for item in selected]


def parse_time(query: str, current_year: int = 2026, current_month: int = 8,
               current_day: int = 20, current_hour: int = 15, current_minute: int = 0) -> TimeExpr:
    """统一时间表达式解析 v3：年/月/日/时/分粒度。
    current_* 为"当前时间"注入（保持纯函数可测）。"""
    ranges = parse_time_ranges(query, current_year, current_month, current_day,
                               current_hour, current_minute)
    return ranges[0] if ranges else TimeExpr()


# 多年份/多时间点分隔符（括号内 或 逗号/顿号/空格连接）
_MULTI_YEAR_SEP = r"[,，、;；/\s]+"
_MULTI_YEAR_RE = re.compile(
    rf"[（(]?(?:\d{{4}}){_MULTI_YEAR_SEP}(?:\d{{4}})(?:{_MULTI_YEAR_SEP}\d{{4}})*[）)]?")


def parse_multi_years(query: str) -> List[TimeExpr]:
    """多年份解析："2014,2018,2022" / "（2014,2018,2022）" / "2014、2018、2022年"
    返回多个 exact 时间点（≥2 个）；单个年份/年份范围不在此列。
    """
    # 排除范围写法（2019-2024 / 2019到2024）——分隔符不含 - 和“到”，天然隔离
    m = _MULTI_YEAR_RE.search(query)
    if not m:
        return []
    years = re.findall(r"\d{4}", m.group(0))
    if len(years) < 2:
        return []
    exprs = []
    for y in years:
        exprs.append(TimeExpr(op="exact", exact=y, raw=m.group(0)))
    return exprs


_YEAR_MONTH_MATRIX_RE = re.compile(
    r"(?P<years>(?:19|20)\d{2}\s*年(?:\s*(?:和|与|、|,|，)\s*(?:19|20)\d{2}\s*年)+)"
    r"\s*的?\s*"
    r"(?P<months>\d{1,2}\s*月(?:份)?(?:\s*(?:和|与|、|,|，)\s*\d{1,2}\s*月(?:份)?)+)"
)


def parse_year_month_matrix(query: str) -> List[TimeExpr]:
    """解析“2025年和2026年的6、7、8月”，展开为年×月的精确月份集合。"""
    match = _YEAR_MONTH_MATRIX_RE.search(query)
    if not match:
        return []
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", match.group("years"))]
    months = [int(value) for value in re.findall(r"(\d{1,2})\s*月", match.group("months"))]
    if not years or not months or any(month < 1 or month > 12 for month in months):
        return []
    exprs = []
    for year in dict.fromkeys(years):
        for month in dict.fromkeys(months):
            start, end = _month_range(year, month)
            exprs.append(TimeExpr(
                op="between", from_=start, to=end, granularity="month",
                raw=f"{year}年{month}月",
            ))
    return exprs


# ================= 实体提取 =================

def parse_exclude_phrase(phrase: str) -> Dict:
    body = re.sub(r"^(不要|除了|排除|不含|别给|别|剔除|去掉|不看)", "", phrase)
    ex = {}
    if not body:
        return ex
    for a in AUTHORS:
        if a in body:
            ex.setdefault("authors", []).append(a)
            return ex
    for w, code in LANG_WORDS.items():
        if w in body:
            ex.setdefault("languages", []).append(code)
            return ex
    for w, code in TYPE_WORDS.items():
        if w in body:
            ex.setdefault("doc_types", []).append(code)
            return ex
    if "翻译" in body:
        ex["translated"] = True
        return ex
    m = re.search(r"(19|20)\d{2}|(?<!\d)\d{2}年", body)
    if m:
        y = m.group(0).replace("年", "")
        ex.setdefault("years", []).append(int(y) + 2000 if len(y) == 2 and int(y) < 100 else int(y))
        return ex
    for did in DOC_IDS:
        if did.lower() in body.lower():
            ex.setdefault("doc_ids", []).append(did)
            return ex
    return ex


def extract_entities(q: str, session: Dict = None) -> Dict:
    """实体提取：author/venue/language/doc_type/doc_ids/exclude（纯函数）"""
    session = session or {}
    ent: Dict = {"author": None, "venue": None, "language": None, "doc_type": None,
                 "doc_ids": [], "exclude": {}}

    # 排除短语（先抓，防污染其他实体）
    m = re.search(r"(?:不要|除了|排除|不含|别给|(?<!分)别|剔除|去掉|不看)[^，。;；]{1,15}", q)
    if m:
        ent["exclude"] = parse_exclude_phrase(m.group(0))

    # 文档定位
    for did in DOC_IDS:
        if did.lower() in q.lower():
            ent["doc_ids"].append(did)
    m = re.search(r"(\d{4}\.\d{5})", q)
    if m:
        ent["doc_ids"].append(m.group(1))
    ent["doc_ids"] = list(dict.fromkeys(ent["doc_ids"]))

    # 指代消解（会话上下文）
    has_ref = bool(re.search(STRONG_REF_PATTERN + r"|它", q))
    if has_ref and session.get("doc_ids"):
        ent["doc_ids"] = list(dict.fromkeys(session["doc_ids"] + ent["doc_ids"]))

    # 属性实体
    for a in AUTHORS:
        if a.lower() in q.lower():
            ent["author"] = a
            break
    for v in VENUES:
        if v.lower() in q.lower():
            ent["venue"] = v
            break
    for w, code in LANG_WORDS.items():
        if w in q:
            ent["language"] = code
            break
    for w, code in TYPE_WORDS.items():
        if w in q:
            ent["doc_type"] = code
            break
    if "翻译版" in q or "翻译的" in q:
        ent["exclude"].setdefault("translated", True)
    return ent


def needs_clarification(q: str, ent: Dict, session: Dict) -> Optional[str]:
    """指代不明 / 真歧义 → 返回澄清原因
    强指代（上面/刚才提到）→ 必须澄清；弱指代（那篇）+ 无定位信息 → 澄清；弱指代 + 有信息 → 不澄清
    """
    has_ref = bool(re.search(STRONG_REF_PATTERN + r"|它", q))
    has_ctx_ref = bool(re.search(r"上面|刚才|之前说的|前面提到", q))
    if has_ref and not ent["doc_ids"]:
        has_info = (
            bool(re.search(r"(?:19|20)\d{2}|(?<!\d)\d{2}年", q))
            or any(w in q for w in TOPIC_WORDS)
            or any(a in q for a in AUTHORS)
            or any(v.lower() in q.lower() for v in VENUES)
        )
        no_identifier = not re.search(r"\d{4}\.\d{5}|\.pdf|S\d{11}|-00222", q)
        if (has_ctx_ref or not has_info) and no_identifier:
            return "指代不明：需要先指定文档（如文件名或上一篇提到的文档）"
    if "引用格式" in q or "参考文献格式" in q:
        return "「引用格式」歧义：是要查某篇论文的参考文献列表，还是要引用书写格式？"
    return None


# ================= 意图规则表（声明式） =================

def _has_semantic(q):
    return bool(re.search(SEMANTIC_PATTERN, q)) or any(w in q for w in TOPIC_WORDS)


def _has_attr(ent):
    return bool(ent.get("author") or ent.get("venue") or ent.get("language") or ent.get("doc_type"))


def _has_enum(q):
    return any(w in q for w in ENUM_WORDS)


def _has_stat(q):
    return any(w in q for w in STAT_WORDS)


def _has_strong_ref(q):
    return bool(re.search(STRONG_REF_PATTERN, q))


def _meta_requested(q):
    return any(w in q for w in META_WORDS) and not re.search(CONTENT_WORD_PATTERN, q)


def _content_requested(q):
    return bool(re.search(CONTENT_WORD_PATTERN, q))


def _has_calc(q):
    return any(w in q for w in CALC_WORDS) or bool(re.search(r"翻[了几]?[倍番]", q))


def _has_collection_scope(q):
    """当前/既有文档集合范围，不把无范围的“有没有关于X”误判成 hybrid。"""
    return bool(re.search(
        r"(?:这|那|上述|前面|刚才)?(?:\d+|[一二两三四五六七八九十]+)?篇(?:文献|论文)?(?:里|中)|"
        r"(?:这些|那些|上述|前述|当前)(?:文献|论文|文章)(?:里|中)", q))


# 规则表：(名称, 判定函数(q, ent, time) -> bool, 意图, 说明)
INTENT_RULES = [
    # 数值计算（强信号优先：计算不是检索）
    ("calc_strong", lambda q, e, t: _has_calc(q),
     "numerical_calculation", "数值计算（增长率/差值/占比/倍数）"),
    ("exclude_citation", lambda q, e, t: bool(re.search(r"(不要|除了|排除|不含|别|剔除|去掉).{0,6}(引用|参考文献)", q)),
     "inventory", "排除语境：盘点库内文档"),
    ("citation", lambda q, e, t: any(w in q.lower() for w in CITATION_WORDS) and "引用格式" not in q,
     "citation_query", "参考文献查询"),
    ("compare_docs", lambda q, e, t: any(w in q for w in COMPARE_WORDS)
     and (bool(re.search(r"论文|文献|那篇|这篇|两篇|文件", q)) or len(e.get("doc_ids", [])) >= 1),
     "cross_doc_synthesis", "文档对比"),
    ("compare_concepts", lambda q, e, t: any(w in q for w in COMPARE_WORDS),
     "semantic_retrieval", "概念对比→语义"),
    ("metadata_doc", lambda q, e, t: (e.get("doc_ids") or _has_strong_ref(q)) and _meta_requested(q),
     "metadata_query", "定位+元数据词"),
    ("doc_qa_content", lambda q, e, t: (e.get("doc_ids") or _has_strong_ref(q) or e.get("author"))
     and (_content_requested(q) or (e.get("author") and _has_strong_ref(q))),
     "doc_qa", "定位+内容索取"),
    ("doc_qa_plain", lambda q, e, t: (e.get("doc_ids") or _has_strong_ref(q))
     and not t.op and not re.search(FIND_VERB_PATTERN, q),
     "doc_qa", "定位+无年份"),
    ("scope_semantic", lambda q, e, t: _has_collection_scope(q) and _has_semantic(q),
     "hybrid", "已有文档集合+语义条件"),
    ("year_semantic", lambda q, e, t: t.op is not None and _has_semantic(q),
     "hybrid", "年份+语义"),
    ("year_stat", lambda q, e, t: t.op is not None and _has_stat(q),
     "inventory", "年份+统计"),
    ("year_enum", lambda q, e, t: t.op is not None and _has_enum(q),
     "attribute_filter", "年份+枚举"),
    ("year_plain", lambda q, e, t: t.op is not None,
     "attribute_filter", "纯年份"),
    ("enum_stat", lambda q, e, t: _has_enum(q) and _has_stat(q),
     "inventory", "枚举+统计"),
    ("enum_attr", lambda q, e, t: _has_enum(q) and (_has_attr(e) or e.get("exclude") or t.sort),
     "attribute_filter", "枚举+属性"),
    ("enum_semantic", lambda q, e, t: _has_enum(q) and _has_semantic(q),
     "hybrid", "枚举+语义"),
    ("enum_plain", lambda q, e, t: _has_enum(q),
     "inventory", "纯枚举"),
    ("exclude_only", lambda q, e, t: bool(e.get("exclude")),
     "attribute_filter", "纯排除（排除翻译版/某作者/某年等）"),
    ("attr_plain", lambda q, e, t: _has_attr(e) and not _has_semantic(q),
     "attribute_filter", "纯属性"),
]

# 1.0 只保留给无歧义的终止类判定。宽泛规则使用分级置信度，供 Router 决定
# 是直接采纳还是继续走 kNN/LLM；这不会改变规则命中的意图标签。
_RULE_CONFIDENCE = {
    "calc_strong": 0.99, "exclude_citation": 0.99, "citation": 0.98,
    "compare_docs": 0.98, "compare_concepts": 0.96,
    "metadata_doc": 0.99, "doc_qa_content": 0.99, "doc_qa_plain": 0.97,
    "scope_semantic": 0.98,
    "year_semantic": 0.98, "year_stat": 0.98, "year_enum": 0.97,
    "year_plain": 0.95, "enum_stat": 0.98, "enum_attr": 0.97,
    "enum_semantic": 0.97, "enum_plain": 0.94, "exclude_only": 0.98,
    "attr_plain": 0.95,
}


def _intent_rule_confidence(name: str, q: str, time_expr: TimeExpr,
                            time_ranges: List[Dict]) -> float:
    confidence = _RULE_CONFIDENCE.get(name, 0.95)
    precise_time = time_expr.granularity in ("day", "hour", "minute")
    compound = len(time_ranges) > 1 or len(q) > 80 or len(re.findall(r"[,，;；]|以及|同时", q)) >= 2
    # “有时间 + 所有/哪些”能确定过滤方向，但复杂事件查询未必属于文献属性筛选，
    # 因此保留为候选并允许模型层复核，而不是用 1.0 截断。
    if name in ("year_enum", "year_plain") and (precise_time or compound):
        return 0.88 if name == "year_enum" else 0.82
    return confidence


# ================= 总入口（纯函数） =================

def analyze(query: str, session: Dict = None, current_year: int = 2026,
            debug: bool = False) -> StructuredQuery:
    """意图识别 + 问题结构化。纯函数，无外部依赖。"""
    from cleaner import clean as clean_text
    cr = clean_text(query)
    sq = StructuredQuery(query=cr.cleaned)
    q = sq.query
    for name, before, after in cr.steps:
        sq.steps.append(ParseStep(f"clean_{name}", f"{before} → {after}"))

    def step(name, detail):
        sq.steps.append(ParseStep(name, detail))

    # 1. 无效输入
    if is_invalid(q):
        sq.intent, sq.source = "invalid", "invalid"
        sq.confidence = 1.0
        step("invalid", "输入为空或纯符号")
        return sq

    # 2. 非知识库任务（操作/闲聊，不依赖上下文）
    if is_non_kb(q):
        sq.intent, sq.source = "non_kb", "non_kb"
        sq.confidence = 1.0
        step("non_kb", f"命中非KB模式: {q}")
        return sq

    # 3. 实体 + 时间解析
    sq.entities = extract_entities(q, session)
    step("entities", str(sq.entities))
    parsed_times = parse_year_month_matrix(q) or parse_time_ranges(q, current_year)
    te = parsed_times[0] if parsed_times else TimeExpr()
    sq.time = te.to_dict()
    step("time", str(te.to_dict()) if te.to_dict() else "无时间表达")

    # 3.4 多时间段：保留每个区间的原粒度与时区；兼容旧的离散年份合并视图。
    if len(parsed_times) >= 2:
        sq.time_ranges = [t.to_dict() for t in parsed_times]
        step("time_ranges", f"识别 {len(parsed_times)} 个独立时间表达式")

    # 旧能力保留：2014,2018,2022 → exact 列表 + 主 time 合并区间
    multi = [] if parse_year_month_matrix(q) else parse_multi_years(q)
    if len(multi) >= 2:
        years = sorted(int(t.exact) for t in multi)
        sq.time_ranges = [t.to_dict() for t in multi]
        sq.time = {"op": "between", "from": str(years[0]), "to": str(years[-1]),
                   "granularity": "year", "raw": multi[0].raw}
        step("multi_years", f"{len(multi)} 个时间点 {years[0]}~{years[-1]}")

    # 3.5 时间非法/模糊 → 澄清；检查所有区间，不能只看第一段。
    invalid_time = next((t for t in parsed_times if t.invalid_reason), None)
    if invalid_time:
        sq.intent, sq.source = "clarification", "clarification"
        sq.confidence = 1.0
        sq.clarification = f"时间表达「{invalid_time.raw}」无效：{invalid_time.invalid_reason}"
        step("invalid_time", sq.clarification)
        return sq
    fuzzy_time = next((t for t in parsed_times if t.fuzzy), None)
    if fuzzy_time:
        sq.intent, sq.source = "clarification", "clarification"
        sq.confidence = 1.0
        sq.clarification = f"时间表达模糊「{fuzzy_time.raw}」，请明确具体年份"
        step("fuzzy_time", sq.clarification)
        return sq

    # 4. 澄清判定（指代不明 / 真歧义）
    reason = needs_clarification(q, sq.entities, session or {})
    if reason:
        sq.intent, sq.source = "clarification", "clarification"
        sq.confidence = 1.0
        sq.clarification = reason
        step("clarification", reason)
        return sq

    # 5. 规则表判定（声明式，按优先级）
    for name, match, intent, desc in INTENT_RULES:
        try:
            if match(q, sq.entities, te):
                sq.intent, sq.source = intent, f"rule_{name}"
                sq.confidence = _intent_rule_confidence(name, q, te, sq.time_ranges)
                step(name, f"{desc}；规则置信度 {sq.confidence:.2f}")
                return sq
        except Exception as ex:
            step(name, f"规则异常: {ex}")
    sq.source = "unclassified"
    step("unclassified", "规则层未命中，下沉 kNN/LLM")
    return sq


# ================= CLI 查看入口 =================

if __name__ == "__main__":
    import sys
    import json
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "找25年之前的文献", "19年之后的文献", "23到25年的文献有哪些",
        "1990年之前的文献", "2024年关于缺失值插补的论文",
        "那篇NeurIPS论文的作者是谁", "不要参考文献里的，只要知识库中的",
        "帮我写个Python脚本", "这篇论文的方法是什么", "？",
    ]
    for t in tests:
        r = analyze(t, debug=True)
        print(f"Q: {t}")
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=1))
        print("-" * 60)
