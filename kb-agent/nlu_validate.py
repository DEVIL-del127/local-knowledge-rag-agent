# -*- coding: utf-8 -*-
"""NLU 验证工具：输入问题 → 清洗 → 时间解析 → 实体提取 → 意图判定 → 结构化呈现

纯函数实现（仅依赖 nlu.py），不加载模型、不连 ES——任何环境秒开。
用途：验证「模型对我的问题理解得对不对」。

用法：
  python nlu_validate.py                 # 交互模式：输入问题看结构化
  python nlu_validate.py -b 问题文件.txt  # 批量：每行一个问题，输出报告文件
"""
import sys
import os
import json
import re
import argparse
from datetime import datetime
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from nlu import analyze, parse_time, extract_entities, clean

BANNER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NLU 验证：清洗 → 时间 → 实体 → 意图（纯解析，秒开）
  输入问题查看结构化结果；[回车=对 x=错 s=跳过 q=退出]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# 标注文件（追加写，防丢；对/错 → 训练数据闭环）
ANNOTATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "annotations.jsonl")

# ---------- 标注落盘 / 统计（纯函数，可单测） ----------

_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_WINDOW_UNITS = {"个月": "month", "年": "year", "周": "week", "天": "day"}


def _parse_window(s: str):
    """期望窗口解析："6个月"→(6,"month") / "半年"→(6,"month") / 无效→None"""
    if not s or not s.strip():
        return None
    s = s.strip()
    if s == "半年":
        return (6, "month")
    m = re.search(r"(\d+|[一二两三四五六七八九十]+)\s*(个月|年|周|天)", s)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    n = int(num) if num.isdigit() else _CN_NUM.get(num)
    if n is None:
        return None
    return (n, _WINDOW_UNITS[unit])


def _save_annotation(rec: dict, path: str = None):
    """追加写一条标注（jsonl）。目录自动建，坏行不影响后续。"""
    path = path or ANNOTATION_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_annotations(path: str = None) -> list:
    """读全部标注；文件不存在→[]；坏行跳过。"""
    path = path or ANNOTATION_PATH
    if not os.path.exists(path):
        return []
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return recs


def _annotations_stats(path: str = None) -> dict:
    """标注统计：total / labels 分布 / accuracy（correct/(correct+wrong)）"""
    recs = _load_annotations(path)
    labels = Counter(r.get("label", "?") for r in recs)
    c, w = labels.get("correct", 0), labels.get("wrong", 0)
    return {"total": len(recs), "labels": dict(labels),
            "accuracy": (c / (c + w)) if (c + w) else 0.0}


def print_stats(path: str = None):
    st = _annotations_stats(path)
    lab = "  ".join(f"{k}={v}" for k, v in sorted(st["labels"].items()))
    print(f"\n标注统计: 共 {st['total']} 条 | {lab} | 准确率 {st['accuracy']*100:.1f}%")


def _record_annotation(q: str, label: str, sq=None,
                       expected_intent: str = "", note: str = "",
                       window: tuple = None, path: str = None) -> dict:
    """组装标注记录并落盘。返回记录（含 ts）。"""
    if sq is None:
        sq = analyze(q)
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "query": q,
        "cleaned": sq.query,
        "intent": sq.intent,
        "source": sq.source,
        "label": label,
        "time": sq.time,
        "entities": sq.entities,
    }
    if expected_intent:
        rec["expected_intent"] = expected_intent
    if note:
        rec["note"] = note
    if window and sq.time.get("fuzzy_word"):
        rec["time_expected"] = {"fuzzy_word": sq.time["fuzzy_word"],
                                "window": window[0], "unit": window[1]}
    _save_annotation(rec, path)
    return rec


def fmt_time(t: dict) -> str:
    if not t:
        return "（无时间表达）"
    if t.get("invalid_reason"):
        return f"无效时间「{t.get('raw')}」：{t['invalid_reason']}"
    if t.get("fuzzy"):
        return f"高度模糊「{t.get('raw')}」→ 需澄清"
    op = t.get("op")
    f, to, ex = t.get("from"), t.get("to"), t.get("exact")
    if op == "between":
        s = f"{f} ~ {to}"
    elif op == "lte":
        s = f"≤ {to}"
    elif op == "gte":
        s = f"≥ {f}"
    elif op == "lt":
        s = f"< {to}"
    elif op == "gt":
        s = f"> {f}"
    elif op == "exact":
        s = f"= {ex}"
    elif t.get("sort") == "year_desc":
        s = "按年份倒序"
    elif t.get("sort") == "latest_before_anchor":
        s = "锚点之前按日期倒序取最近一项"
    else:
        s = str(op or "")
    if t.get("sort") == "year_desc" and op:
        s += "，按年份倒序"
    g = t.get("granularity", "year")
    label = {"year": "年", "month": "月", "day": "日", "hour": "时", "minute": "分"}.get(g, g)
    s += f"（{label}粒度）"
    if t.get("fuzzy_word"):
        s += f"【模糊词「{t['fuzzy_word']}」按默认窗口】"
    if t.get("timezone"):
        s += f"【时区 {t['timezone']}】"
    return f"「{t.get('raw')}」→ {s}"


def fmt_time_ranges(trs: list) -> str:
    """多时间段显示：T1: 2020~2021；T2: 2024~2026"""
    if not trs:
        return ""
    parts = []
    for i, t in enumerate(trs, 1):
        s = fmt_time(t)
        if len(trs) > 1:
            s = f"T{i}: {s}"
        parts.append(s)
    return " ｜ ".join(parts)


def fmt_entities(e: dict) -> str:
    items = []
    if e.get("author"):
        items.append(f"作者: {e['author']}")
    if e.get("venue"):
        items.append(f"期刊/会议: {e['venue']}")
    if e.get("language"):
        items.append(f"语言: {e['language']}")
    if e.get("doc_type"):
        items.append(f"类型: {e['doc_type']}")
    if e.get("doc_ids"):
        items.append(f"文档: {', '.join(e['doc_ids'])}")
    ex = e.get("exclude") or {}
    if ex:
        parts = []
        if ex.get("authors"): parts.append("作者:" + ",".join(ex["authors"]))
        if ex.get("years"): parts.append("年份:" + ",".join(map(str, ex["years"])))
        if ex.get("languages"): parts.append("语言:" + ",".join(ex["languages"]))
        if ex.get("doc_types"): parts.append("类型:" + ",".join(ex["doc_types"]))
        if ex.get("doc_ids"): parts.append("文档:" + ",".join(ex["doc_ids"]))
        if ex.get("translated"): parts.append("翻译版")
        items.append("排除: " + "、".join(parts))
    return "\n        ".join(items) if items else "（无）"


def render(q: str, session: dict = None, emb=None, llm=None) -> str:
    """先拆分复合问题 → 每个子问题独立结构化（含继承标注）
    emb 传入时启用向量语义断层切分；llm 传入时复杂问题走 LLM 结构解析"""
    from cleaner import clean as clean_text
    from splitter import split
    q_clean = clean_text(q).cleaned  # 清洗先行（引导壳/噪声），拆分与解析共享
    # LLM 通道：复杂信号命中 → LLM 结构解析（主视图）+ 规则视图（调试）
    if llm is not None:
        from llm_parser import need_llm
        if need_llm(q_clean):
            r = llm.parse(q_clean)
            if r:
                return render_llm(q, r, q_clean)
    subs = split(q_clean, emb=emb, mode="semantic" if emb else "rule", analyze_fn=analyze)
    if len(subs) > 1:
        return render_multi(q_clean, subs, session)
    return render_single(q_clean, session)


def render_llm(q: str, r, q_clean: str = None) -> str:
    """LLM 结构解析主视图（复杂问题）"""
    lines = []
    lines.append(f"问题: {q}")
    if q_clean and q_clean != q.strip():
        lines.append(f"清洗后: {q_clean}")
    lines.append("-" * 56)
    lines.append(f"[LLM 结构解析]（置信度 {r.confidence:.2f}）")
    lines.append(f"┌─ 主问题 : {r.main_query}")
    tr = r.time_range or {}
    if tr:
        f, t = tr.get("from"), tr.get("to")
        span = f"{f} ~ {t or '至今'}" if f or t else "（事件锚点，见备注）"
        lines.append(f"│  时间范围: {span}（{tr.get('granularity', '-')}粒度）")
    if r.constraints:
        for c in r.constraints:
            lines.append(f"│  约束    : [{c.get('type')}] {c.get('value')}")
    if r.output:
        lines.append(f"│  输出要求: {r.output}")
    if r.note:
        lines.append(f"│  备注    : {r.note}")
    lines.append(f"└─ 说明    : 复杂问题走 LLM 通道；规则层视图见下方调试区")
    # 规则层调试区
    from cleaner import clean as clean_text
    from splitter import split
    subs = split(clean_text(q_clean or q).cleaned, analyze_fn=analyze)
    lines.append("")
    lines.append("── 规则层调试（未参与最终解析） ──")
    for i, sub in enumerate(subs, 1):
        role = sub.role
        lines.append(f"  [{role}] {sub.text} time={fmt_time(sub.time)}")
    return "\n".join(lines)


def render_single(q: str, session: dict = None) -> str:
    sq = analyze(q, session or {}, debug=True)
    lines = []
    lines.append(f"问题: {q}")
    lines.append("-" * 56)

    # ① 清洗
    cleaned = sq.query
    flag = "" if cleaned == q.strip() else f"（已从「{q.strip()}」清洗）"
    lines.append(f"① 清洗后 : {cleaned} {flag}")

    # ② 时间
    lines.append(f"② 时间   : {fmt_time(sq.time)}")
    if sq.time_ranges:
        lines.append(f"② 多时间段: {fmt_time_ranges(sq.time_ranges)}")

    # ③ 实体
    lines.append(f"③ 实体   : {fmt_entities(sq.entities)}")

    # ④ 意图
    if sq.intent:
        src = sq.source
        conf = f"置信度 {sq.confidence:.2f}"
        lines.append(f"④ 意图   : {sq.intent}（{src}，{conf}）")
    else:
        lines.append("④ 意图   : （规则层未命中 → 将由 kNN 模型层判定）")
    if sq.clarification:
        lines.append(f"⑤ 澄清   : {sq.clarification}")

    # ⑥ 轨迹
    steps = " → ".join(f"{s.name}({s.detail[:40]})" for s in sq.steps)
    lines.append(f"⑥ 轨迹   : {steps}")
    return "\n".join(lines)


def _route_source_label(source: str) -> str:
    """Translate Router sources into labels suitable for an intent-only view."""
    if source == "clarification_subquery_limit":
        return "子问题数量保护（未调用模型）"
    if source == "workflow_rule":
        return "依赖工作流规则（未调用 kNN/LLM）"
    if source.endswith("_candidate_fallback"):
        return "L0 低置信候选（模型链无更可靠结果后回落）"
    if source.startswith("rule_") or source in {"invalid", "non_kb", "clarification"}:
        return "L0 规则解析"
    if source == "L3_context":
        return "上下文继承"
    if source == "L3":
        return "L3 向量 kNN（高置信）"
    if source == "L3_low":
        return "L3 向量 kNN（低置信，LLM 未覆盖）"
    if source == "L2_llm":
        return "L2 本地 Ollama LLM"
    if source == "fallback":
        return "最终兜底：semantic_retrieval"
    return source or "未知来源"


def render_router(q: str, router, session: dict = None) -> str:
    """展示拆分后的完整意图路由；附属子句不单独调用模型或检索。"""
    lines = [render(q, session)]
    lines.append("")
    lines.append("── 完整意图路由（不执行检索） ──")
    try:
        plan = router.route_many(q, session)
    except Exception as exc:
        lines.append(f"路由未完成: {type(exc).__name__}: {exc}")
        lines.append("请检查 Elasticsearch、intent_examples 索引和本地向量模型是否可用。")
        return "\n".join(lines)

    mains = [item for item in plan if item.route is not None]
    executable = sum(item.route is not None and item.executable for item in plan)
    lines.append(f"执行计划 : {len(mains)} 个主步骤，{len(plan) - len(mains)} 个附属子句，{executable} 个可直接检索")
    role_cn = {"main": "主问题", "context": "上下文", "time": "时间限定", "constraint": "约束", "output": "输出要求"}
    for item in plan:
        lines.append("")
        lines.append(f"[{role_cn.get(item.role, item.role)} {item.index}] {item.text}")
        if item.route is None:
            if item.parameters:
                lines.append(f"结构参数 : {item.parameters}")
            if item.role == "output":
                lines.append(f"处理方式 : {item.blocked_reason}，不调用 kNN/LLM，不单独检索")
            else:
                lines.append("处理方式 : 作为主问题的继承条件，不调用 kNN/LLM，不单独检索")
            continue
        route = item.route
        lines.append(f"步骤类型 : {item.step_type}")
        if item.depends_on:
            lines.append(f"依赖步骤 : {item.depends_on}")
        if item.requires_tools:
            lines.append(f"所需工具 : {', '.join(item.requires_tools)}")
        if item.parameters:
            lines.append(f"结构参数 : {item.parameters}")
        if item.effective_query and item.effective_query != item.text:
            lines.append(f"路由输入 : {item.effective_query}（已附加继承条件）")
        lines.append(f"最终意图 : {route.intent}")
        lines.append(f"命中层级 : {_route_source_label(route.source)} ({route.source})")
        lines.append(f"置信度   : {route.confidence:.3f}")
        lines.append(f"结构化条件: {route.entities.to_dict() or '（无）'}")
        if route.clarification_needed or route.clarification_reason:
            lines.append(f"澄清信息 : {route.clarification_reason or '需要补充信息'}")
        if item.executable:
            lines.append("检索状态 : 未执行；实际检索时仅使用本主问题，不使用整段原问题")
        else:
            lines.append(f"检索状态 : 已阻止直接检索；{item.blocked_reason}")
    return "\n".join(lines)


def render_multi(q: str, subs: list, session: dict = None) -> str:
    """复合问题视图：主问题 + 附属成分（约束/输出/时间限定）区分呈现"""
    lines = []
    lines.append(f"问题: {q}")
    lines.append("-" * 56)
    mains = [s for s in subs if s.role == "main"]
    attrs = [s for s in subs if s.role != "main"]
    role_cn = {"main": "主问题", "context": "上下文", "time": "时间限定", "constraint": "约束", "output": "输出要求"}
    lines.append(f"拆分: {len(mains)} 个主问题 + {len(attrs)} 个附属成分")
    for i, sub in enumerate(subs, 1):
        lines.append(f"")
        inh = ""
        if sub.inherit_from:
            keys = "、".join(sub.inherited.keys())
            inh = f"  ⬅ 继承自{sub.inherit_from.replace('sub_','')}（{keys}）"
        lines.append(f"┌─ [{role_cn.get(sub.role, sub.role)}] {i}: {sub.text}{inh}")
        if sub.step_type != "retrieval":
            lines.append(f"│  步骤   : {sub.step_type} | 依赖 {sub.depends_on or '-'} | 工具 {sub.requires_tools or '-'}")
        if sub.parameters:
            lines.append(f"│  参数   : {sub.parameters}")
        if sub.role != "output":
            if sub.time_ranges:
                lines.append(f"│  时间   : {fmt_time_ranges(sub.time_ranges)}")
            else:
                lines.append(f"│  时间   : {fmt_time(sub.time)}")
            lines.append(f"│  实体   : {fmt_entities(sub.entities)}")
            if sub.role != "main":
                lines.append(f"│  意图   : 附属{sub.role}（不单独路由）")
            elif sub.step_type in ("data_query", "date_derivation") and not sub.executable:
                lines.append("│  意图   : agent_workflow（依赖工作流规则，不调用模型）")
            elif sub.intent:
                lines.append(f"│  意图   : {sub.intent}（{sub.source}，置信度 {sub.confidence:.2f}）")
            else:
                lines.append(f"│  意图   : （规则层未命中 → 由 kNN 模型层判定）")
        lines.append(f"└─ 子句   : {sub.text}")
    return "\n".join(lines)



def _annotate_one(q: str, session: dict, emb=None, parser=None,
                 path: str = None, no: int = None) -> str:
    """单条标注：展示解析 → 用户判定 → 落盘。返回 label 或 "quit"。"""
    print(render(q, session, emb, parser))
    tag = f"[{no}] " if no else ""
    try:
        ans = input(f"\n{tag}判定 [回车=对 x=错 s=跳过 q=退出]> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n退出标注")
        return "quit"
    if ans in ("q", "quit", "exit"):
        return "quit"
    if ans == "s":
        _record_annotation(q, "skip", path=path)
        return "skip"
    sq = analyze(q)
    expected_intent, note, window = "", "", None
    if ans == "x":
        try:
            expected_intent = input("  期望意图 [回车跳过]> ").strip()
            note = input("  备注 [回车跳过]> ").strip()
            if sq.time.get("fuzzy_word"):
                w = input(f"  含模糊时间词「{sq.time['fuzzy_word']}」，期望窗口 [如 6个月，回车跳过]> ").strip()
                window = _parse_window(w) if w else None
        except (EOFError, KeyboardInterrupt):
            pass
    label = "correct" if ans != "x" else "wrong"
    rec = _record_annotation(q, label, sq=sq, expected_intent=expected_intent,
                             note=note, window=window, path=path)
    extra = ""
    if "time_expected" in rec:
        e = rec["time_expected"]
        extra = f"（期望窗口: {e['fuzzy_word']}→{e['window']}{e['unit']}）"
    print(f"  已记录 [{label}]{extra}")
    return label


_RENDERED_OUTPUT_PREFIXES = (
    "① 清洗后", "② 时间", "② 多时间", "③ 实体", "④ 意图", "⑤ 澄清", "⑥ 轨迹",
    "最终意图", "命中层级", "置信度", "结构化条件", "澄清信息", "检索状态",
    "拆分:", "路由输入", "执行计划", "处理方式", "步骤类型", "依赖步骤", "所需工具", "结构参数",
    "┌─", "│", "└─", "──", "[完整意图路由", "[已忽略", "[主问题", "[上下文", "[时间限定", "[约束", "[输出要求",
)


def _normalize_interactive_input(text: str) -> str:
    """过滤误粘贴回 REPL 的报告行；保留并清理“问题: 原问题”这一行。"""
    q = text.strip()
    if not q or re.fullmatch(r"[-━─=]{3,}", q):
        return ""
    while re.match(r"^问题\s*[:：]", q):
        q = re.sub(r"^问题\s*[:：]\s*", "", q, count=1).strip()
    if not q or re.match(r"^\d+\]\s*", q) or q.startswith(_RENDERED_OUTPUT_PREFIXES):
        return ""
    return q


def interactive(semantic: bool = False, llm: bool = False, annotate: bool = True,
                path: str = None, router_mode: bool = False):
    print(BANNER)
    emb = None
    parser = None
    router = None
    if semantic:
        from embedder import Embedder
        emb = Embedder()
        print("[向量语义断层切分已启用]")
    if llm:
        from llm_parser import LLMParser
        from config import LLM as LLM_CFG
        try:
            parser = LLMParser(LLM_CFG)
            print(f"[LLM 结构解析已启用：复杂问题走本地 Ollama {LLM_CFG.get('model', 'LLM')}]")
        except Exception as e:
            print(f"[LLM 未启用: {e}]")
    if router_mode:
        from router import Router
        router = Router()
        if annotate:
            print("[完整路由模式: 仅展示，不写规则层标注]")
            annotate = False
        print("[完整意图路由已启用: L0 -> 上下文 -> 向量 kNN -> 本地 LLM -> fallback]")
    if annotate:
        print(f"[标注模式: 回车=对 x=错 s=跳过 q=退出 → {path or ANNOTATION_PATH}]")
    session = {"doc_ids": []}
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not q:
            continue
        q = _normalize_interactive_input(q)
        if not q:
            print("[已忽略粘贴的程序输出行]")
            continue
        if q in ("exit", "/exit"):
            print("再见")
            break
        if q in ("stats", "/stats"):
            print_stats(path)
            continue
        if router is not None:
            print()
            print(render_router(q, router, session))
        elif annotate:
            label = _annotate_one(q, session, emb, parser, path)
            if label == "quit":
                print("再见")
                break
        else:
            print()
            print(render(q, session, emb, parser))


def batch(path: str, semantic: bool = False, llm: bool = False,
          annotate: bool = False, ann_path: str = None, router_mode: bool = False):
    emb = None
    parser = None
    router = None
    if semantic:
        from embedder import Embedder
        emb = Embedder()
    if llm:
        from llm_parser import LLMParser
        from config import LLM as LLM_CFG
        try:
            parser = LLMParser(LLM_CFG)
        except Exception:
            parser = None
    if router_mode:
        from router import Router
        router = Router()
        if annotate:
            raise ValueError("--router 与 --annotate 不能同时使用；完整路由模式不写规则层标注")
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    # ---- 标注模式：逐条展示 + 判定 + 落盘 ----
    if annotate:
        print(f"批量标注 {len(lines)} 条 → {ann_path or ANNOTATION_PATH}（q=提前退出）")
        session = {"doc_ids": []}
        for i, q in enumerate(lines, 1):
            print(f"\n{'='*56}\n[{i}/{len(lines)}]")
            if _annotate_one(q, session, emb, parser, ann_path, no=i) == "quit":
                break
        print_stats(ann_path)
        return

    rows = []
    for q in lines:
        if router is not None:
            for item in router.route_many(q):
                route = item.route
                rows.append({
                    "query": q, "subquery": item.text, "role": item.role,
                    "cleaned": item.text,
                    "intent": route.intent if route else None,
                    "source": route.source if route else "attached_condition",
                    "confidence": route.confidence if route else 0.0,
                    "time": item.time,
                    "entities": route.entities.to_dict() if route else item.entities,
                    "clarification": route.clarification_reason if route else "",
                })
        else:
            sq = analyze(q, debug=True)
            rows.append({
                "query": q, "subquery": q, "role": "main", "cleaned": sq.query,
                "intent": sq.intent, "source": sq.source, "confidence": sq.confidence,
                "time": sq.time, "entities": sq.entities, "clarification": sq.clarification,
            })
    # 报告
    out = f"F:/A_ShiXi/Project/STUDY/kb-agent/data/nlu_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# NLU 验证报告\n\n")
        f.write(f"共 {len(rows)} 条\n\n")
        f.write("| # | 问题 | 意图 | 规则来源 | 时间解析 | 实体 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            ent = r["entities"]
            ent_s = " ".join(k for k, v in ent.items() if v) or "-"
            shown_query = r['subquery'] if r['subquery'] == r['query'] else f"[{r['role']}] {r['subquery']}"
            f.write(f"| {i} | {shown_query} | {r['intent'] or '附属条件'} | {r['source']} "
                    f"| {fmt_time(r['time'])} | {ent_s} |\n")
        f.write("\n## 明细\n\n")
        for i, r in enumerate(rows, 1):
            f.write(f"### {i}. [{r['role']}] {r['subquery']}\n\n")
            f.write(f"- 清洗: {r['cleaned']}\n")
            f.write(f"- 意图: {r['intent'] or '（规则层未命中→kNN）'} ({r['source']}, 置信度 {r['confidence']:.3f})\n")
            f.write(f"- 时间: {fmt_time(r['time'])}\n")
            f.write(f"- 实体: {fmt_entities(r['entities'])}\n")
            if r["clarification"]:
                f.write(f"- 澄清: {r['clarification']}\n")
            f.write("\n")
    print(f"报告已生成: {out}")
    print(f"共 {len(rows)} 条，明细如下：\n")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. [{r['role']}] {r['subquery']}\n    → {r['intent'] or '附属条件'} ({r['source']}) | "
              f"时间: {fmt_time(r['time'])} | 澄清: {r['clarification'] or '-'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NLU 验证工具（含对/错标注闭环）")
    ap.add_argument("-b", "--batch", help="批量验证：文件路径（每行一个问题）")
    ap.add_argument("-s", "--semantic", action="store_true",
                    help="启用向量语义断层切分（需加载 bge 模型，默认纯规则秒开）")
    ap.add_argument("-l", "--llm", action="store_true",
                    help="启用 LLM 结构解析（复杂问题走本地 DeepSeek）")
    ap.add_argument("-a", "--annotate", action="store_true",
                    help="标注模式：逐条判定对/错并落盘（批量时用；交互默认开启）")
    ap.add_argument("--no-annotate", action="store_true",
                    help="交互模式关闭标注（只看解析不落盘）")
    ap.add_argument("--ann", dest="ann_path", help="标注文件路径（默认 data/annotations.jsonl）")
    ap.add_argument("--stats", action="store_true", help="查看标注统计")
    ap.add_argument("--router", action="store_true",
                    help="完整意图路由但不检索：L0→上下文→向量kNN→本地LLM→fallback")
    args = ap.parse_args()
    if args.stats:
        print_stats(args.ann_path)
    elif args.batch:
        batch(args.batch, args.semantic, args.llm, args.annotate, args.ann_path, args.router)
    else:
        interactive(args.semantic, args.llm, not args.no_annotate, args.ann_path, args.router)
