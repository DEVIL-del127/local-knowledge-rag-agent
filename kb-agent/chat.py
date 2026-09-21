# -*- coding: utf-8 -*-
"""交互式测试入口：自己输入问题，看 路由 → 结构化 → 检索结果

命令：
  直接输入问题       完整流程（路由 + 结构化 + 检索）
  /nlu <问题>        只看结构化（不检索）
  /session           查看会话状态（指代消解用的上文文档）
  /clear             清空会话
  /exit 或 exit      退出

示例：
  > 找25年之前的文献
  > 这篇论文用了什么方法        ← 自动带上文第一篇文档（指代消解）
"""
import sys

# 终端统一 UTF-8（PTY/Windows Terminal 环境）；老 cmd 若中文乱码请先 chcp 65001
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from router import Router
from search import Searcher
from nlu import analyze
from es_client import ESClient
from embedder import Embedder
from feedback import log_route
from cleaner import clean as clean_text
from splitter import split

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  kb-agent 交互式测试（意图路由 → 结构化 → 三路召回）        ║
║  输入问题测试；/nlu <问题> 只看结构化；/session /clear /exit ║
╚══════════════════════════════════════════════════════════╝
"""


def fmt_time(t: dict) -> str:
    if not t:
        return "无时间表达"
    if t.get("fuzzy"):
        return f"模糊({t.get('raw')})→需澄清"
    parts = []
    op = t.get("op")
    if op == "between":
        parts.append(f"{t['from']}~{t['to']}")
    elif op == "lte":
        parts.append(f"≤{t['to']}")
    elif op == "gte":
        parts.append(f"≥{t['from']}")
    elif op == "lt":
        parts.append(f"<{t['to']}")
    elif op == "gt":
        parts.append(f">{t['from']}")
    elif op == "exact":
        parts.append(f"={t['exact']}")
    if t.get("sort"):
        parts.append(f"sort:{t['sort']}")
    return " ".join(parts)


def fmt_entities(e: dict) -> str:
    parts = []
    if e.get("author"):
        parts.append(f"作者:{e['author']}")
    if e.get("venue"):
        parts.append(f"期刊:{e['venue']}")
    if e.get("language"):
        parts.append(f"语言:{e['language']}")
    if e.get("doc_type"):
        parts.append(f"类型:{e['doc_type']}")
    if e.get("doc_ids"):
        parts.append(f"文档:{','.join(e['doc_ids'])}")
    if e.get("exclude"):
        parts.append(f"排除:{e['exclude']}")
    return "；".join(parts) if parts else "-"


def show_docs(res, limit=6):
    for d in res.docs[:limit]:
        if "snippets" in d:
            print(f"  · {d['doc_id'][:42]} (year={d.get('year')})")
            for s in d["snippets"][:2]:
                print(f"      {s[:90]}")
        elif "references" in d:
            print(f"  · {d['doc_id'][:42]} 参考文献 {len(d['references'])} 字符")
            print(f"      {d['references'][:100]}")
        else:
            ident = d.get("doc_id") or d.get("filename") or ""
            print(f"  · {ident[:42]} year={d.get('year')} type={d.get('doc_type')} "
                  f"venue={str(d.get('venue'))[:14]} lang={d.get('language')} tr={d.get('is_translated')}")
            if d.get("snippet"):
                print(f"      {d['snippet'][:90]}")
    if len(res.docs) > limit:
        print(f"  … 共 {len(res.docs)} 篇（limit {limit}）")
    if not res.docs and res.message:
        print(f"  （空结果）")


def run_chat():
    print(BANNER)
    router = Router()
    searcher = Searcher()
    es = ESClient()
    emb = Embedder()
    session = {"doc_ids": []}

    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not q:
            continue
        if q in ("exit", "quit", "/exit"):
            print("再见")
            break
        if q == "/clear":
            session = {"doc_ids": []}
            print("[会话已清空]")
            continue
        if q == "/session":
            print(f"[会话] doc_ids={session['doc_ids']}")
            continue
        if q.startswith("/nlu "):
            import json
            cleaned = clean_text(q[5:]).cleaned
            subs = split(cleaned, analyze_fn=analyze)
            print(json.dumps({"query": q[5:], "subqueries": [s.to_dict() for s in subs]},
                             ensure_ascii=False, indent=1))
            continue

        print("-" * 60)
        plan = router.route_many(q, session)
        mains = [item for item in plan if item.route is not None]
        if len(mains) > 1:
            print(f"[拆分] {len(mains)} 个主问题分别检索")
        collected_doc_ids = []
        for pos, item in enumerate(mains, 1):
            route = item.route
            if len(mains) > 1:
                print(f"\n[子问题 {pos}] {item.text}")
            src = route.source if route.source.startswith("rule_") else f"{route.source}(模型层)"
            print(f"[路由] {route.intent} | {src} | conf={route.confidence:.2f}")
            if route.clarification_reason:
                print(f"[澄清] {route.clarification_reason}")
                continue
            if not item.executable:
                tools = ", ".join(item.requires_tools) or "Planner"
                print(f"[待 Agent 执行] 步骤={item.step_type} | 依赖={item.depends_on or '-'} | 工具={tools}")
                print(f"[未检索] {item.blocked_reason}")
                continue
            res = searcher.execute(item.text, route)
            print(f"[结构] 时间段={route.entities.time_ranges or '-'} 实体={route.entities.to_dict()}")
            print(f"[结果] {res.message}")
            show_docs(res)
            try:
                log_route(es, emb, item.text, route, hits=len(res.docs))
            except Exception:
                pass
            collected_doc_ids.extend(d.get("doc_id") for d in res.docs if d.get("doc_id"))
        if collected_doc_ids:
            session["doc_ids"] = list(dict.fromkeys(collected_doc_ids))[:3]


if __name__ == "__main__":
    run_chat()
