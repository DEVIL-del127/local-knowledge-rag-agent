# -*- coding: utf-8 -*-
"""kb-agent 统一 CLI 入口
用法:
  python query.py "找25年之前的文献"                 # 文本输出
  python query.py "这篇用了什么方法" --doc 2210.02040  # 带上文文档（指代消解）
  python query.py "..." --json                        # 结构化输出（供 agent 解析）
  python query.py --regression                        # 跑 95 例回归
"""
import sys
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from router import Router
from search import Searcher
from feedback import log_route
from es_client import ESClient
from embedder import Embedder


def format_result(q, route, res):
    lines = []
    lines.append(f"[意图] {route.intent} (conf={route.confidence:.2f}, src={route.source})")
    if route.clarification_reason:
        lines.append(f"[澄清] {route.clarification_reason}")
    if res.message:
        lines.append(f"[结果] {res.message}")
    if res.stats:
        for k, v in res.stats.items():
            lines.append(f"  {k}: {v}")
    for d in res.docs:
        if "snippets" in d:
            lines.append(f"  · {d['doc_id'][:40]} (year={d.get('year')})")
            for s in d["snippets"][:2]:
                lines.append(f"      {s[:100]}")
        elif "references" in d:
            lines.append(f"  · {d['doc_id'][:40]} 参考文献 {len(d['references'])} 字符")
            lines.append(f"      {d['references'][:120]}")
        else:
            ident = d.get("doc_id") or d.get("filename") or ""
            lines.append(f"  · {ident[:44]} year={d.get('year')} type={d.get('doc_type')} "
                         f"venue={d.get('venue')} lang={d.get('language')} tr={d.get('is_translated')}")
            if d.get("snippet"):
                lines.append(f"      {d['snippet'][:100]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="查询问题")
    ap.add_argument("--doc", action="append", default=[], help="会话上下文文档 id（可多次）")
    ap.add_argument("--nlu", action="store_true", help="只查看结构化结果（不检索）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--regression", action="store_true", help="跑回归测试")
    args = ap.parse_args()

    if args.regression:
        import test_regression
        test_regression.main()
        return

    if not args.query:
        print("用法: python query.py \"问题\" [--doc 文档id] [--nlu] [--json]")
        return

    # NLU 查看模式：只跑结构化，不检索
    if args.nlu:
        import json as _json
        from nlu import analyze
        from cleaner import clean as clean_text
        from splitter import split
        cleaned = clean_text(args.query).cleaned
        subs = split(cleaned, analyze_fn=analyze)
        out = {"query": args.query, "cleaned": cleaned,
               "subqueries": [sub.to_dict() for sub in subs]}
        print(_json.dumps(out, ensure_ascii=False, indent=2))
        return

    router = Router()
    searcher = Searcher()
    es = ESClient()
    emb = Embedder()

    session = {"doc_ids": args.doc} if args.doc else {}
    plan = router.route_many(args.query, session)
    executed = []
    blocked = []
    for item in plan:
        if item.route is None:
            continue
        if not item.executable:
            blocked.append(item)
            continue
        # 只用当前主问题做 BM25/向量检索；继承条件已经在 route.entities 中。
        res = searcher.execute(item.text, item.route)
        executed.append((item, res))
        try:
            log_route(es, emb, item.text, item.route, hits=len(res.docs))
        except Exception:
            pass

    if args.json:
        if len(plan) == 1 and len(executed) == 1:
            item, res = executed[0]
            out = {"query": args.query, "route": item.route.to_dict(),
                   "result": res.to_dict(), "clarification": res.clarification}
        else:
            result_by_index = {item.index: res for item, res in executed}
            out = {"query": args.query, "subqueries": []}
            for item in plan:
                record = item.to_dict()
                res = result_by_index.get(item.index)
                record["result"] = res.to_dict() if res else None
                out["subqueries"].append(record)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for item in blocked:
            tools = ", ".join(item.requires_tools) or "Planner"
            print(f"[待 Agent 执行] {item.text}")
            print(f"  步骤={item.step_type} 依赖={item.depends_on or '-'} 工具={tools}")
            print(f"  原因={item.blocked_reason}")
        if len(executed) > 1:
            print(f"[拆分] {len(executed)} 个主问题分别检索")
        for pos, (item, res) in enumerate(executed, 1):
            if len(executed) > 1:
                print(f"\n[子问题 {pos}] {item.text}")
            print(format_result(item.text, item.route, res))


if __name__ == "__main__":
    main()
