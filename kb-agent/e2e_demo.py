# -*- coding: utf-8 -*-
"""端到端演示：路由 → 检索 → 留痕 → 反馈回写（闭环）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from router import Router
from search import Searcher
from feedback import log_route, weekly_sync
from es_client import ESClient
from embedder import Embedder

router = Router()
searcher = Searcher()
es = ESClient()
emb = Embedder()

DEMO = [
    "找25年之前的文献",
    "不要参考文献里的，只要知识库中的",
    "2024年关于缺失值插补的论文",
    "那篇NeurIPS论文的作者是谁",
    "讲讲回声状态网络的训练方法",
    "库里总共有多少篇文献",
    "帮我写个Python脚本",
]

print("=" * 70)
for q in DEMO:
    print(f"\nQ: {q}")
    for item in router.route_many(q):
        if item.route is None or not item.executable:
            continue
        route = item.route
        res = searcher.execute(item.text, route)
        print(f"  [子问题] {item.text}")
        print(f"  [路由] {route.intent} conf={route.confidence} src={route.source}")
        print(f"  [消息] {res.message}")
        for d in res.docs[:4]:
            if "snippets" in d:
                print(f"    · {d['doc_id'][:38]} 片段x{len(d['snippets'])}")
                for s in d["snippets"][:1]:
                    print(f"      └ {s[:70]}")
            elif "references" in d:
                print(f"    · {d['doc_id'][:38]} 引用列表 {len(d['references'])} 字符")
            else:
                print(f"    · {str(d.get('doc_id') or d.get('filename'))[:38]} year={d.get('year')} "
                      f"snip={str(d.get('snippet'))[:60]}")
        log_route(es, emb, item.text, route, hits=len(res.docs), feedback="correct")
print("\n" + "=" * 70)
print("反馈回写（示例库自增长）:")
print(weekly_sync(es, emb))
print(f"intent_examples 现有: {es.count('intent_examples')}")
print(f"query_logs 现有: {es.count('query_logs')}")
