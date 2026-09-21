#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多路检索召回准确性测试(独立脚本, 与主程序功能分离)

用法:
    python tests/test_retrieval.py                # 默认测测试库(test)
    python tests/test_retrieval.py --db main      # 测正式库
    python tests/test_retrieval.py --brief        # 只出汇总指标, 不打印每条详情

测试集格式(test_data/queries.json 或 main_data/queries.json):
    [
        {"query": "误差补偿回声状态网络",
         "relevant": ["具有双储层结构的动态误差补偿回声状态网络_张昭昭.pdf"]},
        {"query": "Bayesian spatio-temporal models",
         "relevant": ["贝叶斯时空统计方法及应用进展与趋势_李俊明.pdf", "2210.02040v3.pdf"]}
    ]

指标:
    Recall@k   : 相关文档在 top-k 中的覆盖率(相关文档数/全部相关文档数)
    MRR@10     : 首个相关文档排名的倒数(0~1, 越大越好)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PDFSearchSystem  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_FILES = {
    'main': os.path.join(PROJECT_ROOT, 'tests', 'data', 'queries_main.json'),
    'test': os.path.join(PROJECT_ROOT, 'tests', 'data', 'queries.json'),
}
MODES = ('es', 'vec', 'hybrid')
TOPK_LIST = (1, 3, 5, 10)


def load_queries(db: str) -> list:
    path = QUERY_FILES[db]
    if not os.path.exists(path):
        print(f"[错误] 测试集不存在: {path}")
        print(f"请创建测试集, 格式参考: test_data/queries.example.json")
        sys.exit(1)
    with open(path, encoding='utf-8') as f:
        queries = json.load(f)
    if not queries:
        print(f"[警告] 测试集为空: {path}")
    return queries


def run_mode(system: PDFSearchSystem, mode: str, queries: list):
    """对一种模式跑全部查询, 返回 (recall指标dict, mrr, 每查询详情)"""
    recall_sums = {k: 0.0 for k in TOPK_LIST}
    mrr_sum = 0.0
    details = []

    for q in queries:
        query = q.get('query', '')
        relevant = set(q.get('relevant', []))
        if not query or not relevant:
            details.append({'query': query, 'skip': '查询或relevant为空'})
            continue

        if mode == 'es':
            hits = system.es_search(query, size=max(TOPK_LIST))
        elif mode == 'vec':
            hits = system.vec_search(query, top_k=max(TOPK_LIST))
        else:
            hits = system.hybrid_search(query, top_n=max(TOPK_LIST))

        files = [h['filename'] for h in hits]

        # Recall@k
        for k in TOPK_LIST:
            hit_count = len(set(files[:k]) & relevant)
            recall_sums[k] += hit_count / len(relevant)

        # MRR@10
        mrr = 0.0
        for i, fn in enumerate(files[:10]):
            if fn in relevant:
                mrr = 1.0 / (i + 1)
                break
        mrr_sum += mrr

        details.append({
            'query': query,
            'relevant': sorted(relevant),
            'top10': files[:10],
            'mrr': mrr,
        })

    n = len(queries)
    recall = {f'Recall@{k}': recall_sums[k] / n for k in TOPK_LIST}
    mrr = mrr_sum / n
    return recall, mrr, details


def print_report(queries, results, brief=False):
    print("\n" + "=" * 66)
    print(f"多路检索召回准确性测试")
    print(f"查询数: {len(queries)} | 相关性: 文件级 | 指标: Recall@k / MRR@10")
    print("=" * 66)

    header = f"{'模式':<8}" + "".join(f"{f'Recall@{k}':>12}" for k in TOPK_LIST) + f"{'MRR@10':>10}"
    print(header)
    print("-" * 66)
    for mode in MODES:
        recall, mrr, _ = results[mode]
        row = f"{mode:<8}" + "".join(f"{recall[f'Recall@{k}']:>12.3f}" for k in TOPK_LIST) + f"{mrr:>10.3f}"
        print(row)
    print("-" * 66)

    # 各查询详细命中情况(定位短板)
    if not brief:
        print("\n=== 各查询命中详情(top10) ===")
        for mode in MODES:
            print(f"\n--- 模式: {mode} ---")
            for d in results[mode][2]:
                if d.get('skip'):
                    print(f"  [跳过] {d['query']}: {d['skip']}")
                    continue
                hit = [f for f in d['top10'] if f in set(d['relevant'])]
                miss = [f for f in d['top10'] if f not in set(d['relevant'])]
                print(f"  查询: {d['query']} (MRR: {d['mrr']:.3f})")
                print(f"    相关: {', '.join(d['relevant'])}")
                if hit:
                    print(f"    命中: {', '.join(hit)}")
                else:
                    print(f"    命中: 无 ✗")
                if miss:
                    print(f"    误召回: {', '.join(miss[:5])}{'...' if len(miss) > 5 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description='多路检索召回准确性测试')
    parser.add_argument('--db', choices=['test', 'main'], default='test',
                        help='测哪个库: test=测试库(默认), main=正式库')
    parser.add_argument('--brief', action='store_true', help='只输出汇总指标')
    args = parser.parse_args()

    print(f"加载测试集: {QUERY_FILES[args.db]}")
    queries = load_queries(args.db)
    if not queries:
        return

    print(f"构建[{args.db}]库检索系统(连接 ES + 向量库)...")
    system = PDFSearchSystem(db=args.db)

    results = {}
    for mode in MODES:
        recall, mrr, details = run_mode(system, mode, queries)
        results[mode] = (recall, mrr, details)

    print_report(queries, results, brief=args.brief)


if __name__ == '__main__':
    main()
