#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动化召回率测试(零标注一键评测)

流程: 检查库就绪(没有则自动建库) → 从库内文档自动抽取文本块生成查询
      → 三种模式(es/vec/hybrid)分别检索 → 输出 Recall@k / MRR@10 对比

自动生成的查询来自原文片段, 相关文档=来源文档,
得分反映"检索链路上限"; 真实场景建议配合 test_data/queries.json 人工标注集。

用法:
    python tests/auto_retrieval_test.py                 # 测试库, 自动建库
    python tests/auto_retrieval_test.py --db main       # 正式库
    python tests/auto_retrieval_test.py --per-doc 3     # 每文档抽3条查询
    python tests/auto_retrieval_test.py --no-build      # 库未就绪时不自动建库
    python tests/auto_retrieval_test.py --reuse         # 复用已生成的自动测试集
    python tests/auto_retrieval_test.py --brief         # 只看汇总
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PDFSearchSystem  # noqa: E402
from core.embedder import clean_embed_text  # noqa: E402
from test_retrieval import run_mode, print_report, MODES  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO_QUERY_FILE = {
    'main': os.path.join(PROJECT_ROOT, 'tests', 'data', 'auto_queries_main.json'),
    'test': os.path.join(PROJECT_ROOT, 'tests', 'data', 'auto_queries.json'),
}
MIN_QUERY_LEN = 60    # 抽取文本最短长度(太短无检索意义)
MAX_QUERY_LEN = 120   # 查询片段最大长度


def check_ready(system: PDFSearchSystem) -> bool:
    """库是否已建好: 向量库非空 且 ES 索引存在"""
    try:
        vec_ok = system.vector_store.count() > 0
        es_ok = system.es_manager.es.indices.exists(
            index=system.es_manager.index_name
        )
        return vec_ok and es_ok
    except Exception:
        return False


def auto_generate_queries(system: PDFSearchSystem, per_doc: int = 1,
                          seed: int = 42) -> list:
    """从库内文档的解析结果(output_*/*.json)自动抽取查询
    查询 = 原文随机片段, 相关文档 = 来源文档
    """
    queries = []
    json_files = sorted(glob.glob(os.path.join(system.output_dir, '*.json')))
    rng = random.Random(seed)

    for jf in json_files:
        try:
            with open(jf, encoding='utf-8') as f:
                parsed = json.load(f)
        except Exception:
            continue

        filename = parsed.get('filename', os.path.basename(jf))

        # 收集够长的页文本
        texts = []
        for page in parsed.get('pages', []):
            t = re.sub(r'\s+', ' ', page.get('text', '')).strip()
            if len(t) >= MIN_QUERY_LEN:
                texts.append(t)
        if not texts:
            continue

        for _ in range(per_doc):
            t = rng.choice(texts)
            max_start = max(0, len(t) - MIN_QUERY_LEN)
            start = rng.randint(0, max_start)
            qlen = rng.randint(MIN_QUERY_LEN, min(MAX_QUERY_LEN, len(t) - start))
            q = t[start:start + qlen].strip()
            # 清洗特殊符号(公式/数学字符), 过滤清洗后过短的查询
            q = clean_embed_text(q)
            if len(q) >= 30:
                queries.append({'query': q, 'relevant': [filename]})

    return queries


def main():
    parser = argparse.ArgumentParser(description='自动化召回率测试(零标注)')
    parser.add_argument('--db', choices=['test', 'main'], default='test')
    parser.add_argument('--per-doc', type=int, default=1, help='每文档抽几条查询')
    parser.add_argument('--seed', type=int, default=42, help='随机种子(可复现)')
    parser.add_argument('--no-build', action='store_true', help='库未就绪时不自动建库')
    parser.add_argument('--reuse', action='store_true', help='复用已生成的自动测试集')
    parser.add_argument('--brief', action='store_true', help='只看汇总指标')
    args = parser.parse_args()

    print(f"构建[{args.db}]库检索系统...")
    system = PDFSearchSystem(db=args.db)

    # 1. 库就绪检查 / 自动建库
    if not check_ready(system):
        if args.no_build:
            print("[错误] 库未就绪(向量库为空或ES索引不存在), 请先建库(主菜单1/2)")
            sys.exit(1)
        print(f"[{args.db}库] 未检测到已建好的知识库, 开始自动建库...")
        system.process_pdfs()
        if not check_ready(system):
            print("[错误] 自动建库后仍不可用, 请检查 ES/Ollama 连接")
            sys.exit(1)
        print("[建库完成]")
    else:
        print(f"[{args.db}库] 知识库已就绪 (向量块数: {system.vector_store.count()})")

    # 2. 生成/复用自动测试集
    qfile = AUTO_QUERY_FILE[args.db]
    if args.reuse and os.path.exists(qfile):
        with open(qfile, encoding='utf-8') as f:
            queries = json.load(f)
        print(f"复用自动测试集: {qfile} ({len(queries)} 条查询)")
    else:
        print(f"自动生成测试查询 (每文档 {args.per_doc} 条, seed={args.seed})...")
        queries = auto_generate_queries(system, per_doc=args.per_doc, seed=args.seed)
        if not queries:
            print("[错误] 没有生成任何查询, 请检查库内文档解析结果")
            sys.exit(1)
        with open(qfile, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"已保存自动测试集: {qfile} ({len(queries)} 条查询)")

    # 3. 三种模式评测
    print("开始评测(es / vec / hybrid)...")
    results = {}
    for mode in MODES:
        recall, mrr, details = run_mode(system, mode, queries)
        results[mode] = (recall, mrr, details)

    # 4. 输出报告
    print_report(queries, results, brief=args.brief)
    print("注: 自动查询来自原文片段, 召回率反映检索链路上限;\n"
          "    真实场景建议使用 test_data/queries.json 人工标注集评测。")


if __name__ == '__main__':
    main()
