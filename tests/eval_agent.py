#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent 评测: 意图路由准确率 + 证据命中率, 支持 LangSmith 上报

本地模式(无需 LangSmith):
    ./venv/bin/python tests/eval_agent.py --db test

LangSmith 模式(需 LANGSMITH_TRACING=true + LANGSMITH_API_KEY):
    ./venv/bin/python tests/eval_agent.py --db test --langsmith

golden 集: tests/data/agent_golden.json (可自行扩充)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_service import AgentSettings, PrivateKnowledgeAgent  # noqa: E402
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings  # noqa: E402
from main import PDFSearchSystem  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_FILE = os.path.join(PROJECT_ROOT, "tests", "data", "agent_golden.json")


def load_golden() -> list[dict]:
    if not os.path.exists(GOLDEN_FILE):
        print(f"[错误] golden 集不存在: {GOLDEN_FILE}")
        sys.exit(1)
    with open(GOLDEN_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    if not data:
        print("[警告] golden 集为空")
    return data


def build_agent(db: str) -> PrivateKnowledgeAgent:
    system = PDFSearchSystem(db=db)
    settings = AgentSettings(state_dir=os.path.join(PROJECT_ROOT, "agent_state"))
    try:
        client = DeepSeekClient(DeepSeekSettings.from_env())
    except Exception as exc:
        print(f"[警告] DeepSeek 配置不可用({exc}), 意图评测中 LLM 路由将降级为关键词")
        client = None
    return PrivateKnowledgeAgent(search_backend=system, deepseek_client=client, settings=settings)


def evaluate_locally(agent: PrivateKnowledgeAgent, golden: list[dict]) -> dict:
    """本地评测: 意图准确率 + 证据命中率"""
    intent_correct = 0
    evidence_hits = 0
    evidence_total = 0
    router_sources: dict[str, int] = {}

    rows = []
    for item in golden:
        query = item["query"]
        expected_intent = item.get("expected_intent", "")
        expected_evidence = set(item.get("expected_evidence", []))

        intent = agent.plan(query)
        router_sources[intent.router_source] = router_sources.get(intent.router_source, 0) + 1
        intent_ok = intent.intent.value == expected_intent
        intent_correct += 1 if intent_ok else 0

        # 证据命中: 直接跑检索技能(不调 LLM 回答), 取前 top_k 证据文件
        evidence_files: set[str] = set()
        if expected_evidence:
            try:
                result = agent.registry.execute(
                    "search_private_kb", {"query": query, "top_k": 5}
                )
                evidence_files = {
                    str(ev.get("source", ""))
                    for ev in result.get("evidence", [])
                    if ev.get("source")
                }
            except Exception as exc:
                print(f"  [检索失败] {query}: {exc}")
            hit = len(evidence_files & expected_evidence)
            evidence_hits += hit
            evidence_total += len(expected_evidence)

        rows.append(
            {
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": intent.intent.value,
                "router_source": intent.router_source,
                "intent_ok": intent_ok,
                "expected_evidence": sorted(expected_evidence),
                "evidence_hit": sorted(evidence_files & expected_evidence),
                "evidence_miss": sorted(expected_evidence - evidence_files),
            }
        )

    n = len(golden)
    report = {
        "total": n,
        "intent_accuracy": intent_correct / n if n else 0.0,
        "evidence_recall": evidence_hits / evidence_total if evidence_total else None,
        "router_sources": router_sources,
        "rows": rows,
    }
    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("Agent 评测报告")
    print("=" * 70)
    print(f"查询数: {report['total']}")
    print(f"意图准确率: {report['intent_accuracy']:.3f}")
    if report["evidence_recall"] is not None:
        print(f"证据命中率(Recall): {report['evidence_recall']:.3f}")
    print(f"路由来源分布: {report['router_sources']}")
    print("-" * 70)
    for row in report["rows"]:
        mark = "✓" if row["intent_ok"] else "✗"
        print(f"[{mark}] {row['query'][:40]}")
        print(f"    期望={row['expected_intent']} 实际={row['actual_intent']}({row['router_source']})")
        if row["evidence_miss"]:
            print(f"    证据缺失: {', '.join(row['evidence_miss'])}")
    print("=" * 70)


def run_langsmith(agent: PrivateKnowledgeAgent, golden: list[dict], db: str) -> None:
    """上报 LangSmith: 创建数据集 + evaluate 意图准确率"""
    try:
        from langsmith import Client
    except ImportError:
        print("[错误] 未安装 langsmith: pip install langsmith")
        sys.exit(1)

    client = Client()
    dataset_name = f"agent-intent-golden-{db}"
    try:
        dataset = client.create_dataset(
            dataset_name,
            description=f"Agent 意图路由 golden 集 ({db}库)",
        )
    except Exception:
        dataset = client.get_dataset(dataset_name)

    examples = [
        {
            "inputs": {"query": item["query"]},
            "outputs": {"expected_intent": item.get("expected_intent", "")},
        }
        for item in golden
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"数据集就绪: {dataset_name} ({len(examples)} 条)")

    def intent_accuracy_evaluator(run, example):
        actual = (run.outputs or {}).get("intent", "")
        expected = (example.outputs or {}).get("expected_intent", "")
        return {"key": "intent_accuracy", "score": 1.0 if actual == expected else 0.0}

    def runner(inputs: dict) -> dict:
        intent = agent.plan(str(inputs.get("query", "")))
        return {"intent": intent.intent.value}

    print("开始 LangSmith evaluate (意图准确率)...")
    result = client.evaluate(
        runner,
        data=dataset_name,
        evaluators=[intent_accuracy_evaluator],
        experiment_prefix=f"agent-intent-{db}",
        metadata={"db": db, "golden": len(golden)},
    )
    print("评测完成, 结果: ", end="")
    for summary in result:
        print(summary)


def main():
    parser = argparse.ArgumentParser(description="Agent 评测(意图 + 证据命中)")
    parser.add_argument("--db", choices=["test", "main"], default="test")
    parser.add_argument("--langsmith", action="store_true", help="上报 LangSmith")
    args = parser.parse_args()

    print(f"加载 golden 集: {GOLDEN_FILE}")
    golden = load_golden()
    if not golden:
        return

    print(f"构建 Agent({args.db}库)...")
    agent = build_agent(args.db)

    report = evaluate_locally(agent, golden)
    print_report(report)

    if args.langsmith:
        run_langsmith(agent, golden, args.db)


if __name__ == "__main__":
    main()
