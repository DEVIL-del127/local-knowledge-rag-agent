from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from agent.v1a_acceptance import evaluate_category_evidence
from core.embedder import OllamaEmbedder, clean_embed_text


def _cases():
    topics = ["ESN", "回声状态网络", "储备池计算", "Transformer", "图神经网络",
              "时间序列预测", "液态神经网络", "脉冲神经网络", "强化学习", "异常检测"]
    rows = []
    for pattern in ["查找{t}论文", "搜索{t}相关文献", "有哪些关于{t}的研究", "列出{t}期刊文章", "知识库中检索{t}资料"]:
        rows.extend({"category": "literature_find", "query": pattern.format(t=t)} for t in topics)
    for pattern in ["比较两篇{t}论文", "对比{t}研究的方法", "分析两篇{t}文献的差异", "比较{t}论文的实验", "对照{t}研究的结论"]:
        rows.extend({"category": "literature_compare", "query": pattern.format(t=t)} for t in topics)
    chat = [
        "你好", "您好", "早上好", "下午好", "晚上好", "嗨", "很高兴见到你", "最近怎么样", "谢谢", "多谢",
        "谢谢你的帮助", "辛苦了", "再见", "回头见", "祝你今天愉快", "讲个笑话", "说个冷笑话", "逗我开心一下", "讲个有趣故事", "来个段子",
        "写一句生日祝福", "写一条周末问候", "给同事一句鼓励", "写一封请假邮件", "帮我拟会议标题",
        "翻译成英文：早上好", "润色这句话", "把语气改礼貌", "写一段自我介绍", "给我三个学习建议",
        "解释什么是递归", "简单介绍北京", "说说时间管理", "如何保持专注", "给团队写感谢语",
        "推荐一本小说", "推荐一部电影", "晚饭吃什么", "周末做什么", "今天心情不错",
        "请简短回答", "换一种说法", "写得正式一点", "写得轻松一点", "给个生活建议",
        "祝考试顺利", "写一句欢迎语", "帮我想个昵称", "说一句晚安", "谢谢再见",
    ]
    rows.extend({"category": "general_chat", "query": query} for query in chat)
    assert len(rows) == 150
    return rows


def _validation_cases():
    topics = ["联邦学习", "扩散模型", "视觉语言模型", "知识图谱", "因果推断",
              "多模态检索", "神经架构搜索", "自监督学习", "边缘计算", "数字孪生"]
    rows = []
    for pattern in ["帮我搜集{t}学术论文", "找出讨论{t}的文章", "给出{t}领域文献清单",
                    "查阅{t}相关出版物", "寻找{t}方向代表作", "从库中筛选{t}研究"]:
        rows.extend({"category": "literature_find", "query": pattern.format(t=t)} for t in topics)
    for pattern in ["对照两项{t}研究", "比较{t}文章的技术路线", "指出多篇{t}论文的差别",
                    "评比{t}研究的实验表现", "对比{t}文献使用的数据", "归纳{t}论文结论的异同"]:
        rows.extend({"category": "literature_compare", "query": pattern.format(t=t)} for t in topics)
    chat = [
        "早安", "午安", "晚上见", "最近过得好吗", "很开心认识你", "感谢解答", "非常感谢", "麻烦你了", "辛苦啦", "拜拜",
        "说个幽默故事", "来一句俏皮话", "逗我笑一笑", "讲个睡前故事", "说个轻松段子",
        "写一句毕业祝福", "写一条欢迎消息", "给朋友一句鼓励", "起草普通通知", "拟一个活动名称",
        "把这句话译成中文", "调整这段话的语气", "改成更正式的表达", "写一个个人简介", "提供复习建议",
        "通俗解释循环", "简单介绍上海", "谈谈如何安排时间", "怎样改善睡眠", "写一段感谢同事的话",
        "推荐轻松的漫画", "推荐喜剧电影", "推荐舒缓音乐", "推荐家常菜", "推荐周末活动",
        "今天有点开心", "晚餐有什么建议", "周末去哪里玩", "聊聊兴趣爱好", "说一句晚安",
        "回答简洁一些", "换个更自然的说法", "改得活泼一点", "改得严谨一点", "给我一个生活小技巧",
        "祝面试成功", "写一句开场白", "想一个网名", "写一条节日问候", "感谢并告别",
        "解释什么是变量", "说明缓存的用途", "介绍常见排序方法", "解释云计算概念", "说明什么是接口",
        "写一封预约邮件", "写一个会议提醒", "润色工作总结", "翻译日常问候", "给新人一些建议",
    ]
    rows.extend({"category": "general_chat", "query": q} for q in chat)
    assert len(rows) == 180
    return rows


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="data/routing/v1a/package.json")
    parser.add_argument("--output", default="docs/v1a-semantic-evidence.json")
    parser.add_argument("--dataset", choices=("development", "validation"), default="development")
    args = parser.parse_args()
    package_path = Path(args.package)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    matrix = np.load(package_path.with_name(package["matrix_file"]), allow_pickle=False)
    embedder = OllamaEmbedder(model=package["encoder"]["model"], batch_size=16, timeout=30)
    cases = _cases() if args.dataset == "development" else _validation_cases()
    embedder.embed_query("预热bge-m3")
    started = time.perf_counter()
    vectors = np.asarray(embedder.embed([clean_embed_text(x["query"]) for x in cases]), dtype=np.float32)
    batch_ms = (time.perf_counter() - started) * 1000.0
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    evidence = []
    for case, vector in zip(cases, vectors):
        scores = matrix @ vector
        grouped = {}
        for example, score in zip(package["examples"], scores.tolist()):
            grouped.setdefault(example["category"], {})[example["cluster_id"]] = float(score)
        ranked = sorted(((cat, float(np.mean(sorted(vals.values(), reverse=True)[:3])))
                         for cat, vals in grouped.items()), key=lambda x: x[1], reverse=True)
        top, second = ranked[0], ranked[1]
        policy = package["categories"][top[0]]
        released = top[1] >= policy["threshold"] and top[1] - second[1] >= policy["margin"]
        evidence.append({**case, "predicted": top[0], "score": top[1],
                         "margin": top[1] - second[1], "released": released,
                         "correct": top[0] == case["category"], "critical_error": False})
    category_report = evaluate_category_evidence(evidence, minimum_releases=50)
    # Individual warm-query latency uses the production single-query path.
    latencies = []
    for item in cases[:30]:
        tick = time.perf_counter()
        embedder.embed_query(clean_embed_text(item["query"]))
        latencies.append((time.perf_counter() - tick) * 1000.0)
    report = {"schema_version": "v1a-semantic-evidence-1", "model": package["encoder"]["model"],
              "samples": len(cases), "batch_latency_ms": batch_ms,
              "warm_single_query": {"count": len(latencies), "p50_ms": _percentile(latencies, 0.5),
                                    "p95_ms": _percentile(latencies, 0.95), "max_ms": max(latencies)},
              "category_evidence": category_report, "cases": evidence}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("samples", "batch_latency_ms", "warm_single_query", "category_evidence")}, ensure_ascii=False, indent=2))
    return 0 if category_report["passed"] and report["warm_single_query"]["p95_ms"] <= 1000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
