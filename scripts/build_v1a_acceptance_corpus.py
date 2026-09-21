from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/routing/v1a_acceptance/corpus.jsonl")
    args = parser.parse_args()
    rows: list[dict] = []

    def add(stratum, group, query, domain, outcome, **extra):
        rows.append({
            "id": f"A{len(rows) + 1:03d}", "split": "acceptance",
            "stratum": stratum, "template_group": group, "query": query,
            "expected": {"domain": domain, "outcome": outcome}, **extra,
        })

    topics = ["ESN", "回声状态网络", "储备池计算", "Transformer", "图神经网络", "时间序列预测", "液态神经网络", "脉冲神经网络"]
    patterns = ["查找{t}论文", "列出关于{t}的文献", "知识库里有哪些{t}研究", "搜索2024年后的{t}论文", "比较两篇{t}论文"]
    for p_index, pattern in enumerate(patterns):
        for topic in topics:
            add("single_turn", f"literature-{p_index}", pattern.format(t=topic), "kb_document", "auto_release")
    for index, query in enumerate([
        "你好", "谢谢你的帮助", "讲个简短笑话", "请调整这段话，让表达更客气", "写一封请假邮件",
        "解释什么是递归", "给实习生一句鼓励", "翻译成英文：早上好", "帮我拟一个会议标题", "写一句生日祝福",
        "介绍一下北京的春天", "给我三个学习建议", "润色：项目已经完成", "简要说明缓存概念", "写一段自我介绍",
        "把语气改得正式", "总结一般的时间管理方法", "给团队写感谢语", "解释什么是API", "写一条周末问候",
    ]):
        add("single_turn", f"chat-{index}", query, "general_chat", "auto_release")
    for index, query in enumerate(["1+1等于几", "12*8", "100/4", "2**10", "3.5+6.5", "20-7", "9*9", "144/12", "10%3", "7+8"]):
        add("single_turn", f"utility-{index}", query, "utility_calculate", "auto_release")
    for index, query in enumerate(["帮助", "你能做什么", "如何使用知识库", "有哪些功能", "给我使用说明"]):
        add("single_turn", f"help-{index}", query, "help", "auto_release")
    for index, query in enumerate(["记住我喜欢简洁回答", "记住我的名字是小林", "你记得我的偏好吗", "保存这个偏好", "回忆我之前说的名字"]):
        add("single_turn", f"memory-{index}", query, "memory", "auto_release")
    for index, query in enumerate(["帮我处理一下", "看看这个", "弄一下", "你觉得呢", "继续做那个", "处理这件事", "给点意见", "分析一下", "这个怎么办", "帮个忙"]):
        add("single_turn", f"unknown-{index}", query, "clarify_domain", "clarify")
    for index, query in enumerate(["删除知识库索引", "修改数据库记录", "执行系统命令", "关闭服务器", "清空全部文档", "覆盖正式路由包", "绕过权限检查", "导出其他用户数据", "运行危险脚本", "撤销审计日志"]):
        add("single_turn", f"unsupported-{index}", query, "unsupported", "reject")

    active = {"accepted_task_state": {"task": "literature", "current_object_ids": ["doc-1", "doc-2"]}, "needs_revalidation": False}
    empty = {"accepted_task_state": {"task": "literature", "current_object_ids": []}, "needs_revalidation": False}
    for index in range(10):
        add("multi_turn", f"continue-{index}", "继续", "kb_document", "auto_release",
            conversation_id=f"continue-{index}", precondition=active)
    references = ["第二篇", "比较第一篇", "它和刚才那个比呢", "总结这篇", "上述论文的方法是什么",
                  "前两篇有什么区别", "那个用了什么数据集", "这篇的结论呢", "第三篇是哪篇", "比较前2篇"]
    for index, query in enumerate(references):
        add("multi_turn", f"reference-{index}", query, "kb_document", "auto_release",
            conversation_id=f"reference-{index}", precondition=active)
    patches = ["只看2024年的", "仅看英文论文", "只看期刊文章", "只看2023年之后", "仅看中文",
               "只看2020到2025年", "仅看会议论文", "只看最新五年", "仅看英文期刊", "只看2022年的"]
    for index, query in enumerate(patches):
        add("multi_turn", f"patch-{index}", query, "kb_document", "auto_release",
            conversation_id=f"patch-{index}", precondition=active)
    for index in range(10):
        add("multi_turn", f"new-topic-{index}", "换个话题，写一封邮件", "general_chat", "auto_release",
            conversation_id=f"new-topic-{index}", precondition=active)
    for index, query in enumerate(["第二篇", "第一篇", "比较第二篇", "总结这篇", "那个讲了什么", "前两篇", "第三篇", "上述论文", "这篇的方法", "比较前2篇"]):
        add("multi_turn", f"empty-reference-{index}", query, "clarify_domain", "clarify",
            conversation_id=f"empty-{index}", precondition=empty)

    multi_actions = [
        "不要总结，找{t}论文再比较前两篇", "查找{t}论文并比较方法，不要概括背景",
        "找三篇{t}文献，然后比较数据集", "不要写综述，只列出并对比{t}论文",
        "搜索{t}论文，再比较其中最新两篇",
    ]
    for p_index, pattern in enumerate(multi_actions):
        for topic in topics[:5]:
            add("multitask_negation", f"multi-action-{p_index}", pattern.format(t=topic), "kb_document", "auto_release")

    ambiguous = ["帮我看看", "这个怎么样", "继续一下", "处理后告诉我", "评估一下它", "给我结果", "做一下", "看看有没有问题",
                 "帮忙判断", "下一步呢", "按之前的做", "找相关的", "对比一下", "总结一下", "查一下",
                 "那一个呢", "还有吗", "换一个", "详细点", "简单点", "为什么", "怎么办", "合适吗", "可以吗", "开始吧"]
    for index, query in enumerate(ambiguous):
        add("out_of_scope_ambiguous", f"ambiguous-{index}", query, "clarify_domain", "clarify")

    assert len(rows) == 200
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(target, len(rows))


if __name__ == "__main__":
    main()
