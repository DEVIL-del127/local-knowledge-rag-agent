from __future__ import annotations

from typing import Mapping
import inspect
import uuid

from agent.agent_service import PrivateKnowledgeAgent

MAX_HISTORY_ROUNDS = 12  # 保留最近 12 轮对话(24 条消息), 防无限增长


def run_agent_loop(
    agent: PrivateKnowledgeAgent,
    *,
    label: str = "正式",
    user_id: str = "default",
    memory=None,
    session_id: str | None = None,
) -> None:
    """检索词记录 + 记忆写入挂载: memory.ingest_message 每轮调用"""
    print("\n" + "=" * 64)
    print(f"  私人知识库 Agent 已启动（当前连接: {label}库 | 用户: {user_id}）")
    print("  能力: DeepSeek 对话 + 意图识别 + 本地知识库检索")
    print("  输入 quit / exit / q 返回主菜单")
    print("=" * 64)

    history: list[Mapping[str, str]] = []
    round_no = 0

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n返回主菜单。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q", "back", "menu", "/quit", "/exit", "/q"}:
            print("返回主菜单。")
            break

        round_no += 1
        request_id = uuid.uuid4().hex
        try:
            # Inspect before dispatch: retrying on TypeError could repeat effects.
            parameters = inspect.signature(agent.chat).parameters
            request_options = {}
            if "request_id" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                request_options["request_id"] = request_id
            reply = agent.chat(
                user_input, history=history, user_id=user_id, session_id=session_id or "", **request_options,
            )
        except Exception as exc:
            # 双保险: 即使 agent 内部漏网也保证会话不崩、历史不丢
            print(f"\n[错误] 处理请求失败: {exc}")
            print("      已保留当前会话，请重试或换个说法。")
            error_context = f"上一条请求处理失败：{type(exc).__name__}: {str(exc)[:300]}"
            history = [*history,
                       {"role": "user", "content": user_input},
                       {"role": "assistant", "content": error_context}]
            if len(history) > MAX_HISTORY_ROUNDS * 2:
                history = history[-(MAX_HISTORY_ROUNDS * 2):]
            continue

        history = [*history,
                   {"role": "user", "content": user_input},
                   {"role": "assistant", "content": reply.answer}]
        # 裁剪历史, 防止长会话内存无限增长
        if len(history) > MAX_HISTORY_ROUNDS * 2:
            history = history[-(MAX_HISTORY_ROUNDS * 2):]

        # 记忆写入(旁路, 内部节流; 失败不影响主流程)
        if memory is not None and session_id:
            try:
                memory.ingest_message(
                    round_no=round_no,
                    user_msg=user_input,
                    agent_reply=reply.answer,
                    intent=reply.intent,
                    session_id=session_id,
                    user_id=user_id,
                )
            except Exception:
                logger_exc = __import__("logging").getLogger(__name__)
                logger_exc.exception("记忆写入失败(已忽略)")

        print(
            f"\n[意图] {reply.intent.intent.value}"
            f" | 路由={reply.intent.router_source}"
            f" | 置信度={reply.intent.confidence:.2f}"
        )
        print(f"\nAgent: {reply.answer}")

        if reply.evidence:
            print("\n参考来源:")
            for item in reply.evidence:
                print(f"- {item.citation_label()}")
