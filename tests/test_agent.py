#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent 层单元测试(不依赖外部服务, 全部本地/mock)
运行: venv/bin/python tests/test_agent.py
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_limits import (  # noqa: E402
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    PersistentCache,
    RateLimiter,
    TokenBudget,
    estimate_tokens,
)
from agent.agent_models import AgentReply, IntentPrediction, IntentType  # noqa: E402
from agent.intent_router import IntentRecognizer  # noqa: E402

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


# ============ 1. token 估算 ============
def test_estimate_tokens():
    print("[test_estimate_tokens]")
    check("中文每字约1token", estimate_tokens("人工之恶能") == 5)
    check("英文4字符约1token", abs(estimate_tokens("hello world") - 3) <= 1)
    check("空串为0", estimate_tokens("") == 0)


# ============ 2. PersistentCache ============
def test_persistent_cache():
    print("[test_persistent_cache]")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        cache = PersistentCache(path, max_entries=2, ttl_seconds=60)
        cache.put("a", {"answer": "你好"})
        check("写入可读", cache.get("a") == {"answer": "你好"})

        # 跨进程持久化: 新实例读同一文件
        cache2 = PersistentCache(path, max_entries=2, ttl_seconds=60)
        check("跨进程持久化", cache2.get("a") == {"answer": "你好"})

        # 容量上限淘汰最旧
        cache.put("b", 1)
        cache.put("c", 2)
        check("容量上限淘汰", cache.get("a") is None and cache.get("c") == 2)

        # TTL 过期
        cache3 = PersistentCache(path, max_entries=10, ttl_seconds=1)
        cache3.put("k", "v")
        time.sleep(1.2)
        check("TTL 过期失效", cache3.get("k") is None)


# ============ 3. TokenBudget ============
def test_token_budget():
    print("[test_token_budget]")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "usage.json")
        budget = TokenBudget(path, session_limit=100, daily_limit=1000)
        ok, reason = budget.check_and_consume(60, user_id="u1")
        check("会话内消费放行", ok and reason is None)
        ok, reason = budget.check_and_consume(60, user_id="u1")
        check("会话预算超限拒绝", not ok and "会话" in reason)
        ok, _ = budget.check_and_consume(10, user_id="u2")
        check("不同用户独立预算", ok)

        # 每日预算跨进程持久化(拒绝的调用不记账, 所以 u1 只成功扣了 60)
        budget2 = TokenBudget(path, session_limit=100, daily_limit=1000)
        snap = budget2.snapshot("u1")
        check("每日用量持久化", snap["daily_used"] == 60, f"got {snap}")
        ok, reason = budget2.check_and_consume(60, user_id="u1")
        check("新实例继续扣减", ok and budget2.snapshot("u1")["daily_used"] == 120)


# ============ 4. RateLimiter ============
def test_rate_limiter():
    print("[test_rate_limiter]")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rate.json")
        limiter = RateLimiter(path, max_calls=2, window_seconds=60)
        check("前两次放行", limiter.allow("u1")[0] and limiter.allow("u1")[0])
        ok, retry = limiter.allow("u1")
        check("第三次拒绝且给出等待秒数", (not ok) and retry > 0)
        check("其他用户不受影响", limiter.allow("u2")[0])

        # 跨进程持久化
        limiter2 = RateLimiter(path, max_calls=2, window_seconds=60)
        check("限流计数跨进程持久化", not limiter2.allow("u1")[0])


# ============ 5. CircuitBreaker ============
def test_circuit_breaker():
    print("[test_circuit_breaker]")
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=1)

    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise RuntimeError("boom")

    def ok_fn():
        return "ok"

    try:
        breaker.call(fail)
    except RuntimeError:
        pass
    try:
        breaker.call(fail)
    except RuntimeError:
        pass
    check("连续失败后熔断打开", breaker.state() == "open")
    try:
        breaker.call(ok_fn)
        check("熔断打开时拒绝调用", False)
    except CircuitOpenError:
        check("熔断打开时拒绝调用", True)

    time.sleep(1.2)  # 冷却期结束, is_open() 触发 open -> half-open
    check("冷却后进入半开", (not breaker.is_open()) and breaker.state() == "half-open")
    check("半开成功恢复关闭", breaker.call(ok_fn) == "ok" and breaker.state() == "closed")

    # 半开失败 -> 重新打开
    breaker.record_failure()
    breaker.record_failure()
    check("再次熔断", breaker.state() == "open")


# ============ 6. 意图路由规则(含否定词) ============
def test_intent_rules():
    print("[test_intent_rules]")
    recognizer = IntentRecognizer(llm_client=None)  # 只测规则路径, 不触 LLM

    def rule(message):
        return recognizer._match_fast_path(message, [{"name": "search_private_kb"}])

    check("问候走闲聊", rule("你好").intent == IntentType.CHAT_DAILY)
    check("搜索关键词走检索", rule("帮我查一下误差补偿的论文").intent == IntentType.KB_SEARCH)
    check("教学请求走讲解", rule("我不懂GAN，你能给我讲讲吗").intent == IntentType.KB_TUTOR)
    check("入门请求走讲解", rule("GAN怎么入门").intent == IntentType.KB_TUTOR)
    check("分析关键词走分析", rule("总结一下贝叶斯方法的进展").intent == IntentType.KB_ANALYSIS)
    check("否定词抑制分析", rule("我不需要你总结") is None)
    check("否定词抑制搜索", rule("不用检索了") is None)
    check("短消息走澄清", rule("嗯").intent == IntentType.CLARIFY)
    check("空消息走澄清", rule("").intent == IntentType.CLARIFY)
    check("元问题走闲聊", rule("你能做什么").intent == IntentType.CHAT_DAILY)
    check("未命中返回None", rule("今天天气怎么样") is None)


# ============ 7. Agent 缓存序列化往返 ============
def test_reply_marshal():
    print("[test_reply_marshal]")
    from agent.agent_service import PrivateKnowledgeAgent

    intent = IntentPrediction(
        intent=IntentType.KB_SEARCH,
        needs_retrieval=True,
        confidence=0.8,
        user_goal="查误差补偿",
        router_source="rule",
    )
    reply = AgentReply(answer="找到了", intent=intent)
    raw = reply.to_dict()
    restored = PrivateKnowledgeAgent._reply_from_dict(raw)
    check("AgentReply 序列化往返", restored is not None and restored.answer == "找到了")
    check("意图字段往返", restored.intent.intent == IntentType.KB_SEARCH)


# ============ 8. chat 异常兜底(不崩) ============
def test_chat_error_guard():
    print("[test_chat_error_guard]")
    import agent_limits
    from agent.agent_models import RetrievedEvidence
    from agent.agent_service import PrivateKnowledgeAgent

    class BoomClient:
        settings = type("S", (), {"router_strict": False})()

        def invoke_router(self, **kwargs):
            raise RuntimeError("api down")

        def invoke_text(self, **kwargs):
            raise RuntimeError("api down")

    class FakeBackend:
        def hybrid_search(self, query, top_n=10, include_chunks=False):
            if include_chunks:
                return [{"filename": "x.pdf", "rrf": 0.1}], [
                    {"filename": "x.pdf", "page": 1, "text": "证据内容", "score": 0.9}
                ]
            return [{"filename": "x.pdf", "rrf": 0.1}]

        def vec_search(self, query, top_k=10):
            return []

    agent = PrivateKnowledgeAgent(
        search_backend=FakeBackend(),
        deepseek_client=BoomClient(),
        settings=type(
            "S",
            (),
            {
                "default_top_k": 5,
                "max_history_turns": 6,
                "answer_temperature": 0.1,
                "low_confidence_threshold": 0.55,
                "cache_size": 8,
                "cache_ttl_seconds": 60,
                "blocked_ttl_seconds": 300,
                "state_dir": tempfile.mkdtemp(),
            },
        )(),
    )
    # 规则路径: 搜索关键词 -> 规则意图, 不触 LLM; 检索后回答走 BoomClient -> 抛异常 -> 兜底
    reply = agent.chat("帮我查一下误差补偿", user_id="tester")
    check("API 异常被兜底为回复", reply.answer.startswith("处理你的问题时") or "证据" in reply.answer)
    check("兜底回复不抛异常", isinstance(reply, AgentReply))


def main():
    test_estimate_tokens()
    test_persistent_cache()
    test_token_budget()
    test_rate_limiter()
    test_circuit_breaker()
    test_intent_rules()
    test_reply_marshal()
    test_chat_error_guard()

    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
