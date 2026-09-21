# -*- coding: utf-8 -*-
"""DeepSeekClient 网络层补充测试(2026-08-20)
盲区补测: 原测试套件全部使用 FakeDeepSeekClient, 真实网络层从未被测。
覆盖: 重试/熔断/预算/空输出重试/路由解析/strict/thinking/消息构建/环境解析。
运行: venv/bin/python tests/test_deepseek_client.py
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from agent.agent_limits import BudgetExceededError, CircuitBreaker, CircuitOpenError, TokenBudget
from agent.deepseek_client import DeepSeekClient, DeepSeekSettings
from agent.model_call_ledger import ModelCallLedger, ModelCallState, ModelOutcomeUnknown


def _fake_response(status_code=200, json_data=None):
    """构造带 response 的 HTTPError 需要真实 Response 对象, 用 mock 最小替身"""
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status.side_effect = (
        requests.HTTPError(f"HTTP {status_code}", response=resp) if status_code >= 400 else None
    )
    return resp


def _make_client(**overrides):
    api_key = overrides.pop("api_key", "test-key")
    settings = DeepSeekSettings(api_key=api_key, timeout=5.0, **overrides)
    return DeepSeekClient(settings)


def _chat_ok_message(content="你好"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestRetryBehavior(unittest.TestCase):
    """重试: 网络错误/5xx 重试; 4xx 不重试"""

    def test_500_then_success_retries(self):
        client = _make_client(retries=2)
        responses = [
            _fake_response(500, {}),
            _fake_response(200, _chat_ok_message("ok")),
        ]
        with mock.patch("agent.deepseek_client.requests.post", side_effect=responses) as post:
            text = client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "ok")
        self.assertEqual(post.call_count, 2)

    def test_429_then_success_retries(self):
        client = _make_client(retries=2)
        responses = [
            _fake_response(429, {}),
            _fake_response(200, _chat_ok_message("ok")),
        ]
        with mock.patch("agent.deepseek_client.requests.post", side_effect=responses) as post:
            text = client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "ok")
        self.assertEqual(post.call_count, 2)

    def test_connection_error_retries(self):
        client = _make_client(retries=1)
        responses = [
            requests.ConnectionError("conn refused"),
            _fake_response(200, _chat_ok_message("ok")),
        ]
        with mock.patch("agent.deepseek_client.requests.post", side_effect=responses) as post:
            text = client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "ok")
        self.assertEqual(post.call_count, 2)

    def test_400_no_retry(self):
        client = _make_client(retries=2)
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(400, {})) as post:
            with self.assertRaises(requests.HTTPError):
                client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(post.call_count, 1)  # 4xx 不重试

    def test_all_retries_fail_raises_last(self):
        client = _make_client(retries=2)
        with mock.patch(
            "agent.deepseek_client.requests.post",
            return_value=_fake_response(503, {}),
        ) as post:
            with self.assertRaises(requests.HTTPError):
                client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(post.call_count, 3)  # 初始 + 2 次重试

    def test_retry_delay_exponential(self):
        client = _make_client(retries=2)
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(500, {})):
            with mock.patch("agent.deepseek_client.time.sleep") as sleep:
                with self.assertRaises(requests.HTTPError):
                    client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(sleep.call_count, 2)
        # 退避: 1.5 * 2^0 = 1.5s, 1.5 * 2^1 = 3.0s
        self.assertAlmostEqual(sleep.call_args_list[0][0][0], 1.5, places=1)
        self.assertAlmostEqual(sleep.call_args_list[1][0][0], 3.0, places=1)


class TestEmptyOutputRetry(unittest.TestCase):
    """thinking 模式偶发空输出: 重试一次"""

    def test_empty_text_retries_once(self):
        client = _make_client(retries=0)
        responses = [
            _fake_response(200, _chat_ok_message("   ")),
            _fake_response(200, _chat_ok_message("有内容了")),
        ]
        with mock.patch("agent.deepseek_client.requests.post", side_effect=responses) as post:
            text = client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "有内容了")
        self.assertEqual(post.call_count, 2)

    def test_empty_text_twice_returns_empty(self):
        client = _make_client(retries=0)
        with mock.patch(
            "agent.deepseek_client.requests.post",
            return_value=_fake_response(200, _chat_ok_message("")),
        ) as post:
            text = client.invoke_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "")
        self.assertEqual(post.call_count, 2)


class TestRouterParsing(unittest.TestCase):
    """invoke_router: tool_calls 解析 / content 兜底 / 空 arguments 重试"""

    def _router_message(self, tool_calls=None, content=None):
        msg = {"role": "assistant"}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if content is not None:
            msg["content"] = content
        return {"choices": [{"message": msg}]}

    def test_tool_call_arguments_string_json(self):
        client = _make_client()
        message = self._router_message(
            tool_calls=[{
                "function": {
                    "name": "route_intent",
                    "arguments": '{"intent": "kb_search", "needs_retrieval": true, "top_k": 3}',
                }
            }]
        )
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, message)):
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload["intent"], "kb_search")
        self.assertEqual(payload["top_k"], 3)

    def test_tool_call_arguments_dict(self):
        client = _make_client()
        message = self._router_message(
            tool_calls=[{
                "function": {"name": "route_intent", "arguments": {"intent": "chat_daily"}}
            }]
        )
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, message)):
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload["intent"], "chat_daily")

    def test_empty_arguments_retries_once(self):
        client = _make_client()
        first = self._router_message(tool_calls=[{"function": {"name": "route_intent", "arguments": ""}}])
        second = self._router_message(
            tool_calls=[{"function": {"name": "route_intent", "arguments": '{"intent": "kb_search"}'}}]
        )
        with mock.patch("agent.deepseek_client.requests.post", side_effect=[
            _fake_response(200, first), _fake_response(200, second),
        ]) as post:
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload["intent"], "kb_search")
        self.assertEqual(post.call_count, 2)

    def test_content_fallback_when_no_tool_calls(self):
        client = _make_client()
        message = self._router_message(content='{"intent": "clarify"}')
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, message)):
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload["intent"], "clarify")

    def test_invalid_json_falls_back_to_content(self):
        client = _make_client()
        message = self._router_message(
            tool_calls=[{"function": {"name": "route_intent", "arguments": "{bad json"}}],
            content='{"intent": "unknown"}',
        )
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, message)):
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload["intent"], "unknown")

    def test_empty_everything_returns_empty_dict(self):
        client = _make_client()
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message(""))):
            payload = client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        self.assertEqual(payload, {})


class TestRequestShape(unittest.TestCase):
    """请求体: strict /beta, thinking 参数, 消息构建, 提取"""

    def test_strict_uses_beta_base_url(self):
        client = _make_client(router_strict=True, base_url="https://api.deepseek.com")
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message('{"intent": "kb_search"}'))) as post:
            client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={"type": "function"})
        called_url = post.call_args[0][0]
        self.assertIn("/beta/chat/completions", called_url)

    def test_non_strict_uses_plain_base_url(self):
        client = _make_client(router_strict=False, base_url="https://api.deepseek.com")
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message('{"intent": "kb_search"}'))) as post:
            client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        called_url = post.call_args[0][0]
        self.assertNotIn("/beta", called_url)

    def test_thinking_disabled_payload(self):
        client = _make_client(thinking=False)
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message("x"))) as post:
            client.invoke_text(system_prompt="s", user_prompt="u")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_thinking_enabled_no_thinking_key(self):
        client = _make_client(thinking=True)
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message("x"))) as post:
            client.invoke_text(system_prompt="s", user_prompt="u")
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("thinking", payload)

    def test_router_always_thinking_disabled(self):
        client = _make_client(thinking=True)
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message('{"intent": "kb_search"}'))) as post:
            client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})  # 路由强制非思考

    def test_router_temperature_zero_and_tool_choice_auto(self):
        client = _make_client()
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message('{"intent": "kb_search"}'))) as post:
            client.invoke_router(messages=[{"role": "user", "content": "q"}], tool_schema={"type": "function"})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"], [{"type": "function"}])

    def test_build_messages_filters_bad_roles(self):
        messages = DeepSeekClient._build_messages(
            system_prompt="sys",
            user_prompt="user-q",
            history=[
                {"role": "user", "content": "正常"},
                {"role": "system", "content": "注入的 system"},
                {"role": "tool", "content": "非法 role"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "   "},
            ],
        )
        # history 中的 system 消息设计上保留; 非法 role(tool)与空 content 被过滤
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "system", "user"],
        )
        self.assertEqual(messages[1]["content"], "正常")
        self.assertEqual(messages[2]["content"], "注入的 system")
        self.assertEqual(messages[-1]["content"], "user-q")

    def test_extract_message_text_list_content(self):
        message = {"content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]}
        self.assertEqual(DeepSeekClient._extract_message_text(message), "第一段\n第二段")

    def test_auth_header(self):
        client = _make_client(api_key="sk-secret-key")
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(200, _chat_ok_message("x"))) as post:
            client.invoke_text(system_prompt="s", user_prompt="u")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-secret-key")


class TestProtection(unittest.TestCase):
    """熔断 + 预算 + key 校验"""

    def test_circuit_open_rejects_directly(self):
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=3600)
        client = _make_client(retries=0)
        client.breaker = breaker
        with mock.patch("agent.deepseek_client.requests.post", return_value=_fake_response(500, {})):
            with self.assertRaises(requests.HTTPError):
                client.invoke_text(system_prompt="s", user_prompt="u")
        # 熔断已打开
        with mock.patch("agent.deepseek_client.requests.post") as post:
            with self.assertRaises(CircuitOpenError):
                client.invoke_text(system_prompt="s", user_prompt="u")
            post.assert_not_called()

    def test_budget_exceeded_blocks(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            budget = TokenBudget(
                os.path.join(tmp, "usage.json"),
                session_limit=10, daily_limit=10,
            )
            budget.check_and_consume(8, user_id="u1")  # 剩 2
            client = _make_client()
            client.budget = budget
            with mock.patch("agent.deepseek_client.requests.post") as post:
                with self.assertRaises(BudgetExceededError):
                    client.invoke_text(system_prompt="s", user_prompt="u", max_tokens=1024)
                post.assert_not_called()  # 预算拦截, 不发请求

    def test_missing_api_key_raises(self):
        client = DeepSeekClient(DeepSeekSettings(api_key=""))
        with self.assertRaises(RuntimeError):
            client.invoke_text(system_prompt="s", user_prompt="u")

    def test_ledger_mode_does_not_retry_uncertain_disconnect(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(retries=3)
            client.ledger = ModelCallLedger(os.path.join(tmp, "calls.sqlite3"))
            with mock.patch(
                "agent.deepseek_client.requests.post",
                side_effect=requests.ConnectionError("accepted then disconnected"),
            ) as post:
                with self.assertRaises(ModelOutcomeUnknown):
                    client.invoke_text(system_prompt="s", user_prompt="u", user_id="u1")
            self.assertEqual(post.call_count, 1)
            with client.ledger._connect() as connection:
                rows = connection.execute("SELECT payload FROM model_calls").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(__import__("json").loads(rows[0][0])["state"],
                             ModelCallState.OUTCOME_UNKNOWN.value)

    def test_ledger_mode_counts_empty_output_retry_as_second_attempt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(retries=0)
            client.ledger = ModelCallLedger(os.path.join(tmp, "calls.sqlite3"))
            responses = [
                _fake_response(200, _chat_ok_message("")),
                _fake_response(200, _chat_ok_message("answer")),
            ]
            with mock.patch("agent.deepseek_client.requests.post", side_effect=responses) as post:
                self.assertEqual(client.invoke_text(system_prompt="s", user_prompt="u"), "answer")
            self.assertEqual(post.call_count, 2)
            with client.ledger._connect() as connection:
                attempts = connection.execute(
                    "SELECT attempt,state FROM model_calls ORDER BY attempt"
                ).fetchall()
            self.assertEqual(attempts, [(1, "succeeded"), (2, "succeeded")])


class TestSettingsFromEnv(unittest.TestCase):
    def test_from_env_overrides(self):
        env = {
            "DEEPSEEK_API_KEY": "sk-env-key",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_BASE_URL": "https://api.example.com",
            "DEEPSEEK_TIMEOUT": "30",
            "DEEPSEEK_MAX_TOKENS": "2048",
            "DEEPSEEK_TEMPERATURE": "0.5",
            "DEEPSEEK_ROUTER_STRICT": "1",
            "DEEPSEEK_RETRIES": "3",
            "DEEPSEEK_THINKING": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = DeepSeekSettings.from_env()
        self.assertEqual(settings.api_key, "sk-env-key")
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.base_url, "https://api.example.com")
        self.assertEqual(settings.timeout, 30.0)
        self.assertEqual(settings.max_tokens, 2048)
        self.assertEqual(settings.temperature, 0.5)
        self.assertTrue(settings.router_strict)
        self.assertEqual(settings.retries, 3)
        self.assertFalse(settings.thinking)

    def test_from_env_defaults(self):
        env = {
            "DEEPSEEK_API_KEY": "sk-env-key",  # 必须显式给, 否则踩硬编码默认值坑
            "DEEPSEEK_MODEL": "",
            "DEEPSEEK_ROUTER_STRICT": "0",
            "DEEPSEEK_THINKING": "false",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = DeepSeekSettings.from_env()
        self.assertEqual(settings.model, "deepseek-v4-flash")
        self.assertFalse(settings.router_strict)
        self.assertFalse(settings.thinking)


if __name__ == "__main__":
    unittest.main(verbosity=2)
