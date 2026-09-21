from __future__ import annotations

import json
import logging
import os
import time
import uuid
import hashlib
import contextvars
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import requests

from agent.agent_limits import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    TokenBudget,
    estimate_messages_tokens,
)
from agent.agent_models import parse_json_like
from agent.model_call_ledger import (
    ModelCallConflict, ModelCallLedger, ModelCallState, ModelOutcomeUnknown,
)
from agent.model_execution_context import current_operation, model_request_id

# LangSmith 可观测性(可选): 配置 LANGSMITH_TRACING=true + LANGSMITH_API_KEY 后生效;
# 未安装/未配置时完全无感, 不影响主流程
try:
    from langsmith import traceable as _langsmith_traceable

    _HAS_LANGSMITH = True
except ImportError:  # pragma: no cover
    _HAS_LANGSMITH = False


def _maybe_trace(fn):
    """条件装饰: 仅当 langsmith 可用且 LANGSMITH_TRACING=true 时启用 trace"""
    if _HAS_LANGSMITH and os.environ.get("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}:
        return _langsmith_traceable(run_type="llm", name=f"deepseek.{fn.__name__}")(fn)
    return fn

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# 网络错误/5xx 才重试; 4xx(参数/鉴权)不重试
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class DeepSeekSettings:
    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.1
    router_strict: bool = True
    retries: int = 2
    # 思考模式: True=v4-flash/reasoner 推理模式(质量高, 慢, 费token)
    # False=关闭思考(快/省; 环境变量 DEEPSEEK_THINKING=0)
    thinking: bool = True

    @classmethod
    def from_env(cls) -> "DeepSeekSettings":
        # 注意: 严禁把 API key 硬编码为默认值(会随代码泄露)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        return cls(
            api_key=api_key,
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
            or DEFAULT_DEEPSEEK_MODEL,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).strip()
            or DEFAULT_DEEPSEEK_BASE_URL,
            timeout=float(os.environ.get("DEEPSEEK_TIMEOUT", "60")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.1")),
            router_strict=os.environ.get("DEEPSEEK_ROUTER_STRICT", "0").strip()
            not in {"0", "false", "False"},
            retries=int(os.environ.get("DEEPSEEK_RETRIES", "2")),
            thinking=os.environ.get("DEEPSEEK_THINKING", "1").strip() not in {"0", "false", "False"},
        )


class DeepSeekClient:
    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        *,
        budget: TokenBudget | None = None,
        breaker: CircuitBreaker | None = None,
        ledger: ModelCallLedger | None = None,
    ) -> None:
        self.settings = settings or DeepSeekSettings.from_env()
        self.budget = budget
        self.breaker = breaker or CircuitBreaker()
        self.ledger = ledger
        self._last_attempts = contextvars.ContextVar(
            "deepseek_last_attempts", default=0
        )

    def last_attempt_count(self) -> int:
        return int(self._last_attempts.get())

    def require_api_key(self) -> None:
        if not self.settings.api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。"
                "请先设置环境变量: setx DEEPSEEK_API_KEY \"sk-...\" (PowerShell) "
                "或 export DEEPSEEK_API_KEY=sk-... (Linux/WSL)。"
            )

    def _check_budget(self, *, messages: list[dict[str, Any]], max_tokens: int, user_id: str) -> None:
        if self.budget is None:
            return
        estimated = estimate_messages_tokens(messages, max_tokens)
        allowed, reason = self.budget.check_and_consume(estimated, user_id=user_id)
        if not allowed:
            raise BudgetExceededError(reason)

    # ---------- 公开接口 ----------
    @_maybe_trace
    def invoke_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: Iterable[Mapping[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        user_id: str = "default",
        retry_empty: bool = True,
        thinking: bool | None = None,
    ) -> str:
        messages = self._build_messages(
            system_prompt=system_prompt, user_prompt=user_prompt, history=history
        )
        self.require_api_key()
        resolved_max_tokens = self.settings.max_tokens if max_tokens is None else max_tokens
        self._check_budget(messages=messages, max_tokens=resolved_max_tokens, user_id=user_id)

        logical_request_id = model_request_id("answer")
        message = self._invoke_chat(
            messages=messages,
            temperature=self.settings.temperature if temperature is None else temperature,
            max_tokens=resolved_max_tokens,
            user_id=user_id,
            logical_request_id=logical_request_id,
            purpose="answer",
            thinking=thinking,
        )
        text = self._extract_message_text(message)
        if not text.strip():
            if not retry_empty:
                raise RuntimeError("model_empty_response")
            # thinking 模式偶发空输出(reasoning 吃光 token): 重试一次
            logger.warning("DeepSeek 返回空内容(user=%s), 重试一次", user_id)
            self._check_budget(
                messages=messages, max_tokens=resolved_max_tokens, user_id=user_id
            )
            message = self._invoke_chat(
                messages=messages,
                temperature=self.settings.temperature if temperature is None else temperature,
                max_tokens=resolved_max_tokens,
                user_id=user_id,
                logical_request_id=logical_request_id,
                purpose="answer",
                force_new_attempt=True,
            )
            text = self._extract_message_text(message)
        return text

    @_maybe_trace
    def invoke_router(
        self,
        *,
        messages: list[dict[str, str]],
        tool_schema: dict[str, Any],
        user_id: str = "default",
    ) -> dict[str, Any]:
        self.require_api_key()
        self._check_budget(messages=messages, max_tokens=2048, user_id=user_id)

        strict = bool(self.settings.router_strict)
        base_url = self._beta_base_url(self.settings.base_url) if strict else self.settings.base_url
        logical_request_id = model_request_id("router")
        message = self._invoke_chat(
            messages=messages,
            temperature=0.0,
            # thinking 模式的 reasoning 会吃掉大量 token, 给足空间避免输出截断
            max_tokens=2048,
            tools=[tool_schema],
            # 注意: deepseek-v4-flash 的 thinking 模式不支持强制 tool_choice(指定函数名会 400)
            # 用 auto 让模型自行决定是否调用; 返回无 tool_calls 时走 content 解析兜底
            tool_choice="auto",
            base_url=base_url,
            user_id=user_id,
            # 意图路由是分类任务: 强制非思考(快/稳/省, 避开 reasoning 挤占 token)
            thinking=False,
            logical_request_id=logical_request_id,
            purpose="router",
        )
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function_payload = tool_calls[0].get("function", {})
            arguments = function_payload.get("arguments", {})
            if isinstance(arguments, str):
                if arguments.strip():
                    try:
                        return parse_json_like(arguments)
                    except Exception:
                        pass  # 非法 JSON, 继续尝试 content 兜底
            elif isinstance(arguments, Mapping):
                return dict(arguments)
            # arguments 为空/非法: 重试一次(thinking 关闭后偶发空输出)
            logger.warning("路由返回空 arguments, 重试一次")
            self._check_budget(messages=messages, max_tokens=2048, user_id=user_id)
            message = self._invoke_chat(
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                tools=[tool_schema],
                tool_choice="auto",
                base_url=base_url,
                user_id=user_id,
                thinking=False,
                logical_request_id=logical_request_id,
                purpose="router",
                force_new_attempt=True,
            )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                function_payload = tool_calls[0].get("function", {})
                arguments = function_payload.get("arguments", {})
                if isinstance(arguments, str) and arguments.strip():
                    try:
                        return parse_json_like(arguments)
                    except Exception:
                        pass
                elif isinstance(arguments, Mapping):
                    return dict(arguments)

        content = self._extract_message_text(message)
        if not content:
            return {}
        return parse_json_like(content)

    # ---------- 底层调用(重试 + 熔断) ----------
    def _invoke_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        user_id: str,
        base_url: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        thinking: bool | None = None,
        logical_request_id: str = "",
        purpose: str = "model",
        force_new_attempt: bool = False,
    ) -> dict[str, Any]:
        url = f"{(base_url or self.settings.base_url).rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        # 关闭思考模式(v4-flash 等支持 thinking.disabled)
        if (self.settings.thinking if thinking is None else thinking) is False:
            payload["thinking"] = {"type": "disabled"}

        execution = current_operation()
        ledger = ModelCallLedger(execution[0].path) if execution else self.ledger
        if ledger is not None:
            request_id = logical_request_id or uuid.uuid4().hex
            try:
                return self.breaker.call(lambda: self._ledger_request(
                    url=url, payload=payload, user_id=user_id,
                    logical_request_id=request_id,
                    purpose=purpose, force_new_attempt=force_new_attempt,
                    ledger=ledger,
                ))
            finally:
                self._last_attempts.set(len(ledger.list_for(request_id)))

        def _do_request() -> dict[str, Any]:
            data = self._post(url, payload)
            return data["choices"][0]["message"]

        # 熔断保护包裹整个重试过程: 熔断打开时直接拒绝
        try:
            return self.breaker.call(lambda: self._with_retries(_do_request, user_id))
        finally:
            # Legacy transport does not expose durable per-attempt accounting;
            # production composition always supplies ModelCallLedger.
            self._last_attempts.set(1)

    def _ledger_request(
        self, *, url: str, payload: dict[str, Any], user_id: str,
        logical_request_id: str, purpose: str, force_new_attempt: bool = False,
        ledger: ModelCallLedger | None = None,
    ) -> dict[str, Any]:
        ledger = ledger or self.ledger
        assert ledger is not None
        payload_digest = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        endpoint_identity = hashlib.sha256(url.encode("utf-8")).hexdigest()
        execution = current_operation()
        if execution:
            payload_digest = execution[0]._operation_hmac(payload)
        existing = ledger.list_for(logical_request_id)
        for prior in existing:
            if (prior.provider != "deepseek" or prior.purpose != purpose
                    or prior.endpoint_identity != endpoint_identity
                    or prior.model != self.settings.model
                    or prior.payload_digest != payload_digest
                    or prior.session_id != user_id):
                raise ModelCallConflict("逻辑请求标识已绑定不同的调用内容或会话")
        if existing and not force_new_attempt:
            latest = existing[-1]
            if latest.state == ModelCallState.SUCCEEDED:
                return ledger.load_response(latest)["choices"][0]["message"]
            if latest.state == ModelCallState.DISPATCH_INTENT:
                # Another worker may still own the HTTP attempt.  Wait only for
                # its durable terminal record; never rewrite a live intent as
                # OUTCOME_UNKNOWN from a competing request.
                for _ in range(100):
                    time.sleep(0.01)
                    current = ledger.get(latest.call_id)
                    if current is not None and current.state != ModelCallState.DISPATCH_INTENT:
                        latest = current
                        break
                if latest.state == ModelCallState.SUCCEEDED:
                    return ledger.load_response(latest)["choices"][0]["message"]
            if latest.state in {ModelCallState.DISPATCH_INTENT, ModelCallState.OUTCOME_UNKNOWN}:
                raise ModelOutcomeUnknown("模型调用结果不确定，已禁止自动重发。")
            if latest.state == ModelCallState.FAILED_FINAL:
                raise ModelCallConflict("模型调用已终止，不能重放")
        # PREPARED is a reserved logical attempt, not a completed failure.
        # Concurrent workers must contend for that same CAS record instead of
        # allocating attempt N+1 and producing a second HTTP dispatch.
        attempt = (
            existing[-1].attempt
            if existing and existing[-1].state == ModelCallState.PREPARED
            else len(existing) + 1
        )
        record = ledger.prepare(
            provider="deepseek", purpose=purpose,
            logical_request_id=logical_request_id, attempt=attempt,
            endpoint_identity=endpoint_identity,
            model=self.settings.model, payload_digest=payload_digest,
            estimated_tokens=(
                len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                + int(payload.get("max_tokens", 0)) + 256
                if execution else estimate_messages_tokens(
                    list(payload.get("messages") or []), int(payload.get("max_tokens", 0))
                )
            ),
            session_id=user_id,
            operation_id=execution[1].operation_id if execution else None,
            operation_owner=execution[1].owner if execution else None,
            token_scopes=(
                {
                    execution[0]._operation_hmac(["token-session-v1", execution[1].scope.split("\x1f")[:2]]):
                        getattr(self.budget, "session_limit", 100_000),
                    execution[0]._operation_hmac(["token-daily-v1", execution[1].scope.split("\x1f")[0]]):
                        getattr(self.budget, "daily_limit", 500_000),
                } if execution else None
            ),
            daily_scope=(execution[0]._operation_hmac(["token-daily-v1", execution[1].scope.split("\x1f")[0]]) if execution else None),
        )
        if record.state == ModelCallState.SUCCEEDED:
            return ledger.load_response(record)["choices"][0]["message"]
        if record.state == ModelCallState.DISPATCH_INTENT:
            for _ in range(500):
                time.sleep(0.01)
                current = ledger.get(record.call_id)
                if current is not None and current.state != ModelCallState.DISPATCH_INTENT:
                    record = current
                    break
            if record.state == ModelCallState.SUCCEEDED:
                return ledger.load_response(record)["choices"][0]["message"]
        if record.state in {ModelCallState.DISPATCH_INTENT, ModelCallState.OUTCOME_UNKNOWN}:
            raise ModelOutcomeUnknown("模型调用结果不确定，已禁止自动重发。")
        try:
            record = ledger.transition(
                record.call_id, expected=ModelCallState.PREPARED,
                target=ModelCallState.DISPATCH_INTENT,
                operation_id=execution[1].operation_id if execution else None,
                operation_owner=execution[1].owner if execution else None,
            )
        except ModelCallConflict:
            # A concurrent worker won the unique logical-attempt CAS.  Reuse
            # its durable response, or conservatively report uncertainty.
            for _ in range(100):
                time.sleep(0.01)
                current = ledger.get(record.call_id)
                if current is not None and current.state != ModelCallState.DISPATCH_INTENT:
                    if current.state == ModelCallState.SUCCEEDED:
                        return ledger.load_response(current)["choices"][0]["message"]
                    break
            raise ModelOutcomeUnknown("并发模型调用的持久化结果尚不确定。")
        try:
            data = self._post(url, payload)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            ledger.transition(
                record.call_id, expected=ModelCallState.DISPATCH_INTENT,
                target=ModelCallState.FAILED_FINAL,
                error_class=f"http_{status or 'unknown'}",
            )
            raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            ledger.transition(
                record.call_id, expected=ModelCallState.DISPATCH_INTENT,
                target=ModelCallState.OUTCOME_UNKNOWN,
                error_class=type(exc).__name__,
            )
            raise ModelOutcomeUnknown("模型调用结果不确定，已禁止自动重发。") from exc
        response_digest = hashlib.sha256(json.dumps(
            data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()
        usage = dict(data.get("usage") or {})
        ledger.transition(
            record.call_id, expected=ModelCallState.DISPATCH_INTENT,
            target=ModelCallState.SUCCEEDED,
            response_digest=response_digest,
            actual_tokens=int(usage.get("total_tokens", 0) or 0),
            response_payload=data,
        )
        return data["choices"][0]["message"]

    def _with_retries(self, fn, user_id: str) -> dict[str, Any]:
        """指数退避重试: 网络错误/5xx 重试, 4xx 直接抛"""
        last_exc: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                return fn()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in _RETRYABLE_STATUS:
                    raise
                last_exc = exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
            if attempt < self.settings.retries:
                delay = 1.5 * (2 ** attempt)
                logger.warning(
                    "DeepSeek 调用失败(第%d次, user=%s): %s; %.1fs 后重试",
                    attempt + 1, user_id, last_exc, delay,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _post(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.settings.timeout)
        response.raise_for_status()
        return response.json()

    # ---------- 工具方法 ----------
    @staticmethod
    def _build_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        history: Iterable[Mapping[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for message in history or []:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _extract_message_text(message: Mapping[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
            return "\n".join(part.strip() for part in parts if part).strip()
        return str(content).strip()

    @staticmethod
    def _beta_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/beta"):
            return normalized
        return f"{normalized}/beta"


def dump_json_payload(payload: Mapping[str, Any]) -> str:
    """调试辅助: 安全打印请求体(截断)"""
    return json.dumps(payload, ensure_ascii=False)[:2000]
