"""NLU provider admission through the request's durable model ledger."""
from __future__ import annotations

import hashlib
import json
import time

from agent.model_execution_context import current_operation
from agent.model_call_ledger import ModelCallLedger, ModelCallState, ModelOutcomeUnknown


class BudgetedNluProvider:
    supports_semantic_choice = True

    def __init__(self, provider, *, require_operation=False, session_limit=100_000, daily_limit=500_000):
        self.provider = provider
        self.require_operation = require_operation
        self.session_limit = session_limit
        self.daily_limit = daily_limit

    def complete(self, prompt, *, deadline, attempt_budget, response_schema=None, purpose="nlu_extract"):
        from nlu_v2.llm_extractor import ProviderResponse
        if attempt_budget < 1:
            return ProviderResponse("", attempts=0)
        if deadline <= time.monotonic():
            raise TimeoutError("nlu_deadline_before_dispatch")
        if purpose not in {"nlu_extract", "nlu_repair", "candidate"}:
            raise ValueError("invalid_nlu_purpose")
        execution = current_operation()
        if not execution:
            if self.require_operation:
                raise RuntimeError("nlu_operation_required")
            return self.provider.complete(prompt, deadline=deadline, attempt_budget=1,
                                          response_schema=response_schema)
        store, claim = execution
        ledger = ModelCallLedger(store.path)
        config = {name: getattr(self.provider, name, None) for name in (
            "model", "base_url", "max_response_tokens", "max_context_tokens")}
        payload = {"prompt": prompt, "schema": response_schema, "config": config}
        digest = store._operation_hmac(payload)
        logical_id = f"{claim.operation_id}:understand:{purpose}:{digest}"
        parts = claim.scope.split("\x1f")
        daily = store._operation_hmac(["token-daily-v1", parts[0]])
        session = store._operation_hmac(["token-session-v1", parts[:2]])
        record = ledger.prepare(
            provider="ollama", purpose=purpose, logical_request_id=logical_id, attempt=1,
            endpoint_identity=hashlib.sha256(str(config["base_url"]).encode()).hexdigest(),
            model=str(config["model"]), payload_digest=digest,
            estimated_tokens=len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                             + int(config["max_response_tokens"] or 1024) + 256,
            session_id=parts[0], operation_id=claim.operation_id, operation_owner=claim.owner,
            token_scopes={session: self.session_limit, daily: self.daily_limit}, daily_scope=daily,
        )
        if record.state == ModelCallState.SUCCEEDED:
            saved = ledger.load_response(record)
            return ProviderResponse(saved["text"], attempts=0, stop_reason=saved["stop_reason"])
        if record.state != ModelCallState.PREPARED:
            raise ModelOutcomeUnknown("nlu_attempt_not_replayable")
        ledger.transition(record.call_id, expected=ModelCallState.PREPARED,
                          target=ModelCallState.DISPATCH_INTENT,
                          operation_id=claim.operation_id, operation_owner=claim.owner)
        try:
            response = self.provider.complete(prompt, deadline=deadline, attempt_budget=1,
                                              response_schema=response_schema)
            text = response if isinstance(response, str) else response.text
            stop = "" if isinstance(response, str) else response.stop_reason
            saved = {"text": text, "stop_reason": stop}
            if not isinstance(response, str) and response.attempts != 1:
                raise RuntimeError("nlu_provider_attempt_contract_broken")
        except Exception:
            ledger.transition(record.call_id, expected=ModelCallState.DISPATCH_INTENT,
                              target=ModelCallState.OUTCOME_UNKNOWN, error_class="nlu_provider_failed")
            raise ModelOutcomeUnknown("nlu_provider_outcome_unknown") from None
        ledger.transition(record.call_id, expected=ModelCallState.DISPATCH_INTENT,
                          target=ModelCallState.SUCCEEDED, response_payload=saved,
                          response_digest=hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest())
        return ProviderResponse(text, attempts=1, stop_reason=stop)
