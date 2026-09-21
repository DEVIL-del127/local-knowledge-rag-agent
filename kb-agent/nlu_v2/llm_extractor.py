"""Bounded, optional LLM extraction of candidate IR fragments."""
from __future__ import annotations

import json
import inspect
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol

from .patch_protocol import (
    GapRequest, SemanticPatchV1, atomic_operation_type, atomic_response_schema,
    parse_atomic_patch_text, parse_patch_text,
)
from .semantic_candidates import (
    CANDIDATE_CHOICE_PROTOCOL,
    CandidateDecision, CandidateMenu,
    SemanticCandidate,
    parse_candidate_decision,
)


PROMPT_VERSION = "semantic-patch-v2.4.0"
CANDIDATE_SEMANTIC_GUARD_VERSION = "candidate-semantic-guard-v1"
DEFAULT_LLM_TOTAL_DEADLINE_SECONDS = 75.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 70.0
DEFAULT_MAX_RESPONSE_TOKENS = 512
SYSTEM_PROMPT = (
    "你是企业只读请求的语义解析器。你只输出符合用户消息约束的 JSON 对象。"
    "禁止回显 schema、枚举说明或占位符，禁止输出工具调用、执行计划、答案和解释。"
    "每个枚举字段只能选择一个允许值，任何包含竖线字符 | 的值都是无效输出。"
)


class LLMProvider(Protocol):
    """Providers must honor attempt_budget and include transport retries in attempts."""

    def complete(self, prompt: str, *, deadline: float,
                 attempt_budget: int,
                 response_schema: Mapping[str, Any] | None = None) -> "ProviderResponse | str":
        ...


@dataclass(slots=True)
class ProviderResponse:
    text: str
    attempts: int = 1
    stop_reason: str = ""


@dataclass(slots=True)
class LLMCandidate:
    payload: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    error: str = ""
    valid_json: bool = False
    patch: SemanticPatchV1 | None = None
    protocol: str = ""
    failure_kind: str = ""  # provider / schema / capacity / none
    prompt_chars: int = 0
    response_chars: int = 0
    raw_response_excerpt: str = ""
    parsed_json_excerpt: str = ""
    selected_gap_ids: list[str] = field(default_factory=list)
    atomic_operation: str = ""
    stop_reason: str = ""


class OllamaOpenAIProvider:
    """Local Ollama provider with no hidden retries or remote fallback."""

    supports_semantic_choice = True

    def __init__(self, model: str | None = None,
                 base_url: str | None = None, timeout: float | None = None,
                 *, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
                 read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
                 max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
                 max_context_tokens: int = 3072):
        self.model = model or os.environ.get("KB_NLU_LLM_MODEL", "qwen2.5:7b")
        self.base_url = (
            base_url or os.environ.get("KB_OLLAMA_URL")
            or os.environ.get("OLLAMA_URL") or _default_ollama_url()
        ).rstrip("/")
        # ``timeout`` is retained as a compatibility alias for callers that
        # previously supplied one combined requests timeout.  New callers get
        # independent connect/read budgets, always further bounded by the
        # extractor's total deadline.
        if timeout is not None:
            read_timeout = timeout
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.read_timeout = max(0.1, float(read_timeout))
        self.timeout = self.read_timeout
        self.max_response_tokens = max(64, min(int(max_response_tokens), 1024))
        self.max_context_tokens = max(1024, min(int(max_context_tokens), 4096))

    def complete(self, prompt: str, *, deadline: float,
                 attempt_budget: int,
                 response_schema: Mapping[str, Any] | None = None) -> ProviderResponse:
        if attempt_budget < 1:
            return ProviderResponse("", attempts=0)
        import requests

        available = deadline - time.monotonic()
        if available <= 0:
            raise TimeoutError("Ollama deadline exceeded before request")
        connect_timeout = max(0.1, min(self.connect_timeout, available))
        read_timeout = max(0.1, min(self.read_timeout, available))
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": dict(response_schema) if response_schema else "json",
                "keep_alive": "10m",
                "options": {
                    "temperature": 0, "seed": 0,
                    "num_ctx": self.max_context_tokens,
                    "num_predict": self.max_response_tokens,
                },
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=(connect_timeout, read_timeout),
        )
        response.raise_for_status()
        data = response.json()
        return ProviderResponse(
            str(data.get("message", {}).get("content", "")), attempts=1,
            stop_reason=str(data.get("done_reason") or ("stop" if data.get("done") else "")),
        )

    def health(self) -> dict[str, Any]:
        import requests

        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=(min(self.connect_timeout, 5.0), min(self.read_timeout, 5.0)),
        )
        response.raise_for_status()
        models = [str(item.get("name", "")) for item in response.json().get("models", [])]
        return {
            "base_url": self.base_url,
            "model": self.model,
            "available": self.model in models or f"{self.model}:latest" in models,
            "models": models,
        }


def _guard_candidate_semantics(
    menu: CandidateMenu,
) -> tuple[CandidateDecision, str] | None:
    """Apply conservative source-to-payload invariants after an LLM choice.

    This guard is deliberately independent of oracle ids, candidate digests and
    candidate ordering.  It recognizes only typed payload families whose source
    constraints can be checked without inventing missing context.  Unrecognized
    menus keep the provider decision unchanged.
    """
    candidates = list(menu.candidates)
    if not candidates:
        return None
    payloads = [dict(item.semantic_payload) for item in candidates]
    if all(str(item.get("op", "")) == "resolve_reference" for item in payloads):
        return _guard_reference_candidates(menu.source_clause, candidates, payloads)
    if all("field" in item and "cmp" in item for item in payloads):
        return _guard_event_candidates(menu.source_clause, candidates, payloads)
    if all("inputs" in item and "grain" in item for item in payloads):
        return _guard_set_candidates(menu.source_clause, candidates, payloads)
    if all("measure" in item for item in payloads):
        return _guard_aggregate_candidates(menu.source_clause, candidates, payloads)
    if all("operands" in item for item in payloads):
        return _guard_calculation_candidates(menu.source_clause, candidates, payloads)
    return None


def _guard_event_candidates(source: str, candidates, payloads):
    lowered = source.casefold()
    field_aliases = {
        "temperature": ("温度", "temperature"),
        "pressure": ("压力", "pressure"),
        "speed": ("转速", "speed"),
        "acceleration": ("加速度", "acceleration"),
        "noise": ("噪音", "噪声", "noise"),
    }
    fields = _mentioned_identifiers(lowered, field_aliases)
    if ("或" in lowered or " or " in f" {lowered} ") and len(fields) > 1:
        return CandidateDecision("ambiguous"), "explicit_alternative_fields"
    survivors = list(range(len(candidates)))
    if fields:
        survivors = [i for i in survivors if str(payloads[i].get("field")) in fields]
    if "边界" in lowered or "boundary" in lowered:
        comparisons = {str(payloads[i].get("cmp")) for i in survivors}
        if len(comparisons) > 1:
            return CandidateDecision("ambiguous"), "comparison_boundary_unspecified"
    expected_cmp = (
        "gt" if re.search(r"超过|高于|大于|\babove\b|\bover\b", lowered) else
        "lt" if re.search(r"低于|小于|\bbelow\b|\bunder\b", lowered) else None
    )
    if expected_cmp:
        survivors = [i for i in survivors if str(payloads[i].get("cmp")) == expected_cmp]
    if "额定" in lowered or "rated" in lowered:
        percent = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
        expected_factor = float(percent.group(1)) / 100.0 if percent else None
        survivors = [
            i for i in survivors
            if "rated" in str(payloads[i].get("baseline", "")).casefold()
            and (expected_factor is None or abs(float(payloads[i].get("factor", -1)) - expected_factor) < 1e-9)
        ]
    expected_unit = next((canonical for pattern, canonical in (
        (r"m/s(?:²|2)", "m/s2"), (r"mpa", "mpa"), (r"kpa", "kpa"),
        (r"db", "db"), (r"℃|°c|celsius", "celsius"), (r"rpm", "rpm"),
    ) if re.search(pattern, lowered, re.I)), None)
    if expected_unit:
        survivors = [
            i for i in survivors
            if str(payloads[i].get("unit", "")).casefold() == expected_unit
        ]
    duration = re.search(r"(\d+(?:\.\d+)?)\s*(分钟|分|min(?:ute)?s?|秒|s(?:ec(?:ond)?s?)?|小时|h(?:ours?)?)", lowered)
    if duration:
        value = float(duration.group(1))
        unit = duration.group(2)
        multiplier = 60 if re.match(r"分钟|分|min", unit) else 3600 if re.match(r"小时|h", unit) else 1
        expected_seconds = value * multiplier
        survivors = [
            i for i in survivors
            if abs(float(payloads[i].get("duration_s", -1)) - expected_seconds) < 1e-9
        ]
    return _guard_survivors(candidates, survivors, "event_constraints")


def _guard_set_candidates(source: str, candidates, payloads):
    lowered = source.casefold()
    survivors = list(range(len(candidates)))
    expected_op = (
        "difference" if re.search(r"排除|剔除|差集|difference", lowered) else
        "union" if re.search(r"并集|union", lowered) else
        "intersection" if re.search(r"交集|重合|intersection", lowered) else None
    )
    if expected_op:
        survivors = [i for i in survivors if str(payloads[i].get("op")) == expected_op]
    elif re.search(r"组合|combine", lowered):
        operations = {str(payloads[i].get("op")) for i in survivors}
        if len(operations) > 1:
            return CandidateDecision("ambiguous"), "set_operation_unspecified"
    expected_grain = (
        "interval" if re.search(r"时间段|时段|区间|interval", lowered) else
        "date" if re.search(r"日期|date", lowered) else None
    )
    if expected_grain:
        survivors = [i for i in survivors if str(payloads[i].get("grain")) == expected_grain]
    elif len({str(payloads[i].get("grain")) for i in survivors}) > 1:
        return CandidateDecision("ambiguous"), "set_grain_unspecified"
    if re.search(r"两类|two\s+(?:classes|types)", lowered):
        survivors = [
            i for i in survivors
            if len(list(payloads[i].get("inputs") or [])) == 2
            and len(set(payloads[i].get("inputs") or [])) == 2
        ]
    if expected_op == "difference" and re.search(r"a.*排除.*b|a.*exclude.*b", lowered, re.I):
        survivors = [
            i for i in survivors
            if len(payloads[i].get("inputs") or []) == 2
            and str(payloads[i]["inputs"][0]).casefold().endswith("a")
            and str(payloads[i]["inputs"][1]).casefold().endswith("b")
        ]
    return _guard_survivors(candidates, survivors, "set_constraints")


def _guard_aggregate_candidates(source: str, candidates, payloads):
    lowered = source.casefold()
    if re.search(r"两个指标|two\s+(?:metrics|measures)", lowered):
        measures = {str(item.get("measure")) for item in payloads}
        if len(measures) > 1:
            return CandidateDecision("ambiguous"), "aggregate_measure_unspecified"
    survivors = list(range(len(candidates)))
    expected_op = (
        "stddev" if re.search(r"标准差|波动率|stddev|standard deviation", lowered) else
        "max" if re.search(r"峰值|最大值|\bmax(?:imum)?\b", lowered) else
        "min" if re.search(r"最小值|\bmin(?:imum)?\b", lowered) else
        "sum" if re.search(r"总额|总和|合计|\bsum\b", lowered) else
        "avg" if re.search(r"平均|均值|\bavg\b|\bmean\b", lowered) else None
    )
    if expected_op:
        survivors = [i for i in survivors if str(payloads[i].get("op")) == expected_op]
    measure_aliases = {
        "efficiency": ("转换效率", "efficiency"),
        "irradiance": ("辐照度", "irradiance"),
        "total_spend": ("累计消费金额", "消费金额", "total spend"),
        "sales_amount": ("销售总额", "销售额", "sales amount"),
        "order_count": ("订单数", "order count"),
        "acceleration": ("加速度", "acceleration"),
        "voltage": ("电压", "voltage"),
        "temperature": ("温度", "temperature"),
        "pressure": ("压力", "pressure"),
    }
    measures = _mentioned_identifiers(lowered, measure_aliases)
    if measures:
        survivors = [i for i in survivors if str(payloads[i].get("measure")) in measures]
    if re.search(r"筛选用户|filtered users?", lowered):
        survivors = [
            i for i in survivors
            if "filtered" in str(payloads[i].get("scope", "")).casefold()
        ]
    return _guard_survivors(candidates, survivors, "aggregate_constraints")


def _guard_calculation_candidates(source: str, candidates, payloads):
    lowered = source.casefold()
    if re.search(r"两者的比值|两个分组均值的差|ratio of (?:the )?two|difference between", lowered):
        return CandidateDecision("ambiguous"), "noncommutative_order_unspecified"
    survivors = list(range(len(candidates)))
    cumulative_ratio = bool(re.search(r"累计时长.*(?:占|相对).*总时长|cumulative duration.*total duration", lowered))
    expected_ops = (
        {"cumulative_duration_ratio", "ratio"} if cumulative_ratio else
        {"correlation"} if re.search(r"相关系数|correlation", lowered) else
        {"stddev"} if re.search(r"标准差|波动率|stddev|standard deviation", lowered) else
        {"difference"} if re.search(r"减去|差值|difference|minus", lowered) else
        {"ratio"} if re.search(r"比值|比例|除以|ratio|divide", lowered) else set()
    )
    if expected_ops:
        survivors = [i for i in survivors if str(payloads[i].get("op")) in expected_ops]
    operand_aliases = {
        "max_temperature": ("温度峰值", "maximum temperature"),
        "max_speed": ("转速峰值", "maximum speed"),
        "defect_density": ("缺陷密度", "defect density"),
        "yield_rate": ("良率", "yield rate"),
        "filtered_avg": ("筛选均值", "filtered average"),
        "global_avg": ("全站均值", "global average"),
        "abnormal_duration": ("异常累计时长", "abnormal duration"),
        "period_duration": ("当日总时长", "周期总时长", "period duration", "total duration"),
        "acceleration": ("加速度", "acceleration"),
        "voltage": ("电压", "voltage"),
        "temperature": ("温度", "temperature"),
        "pressure": ("压力", "pressure"),
    }
    operands = _mentioned_identifiers_in_order(lowered, operand_aliases)
    if operands:
        if expected_ops == {"correlation"}:
            expected_set = set(operands)
            survivors = [
                i for i in survivors
                if len(payloads[i].get("operands") or []) == len(expected_set)
                and set(payloads[i].get("operands") or []) == expected_set
            ]
        else:
            survivors = [i for i in survivors if list(payloads[i].get("operands") or []) == operands]
    return _guard_survivors(candidates, survivors, "calculation_constraints")


def _guard_reference_candidates(source: str, candidates, payloads):
    lowered = source.casefold()
    valid = [
        i for i, payload in enumerate(payloads)
        if str(payload.get("lineage", "")).casefold() not in {"stale", "missing"}
    ]
    if not valid:
        return CandidateDecision("none_of_above"), "reference_lineage_invalid"
    if re.search(r"这些结果|these results", lowered):
        return CandidateDecision("ambiguous"), "reference_type_unspecified"
    if re.search(r"(?:在)?交集内|within (?:the )?intersection", lowered) and not re.search(
        r"日期|时间段|时段|区间|date|interval", lowered,
    ):
        types = {str(payloads[i].get("type")) for i in valid}
        if len(types) > 1:
            return CandidateDecision("ambiguous"), "intersection_reference_grain_unspecified"
    expected_type = (
        "interval" if re.search(r"时间段|时段|区间|interval", lowered) else
        "date" if re.search(r"日期|date", lowered) else
        "entity_set" if re.search(r"用户|users?", lowered) else
        "number" if re.search(r"平均值|均值|average|mean", lowered) else None
    )
    if expected_type:
        valid = [i for i in valid if str(payloads[i].get("type")) == expected_type]
    if re.search(r"上述用户|这些用户|above users?|these users?", lowered):
        narrowed = [
            i for i in valid
            if not re.search(r"(?:^|_)(?:all|global)(?:_|$)", str(payloads[i].get("target", "")), re.I)
        ]
        if narrowed:
            valid = narrowed
    return _guard_survivors(candidates, valid, "reference_constraints")


def _guard_survivors(candidates, survivors: list[int], reason: str):
    survivors = list(dict.fromkeys(survivors))
    if len(survivors) == 1:
        return CandidateDecision("select", candidates[survivors[0]].candidate_id), reason
    if not survivors:
        return CandidateDecision("none_of_above"), reason
    return CandidateDecision("ambiguous"), reason


def _mentioned_identifiers(source: str, aliases: Mapping[str, tuple[str, ...]]) -> set[str]:
    return {
        identifier for identifier, values in aliases.items()
        if any(value.casefold() in source for value in values)
    }


def _mentioned_identifiers_in_order(
    source: str, aliases: Mapping[str, tuple[str, ...]],
) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    for identifier, values in aliases.items():
        for value in values:
            start = source.find(value.casefold())
            if start >= 0:
                matches.append((start, start + len(value), identifier))
    selected: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, identifier in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        selected.append((start, end, identifier))
        occupied.append((start, end))
    return [identifier for _, _, identifier in sorted(selected)]


class BoundedLLMExtractor:
    def __init__(self, provider: LLMProvider, *, timeout: float = DEFAULT_LLM_TOTAL_DEADLINE_SECONDS,
                 max_remote_attempts: int = 2, allow_json_repair: bool = False,
                 max_concurrency: int = 4, max_query_chars: int = 12000,
                 max_catalog_chars: int = 12000, max_existing_ir_chars: int = 12000,
                 max_prompt_chars: int = 36000):
        self.provider = provider
        self.timeout = max(60.0, min(float(timeout), 90.0))
        self.max_remote_attempts = max(1, min(max_remote_attempts, 2))
        self.allow_json_repair = allow_json_repair
        self.max_query_chars = max(512, int(max_query_chars))
        self.max_catalog_chars = max(1024, int(max_catalog_chars))
        self.max_existing_ir_chars = max(1024, int(max_existing_ir_chars))
        self.max_prompt_chars = max(4096, int(max_prompt_chars))
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency))

    def cache_identity(self) -> str:
        model = str(getattr(self.provider, "model", type(self.provider).__name__))
        endpoint = str(getattr(self.provider, "base_url", "local-provider"))
        return (
            f"{PROMPT_VERSION}:{model}:{endpoint}:"
            f"repair={int(self.allow_json_repair)}"
        )

    def choose_candidate(self, menu: CandidateMenu) -> LLMCandidate:
        """Perform one opaque candidate-ID choice with no remote repair."""
        deadline = time.monotonic() + self.timeout
        aliases = {
            f"choice_{index}": item.candidate_id
            for index, item in enumerate(menu.candidates, start=1)
        }
        prompt = self._candidate_choice_prompt(menu, aliases)
        audit = {
            "prompt_chars": len(prompt),
            "selected_gap_ids": [menu.gap_id],
            "atomic_operation": "candidate_select",
        }
        if len(prompt) > self.max_prompt_chars:
            return LLMCandidate(
                attempts=0, error="candidate choice prompt exceeds bounded size",
                failure_kind="capacity", protocol=CANDIDATE_CHOICE_PROTOCOL,
                stop_reason="prompt_capacity", **audit,
            )
        if not self._semaphore.acquire(timeout=self.timeout):
            return LLMCandidate(
                attempts=0, error="LLM concurrency limit reached",
                failure_kind="capacity", protocol=CANDIDATE_CHOICE_PROTOCOL,
                stop_reason="concurrency_capacity", **audit,
            )
        schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "const": "select"},
                        "candidate_id": {"type": "string", "enum": list(aliases)},
                    },
                    "required": ["decision", "candidate_id"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": [
                            "none_of_above", "ambiguous",
                        ]},
                    },
                    "required": ["decision"],
                    "additionalProperties": False,
                },
            ],
        }
        try:
            response = self._complete(
                prompt, deadline=deadline, attempt_budget=1, response_schema=schema, purpose="candidate",
            )
            text, used, stop_reason = self._unwrap(response)
            alias_candidates = tuple(
                SemanticCandidate(
                    alias, item.operation_type, item.semantic_payload,
                    item.evidence_summary,
                )
                for alias, item in zip(aliases, menu.candidates)
            )
            alias_menu = replace(menu, candidates=alias_candidates)
            provider_decision = parse_candidate_decision(text, alias_menu)
            decision = provider_decision
            guard_reason = ""
            if not provider_decision.error:
                guarded = _guard_candidate_semantics(alias_menu)
                if guarded is not None:
                    decision, guard_reason = guarded
            resolved_candidate_id = aliases.get(decision.candidate_id, "")
            payload = {
                "decision": decision.decision,
                **({"candidate_id": resolved_candidate_id} if resolved_candidate_id else {}),
            }
            return LLMCandidate(
                payload=payload if not decision.error else {},
                attempts=max(1, used), error=decision.error,
                valid_json=not decision.error, protocol=CANDIDATE_CHOICE_PROTOCOL,
                failure_kind="" if not decision.error else "schema",
                response_chars=len(text), raw_response_excerpt=_audit_excerpt(text),
                parsed_json_excerpt=_parsed_json_excerpt(text),
                stop_reason=(
                    f"{stop_reason or 'stop'};{CANDIDATE_SEMANTIC_GUARD_VERSION}:{guard_reason}"
                    if guard_reason else
                    stop_reason or ("choice_rejected" if decision.error else "stop")
                ),
                **audit,
            )
        except Exception as exc:
            return LLMCandidate(
                attempts=1, error=f"{type(exc).__name__}: {exc}",
                failure_kind="provider", protocol=CANDIDATE_CHOICE_PROTOCOL,
                stop_reason="provider_error", **audit,
            )
        finally:
            self._semaphore.release()

    @staticmethod
    def _candidate_choice_prompt(menu: CandidateMenu,
                                 aliases: Mapping[str, str]) -> str:
        candidates = [
            {
                "candidate_id": alias,
                "evidence_summary": item.evidence_summary,
                "semantic_payload": item.semantic_payload,
            }
            for alias, item in zip(aliases, menu.candidates)
        ]
        return (
            "你只负责从本地编译器提供的语义候选中选择，不得创建或修改候选。"
            "如果恰好一个候选符合原文，返回 select 和其 candidate_id；"
            "如果都不符合，返回 none_of_above；如果原文无法区分多个候选，返回 ambiguous。"
            "对于比值、除法、差值等非交换运算，如果候选仅操作数顺序不同，而原文只说‘两者’、"
            "‘A和B’且未明确‘A除以B’、分子分母或比较方向，必须返回 ambiguous，不得任选。"
            "如果原文明示‘A除以B’、‘A占B’或‘A相对B’，应选择保持该左右顺序的唯一候选。"
            "解析‘这些日期/时间段/结果’等引用时，必须按 semantic_payload.type 选择分别为"
            " date/interval/value 的唯一匹配候选；类型不匹配的候选不符合原文。"
            "按以下语义判据逐项核对：原文只说‘边界’而候选区分 > 与 >= 时是 ambiguous；"
            "只说‘组合/重合结果’而候选区分交并集或日期/区间粒度时也是 ambiguous。"
            "‘交集时间段’明确要求 interval，不得选择 date；‘峰值’等于最大值。"
            "聚合与相关系数候选必须保留原文的被聚合字段及全部操作数；没有一个完整匹配时返回"
            " none_of_above，不得因两个都错而返回 ambiguous。"
            "指代词的类型具有约束力：‘这些日期’只能指 date，‘上述用户’指前文筛选得到的用户集合，"
            "‘该平均值’只能指数值；若‘这些结果/交集’没有足够词语区分两个不同类型候选，则返回"
            " ambiguous，而不是 none_of_above。"
            "只输出一个 JSON 对象，不得输出解释、Markdown、字段、公式、IR 或 Patch。\n"
            f"PROTOCOL={CANDIDATE_CHOICE_PROTOCOL}\n"
            f"SOURCE_CLAUSE={json.dumps(menu.source_clause, ensure_ascii=False)}\n"
            f"TARGET={json.dumps(menu.target_summary, ensure_ascii=False, sort_keys=True)}\n"
            f"CANDIDATES={json.dumps(candidates, ensure_ascii=False)}"
        )

    def extract(self, query: str, catalog_payload: list[dict[str, Any]],
                existing_ir: Mapping[str, Any] | None = None, *,
                gaps: list[GapRequest] | None = None,
                base_ir_digest: str = "",
                enable_patch_v1: bool = False,
                enable_atomic_patch_v2: bool = False) -> LLMCandidate:
        deadline = time.monotonic() + self.timeout
        bounded_query = _truncate_text(query, self.max_query_chars)
        bounded_catalog = _compact_catalog_payload(catalog_payload, self.max_catalog_chars)
        bounded_existing_ir = _compact_mapping(existing_ir or {}, self.max_existing_ir_chars)
        gap_requests = list(gaps or [])
        if not bool(getattr(self.provider, "supports_semantic_choice", False)):
            gap_requests = [
                item.model_copy(update={"response_mode": "legacy_node"})
                for item in gap_requests
            ]
        # A connected RepairUnit can require several dependent operations.
        # Atomic v2 is only valid for a single-gap unit; multi-gap units use
        # the typed V1 operation list and are submitted atomically together.
        atomic_gap = gap_requests[0] if enable_atomic_patch_v2 and len(gap_requests) == 1 else None
        response_schema = (
            atomic_response_schema(atomic_gap) if atomic_gap is not None
            else SemanticPatchV1.model_json_schema() if enable_patch_v1 else None
        )
        prompt = (
            self._atomic_prompt(
                bounded_query, bounded_catalog, bounded_existing_ir, atomic_gap, base_ir_digest,
            ) if atomic_gap is not None else self._patch_prompt(
                bounded_query, bounded_catalog, bounded_existing_ir, gap_requests, base_ir_digest,
            )
            if enable_patch_v1 else self._prompt(bounded_query, bounded_catalog, bounded_existing_ir)
        )
        audit = {
            "prompt_chars": len(prompt),
            "selected_gap_ids": [item.gap_id for item in gap_requests],
            "atomic_operation": atomic_operation_type(atomic_gap) if atomic_gap else "",
        }
        if len(prompt) > self.max_prompt_chars:
            return LLMCandidate(
                error=(f"LLM prompt exceeds bounded size "
                       f"({len(prompt)}>{self.max_prompt_chars})"),
                failure_kind="capacity", stop_reason="prompt_capacity", **audit,
            )
        attempts = 0
        if not self._semaphore.acquire(timeout=self.timeout):
            return LLMCandidate(
                error="LLM concurrency limit reached", failure_kind="capacity",
                stop_reason="concurrency_capacity", **audit,
            )
        try:
            # Count the request before transport starts so timeout/error paths remain auditable.
            before = attempts
            attempts += 1
            response = self._complete(
                prompt, deadline=deadline,
                # A transport/provider failure is terminal.  A separate,
                # schema-only format repair below is the sole second call.
                attempt_budget=1,
                response_schema=response_schema,
            )
            text, used, stop_reason = self._unwrap(response)
            parsed_json_excerpt = _parsed_json_excerpt(text)
            attempts = before + max(1, used)
            if enable_patch_v1:
                parsed, protocol, error = (
                    parse_atomic_patch_text(text, base_ir_digest=base_ir_digest, gap=atomic_gap)
                    if atomic_gap is not None else
                    parse_patch_text(text, base_ir_digest=base_ir_digest, gaps=gap_requests)
                )
                if parsed is not None:
                    return LLMCandidate(
                        payload=parsed.model_dump(mode="json"), attempts=attempts,
                        valid_json=True, patch=parsed, protocol=protocol,
                        response_chars=len(text), raw_response_excerpt=_audit_excerpt(text),
                        parsed_json_excerpt=parsed_json_excerpt,
                        stop_reason=stop_reason, **audit,
                    )
            else:
                payload = _parse_json(text)
                error = "" if payload is not None else "invalid LLM JSON"
                if payload is not None:
                    return LLMCandidate(
                        payload=payload, attempts=attempts, valid_json=True,
                        response_chars=len(text), raw_response_excerpt=_audit_excerpt(text),
                        parsed_json_excerpt=parsed_json_excerpt,
                        stop_reason=stop_reason, **audit,
                    )
            # A second request is a formatting-only repair.  A syntactically
            # valid response that violates Patch semantics is not repaired or
            # retried: that would be a semantic retry and can manufacture new
            # facts under a provider failure budget.
            json_syntax_damaged = not _has_valid_json_syntax(text)
            if (not self.allow_json_repair or not json_syntax_damaged
                    or attempts >= self.max_remote_attempts):
                return LLMCandidate(
                    attempts=attempts, error=error or "invalid LLM JSON", failure_kind="schema",
                    response_chars=len(text), raw_response_excerpt=_audit_excerpt(text),
                    parsed_json_excerpt=parsed_json_excerpt,
                    stop_reason=stop_reason or "schema_rejected", **audit,
                )
            if time.monotonic() >= deadline:
                return LLMCandidate(
                    attempts=attempts, error="LLM deadline exceeded before repair",
                    failure_kind="provider", response_chars=len(text),
                    raw_response_excerpt=_audit_excerpt(text),
                    parsed_json_excerpt=parsed_json_excerpt,
                    stop_reason="deadline", **audit,
                )
            before = attempts
            attempts += 1
            repair = self._complete(
                self._repair_prompt(text, response_schema), deadline=deadline,
                attempt_budget=1,
                response_schema=response_schema,
                purpose="nlu_repair",
            )
            repaired_text, used, repaired_stop_reason = self._unwrap(repair)
            repaired_json_excerpt = _parsed_json_excerpt(repaired_text)
            attempts = before + max(1, used)
            if enable_patch_v1:
                parsed, protocol, error = (
                    parse_atomic_patch_text(repaired_text, base_ir_digest=base_ir_digest, gap=atomic_gap)
                    if atomic_gap is not None else
                    parse_patch_text(repaired_text, base_ir_digest=base_ir_digest, gaps=gap_requests)
                )
                return LLMCandidate(
                    payload=parsed.model_dump(mode="json") if parsed else {},
                    attempts=attempts, error="" if parsed else error,
                    valid_json=parsed is not None, patch=parsed, protocol=protocol,
                    failure_kind="" if parsed else "schema",
                    response_chars=len(repaired_text),
                    raw_response_excerpt=_audit_excerpt(repaired_text),
                    parsed_json_excerpt=repaired_json_excerpt,
                    stop_reason=repaired_stop_reason or stop_reason, **audit,
                )
            payload = _parse_json(repaired_text)
            return LLMCandidate(
                payload=payload or {}, attempts=attempts,
                error="" if payload is not None else "invalid repaired LLM JSON",
                valid_json=payload is not None,
                failure_kind="" if payload is not None else "schema",
                response_chars=len(repaired_text),
                raw_response_excerpt=_audit_excerpt(repaired_text),
                parsed_json_excerpt=repaired_json_excerpt,
                stop_reason=repaired_stop_reason or stop_reason, **audit,
            )
        except Exception as exc:
            return LLMCandidate(attempts=attempts, error=f"{type(exc).__name__}: {exc}",
                                failure_kind="provider", stop_reason="provider_error", **audit)
        finally:
            self._semaphore.release()

    @staticmethod
    def _unwrap(response: ProviderResponse | str) -> tuple[str, int, str]:
        if isinstance(response, ProviderResponse):
            return response.text, max(0, response.attempts), response.stop_reason
        return str(response), 1, ""

    def _complete(self, prompt: str, *, deadline: float, attempt_budget: int,
                  response_schema: Mapping[str, Any] | None, purpose: str = "nlu_extract") -> ProviderResponse | str:
        """Pass JSON Schema when supported without breaking frozen local providers."""
        parameters = inspect.signature(self.provider.complete).parameters
        kwargs = {"deadline": deadline, "attempt_budget": attempt_budget}
        if "purpose" in parameters:
            kwargs["purpose"] = purpose
        if "response_schema" in parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        ):
            kwargs["response_schema"] = response_schema
        return self.provider.complete(prompt, **kwargs)

    @staticmethod
    def _patch_prompt(query: str, catalog_payload: list[dict[str, Any]],
                      existing_ir: Mapping[str, Any], gaps: list[GapRequest],
                      base_ir_digest: str) -> str:
        allowed = sorted({value for gap in gaps for value in gap.allowed_operations})
        return (
            "你是企业只读请求的语义缺口修补器，不是整题解析器。只处理 GAP_REQUESTS 中列出的缺口，"
            "不得改写、删除或重复 EXISTING_IR 节点。输出必须符合 SemanticPatchV1 JSON Schema。"
            "base_ir_digest 必须逐字复制 BASE_IR_DIGEST；每个 operation.gap_id 必须来自 GAP_REQUESTS，"
            "operation_type 必须同时出现在该 gap.allowed_operations 和 ALLOWED_OPERATIONS。"
            "每个 operation.anchor_ids 必须从对应 gap.allowed_anchor_ids 中选择，不能为空。"
            "所有事实必须引用 QUERY 中逐字一致的 evidence；可省略 start/end，但 evidence.text 在 QUERY 中"
            "必须唯一出现。只能引用 gap 允许的 Catalog symbol、hypothesis、metric 和 output ref。"
            "依赖不存在、证据不足或不能安全关闭缺口时，不要猜测，返回 operations=[]。"
            "禁止输出解释、Markdown、答案、执行计划、confidence 或 Schema 外字段。\n"
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            f"BASE_IR_DIGEST={base_ir_digest}\n"
            f"ALLOWED_OPERATIONS={json.dumps(allowed, ensure_ascii=False)}\n"
            f"GAP_REQUESTS={json.dumps([item.model_dump(mode='json') for item in gaps], ensure_ascii=False)}\n"
            f"RELEVANT_CATALOG={json.dumps(catalog_payload, ensure_ascii=False)}\n"
            f"EXISTING_IR={json.dumps(existing_ir, ensure_ascii=False)}\n"
            f"QUERY={json.dumps(query, ensure_ascii=False)}"
        )

    @staticmethod
    def _atomic_prompt(query: str, catalog_payload: list[dict[str, Any]],
                       existing_ir: Mapping[str, Any], gap: GapRequest,
                       base_ir_digest: str) -> str:
        operation_type = atomic_operation_type(gap)
        operation_guidance = {
            "add_event": (
                "event.span 必须覆盖完整的字段条件到持续时长；condition.span 必须包含可证明字段别名的"
                "条件短语；duration.span 必须包含数值和时间单位。"
            ),
            "add_set_operation": "inputs 只能引用 AVAILABLE_OUTPUT_REFS 中两个不同输出。",
            "add_aggregate": "input 必须使用允许的 symbol 或 output_ref，scope 必须明确。",
            "add_calculation": "expression 只能使用允许的 symbol/output_ref 和安全算术。",
            "add_sequence": (
                "steps 必须按原文顺序给出至少两个条件；每个步骤只使用允许字段，"
                "partition_by/order_by 必须是允许的 Catalog 字段。"
            ),
            "add_turn_directive": "只表达当前原文明确的 inherit/replace/cancel，不执行副作用。",
        }.get(operation_type, "不得补充原文不存在的事实。")
        if gap.response_mode == "semantic_choice":
            work_order = {
                "gap_type": gap.gap_type,
                "required_slots": gap.required_slots,
                "target_summary": gap.existing_node_summary,
            }
            return (
                "你只完成一个已通过本地就绪检查的语义选择题。只输出当前 JSON Schema 要求的对象。"
                "所有 ref 字段只能逐字选择 ALLOWED_CHOICES 中的值，不得把自然语言描述写成 ref。"
                "evidence_text 必须从 QUERY_SLICE 逐字复制一个能够证明整个操作且只出现一次的连续片段。"
                "不要输出 ID、Anchor、Span 坐标、递归 AST、解释、Markdown、答案或工具调用；"
                "这些结构由本地编译器生成。\n"
                f"OPERATION={operation_type}\n"
                f"GUIDANCE={operation_guidance}\n"
                f"WORK_ORDER={json.dumps(work_order, ensure_ascii=False)}\n"
                f"ALLOWED_CHOICES={json.dumps(gap.choice_domain, ensure_ascii=False)}\n"
                f"QUERY_SLICE={json.dumps(gap.query_slice, ensure_ascii=False)}"
            )
        return (
            "你只修补一个原子语义缺口。输出对象必须符合当前唯一 JSON Schema。"
            "不要输出 schema_version、base_ir_digest、gap_id、operation_id、start 或 end；这些由本地注入。"
            "evidence_text 必须从 QUERY_SLICE 逐字复制且只出现一次；"
            "若没有更小且唯一的原文片段，可直接逐字复制 CANONICAL_EVIDENCE_TEXT 的完整值。"
            "anchor_ids 必须从 GAP.allowed_anchor_ids 中选择；只能使用 GAP 中允许的 Catalog symbol、Schema hypothesis、metric 和 output ref。"
            "证据不足时返回空对象，不得猜测，不得输出解释、Markdown、答案或工具调用。\n"
            f"OPERATION_GUIDANCE={operation_guidance}\n"
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            # Kept visible for migration fixtures; the atomic response schema cannot output it.
            f"BASE_IR_DIGEST={base_ir_digest}\n"
            f"ATOMIC_OPERATION={operation_type}\n"
            f"GAP_REQUESTS={json.dumps([gap.model_dump(mode='json')], ensure_ascii=False)}\n"
            f"RELEVANT_CATALOG={json.dumps(catalog_payload, ensure_ascii=False)}\n"
            f"EXISTING_IR={json.dumps(existing_ir, ensure_ascii=False)}\n"
            f"CANONICAL_EVIDENCE_TEXT={json.dumps(gap.query_slice, ensure_ascii=False)}\n"
            f"QUERY_SLICE={json.dumps(gap.query_slice, ensure_ascii=False)}\n"
            f"QUERY={json.dumps(query, ensure_ascii=False)}"
        )

    @staticmethod
    def _prompt(query: str, catalog_payload: list[dict[str, Any]],
                existing_ir: Mapping[str, Any]) -> str:
        span_schema = {
            "text": "逐字复制 QUERY 中能证明该节点的唯一连续原文；禁止输出 SPAN"
        }
        schema = {
            "goals": [{"type": "retrieve", "span": span_schema}],
            "sources": [{"source_id": "catalog source id", "span": span_schema}],
            "projections": [{"field_id": "catalog field id", "span": span_schema}],
            "hypothesis_references": [{"hypothesis_id": "EXISTING_IR hypothesis id", "span": span_schema}],
            "sampling": [{"interval_seconds": "positive integer", "span": span_schema}],
            "filters": [{
                "field_id": "catalog field id", "operator": "gt|gte|lt|lte|eq|ne|between|in|contains",
                "value": "number|string|array", "unit": "canonical unit or null", "span": span_schema,
            }],
            "events": [{
                "event_id": "short local id",
                "metric_id": "catalog derived metric id",
                "logic": "and|or",
                "conditions": [{
                    "field_id": "catalog field id", "operator": "gt|gte|lt|lte|eq|ne|between",
                    "value": "number or array", "unit": "canonical unit or null", "span": span_schema,
                }],
                "duration": {"operator": "gt|gte", "value": "number", "unit": "second|minute|hour|day", "span": span_schema},
                "group_by": ["local_date"], "span": span_schema,
            }],
            "set_operations": [{
                "operation": "intersection|union|difference",
                "inputs": ["event_id"], "granularity": "date", "span": span_schema,
            }],
            "calculations": [{
                "type": "mom|yoy|growth|difference|ratio|volatility",
                "expression": "safe declarative expression", "parameters": {}, "span": span_schema,
            }],
            "aggregates": [{
                "id": "local id", "function": "sum|avg|min|max|count|stddev",
                "input_id": "catalog field or hypothesis id",
                "scope": "filtered|global|relation", "scope_ref": "existing output id or null",
                "span": span_schema,
            }],
            "comparisons": [{
                "left_ref": "aggregate output", "operator": "gt|gte|lt|lte|eq|ne",
                "right": "typed expression object", "span": span_schema,
            }],
            "unresolved": [{"name": "source text", "code": "reason_code", "message": "reason", "span": span_schema}],
        }
        return (
            "你是只读请求语义解析器。请独立检查 QUERY 的完整语义；EXISTING_IR 只是另一组证据，"
            "不是必须服从的裁判。只返回一个 JSON 对象，"
            "禁止解释、Markdown、工具调用、执行计划和答案。只能引用 CATALOG 中存在的 source_id、"
            "field_id、metric_id 或 EXISTING_IR 已声明的 hypothesis_id；hypothesis 永远不是 Catalog 绑定。"
            "Catalog source/field 也只是候选，本地程序会用原文别名重新验证，不能自行授权。"
            "不能把‘累计、连续、总时长’伪造为字段。连续/累计语义必须放 events。"
            "每个候选节点必须提供 span.text，它必须是从 QUERY 逐字复制且只出现一次的连续片段；"
            "start/end 由本地程序计算，不要输出占位词 SPAN。"
            "派生指标选择铁律：‘连续、一直、持续、保持了N分钟’使用 consecutive_duration；"
            "只有‘累计、总时长、合计时长’才使用 cumulative_duration。"
            "当 unresolved 包含 event_structure_incomplete 时，events.conditions 必须复用 QUERY 中的"
            "数值条件；即使该条件已存在于 EXISTING_IR.filters，也必须在 event.conditions 中再次声明。"
            "连续事件必须同时提供 conditions、duration、metric_id 和覆盖完整事件短语的 span；"
            "不能仅重复 unresolved。"
            "日期范围属于 temporal，规则层已经解析，不得把 timestamp 日期范围重复放入 event.conditions。"
            "不确定内容放 unresolved，不要猜测。忽略 confidence，禁止输出 confidence。"
            "首先检查 EXISTING_IR.unresolved：如果原文证据和 CATALOG 足以补全，必须返回对应结构；"
            "如果仍不足，则保留 unresolved。type/operator 等枚举必须从 ALLOWED_ENUMS 选择一个值，"
            "绝不能复制包含 | 的类型说明。"
            "可以返回与规则证据重复的节点，本地 ClaimReconciler 会去重；不确定时放 unresolved，"
            "不得为了结构完整而猜测。没有可验证候选时返回 {}。\n"
            f"PROMPT_VERSION={PROMPT_VERSION}\n"
            "ALLOWED_ENUMS={\"goal\":[\"retrieve\",\"aggregate\",\"compare\",\"compute\",\"present\"],"
            "\"operator\":[\"eq\",\"ne\",\"gt\",\"gte\",\"lt\",\"lte\",\"between\",\"in\",\"contains\"],"
            "\"calculation\":[\"mom\",\"yoy\",\"growth\",\"difference\",\"ratio\",\"volatility\"]}\n"
            f"OUTPUT_SCHEMA={json.dumps(schema, ensure_ascii=False)}\n"
            f"CATALOG={json.dumps(catalog_payload, ensure_ascii=False)}\n"
            f"EXISTING_IR={json.dumps(existing_ir, ensure_ascii=False)}\n"
            f"QUERY={json.dumps(query, ensure_ascii=False)}"
        )

    @staticmethod
    def _repair_prompt(text: str,
                       response_schema: Mapping[str, Any] | None = None) -> str:
        return (
            "Repair only JSON structure to match the supplied schema without adding or changing facts. "
            "Return JSON only. If impossible return an object with the same base_ir_digest and an empty "
            "operations array.\nSCHEMA="
            + json.dumps(response_schema or {}, ensure_ascii=False)
            + "\nINVALID_OUTPUT=" + text[:8000]
        )


def _truncate_text(value: str, limit: int) -> str:
    """Keep a whole-prefix request budget without silently growing a prompt."""
    if len(value) <= limit:
        return value
    suffix = "…[truncated]"
    return value[:max(0, limit - len(suffix))] + suffix


def _audit_excerpt(value: str, limit: int = 4096) -> str:
    """Retain bounded response evidence for parser debugging, never full prompts."""
    return _truncate_text(_redact_audit_text(value), limit)


def _parsed_json_excerpt(value: str, limit: int = 4096) -> str:
    """Store the parsed object separately from the raw model response."""
    parsed = _parse_json(value)
    if parsed is None:
        return ""
    return _truncate_text(
        json.dumps(_redact_json(parsed), ensure_ascii=False, sort_keys=True), limit,
    )


def _redact_json(value: Any, key: str = "") -> Any:
    if re.search(r"token|secret|password|api[_-]?key|authorization", key, re.I):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact_json(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return _redact_audit_text(value)
    return value


def _redact_audit_text(value: str) -> str:
    result = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+\-/]+=*", r"\1[REDACTED]", value,
    )
    result = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|token|secret|password|authorization)[\"']?"
        r"\s*[=:]\s*[\"']?)"
        r"[^\"'\s,}]+", r"\1[REDACTED]", result,
    )
    result = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", result)
    return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[REDACTED_PHONE]", result)


def _compact_catalog_payload(payload: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return a JSON-valid prefix of the catalog, preserving source/field IDs.

    This is deliberately deterministic: a small local model sees the same
    candidate catalog for the same ordered snapshot, while a broad catalog can
    never consume the complete request budget.
    """
    result: list[dict[str, Any]] = []
    for source in payload:
        item = dict(source)
        fields = list(item.get("fields") or [])
        item["fields"] = []
        if _json_length(result + [item]) > limit:
            break
        result.append(item)
        for field in fields:
            candidate = dict(item)
            candidate["fields"] = [*item["fields"], field]
            if _json_length([*result[:-1], candidate]) > limit:
                return result
            item = candidate
            result[-1] = item
    return result


def _compact_mapping(payload: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """Bound the existing-IR payload while retaining valid JSON structure."""
    value = dict(payload)
    if _json_length(value) <= limit:
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        remaining = max(128, limit - _json_length(result) - len(str(key)) - 8)
        compact = _compact_json_value(item, remaining)
        candidate = {**result, key: compact}
        if _json_length(candidate) > limit:
            break
        result = candidate
    return result


def _compact_json_value(value: Any, limit: int) -> Any:
    if _json_length(value) <= limit:
        return value
    if isinstance(value, str):
        return _truncate_text(value, max(16, limit // 2))
    if isinstance(value, Mapping):
        return _compact_mapping(value, limit)
    if isinstance(value, list):
        result: list[Any] = []
        for item in value:
            candidate = [*result, _compact_json_value(item, max(64, limit // max(1, len(value))))]
            if _json_length(candidate) > limit:
                break
            result = candidate
        return result
    return None


def _json_length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _json_candidate_text(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    return value


def _has_valid_json_syntax(text: str) -> bool:
    try:
        json.loads(_json_candidate_text(text))
    except (TypeError, ValueError):
        return False
    return True


def _parse_json(text: str) -> dict[str, Any] | None:
    value = _json_candidate_text(text)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _default_ollama_url() -> str:
    """Use the Windows host gateway when this process runs under WSL NAT."""
    if not (os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")):
        return "http://127.0.0.1:11434"
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        parts = result.stdout.split()
        if "via" in parts:
            gateway = parts[parts.index("via") + 1]
            if gateway:
                return f"http://{gateway}:11434"
    except (OSError, subprocess.SubprocessError):
        pass
    return "http://127.0.0.1:11434"
