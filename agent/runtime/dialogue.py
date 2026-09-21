from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DialogueActKind(str, Enum):
    NEW_REQUEST = "new_request"
    ACCEPT = "accept"
    REJECT = "reject"
    CONTINUE_ACTION = "continue_action"
    REPHRASE = "rephrase"
    REQUEST_EXAMPLE = "request_example"
    ANSWER_CLARIFICATION = "answer_clarification"
    CANCEL = "cancel"


class ResponseStyle(str, Enum):
    DEFAULT = "default"
    PLAIN = "plain"
    EXAMPLE = "example"
    CONCISE = "concise"


@dataclass(frozen=True, slots=True)
class DialogueAct:
    kind: DialogueActKind
    target_action_id: str | None = None
    style: ResponseStyle = ResponseStyle.DEFAULT
    reason_code: str = ""
    schema_version: str = "dialogue-act-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["style"] = self.style.value
        return payload


@dataclass(slots=True)
class PendingInteraction:
    action_id: str
    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    payload_digest: str = ""
    source_turn_id: str = ""
    generation_snapshot_digest: str | None = None
    state: str = "pending"
    expires_at_epoch: float = 0.0
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.payload_digest:
            self.payload_digest = _digest(self.payload)

    def expired(self, now: float | None = None) -> bool:
        return bool(self.expires_at_epoch and self.expires_at_epoch <= (now or time.time()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingInteraction":
        return cls(**payload)


_ACCEPT = re.compile(r"^\s*(?:行|好|好的|可以|嗯|嗯嗯|ok|okay|继续|来吧|没问题)[。.!！]?\s*$", re.I)
_REJECT = re.compile(r"^\s*(?:不用|不要|算了|取消|不必|no|不用了)[。.!！]?\s*$", re.I)
_EXAMPLE = re.compile(r"举(?:个|一个)?例子|生活化|打个比方|用例子|example", re.I)
_REPHRASE = re.compile(r"简单|通俗|白话|容易懂|换(?:个|一种)说法|再解释|重新解释|没听懂|不太懂", re.I)
_FOLLOWUP_EXAMPLE = re.compile(
    r"(?:请|可以|能不能|能|再|帮我|给我|用)?\s*(?:举(?:个|一个)?例子|打个比方|用例子(?:解释)?|生活化(?:一点|一些)?|example)"
    r"(?:说明|解释|讲讲)?(?:一下|一点|一些)?(?:吗|吧)?[。.!！?？]*", re.I,
)
_FOLLOWUP_REPHRASE = re.compile(
    r"(?:请|可以|能不能|能|再|帮我)?\s*(?:简单|通俗|白话|容易懂)(?:点|一点|一些)?"
    r"(?:地)?(?:讲讲|解释|说说|说|介绍)?(?:一下|一点|一些)?(?:吗|吧)?(?:[，,]?我(?:不太懂|没听懂))?[。.!！?？]*|"
    r"(?:请|再)?(?:换(?:个|一种)说法|再解释|重新解释|没听懂|不太懂)(?:一下|一次)?[。.!！?？]*", re.I,
)
_OFFER = re.compile(
    r"(?:要不要|是否需要|需要我|我可以).{0,28}(?:举例|例子|生活化|通俗|再解释)", re.I | re.S
)


class DialogueActResolver:
    def resolve(
        self,
        message: str,
        *,
        pending: PendingInteraction | None,
        has_prior_substantive_turn: bool,
    ) -> DialogueAct:
        value = str(message or "").strip()
        active = pending if pending and pending.state == "pending" and not pending.expired() else None
        if _ACCEPT.fullmatch(value):
            if active:
                return DialogueAct(
                    DialogueActKind.ACCEPT, active.action_id,
                    ResponseStyle.EXAMPLE if active.action_type == "example" else ResponseStyle.DEFAULT,
                    "accepted_pending_action",
                )
            return DialogueAct(DialogueActKind.ACCEPT, reason_code="confirmation_without_pending")
        if _REJECT.fullmatch(value):
            return DialogueAct(
                DialogueActKind.REJECT,
                active.action_id if active else None,
                reason_code="rejected_pending_action" if active else "rejection_without_pending",
            )
        if _FOLLOWUP_EXAMPLE.fullmatch(value) and has_prior_substantive_turn:
            return DialogueAct(DialogueActKind.REQUEST_EXAMPLE, style=ResponseStyle.EXAMPLE,
                               reason_code="explicit_example_followup")
        if _FOLLOWUP_REPHRASE.fullmatch(value) and has_prior_substantive_turn:
            return DialogueAct(DialogueActKind.REPHRASE, style=ResponseStyle.PLAIN,
                               reason_code="explicit_rephrase_followup")
        return DialogueAct(DialogueActKind.NEW_REQUEST, reason_code="substantive_or_unbound_turn")


def suggested_interaction_from_reply(
    answer: str,
    *,
    source_turn_id: str,
    generation_snapshot_digest: str | None,
    ttl_seconds: int = 900,
) -> PendingInteraction | None:
    """Compatibility bridge: persist only an action the answer explicitly offered."""
    if not _OFFER.search(str(answer or "")):
        return None
    payload = {"style": ResponseStyle.EXAMPLE.value, "source": "explicit_answer_offer"}
    action_id = hashlib.sha256(
        f"{source_turn_id}:example:{_digest(payload)}".encode("utf-8")
    ).hexdigest()[:24]
    return PendingInteraction(
        action_id=action_id,
        action_type="example",
        payload=payload,
        source_turn_id=source_turn_id,
        generation_snapshot_digest=generation_snapshot_digest or None,
        expires_at_epoch=time.time() + max(1, ttl_seconds),
    )


def render_followup(previous_answer: str, *, style: ResponseStyle) -> str:
    previous = str(previous_answer or "").strip()
    if not previous:
        return "我没有找到可复述的上一条实质回答，请把要解释的内容再发一次。"
    if style == ResponseStyle.EXAMPLE:
        return "可以。把它先当成一个生活化的直觉来理解：\n\n" + previous
    if style == ResponseStyle.PLAIN:
        return "换成更直白的说法：\n\n" + previous
    return previous


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()
