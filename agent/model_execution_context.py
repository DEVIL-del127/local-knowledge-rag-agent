"""Optional model identity context; imports no runtime or storage implementation."""
from __future__ import annotations

import contextvars
import uuid

operation_context = contextvars.ContextVar("model_operation_context", default=None)


def current_operation():
    return operation_context.get()


def model_request_id(purpose: str) -> str:
    execution = current_operation()
    return f"{execution[1].operation_id}:{purpose}:output" if execution else uuid.uuid4().hex
