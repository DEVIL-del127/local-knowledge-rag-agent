from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_agent(*, mode: str, legacy_factory: Callable[[], Any], runtime_factory: Callable[[Any, str], Any] | None = None) -> Any:
    """Composition-root hard bypass.

    The off branch intentionally does not import any runtime graph/checkpoint module.
    """
    normalized = str(mode or "off").strip().lower()
    if normalized not in {"shadow", "enforce"}:
        return legacy_factory()
    legacy = legacy_factory()
    if runtime_factory is not None:
        return runtime_factory(legacy, normalized)
    from agent.runtime.facade import RuntimeAgentFacade

    return RuntimeAgentFacade(legacy=legacy, mode=normalized)
