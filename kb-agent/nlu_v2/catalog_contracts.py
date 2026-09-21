"""Deterministic readiness preflight for frozen or imported source contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CatalogContractReadiness:
    contract_id: str
    readiness_level: str
    blockers: tuple[str, ...]


class CatalogContractPreflight:
    """Evaluate capabilities and policies without domain-field inference."""

    def evaluate(self, contract: Mapping[str, Any]) -> CatalogContractReadiness:
        capabilities = set(contract.get("capabilities", []))
        policies = dict(contract.get("policies", {}))
        sampling = dict(contract.get("sampling", {}))
        group = str(contract.get("matched_group", ""))
        blockers = []
        level = "logical_plan_ready"
        if not contract.get("read_only", False):
            return CatalogContractReadiness(str(contract.get("contract_id", "")),
                                            "source_authorization_pending", ("source_not_read_only",))
        if group == "device":
            if "event_segmentation" not in capabilities:
                level, blockers = "event_capability_pending", ["event_segmentation"]
            elif not policies.get("max_gap"):
                level, blockers = "event_policy_pending", ["max_gap"]
        elif group == "manufacturing":
            if "duration_accumulation" not in capabilities:
                level, blockers = "duration_capability_pending", ["duration_accumulation"]
            elif not policies.get("integration_method"):
                level, blockers = "duration_policy_pending", ["integration_method"]
        elif group == "market":
            if "market_calendar" not in capabilities or not sampling.get("calendar"):
                level, blockers = "window_capability_pending", ["market_calendar"]
            elif policies.get("window_semantics") != "trading_day":
                level, blockers = "window_policy_pending", ["window_semantics"]
        else:
            level, blockers = "source_contract_unsupported", ["matched_group"]
        return CatalogContractReadiness(str(contract.get("contract_id", "")), level, tuple(blockers))
