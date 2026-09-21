from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class NluExecutionProfile:
    """Single immutable switchboard for the authoritative NLU v2 engine."""

    profile_id: str = "latest-enforce-v1"
    enable_vector: bool = True
    enable_llm: bool = True
    llm_on_complex: bool = False
    enable_patch_v1: bool = True
    enable_requirement_ir_v2: bool = True
    enable_operator_registry: bool = True
    enable_atomic_patch_v2: bool = True
    enable_semantic_repair: bool = True
    enable_requirement_normalizer: bool = True
    enable_semantic_linker: bool = True
    enable_event_closure_contract: bool = True
    enable_semantic_target_gate: bool = True
    enable_candidate_choice_v3: bool = True
    enable_m15_requirement_graph_shadow: bool = True
    enable_m15_field_binding_shadow: bool = True
    enable_m15_quantity_contract: bool = True
    enable_m15_temporal_contracts: bool = True
    enable_m15_typed_lineage: bool = True
    enable_m15_logical_plan: bool = True

    def engine_kwargs(self) -> dict[str, bool]:
        values = asdict(self)
        values.pop("profile_id")
        return values

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    @classmethod
    def latest(cls, *, enable_llm: bool = True) -> "NluExecutionProfile":
        return cls(enable_llm=bool(enable_llm))
