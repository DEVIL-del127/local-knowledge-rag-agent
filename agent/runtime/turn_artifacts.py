from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentResult:
    document_id: str
    filename: str
    title: str
    authors: tuple[str, ...]
    publication_year: int | None
    rank: int


@dataclass(frozen=True, slots=True)
class TurnResultArtifact:
    turn_id: str
    task: str
    generation_id: str
    snapshot_digest: str
    documents: tuple[DocumentResult, ...]
    topic: str | None
    filters: dict[str, Any]
    artifact_digest: str = ""
    schema_version: str = "turn-result-artifact-v1"

    def __post_init__(self) -> None:
        if not self.artifact_digest:
            payload = asdict(self)
            payload["artifact_digest"] = ""
            object.__setattr__(self, "artifact_digest", hashlib.sha256(json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def restore(cls, payload: dict[str, Any]) -> "TurnResultArtifact":
        documents = tuple(DocumentResult(**item) for item in payload.get("documents", []))
        restored = cls(**{**payload, "documents": documents})
        expected = cls(**{**payload, "documents": documents, "artifact_digest": ""}).artifact_digest
        if restored.artifact_digest != expected:
            raise ValueError("TurnResultArtifact digest mismatch")
        return restored
