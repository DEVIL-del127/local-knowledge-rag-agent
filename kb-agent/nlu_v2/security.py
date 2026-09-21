"""Authorization boundary for catalog data exposed to the NLU pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .catalog import CatalogSnapshot, DataSourceSpec, FieldSpec, _version_for


@dataclass(frozen=True, slots=True)
class SecurityContext:
    """Trusted caller context. It must never be populated from user text or an LLM."""

    tenant_id: str
    principal_id: str
    roles: tuple[str, ...] = ()
    allowed_sources: tuple[str, ...] = ("*",)
    allowed_fields: tuple[str, ...] = ("*",)
    policy_version: str = "local-v1"
    data_classification: str = "private"

    @classmethod
    def local_development(cls) -> "SecurityContext":
        return cls(
            tenant_id="local",
            principal_id="local-cli",
            roles=("owner",),
        )

    def digest(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "roles": sorted(self.roles),
            "allowed_sources": sorted(self.allowed_sources),
            "allowed_fields": sorted(self.allowed_fields),
            "policy_version": self.policy_version,
            "data_classification": self.data_classification,
        }
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class CatalogAuthorizer:
    """Creates the only catalog snapshot that rules, vectors, and models may see."""

    def authorize(self, catalog: CatalogSnapshot,
                  context: SecurityContext) -> CatalogSnapshot:
        sources = []
        for source in catalog.sources:
            if not self._allowed(source.source_id, context.allowed_sources):
                continue
            fields = [
                field for field in source.fields
                if self._allowed(field.field_id, context.allowed_fields)
            ]
            sources.append(self._copy_source(
                source, fields, allow_all_fields="*" in context.allowed_fields
            ))
        # Include the upstream version so a policy change cannot alias an old snapshot.
        version_seed = [
            DataSourceSpec(
                source_id=f"authorized:{catalog.version}:{context.policy_version}",
                aliases=[], kind="scope", capabilities=[], fields=[],
            ),
            *sources,
        ]
        return CatalogSnapshot(_version_for(version_seed), sources)

    @staticmethod
    def _allowed(identifier: str, allowed: tuple[str, ...]) -> bool:
        return "*" in allowed or identifier in allowed

    @staticmethod
    def _copy_source(source: DataSourceSpec,
                     fields: list[FieldSpec], *,
                     allow_all_fields: bool) -> DataSourceSpec:
        allowed_ids = {item.field_id for item in fields}
        all_source_fields_allowed = all(
            item.field_id in allowed_ids for item in source.fields
        )
        return DataSourceSpec(
            source_id=source.source_id,
            aliases=list(source.aliases),
            kind=source.kind,
            capabilities=list(source.capabilities),
            fields=list(fields),
            read_only=source.read_only,
            version=source.version,
            join_keys=[item for item in source.join_keys
                       if any(field.field_id == item for field in fields)],
            derived_metrics=(
                list(source.derived_metrics)
                if fields and (allow_all_fields or all_source_fields_allowed) else []
            ),
            metadata=CatalogAuthorizer._sanitize_metadata(
                source.metadata, allowed_ids
            ),
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict, allowed_ids: set[str]) -> dict:
        result = dict(metadata)
        timestamp = result.get("timestamp_field")
        if isinstance(timestamp, str) and timestamp not in allowed_ids:
            result.pop("timestamp_field", None)
        partitions = result.get("partition_fields")
        if isinstance(partitions, list):
            result["partition_fields"] = [
                item for item in partitions if item in allowed_ids
            ]
        return result
