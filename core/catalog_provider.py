from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any
from pathlib import Path


def catalog_builder_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CatalogFieldContract:
    canonical_id: str
    backend_field: str
    data_type: str
    aliases: tuple[str, ...]
    allowed_operators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogSourceContract:
    source_id: str
    aliases: tuple[str, ...]
    capabilities: tuple[str, ...]
    fields: tuple[CatalogFieldContract, ...]
    version: str


@dataclass(frozen=True, slots=True)
class CatalogContractSnapshot:
    version: str
    sources: tuple[CatalogSourceContract, ...] = field(default_factory=tuple)
    schema_version: str = "catalog-contract-v2"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [asdict(source) for source in self.sources],
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.identity_payload(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_identity_payload(cls, value: dict, *, version: str):
        def text_value(item):
            return isinstance(item, str) and bool(item.strip())

        def text_array(item):
            return (isinstance(item, (list, tuple)) and all(text_value(part) for part in item)
                    and len(set(item)) == len(item))

        if not text_value(version):
            raise ValueError("invalid catalog version")
        if not isinstance(value, dict) or set(value) != {"schema_version", "sources"} or value["schema_version"] != "catalog-contract-v2":
            raise ValueError("invalid catalog schema")
        if not isinstance(value["sources"], (list, tuple)):
            raise ValueError("invalid catalog sources")
        sources = []
        source_ids = set()
        for source in value["sources"]:
            if not isinstance(source, dict) or set(source) != {"source_id", "aliases", "capabilities", "fields", "version"}:
                raise ValueError("invalid catalog source")
            if (not all(text_value(source[key]) for key in ("source_id", "version"))
                    or not all(text_array(source[key]) for key in ("aliases", "capabilities"))
                    or not isinstance(source["fields"], (list, tuple))
                    or source["source_id"] in source_ids):
                raise ValueError("invalid catalog source values")
            source_ids.add(source["source_id"])
            fields = []
            field_ids, backend_fields = set(), set()
            for item in source["fields"]:
                if not isinstance(item, dict) or set(item) != {"canonical_id", "backend_field", "data_type", "aliases", "allowed_operators"}:
                    raise ValueError("invalid catalog field")
                if (not all(text_value(item[key]) for key in ("canonical_id", "backend_field", "data_type"))
                        or not all(text_array(item[key]) for key in ("aliases", "allowed_operators"))
                        or item["canonical_id"] in field_ids or item["backend_field"] in backend_fields):
                    raise ValueError("invalid catalog field values")
                field_ids.add(item["canonical_id"])
                backend_fields.add(item["backend_field"])
                fields.append(CatalogFieldContract(**{
                    **item, "aliases": tuple(item["aliases"]), "allowed_operators": tuple(item["allowed_operators"]),
                }))
            sources.append(CatalogSourceContract(**{
                **source, "aliases": tuple(source["aliases"]), "capabilities": tuple(source["capabilities"]),
                "fields": tuple(fields),
            }))
        return cls(version=version, sources=tuple(sources), schema_version=value["schema_version"])


class MainProjectCatalogReader:
    """Read the official Elasticsearch client mapping without exposing query APIs."""

    _ALIASES = {
        "filename": ("文件名", "文档名", "论文名"),
        "content": ("内容", "正文"),
        "content_length": ("内容长度",),
        "page_count": ("页数", "总页数"),
        # This is ingestion time, never publication time.
        "created_date": ("建库时间", "收录时间", "索引时间"),
        "language": ("语言",),
        "document_type": ("文档类型", "论文类型"),
    }

    def __init__(self, es_client: Any, index_name: str) -> None:
        self.es = es_client
        self.index_name = index_name

    def snapshot(self) -> CatalogContractSnapshot:
        mapping = self.es.indices.get_mapping(index=self.index_name)
        index_mapping = mapping.get(self.index_name, {})
        mappings = index_mapping.get("mappings", {})
        properties = mappings.get("properties", {})
        fields = tuple(self._flatten_fields(properties))
        source_version = str(mappings.get("_meta", {}).get("version", "1"))
        source = CatalogSourceContract(
            source_id=self.index_name,
            aliases=(
                self.index_name, "私人知识库", "文档库", "PDF库",
                "论文库", "论文", "文献库", "文献", "库里", "知识库",
            ),
            # `filter` is the canonical NLU planner capability; metadata_filter
            # describes the backend implementation.  Publish both so a temporal
            # predicate can be bound without weakening planner validation.
            capabilities=(
                "retrieve", "text_search", "filter", "metadata_filter", "sort",
                "aggregate", "inventory",
            ),
            fields=fields,
            version=source_version,
        )
        provisional = CatalogContractSnapshot(version="", sources=(source,))
        version = f"{source_version}:{provisional.digest()[:16]}"
        return CatalogContractSnapshot(version=version, sources=(source,))

    def _flatten_fields(self, properties: dict[str, Any], prefix: str = ""):
        for name, spec in properties.items():
            backend_field = f"{prefix}.{name}" if prefix else name
            nested = spec.get("properties") if isinstance(spec, dict) else None
            if isinstance(nested, dict):
                yield from self._flatten_fields(nested, backend_field)
                continue
            data_type = str(spec.get("type", "object")) if isinstance(spec, dict) else "object"
            operators = (
                ("eq", "ne", "gt", "gte", "lt", "lte", "between", "in")
                if data_type in {"integer", "long", "float", "double", "date"}
                else ("eq", "ne", "in", "contains", "exists")
            )
            aliases = (name, *self._ALIASES.get(backend_field, ()))
            yield CatalogFieldContract(
                canonical_id=f"{self.index_name}.{backend_field}",
                backend_field=backend_field,
                data_type=data_type,
                aliases=tuple(dict.fromkeys(aliases)),
                allowed_operators=operators,
            )
