"""Versioned catalogs for static schemas, Elasticsearch mappings, skills and MCP."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from .models import QuerySchemaSnapshot, SourceSpan


@dataclass(slots=True)
class FieldSpec:
    field_id: str
    aliases: list[str]
    data_type: str
    unit: str | None = None
    allowed_operators: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DerivedMetricSpec:
    metric_id: str
    aliases: list[str]
    result_type: str
    unit: str | None = None
    parameters: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    missing_data_policy: str = ""


@dataclass(slots=True)
class DataSourceSpec:
    source_id: str
    aliases: list[str]
    kind: str
    capabilities: list[str]
    fields: list[FieldSpec]
    read_only: bool = True
    version: str = "1"
    join_keys: list[str] = field(default_factory=list)
    derived_metrics: list[DerivedMetricSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FieldMatch:
    alias: str
    start: int
    end: int
    field_ids: list[str]


@dataclass(slots=True)
class CatalogSnapshot:
    version: str
    sources: list[DataSourceSpec]

    def source(self, source_id: str) -> DataSourceSpec | None:
        return next((item for item in self.sources if item.source_id == source_id), None)

    def field(self, field_id: str) -> FieldSpec | None:
        for source in self.sources:
            for item in source.fields:
                if item.field_id == field_id:
                    return item
        return None

    def source_for_field(self, field_id: str) -> str | None:
        for source in self.sources:
            if any(item.field_id == field_id for item in source.fields):
                return source.source_id
        return None

    def derived_metric(self, metric_id: str) -> DerivedMetricSpec | None:
        for source in self.sources:
            for item in source.derived_metrics:
                if item.metric_id == metric_id:
                    return item
        return None

    def source_for_derived_metric(self, metric_id: str) -> str | None:
        for source in self.sources:
            if any(item.metric_id == metric_id for item in source.derived_metrics):
                return source.source_id
        return None

    def resolve_field(self, raw_name: str) -> list[str]:
        needle = raw_name.strip().lower()
        matches = []
        for source in self.sources:
            for item in source.fields:
                names = [item.field_id.rsplit(".", 1)[-1], *item.aliases]
                if any(needle == name.strip().lower() for name in names):
                    matches.append(item.field_id)
        return list(dict.fromkeys(matches))

    def find_field_mentions(self, query: str) -> list[FieldMatch]:
        candidates = []
        lowered = query.lower()
        for source in self.sources:
            for item in source.fields:
                for alias in sorted(set(item.aliases), key=len, reverse=True):
                    if not alias:
                        continue
                    start = lowered.find(alias.lower())
                    while start >= 0:
                        candidates.append((start, start + len(alias), alias, item.field_id))
                        start = lowered.find(alias.lower(), start + len(alias))
        candidates.sort(key=lambda row: (row[0], -(row[1] - row[0])))
        merged: list[FieldMatch] = []
        for start, end, alias, field_id in candidates:
            existing = next((item for item in merged if item.start == start and item.end == end), None)
            if existing:
                if field_id not in existing.field_ids:
                    existing.field_ids.append(field_id)
                continue
            if any(start < item.end and end > item.start for item in merged):
                continue
            merged.append(FieldMatch(alias, start, end, [field_id]))
        return sorted(merged, key=lambda item: item.start)

    def source_alias_matches(self, query: str) -> list[str]:
        matches = []
        lowered = query.lower()
        for source in self.sources:
            if any(alias.lower() in lowered for alias in source.aliases):
                matches.append(source.source_id)
        return list(dict.fromkeys(matches))

    def prove_source_alias(self, source_id: str, evidence: str) -> bool:
        """A model-proposed source is binding only when its evidence names it uniquely."""
        source = self.source(source_id)
        if not source:
            return False
        needle = evidence.strip().lower()
        matched = [
            item.source_id for item in self.sources
            if any(name and name.lower() in needle for name in [item.source_id, *item.aliases])
        ]
        return matched == [source_id]

    def prove_field_alias(self, field_id: str, evidence: str) -> bool:
        """Prove a field against the already-authorized snapshot, never query hypotheses."""
        field = self.field(field_id)
        if not field:
            return False
        needle = evidence.strip().lower()
        matched = []
        for source in self.sources:
            for item in source.fields:
                names = [item.field_id, item.field_id.rsplit(".", 1)[-1], *item.aliases]
                if any(name and name.lower() in needle for name in names):
                    matched.append(item.field_id)
        return list(dict.fromkeys(matched)) == [field_id]

    def query_schema(self, query: str, *, hypotheses: Iterable[str] = ()) -> QuerySchemaSnapshot:
        """Build a query-local schema without turning candidates into bindings.

        The ordering is fixed: explicit field IDs in the query, fields of an
        explicitly named source, then local candidate matches.  A generic
        alias such as ``温度`` remains a candidate when the query does not name
        a source, which prevents accidental cross-domain binding.
        """
        evidence: list[SourceSpan] = []
        field_ids: list[str] = []
        for match in re.finditer(r"`(?P<field>[A-Za-z_][A-Za-z0-9_.]*)`", query):
            field_id = match.group("field")
            if self.field(field_id):
                field_ids.append(field_id)
                evidence.append(SourceSpan(match.start(), match.end(), match.group(0)))
        source_ids = self.source_alias_matches(query)
        for source_id in source_ids:
            source = self.source(source_id)
            if not source:
                continue
            alias = next((name for name in [source_id, *source.aliases]
                          if name.lower() in query.lower()), source_id)
            start = query.lower().find(alias.lower())
            if start >= 0:
                evidence.append(SourceSpan(start, start + len(alias), query[start:start + len(alias)]))
        candidate_field_ids = [field_id for match in self.find_field_mentions(query)
                               for field_id in match.field_ids]
        if not source_ids:
            owner_sets = [
                {self.source_for_field(field_id) for field_id in match.field_ids
                 if self.source_for_field(field_id)}
                for match in self.find_field_mentions(query)
            ]
            owner_sets = [owners for owners in owner_sets if owners]
            consensus = set.intersection(*owner_sets) if len(owner_sets) >= 2 else set()
            if len(consensus) == 1:
                # Multiple independent field mentions can prove one source
                # without guessing any individual ambiguous alias.
                source_ids = [next(iter(consensus))]
        if source_ids:
            candidate_field_ids = [field_id for field_id in candidate_field_ids
                                   if self.source_for_field(field_id) in source_ids]
        # Field IDs are executable only when directly named; a source mention
        # establishes local Catalog scope but not an unmentioned field binding.
        basis = (
            "explicit_query" if field_ids else
            "source_catalog" if self.source_alias_matches(query) else
            "field_consensus" if source_ids else "candidate_only"
        )
        return QuerySchemaSnapshot(
            source_ids=list(dict.fromkeys(source_ids)),
            field_ids=list(dict.fromkeys(field_ids)),
            candidate_field_ids=list(dict.fromkeys(candidate_field_ids)),
            hypotheses=list(hypotheses), evidence=evidence, binding_basis=basis,
        )


class CatalogProvider(Protocol):
    def snapshot(self) -> CatalogSnapshot:
        ...


def _version_for(sources: Iterable[DataSourceSpec]) -> str:
    payload = json.dumps([asdict(item) for item in sources], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class StaticCatalogProvider:
    def __init__(self, sources: list[DataSourceSpec] | None = None):
        self.sources = sources or default_sources()

    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot(_version_for(self.sources), list(self.sources))


class CompositeCatalogProvider:
    def __init__(self, providers: Iterable[CatalogProvider]):
        self.providers = list(providers)

    def snapshot(self) -> CatalogSnapshot:
        merged: dict[str, DataSourceSpec] = {}
        for provider in self.providers:
            for source in provider.snapshot().sources:
                merged[source.source_id] = source
        sources = list(merged.values())
        return CatalogSnapshot(_version_for(sources), sources)


class ESCatalogProvider:
    """Read-only adapter around the existing REST ESClient; no request occurs at import."""

    def __init__(self, es_client: Any, indexes: Iterable[str]):
        self.es = es_client
        self.indexes = list(indexes)

    def snapshot(self) -> CatalogSnapshot:
        sources = []
        for index in self.indexes:
            mapping = self.es._req("GET", f"/{index}/_mapping")
            properties = mapping.get(index, {}).get("mappings", {}).get("properties", {})
            fields = []
            for name, spec in properties.items():
                data_type = str(spec.get("type", "object"))
                fields.append(FieldSpec(
                    field_id=f"{index}.{name}", aliases=[name], data_type=data_type,
                    allowed_operators=_default_operators(data_type),
                    aggregations=_default_aggregations(data_type),
                ))
            sources.append(DataSourceSpec(
                source_id=index, aliases=[index], kind="elasticsearch",
                capabilities=["retrieve", "filter", "aggregate", "sort"], fields=fields,
                version=str(mapping.get(index, {}).get("mappings", {}).get("_meta", {}).get("version", "1")),
            ))
        return CatalogSnapshot(_version_for(sources), sources)


class SkillCatalogProvider:
    """Convert trusted Skill/MCP input schemas into catalog capabilities."""

    def __init__(self, skill_specs: Iterable[Mapping[str, Any]]):
        self.skill_specs = list(skill_specs)

    def snapshot(self) -> CatalogSnapshot:
        sources = []
        for spec in self.skill_specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            schema = spec.get("input_schema") or spec.get("parameters") or {}
            properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
            fields = [
                FieldSpec(
                    field_id=f"{name}.{field_name}", aliases=[field_name],
                    data_type=str(field_schema.get("type", "string")),
                    allowed_operators=_default_operators(str(field_schema.get("type", "string"))),
                )
                for field_name, field_schema in properties.items()
                if isinstance(field_schema, Mapping)
            ]
            sources.append(DataSourceSpec(
                source_id=name, aliases=[name, *list(spec.get("aliases", []))],
                kind=str(spec.get("kind", "tool")), capabilities=list(spec.get("capabilities", ["retrieve"])),
                fields=fields, read_only=bool(spec.get("read_only", True)),
                version=str(spec.get("version", "1")),
            ))
        return CatalogSnapshot(_version_for(sources), sources)


def _default_operators(data_type: str) -> list[str]:
    if data_type in {"integer", "long", "float", "double", "number", "date"}:
        return ["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in"]
    return ["eq", "ne", "in", "not_in", "contains", "exists"]


def _default_aggregations(data_type: str) -> list[str]:
    if data_type in {"integer", "long", "float", "double", "number"}:
        return ["sum", "avg", "min", "max", "count"]
    return ["count"]


def _field(source: str, name: str, aliases: list[str], data_type: str,
           unit: str | None = None) -> FieldSpec:
    return FieldSpec(
        field_id=f"{source}.{name}", aliases=aliases, data_type=data_type, unit=unit,
        allowed_operators=_default_operators(data_type), aggregations=_default_aggregations(data_type),
    )


def _derived(source: str, name: str, aliases: list[str], result_type: str,
             unit: str | None, parameters: list[str], capabilities: list[str],
             **defaults: Any) -> DerivedMetricSpec:
    return DerivedMetricSpec(
        metric_id=f"{source}.{name}", aliases=aliases, result_type=result_type,
        unit=unit, parameters=parameters, required_capabilities=capabilities,
        defaults=defaults,
        missing_data_policy=str(defaults.get("missing_data_policy", "")),
    )


def default_sources() -> list[DataSourceSpec]:
    return [
        DataSourceSpec(
            "device_telemetry",
            ["监测设备", "设备传感器", "温度传感器", "传感器", "设备遥测", "设备数据"],
            "time_series",
            [
                "retrieve", "filter", "aggregate", "sort", "time_series_scan",
                "ordered_partition", "event_segmentation", "duration_aggregation",
                "window_aggregation", "set_operation", "project_local_date",
            ],
            [
                _field("device_telemetry", "device_id", ["设备ID", "设备编号", "设备"], "string"),
                _field("device_telemetry", "timestamp", ["时间戳", "采样时间", "记录时间", "日期"], "date"),
                _field("device_telemetry", "temperature", ["温度", "温度值"], "number", "celsius"),
                _field("device_telemetry", "voltage", ["电压", "电压值"], "number", "volt"),
            ],
            derived_metrics=[
                _derived(
                    "device_telemetry", "consecutive_duration",
                    ["连续时长", "持续时长", "连续超过"], "duration", "second",
                    ["condition", "partition_by", "order_by", "max_gap_seconds"],
                    ["ordered_partition", "event_segmentation"],
                    missing_data_policy="break_segment", boundary_policy="clip_to_window",
                    max_gap_multiplier=2,
                ),
                _derived(
                    "device_telemetry", "cumulative_duration",
                    ["累计时长", "总时长", "累计超过"], "duration", "second",
                    ["condition", "window", "partition_by", "order_by"],
                    ["duration_aggregation"],
                    missing_data_policy="exclude_unknown", integration_method="step",
                    boundary_policy="clip_to_calendar_day",
                ),
                _derived(
                    "device_telemetry", "voltage_volatility",
                    ["电压波动率", "波动率"], "number", None,
                    ["input_field", "window", "group_by", "method"],
                    ["window_aggregation"], method="stddev",
                ),
                _derived(
                    "device_telemetry", "local_date",
                    ["本地日期", "日期"], "date", None,
                    ["timestamp", "timezone"], ["project_local_date"],
                    timezone="Asia/Shanghai",
                ),
            ],
            metadata={
                "requires_source_context": True,
                "timestamp_field": "device_telemetry.timestamp",
                "partition_fields": ["device_telemetry.device_id"],
                "expected_sampling_interval_seconds": 60,
                "default_timezone": "Asia/Shanghai",
                "max_scan_points": 2000000,
                "scan_chunk_size": 100000,
                "max_partitions": 10000,
            },
        ),
        DataSourceSpec(
            "private_kb", ["知识库", "文献库", "论文库", "私人资料库", "文献", "论文", "资料"], "elasticsearch",
            ["retrieve", "filter", "aggregate", "sort"], [
                _field("private_kb", "title", ["标题", "题名"], "string"),
                _field("private_kb", "author", ["作者"], "string"),
                _field("private_kb", "year", ["年份", "发表年份"], "integer"),
                _field("private_kb", "venue", ["期刊", "会议"], "string"),
                _field("private_kb", "language", ["语言"], "string"),
                _field("private_kb", "doc_type", ["文档类型", "论文类型"], "string"),
                _field("private_kb", "content", ["内容", "主题", "方法"], "text"),
            ],
        ),
        DataSourceSpec(
            "weather_observations", ["天气数据", "气象数据", "天气观测"], "external",
            ["retrieve", "filter", "aggregate", "sort"], [
                _field("weather_observations", "temperature", ["温度", "气温"], "number", "celsius"),
                _field("weather_observations", "wind_speed", ["风速"], "number", "m/s"),
                _field("weather_observations", "humidity", ["湿度", "相对湿度"], "number", "percent"),
                _field("weather_observations", "precipitation", ["降水量", "降雨量"], "number", "mm"),
                _field("weather_observations", "city", ["城市", "地区"], "string"),
                _field("weather_observations", "observed_at", ["观测时间", "日期"], "date"),
            ],
        ),
        DataSourceSpec(
            "sales_data", ["销售数据", "零售数据", "SKU数据"], "external",
            ["retrieve", "filter", "aggregate", "sort"], [
                _field("sales_data", "sku", ["SKU", "库存单位"], "string"),
                _field("sales_data", "sales_total", ["销售总额", "销售额", "营业额"], "number", "currency"),
                _field("sales_data", "quantity", ["销量", "销售数量"], "number"),
                _field("sales_data", "store", ["门店", "零售店"], "string"),
                _field("sales_data", "sold_at", ["销售日期", "日期"], "date"),
            ],
        ),
        DataSourceSpec(
            "market_data", ["股票数据", "行情数据", "市场数据"], "external",
            ["retrieve", "filter", "aggregate", "sort", "market_calendar"], [
                _field("market_data", "ticker", ["股票代码", "证券代码"], "string"),
                _field("market_data", "price", ["股价", "价格"], "number", "currency"),
                _field("market_data", "high_price", ["最高价"], "number", "currency"),
                _field("market_data", "low_price", ["最低价"], "number", "currency"),
                _field("market_data", "trading_date", ["交易日期", "交易日"], "date"),
            ],
        ),
    ]
