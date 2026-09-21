"""Quantity semantics and safe conversion contracts for M1.5."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantitySpec:
    unit: str
    dimension: str
    quantity_kind: str
    transform_kind: str
    canonical_unit: str
    scale: float = 1.0
    offset: float = 0.0
    allowed_operations: tuple[str, ...] = ("compare", "add", "subtract")
    converter_id: str = ""


@dataclass(frozen=True, slots=True)
class QuantityConversion:
    source: QuantitySpec
    target: QuantitySpec
    allowed: bool
    reason: str = ""

    def convert(self, value: float) -> float:
        if not self.allowed:
            raise ValueError(self.reason or "quantity conversion is not allowed")
        canonical = value * self.source.scale + self.source.offset
        return (canonical - self.target.offset) / self.target.scale


class QuantityRegistry:
    def __init__(self):
        self._specs = _default_specs()

    def get(self, unit: str, quantity_kind: str = "") -> QuantitySpec | None:
        normalized = _ALIASES.get(unit.strip(), unit.strip())
        candidates = [item for item in self._specs if item.unit == normalized]
        if quantity_kind:
            candidates = [item for item in candidates if item.quantity_kind == quantity_kind]
        return candidates[0] if len(candidates) == 1 else None

    def conversion(self, source_unit: str, target_unit: str, *, quantity_kind: str = "") -> QuantityConversion:
        source = self.get(source_unit, quantity_kind)
        target = self.get(target_unit, quantity_kind)
        if not source or not target:
            placeholder = source or target or QuantitySpec("unknown", "unknown", "unknown", "contextual", "")
            return QuantityConversion(source or placeholder, target or placeholder, False, "unit_binding_pending")
        if (source.dimension, source.quantity_kind) != (target.dimension, target.quantity_kind):
            return QuantityConversion(source, target, False, "incompatible_quantity_kind")
        if "logarithmic" in {source.transform_kind, target.transform_kind}:
            allowed = source.unit == target.unit and source.converter_id == target.converter_id
            return QuantityConversion(source, target, allowed,
                                      "logarithmic_reference_required" if not allowed else "")
        if "contextual" in {source.transform_kind, target.transform_kind}:
            return QuantityConversion(source, target, False, "contextual_conversion_source_required")
        return QuantityConversion(source, target, True)


_ALIASES = {"℃": "celsius", "°C": "celsius", "℉": "fahrenheit", "°F": "fahrenheit",
            "秒": "second", "分钟": "minute", "小时": "hour", "%": "percent"}


def _default_specs():
    common = ("compare", "add", "subtract")
    return [
        QuantitySpec("celsius", "temperature", "absolute_temperature", "affine", "kelvin", 1.0, 273.15, common),
        QuantitySpec("fahrenheit", "temperature", "absolute_temperature", "affine", "kelvin", 5 / 9, 255.3722222222, common),
        QuantitySpec("kelvin", "temperature", "absolute_temperature", "linear", "kelvin", 1.0, 0.0, common),
        QuantitySpec("celsius", "temperature", "temperature_delta", "linear", "kelvin_delta", 1.0, 0.0, common),
        QuantitySpec("fahrenheit", "temperature", "temperature_delta", "linear", "kelvin_delta", 5 / 9, 0.0, common),
        QuantitySpec("second", "time", "duration", "linear", "second", 1.0),
        QuantitySpec("minute", "time", "duration", "linear", "second", 60.0),
        QuantitySpec("hour", "time", "duration", "linear", "second", 3600.0),
        QuantitySpec("percent", "ratio", "ratio", "linear", "ratio", 0.01),
        QuantitySpec("ratio", "ratio", "ratio", "linear", "ratio", 1.0),
        QuantitySpec("dB", "sound_level", "log_level", "logarithmic", "dB", converter_id="decibel_same_reference"),
        QuantitySpec("currency", "currency", "money", "contextual", "currency", converter_id="fx_at_timestamp"),
        QuantitySpec("m/s", "length/time", "speed", "linear", "m/s", 1.0),
        QuantitySpec("km/h", "length/time", "speed", "linear", "m/s", 1 / 3.6),
        QuantitySpec("m/s2", "length/time2", "acceleration", "linear", "m/s2", 1.0),
        QuantitySpec("pa", "pressure", "pressure", "linear", "pa", 1.0),
        QuantitySpec("kpa", "pressure", "pressure", "linear", "pa", 1000.0),
        QuantitySpec("watt", "power", "power", "linear", "watt", 1.0),
        QuantitySpec("kw", "power", "power", "linear", "watt", 1000.0),
        QuantitySpec("volt", "electric_potential", "voltage", "linear", "volt", 1.0),
        QuantitySpec("ampere", "electric_current", "current", "linear", "ampere", 1.0),
        QuantitySpec("kg/m3", "mass/volume", "density", "linear", "kg/m3", 1.0),
        QuantitySpec("m3/s", "volume/time", "flow_rate", "linear", "m3/s", 1.0),
        QuantitySpec("count/s", "count/time", "count_rate", "linear", "count/s", 1.0),
        QuantitySpec("joule", "energy", "energy", "linear", "joule", 1.0),
        QuantitySpec("kwh", "energy", "energy", "linear", "joule", 3_600_000.0),
        QuantitySpec("meter", "length", "distance", "linear", "meter", 1.0),
        QuantitySpec("km", "length", "distance", "linear", "meter", 1000.0),
    ]
