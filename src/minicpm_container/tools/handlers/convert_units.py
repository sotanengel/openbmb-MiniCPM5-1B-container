"""Unit conversion tool handler."""

from __future__ import annotations

from typing import Any

_UNIT_TO_BASE: dict[str, tuple[str, float]] = {
    "m": ("length_m", 1.0),
    "meter": ("length_m", 1.0),
    "meters": ("length_m", 1.0),
    "km": ("length_m", 1000.0),
    "kilometer": ("length_m", 1000.0),
    "kilometers": ("length_m", 1000.0),
    "cm": ("length_m", 0.01),
    "centimeter": ("length_m", 0.01),
    "centimeters": ("length_m", 0.01),
    "mm": ("length_m", 0.001),
    "millimeter": ("length_m", 0.001),
    "millimeters": ("length_m", 0.001),
    "g": ("mass_kg", 0.001),
    "gram": ("mass_kg", 0.001),
    "grams": ("mass_kg", 0.001),
    "kg": ("mass_kg", 1.0),
    "kilogram": ("mass_kg", 1.0),
    "kilograms": ("mass_kg", 1.0),
    "b": ("bytes", 1.0),
    "byte": ("bytes", 1.0),
    "bytes": ("bytes", 1.0),
    "kb": ("bytes", 1024.0),
    "kilobyte": ("bytes", 1024.0),
    "kilobytes": ("bytes", 1024.0),
    "mb": ("bytes", 1024.0**2),
    "megabyte": ("bytes", 1024.0**2),
    "megabytes": ("bytes", 1024.0**2),
}

_TEMPERATURE_UNITS = frozenset({"celsius", "c", "fahrenheit", "f"})


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    from_key = from_unit.strip().lower()
    to_key = to_unit.strip().lower()

    if from_key in _TEMPERATURE_UNITS and to_key in _TEMPERATURE_UNITS:
        from_c = value if from_key in {"celsius", "c"} else (value - 32) * 5 / 9
        result = from_c if to_key in {"celsius", "c"} else from_c * 9 / 5 + 32
        return str(result)

    from_entry = _UNIT_TO_BASE.get(from_key)
    to_entry = _UNIT_TO_BASE.get(to_key)
    if not from_entry or not to_entry:
        raise ValueError("unsupported unit; use length, mass, temperature, or bytes units")
    if from_entry[0] != to_entry[0]:
        raise ValueError("cannot convert between different unit categories")

    base_value = value * from_entry[1]
    converted = base_value / to_entry[1]
    if converted.is_integer():
        return str(int(converted))
    return str(converted)


def handle_convert_units(arguments: dict[str, Any]) -> str:
    from minicpm_container.tools.handlers.common import require_str

    raw_value = arguments.get("value")
    if not isinstance(raw_value, int | float):
        raise ValueError("value must be a number")
    from_unit = require_str(arguments, "from_unit")
    to_unit = require_str(arguments, "to_unit")
    return convert_units(float(raw_value), from_unit, to_unit)
