"""Execute whitelisted tools and return string results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from minicpm_container.tools.calculate import CalculateError, safe_calculate
from minicpm_container.tools.http_client import HttpClientError, http_get, web_search
from minicpm_container.tools.limits import (
    MAX_TEXT_ARG_CHARS,
    MAX_TOOL_RESULT_CHARS,
)
from minicpm_container.tools.registry import ALL_TOOL_IDS

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


def _truncate(result: str) -> str:
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return result[: MAX_TOOL_RESULT_CHARS - 1] + "…"


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _convert_units(value: float, from_unit: str, to_unit: str) -> str:
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


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name not in ALL_TOOL_IDS:
        return f"error: unknown tool {name!r}"

    if not isinstance(arguments, dict):
        return "error: arguments must be an object"

    try:
        if name == "calculate":
            expression = _require_str(arguments, "expression")
            return _truncate(safe_calculate(expression))
        if name == "current_datetime":
            return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017
        if name == "count_text":
            text = _require_str(arguments, "text")
            if len(text) > MAX_TEXT_ARG_CHARS:
                raise ValueError(f"text exceeds {MAX_TEXT_ARG_CHARS} characters")
            unit = _require_str(arguments, "unit").lower()
            if unit == "characters":
                return str(len(text))
            if unit == "words":
                return str(len(text.split()))
            if unit == "lines":
                return str(len(text.splitlines()))
            raise ValueError("unit must be characters, words, or lines")
        if name == "convert_units":
            raw_value = arguments.get("value")
            if not isinstance(raw_value, int | float):
                raise ValueError("value must be a number")
            from_unit = _require_str(arguments, "from_unit")
            to_unit = _require_str(arguments, "to_unit")
            return _truncate(_convert_units(float(raw_value), from_unit, to_unit))
        if name == "http_get":
            url = _require_str(arguments, "url")
            return _truncate(http_get(url))
        if name == "web_search":
            query = _require_str(arguments, "query")
            return _truncate(web_search(query))
    except (CalculateError, HttpClientError, ValueError) as exc:
        return f"error: {exc}"

    return f"error: tool {name!r} is not implemented"
