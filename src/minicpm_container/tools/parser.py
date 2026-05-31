"""Parse MiniCPM5 XML-style tool calls from model output."""

from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from minicpm_container.generation_decode import strip_thinking_blocks

TOOL_CALL_REGEX = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
FUNC_CALL_REGEX = re.compile(r"<function.*?</function>", re.DOTALL)
FUNC_NAME_REGEX = re.compile(r'<function\s+name=[\'"]([^\'"]+)[\'"][^>]*>')
PARAM_WITH_NAME_REGEX = re.compile(
    r'<param\s+name=[\'"]([^\'"]+)[\'"]>([\s\S]*?)</param>',
    re.DOTALL,
)
PARAM_MISSING_NAME_REGEX = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParseResult:
    normal_text: str
    calls: list[ParsedToolCall]


def _parse_param_value(raw: str, arg_type: str | None) -> Any:
    text = raw.strip()
    if arg_type != "string":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError, TypeError):
                return text
    return text


def _schema_lookup(
    tool_schemas: list[dict[str, Any]],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    names: set[str] = set()
    allowed_props: dict[str, set[str]] = {}
    required_props: dict[str, set[str]] = {}
    prop_types: dict[str, dict[str, str]] = {}

    for tool in tool_schemas:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        names.add(name)
        params = function.get("parameters")
        if not isinstance(params, dict):
            continue
        properties = params.get("properties")
        if isinstance(properties, dict):
            allowed_props[name] = set(properties.keys())
            prop_types[name] = {
                key: value.get("type", "string")
                for key, value in properties.items()
                if isinstance(value, dict)
            }
        required = params.get("required")
        if isinstance(required, list):
            required_props[name] = {item for item in required if isinstance(item, str)}
        else:
            required_props[name] = set()

    return names, allowed_props, required_props, prop_types


def _parse_function_block(
    block: str,
    *,
    tool_names: set[str],
    allowed_props: dict[str, set[str]],
    required_props: dict[str, set[str]],
    prop_types: dict[str, dict[str, str]],
) -> ParsedToolCall | None:
    func_name: str | None = None
    arguments: dict[str, Any] = {}
    parsed_ok = False
    param_invalid = False

    try:
        root = ET.fromstring(block)
        func_node = root if root.tag == "function" else root.find("function")
        if func_node is not None:
            func_name = (func_node.attrib.get("name") or "").strip()
            param_nodes = list(func_node.findall("param"))
            args_node = func_node.find("arguments")
            if args_node is not None and not param_nodes:
                param_nodes = list(args_node.findall("param"))

            if func_name in tool_names:
                allowed = allowed_props.get(func_name, set())
                seen: set[str] = set()
                for param in param_nodes:
                    key = param.attrib.get("name")
                    if not key or (allowed and key not in allowed):
                        param_invalid = True
                        break
                    if key in seen:
                        param_invalid = True
                        break
                    seen.add(key)
                    arg_type = prop_types.get(func_name, {}).get(key)
                    arguments[key] = _parse_param_value(param.text or "", arg_type)
                if not param_invalid:
                    parsed_ok = bool(func_name)
    except ET.ParseError:
        parsed_ok = False

    if not parsed_ok:
        match = FUNC_NAME_REGEX.search(block)
        if match:
            func_name = (match.group(1) or "").strip()
            param_invalid = PARAM_MISSING_NAME_REGEX.search(block) is not None
            if func_name in tool_names and not param_invalid:
                allowed = allowed_props.get(func_name, set())
                seen: set[str] = set()
                for param_match in PARAM_WITH_NAME_REGEX.finditer(block):
                    key = param_match.group(1).strip()
                    if allowed and key not in allowed:
                        param_invalid = True
                        break
                    if key in seen:
                        param_invalid = True
                        break
                    seen.add(key)
                    val_text = (param_match.group(2) or "").strip()
                    arg_type = prop_types.get(func_name, {}).get(key)
                    arguments[key] = _parse_param_value(val_text, arg_type)
                parsed_ok = bool(func_name) and not param_invalid

    if not func_name or func_name not in tool_names or param_invalid:
        return None

    req = required_props.get(func_name, set())
    if req and not req.issubset(arguments.keys()):
        return None

    return ParsedToolCall(name=func_name, arguments=arguments)


def _strip_tool_call_wrappers(text: str) -> str:
    """Keep inner <function> blocks; drop <tool_call> wrapper tokens from normal text."""
    return TOOL_CALL_REGEX.sub("", text)


def _normalize_tool_text(text: str) -> str:
    cleaned = strip_thinking_blocks(text)
    cleaned = cleaned.replace("<|im_sep|>", "")
    cleaned = re.sub(r"<+(\{)", r"\1", cleaned)
    return cleaned.strip()


def _extract_json_object_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        depth = 0
        for offset in range(start, len(text)):
            char = text[offset]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : offset + 1]
                    if '"name"' in candidate:
                        spans.append((start, offset + 1, candidate))
                    index = offset + 1
                    break
        else:
            break
    return spans


def _validate_arguments(
    func_name: str,
    arguments: dict[str, Any],
    *,
    allowed_props: dict[str, set[str]],
    required_props: dict[str, set[str]],
) -> bool:
    allowed = allowed_props.get(func_name, set())
    if allowed and not set(arguments.keys()).issubset(allowed):
        return False
    req = required_props.get(func_name, set())
    return not req or req.issubset(arguments.keys())


def _parse_json_tool_call(
    payload: dict[str, Any],
    *,
    tool_names: set[str],
    allowed_props: dict[str, set[str]],
    required_props: dict[str, set[str]],
) -> ParsedToolCall | None:
    name = payload.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None

    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {
            key: value
            for key, value in payload.items()
            if key not in {"name", "type", "function"}
            and isinstance(value, (str, int, float, bool))
        }
    if not isinstance(arguments, dict) or not arguments:
        return None
    normalized = {str(key): value for key, value in arguments.items()}
    if not _validate_arguments(
        name,
        normalized,
        allowed_props=allowed_props,
        required_props=required_props,
    ):
        return None
    return ParsedToolCall(name=name, arguments=normalized)


def _parse_json_tool_calls(
    text: str,
    *,
    tool_names: set[str],
    allowed_props: dict[str, set[str]],
    required_props: dict[str, set[str]],
) -> tuple[list[ParsedToolCall], str]:
    calls: list[ParsedToolCall] = []
    remove_ranges: list[tuple[int, int]] = []
    for start, end, candidate in _extract_json_object_spans(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        parsed = _parse_json_tool_call(
            payload,
            tool_names=tool_names,
            allowed_props=allowed_props,
            required_props=required_props,
        )
        if parsed is None:
            continue
        calls.append(parsed)
        remove_ranges.append((start, end))

    if not remove_ranges:
        return calls, text

    normal_parts: list[str] = []
    cursor = 0
    for start, end in sorted(remove_ranges):
        normal_parts.append(text[cursor:start])
        cursor = end
    normal_parts.append(text[cursor:])
    normal_text = "".join(normal_parts).strip()
    return calls, normal_text


def parse_tool_calls(text: str, tool_schemas: list[dict[str, Any]] | None) -> ParseResult:
    if not tool_schemas:
        return ParseResult(normal_text=text, calls=[])

    cleaned = _normalize_tool_text(text)
    tool_names, allowed_props, required_props, prop_types = _schema_lookup(tool_schemas)
    normal_parts: list[str] = []
    calls: list[ParsedToolCall] = []
    last_end = 0

    for match in FUNC_CALL_REGEX.finditer(cleaned):
        if match.start() > last_end:
            normal_parts.append(cleaned[last_end : match.start()])

        parsed = _parse_function_block(
            match.group(0),
            tool_names=tool_names,
            allowed_props=allowed_props,
            required_props=required_props,
            prop_types=prop_types,
        )
        if parsed is not None:
            calls.append(parsed)
        else:
            normal_parts.append(match.group(0))
        last_end = match.end()

    if last_end < len(cleaned):
        normal_parts.append(cleaned[last_end:])

    normal_text = _strip_tool_call_wrappers("".join(normal_parts)).strip()

    if not calls:
        json_calls, normal_text = _parse_json_tool_calls(
            cleaned,
            tool_names=tool_names,
            allowed_props=allowed_props,
            required_props=required_props,
        )
        calls.extend(json_calls)

    return ParseResult(normal_text=normal_text, calls=calls)
