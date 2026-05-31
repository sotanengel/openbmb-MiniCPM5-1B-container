"""Login-time CLI arguments for chat-login."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from minicpm_container.tools.registry import NETWORK_TOOL_IDS, parse_enabled_tools

CHAT_TOOLS_ENV = "CHAT_TOOLS"


def build_login_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-login",
        description="Authenticate and start the secure MiniCPM5 chat session.",
    )
    parser.add_argument(
        "--tools",
        metavar="TOOLS",
        default=None,
        help=(
            "Comma-separated tool ids to enable (none disables). "
            "Example: calculate,http_get,web_search"
        ),
    )
    parser.add_argument(
        "--tools-prompt",
        action="store_true",
        help="Interactively ask which tools to enable after login.",
    )
    return parser


def resolve_enabled_tools(
    args: argparse.Namespace,
    *,
    prompt_callback: Callable[[], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    if args.tools is not None:
        return parse_enabled_tools(args.tools)

    env_value = os.environ.get(CHAT_TOOLS_ENV, "").strip()
    if env_value:
        return parse_enabled_tools(env_value)

    if args.tools_prompt and prompt_callback is not None:
        return prompt_callback()

    return ()


def warn_if_network_tools_without_egress(enabled_tools: tuple[str, ...]) -> None:
    if not enabled_tools:
        return
    if not NETWORK_TOOL_IDS.intersection(enabled_tools):
        return
    from minicpm_container.tools.http_client import is_network_enabled

    if is_network_enabled():
        return
    import sys

    print(
        "Warning: http_get/web_search are enabled but CHAT_NETWORK_ENABLED is not set. "
        "Start with MINICPM_NETWORK=1 ./scripts/run.sh (or docker-compose.network.yml).",
        file=sys.stderr,
    )
