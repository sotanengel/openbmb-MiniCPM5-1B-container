"""Authenticated chat-login entry point."""

from __future__ import annotations

import sys

from minicpm_container.auth import AuthenticationError, authenticate
from minicpm_container.chat_cli import configure_stdio, run_chat_loop
from minicpm_container.generation_config_prompt import (
    prompt_enabled_tools,
    prompt_generation_config,
)
from minicpm_container.login_cli import (
    build_login_parser,
    resolve_enabled_tools,
    warn_if_network_tools_without_egress,
)


def main(argv: list[str] | None = None) -> None:
    configure_stdio()

    parser = build_login_parser()
    args = parser.parse_args(argv)

    try:
        authenticate()
    except AuthenticationError as exc:
        print(f"Login denied: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    enabled_tools = resolve_enabled_tools(
        args,
        prompt_callback=lambda: prompt_enabled_tools(()),
    )
    warn_if_network_tools_without_egress(enabled_tools)

    config = prompt_generation_config(
        enabled_tools=enabled_tools,
        prompt_tools=args.tools_prompt,
    )
    run_chat_loop(config=config)


if __name__ == "__main__":
    main()
