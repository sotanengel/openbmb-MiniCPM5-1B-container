#!/usr/bin/env bash
# E2E: piped chat-login against running minicpm5-1b-chat container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-1b-chat}"
export E2E_PASS="${E2E_PASS:-e2e-verify-local-only}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
export TOOLS="${1:?usage: e2e_chat.sh TOOL_ID [user message]}"
export USER_MSG="${2:-Use the calculate tool to compute 17 times 23. Answer briefly in English.}"
export NETWORK="${NETWORK:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
export CONTAINER_NAME

if ! docker exec "${CONTAINER_NAME}" test -S /run/model.sock 2>/dev/null; then
  echo "Model socket not ready. Run ./scripts/sync.sh first." >&2
  exit 1
fi

python3 - <<'PY'
import os
import subprocess
import sys

password = os.environ["E2E_PASS"]
user_msg = os.environ["USER_MSG"]
tools = os.environ["TOOLS"]
timeout = int(os.environ["TIMEOUT_SEC"])
container = os.environ["CONTAINER_NAME"]
network = os.environ.get("NETWORK") == "1"
max_tokens = os.environ.get("MAX_NEW_TOKENS", "").strip()
if not max_tokens:
    max_tokens = "256" if tools == "web_search" else "128"

stdin = "\n".join(
    [
        password,
        max_tokens,
        "no",
        "",
        "",
        "",
        user_msg,
        "/exit",
        "",
    ]
)

cmd = [
    "docker",
    "exec",
    "-u",
    "chat",
    "-i",
    "-e",
    "LANG=C.UTF-8",
    "-e",
    "LC_ALL=C.UTF-8",
    "-e",
    "PYTHONIOENCODING=utf-8",
]
if network:
    cmd.extend(["-e", "CHAT_NETWORK_ENABLED=1"])
cmd.extend([container, "chat-login", "--tools", tools])

print(
    f"E2E tools={tools} network={network} max_new_tokens={max_tokens} timeout={timeout}s",
    flush=True,
)
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
try:
    out, _ = proc.communicate(input=stdin, timeout=timeout)
except subprocess.TimeoutExpired:
    proc.kill()
    out, _ = proc.communicate()
    print(out, end="")
    print("E2E timed out", file=sys.stderr)
    sys.exit(1)

print(out, end="")
if proc.returncode:
    sys.exit(proc.returncode)

if tools == "calculate" and "391" not in out:
    print("E2E failed: expected 391 in output", file=sys.stderr)
    sys.exit(1)

if tools == "web_search":
    lowered = out.lower()
    if "bots use duckduckgo" in lowered:
        print("E2E failed: DuckDuckGo CAPTCHA text in output", file=sys.stderr)
        sys.exit(1)
    if "コナン" not in out:
        print("E2E failed: expected コナン in output", file=sys.stderr)
        sys.exit(1)
    if not any(keyword in out for keyword in ("漫画", "青山", "推理", "manga")):
        print("E2E failed: expected manga/author/detective context in output", file=sys.stderr)
        sys.exit(1)
    print("E2E web_search assertions passed", flush=True)

if tools == "calculate":
    print("E2E calculate assertions passed", flush=True)

sys.exit(0)
PY
