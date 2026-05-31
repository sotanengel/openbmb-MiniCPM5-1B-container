"""Limits for tool execution and agent loop."""

MAX_TOOL_RESULT_CHARS = 4096
MAX_WEB_SEARCH_RESULT_CHARS = 1200
MAX_TOOL_ROUNDS = 5
MAX_TOOL_CALLS_PER_RESPONSE = 3
MAX_EXPRESSION_CHARS = 256
MAX_TEXT_ARG_CHARS = 8192
MAX_URL_CHARS = 2048
MAX_HTTP_RESPONSE_BYTES = 32 * 1024
HTTP_TIMEOUT_SECONDS = 10
HTTP_MAX_REDIRECTS = 3
MAX_SEARCH_QUERY_CHARS = 500

NETWORK_REQUIRED_MESSAGE = (
    "Network tools require egress. Start the container with network enabled "
    "(e.g. MINICPM_NETWORK=1 ./scripts/run.sh or docker-compose.network.yml)."
)
