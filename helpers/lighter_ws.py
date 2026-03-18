import os


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_lighter_ws_url() -> str:
    """Build Lighter WebSocket URL with optional legacy server ping fallback."""
    base_url = os.getenv("LIGHTER_WS_URL", "wss://mainnet.zklighter.elliot.ai/stream").strip()
    if not base_url:
        base_url = "wss://mainnet.zklighter.elliot.ai/stream"

    use_server_pings = _is_truthy(os.getenv("LIGHTER_WS_SERVER_PINGS", "false"))
    if not use_server_pings:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}server_pings=true"


def lighter_ws_connect_kwargs() -> dict:
    """
    WebSocket settings compatible with both old and new Lighter WS behavior.
    - Send client ping frames regularly (required by new server behavior).
    - Keep permessage-deflate enabled.
    """
    return {
        "ping_interval": 50,
        "ping_timeout": 20,
        "compression": "deflate",
        "max_queue": 1024,
    }
