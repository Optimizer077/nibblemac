"""OpenAI-compatible HTTP server for one pinfer model (``pinfer serve``).

    import pinfer
    from pinfer.server import EngineConfig, serve

    with pinfer.Model.load("qwen35-0.8b-q4k-headq4k.pq") as m:
        serve(m, EngineConfig(model_name="qwen3.5-0.8b"), host="0.0.0.0", port=8080)

Standard library only (``http.server``). Endpoints, options and the prefix-reuse design:
``docs/library.md`` "Server"; modules: :mod:`.protocol` (wire format), :mod:`.engine` (the
model context), :mod:`.history` (prompt tokens of a conversation), :mod:`.cache` (snapshots,
replies, request queue), :mod:`.httpd` (HTTP).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .engine import Engine, EngineConfig, RunStats
from .httpd import Server, make_server
from .protocol import APIError, Defaults

if TYPE_CHECKING:
    from ..model import Model

logger = logging.getLogger("pinfer.server")

__all__ = ["APIError", "Defaults", "Engine", "EngineConfig", "RunStats", "Server", "make_server",
           "serve"]


def serve(model: Model, cfg: EngineConfig, host: str = "127.0.0.1", port: int = 8080,
          api_key: str | None = None) -> None:
    """Serve ``model`` until interrupted (Ctrl-C / SIGINT)."""
    engine = Engine(model, cfg)
    server = make_server(engine, host, port, api_key)
    spec = (f"speculative k={cfg.spec_k or 'default'}" if engine.speculative
            else "speculative off")
    logger.info("serving %s on http://%s:%d/v1 (max_ctx %d, kv %s, %s, reuse %s, history %s, "
                "state cache %.0f MB%s)", cfg.model_name, host, server.server_address[1],
                model.max_ctx, model.kv_dtype, spec, "on" if cfg.reuse else "off",
                cfg.history, cfg.state_cache_mb if engine.snapshots else 0,
                ", API key required" if api_key else "")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    finally:
        server.server_close()
