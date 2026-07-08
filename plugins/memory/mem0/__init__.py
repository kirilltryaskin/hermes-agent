"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API or self-hosted instance.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 API key (required for cloud, optional for self-hosted)
  MEM0_HOST          — Self-hosted Mem0 URL (default: https://api.mem0.ai)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Session batching: accumulate turns in a RAM buffer and flush to the
# extraction LLM on session boundaries (switch / end / shutdown) instead of
# every turn. Cuts per-turn extraction-LLM load (~N× fewer relay calls) while
# keeping quality — the LLM sees the whole session at once. On Fly the VM is
# suspended (frozen to RAM), not killed, so the buffer survives idle sleep.
# Backstop flush when a session never rotates (slow drip / no compression):
_FLUSH_CAP_MESSAGES = 80  # ~40 turns

# Prefetch cadence: run semantic search every Nth turn, not every turn — each
# search embeds the query on the user's OpenRouter key directly (the relay
# serves no embedding models), so per-turn search is a direct provider cost.
_DEFAULT_PREFETCH_CADENCE = 3

# Trivial prompts that don't warrant a memory search.
_TRIVIAL_PROMPTS = {
    "да", "нет", "ок", "окей", "ага", "угу", "спасибо", "пасибо", "спс",
    "ok", "okay", "yes", "no", "yep", "nope", "thanks", "thx", "ty", "+",
}


def _is_trivial_prompt(query: str) -> bool:
    """True for empty/slash/very-short/acknowledgement prompts — skip search."""
    s = (query or "").strip().lower()
    if not s or s.startswith("/") or len(s) <= 3:
        return True
    return s in _TRIVIAL_PROMPTS


# One-time rotating-file handler for batching debug traces, capped at ~10 MB
# (5 MB active + 1 backup) so it never grows unbounded on the VM.
_batch_log_ready = False


def _setup_batch_logger() -> None:
    global _batch_log_ready
    if _batch_log_ready:
        return
    try:
        from logging.handlers import RotatingFileHandler
        from hermes_constants import get_hermes_home

        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            str(log_dir / "mem0-batch.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=1,             # + 1 backup → ~10 MB hard ceiling
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # don't duplicate into the agent's stdout log
        _batch_log_ready = True
    except Exception:
        # Logging setup must never break memory.
        pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": False,
        "keyword_search": False,
        "prefetch_cadence": _DEFAULT_PREFETCH_CADENCE,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search.

    Supports both Mem0 Cloud (api.mem0.ai) and self-hosted instances
    via the ``host`` config key or ``MEM0_HOST`` env var.
    """

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._host = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Session batching state
        self._session_id = ""
        self._session_batch: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._prefetch_cadence = _DEFAULT_PREFETCH_CADENCE
        self._turn_count = 0
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        host = cfg.get("host", "")
        api_key = cfg.get("api_key", "")
        return bool(host) or bool(api_key)

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 API key (cloud or self-hosted)", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 URL (e.g. http://localhost:24220)", "default": "", "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall (extra call per search)", "default": "false", "choices": ["true", "false"]},
            {"key": "prefetch_cadence", "description": "Run memory search every Nth turn (1 = every turn)", "default": str(_DEFAULT_PREFETCH_CADENCE)},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from mem0 import MemoryClient
                kwargs = {}
                if self._host:
                    kwargs["host"] = self._host
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                elif not self._host:
                    raise ValueError("Mem0: either api_key or host is required")
                self._client = MemoryClient(**kwargs)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", False)
        try:
            self._prefetch_cadence = max(1, int(
                self._config.get("prefetch_cadence", _DEFAULT_PREFETCH_CADENCE)
            ))
        except (TypeError, ValueError):
            self._prefetch_cadence = _DEFAULT_PREFETCH_CADENCE
        self._session_id = session_id or ""
        _setup_batch_logger()
        logger.info(
            "mem0 initialized: session=%s user=%s cadence=%d rerank=%s",
            self._session_id, self._user_id, self._prefetch_cadence, self._rerank,
        )

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        target = self._host or "cloud"
        return (
            f"# Mem0 Memory ({target})\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        # Cadence gate: search on turns 1, 1+N, 1+2N, … — skip the rest so we
        # don't embed the query on the user's OpenRouter key every turn.
        self._turn_count += 1
        if self._prefetch_cadence > 1 and (self._turn_count - 1) % self._prefetch_cadence != 0:
            logger.debug(
                "mem0 prefetch skipped: cadence gate (turn=%d, cadence=%d)",
                self._turn_count, self._prefetch_cadence,
            )
            return
        if _is_trivial_prompt(query):
            logger.debug("mem0 prefetch skipped: trivial prompt")
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=self._rerank,
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Buffer the turn; extraction is batched and flushed on session
        boundaries (switch / end / shutdown / buffer-cap), not per turn.

        This is the load fix: instead of one extraction-LLM call per turn, the
        whole session is sent to the extraction LLM once, on a boundary. The
        LLM sees the full conversation → cleaner deduped facts, ~N× fewer calls.
        """
        if session_id:
            self._session_id = session_id
        with self._batch_lock:
            self._session_batch.append({"role": "user", "content": user_content})
            self._session_batch.append({"role": "assistant", "content": assistant_content})
            size = len(self._session_batch)
        logger.debug(
            "mem0 buffer: +2 msg → %d total (session=%s)", size, self._session_id
        )
        # Backstop: a session that never rotates (slow drip, no compression)
        # would grow the buffer unbounded — flush at the cap.
        if size >= _FLUSH_CAP_MESSAGES:
            logger.info("mem0 buffer cap hit (%d msg) → flush", size)
            self._flush_session("buffer-cap")

    def _flush_session(self, trigger: str) -> None:
        """Drain the session buffer to the extraction LLM (non-blocking).

        Called on session boundaries. ``trigger`` labels the cause for debug
        traces. Buffer is preserved (not dropped) when the breaker is open, so
        the next trigger retries.
        """
        if self._is_breaker_open():
            logger.warning("mem0 flush deferred: breaker open (trigger=%s)", trigger)
            return
        with self._batch_lock:
            if not self._session_batch:
                logger.debug("mem0 flush skipped: empty buffer (trigger=%s)", trigger)
                return
            batch = self._session_batch
            self._session_batch = []
            session_id = self._session_id
        n = len(batch)
        logger.info(
            "mem0 flush: %d msg → extraction (trigger=%s, session=%s)",
            n, trigger, session_id,
        )

        def _sync():
            try:
                client = self._get_client()
                # infer=True: server-side extraction over the whole session →
                # clean deduped facts (not verbatim raw turns).
                client.add(batch, **self._write_filters(), infer=True)
                self._record_success()
                logger.info("mem0 flush ok: %d msg written (trigger=%s)", n, trigger)
            except Exception as e:
                self._record_failure()
                logger.warning("mem0 flush failed (trigger=%s): %s", trigger, e)

        # Wait for any previous flush before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-flush")
        self._sync_thread.start()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        # Fires on /new, /branch, /resume AND automatic context-compression
        # rotation. Any rotation = the old session is done → flush it, then
        # start accumulating under the new session_id.
        self._flush_session("switch-reset" if reset else "switch")
        self._session_id = new_session_id or self._session_id
        self._turn_count = 0

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._flush_session("session-end")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # Flush right before context compression discards old turns — facts
        # land in mem0 exactly as they leave the live context.
        self._flush_session("pre-compress")
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        # Emergency flush before teardown (redeploy / SIGTERM) so an in-flight
        # session isn't lost. entrypoint.sh traps TERM/INT → this runs.
        self._flush_session("shutdown")
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
