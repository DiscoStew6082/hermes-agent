"""Mem0 memory plugin — MemoryProvider interface.

Supports two modes:
1. Platform mode: Uses Mem0 Platform API with server-side LLM extraction
2. Local/OSS mode: Self-hosted with custom LLM, embedder, and vector store

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables (Platform mode):
  MEM0_API_KEY       — Mem0 Platform API key (required for platform mode)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.

For Local/OSS mode, configure in mem0.json:
{
  "mode": "local",
  "llm": {"provider": "lmstudio", ...},
  "embedder": {"provider": "lmstudio", ...},
  "vector_store": {"provider": "faiss", ...}
}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import get_hermes_home

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    
    Supports both Platform mode and Local/OSS mode:
    - Platform: requires MEM0_API_KEY, uses MemoryClient
    - Local: requires mode="local" plus llm/embedder/vector_store config
    """
    # Base config with all possible fields (defaults)
    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
        # Local mode defaults
        "llm_provider": "lmstudio",
        "llm_model": "google/gemma-4-26b-a4b",
        "llm_base_url": os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "embedder_provider": "lmstudio",
        "embedder_model": "text-embedding-kalm-embedding-gemma3-12b-2511-i1",
        "embedder_dims": 3840,
        "vector_store_provider": "faiss",
        "vector_store_path": None,  # Will default to HERMES_HOME/mem0_vectors
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
    """Mem0 memory with Platform or Local/OSS mode support.

    Mode detection:
    - If "mode": "local" in mem0.json → use Memory.from_config (OSS)
    - Otherwise require MEM0_API_KEY for platform mode
    """

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._mode = "platform"  # "platform" or "local"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "")
        
        # Local mode: check for flat config keys (llm_provider, embedder_provider)
        if mode == "local":
            has_llm = bool(cfg.get("llm_provider") or cfg.get("llm"))
            has_embedder = bool(cfg.get("embedder_provider") or cfg.get("embedder"))
            return has_llm and has_embedder
        
        # Platform mode: require API key
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Mode: platform (API) or local (self-hosted)", "default": "platform", "choices": ["platform", "local"]},
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            # Platform mode options
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
            # Local mode options - LLM
            {"key": "llm_provider", "description": "Local LLM provider (lmstudio, ollama, openai)", "when": {"mode": "local"}},
            {"key": "llm_model", "description": "LLM model name for fact extraction", "default_from": {"field": "llm_provider", "map": {"lmstudio": "google/gemma-4-26b-a4b", "ollama": "llama3.1:latest", "openai": "gpt-4o-mini"}}},
            {"key": "llm_base_url", "description": "Base URL for local LLM (e.g., http://localhost:1234/v1)", "default": "http://localhost:1234/v1"},
            # Local mode options - Embedder
            {"key": "embedder_provider", "description": "Embedder provider (lmstudio, ollama, openai)", "when": {"mode": "local"}},
            {"key": "embedder_model", "description": "Embedding model name", "default_from": {"field": "embedder_provider", "map": {"lmstudio": "text-embedding-kalm-embedding-gemma3-12b-2511-i1", "ollama": "nomic-embed-text:latest"}}},
            {"key": "embedder_dims", "description": "Embedding dimensions", "default_from": {"field": "embedder_model", "map": {"text-embedding-kalm-embedding-gemma3-12b-2511-i1": 3840, "nomic-embed-text:latest": 768}}, "default": 1536},
            # Local mode options - Vector Store
            {"key": "vector_store_provider", "description": "Vector store provider (faiss, qdrant, chroma)", "when": {"mode": "local"}, "default": "faiss"},
            {"key": "vector_store_path", "description": "Path for faiss/chroma storage (defaults to HERMES_HOME/mem0_vectors)", "default": ""},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization.
        
        Platform mode: uses MemoryClient (API-based)
        Local mode: uses Memory.from_config() (OSS self-hosted)
        """
        with self._client_lock:
            if self._client is not None:
                return self._client
            
            try:
                from mem0 import Memory
                
                if self._mode == "local":
                    # Build OSS config - matching skill doc format exactly
                    embedder_dims = int(self._config.get("embedder_dims", 3840))
                    
                    config = {
                        "llm": {
                            "provider": self._config.get("llm_provider", "lmstudio"),
                            "config": {
                                "model": self._config.get("llm_model", "google/gemma-4-26b-a4b"),
                                "lmstudio_base_url": self._config.get("llm_base_url", "http://localhost:1234/v1"),
                            }
                        },
                        # Note: embedder config structure matches skill doc
                        "embedder": {
                            "provider": self._config.get("embedder_provider", "lmstudio"),
                            "config": {
                                "model": self._config.get("embedder_model", "text-embedding-kalm-embedding-gemma3-12b-2511-i1"),
                                "lmstudio_base_url": self._config.get("llm_base_url", "http://localhost:1234/v1"),
                                # Key names from skill doc - embedding_dims at top level of config
                            }
                        },
                        # Handle both naming conventions for dimensions
                    }
                    
                    # Add dimensions to both locations (handles different mem0 versions)
                    if embedder_dims:
                        config["embedder"]["config"]["embedding_dims"] = embedder_dims
                        # Default to HERMES_HOME/mem0_vectors if not set
                    default_vector_path = str(Path(get_hermes_home()) / "mem0_vectors")

                    config["vector_store"] = {
                        "provider": self._config.get("vector_store_provider", "faiss"),
                        "config": {
                            "path": self._config.get("vector_store_path") or default_vector_path,
                            # Key name from skill doc
                            "embedding_model_dims": embedder_dims,
                        }
                    }

                    openai_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_key and self._config.get("llm_provider") == "openai":
                        logger.warning("Local mode with openai provider but no OPENAI_API_KEY set")
                    
                    logger.info(f"Mem0 local mode: llm={config['llm']['provider']}/{config['llm']['config'].get('model')}, embedder={config['embedder']['provider']}/{config['embedder']['config'].get('model')}")
                    
                    self._client = Memory.from_config(config)
                else:
                    # Platform mode
                    from mem0 import MemoryClient
                    if not self._api_key:
                        raise RuntimeError("MEM0_API_KEY required for platform mode")
                    self._client = MemoryClient(api_key=self._api_key)
                    
                return self._client
                
            except ImportError as e:
                logger.error(f"Failed to import mem0 package: {e}")
                if "Memory.from_config" in str(e):
                    raise RuntimeError("mem0ai package version too old for local mode. Run: uv pip install --upgrade mem0ai")
                raise RuntimeError("mem0 package not installed. Run: uv pip install mem0ai")

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
        self._user_id = self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)
        
        # Detect mode - check config first, then env var
        cfg_mode = self._config.get("mode", "").lower()
        env_mode = os.environ.get("MEM0_MODE", "").lower()
        self._mode = cfg_mode or env_mode or "platform"
        
        logger.info(f"Mem0 initializing in {self._mode} mode")

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
        return (
            "# Mem0 Memory\n"
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

        def _run():
            try:
                client = self._get_client()
                # Platform uses filters+top_k, OSS uses user_id+limit
                search_kwargs = dict(
                    query=query,
                    user_id=self._user_id,
                    rerank=self._rerank,
                    limit=5,
                )
                if self._mode != "local":
                    search_kwargs.update(filters=self._read_filters(), top_k=5)
                
                results = self._unwrap_results(client.search(**search_kwargs))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {line}" for line in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                # Pass both platform and OSS style - each client takes what it needs
                client.add(messages, **self._write_filters(), infer=True)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

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
            return json.dumps({"error": str(e)})

        if tool_name == "mem0_profile":
            try:
                # Platform uses filters, OSS uses user_id+limit - pass both
                get_all_kwargs = dict(
                    user_id=self._user_id,
                    limit=100,
                )
                if self._mode != "local":
                    get_all_kwargs["filters"] = self._read_filters()
                
                memories = self._unwrap_results(client.get_all(**get_all_kwargs))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Failed to fetch profile: {e}"})

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            rerank = args.get("rerank", False)
            limit = min(int(args.get("top_k", 10)), 50)
            try:
                # Platform uses filters+top_k, OSS uses user_id+limit - pass both, each takes what it needs
                search_kwargs = dict(
                    query=query,
                    user_id=self._user_id,
                    rerank=rerank,
                    limit=limit,
                )
                if self._mode != "local":
                    # Platform mode also wants filters and top_k (OSS rejects these)
                    search_kwargs.update(filters=self._read_filters(), top_k=limit)
                
                results = self._unwrap_results(client.search(**search_kwargs))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Search failed: {e}"})

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return json.dumps({"error": "Missing required parameter: conclusion"})
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
                return json.dumps({"error": f"Failed to store: {e}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
