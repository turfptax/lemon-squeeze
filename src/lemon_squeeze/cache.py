"""Process-local cache for hot read paths.

The `/route` endpoint hits `stats_by_tag` on every request, which re-runs the
PromptTag → Run → Eval → Model aggregation. In steady state (no new evals)
the result is identical across thousands of requests. This module caches it
in memory and invalidates when the underlying tables change.

Design choices:
  - Process-local only. The HTTP server with multiple workers each gets its
    own cache; that's fine — they all hit the same SQLite file and the cache
    is purely a hot-path acceleration, not a coherence layer.
  - Invalidation is push-based, not pull. SQLAlchemy `after_flush` events fire
    `bump()` when any of (Run, Evaluation, PromptTag, Model) inserts. Reads
    that come after see the new key version and miss.
  - Plus a TTL safety net (default 60s) so a write that somehow bypassed
    the ORM can't pin a stale view forever.
  - Stats: hits, misses, invalidations exposed via `cache_stats()` for the
    /metrics endpoint.

NOT cached: anything that takes parameters that vary per request beyond the
known few keys (the prompt text in /classify, for instance). Only the aggregate
queries are stable enough to be worth caching.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Cache config — tuned conservatively.
DEFAULT_MAX_ENTRIES = 256
DEFAULT_TTL_SECONDS = 60.0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    size: int = 0
    entries_evicted: int = 0


@dataclass
class _Entry:
    value: Any
    expires_at: float
    version: int


class _LRU:
    """LRU + TTL + version-tag invalidation. Thread-safe."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[Any, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._version = 0  # monotonic; bumps on writes
        self._stats = CacheStats()

    def get(self, key: Any) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return _MISS
            if entry.version < self._version or entry.expires_at < now:
                # Stale — drop it.
                del self._data[key]
                self._stats.misses += 1
                return _MISS
            self._data.move_to_end(key)  # mark as recently used
            self._stats.hits += 1
            return entry.value

    def put(self, key: Any, value: Any, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        """Insert a value with per-entry TTL. Default is module-wide TTL; tests
        and code with short-lived values can pass a smaller window."""
        with self._lock:
            self._data[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + ttl_seconds,
                version=self._version,
            )
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self._stats.entries_evicted += 1

    def bump_version(self) -> None:
        """Invalidate every cached entry. Cheap; readers detect via version compare."""
        with self._lock:
            self._version += 1
            self._stats.invalidations += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._version += 1
            self._stats.invalidations += 1

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                invalidations=self._stats.invalidations,
                size=len(self._data),
                entries_evicted=self._stats.entries_evicted,
            )


_MISS = object()
_aggregations_cache = _LRU()


def aggregations_cache() -> _LRU:
    """Public handle to the aggregation cache."""
    return _aggregations_cache


def cache_stats() -> dict[str, CacheStats]:
    """All known caches, keyed by name. Returned for /metrics."""
    return {"aggregations": _aggregations_cache.stats()}


# ---------- ORM event hooks --------------------------------------------------


def install_invalidation_hooks() -> None:
    """Wire SQLAlchemy `after_flush` events to bump cache versions.

    Idempotent. Called from `db.session.get_engine` so it runs once when the
    engine is built. If callers somehow bypass the ORM (raw INSERT via
    `Connection.execute`), the TTL is the safety net.
    """
    from sqlalchemy import event

    from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run
    from lemon_squeeze.db.session import _sessionmaker  # type: ignore

    import itertools
    watched = (Prompt, PromptTag, Model, Run, Evaluation)

    def _after_flush(session, flush_context):
        touched = itertools.chain(session.new, session.dirty, session.deleted)
        if any(isinstance(o, watched) for o in touched):
            _aggregations_cache.bump_version()

    sm = _sessionmaker()
    if getattr(sm, "_lemon_cache_hook_installed", False):
        return
    event.listen(sm, "after_flush", _after_flush)
    sm._lemon_cache_hook_installed = True  # type: ignore[attr-defined]


# ---------- Cache key helpers ------------------------------------------------


def aggregations_key(
    *,
    fn: str,
    rubrics: Sequence[str],
    tags: Sequence[str] | None,
    model_names: Sequence[str] | None,
    prompt_ids: Sequence[int] | None,
) -> tuple:
    """Hashable cache key for an aggregations call."""
    return (
        fn,
        tuple(sorted(rubrics)),
        tuple(sorted(tags)) if tags is not None else None,
        tuple(sorted(model_names)) if model_names is not None else None,
        tuple(sorted(prompt_ids)) if prompt_ids is not None else None,
    )
