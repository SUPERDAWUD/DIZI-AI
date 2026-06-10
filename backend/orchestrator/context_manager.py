from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KVCacheStats:
    """Tracks how key-value cache size and accesses evolve."""
    entries: int = 0
    reads: int = 0
    writes: int = 0
    misses: int = 0
    max_entries: int = 0

    def record_read(self, hit: bool) -> None:
        self.reads += 1
        if not hit:
            self.misses += 1

    def record_write(self, count: int = 1) -> None:
        self.writes += 1
        self.entries += count
        self.max_entries = max(self.max_entries, self.entries)


@dataclass
class ContextManager:
    """Manages the shared context and history across agent runs."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    kv_cache: KVCacheStats = field(default_factory=KVCacheStats)

    def update(self, data: Dict[str, Any]) -> None:
        """Update the shared context with results from an agent."""
        self.state.update(data)
        self.history.append(data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def record_kv_read(self, hit: bool) -> None:
        self.kv_cache.record_read(hit)

    def record_kv_write(self, count: int = 1) -> None:
        self.kv_cache.record_write(count)
