from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheStats:
    entries: int = 0
    reads: int = 0
    writes: int = 0
    misses: int = 0
    capacity: int = 2048

    def read(self, hit: bool) -> None:
        self.reads += 1
        if not hit:
            self.misses += 1

    def write(self, count: int = 1) -> None:
        self.writes += 1
        self.entries += count
        if self.entries > self.capacity:
            self.entries = self.capacity

    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.entries / self.capacity
