from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryResult:
    bandwidth_gbps: float
    latency_ns: float
    bottleneck: str


class FakeMemoryModel:
    """Simulates SRAM and DRAM bandwidth and latency."""

    def simulate(self, access_size_bytes: int, memory_type: str = "SRAM") -> MemoryResult:
        if memory_type.upper() == "SRAM":
            bandwidth = 1024.0
            latency = 10.0
        else:
            bandwidth = 256.0
            latency = 80.0

        time_ns = (access_size_bytes / (bandwidth * 1e9)) * 1e9 + latency
        bottleneck = "bandwidth" if access_size_bytes > bandwidth * 1e6 else "latency"
        return MemoryResult(bandwidth_gbps=bandwidth, latency_ns=time_ns, bottleneck=bottleneck)
