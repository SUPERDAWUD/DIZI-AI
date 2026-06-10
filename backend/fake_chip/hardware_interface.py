from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .kv_cache import KVCacheStats
from .matmul import FakeMatmul, MatmulTileResult
from .memory import FakeMemoryModel, MemoryResult
from .power import FakePowerModel, PowerResult


@dataclass
class HardwareProfile:
    matmul_latency_ms: float
    matmul_energy_mj: float
    memory_latency_ns: float
    memory_bandwidth_gbps: float
    power_energy_mj: float
    power_thermal_w: float
    kv_cache_utilization: float
    kv_reads: int
    kv_writes: int
    kv_misses: int


class FakeChipInterface:
    """Simulates a fake accelerator for profiling local model execution."""

    def __init__(self) -> None:
        self.matmul = FakeMatmul()
        self.memory = FakeMemoryModel()
        self.power = FakePowerModel()
        self.kv_cache = KVCacheStats()

    def execute_attention(self, seq_len: int, head_dim: int, num_heads: int, precision: str = "int8") -> HardwareProfile:
        matmul = self.matmul.simulate(seq_len, seq_len, head_dim * num_heads, precision=precision)
        memory = self.memory.simulate(access_size_bytes=seq_len * head_dim * num_heads * 4, memory_type="SRAM")
        power = self.power.estimate(active_units=num_heads, runtime_ms=matmul.latency_ms, precision=precision)
        self.kv_cache.read(hit=True)
        self.kv_cache.write(count=1)
        return HardwareProfile(
            matmul_latency_ms=matmul.latency_ms,
            matmul_energy_mj=matmul.energy_mj,
            memory_latency_ns=memory.latency_ns,
            memory_bandwidth_gbps=memory.bandwidth_gbps,
            power_energy_mj=power.energy_mj,
            power_thermal_w=power.thermal_w,
            kv_cache_utilization=self.kv_cache.utilization(),
            kv_reads=self.kv_cache.reads,
            kv_writes=self.kv_cache.writes,
            kv_misses=self.kv_cache.misses,
        )

    def execute_token(self, token_count: int, precision: str = "int8") -> HardwareProfile:
        matmul = self.matmul.simulate(token_count, token_count, 1024, precision=precision)
        memory = self.memory.simulate(access_size_bytes=token_count * 1024 * 4, memory_type="DRAM")
        power = self.power.estimate(active_units=4, runtime_ms=matmul.latency_ms, precision=precision)
        self.kv_cache.read(hit=False)
        self.kv_cache.write(count=token_count)
        return HardwareProfile(
            matmul_latency_ms=matmul.latency_ms,
            matmul_energy_mj=matmul.energy_mj,
            memory_latency_ns=memory.latency_ns,
            memory_bandwidth_gbps=memory.bandwidth_gbps,
            power_energy_mj=power.energy_mj,
            power_thermal_w=power.thermal_w,
            kv_cache_utilization=self.kv_cache.utilization(),
            kv_reads=self.kv_cache.reads,
            kv_writes=self.kv_cache.writes,
            kv_misses=self.kv_cache.misses,
        )

    def current_stats(self) -> Dict[str, float]:
        return {
            "kv_utilization": self.kv_cache.utilization(),
            "kv_reads": float(self.kv_cache.reads),
            "kv_writes": float(self.kv_cache.writes),
            "kv_misses": float(self.kv_cache.misses),
        }
