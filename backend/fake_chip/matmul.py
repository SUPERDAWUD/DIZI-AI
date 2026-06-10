from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatmulTileResult:
    latency_ms: float
    energy_mj: float
    operations: int


class FakeMatmul:
    """Simulates a tiled matrix multiply operation."""

    def __init__(self, tile_size: int = 16) -> None:
        self.tile_size = tile_size

    def simulate(self, m: int, n: int, k: int, precision: str = "int8") -> MatmulTileResult:
        tiles = ((m + self.tile_size - 1) // self.tile_size) * ((n + self.tile_size - 1) // self.tile_size)
        ops_per_tile = self.tile_size * self.tile_size * k * 2
        total_ops = ops_per_tile * tiles
        precision_factor = {"int4": 0.8, "int8": 1.0, "fp16": 1.6}.get(precision, 1.0)
        latency_ms = tiles * 0.2 * precision_factor
        energy_mj = total_ops * 1e-6 * precision_factor
        return MatmulTileResult(latency_ms=latency_ms, energy_mj=energy_mj, operations=total_ops)
