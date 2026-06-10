from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerResult:
    energy_mj: float
    thermal_w: float


class FakePowerModel:
    """Simulates energy and thermal estimates for a fake accelerator."""

    def estimate(self, active_units: int, runtime_ms: float, precision: str = "int8") -> PowerResult:
        base_power = 25.0
        precision_scale = {"int4": 0.7, "int8": 1.0, "fp16": 1.4}.get(precision, 1.0)
        energy_mj = base_power * active_units * (runtime_ms / 1000.0) * precision_scale
        thermal_w = base_power * active_units * precision_scale
        return PowerResult(energy_mj=energy_mj, thermal_w=thermal_w)
