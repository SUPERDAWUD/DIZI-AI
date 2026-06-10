from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CustomModelConfig:
    key: str
    name: str
    weights_path: str
    description: str
    role: str = "custom"
    device: str = "auto"


CUSTOM_MODEL_DIR = Path("models/custom")
CUSTOM_MODEL_WEIGHTS_DIR = CUSTOM_MODEL_DIR / "weights"


def discover_custom_models() -> Dict[str, CustomModelConfig]:
    models: Dict[str, CustomModelConfig] = {}
    if not CUSTOM_MODEL_WEIGHTS_DIR.exists():
        return models

    for path in CUSTOM_MODEL_WEIGHTS_DIR.glob("*.pth"):
        key = path.stem
        models[key] = CustomModelConfig(
            key=key,
            name=f"Custom model {key}",
            weights_path=str(path),
            description="Custom local model integrated into Phase 1.",
            role="custom",
            device="auto",
        )
    return models


def load_custom_model(weights_path: Optional[str] = None, device: str = "auto") -> Any:
    try:
        from .custom_llm import load_custom_llm
    except ImportError as exc:
        raise RuntimeError(f"Custom model support is unavailable: {exc}") from exc

    weights_path = weights_path or str(CUSTOM_MODEL_WEIGHTS_DIR / "custom_llm_weights.pth")
    return load_custom_llm(weights_path=weights_path, device=device)


def custom_model_info() -> Dict[str, Any]:
    return {k: v.__dict__ for k, v in discover_custom_models().items()}
