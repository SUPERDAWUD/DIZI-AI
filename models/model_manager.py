from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None  # type: ignore
    _LLAMA_CPP_AVAILABLE = False


@dataclass
class LocalModelConfig:
    key: str
    name: str
    repo: str
    filename: str
    description: str
    role: str
    context_size: int = 4096


@dataclass
class CustomModelConfig:
    key: str
    name: str
    weights_path: str
    description: str
    role: str = "custom"
    device: str = "auto"


class ModelManager:
    MODEL_BASE = Path("models")
    CUSTOM_MODEL_DIR = MODEL_BASE / "custom"
    CUSTOM_MODEL_WEIGHTS_DIR = CUSTOM_MODEL_DIR / "weights"

    MODEL_REGISTRY: Dict[str, LocalModelConfig] = {
        "phi3": LocalModelConfig(
            key="phi3",
            name="Phi-3 Mini",
            repo="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            description="Best for reasoning, rewriting, and text improvement tasks.",
            role="reasoning",
            context_size=4096,
        ),
        "llama3": LocalModelConfig(
            key="llama3",
            name="Llama 3 8B Instruct",
            repo="meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
            filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            description="Best general-purpose model for summarization and instruction tasks.",
            role="general",
            context_size=4096,
        ),
        "mistral": LocalModelConfig(
            key="mistral",
            name="Mistral 7B Instruct",
            repo="TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
            filename="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
            description="Fast and stable model for latency-sensitive agents.",
            role="fast",
            context_size=4096,
        ),
        "deepseek": LocalModelConfig(
            key="deepseek",
            name="DeepSeek R1 7B",
            repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            filename="deepseek-r1-distill-qwen-7b-q4.gguf",
            description="Strong reasoning and coding model for logic and development agents.",
            role="coding",
            context_size=4096,
        ),
    }

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else self.MODEL_BASE
        self._loaded_models: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return _LLAMA_CPP_AVAILABLE

    def discover_custom_models(self) -> Dict[str, CustomModelConfig]:
        models: Dict[str, CustomModelConfig] = {}
        if not self.CUSTOM_MODEL_WEIGHTS_DIR.exists():
            return models

        for path in self.CUSTOM_MODEL_WEIGHTS_DIR.glob("*.pth"):
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

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, config in self.MODEL_REGISTRY.items():
            model_path = self._get_model_path(key)
            result[key] = {
                "key": key,
                "name": config.name,
                "repo": config.repo,
                "filename": config.filename,
                "description": config.description,
                "role": config.role,
                "path": str(model_path),
                "available": model_path.exists(),
                "type": "gguf",
            }

        for key, custom_config in self.discover_custom_models().items():
            result[key] = {
                "key": key,
                "name": custom_config.name,
                "description": custom_config.description,
                "role": custom_config.role,
                "path": custom_config.weights_path,
                "available": os.path.exists(custom_config.weights_path),
                "type": "custom",
            }

        return result

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        if model_key in self.MODEL_REGISTRY:
            config = self.MODEL_REGISTRY[model_key]
            return {
                "key": config.key,
                "name": config.name,
                "repo": config.repo,
                "filename": config.filename,
                "description": config.description,
                "role": config.role,
                "path": str(self._get_model_path(model_key)),
                "available": self._get_model_path(model_key).exists(),
                "type": "gguf",
            }

        custom_models = self.discover_custom_models()
        if model_key in custom_models:
            config = custom_models[model_key]
            return {
                "key": config.key,
                "name": config.name,
                "description": config.description,
                "role": config.role,
                "path": config.weights_path,
                "available": os.path.exists(config.weights_path),
                "type": "custom",
            }

        raise KeyError(f"Model key '{model_key}' is not registered.")

    def _get_model_path(self, model_key: str) -> Path:
        config = self.MODEL_REGISTRY.get(model_key)
        if config is None:
            raise KeyError(f"Unknown model key: {model_key}")
        return self.root_dir / config.key / config.filename

    def _load_llama_model(self, config: LocalModelConfig) -> Any:
        if not self.is_available():
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with 'pip install llama-cpp-python'."
            )

        model_path = self._get_model_path(config.key)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model file not found for '{config.key}': {model_path}"
            )

        return Llama(
            model_path=str(model_path),
            n_ctx=config.context_size,
            verbose=False,
        )

    def _load_custom_model(self, model_key: str) -> Any:
        custom_models = self.discover_custom_models()
        config = custom_models.get(model_key)
        if config is None:
            raise KeyError(f"Unknown custom model key: {model_key}")

        try:
            from .custom import load_custom_model
        except Exception as exc:
            raise RuntimeError(f"Unable to load custom model support: {exc}") from exc

        return load_custom_model(weights_path=config.weights_path, device=config.device)

    def load_model(self, model_key: str) -> Any:
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        if model_key in self.MODEL_REGISTRY:
            model = self._load_llama_model(self.MODEL_REGISTRY[model_key])
        else:
            model = self._load_custom_model(model_key)

        self._loaded_models[model_key] = model
        return model

    def generate(self, model_key: str, prompt: str, max_tokens: int = 256, stop: Optional[list] = None) -> str:
        if model_key in self.MODEL_REGISTRY:
            if not self.is_available():
                raise RuntimeError("Local GGUF model runtime is unavailable. Install llama-cpp-python.")

            model = self.load_model(model_key)
            response = model(prompt, max_tokens=max_tokens, stop=stop)
            choice = response.get("choices", [{}])[0]
            return str(choice.get("text", "")).strip()

        model = self.load_model(model_key)
        try:
            import torch

            device = next(model.parameters()).device
            prompt_vector = torch.full((1, 768), min(len(prompt), 100) / 100.0, dtype=torch.float32, device=device)
            output = model(prompt_vector)
            result = output.detach().cpu().numpy()
            return " ".join(str(x) for x in result.flatten().tolist()[:16])
        except Exception as exc:
            return f"[custom model generation error: {exc}]"

    def get_model_for_agent(self, agent_name: str) -> str:
        mapping = {
            "ReaderAgent": "llama3",
            "SummarizerAgent": "llama3",
            "CheckerAgent": "phi3",
            "ConversationAgent": "llama3",
            "CodeAgent": "deepseek",
            "LogicAgent": "deepseek",
            "FastAgent": "mistral",
        }
        return mapping.get(agent_name, "llama3")
