from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent


class FastAgent(BaseAgent):
    """Produces fast responses for low-latency tasks."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        instruction = context.get("summary") or context.get("text") or context.get("task") or ""
        self._log("start", {"instruction_length": len(instruction)})

        if not instruction:
            message = "No input available for the fast agent."
            self._log("error", {"message": message})
            return {"error": message}

        model_key = self.model_manager.get_model_for_agent(self.__class__.__name__) if self.model_manager else "mistral"
        prompt = (
            "You are a fast assistant. Provide a short, clear answer or rewrite quickly.\n\n"
            f"Input:\n{instruction}\n\nAnswer:"
        )

        if self.model_manager is not None and self.model_manager.is_available():
            try:
                result = self.model_manager.generate(model_key, prompt, max_tokens=160)
                self._log("success", {"model": model_key, "result_length": len(result)})
                return {"fast_result": result, "fast_model": model_key, "fast_at": datetime.utcnow().isoformat()}
            except Exception as exc:
                self._log("generation_failed", {"error": str(exc), "model": model_key})

        fallback_result = f"[Fallback fast] {instruction[:256]}"
        self._log("fallback", {"result_length": len(fallback_result)})
        return {"fast_result": fallback_result, "fast_model": "fallback", "fast_at": datetime.utcnow().isoformat()}
