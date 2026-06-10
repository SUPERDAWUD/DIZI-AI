from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent


class LogicAgent(BaseAgent):
    """Handles reasoning and coding tasks in the Phase 1 orchestrator."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("summary") or context.get("text") or context.get("task") or ""
        self._log("start", {"query_length": len(query)})

        if not query:
            message = "No query available for logic reasoning."
            self._log("error", {"message": message})
            return {"error": message}

        model_key = self.model_manager.get_model_for_agent(self.__class__.__name__) if self.model_manager else "phi3"
        prompt = (
            "You are a reasoning assistant. Improve the input, fix bugs, and explain the reasoning clearly.\n\n"
            f"Input:\n{query}\n\nResponse:"
        )

        if self.model_manager is not None and self.model_manager.is_available():
            try:
                result = self.model_manager.generate(model_key, prompt, max_tokens=256)
                self._log("success", {"model": model_key, "result_length": len(result)})
                return {"logic_result": result, "logic_model": model_key, "logic_at": datetime.utcnow().isoformat()}
            except Exception as exc:
                self._log("generation_failed", {"error": str(exc), "model": model_key})

        fallback_result = f"[Fallback logic] {query[:512]}"
        self._log("fallback", {"result_length": len(fallback_result)})
        return {"logic_result": fallback_result, "logic_model": "fallback", "logic_at": datetime.utcnow().isoformat()}
