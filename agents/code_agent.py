from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent


class CodeAgent(BaseAgent):
    """Writes, explains, or improves code from a user request."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = str(context.get("prompt") or context.get("request") or "").strip()
        code_snippet = str(context.get("extra") or context.get("code") or "").strip()
        self._log("start", {"request_length": len(request), "code_length": len(code_snippet)})

        if not request and not code_snippet:
            output = "Describe the code task or paste a snippet, and I can help write, explain, or improve it."
            self._log("fallback", {"output_length": len(output)})
            return {"output": output, "code_at": datetime.utcnow().isoformat()}

        output = ""
        if self.model_manager is not None and hasattr(self.model_manager, "generate"):
            try:
                model_key = self.model_manager.get_model_for_agent("CodeAgent")
                prompt = (
                    "You are DIZI-AI's coding agent. Write correct, practical code and include a concise explanation.\n\n"
                    f"Request:\n{request or 'Improve or explain the provided code.'}\n\n"
                    f"Code snippet:\n{code_snippet or '(none)'}\n\n"
                    "Answer with code first when code is useful, then a short explanation."
                )
                output = self.model_manager.generate(model_key, prompt, max_tokens=700)
            except Exception as exc:
                self._log("generation_failed", {"error": str(exc)})

        output = str(output or "").strip()
        if not output:
            if code_snippet:
                output = (
                    "```text\n"
                    f"{code_snippet}\n"
                    "```\n\n"
                    "Explanation: I can review or improve this snippet once a local code model is available."
                )
            else:
                output = (
                    "```python\n"
                    "def solution():\n"
                    "    pass\n"
                    "```\n\n"
                    f"Explanation: This is a placeholder for the requested task: {request}"
                )

        self._log("success", {"output_length": len(output)})
        return {"output": output, "code_at": datetime.utcnow().isoformat()}
