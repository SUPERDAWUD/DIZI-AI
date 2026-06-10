from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .base_agent import BaseAgent


class ReaderAgent(BaseAgent):
    """Reads text from a file or raw prompt and stores it into shared context.

    This agent is the first step in a pipeline that turns disk data
    into something other agents can summarize, analyze, or verify.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        filepath = context.get("filepath")
        raw_text = str(context.get("text") or context.get("prompt") or "").strip()
        self._log("start", {"filepath": filepath, "raw_text_length": len(raw_text)})

        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
            except Exception as exc:
                message = f"Could not read file: {exc}"
                self._log("error", {"message": message})
                return {"error": message}

            self._log("success", {"length": len(text), "source": "file"})
            return {
                "text": text,
                "source": filepath,
                "read_at": datetime.utcnow().isoformat(),
            }

        if raw_text:
            self._log("success", {"length": len(raw_text), "source": "prompt"})
            return {
                "text": raw_text,
                "source": "prompt",
                "read_at": datetime.utcnow().isoformat(),
            }

        if filepath:
            message = f"File not found: {filepath}"
        else:
            message = "No filepath or prompt text provided to ReaderAgent."
            self._log("error", {"message": message})
        return {"error": message}
