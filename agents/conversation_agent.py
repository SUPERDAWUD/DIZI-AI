from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent


class ConversationAgent(BaseAgent):
    """Produces natural assistant-style conversational replies."""

    @staticmethod
    def _same_text(left: str, right: str) -> bool:
        return left.strip().rstrip(".!?").lower() == right.strip().rstrip(".!?").lower()

    @staticmethod
    def _fallback_reply(prompt: str) -> str:
        lowered = prompt.lower()
        words = set(re.findall(r"[a-z']+", lowered))
        if words.intersection({"hello", "hi", "hey"}):
            return "Hey, I am here. What would you like to work on?"
        if any(word in lowered for word in ("code", "function", "bug", "error", "script")):
            return "I can help with that. Share the relevant code or describe the behavior you want, and I will turn it into a concrete fix."
        if any(word in lowered for word in ("image", "picture", "draw", "generate")):
            return "I can help shape that into an image prompt or generate a visual from the Image mode."
        if "?" in prompt:
            return "Short answer: yes, I can help reason through that. Give me any constraints that matter, and I will make the next step concrete."
        return "I can help with this. The best next step is to turn it into a clear goal, identify any constraints, and work through it piece by piece."

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(context.get("prompt") or context.get("text") or "").strip()
        self._log("start", {"prompt_length": len(prompt)})

        if not prompt:
            reply = "I'm ready when you are. Send me a prompt to work with."
            self._log("fallback", {"reply_length": len(reply)})
            return {"output": reply, "conversation_at": datetime.utcnow().isoformat()}

        reply = ""
        if self.model_manager is not None and hasattr(self.model_manager, "generate"):
            try:
                model_key = self.model_manager.get_model_for_agent("ConversationAgent")
                chat_prompt = (
                    "You are DIZI-AI, a helpful conversational assistant. "
                    "Reply naturally, clearly, and directly.\n\n"
                    f"User: {prompt}\nAssistant:"
                )
                reply = self.model_manager.generate(model_key, chat_prompt, max_tokens=320)
            except Exception as exc:
                self._log("generation_failed", {"error": str(exc)})

        reply = str(reply or "").strip()
        if not reply or self._same_text(reply, prompt):
            reply = self._fallback_reply(prompt)

        self._log("success", {"reply_length": len(reply)})
        return {"output": reply, "conversation_at": datetime.utcnow().isoformat()}
