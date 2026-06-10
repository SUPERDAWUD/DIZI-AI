from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional


class BaseAgent:
    """Base class for all DIZI-AI agents.

    An agent is a reusable unit of AI work. The orchestrator calls
    its `run()` method with a shared context and the agent returns
    a result dictionary.

    This class provides common logging behavior and a standard interface
    so every agent can be swapped or extended without changing the
    orchestrator.
    """

    def __init__(
        self,
        logger: Optional[Callable[[Dict[str, Any]], Any]] = None,
        model_manager: Optional[Any] = None,
    ) -> None:
        self.logger = logger or (lambda entry: None)
        self.model_manager = model_manager

    def _log(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.logger(
            {
                "agent": self.__class__.__name__,
                "action": action,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on the current context.

        Child classes must implement this method.
        """
        raise NotImplementedError("Agent must implement run(context)")
