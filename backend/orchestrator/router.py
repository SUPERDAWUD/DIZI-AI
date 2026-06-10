from __future__ import annotations

from typing import Any, Callable, Dict, List


class Router:
    """Routes tasks to the appropriate agents."""

    def __init__(self, agent_classes: List[Any]):
        self.agent_classes = {agent.__name__: agent for agent in agent_classes}

    def route(self, task_type: str, context: Dict[str, Any]) -> str:
        """Return the best agent class name for a given task type."""
        type_map = {
            "read": "ReaderAgent",
            "summarize": "SummarizerAgent",
            "check": "CheckerAgent",
            "chat": "ConversationAgent",
            "code": "CodeAgent",
            "image": "ImageGenAgent",
        }
        return type_map.get(task_type, "SummarizerAgent")

    def get_agent_class(self, agent_name: str) -> Any:
        return self.agent_classes.get(agent_name)
