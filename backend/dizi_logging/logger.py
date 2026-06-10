from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class EventLogger:
    """Collects structured events from the orchestrator and agents."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    listeners: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)

    def log(self, event: Dict[str, Any]) -> None:
        """Log an event by appending it to the current list and notify subscribers."""
        self.events.append(event)
        for listener in list(self.listeners):
            try:
                listener(event)
            except Exception:
                continue

    def subscribe(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        self.listeners.append(listener)

    def unsubscribe(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def reset(self) -> None:
        self.events.clear()

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self.events)
