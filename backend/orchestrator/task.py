from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Task:
    """A task represents a unit of work in the DIZI-AI orchestrator."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
