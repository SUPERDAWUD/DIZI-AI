from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


@dataclass
class TaskProfile:
    task_id: str
    agent: str
    task_type: str
    duration_seconds: float
    token_count: int = 0
    cpu_usage_percent: Optional[float] = None
    memory_rss_mb: Optional[float] = None


class Profiler:
    """Tracks timing and resource usage for each orchestrator task."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.task_profiles: List[TaskProfile] = []
        self.start_time = time.time()

    def record_task(
        self,
        task_id: str,
        agent: str,
        task_type: str,
        duration_seconds: float,
        token_count: int = 0,
    ) -> None:
        cpu_usage = None
        memory_rss = None
        if psutil is not None:
            process = psutil.Process()
            try:
                cpu_usage = process.cpu_percent(interval=None)
                memory_rss = process.memory_info().rss / 1024 / 1024
            except Exception:
                cpu_usage = None
                memory_rss = None

        self.task_profiles.append(
            TaskProfile(
                task_id=task_id,
                agent=agent,
                task_type=task_type,
                duration_seconds=duration_seconds,
                token_count=token_count,
                cpu_usage_percent=cpu_usage,
                memory_rss_mb=memory_rss,
            )
        )

    def summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        count = len(self.task_profiles)
        total_tokens = sum(p.token_count for p in self.task_profiles)
        return {
            "total_time_seconds": total_time,
            "task_count": count,
            "total_tokens": total_tokens,
            "tasks": [p.__dict__ for p in self.task_profiles],
        }
