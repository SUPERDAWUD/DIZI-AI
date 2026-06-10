from __future__ import annotations

from collections import deque
from typing import Deque, Iterable

from .task import Task


class Scheduler:
    """Simple FIFO scheduler for Phase 1 tasks."""

    def __init__(self) -> None:
        self.queue: Deque[Task] = deque()

    def add(self, task: Task) -> None:
        self.queue.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        self.queue.extend(tasks)

    def next_task(self) -> Task | None:
        if not self.queue:
            return None
        return self.queue.popleft()

    def has_work(self) -> bool:
        return bool(self.queue)
