"""Generic task lifecycle manager with SSE progress streaming."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TaskState:
    id: str
    type: str
    status: str = "running"  # running | completed | failed | cancelled
    percent: float = 0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    progress_queue: queue.Queue = field(default_factory=queue.Queue)
    created_at: float = field(default_factory=time.time)


class TaskManager:
    """Thread-safe registry of long-running tasks with SSE-friendly progress."""

    def __init__(self):
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()

    def create(self, task_type: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        state = TaskState(id=task_id, type=task_type)
        with self._lock:
            self._tasks[task_id] = state
        return task_id

    def update(self, task_id: str, percent: float, message: str = "") -> None:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            return
        t.percent = percent
        t.message = message
        t.progress_queue.put({
            "event": "progress",
            "percent": percent,
            "message": message,
            "status": "running",
        })

    def complete(self, task_id: str, result: dict | None = None) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            return
        t.status = "completed"
        t.percent = 100
        t.result = result
        t.progress_queue.put({
            "event": "complete",
            "percent": 100,
            "message": "Done",
            "result": result,
            "status": "completed",
        })

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            return
        t.status = "failed"
        t.error = error
        t.progress_queue.put({
            "event": "error",
            "percent": t.percent,
            "message": error,
            "status": "failed",
        })

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            return False
        t.cancel_flag.set()
        t.status = "cancelled"
        t.progress_queue.put({
            "event": "cancelled",
            "percent": t.percent,
            "message": "Cancelled",
            "status": "cancelled",
        })
        return True

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            t = self._tasks.get(task_id)
        return t.cancel_flag.is_set() if t else True

    async def stream_events(self, task_id: str):
        """Async generator yielding SSE-formatted strings."""
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            yield f"data: {json.dumps({'event': 'error', 'message': 'Unknown task'})}\n\n"
            return

        q = t.progress_queue
        while True:
            try:
                data = await asyncio.to_thread(q.get, timeout=0.1)
            except queue.Empty:
                with self._lock:
                    current = self._tasks.get(task_id)
                if current and current.status in ("completed", "failed", "cancelled"):
                    # drain any remaining items
                    while not q.empty():
                        try:
                            data = q.get_nowait()
                            yield f"data: {json.dumps(data)}\n\n"
                        except queue.Empty:
                            break
                    return
                yield ""  # heartbeat — keeps connection alive
                await asyncio.sleep(0.2)
                continue

            yield f"data: {json.dumps(data)}\n\n"

            if data.get("event") in ("complete", "error", "cancelled"):
                return

    def cleanup(self, max_age_seconds: float = 300) -> None:
        """Remove completed/failed/cancelled tasks older than max_age_seconds."""
        now = time.time()
        with self._lock:
            stale = [
                tid for tid, t in self._tasks.items()
                if t.status in ("completed", "failed", "cancelled")
                and (now - t.created_at) > max_age_seconds
            ]
            for tid in stale:
                del self._tasks[tid]


# Singleton
task_manager = TaskManager()
