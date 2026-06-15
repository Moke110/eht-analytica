"""Browser-close detection via frontend heartbeat.

The frontend pings ``/api/system/heartbeat`` every 3 s.  If no beat arrives
for 5 s the launcher watchdog shuts down the process.
"""

from __future__ import annotations

import threading
import time


class HeartbeatMonitor:
    """Thread-safe heartbeat tracker.

    Starts dead (``is_alive()`` returns False) until the first ``beat()``
    call. Callers that construct the monitor before the heartbeat source is
    ready must call ``beat()`` to activate it.
    """

    _UNSET: float = 0.0  # sentinel: monitor is dead until first beat()

    def __init__(self, timeout: float = 5.0) -> None:
        self._last_beat: float = self._UNSET
        self._timeout: float = timeout
        self._lock: threading.Lock = threading.Lock()

    def beat(self) -> None:
        """Record a heartbeat (called by the API endpoint)."""
        with self._lock:
            self._last_beat = time.time()

    def is_alive(self) -> bool:
        """Return True if a beat was received within the timeout window."""
        with self._lock:
            return (time.time() - self._last_beat) < self._timeout


heartbeat_monitor = HeartbeatMonitor(timeout=5.0)
