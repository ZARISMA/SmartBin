import threading
from collections import deque
from datetime import datetime

from .warnings import WarningRegistry


class AppState:
    """Thread-safe shared state between the main loop and classifier worker."""

    def __init__(self):
        self._lock = threading.Lock()
        # Protected — written by classifier thread, read by main thread:
        self._label = "Ready"
        self._detail = "Press 'c' to classify. 'a' auto. 'q' quit."
        self._is_classifying = False
        self._history: deque[tuple[str, str]] = deque(maxlen=5)
        # Main-thread only (no lock needed):
        self.auto_classify = False
        self.last_capture_time = 0.0

        # ── Fleet control ────────────────────────────────────────────────
        # These fields drive the admin dashboard and edge /command handler.
        self._active_strategy = "manual"  # "manual" | "auto" | "oak-native"
        self._active_pipeline = "oak"  # "oak" | "oak-native"
        self._running = True
        self._shutdown_requested = False
        self._restart_requested = False
        self._strategy_swap_requested: str | None = None
        self._camera_count = 0
        self.warnings = WarningRegistry()

    # ── read ──────────────────────────────────────────────────────────────

    def get_display(self) -> tuple[str, str, bool]:
        with self._lock:
            return self._label, self._detail, self.auto_classify

    # ── write ─────────────────────────────────────────────────────────────

    def set_status(self, label: str, detail: str) -> None:
        with self._lock:
            self._label = label
            self._detail = detail

    def start_classify(self) -> bool:
        """Atomically check-and-set is_classifying. Returns False if already busy."""
        with self._lock:
            if self._is_classifying:
                return False
            self._is_classifying = True
            return True

    def finish_classify(self) -> None:
        with self._lock:
            self._is_classifying = False

    def add_to_history(self, label: str) -> None:
        """Prepend a completed classification label to the history ring."""
        ts = datetime.now().strftime("%H:%M")
        with self._lock:
            self._history.appendleft((ts, label))

    def get_history(self) -> list[tuple[str, str]]:
        """Return up to 5 recent (time_str, label) pairs, newest first."""
        with self._lock:
            return list(self._history)

    @property
    def is_classifying(self) -> bool:
        with self._lock:
            return bool(self._is_classifying)

    def toggle_auto(self) -> bool:
        """Toggle auto_classify and return the new value."""
        self.auto_classify = not self.auto_classify
        return bool(self.auto_classify)

    # ── Fleet / admin control ────────────────────────────────────────────

    def set_strategy(self, name: str) -> None:
        with self._lock:
            self._active_strategy = name

    def get_strategy(self) -> str:
        with self._lock:
            return self._active_strategy

    def set_pipeline(self, name: str) -> None:
        with self._lock:
            self._active_pipeline = name

    def get_pipeline(self) -> str:
        with self._lock:
            return self._active_pipeline

    def set_camera_count(self, n: int) -> None:
        with self._lock:
            self._camera_count = int(n)

    def get_camera_count(self) -> int:
        with self._lock:
            return self._camera_count

    def set_running(self, running: bool) -> None:
        with self._lock:
            self._running = bool(running)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown_requested = True
            self._running = False

    def request_restart(self) -> None:
        """Ask the supervisor (Docker / systemd) to restart this process."""
        with self._lock:
            self._restart_requested = True
            self._shutdown_requested = True

    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    @property
    def restart_requested(self) -> bool:
        with self._lock:
            return self._restart_requested

    def request_strategy_swap(self, name: str) -> None:
        with self._lock:
            self._strategy_swap_requested = name

    def take_pending_strategy_swap(self) -> str | None:
        with self._lock:
            val = self._strategy_swap_requested
            self._strategy_swap_requested = None
            return val
