import time
import threading
from typing import Callable, Optional
from config.settings import config
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProgressConfig:
    update_interval: float = 5.0
    progress_chars: str = "▁▂▃▄▅▆▇█"
    spinner_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    max_width: int = 40


class ProgressTracker:
    """Visual progress tracking for long-running operations."""

    def __init__(self,
                 total: int,
                 description: str = "Processing",
                 config: Optional[ProgressConfig] = None):
        self.total = total
        self.description = description
        self.config = config or ProgressConfig()
        self._completed = 0
        self._active = False
        self._start_time = 0.0
        self._thread = None

    def start(self):
        """Begin progress tracking."""
        self._active = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_progress,
            daemon=True
        )
        self._thread.start()

    def update(self, increment: int = 1):
        """Update progress count."""
        self._completed += increment

    def stop(self):
        """Stop progress tracking."""
        self._active = False
        if self._thread:
            self._thread.join()
        self._print_final()

    def _run_progress(self):
        """Background thread displaying progress."""
        while self._active and self._completed < self.total:
            self._print_progress()
            time.sleep(self.config.update_interval)
        self._print_progress()

    def _print_progress(self):
        """Render current progress state."""
        elapsed = time.time() - self._start_time
        percent = min(100, (self._completed / self.total) * 100)

        # Animated elements
        spinner_char = self._get_spinner_char()
        progress_bar = self._get_progress_bar(percent)

        # Estimate remaining time
        remaining = self._estimate_remaining(elapsed)

        # Construct message
        message = (
            f"\r{spinner_char} {self.description}: "
            f"{progress_bar} {percent:.1f}% "
            f"({self._completed}/{self.total}) "
            f"[{self._format_time(elapsed)}<{self._format_time(remaining)}]"
        )

        print(message, end="", flush=True)

    def _get_spinner_char(self) -> str:
        """Get current spinner character."""
        elapsed_cycles = int(time.time() / 0.1)
        index = elapsed_cycles % len(self.config.spinner_chars)
        return self.config.spinner_chars[index]

    def _get_progress_bar(self, percent: float) -> str:
        """Generate visual progress bar."""
        filled_width = int((percent / 100) * self.config.max_width)
        bar = (self.config.progress_chars[-1] * filled_width).ljust(
            self.config.max_width, ' ')

        # Add partial character if needed
        partial_index = int(
            (percent % (100/self.config.max_width)) *
            (len(self.config.progress_chars)-1) /
            (100/self.config.max_width)
        )
        if filled_width < self.config.max_width and partial_index > 0:
            bar = bar[:filled_width] + \
                self.config.progress_chars[partial_index] + \
                bar[filled_width+1:]

        return f"[{bar}]"

    def _estimate_remaining(self, elapsed: float) -> float:
        """Calculate estimated remaining time."""
        if self._completed == 0:
            return 0
        return (elapsed / self._completed) * (self.total - self._completed)

    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        return f"{seconds/60:.1f}m"

    def _print_final(self):
        """Print final completed state."""
        elapsed = time.time() - self._start_time
        print(
            f"\r✓ {self.description}: Completed {self.total} items "
            f"in {self._format_time(elapsed)}"
            " " * (self.config.max_width + 20)  # Clear line
        )


def track_progress(func: Callable):
    """Decorator for automatic progress tracking."""
    def wrapper(*args, **kwargs):
        # Get total from function if available
        total = getattr(func, "__progress_total__", 100)

        tracker = ProgressTracker(
            total=total,
            description=func.__name__.replace('_', ' ').title()
        )
        tracker.start()

        try:
            # Pass tracker to function if it accepts it
            if "progress_tracker" in func.__code__.co_varnames:
                kwargs["progress_tracker"] = tracker

            result = func(*args, **kwargs)
            tracker.stop()
            return result
        except Exception as e:
            tracker.stop()
            raise e

    return wrapper
