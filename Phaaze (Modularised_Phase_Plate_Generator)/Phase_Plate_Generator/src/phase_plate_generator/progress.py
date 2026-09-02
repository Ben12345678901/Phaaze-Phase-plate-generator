"""Terminal progress reporting for iterative phase retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time


@dataclass
class TerminalProgress:
    """Render a compact in-place progress bar without extra dependencies."""

    total: int
    updates: int = 20
    width: int = 30
    label: str = "GS iterations"
    _started_at: float = field(default=0.0, init=False)
    _last_length: int = field(default=0, init=False)

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._render(0, None, final=False)

    def update(
        self,
        completed: int,
        total: int,
        metrics: dict[str, float],
    ) -> None:
        if total != self.total:
            self.total = total
        interval = max(1, math.ceil(self.total / self.updates))
        if completed != self.total and completed % interval:
            return
        self._render(completed, metrics, final=completed >= self.total)

    def skipped(self, reason: str) -> None:
        padding = " " * self._last_length
        print(f"\r{padding}\r{self.label}: skipped ({reason})")

    def _render(
        self,
        completed: int,
        metrics: dict[str, float] | None,
        *,
        final: bool,
    ) -> None:
        fraction = min(max(completed / max(self.total, 1), 0.0), 1.0)
        filled = round(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._started_at
        metric_text = ""
        if metrics is not None:
            metric_text = (
                f"  PCC {metrics['pcc']:.4f}"
                f"  NRMSE {metrics['relative_nrmse']:.4f}"
            )
        message = (
            f"{self.label}: [{bar}] {fraction * 100:5.1f}% "
            f"({completed}/{self.total}){metric_text}  {elapsed:.1f}s"
        )
        padding = " " * max(0, self._last_length - len(message))
        print(
            f"\r{message}{padding}",
            end="\n" if final else "",
            flush=True,
        )
        self._last_length = len(message)
