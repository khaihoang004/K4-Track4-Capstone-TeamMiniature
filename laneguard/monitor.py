"""Temporal state is discarded immediately when geometry becomes unreliable."""

from dataclasses import dataclass
from enum import Enum
import math


class State(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "LANE DEPARTURE WARNING"
    UNRELIABLE = "LANE NOT RELIABLE"


@dataclass(frozen=True)
class Status:
    state: State
    offset: float | None
    pending: int = 0


class DepartureMonitor:
    def __init__(self, threshold=0.65, clear_threshold=0.50, persistence=5, alpha=0.35):
        if not (0 < clear_threshold < threshold and 0 < alpha <= 1 and persistence >= 2):
            raise ValueError("Require 0 < clear < threshold, 0 < alpha <= 1, persistence >= 2")
        self.threshold = threshold
        self.clear_threshold = clear_threshold
        self.persistence = persistence
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.smoothed = None
        self.count = 0
        self.direction = 0
        self.warning = False

    def update(self, geometry):
        if geometry is None or not math.isfinite(geometry.offset):
            self.reset()
            return Status(State.UNRELIABLE, None)
        raw = geometry.offset
        # A direct sign reversal must earn persistence in the new direction.
        direction = 1 if raw >= 0 else -1
        if self.direction and direction != self.direction:
            self.reset()
        self.direction = direction
        self.smoothed = raw if self.smoothed is None else (
            self.alpha * raw + (1 - self.alpha) * self.smoothed)
        if abs(raw) >= self.threshold and abs(self.smoothed) >= self.threshold:
            self.count += 1
            if self.count >= self.persistence:
                self.warning = True
        else:
            self.count = 0
        if abs(raw) < self.clear_threshold and abs(self.smoothed) < self.clear_threshold:
            self.warning = False
        return Status(State.WARNING if self.warning else State.NORMAL, self.smoothed, self.count)
