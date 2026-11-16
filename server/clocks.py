"""Clock offset of an edge camera node, estimated from its heartbeats.

A node stamps every heartbeat with its own wall clock (send time); the control room notes when the
heartbeat arrived on its clock (receive time). One sample is

    receive - send  =  (control-room clock - node clock)  +  network delay

The network delay is never negative, so the smallest sample in a recent window is the one that
travelled fastest and is the best estimate of the clock offset. Taking the minimum is also the
smoothing: a slow or queued heartbeat (a larger sample) cannot move the estimate, and neither can an
old signed heartbeat that somebody replays. The window slides, so a node whose clock drifts or is
stepped by NTP is followed again after at most `window` seconds.

    node time + offset = control-room time

The estimate is high by the fastest heartbeat's travel time (milliseconds on a LAN), which is far
below what the feasibility gate can tell apart.
"""
from __future__ import annotations

import math
from collections import deque

WINDOW = 30.0      # seconds of heartbeats the minimum is taken over
MAX_SAMPLES = 256  # the node sends a heartbeat with every frame (~8 per second)
WARN_OFFSET = 2.0  # seconds of skew above which the control room logs it for the officer


class ClockOffset:
    """Sliding-window minimum of (receive time - send time) for one camera node."""

    def __init__(self, window: float = WINDOW, max_samples: int = MAX_SAMPLES):
        self.window = window
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)  # (receive time, receive - send)

    def add(self, sent: float, received: float) -> float | None:
        """Record one heartbeat and return the current estimate."""
        if not (math.isfinite(sent) and math.isfinite(received)):
            return self.value
        self._samples.append((received, received - sent))
        while self._samples and received - self._samples[0][0] > self.window:
            self._samples.popleft()
        return self.value

    @property
    def value(self) -> float | None:
        """Seconds to add to a node timestamp to get control-room time; None before the first heartbeat."""
        return min(d for _, d in self._samples) if self._samples else None

    def correct(self, node_time: float) -> float:
        """A node timestamp on the control room's clock (unchanged while the offset is unknown)."""
        off = self.value
        return node_time if off is None else node_time + off


def describe(offset: float) -> str:
    """'61.0 s ahead of' / '90.0 s behind' — the node's clock relative to the control room's."""
    return f"{abs(offset):.1f} s {'behind' if offset > 0 else 'ahead of'}"
