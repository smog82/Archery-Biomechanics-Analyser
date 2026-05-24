"""src/metrics/stability.py — Per-session stability tracker (no global state)."""
import numpy as np
from collections import deque
from typing import Optional


class StabilityTracker:
    """
    Rolling standard deviation of any metric over the last `window` frames.
    Instantiate once per session/shot to avoid cross-session contamination.
    """

    def __init__(self, window: int = 30):
        self._buf = deque(maxlen=window)
        self.window = window

    def update(self, value: Optional[float]) -> Optional[float]:
        if value is None or np.isnan(value):
            return None
        self._buf.append(value)
        if len(self._buf) < 5:
            return None
        return float(np.std(self._buf))

    def reset(self):
        self._buf.clear()
