"""
src/metrics/kinematics.py
Kinematic derivative metrics: angular velocity and shoulder sway.
"""

import numpy as np
from collections import deque
from typing import List, Optional


# ── Angular velocity ──────────────────────────────────────────────────────────

def angular_velocity_series(
    angles: List[Optional[float]],
    fps: float = 30.0,
    smooth: bool = True,
) -> List[Optional[float]]:
    """
    Compute frame-to-frame angular velocity in °/s.

    Parameters
    ----------
    angles  : per-frame angle values (None = not detected)
    fps     : frames per second
    smooth  : apply 3-point centred moving average (reduces noise)

    Returns
    -------
    List of same length; first element is always None.
    """
    n = len(angles)
    raw: List[Optional[float]] = [None] * n

    for i in range(1, n):
        if angles[i] is not None and angles[i - 1] is not None:
            raw[i] = (angles[i] - angles[i - 1]) * fps

    if not smooth or n < 3:
        return raw

    out: List[Optional[float]] = [None] * n
    out[0] = raw[0]
    out[-1] = raw[-1]
    for i in range(1, n - 1):
        vals = [v for v in (raw[i - 1], raw[i], raw[i + 1]) if v is not None]
        out[i] = float(np.mean(vals)) if vals else None
    return out


class AngularVelocityTracker:
    """
    Per-frame tracker that computes angular velocity on the fly (°/s).
    Keeps the previous value to compute incremental differences.
    """

    def __init__(self, fps: float = 30.0):
        self._fps  = fps
        self._prev: Optional[float] = None

    def update(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            self._prev = None
            return None
        vel = None
        if self._prev is not None:
            vel = (angle - self._prev) * self._fps
        self._prev = angle
        return vel

    def reset(self):
        self._prev = None


# ── Shoulder sway ─────────────────────────────────────────────────────────────

class SwayTracker:
    """
    Rolling 2-D position dispersion of a landmark (normalised coords × 1000).
    A value of 1.0 means the landmark moved ±0.001 in normalised coords
    (roughly 1–2 mm at typical filming distances).
    """

    def __init__(self, window: int = 20):
        self._bx: deque = deque(maxlen=window)
        self._by: deque = deque(maxlen=window)
        self.window = window

    def update(
        self,
        x: Optional[float],
        y: Optional[float],
    ) -> Optional[float]:
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        self._bx.append(x)
        self._by.append(y)
        if len(self._bx) < 5:
            return None
        # Combined Euclidean std (normalised units × 1000 for readability)
        sway = float(np.sqrt(np.std(self._bx) ** 2 + np.std(self._by) ** 2)) * 1000
        return round(sway, 4)

    def reset(self):
        self._bx.clear()
        self._by.clear()
