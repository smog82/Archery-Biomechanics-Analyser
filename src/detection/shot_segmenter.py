"""
src/detection/shot_segmenter.py
Automatic detection of individual shot cycles in a long video.

Strategy
--------
A shot cycle has the following elbow-angle signature:
  1. pre_draw  — elbow angle ≥ pre_draw_threshold (arm extended)
  2. draw      — angle falls toward full_draw
  3. full_draw — angle reaches local minimum ≤ full_draw_threshold
  4. release   — angle rises rapidly (≥ release_rise above minimum)

Gaps in detection (None values) are filled by linear interpolation before
scanning, then the raw frame indices are preserved in the output.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ShotSegment:
    index:           int    # 1-based shot number
    start_frame:     int    # first frame of pre-draw
    end_frame:       int    # last frame of follow-through
    peak_draw_frame: int    # frame with minimum elbow angle
    min_elbow_angle: float  # angle at full draw (degrees)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration_s(self, fps: float) -> float:
        return self.duration_frames / max(fps, 1.0)

    def __str__(self) -> str:
        return (f"Shot {self.index}: frames {self.start_frame}–{self.end_frame} "
                f"({self.duration_frames} fr), min elbow={self.min_elbow_angle:.1f}°")


def segment_shots(
    elbow_angles: List[Optional[float]],
    fps: float = 30.0,
    pre_draw_threshold: float = 150.0,
    full_draw_threshold: float = 120.0,
    release_rise: float = 25.0,
    min_shot_s: float = 1.5,
    max_shot_s: float = 20.0,
) -> List[ShotSegment]:
    """
    Detect individual shot cycles from an elbow-angle time series.

    Parameters
    ----------
    elbow_angles        : per-frame elbow angles (None = not detected)
    fps                 : frames per second of the video
    pre_draw_threshold  : minimum angle to qualify as pre-draw (arm extended)
    full_draw_threshold : maximum angle at full draw (arm fully bent)
    release_rise        : how many degrees the angle must rise after full draw
                          to count as a release event
    min_shot_s / max_shot_s : valid shot duration bounds in seconds

    Returns
    -------
    list of ShotSegment (sorted by start_frame)
    """
    filled = _interpolate(elbow_angles)
    n      = len(filled)
    min_f  = int(min_shot_s * fps)
    max_f  = int(max_shot_s * fps)
    shots  = []
    shot_n = 1
    i      = 0

    while i < n:
        # ① Find arm-extended start
        if filled[i] is None or filled[i] < pre_draw_threshold:
            i += 1
            continue
        pre_start = i

        # ② Scan until angle drops below full_draw_threshold (draw begins)
        j = i + 1
        while j < n and (filled[j] is None or filled[j] > full_draw_threshold):
            j += 1
        if j >= n:
            break

        # ③ Find local minimum (full draw = deepest bend)
        peak = j
        while (peak + 1 < n
               and filled[peak + 1] is not None
               and filled[peak + 1] <= filled[peak]):
            peak += 1

        min_angle = filled[peak] if filled[peak] is not None else 999.0
        if min_angle > full_draw_threshold:
            i = j + 1
            continue

        # ④ Find release: angle rises ≥ release_rise above minimum
        target = min(min_angle + release_rise, 145.0)
        k = peak + 1
        while k < n and (filled[k] is None or filled[k] < target):
            k += 1

        shot_len = k - pre_start
        if min_f <= shot_len <= max_f:
            shots.append(ShotSegment(
                index           = shot_n,
                start_frame     = pre_start,
                end_frame       = k,
                peak_draw_frame = peak,
                min_elbow_angle = round(min_angle, 2),
            ))
            shot_n += 1
            i = k + 1
        else:
            i = pre_start + 1

    return shots


# ── Internal ──────────────────────────────────────────────────────────────────

def _interpolate(series: List[Optional[float]]) -> List[Optional[float]]:
    """Linear interpolation over None gaps; edge-fills at boundaries."""
    result = list(series)
    n = len(result)
    for i in range(n):
        if result[i] is not None:
            continue
        left = i - 1
        while left >= 0 and result[left] is None:
            left -= 1
        right = i + 1
        while right < n and result[right] is None:
            right += 1
        if left >= 0 and right < n:
            t = (i - left) / (right - left)
            result[i] = result[left] + t * (result[right] - result[left])
        elif left >= 0:
            result[i] = result[left]
        elif right < n:
            result[i] = result[right]
    return result
