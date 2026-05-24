"""
src/metrics/shot_phase_detector.py
Classifies each frame into a shot-execution phase based on elbow angle dynamics,
and records the timing (in seconds) of each phase transition.

Phases (in order):
  pre_draw      — arm extended, elbow > 160°
  draw          — elbow bending, 90°–160°, decreasing
  full_draw     — elbow stable (rolling std < 3° over 10 frames)
  release       — abrupt angle change > 15° in 3 frames
  follow_through — post-release

Usage
-----
    detector = ShotPhaseDetector(fps=30.0)
    phase    = detector.update(elbow_angle_degrees)   # call each frame
    timing   = detector.phase_timing()                # dict: phase → duration_s
    detector.reset()
"""

from collections import deque
from typing import Optional, Dict
import numpy as np


class ShotPhaseDetector:

    PHASES = ("pre_draw", "draw", "full_draw", "release", "follow_through")

    def __init__(
        self,
        fps: float = 30.0,
        stability_window: int = 10,
        release_threshold: float = 15.0,
        rearm_threshold: float = 160.0,
    ):
        self._fps              = fps
        self._stability_window = stability_window
        self._release_threshold = release_threshold
        self._rearm_threshold  = rearm_threshold

        self._history: deque = deque(maxlen=stability_window)
        self._phase: str     = "pre_draw"
        self._locked: bool   = False   # True after release, until arm re-extends
        self._frames_in_follow: int = 0

        # Timing tracking
        self._frame: int          = 0
        self._phase_start: int    = 0
        self._durations: Dict[str, float] = {p: 0.0 for p in self.PHASES}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, elbow_angle: Optional[float]) -> str:
        """
        Consume one elbow-angle measurement and return the current phase.
        Call once per frame (in order).
        """
        if elbow_angle is not None:
            self._history.append(elbow_angle)

        prev_phase = self._phase

        if elbow_angle is None:
            self._frame += 1
            return self._phase

        # Once in follow_through, stay there until the arm re-extends
        # past the re-arm threshold (start of a new shot cycle).
        if self._locked:
            self._frames_in_follow += 1
            # Re-arm only after a minimum dwell in follow-through AND the
            # arm has clearly extended again (new pre-draw).
            if self._frames_in_follow >= self._stability_window and \
                    elbow_angle >= self._rearm_threshold:
                self._commit_transition("pre_draw")
                self._locked = False
                self._frames_in_follow = 0
                self._frame += 1
                return self._phase
            self._frame += 1
            return "follow_through"

        new_phase = self._classify(elbow_angle)

        if new_phase != prev_phase:
            self._commit_transition(new_phase)
            if new_phase == "follow_through":
                self._locked = True
                self._frames_in_follow = 0

        self._frame += 1
        return self._phase

    def _commit_transition(self, new_phase: str):
        """Accumulate the duration of the phase that just ended, then switch."""
        elapsed = (self._frame - self._phase_start) / max(self._fps, 1.0)
        self._durations[self._phase] += elapsed
        self._phase_start = self._frame
        self._phase = new_phase

    def phase_timing(self) -> Dict[str, float]:
        """
        Return total seconds spent in each phase so far.
        Flushes the current open phase into the accumulator.
        """
        result = dict(self._durations)
        if self._frame > self._phase_start:
            elapsed = (self._frame - self._phase_start) / max(self._fps, 1.0)
            result[self._phase] = result.get(self._phase, 0.0) + elapsed
        return {k: round(v, 3) for k, v in result.items()}

    def reset(self):
        """Reset for a new shot."""
        self._history.clear()
        self._phase       = "pre_draw"
        self._locked      = False
        self._frames_in_follow = 0
        self._frame       = 0
        self._phase_start = 0
        self._durations   = {p: 0.0 for p in self.PHASES}

    @property
    def current_phase(self) -> str:
        return self._phase

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify(self, angle: float) -> str:
        # Follow-through already locked above
        # Release: rapid change in last 3 values
        if len(self._history) >= 3:
            recent = list(self._history)[-3:]
            if (abs(recent[-1] - recent[0]) > self._release_threshold
                    and self._phase in ("full_draw", "draw")):
                return "release"
        if self._phase == "release":
            return "follow_through"

        # Full draw: stable angle below 160°
        if len(self._history) >= self._stability_window and angle < 160:
            if float(np.std(list(self._history))) < 3.0:
                return "full_draw"

        # Draw
        if 90 <= angle <= 160 and self._phase in ("pre_draw", "draw"):
            return "draw"

        # Pre-draw
        if angle > 160 and self._phase not in ("release", "follow_through", "full_draw", "draw"):
            return "pre_draw"

        return self._phase


# ── Batch helper ──────────────────────────────────────────────────────────────

def detect_phases(
    elbow_angles: list,
    fps: float = 30.0,
) -> tuple:
    """
    Apply phase detection to a list of elbow-angle values.

    Returns
    -------
    phases : list[str] — phase label per frame
    timing : dict[str, float] — seconds per phase
    """
    detector = ShotPhaseDetector(fps=fps)
    phases   = [detector.update(a) for a in elbow_angles]
    timing   = detector.phase_timing()
    return phases, timing
