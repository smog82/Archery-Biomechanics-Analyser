"""
src/pose/top.py
Pose estimator for the top / zenith (overhead) camera view.
Fuses MediaPipe Pose + Holistic for improved reliability in non-standard viewpoints.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque


class PoseEstimatorTop:
    """
    Detects upper-body landmarks from the zenith view.

    Strategy:
      1. Primary:  MediaPipe Pose (tracking mode, model_complexity=2)
      2. Fallback: MediaPipe Holistic when Pose confidence is low
      3. Last-resort: carry forward the last valid frame

    Applies exponential moving average (EMA) smoothing.
    """

    # Indices we care about: shoulders, elbows, wrists
    UPPER_BODY = [11, 12, 13, 14, 15, 16]

    CONNECTIONS_TO_DRAW = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
    ]

    def __init__(self, buffer_size: int = 5, ema_alpha: float = 0.3):
        self.mp_pose    = mp.solutions.pose
        self.mp_hol     = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        # NOTE: static_image_mode=False  ← critical fix vs original code
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.holistic = self.mp_hol.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.ema_alpha      = ema_alpha
        self._ema_state     = None        # EMA accumulator
        self._last_valid    = None        # last valid full landmark list (33 lms)
        self._buffer: deque = deque(maxlen=buffer_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_pose(self, frame: np.ndarray, draw_landmarks: bool = True):
        """
        Run fused pose detection on a single BGR frame.

        Returns
        -------
        landmarks : list[list[float]] | None
            Full 33-landmark list in normalised coords.
        frame : np.ndarray
            Annotated BGR frame.
        """
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(rgb)
        results_hol  = self.holistic.process(rgb)

        merged = self._fuse(results_pose, results_hol)

        if merged is None:
            return self._last_valid, frame

        smoothed          = self._ema(merged)
        self._last_valid  = smoothed

        if draw_landmarks and results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                connections=self.CONNECTIONS_TO_DRAW,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 255), thickness=2
                ),
            )

        return smoothed, frame

    def reset(self):
        """Clear state between sessions."""
        self._ema_state  = None
        self._last_valid = None
        self._buffer.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fuse(self, results_pose, results_hol):
        """
        Prefer Pose landmarks; fall back to Holistic per-landmark when
        Pose visibility < 0.5.
        """
        pose_lms = results_pose.pose_landmarks
        hol_lms  = results_hol.pose_landmarks

        if pose_lms is None and hol_lms is None:
            return None

        def extract(src_lms):
            if src_lms is None:
                return None
            return [[lm.x, lm.y, lm.z, lm.visibility]
                    for lm in src_lms.landmark]

        pose_arr = extract(pose_lms)
        hol_arr  = extract(hol_lms)

        if pose_arr is None:
            return hol_arr
        if hol_arr is None:
            return pose_arr

        # Fuse: per-landmark winner selection
        fused = []
        for i in range(len(pose_arr)):
            p = pose_arr[i]
            h = hol_arr[i] if i < len(hol_arr) else [0, 0, 0, 0]
            # Prefer pose if visibility ≥ 0.5, else holistic
            fused.append(p if p[3] >= 0.5 else h)

        return fused

    def _ema(self, landmarks: list) -> list:
        arr = np.array(landmarks)
        if self._ema_state is None:
            self._ema_state = arr
        else:
            self._ema_state = self.ema_alpha * arr + (1 - self.ema_alpha) * self._ema_state
        return self._ema_state.tolist()
