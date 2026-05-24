"""
src/pose/sagittal.py
Pose estimator for the sagittal (lateral) camera view.
Uses MediaPipe Pose with temporal smoothing.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque


class PoseEstimatorSagittal:
    """
    Detects and tracks body pose in the sagittal (lateral) view.
    Applies temporal smoothing over a configurable window.
    """

    CONNECTIONS_TO_DRAW = [
        (11, 13), (13, 15),   # Left arm
        (12, 14), (14, 16),   # Right arm
        (11, 12),             # Shoulder line
        (23, 24),             # Hip line
        (11, 23), (12, 24),   # Torso
    ]

    def __init__(self, buffer_size: int = 5):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # static_image_mode=False enables inter-frame tracking (faster + more stable)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_pose(self, frame: np.ndarray, draw_landmarks: bool = True):
        """
        Run pose detection on a single BGR frame.

        Returns
        -------
        landmarks : list[list[float]] | None
            33 landmarks as [x, y, z, visibility] in normalised coords.
        frame : np.ndarray
            Annotated BGR frame.
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, frame

        raw = [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]
        smoothed = self._smooth(raw)

        if draw_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                connections=self.CONNECTIONS_TO_DRAW,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                ),
            )

        return smoothed, frame

    def reset(self):
        """Clear the smoothing buffer (call between shots)."""
        self._buffer.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _smooth(self, landmarks: list) -> list:
        self._buffer.append(landmarks)
        arr      = np.array(self._buffer)   # (N, 33, 4)
        smoothed = np.mean(arr, axis=0)     # (33, 4)
        return smoothed.tolist()
