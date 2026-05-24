"""
src/calibration/aruco_calibrator.py
Pixel-to-mm scale estimation using an ArUco marker.

Workflow
--------
1. Print any ArUco marker from DICT_4X4_50 (IDs 0–49).
   Note its real side length in mm (e.g. a 100 mm = 10 cm square).
2. Place the marker flat on the ground in front of the archer before filming.
3. Call detect_aruco_scale(video_path, marker_real_mm) — returns px_per_mm.
4. Convert any pixel distance:  mm = px_distance / px_per_mm

Fallback
--------
If no ArUco marker is present, use manual_scale(known_px, known_mm) by
measuring a known object in the video (bow length, arrow length, etc.).
"""

import cv2
import numpy as np
from typing import Optional

# ── Build detector once at import time ───────────────────────────────────────
try:
    _dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _params = cv2.aruco.DetectorParameters()
    _det    = cv2.aruco.ArucoDetector(_dict, _params)
    _AVAILABLE = True
except AttributeError:
    _AVAILABLE = False
    try:
        _dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        _params = cv2.aruco.DetectorParameters_create()
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False


def aruco_available() -> bool:
    return _AVAILABLE


def _detect(gray: np.ndarray):
    if not _AVAILABLE:
        return [], None
    try:
        corners, ids, _ = _det.detectMarkers(gray)
    except NameError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _dict, parameters=_params)
    return corners, ids


def _corners_to_px_per_mm(corners, marker_real_mm: float) -> float:
    c = corners[0][0]   # shape (4, 2)
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides)) / marker_real_mm


# ── Public API ────────────────────────────────────────────────────────────────

def detect_aruco_scale(
    video_path: str,
    marker_real_mm: float = 100.0,
    n_frames: int = 90,
) -> Optional[float]:
    """
    Scan the first n_frames of a video for an ArUco marker.

    Returns
    -------
    px_per_mm : float | None
        Scale factor. None if no marker detected.
    """
    if not _AVAILABLE:
        return None
    cap = cv2.VideoCapture(video_path)
    samples = []
    try:
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = _detect(gray)
            if ids is not None and len(corners) > 0:
                samples.append(_corners_to_px_per_mm(corners, marker_real_mm))
    finally:
        cap.release()
    return float(np.median(samples)) if samples else None


def detect_aruco_scale_frame(
    frame: np.ndarray,
    marker_real_mm: float = 100.0,
) -> Optional[float]:
    """Detect ArUco scale from a single BGR frame."""
    if not _AVAILABLE:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect(gray)
    if ids is None or len(corners) == 0:
        return None
    return _corners_to_px_per_mm(corners, marker_real_mm)


def manual_scale(known_px: float, known_mm: float) -> float:
    """Compute px_per_mm from a user-measured reference distance."""
    if known_mm <= 0:
        raise ValueError("known_mm must be > 0")
    return known_px / known_mm


def px_to_mm(px_distance: float, px_per_mm: float) -> float:
    """Convert pixel distance to millimetres."""
    if px_per_mm <= 0:
        return px_distance
    return px_distance / px_per_mm


def annotate_frame(
    frame: np.ndarray,
    marker_real_mm: float = 100.0,
) -> tuple:
    """
    Detect markers and draw them on the frame.

    Returns
    -------
    (annotated_frame, px_per_mm | None)
    """
    if not _AVAILABLE:
        return frame, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect(gray)
    scale = None
    if ids is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        scale = _corners_to_px_per_mm(corners, marker_real_mm)
        c  = corners[0][0]
        cx = int(c[:, 0].mean())
        cy = int(c[:, 1].mean()) - 12
        cv2.putText(frame, f"{scale:.3f} px/mm  ({1/scale:.1f} mm/px)",
                    (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, scale
