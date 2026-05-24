"""
src/metrics/angles.py
Biomechanical angle metrics for archery pose analysis.
All functions receive the full 33-landmark MediaPipe list.
"""

import numpy as np
from typing import Optional


# ── Primitive ─────────────────────────────────────────────────────────────────

def _angle_3pts(p1, p2, p3) -> Optional[float]:
    """Angle at vertex p2, in degrees, formed by p1–p2–p3."""
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _angle_vec_vertical(p1, p2) -> Optional[float]:
    """Angle between vector p1→p2 and the upward vertical axis."""
    vec = np.array(p2[:2]) - np.array(p1[:2])
    vertical = np.array([0.0, -1.0])   # y-axis points down in image coords
    n = np.linalg.norm(vec)
    if n == 0:
        return None
    cos_a = np.clip(np.dot(vec, vertical) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _angle_vec_downward(p1, p2) -> Optional[float]:
    """
    Angle between vector p1→p2 and the downward vertical axis.
    Returns 0° when the vector points straight down (bow held vertical).
    In image coords y increases downward, so the downward axis is [0, +1].
    """
    vec = np.array(p2[:2]) - np.array(p1[:2])
    down = np.array([0.0, 1.0])
    n = np.linalg.norm(vec)
    if n == 0:
        return None
    cos_a = np.clip(np.dot(vec, down) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _safe(lm, *indices):
    """Return landmark tuples only if all indices are present and have visibility > 0.1."""
    pts = []
    for i in indices:
        if lm is None or len(lm) <= i:
            return None
        if lm[i][3] < 0.1:       # visibility guard
            return None
        pts.append(lm[i])
    return pts


# ── Dominant-side auto-detection ──────────────────────────────────────────────

def auto_detect_side(lm) -> str:
    """
    Determine which arm is more visible and return 'right' or 'left'.

    Uses mean visibility of the draw-arm landmarks (shoulder, elbow, wrist).
    Falls back to 'right' if landmarks are missing.
    """
    if lm is None or len(lm) < 17:
        return "right"

    # Right arm: shoulder=12, elbow=14, wrist=16
    right_vis = np.mean([lm[i][3] for i in (12, 14, 16)])
    # Left arm:  shoulder=11, elbow=13, wrist=15
    left_vis  = np.mean([lm[i][3] for i in (11, 13, 15)])

    return "right" if right_vis >= left_vis else "left"


# ── Sagittal metrics ──────────────────────────────────────────────────────────

def get_arm_trunk_angle(lm, side: str = "right") -> Optional[float]:
    """
    Angle between the upper arm and the torso (hip–shoulder–elbow).
    Indicates how high the draw arm is lifted relative to the body.
    Pass side="auto" to detect the dominant arm automatically.
    """
    if side == "auto":
        side = auto_detect_side(lm)
    if side == "right":
        pts = _safe(lm, 24, 12, 14)   # hip, shoulder, elbow
    else:
        pts = _safe(lm, 23, 11, 13)
    if pts is None:
        return None
    return _angle_3pts(pts[0], pts[1], pts[2])


def get_elbow_angle(lm, side: str = "right") -> Optional[float]:
    """
    Elbow flexion angle (shoulder–elbow–wrist).
    Tracks the bend of the draw arm at full draw.
    Pass side="auto" to detect the dominant arm automatically.
    """
    if side == "auto":
        side = auto_detect_side(lm)
    if side == "right":
        pts = _safe(lm, 12, 14, 16)
    else:
        pts = _safe(lm, 11, 13, 15)
    if pts is None:
        return None
    return _angle_3pts(pts[0], pts[1], pts[2])


def get_wrist_angle(lm, side: str = "right") -> Optional[float]:
    """
    Wrist angle (elbow–wrist–fingertip).
    Monitors wrist deviation, a key factor in arrow release accuracy.
    Pass side="auto" to detect the dominant arm automatically.
    """
    if side == "auto":
        side = auto_detect_side(lm)
    if side == "right":
        pts = _safe(lm, 14, 16, 20)
    else:
        pts = _safe(lm, 13, 15, 19)
    if pts is None:
        return None
    return _angle_3pts(pts[0], pts[1], pts[2])


def get_dfl_angle(lm, side: str = "right") -> Optional[float]:
    """
    Draw Force Line (DFL) angle: deviation of the elbow from the
    shoulder–wrist line.  Measures bow-arm alignment quality.
    Pass side="auto" to detect the dominant arm automatically.
    """
    if side == "auto":
        side = auto_detect_side(lm)
    if side == "right":
        pts = _safe(lm, 12, 14, 16)
    else:
        pts = _safe(lm, 11, 13, 15)
    if pts is None:
        return None
    shoulder, elbow, wrist = [np.array(p[:2]) for p in pts]
    v1 = elbow   - shoulder
    v2 = wrist   - shoulder
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def get_bow_cant_angle(lm, draw_side: str = "right") -> Optional[float]:
    """
    Bow cant: angle of the bow arm (shoulder → wrist) relative to the vertical.
    For a right-handed archer (draw hand = right), the bow arm is the left arm.
    Returns 0° when the bow is perfectly vertical; >15° indicates significant cant.
    The reference axis is the downward vertical (bow arm points downward naturally).
    Pass draw_side="auto" to auto-detect the draw arm.
    """
    if draw_side == "auto":
        draw_side = auto_detect_side(lm)
    # Bow arm is the opposite of the draw arm
    if draw_side == "right":
        pts = _safe(lm, 11, 15)   # left shoulder → left wrist
    else:
        pts = _safe(lm, 12, 16)   # right shoulder → right wrist
    if pts is None:
        return None
    return _angle_vec_downward(pts[0], pts[1])


def get_elbow_wrist_vertical_angle(lm, draw_side: str = "right") -> Optional[float]:
    """
    Angle between the bow-arm elbow→wrist vector and the vertical axis.
    Captures the cant of the bow forearm at release.

    The bow arm is the opposite of the draw arm:
      - draw_side="right" → bow arm = left  (elbow 13 → wrist 15)
      - draw_side="left"  → bow arm = right (elbow 14 → wrist 16)
    Pass draw_side="auto" to auto-detect the draw arm.
    """
    if draw_side == "auto":
        draw_side = auto_detect_side(lm)
    if draw_side == "right":
        pts = _safe(lm, 13, 15)    # left elbow → left wrist (bow arm)
    else:
        pts = _safe(lm, 14, 16)    # right elbow → right wrist (bow arm)
    if pts is None:
        return None
    return _angle_vec_vertical(pts[0], pts[1])


# ── Top-view metrics ──────────────────────────────────────────────────────────

def get_shoulder_tilt(lm) -> Optional[float]:
    """
    Horizontal tilt of the shoulder line from the zenith view.
    Should be close to 0° for correct shoulder alignment.
    """
    pts = _safe(lm, 11, 12)   # left, right shoulder
    if pts is None:
        return None
    vec = np.array(pts[1][:2]) - np.array(pts[0][:2])
    n   = np.linalg.norm(vec)
    if n == 0:
        return None
    cos_a = np.clip(np.dot(vec, [1.0, 0.0]) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ── Summary helpers ───────────────────────────────────────────────────────────

def compute_all_sagittal(lm, side: str = "right") -> dict:
    """Return all sagittal metrics in a single dict."""
    if side == "auto":
        side = auto_detect_side(lm)
    return {
        "Angle_Arm_Trunk":        get_arm_trunk_angle(lm, side),
        "Angle_Elbow":            get_elbow_angle(lm, side),
        "Angle_Wrist":            get_wrist_angle(lm, side),
        "DFL_Angle":              get_dfl_angle(lm, side),
        "Elbow_Wrist_Vertical":   get_elbow_wrist_vertical_angle(lm, side),
        "Bow_Cant_Angle":         get_bow_cant_angle(lm, side),
    }


def compute_all_top(lm) -> dict:
    return {
        "Shoulder_Tilt": get_shoulder_tilt(lm),
    }
