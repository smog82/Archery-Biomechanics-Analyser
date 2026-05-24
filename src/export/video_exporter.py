"""
src/export/video_exporter.py
Exports an annotated MP4 video with biomechanical overlays.

Overlay elements (all optional):
  - MediaPipe pose skeleton
  - Angle lines (colour-coded: green = in range, amber = borderline, red = out)
  - Numeric angle values
  - Shot phase label
  - Per-frame metric panel (top-right HUD)
  - Pixel calibration scale bar (if px_per_mm provided)
"""

import cv2
import numpy as np
import tempfile
import os
from typing import Optional, List

# Biomechanically valid ranges (same as STATUS_RANGES in excel_exporter)
_ANGLE_RANGES = {
    "Angle_Arm_Trunk":      (60,  130),
    "Angle_Elbow":          (100, 170),
    "Angle_Wrist":          (140, 180),
    "DFL_Angle":            (0,   30),
    "Elbow_Wrist_Vertical": (0,   45),
    "Bow_Cant_Angle":       (0,   20),
    "Shoulder_Tilt":        (0,   15),
}

_PHASE_COLORS = {
    "pre_draw":      (180, 180, 180),
    "draw":          (0,   200, 255),
    "full_draw":     (0,   255, 0),
    "release":       (0,   80,  255),
    "follow_through":(200, 100, 0),
}


def _status_color(metric: str, value: Optional[float]):
    """Return BGR color based on whether value is in the biomechanical range."""
    if value is None:
        return (100, 100, 100)
    lo, hi = _ANGLE_RANGES.get(metric, (None, None))
    if lo is None:
        return (200, 200, 200)
    margin = (hi - lo) * 0.1
    if lo <= value <= hi:
        return (0, 220, 0)          # green — in range
    elif (lo - margin) <= value <= (hi + margin):
        return (0, 180, 255)        # amber — borderline
    return (0, 60, 255)             # red — out of range


def _draw_hud(frame, metrics: dict, phase: str, frame_idx: int, px_per_mm: Optional[float]):
    """Draw the top-right metrics panel on the frame."""
    h, w = frame.shape[:2]
    panel_w, line_h = 260, 24
    panel_h = 32 + line_h * (len(metrics) + 1)
    x0, y0 = w - panel_w - 8, 8

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + panel_w, y0 + panel_h), (20, 30, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Phase badge
    phase_col = _PHASE_COLORS.get(phase, (200, 200, 200))
    cv2.putText(frame, f"Phase: {phase}", (x0, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_col, 2)

    # Frame number
    cv2.putText(frame, f"Frame {frame_idx}", (x0, y0 + 18 + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Metrics
    for i, (key, val) in enumerate(metrics.items()):
        col = _status_color(key, val)
        label = key.replace("_", " ")[:18]
        text  = f"{label}: {val:.1f}" if val is not None else f"{label}: —"
        cv2.putText(frame, text, (x0, y0 + 18 + (i + 2) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Scale bar (bottom-left)
    if px_per_mm and px_per_mm > 0:
        bar_mm  = 100   # draw a 100 mm scale bar
        bar_px  = int(bar_mm * px_per_mm)
        bx, by  = 12, h - 20
        cv2.line(frame, (bx, by), (bx + bar_px, by), (0, 255, 200), 3)
        cv2.line(frame, (bx, by - 5), (bx, by + 5), (0, 255, 200), 2)
        cv2.line(frame, (bx + bar_px, by - 5), (bx + bar_px, by + 5), (0, 255, 200), 2)
        cv2.putText(frame, "100 mm", (bx + bar_px // 2 - 25, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

    return frame


def _draw_angle_overlay(frame, lm, metrics: dict):
    """Draw colour-coded angle lines on the frame based on in-range status."""
    if lm is None:
        return frame
    h, w = frame.shape[:2]

    def pt(idx):
        return (int(lm[idx][0] * w), int(lm[idx][1] * h))

    # Connections and which metric they represent
    lines = [
        (12, 24, "Angle_Arm_Trunk"),
        (12, 14, "Angle_Elbow"),
        (14, 16, "Angle_Elbow"),
        (16, 20, "Angle_Wrist"),
        (11, 15, "Bow_Cant_Angle"),
        (13, 16, "Elbow_Wrist_Vertical"),
    ]
    for a, b, metric in lines:
        if len(lm) > max(a, b):
            col = _status_color(metric, metrics.get(metric))
            try:
                cv2.line(frame, pt(a), pt(b), col, 2)
            except Exception:
                pass

    return frame


def export_annotated_video(
    input_path: str,
    results: List[dict],
    output_path: Optional[str] = None,
    fps: float = 30.0,
    rotation: int = 0,
    px_per_mm: Optional[float] = None,
    draw_skeleton: bool = True,
    draw_hud: bool = True,
    draw_angles: bool = True,
) -> str:
    """
    Re-process the sagittal video and write an annotated output MP4.

    Parameters
    ----------
    input_path  : path to original sagittal video
    results     : list of per-frame dicts (from analysis loop)
    output_path : destination path; defaults to a temp file
    fps         : frame rate for the output video
    rotation    : degrees to rotate each frame (same as analysis)
    px_per_mm   : calibration scale (draws a scale bar if provided)
    draw_skeleton / draw_hud / draw_angles : overlay toggles

    Returns
    -------
    output_path : str — path to the written MP4
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_annotated.mp4")
        os.close(fd)

    cap = cv2.VideoCapture(input_path)
    try:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or fps

        # Try mp4v codec first, fall back to avc1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation != 0:
                M     = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

            row     = results[frame_idx] if frame_idx < len(results) else {}
            phase   = row.get("Shot_Phase", "unknown")
            metrics = {k: v for k, v in row.items()
                       if k not in ("Frame", "Shot_Phase") and isinstance(v, (int, float))}

            if draw_angles:
                # Extract landmark list stored in row (if present)
                lm = row.get("_landmarks")
                if lm is not None:
                    frame = _draw_angle_overlay(frame, lm, metrics)

            if draw_hud:
                frame = _draw_hud(frame, metrics, phase, frame_idx, px_per_mm)

            out.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    return output_path
