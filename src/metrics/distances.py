"""src/metrics/distances.py"""
import numpy as np
from typing import Optional


def calculate_distances(lm_sagittal, lm_top) -> dict:
    """Calculate key joint distances from sagittal and top-view landmarks."""
    out = {}

    def dist(lm, i, j):
        if lm is None or len(lm) <= max(i, j):
            return None
        a = np.array(lm[i][:2])
        b = np.array(lm[j][:2])
        return float(np.linalg.norm(a - b))

    # Shoulder width (top view preferred)
    if lm_top is not None and len(lm_top) > 12:
        out["Distance_Shoulders"] = dist(lm_top, 11, 12)
    elif lm_sagittal is not None:
        out["Distance_Shoulders"] = dist(lm_sagittal, 11, 12)
    else:
        out["Distance_Shoulders"] = None

    # Wrist separation (top view)
    out["Distance_Wrists"] = dist(lm_top, 15, 16) if lm_top is not None else None

    # Elbow height relative to shoulder (sagittal)
    if lm_sagittal and len(lm_sagittal) > 14:
        sh = np.array(lm_sagittal[12][:2])
        el = np.array(lm_sagittal[14][:2])
        out["Elbow_Height_Rel"] = float(sh[1] - el[1])   # positive = elbow above shoulder
    else:
        out["Elbow_Height_Rel"] = None

    return out
