"""
src/sync/synchroniser.py
Automatic synchronisation of sagittal + top-view videos.

Strategy (lightweight, no audio dependency)
--------------------------------------------
1. If the two clips are within 5 % of each other in frame count, assume the
   cameras started together → offset 0.
2. Otherwise build a per-frame motion-energy signal for each clip (mean
   absolute inter-frame difference on down-scaled greyscale frames) and find
   the lag that maximises the normalised cross-correlation between the two
   signals. This is robust to low-activity intros because it matches the
   *shape* of the activity profile, not a single energy value.

The function returns `offset` (integer):
  offset > 0 → skip `offset` frames in the TOP video
  offset < 0 → skip `-offset` frames in the SAGITTAL video
"""

import cv2
import numpy as np


def sync_videos(sag_path: str, top_path: str, search_range: int = 90) -> int:
    """
    Estimate the temporal offset between two videos.

    Parameters
    ----------
    sag_path     : path to sagittal video
    top_path     : path to top-view video
    search_range : maximum lag to search, in frames (default 90 = 3 s @ 30 fps)

    Returns
    -------
    offset : int  (positive → skip frames in top video)
    """
    cap_s = cv2.VideoCapture(sag_path)
    cap_t = cv2.VideoCapture(top_path)

    try:
        n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
        n_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))

        # If videos are within 5 % of each other in length → assume sync
        if max(n_s, n_t) > 0 and abs(n_s - n_t) / max(n_s, n_t) < 0.05:
            return 0

        # Build motion-energy signals (cap the number of frames analysed)
        max_frames = min(max(n_s, n_t), search_range * 4, 600)
        sig_s = _motion_signal(cap_s, max_frames)
        sig_t = _motion_signal(cap_t, max_frames)

        if sig_s is None or sig_t is None or len(sig_s) < 4 or len(sig_t) < 4:
            return 0

        lag = _best_lag(sig_s, sig_t, search_range)
        # lag > 0 means the sagittal signal leads → top starts later → skip top
        return int(lag)
    finally:
        cap_s.release()
        cap_t.release()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _motion_signal(cap, max_frames: int) -> np.ndarray | None:
    """Per-frame mean absolute difference on 64×64 greyscale frames."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev = None
    energy = []
    for _ in range(max_frames):
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            energy.append(float(np.abs(g - prev).mean()))
        prev = g
    if not energy:
        return None
    arr = np.asarray(energy, dtype=np.float32)
    # Zero-mean, unit-variance normalisation for cross-correlation
    std = arr.std()
    if std < 1e-6:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def _best_lag(a: np.ndarray, b: np.ndarray, search_range: int) -> int:
    """
    Return the lag (in frames) that best aligns signal `b` onto signal `a`.

    Positive lag → `a` leads `b` (b must be advanced / its video skipped).
    """
    best_score = -np.inf
    best_lag = 0
    max_lag = min(search_range, len(a) - 2, len(b) - 2)

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            seg_a = a[lag:]
            seg_b = b[: len(seg_a)]
        else:
            seg_b = b[-lag:]
            seg_a = a[: len(seg_b)]
        n = min(len(seg_a), len(seg_b))
        if n < 4:
            continue
        score = float(np.dot(seg_a[:n], seg_b[:n]) / n)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag
