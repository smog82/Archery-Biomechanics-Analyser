"""
Page 0 — Pixel Calibration
Establishes the pixel-to-mm scale factor for real-world distance measurements.

Two methods:
  A) ArUco marker auto-detection (recommended) — print any ArUco DICT_4X4_50
     marker, note its real size in mm, place in front of archer, upload video.
  B) Manual reference — measure a known object in the video (bow length, arrow).

The resulting px_per_mm value is stored in st.session_state["px_per_mm"]
and used automatically by the Upload & Analyse page.
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, APP_DIR)
from src.calibration.aruco_calibrator import (
    detect_aruco_scale, aruco_available, annotate_frame, manual_scale,
)
import _theme

st.set_page_config(page_title="Calibration", layout="wide")
_theme.inject(st)
_theme.header(st, "Pixel Calibration",
              "Set the pixel-to-mm scale so that joint distances are reported in real-world units")

# ── Session state initialisation ──────────────────────────────────────────────
if "px_per_mm" not in st.session_state:
    st.session_state["px_per_mm"] = None

# ── Current calibration display ───────────────────────────────────────────────
col_status, col_reset = st.columns([3, 1])
with col_status:
    if st.session_state["px_per_mm"]:
        ppm = st.session_state["px_per_mm"]
        st.markdown(f"""
        <div class="scale-display">
            <div class="scale-value">{ppm:.4f}</div>
            <div class="scale-label">px / mm  &nbsp;·&nbsp; 1 mm = {1/ppm:.2f} px</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No calibration set — distances will be in normalised coordinates (0–1).")

with col_reset:
    if st.button("Clear calibration", type="secondary"):
        st.session_state["px_per_mm"] = None
        st.rerun()

st.divider()

# ── Method selection ──────────────────────────────────────────────────────────
method = st.radio(
    "Calibration method",
    ["ArUco marker (recommended)", "Manual reference distance"],
    horizontal=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A — ArUco auto-detection
# ══════════════════════════════════════════════════════════════════════════════
if "ArUco" in method:
    if not aruco_available():
        st.error("OpenCV ArUco module not available. Use manual calibration or update opencv-python.")
        st.stop()

    st.markdown("""
**How to use ArUco calibration:**
1. Print any ArUco marker (DICT_4X4_50) on paper.
2. Measure the printed marker's actual side length in mm.
3. Place the marker flat on the floor in the camera's field of view.
4. Upload the video — the first 90 frames will be scanned for the marker.
""")

    col1, col2 = st.columns([2, 1])
    with col2:
        marker_mm = st.number_input(
            "Actual marker side length (mm)", min_value=10.0, max_value=1000.0,
            value=100.0, step=5.0,
            help="Measure the printed black square (not the white border)"
        )

    with col1:
        calib_file = st.file_uploader(
            "Upload video with ArUco marker visible in first 3 seconds",
            type=["mp4", "avi", "mov", "MP4"],
            key="aruco_calib_file",
        )

    if calib_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(calib_file.read())
            tmp_path = f.name

        col_preview, col_result = st.columns([2, 1])

        with st.spinner("Scanning for ArUco marker…"):
            try:
                # Show first frame with annotation
                cap = cv2.VideoCapture(tmp_path)
                try:
                    ret, frame = cap.read()
                finally:
                    cap.release()

                if ret:
                    frame_ann, frame_scale = annotate_frame(frame.copy(), marker_mm)
                    rgb = cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB)
                    with col_preview:
                        st.image(rgb, caption="First frame — marker detection preview",
                                 width="stretch")

                # Full scan
                px_per_mm = detect_aruco_scale(tmp_path, marker_real_mm=marker_mm)

            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        with col_result:
            if px_per_mm:
                st.success("Marker detected!")
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="val">{px_per_mm:.4f}</div>'
                    f'<div class="lbl">px / mm &nbsp;·&nbsp; 1 mm = {1/px_per_mm:.2f} px</div>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown("")
                if st.button("Apply this calibration", type="primary"):
                    st.session_state["px_per_mm"] = px_per_mm
                    st.success("Calibration saved!")
                    st.rerun()
            else:
                st.error("No ArUco marker detected in the first 90 frames.")
                st.markdown("""
                **Troubleshooting:**
                - Is the marker visible and facing the camera?
                - Is the marker at least 5% of the frame width?
                - Try better lighting or a larger printout
                """)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD B — Manual reference
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
**Manual calibration:** Measure a known object in your video (e.g. arrow = 700 mm, bow riser = 380 mm).
Count its pixel length in a video frame, then enter both values below.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        known_px = st.number_input("Reference object — pixel length", min_value=1.0,
                                   value=350.0, step=1.0)
    with col2:
        known_mm = st.number_input("Reference object — real length (mm)", min_value=1.0,
                                   value=700.0, step=1.0,
                                   help="e.g. 700 for an arrow, 380 for a bow riser")
    with col3:
        st.markdown("&nbsp;")
        if st.button("Calculate & apply", type="primary"):
            ppm = manual_scale(known_px, known_mm)
            st.session_state["px_per_mm"] = ppm
            st.success(f"Scale set: {ppm:.4f} px/mm  ({1/ppm:.2f} mm/px)")
            st.rerun()

    if known_px > 0 and known_mm > 0:
        preview_ppm = known_px / known_mm
        st.info(f"Preview: {preview_ppm:.4f} px/mm — 1 mm = {1/preview_ppm:.2f} px")

# ── Common objects reference ───────────────────────────────────────────────────
st.divider()
_theme.section(st, "Common Reference Objects")
ref_data = {
    "Olympic recurve arrow": "620–710 mm",
    "Compound arrow": "550–620 mm",
    "Recurve bow riser": "~380 mm (25 in)",
    "Standard A4 paper (short side)": "210 mm",
    "Standard A4 paper (long side)": "297 mm",
    "ArUco marker (custom)": "User-defined",
}
import pandas as pd
st.table(pd.DataFrame({"Object": list(ref_data.keys()), "Typical length": list(ref_data.values())}))
