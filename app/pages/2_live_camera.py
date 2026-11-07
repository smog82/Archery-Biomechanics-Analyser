"""
Page 2 — Live Camera
Real-time biomechanical analysis via webcam or connected cameras.
Supports up to two simultaneous camera feeds with independent pose estimators.

Stop mechanism: processes ONE frame per Streamlit rerun and calls st.rerun()
to loop, so the Stop button is evaluated on every script execution.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import sys
import os
import tempfile
import plotly.graph_objects as go
from collections import deque

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_APP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _APP)
from src.pose.sagittal import PoseEstimatorSagittal
from src.metrics.angles import (
    get_arm_trunk_angle, get_elbow_angle, get_wrist_angle,
    get_dfl_angle, get_elbow_wrist_vertical_angle
)
from src.metrics.stability import StabilityTracker
import _theme

st.set_page_config(page_title="Live Camera", layout="wide")
_theme.inject(st)
_theme.header(st, "Live Camera Analysis",
              "Real-time pose estimation and biomechanical metrics from connected cameras")

# ── Sidebar ───────────────────────────────────────────────────────────────────
_SOURCE_OPTIONS = [
    "Device webcam (index 0)",
    "Camera index 1",
    "Camera index 2",
    "Camera index 3",
    "RTSP / URL stream",
]
_SOURCE_MAP = {
    "Device webcam (index 0)": 0,
    "Camera index 1": 1,
    "Camera index 2": 2,
    "Camera index 3": 3,
}

with st.sidebar:
    # ── Camera 1 ─────────────────────────────────────────────────────────────
    _theme.section(st, "Camera 1")
    cam1_source = st.selectbox("Source", _SOURCE_OPTIONS, key="cam1_source")
    cam1_rtsp = ""
    if cam1_source == "RTSP / URL stream":
        cam1_rtsp = st.text_input("Stream URL", placeholder="rtsp://192.168.1.x:port/stream",
                                   key="cam1_rtsp")
    cam1_rotation = st.slider("Rotation (°)", -180, 180, 0, 90, key="cam1_rot")
    cam1_side = st.selectbox("Draw arm side", ["Auto-detect", "Right", "Left"], key="cam1_side")

    st.divider()

    # ── Camera 2 toggle ───────────────────────────────────────────────────────
    cam2_enabled = st.checkbox("Enable second camera", value=False, key="cam2_enabled")

    if cam2_enabled:
        _theme.section(st, "Camera 2")
        cam2_source = st.selectbox("Source", _SOURCE_OPTIONS, key="cam2_source",
                                    index=1)   # default: index 1
        cam2_rtsp = ""
        if cam2_source == "RTSP / URL stream":
            cam2_rtsp = st.text_input("Stream URL", placeholder="rtsp://...",
                                       key="cam2_rtsp")
        cam2_rotation = st.slider("Rotation (°)", -180, 180, 0, 90, key="cam2_rot")
        cam2_side = st.selectbox("Draw arm side", ["Auto-detect", "Right", "Left"],
                                  key="cam2_side")
        st.divider()
    else:
        cam2_source = "Camera index 1"
        cam2_rtsp = ""
        cam2_rotation = 0
        cam2_side = "Auto-detect"

    # ── Shared metric options ─────────────────────────────────────────────────
    _theme.section(st, "Live Metrics")
    show_arm_trunk = st.checkbox("Arm–Trunk angle", value=True)
    show_elbow     = st.checkbox("Elbow angle",     value=True)
    show_wrist     = st.checkbox("Wrist angle",     value=True)
    show_dfl       = st.checkbox("DFL angle",       value=True)
    show_stability = st.checkbox("Stability",       value=True)
    st.divider()
    chart_window = st.slider("Chart history (frames)", 30, 300, 90)


# ── Helper: resolve camera index / URL ───────────────────────────────────────
def _resolve_source(source_str: str, rtsp_url: str):
    if source_str == "RTSP / URL stream":
        return rtsp_url if rtsp_url else 0
    return _SOURCE_MAP.get(source_str, 0)


def _read_one_frame(source):
    """Open camera, grab one frame, release immediately. Returns (ok, frame)."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return False, None
        ret, frame = cap.read()
    finally:
        cap.release()
    return ret, frame if ret else None


def _rotate_frame(frame, degrees: int):
    if degrees == 0:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), degrees, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def _compute_metrics(lm, show_arm_trunk, show_elbow, show_wrist,
                     show_dfl, show_stability, stability_tracker):
    metrics = {}
    if lm is None:
        return metrics
    if show_arm_trunk:
        metrics["Arm_Trunk"]  = get_arm_trunk_angle(lm)
    if show_elbow:
        metrics["Elbow"]      = get_elbow_angle(lm)
    if show_wrist:
        metrics["Wrist"]      = get_wrist_angle(lm)
    if show_dfl:
        metrics["DFL"]        = get_dfl_angle(lm)
    if show_stability:
        metrics["Stability"]  = stability_tracker.update(metrics.get("Arm_Trunk"))
    return metrics


def _metrics_html(metrics: dict) -> str:
    pills = "".join([
        f'<div class="metric-live">'
        f'<div class="val">{v:.1f}{"°" if k != "Stability" else ""}</div>'
        f'<div class="lbl">{k.replace("_", " ")}</div>'
        f'</div>'
        for k, v in metrics.items() if v is not None
    ])
    return f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{pills}</div>'


# ── Session state: camera 1 ───────────────────────────────────────────────────
_CAM1_DEFAULTS = {
    "live_running":       False,
    "live_data":          [],
    "live_frame_count":   0,
    "live_history":       None,
    "live_frames_history": None,
    "live_pose_est":      None,
    "live_stability":     None,
}
# Session state: camera 2
_CAM2_DEFAULTS = {
    "live2_data":          [],
    "live2_frame_count":   0,
    "live2_history":       None,
    "live2_frames_history": None,
    "live2_pose_est":      None,
    "live2_stability":     None,
    "live2_error":         False,
}
for k, v in {**_CAM1_DEFAULTS, **_CAM2_DEFAULTS}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Control buttons ───────────────────────────────────────────────────────────
col_start, col_stop, col_snap = st.columns([1, 1, 2])
start_btn    = col_start.button("Start", type="primary")
stop_btn     = col_stop.button("Stop")
snapshot_btn = col_snap.button("Snapshot")

st.divider()

# ── Main layout ───────────────────────────────────────────────────────────────
if cam2_enabled:
    col_c1, col_c2, col_mtr = st.columns([5, 5, 3])
    with col_c1:
        st.markdown('<div class="section-title">Camera 1 — Live Feed</div>',
                    unsafe_allow_html=True)
        frame1_ph  = st.empty()
        metric1_ph = st.empty()
    with col_c2:
        st.markdown('<div class="section-title">Camera 2 — Live Feed</div>',
                    unsafe_allow_html=True)
        frame2_ph  = st.empty()
        metric2_ph = st.empty()
    with col_mtr:
        st.markdown('<div class="section-title">Real-Time Chart</div>',
                    unsafe_allow_html=True)
        chart_ph = st.empty()
else:
    col_vid, col_metrics = st.columns([3, 2])
    with col_vid:
        st.markdown('<div class="section-title">Live Feed</div>', unsafe_allow_html=True)
        frame1_ph = st.empty()
    with col_metrics:
        st.markdown('<div class="section-title">Real-Time Metrics</div>', unsafe_allow_html=True)
        metric1_ph = st.empty()
    chart_ph   = st.empty()
    frame2_ph  = None
    metric2_ph = None

st.divider()
st.markdown('<div class="section-title">Session Data</div>', unsafe_allow_html=True)
session_ph = st.empty()


# ── Start / Stop ──────────────────────────────────────────────────────────────
if start_btn:
    st.session_state.live_running      = True
    st.session_state.live_data         = []
    st.session_state.live_frame_count  = 0
    st.session_state.live_history      = {
        k: deque(maxlen=chart_window)
        for k in ["Arm_Trunk", "Elbow", "Wrist", "DFL", "Stability"]
    }
    st.session_state.live_frames_history = deque(maxlen=chart_window)
    st.session_state.live_pose_est     = PoseEstimatorSagittal(buffer_size=5)
    st.session_state.live_stability    = StabilityTracker(window=30)

    # Camera 2
    st.session_state.live2_data         = []
    st.session_state.live2_frame_count  = 0
    st.session_state.live2_history      = {
        k: deque(maxlen=chart_window)
        for k in ["Arm_Trunk", "Elbow", "Wrist", "DFL", "Stability"]
    }
    st.session_state.live2_frames_history = deque(maxlen=chart_window)
    st.session_state.live2_pose_est     = PoseEstimatorSagittal(buffer_size=5)
    st.session_state.live2_stability    = StabilityTracker(window=30)
    st.session_state.live2_error        = False

if stop_btn:
    st.session_state.live_running = False


# ── Per-frame processing loop ─────────────────────────────────────────────────
if st.session_state.live_running:
    src1 = _resolve_source(cam1_source, cam1_rtsp)
    ok1, frame1 = _read_one_frame(src1)

    if not ok1 or frame1 is None:
        st.error(f"Cannot open Camera 1 (source: {src1}). Check connection or try a different index.")
        st.session_state.live_running = False
        st.stop()

    frame1 = _rotate_frame(frame1, cam1_rotation)

    # ── Camera 1 pose & metrics ───────────────────────────────────────────────
    pose1     = st.session_state.live_pose_est
    stab1     = st.session_state.live_stability
    hist1     = st.session_state.live_history
    fhist1    = st.session_state.live_frames_history
    cnt1      = st.session_state.live_frame_count

    lm1, frame1_ann = pose1.detect_pose(frame1.copy(), draw_landmarks=True)
    m1 = _compute_metrics(lm1, show_arm_trunk, show_elbow, show_wrist,
                          show_dfl, show_stability, stab1)

    fhist1.append(cnt1)
    for k in hist1:
        hist1[k].append(m1.get(k))
    st.session_state.live_data.append({"frame": cnt1, **m1})
    st.session_state.live_frame_count += 1

    # Display cam1
    rgb1 = cv2.cvtColor(cv2.resize(frame1_ann, (480, 270)), cv2.COLOR_BGR2RGB)
    frame1_ph.image(rgb1, width="stretch")
    metric1_ph.markdown(_metrics_html(m1), unsafe_allow_html=True)

    # ── Camera 2 (if enabled) ─────────────────────────────────────────────────
    frame2_ann = None
    if cam2_enabled and not st.session_state.live2_error:
        src2 = _resolve_source(cam2_source, cam2_rtsp)
        ok2, frame2 = _read_one_frame(src2)

        if not ok2 or frame2 is None:
            st.session_state.live2_error = True
            if frame2_ph:
                frame2_ph.error(
                    f"Cannot open Camera 2 (source: {src2}). "
                    "Check connection. Camera 1 continues running."
                )
        else:
            frame2 = _rotate_frame(frame2, cam2_rotation)

            pose2  = st.session_state.live2_pose_est
            stab2  = st.session_state.live2_stability
            hist2  = st.session_state.live2_history
            fhist2 = st.session_state.live2_frames_history
            cnt2   = st.session_state.live2_frame_count

            lm2, frame2_ann = pose2.detect_pose(frame2.copy(), draw_landmarks=True)
            m2 = _compute_metrics(lm2, show_arm_trunk, show_elbow, show_wrist,
                                  show_dfl, show_stability, stab2)

            fhist2.append(cnt2)
            for k in hist2:
                hist2[k].append(m2.get(k))
            st.session_state.live2_data.append({"frame": cnt2, **m2})
            st.session_state.live2_frame_count += 1

            # Display cam2
            rgb2 = cv2.cvtColor(cv2.resize(frame2_ann, (480, 270)), cv2.COLOR_BGR2RGB)
            if frame2_ph:
                frame2_ph.image(rgb2, width="stretch")
            if metric2_ph:
                metric2_ph.markdown(_metrics_html(m2), unsafe_allow_html=True)

    elif cam2_enabled and st.session_state.live2_error and frame2_ph:
        frame2_ph.warning("Camera 2 unavailable — Camera 1 still running.")

    # ── Snapshot ──────────────────────────────────────────────────────────────
    if snapshot_btn:
        snap1 = os.path.join(tempfile.gettempdir(),
                             f"archery_cam1_snap_{cnt1}.jpg")
        cv2.imwrite(snap1, frame1_ann)
        st.success(f"Snapshot saved: {snap1}")
        if (cam2_enabled and not st.session_state.live2_error
                and frame2_ann is not None):
            snap2 = os.path.join(tempfile.gettempdir(),
                                 f"archery_cam2_snap_{cnt1}.jpg")
            cv2.imwrite(snap2, frame2_ann)

    # ── Live chart (every 5 frames) ───────────────────────────────────────────
    if cnt1 % 5 == 0 and len(fhist1) > 5:
        COLORS = {
            "Arm_Trunk": "#C9A84C", "Elbow": "#4CAF50",
            "Wrist": "#2196F3", "DFL": "#E91E63", "Stability": "#9C27B0"
        }
        fig = go.Figure()
        for k, vals in hist1.items():
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(
                    x=list(fhist1), y=list(vals),
                    name=f"C1 {k.replace('_', ' ')}",
                    line=dict(color=COLORS.get(k, "#fff"), width=2),
                    mode="lines"
                ))
        # Overlay camera-2 arm trunk if available
        if cam2_enabled and not st.session_state.live2_error:
            h2 = st.session_state.live2_history
            fh2 = st.session_state.live2_frames_history
            if any(v is not None for v in h2.get("Arm_Trunk", [])):
                fig.add_trace(go.Scatter(
                    x=list(fh2), y=list(h2["Arm_Trunk"]),
                    name="C2 Arm–Trunk",
                    line=dict(color="#FF9800", width=2, dash="dot"),
                    mode="lines"
                ))

        fig.update_layout(
            paper_bgcolor="#080E14", plot_bgcolor="#0A1522",
            font_color="#C8D8E8", height=280, margin=dict(t=20, b=20),
            legend=dict(bgcolor="#0A1522", bordercolor="#162030", font=dict(size=10))
        )
        fig.update_xaxes(title="Frame", gridcolor="#162030")
        fig.update_yaxes(title="Angle (°)", gridcolor="#162030")
        chart_ph.plotly_chart(fig, width="stretch")

    time.sleep(0.033)   # ~30 fps throttle
    st.rerun()          # next frame — Stop button evaluated on next run

# ── Stopped state ─────────────────────────────────────────────────────────────
else:
    if st.session_state.live_data:
        st.info("Stream stopped.")
        df1 = pd.DataFrame(st.session_state.live_data)

        if cam2_enabled and st.session_state.live2_data:
            tab1, tab2 = st.tabs(["Camera 1 Data", "Camera 2 Data"])
            with tab1:
                st.dataframe(df1.tail(30), width="stretch")
                csv1 = df1.to_csv(index=False).encode()
                st.download_button("Download Camera 1 CSV", csv1,
                                   "live_cam1.csv", "text/csv")
            with tab2:
                df2 = pd.DataFrame(st.session_state.live2_data)
                st.dataframe(df2.tail(30), width="stretch")
                csv2 = df2.to_csv(index=False).encode()
                st.download_button("Download Camera 2 CSV", csv2,
                                   "live_cam2.csv", "text/csv")
        else:
            session_ph.dataframe(df1.tail(20), width="stretch")
            csv = df1.to_csv(index=False).encode()
            st.download_button("Download Session CSV", csv,
                               "live_session.csv", "text/csv")
    else:
        frame1_ph.info("Press **Start** to begin live analysis.")
        st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px;">
  <div class="metric-card" style="text-align:left;">
    <div class="lbl">Device Webcam</div>
    <div style="color:#C8D8E8;font-size:0.88rem;margin-top:10px;">Works immediately — select index 0 or 1</div>
  </div>
  <div class="metric-card" style="text-align:left;">
    <div class="lbl">Action Cam / GoPro</div>
    <div style="color:#C8D8E8;font-size:0.88rem;margin-top:10px;">Connect via USB and select index 1 or 2</div>
  </div>
  <div class="metric-card" style="text-align:left;">
    <div class="lbl">IP Camera / RTSP</div>
    <div style="color:#C8D8E8;font-size:0.88rem;margin-top:10px;">Use the RTSP / URL stream option</div>
  </div>
  <div class="metric-card" style="text-align:left;">
    <div class="lbl">Dual Camera Setup</div>
    <div style="color:#C8D8E8;font-size:0.88rem;margin-top:10px;">Enable second camera in the sidebar for sagittal + top view</div>
  </div>
</div>
""", unsafe_allow_html=True)
