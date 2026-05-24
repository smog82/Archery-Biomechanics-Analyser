"""
Page 1 — Upload & Analyse
Upload sagittal + top-view videos and run full biomechanical analysis.

New in this version:
  - Bow cant angle metric
  - Angular velocity (°/s) for elbow and arm-trunk
  - Shoulder sway (micro-movement during full draw)
  - Shot phase detection + phase timing table
  - Shot segmentation (multi-shot video support)
  - Colour-coded reference overlay on frames (green/amber/red)
  - Annotated video export
  - PDF report export
  - ArUco pixel calibration integration
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import _theme

from src.pose.sagittal import PoseEstimatorSagittal
from src.pose.top import PoseEstimatorTop
from src.metrics.angles import (
    get_arm_trunk_angle, get_elbow_angle, get_wrist_angle,
    get_dfl_angle, get_elbow_wrist_vertical_angle,
    get_shoulder_tilt, get_bow_cant_angle, auto_detect_side,
)
from src.metrics.distances import calculate_distances
from src.metrics.stability import StabilityTracker
from src.metrics.kinematics import AngularVelocityTracker, SwayTracker, angular_velocity_series
from src.metrics.consistency import ConsistencyIndex
from src.metrics.shot_phase_detector import ShotPhaseDetector, detect_phases
from src.detection.shot_segmenter import segment_shots
from src.export.excel_exporter import export_to_excel
from src.export.pdf_exporter import export_to_pdf
from src.export.video_exporter import export_annotated_video
from src.sync.synchroniser import sync_videos

st.set_page_config(page_title="Upload & Analyse", layout="wide")
_theme.inject(st)
_theme.header(st, "Upload & Analyse",
              "Load sagittal + top-view videos and extract biomechanical metrics frame by frame")

# ── Pixel calibration status ──────────────────────────────────────────────────
px_per_mm: float | None = st.session_state.get("px_per_mm")
if px_per_mm:
    st.success(f"Calibration active: **{px_per_mm:.4f} px/mm** ({1/px_per_mm:.2f} mm/px) — "
               "distances reported in mm. Change in **Calibration** page.")
else:
    st.info("No pixel calibration set — distances in normalised coords. "
            "Set one in the **Calibration** page for real-world measurements.")

# ── Reference overlay ranges ──────────────────────────────────────────────────
_REF = {
    "Angle_Arm_Trunk":      (60,  130),
    "Angle_Elbow":          (100, 170),
    "Angle_Wrist":          (140, 180),
    "DFL_Angle":            (0,   30),
    "Elbow_Wrist_Vertical": (0,   45),
    "Bow_Cant_Angle":       (0,   20),
    "Shoulder_Tilt":        (0,   15),
}

PHASE_COLORS_BGR = {
    "pre_draw":       (180, 180, 180),
    "draw":           (255, 200, 0),
    "full_draw":      (0, 220, 0),
    "release":        (255, 80,  0),
    "follow_through": (0, 100, 200),
}


def _line_color(metric: str, value):
    """BGR color based on biomechanical range status."""
    if value is None:
        return (100, 100, 100)
    lo, hi = _REF.get(metric, (None, None))
    if lo is None:
        return (200, 200, 200)
    margin = (hi - lo) * 0.1
    if lo <= value <= hi:
        return (0, 220, 0)
    elif (lo - margin) <= value <= (hi + margin):
        return (0, 180, 255)
    return (0, 60, 255)


def _draw_angle_lines(frame, lm, metrics: dict):
    """Draw colour-coded biomechanical angle lines + numeric labels."""
    if lm is None:
        return frame
    try:
        h, w = frame.shape[:2]

        def pt(idx):
            return (int(lm[idx][0] * w), int(lm[idx][1] * h))

        segments = [
            (12, 24, "Angle_Arm_Trunk"),
            (12, 14, "Angle_Elbow"),
            (14, 16, "Angle_Elbow"),
            (16, 20, "Angle_Wrist"),
            (11, 15, "Bow_Cant_Angle"),
            (13, 16, "Elbow_Wrist_Vertical"),
        ]
        for a, b, metric in segments:
            if len(lm) > max(a, b):
                col = _line_color(metric, metrics.get(metric))
                try:
                    cv2.line(frame, pt(a), pt(b), col, 2)
                except Exception:
                    pass

        # Numeric angle labels at the midpoint of key segments
        label_map = [
            (12, 14, "Angle_Elbow",  "El"),
            (14, 16, "Angle_Wrist",  "Wr"),
            (12, 24, "Angle_Arm_Trunk", "AT"),
            (11, 15, "Bow_Cant_Angle", "BC"),
        ]
        for a, b, metric, short in label_map:
            val = metrics.get(metric)
            if val is None or len(lm) <= max(a, b):
                continue
            mx = (pt(a)[0] + pt(b)[0]) // 2
            my = (pt(a)[1] + pt(b)[1]) // 2
            col = _line_color(metric, val)
            cv2.putText(frame, f"{short}:{val:.0f}", (mx + 4, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    except Exception:
        pass
    return frame


def _draw_phase_overlay(frame, phase: str):
    """Draw shot phase label in top-left corner."""
    col = PHASE_COLORS_BGR.get(phase, (200, 200, 200))
    cv2.rectangle(frame, (4, 4), (190, 28), (20, 30, 20), -1)
    cv2.putText(frame, f"Phase: {phase}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return frame


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    archer_name    = st.text_input("Archer name", placeholder="e.g. Carlos López")
    session_label  = st.text_input("Session label", placeholder="e.g. Training 2025-06")
    draw_landmarks = st.toggle("Draw pose landmarks", value=True)
    draw_angles    = st.toggle("Draw angle overlay (colour-coded)", value=True)
    draw_phase_hud = st.toggle("Draw phase label", value=True)
    rotation_sagittal = st.slider("Sagittal rotation (°)", -180, 180, 90, 10)
    smoothing_window  = st.slider("Landmark smoothing window", 1, 15, 5)
    side_mode = st.selectbox("Draw arm side", ["Auto-detect", "Right", "Left"])
    side = {"Auto-detect": "auto", "Right": "right", "Left": "left"}[side_mode]

    st.divider()
    st.markdown("### 📋 Metrics")
    inc_arm_trunk  = st.checkbox("Arm–Trunk angle",       value=True)
    inc_elbow      = st.checkbox("Elbow angle",           value=True)
    inc_wrist      = st.checkbox("Wrist angle",           value=True)
    inc_dfl        = st.checkbox("DFL angle",             value=True)
    inc_vertical   = st.checkbox("Elbow–Wrist vertical",  value=True)
    inc_bow_cant   = st.checkbox("Bow cant angle",        value=True)
    inc_shoulder   = st.checkbox("Shoulder tilt (top)",   value=True)
    inc_stability  = st.checkbox("Stability (rolling σ)", value=True)
    inc_ang_vel    = st.checkbox("Angular velocity",      value=True)
    inc_sway       = st.checkbox("Shoulder sway",         value=True)
    inc_phases     = st.checkbox("Shot phases",           value=True)
    inc_distances  = st.checkbox("Joint distances",       value=True)

    st.divider()
    st.markdown("### 💾 Export options")
    export_pdf_check   = st.checkbox("PDF report",         value=True)
    export_video_check = st.checkbox("Annotated video",    value=False)
    st.caption("⚠️ Annotated video re-processes the full video — can take several minutes.")


# ── File upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📹 Sagittal view (lateral)")
    sag_file = st.file_uploader("", type=["mp4", "avi", "mov", "MP4"], key="sag")
with col2:
    st.markdown("#### 🔭 Top view (zenith / overhead)")
    top_file = st.file_uploader("", type=["mp4", "avi", "mov", "MP4"], key="top")

if sag_file and top_file:
    st.success("✅ Both videos loaded. Ready to analyse.")

    # Write temp files for video-info display
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        sag_file.seek(0); f.write(sag_file.read()); sag_path = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        top_file.seek(0); f.write(top_file.read()); top_path = f.name

    cap_sag = cv2.VideoCapture(sag_path)
    cap_top = cv2.VideoCapture(top_path)
    try:
        sag_frames = int(cap_sag.get(cv2.CAP_PROP_FRAME_COUNT))
        top_frames = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT))
        sag_fps    = cap_sag.get(cv2.CAP_PROP_FPS) or 30.0
        top_fps    = cap_top.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap_sag.release()
        cap_top.release()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sagittal frames", sag_frames)
    col2.metric("Sagittal FPS",    f"{sag_fps:.1f}")
    col3.metric("Top frames",      top_frames)
    col4.metric("Top FPS",         f"{top_fps:.1f}")

    if abs(sag_frames - top_frames) > sag_fps * 2:
        st.warning(f"⚠️ Videos differ by > 2 s ({sag_frames} vs {top_frames} frames). Auto-sync applied.")

    if st.button("🚀 Run Analysis", type="primary"):
        st.markdown("---")
        st.markdown("### ⏳ Processing…")

        preview_col, chart_col = st.columns([1, 1])
        with preview_col:
            frame_ph = st.empty()
        with chart_col:
            chart_ph = st.empty()

        progress = st.progress(0)
        status   = st.empty()

        # Estimators & trackers
        pose_sag     = PoseEstimatorSagittal(buffer_size=smoothing_window)
        pose_top     = PoseEstimatorTop(buffer_size=smoothing_window)
        stability    = StabilityTracker(window=30)
        vel_elbow    = AngularVelocityTracker(fps=sag_fps)
        vel_arm      = AngularVelocityTracker(fps=sag_fps)
        sway_tracker = SwayTracker(window=20)
        phase_det    = ShotPhaseDetector(fps=sag_fps)

        cap_sag = cv2.VideoCapture(sag_path)
        cap_top = cv2.VideoCapture(top_path)
        results_data = []
        detected_sag = 0

        try:
            offset = sync_videos(sag_path, top_path)
            if offset > 0:
                for _ in range(offset): cap_top.read()
            elif offset < 0:
                for _ in range(-offset): cap_sag.read()

            total  = min(sag_frames, top_frames)
            f_idx  = 0
            UPDATE = 5

            while True:
                ret1, frame_sag = cap_sag.read()
                ret2, frame_top = cap_top.read()
                if not ret1 or not ret2:
                    break

                if rotation_sagittal != 0:
                    h, w = frame_sag.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_sagittal, 1.0)
                    frame_sag = cv2.warpAffine(frame_sag, M, (w, h))

                lm_sag, frame_sag = pose_sag.detect_pose(frame_sag, draw_landmarks)
                lm_top, frame_top = pose_top.detect_pose(frame_top, draw_landmarks)

                row = {"Frame": f_idx}
                metrics_this_frame = {}

                if lm_sag:
                    detected_sag += 1
                    eff_side = auto_detect_side(lm_sag) if side == "auto" else side

                    if inc_arm_trunk:
                        v = get_arm_trunk_angle(lm_sag, eff_side)
                        row["Angle_Arm_Trunk"] = v
                        metrics_this_frame["Angle_Arm_Trunk"] = v
                    if inc_elbow:
                        v = get_elbow_angle(lm_sag, eff_side)
                        row["Angle_Elbow"] = v
                        metrics_this_frame["Angle_Elbow"] = v
                    if inc_wrist:
                        v = get_wrist_angle(lm_sag, eff_side)
                        row["Angle_Wrist"] = v
                        metrics_this_frame["Angle_Wrist"] = v
                    if inc_dfl:
                        v = get_dfl_angle(lm_sag, eff_side)
                        row["DFL_Angle"] = v
                        metrics_this_frame["DFL_Angle"] = v
                    if inc_vertical:
                        v = get_elbow_wrist_vertical_angle(lm_sag, eff_side)
                        row["Elbow_Wrist_Vertical"] = v
                        metrics_this_frame["Elbow_Wrist_Vertical"] = v
                    if inc_bow_cant:
                        v = get_bow_cant_angle(lm_sag, eff_side)
                        row["Bow_Cant_Angle"] = v
                        metrics_this_frame["Bow_Cant_Angle"] = v
                    if inc_stability:
                        row["Stability_Arm_Trunk"] = stability.update(metrics_this_frame.get("Angle_Arm_Trunk"))
                    if inc_ang_vel:
                        row["Vel_Elbow"]    = vel_elbow.update(metrics_this_frame.get("Angle_Elbow"))
                        row["Vel_ArmTrunk"] = vel_arm.update(metrics_this_frame.get("Angle_Arm_Trunk"))
                    if inc_sway and len(lm_sag) > 12:
                        row["Shoulder_Sway"] = sway_tracker.update(lm_sag[12][0], lm_sag[12][1])
                    if inc_phases:
                        phase = phase_det.update(metrics_this_frame.get("Angle_Elbow"))
                        row["Shot_Phase"] = phase
                    else:
                        phase = "unknown"

                    if draw_angles:
                        frame_sag = _draw_angle_lines(frame_sag, lm_sag, metrics_this_frame)
                    if draw_phase_hud and inc_phases:
                        frame_sag = _draw_phase_overlay(frame_sag, phase)

                    # Store landmarks for video export
                    row["_landmarks"] = lm_sag

                if lm_top:
                    if inc_shoulder:
                        v = get_shoulder_tilt(lm_top)
                        row["Shoulder_Tilt"] = v
                        metrics_this_frame["Shoulder_Tilt"] = v
                    if inc_distances:
                        dists = calculate_distances(lm_sag, lm_top)
                        # Convert to mm if calibrated
                        if px_per_mm:
                            dists = {
                                k: (round(v / px_per_mm * 10, 2) if v is not None else None)
                                for k, v in dists.items()
                            }
                        row.update(dists)

                results_data.append(row)

                # Live preview
                if f_idx % UPDATE == 0:
                    sag_rgb = cv2.cvtColor(cv2.resize(frame_sag, (480, 270)), cv2.COLOR_BGR2RGB)
                    top_rgb = cv2.cvtColor(cv2.resize(frame_top, (480, 270)), cv2.COLOR_BGR2RGB)
                    frame_ph.image(np.hstack([sag_rgb, top_rgb]),
                                   caption="Sagittal (colour-coded) | Top view",
                                   use_container_width=True)

                    if len(results_data) > 5:
                        df_live = pd.DataFrame(results_data)
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            subplot_titles=("Arm–Trunk & Elbow (°)", "Wrist & DFL (°)"))
                        C = {"Angle_Arm_Trunk": "#C9A84C", "Angle_Elbow": "#4CAF50",
                             "Angle_Wrist": "#2196F3", "DFL_Angle": "#E91E63",
                             "Bow_Cant_Angle": "#FF9800"}
                        for c in [x for x in ["Angle_Arm_Trunk", "Angle_Elbow"] if x in df_live]:
                            fig.add_trace(go.Scatter(x=df_live["Frame"], y=df_live[c], name=c,
                                                     line=dict(color=C.get(c, "#fff"))), row=1, col=1)
                        for c in [x for x in ["Angle_Wrist", "DFL_Angle", "Bow_Cant_Angle"] if x in df_live]:
                            fig.add_trace(go.Scatter(x=df_live["Frame"], y=df_live[c], name=c,
                                                     line=dict(color=C.get(c, "#fff"))), row=2, col=1)
                        fig.update_layout(paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                                          font_color="#E8F5E9", height=350, margin=dict(t=40, b=20))
                        fig.update_xaxes(gridcolor="#2E6B4F")
                        fig.update_yaxes(gridcolor="#2E6B4F")
                        chart_ph.plotly_chart(fig, use_container_width=True)

                progress.progress(min(f_idx / max(total, 1), 1.0))
                status.caption(f"Frame {f_idx} / {total}")
                f_idx += 1

        finally:
            cap_sag.release()
            cap_top.release()
            # Clean up temp files
            for p in (sag_path, top_path):
                try:
                    if os.path.exists(p): os.unlink(p)
                except OSError:
                    pass

        progress.progress(1.0)
        det_pct = detected_sag / max(f_idx, 1) * 100
        status.caption(f"✅ Complete — {f_idx} frames · {det_pct:.1f}% detection")

        # ── Build DataFrame ────────────────────────────────────────────────────
        df = pd.DataFrame(results_data)
        # Remove internal landmark column from the displayed/stored DataFrame
        # (the raw landmarks remain in results_data for annotated-video export)
        if "_landmarks" in df.columns:
            df = df.drop(columns=["_landmarks"])

        st.session_state["last_df"]      = df
        st.session_state["last_archer"]  = archer_name
        st.session_state["last_session"] = session_label
        st.session_state["last_fps"]     = sag_fps
        # Persist raw sagittal bytes + per-frame results so the annotated-video
        # export still works on later reruns (UploadedFile is not reusable).
        if export_video_check:
            sag_file.seek(0)
            st.session_state["last_sag_bytes"]    = sag_file.read()
            st.session_state["last_results_data"] = results_data
            st.session_state["last_rotation"]     = rotation_sagittal
            st.session_state["last_px_per_mm"]    = px_per_mm

        # Post-process angular velocity on full series (smoother)
        if inc_ang_vel and "Angle_Elbow" in df.columns:
            df["Vel_Elbow"] = angular_velocity_series(df["Angle_Elbow"].tolist(), sag_fps)
        if inc_ang_vel and "Angle_Arm_Trunk" in df.columns:
            df["Vel_ArmTrunk"] = angular_velocity_series(df["Angle_Arm_Trunk"].tolist(), sag_fps)

        # Phase timing summary
        phase_timing = phase_det.phase_timing() if inc_phases else {}

        # Shot segmentation
        shots = []
        if inc_phases and "Angle_Elbow" in df.columns:
            shots = segment_shots(df["Angle_Elbow"].tolist(), fps=sag_fps)

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Results Summary")

        angle_cols = [c for c in df.columns
                      if any(k in c for k in ("Angle", "DFL", "Tilt", "Cant", "Vertical"))
                      and df[c].dropna().count() > 0]
        if angle_cols:
            s_cols = st.columns(min(len(angle_cols), 4))
            for i, col in enumerate(angle_cols):
                mean_v = df[col].dropna().mean()
                std_v  = df[col].dropna().std()
                lo, hi = _REF.get(col, (None, None))
                delta_str = f"± {std_v:.1f}°"
                s_cols[i % 4].metric(col.replace("_", " "), f"{mean_v:.1f}°", delta_str)

        # Phase timing table
        if phase_timing and any(v > 0 for v in phase_timing.values()):
            st.markdown("#### ⏱️ Shot Phase Timing")
            PHASE_COLORS_HEX = {
                "pre_draw": "#888", "draw": "#FFC107", "full_draw": "#4CAF50",
                "release": "#E91E63", "follow_through": "#2196F3",
            }
            ph_cols = st.columns(len([v for v in phase_timing.values() if v > 0]))
            ci = 0
            for phase, secs in phase_timing.items():
                if secs <= 0:
                    continue
                col_hex = PHASE_COLORS_HEX.get(phase, "#888")
                ph_cols[ci].markdown(
                    f'<div style="background:#1B3A2D;border-left:4px solid {col_hex};'
                    f'border-radius:6px;padding:10px;text-align:center;">'
                    f'<div style="color:{col_hex};font-weight:bold;font-size:1.3rem;">{secs:.2f}s</div>'
                    f'<div style="color:#A8D5B5;font-size:0.75rem;">{phase.replace("_"," ")}</div></div>',
                    unsafe_allow_html=True)
                ci += 1

        # Shot segmentation results
        if shots:
            st.markdown(f"#### 🎯 Shot Segmentation — {len(shots)} shot(s) detected")
            for s in shots:
                st.markdown(
                    f'<div class="shot-card">'
                    f'<b>Shot {s.index}</b>  ·  frames {s.start_frame}–{s.end_frame}'
                    f'  ·  duration {s.duration_s(sag_fps):.2f}s'
                    f'  ·  min elbow {s.min_elbow_angle:.1f}°</div>',
                    unsafe_allow_html=True)

        # Full time-series chart
        st.markdown("#### 📈 Full Kinematic Time Series")
        COLORS_ALL = ["#C9A84C", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#FF9800", "#00BCD4"]
        fig2 = go.Figure()
        for i, col in enumerate(angle_cols):
            if col in df:
                fig2.add_trace(go.Scatter(x=df["Frame"], y=df[col],
                                          name=col.replace("_", " "),
                                          line=dict(color=COLORS_ALL[i % len(COLORS_ALL)], width=2)))
        # Phase colour band background
        if inc_phases and "Shot_Phase" in df.columns:
            PHASE_HEX = {"pre_draw": "rgba(128,128,128,0.1)", "draw": "rgba(255,193,7,0.1)",
                         "full_draw": "rgba(76,175,80,0.15)", "release": "rgba(233,30,99,0.15)",
                         "follow_through": "rgba(33,150,243,0.1)"}
            prev_phase, prev_f = None, 0
            for fi, ph in enumerate(df["Shot_Phase"]):
                if ph != prev_phase:
                    if prev_phase is not None:
                        fig2.add_vrect(x0=prev_f, x1=fi, fillcolor=PHASE_HEX.get(prev_phase, "rgba(0,0,0,0)"),
                                       line_width=0, annotation_text=prev_phase if fi-prev_f>20 else "",
                                       annotation_font_size=8, annotation_font_color="#A8D5B5")
                    prev_phase, prev_f = ph, fi
        fig2.update_layout(paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                           font_color="#E8F5E9", height=420,
                           legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F"))
        fig2.update_xaxes(title="Frame", gridcolor="#2E6B4F")
        fig2.update_yaxes(title="Angle (°)", gridcolor="#2E6B4F")
        st.plotly_chart(fig2, use_container_width=True)

        # Angular velocity chart
        vel_cols = [c for c in df.columns if "Vel_" in c and df[c].dropna().count() > 3]
        if vel_cols:
            st.markdown("#### ⚡ Angular Velocity (°/s)")
            fig_vel = go.Figure()
            for i, col in enumerate(vel_cols):
                fig_vel.add_trace(go.Scatter(x=df["Frame"], y=df[col],
                                             name=col.replace("Vel_", "").replace("_", " "),
                                             line=dict(color=COLORS_ALL[i], width=1.5)))
            fig_vel.add_hline(y=0, line_dash="dash", line_color="#666")
            fig_vel.update_layout(paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                                  font_color="#E8F5E9", height=280,
                                  legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F"))
            fig_vel.update_xaxes(title="Frame", gridcolor="#2E6B4F")
            fig_vel.update_yaxes(title="°/s", gridcolor="#2E6B4F")
            st.plotly_chart(fig_vel, use_container_width=True)

        # Distributions
        if angle_cols:
            st.markdown("#### 📉 Angle Distributions")
            fig3 = make_subplots(rows=1, cols=len(angle_cols),
                                 subplot_titles=[c.replace("_", " ") for c in angle_cols])
            for i, col in enumerate(angle_cols):
                fig3.add_trace(go.Histogram(x=df[col].dropna(), marker_color=COLORS_ALL[i % len(COLORS_ALL)],
                                            opacity=0.8, name=col), row=1, col=i + 1)
                lo, hi = _REF.get(col, (None, None))
                if lo is not None:
                    fig3.add_vline(x=lo, line_dash="dash", line_color="#4CAF50", row=1, col=i+1)
                    fig3.add_vline(x=hi, line_dash="dash", line_color="#4CAF50", row=1, col=i+1)
            fig3.update_layout(paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                               font_color="#E8F5E9", height=300, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("🔢 Raw data table"):
            display_df = df.drop(columns=["Shot_Phase", "_landmarks"], errors="ignore")
            st.dataframe(display_df.style.background_gradient(cmap="Greens"), use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Export")

        df_excel = df.drop(columns=["Shot_Phase", "_landmarks"], errors="ignore")
        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            excel_bytes = export_to_excel(df_excel, archer_name, session_label)
            st.download_button("📥 Download Excel",
                               excel_bytes,
                               file_name=f"archery_{archer_name or 'session'}_{session_label or 'report'}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col_ex2:
            csv_data = df_excel.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv_data,
                               file_name="archery_data.csv", mime="text/csv")

        with col_ex3:
            if export_pdf_check:
                with st.spinner("Generating PDF…"):
                    pdf_bytes = export_to_pdf(
                        df_excel,
                        archer_name    = archer_name,
                        session_label  = session_label,
                        phase_timing   = phase_timing,
                        shot_count     = len(shots),
                        px_per_mm      = px_per_mm,
                        detection_rate_pct = det_pct,
                    )
                st.download_button("📄 Download PDF Report",
                                   pdf_bytes,
                                   file_name=f"archery_{archer_name or 'report'}.pdf",
                                   mime="application/pdf")

        # Annotated video
        if export_video_check and "last_sag_bytes" in st.session_state:
            st.markdown("#### 🎬 Annotated Video")
            if st.button("Generate annotated video", type="secondary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(st.session_state["last_sag_bytes"])
                    src_path = f.name
                out_path = src_path.replace(".mp4", "_annotated.mp4")
                with st.spinner("Rendering annotated video…"):
                    export_annotated_video(
                        src_path,
                        st.session_state["last_results_data"],
                        output_path=out_path,
                        rotation=st.session_state.get("last_rotation", 0),
                        px_per_mm=st.session_state.get("last_px_per_mm"),
                    )
                try:
                    os.unlink(src_path)
                except OSError:
                    pass
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button("📥 Download annotated MP4",
                                           f.read(), file_name="archery_annotated.mp4",
                                           mime="video/mp4")
                    try:
                        os.unlink(out_path)
                    except OSError:
                        pass
                else:
                    st.error("Annotated video could not be generated (codec unavailable).")

        # Save profile
        if archer_name:
            from src.export.profile_manager import save_session
            label_to_save = session_label or datetime.now().strftime("session_%Y-%m-%d_%H%M")
            save_session(archer_name, label_to_save, df_excel)
            if not session_label:
                st.info(f"No session label provided — saved as **{label_to_save}**.")
            st.success(f"✅ Session saved to profile: **{archer_name}**")

else:
    st.info("⬆️ Upload both videos above to begin analysis.")
