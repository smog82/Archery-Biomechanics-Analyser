"""
Archery Biomechanics Analyser — Home
Madrid Archery Team · Computer Vision & Biomechanics Platform
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _theme

st.set_page_config(
    page_title="Archery Biomechanics Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
)

_theme.inject(st)
_theme.header(
    st,
    "Archery Biomechanics Analyser",
    "Madrid Archery Team  ·  Computer Vision & Biomechanics Platform",
)

# ── Stat cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("8",           "Biomechanical Metrics"),
    ("2",           "Synchronised Camera Views"),
    ("AI",          "MediaPipe Pose Engine"),
    ("PDF + Excel", "Export Formats"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
    col.markdown(_theme.metric_card(val, lbl), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Module overview table ─────────────────────────────────────────────────────
_theme.section(st, "Platform Modules")

st.markdown("""
<table>
<thead><tr><th>Module</th><th>Description</th></tr></thead>
<tbody>
<tr><td>Upload &amp; Analyse</td><td>Load sagittal and top-view videos simultaneously and extract frame-by-frame biomechanical metrics</td></tr>
<tr><td>Live Camera</td><td>Real-time pose estimation from connected webcam or RTSP stream — requires local execution</td></tr>
<tr><td>Archer Profiles</td><td>Persistent per-archer session history, progression charts, and metadata management</td></tr>
<tr><td>Shot Comparison</td><td>Overlay normalised kinematic curves across multiple sessions with Consistency Index scoring</td></tr>
<tr><td>Radar Charts</td><td>Spider chart technique fingerprint — single archer evolution or multi-archer comparison</td></tr>
<tr><td>Coach Dashboard</td><td>Team-level aggregated analytics, consistency ranking, metric heatmap and distribution plots</td></tr>
<tr><td>Calibration</td><td>Establish pixel-to-mm scale via ArUco marker auto-detection or manual reference measurement</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

# ── Metric reference ──────────────────────────────────────────────────────────
_theme.section(st, "Biomechanical Metrics Reference")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
<table>
<thead><tr><th>Metric</th><th>Normal Range</th></tr></thead>
<tbody>
<tr><td>Arm-Trunk Angle</td><td>60 – 130 deg</td></tr>
<tr><td>Elbow Angle</td><td>100 – 170 deg</td></tr>
<tr><td>Wrist Angle</td><td>140 – 180 deg</td></tr>
<tr><td>DFL Angle</td><td>0 – 30 deg</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

with col_right:
    st.markdown("""
<table>
<thead><tr><th>Metric</th><th>Normal Range</th></tr></thead>
<tbody>
<tr><td>Elbow-Wrist Vertical</td><td>0 – 45 deg</td></tr>
<tr><td>Bow Cant Angle</td><td>0 – 20 deg</td></tr>
<tr><td>Shoulder Tilt (top)</td><td>0 – 15 deg</td></tr>
<tr><td>Stability (rolling std)</td><td>Lower is better</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

# ── Getting started ───────────────────────────────────────────────────────────
_theme.section(st, "Getting Started")

s1, s2, s3, s4 = st.columns(4)
steps = [
    ("01", "Calibration", "Set the pixel-to-mm scale using an ArUco marker or a known reference object. Optional but recommended for real-world distance measurements."),
    ("02", "Upload & Analyse", "Upload a paired sagittal and top-view video. Configure metrics, run the analysis, and download your Excel or PDF report."),
    ("03", "Save Profile", "Confirm session save to the archer's profile. Data persists between sessions for longitudinal tracking."),
    ("04", "Analytics", "Use Shot Comparison, Radar Charts, and the Coach Dashboard to track progress and benchmark against the team."),
]
for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
    col.markdown(
        f'<div class="metric-card" style="text-align:left;padding:22px 20px;">'
        f'<div style="font-size:0.65rem;font-weight:800;letter-spacing:0.18em;'
        f'color:#3A5870;text-transform:uppercase;margin-bottom:10px;">Step {num}</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#D4A843;margin-bottom:8px;">{title}</div>'
        f'<div style="font-size:0.82rem;color:#7A9AB5;line-height:1.55;">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Built with Streamlit · MediaPipe Pose (model complexity 2) · OpenCV · "
    "Plotly · openpyxl · ReportLab · Python 3.10+"
)
