"""
Page 4 — Shot Comparison
Overlay kinematic curves from multiple shots / sessions side by side.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_APP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _APP)
from src.export.profile_manager import list_archers, load_archer_profile
from src.metrics.consistency import ConsistencyIndex
import _theme

st.set_page_config(page_title="Shot Comparison", layout="wide")
_theme.inject(st)
_theme.header(st, "Shot Comparison",
              "Overlay kinematic curves across multiple shots or sessions")

archers = list_archers()

with st.sidebar:
    _theme.section(st, "Select Archer")
    archer = st.selectbox("Archer", ["— Select —"] + archers)
    st.divider()
    _theme.section(st, "Metric to Compare")
    metric = st.selectbox("Primary metric", [
        "Angle_Arm_Trunk", "Angle_Elbow", "Angle_Wrist",
        "DFL_Angle", "Elbow_Wrist_Vertical", "Shoulder_Tilt"
    ])
    normalise = st.checkbox("Normalise time axis (0→1)", value=True)
    show_mean = st.checkbox("Show mean ± std band", value=True)
    show_ci   = st.checkbox("Show Consistency Index", value=True)

if archer == "— Select —":
    st.info("Select an archer in the sidebar to begin comparison.")
    st.stop()

profile  = load_archer_profile(archer)
sessions = profile.get("sessions", [])
valid    = [s for s in sessions if s.get("data") and metric in pd.DataFrame(s["data"]).columns]

if not valid:
    st.warning(f"No sessions with metric **{metric}** found for {archer}.")
    st.stop()

st.markdown(f"### {archer} — {metric.replace('_', ' ')} across {len(valid)} session(s)")

# ── Build series ──────────────────────────────────────────────────────────────
all_series = []
labels     = []
PALETTE    = ["#C9A84C", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#FF9800", "#00BCD4", "#FF5722"]

for s in valid:
    df = pd.DataFrame(s["data"])
    series = df[metric].dropna().values
    if normalise:
        x = np.linspace(0, 1, len(series))
    else:
        x = np.arange(len(series))
    all_series.append((x, series))
    labels.append(f"{s.get('date','')} · {s.get('label','')}")

# ── Overlay chart ─────────────────────────────────────────────────────────────
fig = go.Figure()

if show_mean and len(all_series) > 1:
    # Interpolate all to common 200-point grid
    common_x = np.linspace(0, 1, 200) if normalise else None
    interp_series = []
    for x, y in all_series:
        if normalise:
            y_interp = np.interp(common_x, x, y)
        else:
            y_interp = y
        interp_series.append(y_interp)

    if normalise:
        min_len = 200
    else:
        min_len = min(len(s) for s in interp_series)
        interp_series = [s[:min_len] for s in interp_series]
        common_x = np.arange(min_len)

    arr  = np.array(interp_series)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([common_x, common_x[::-1]]),
        y=np.concatenate([mean + std, (mean - std)[::-1]]),
        fill="toself", fillcolor="rgba(201,168,76,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="± 1 std dev", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=common_x, y=mean,
        name="Mean",
        line=dict(color="#FFFFFF", width=3, dash="dash")
    ))

for i, (x, y) in enumerate(all_series):
    fig.add_trace(go.Scatter(
        x=x, y=y,
        name=labels[i],
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        opacity=0.85
    ))

fig.update_layout(
    paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
    font_color="#E8F5E9", height=450,
    xaxis_title="Normalised time" if normalise else "Frame",
    yaxis_title="Angle (°)",
    legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F")
)
fig.update_xaxes(gridcolor="#2E6B4F")
fig.update_yaxes(gridcolor="#2E6B4F")
st.plotly_chart(fig, use_container_width=True)

# ── Consistency Index ─────────────────────────────────────────────────────────
if show_ci and len(all_series) > 1:
    st.divider()
    st.markdown("### 🎯 Consistency Index")

    arrays = [y for _, y in all_series]
    ci = ConsistencyIndex.compute(arrays)

    if ci >= 80:
        badge_cls = "ci-high"; rating = "Excellent"
    elif ci >= 55:
        badge_cls = "ci-mid";  rating = "Moderate"
    else:
        badge_cls = "ci-low";  rating = "Inconsistent"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<div class="ci-badge {badge_cls}">CI = {ci:.1f} / 100 — {rating}</div>',
                    unsafe_allow_html=True)
        st.caption("CI measures how similar the movement pattern is across shots. 100 = perfectly identical.")
    with col2:
        # Per-session mean ± std
        per_session = pd.DataFrame([
            {"Session": lbl, "Mean (°)": y.mean().round(2), "Std dev (°)": y.std().round(2),
             "Min (°)": y.min().round(2), "Max (°)": y.max().round(2)}
            for lbl, (_, y) in zip(labels, all_series)
        ])
        st.dataframe(per_session, use_container_width=True, hide_index=True)

# ── Phase analysis: pre-release window ───────────────────────────────────────
st.divider()
st.markdown("### ⏱️ Pre-Release Phase Analysis")
st.caption("Zoom into the last 20% of the shot (critical release phase) for fine-grained comparison.")

fig2 = go.Figure()
for i, (x, y) in enumerate(all_series):
    start = int(len(y) * 0.8)
    seg_y = y[start:]
    seg_x = np.linspace(0.8, 1.0, len(seg_y)) if normalise else np.arange(start, start + len(seg_y))
    fig2.add_trace(go.Scatter(x=seg_x, y=seg_y, name=labels[i],
                              line=dict(color=PALETTE[i % len(PALETTE)], width=2)))

fig2.update_layout(
    paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
    font_color="#E8F5E9", height=320,
    xaxis_title="Normalised time (release phase)" if normalise else "Frame",
    yaxis_title="Angle (°)"
)
fig2.update_xaxes(gridcolor="#2E6B4F"); fig2.update_yaxes(gridcolor="#2E6B4F")
st.plotly_chart(fig2, use_container_width=True)
