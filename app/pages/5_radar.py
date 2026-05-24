"""
Page 5 — Radar Charts
Spider / radar chart visualisation of technique profile per archer and session.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_APP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _APP)
from src.export.profile_manager import list_archers, load_archer_profile
import _theme

st.set_page_config(page_title="Radar Charts", layout="wide")
_theme.inject(st)
_theme.header(st, "Radar Charts — Technique Profile",
              "Spider charts give an instant visual fingerprint of each archer's biomechanical signature")

archers = list_archers()

with st.sidebar:
    _theme.section(st, "Settings")
    mode = st.radio("Mode", ["Single archer — sessions over time", "Multi-archer comparison"])
    if mode == "Single archer — sessions over time":
        archer = st.selectbox("Archer", ["— Select —"] + archers)
    else:
        selected_archers = st.multiselect("Select archers (max 5)", archers, max_selections=5)
    st.divider()
    _theme.section(st, "Radar Dimensions")
    st.caption("Each axis is normalised to a reference range for visual balance.")

# ── Reference ranges for normalisation (biomechanically grounded) ─────────────
REFS = {
    "Arm–Trunk Angle":      (60, 130),   # typical range for archery draw
    "Elbow Angle":          (100, 170),
    "Wrist Angle":          (140, 180),
    "DFL Angle":            (0, 30),
    "Elbow–Wrist Vertical": (0, 45),
    "Shoulder Tilt":        (0, 15),
}
METRIC_MAP = {
    "Arm–Trunk Angle":       "Angle_Arm_Trunk",
    "Elbow Angle":           "Angle_Elbow",
    "Wrist Angle":           "Angle_Wrist",
    "DFL Angle":             "DFL_Angle",
    "Elbow–Wrist Vertical":  "Elbow_Wrist_Vertical",
    "Shoulder Tilt":         "Shoulder_Tilt",
}

def normalise(value, lo, hi):
    """Scale value to 0-100 within reference range."""
    if pd.isna(value):   # handles None, np.nan, pd.NA, pd.NaT
        return 50  # neutral
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #RRGGBB hex string to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(201,168,76,{alpha})"
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_radar_trace(name, means, color, fill=True):
    dims   = list(REFS.keys())
    vals   = [normalise(means.get(METRIC_MAP[d], np.nan), *REFS[d]) for d in dims]
    vals  += [vals[0]]  # close polygon
    dims  += [dims[0]]
    return go.Scatterpolar(
        r=vals, theta=dims, name=name,
        fill="toself" if fill else "none",
        line=dict(color=color, width=2),
        fillcolor=_hex_to_rgba(color, 0.15) if fill else None,
        opacity=0.85
    )

PALETTE = ["#C9A84C", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0"]

# ── Mode A: single archer ─────────────────────────────────────────────────────
if mode == "Single archer — sessions over time":
    if archer == "— Select —":
        st.info("Select an archer in the sidebar.")
        st.stop()

    profile  = load_archer_profile(archer)
    sessions = [s for s in profile.get("sessions", []) if s.get("data")]

    if not sessions:
        st.warning("No session data found for this archer.")
        st.stop()

    st.markdown(f"### 🏹 {archer} — Technique Evolution")

    # All-sessions radar
    fig = go.Figure()
    for i, s in enumerate(sessions[-5:]):  # last 5 sessions
        df = pd.DataFrame(s["data"])
        means = {c: df[c].mean() for c in df.columns if not df[c].dropna().empty}
        lbl = f"{s.get('date','')} {s.get('label','')}"
        fig.add_trace(build_radar_trace(lbl, means, PALETTE[i % len(PALETTE)]))

    fig.update_layout(
        polar=dict(
            bgcolor="#1B3A2D",
            radialaxis=dict(visible=True, range=[0, 100], color="#A8D5B5",
                            tickfont=dict(color="#A8D5B5"), gridcolor="#2E6B4F"),
            angularaxis=dict(color="#E8F5E9", gridcolor="#2E6B4F")
        ),
        paper_bgcolor="#0D1F16",
        font_color="#E8F5E9",
        height=520,
        legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Each axis is normalised 0–100 within the biomechanically expected range. "
               "Outer = higher value, not necessarily better — interpretation depends on the metric.")

    # Per-metric trend bars
    if len(sessions) > 1:
        st.divider()
        st.markdown("### 📊 Metric evolution per session")
        prog_data = []
        for s in sessions:
            df = pd.DataFrame(s["data"])
            row = {"Session": f"{s.get('date','')} {s.get('label','')}"}
            for label, col in METRIC_MAP.items():
                if col in df.columns:
                    row[label] = round(df[col].mean(), 2)
            prog_data.append(row)
        prog_df = pd.DataFrame(prog_data)

        dims = [d for d in REFS if d in prog_df.columns]
        n    = len(dims)
        if n:
            fig2 = go.Figure()
            for i, dim in enumerate(dims):
                fig2.add_trace(go.Scatter(
                    x=prog_df["Session"], y=prog_df[dim],
                    name=dim, mode="lines+markers",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2)
                ))
            fig2.update_layout(
                paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                font_color="#E8F5E9", height=350,
                legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F")
            )
            fig2.update_xaxes(gridcolor="#2E6B4F"); fig2.update_yaxes(gridcolor="#2E6B4F")
            st.plotly_chart(fig2, use_container_width=True)

# ── Mode B: multi-archer comparison ──────────────────────────────────────────
else:
    if not selected_archers:
        st.info("Select 2–5 archers in the sidebar.")
        st.stop()

    st.markdown(f"### 🏹 Multi-Archer Technique Comparison")

    fig = go.Figure()
    for i, name in enumerate(selected_archers):
        profile  = load_archer_profile(name)
        sessions = [s for s in profile.get("sessions", []) if s.get("data")]
        if not sessions:
            continue
        # Use mean across all sessions
        all_dfs = [pd.DataFrame(s["data"]) for s in sessions]
        combined = pd.concat(all_dfs, ignore_index=True)
        means = {c: combined[c].mean() for c in combined.columns if not combined[c].dropna().empty}
        fig.add_trace(build_radar_trace(name, means, PALETTE[i % len(PALETTE)]))

    fig.update_layout(
        polar=dict(
            bgcolor="#1B3A2D",
            radialaxis=dict(visible=True, range=[0, 100], color="#A8D5B5",
                            tickfont=dict(color="#A8D5B5"), gridcolor="#2E6B4F"),
            angularaxis=dict(color="#E8F5E9", gridcolor="#2E6B4F")
        ),
        paper_bgcolor="#0D1F16",
        font_color="#E8F5E9",
        height=550,
        legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    st.divider()
    st.markdown("### 📋 Metric Summary Table")
    rows = []
    for name in selected_archers:
        profile  = load_archer_profile(name)
        sessions = [s for s in profile.get("sessions", []) if s.get("data")]
        if not sessions:
            continue
        all_dfs  = [pd.DataFrame(s["data"]) for s in sessions]
        combined = pd.concat(all_dfs, ignore_index=True)
        row = {"Archer": name}
        for label, col in METRIC_MAP.items():
            if col in combined.columns:
                row[f"{label} (mean °)"] = round(combined[col].mean(), 2)
                row[f"{label} (std °)"]  = round(combined[col].std(), 2)
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
