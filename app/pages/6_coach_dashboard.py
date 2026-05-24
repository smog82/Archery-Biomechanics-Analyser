"""
Page 6 — Coach Dashboard
Aggregated team view: all archers, ranking, consistency comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_APP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _APP)
from src.export.profile_manager import list_archers, load_archer_profile, get_progression_df
from src.metrics.consistency import ConsistencyIndex
import _theme

st.set_page_config(page_title="Coach Dashboard", layout="wide")
_theme.inject(st)
_theme.header(st, "Coach Dashboard",
              "Aggregated team overview · Consistency rankings · Metric benchmarks")

archers = list_archers()

if not archers:
    st.info("No archer profiles yet. Create profiles and run analyses in the other modules.")
    st.stop()

with st.sidebar:
    _theme.section(st, "Dashboard Settings")
    rank_metric = st.selectbox("Rank archers by consistency of", [
        "Angle_Arm_Trunk", "Angle_Elbow", "Angle_Wrist", "DFL_Angle"
    ])
    min_sessions = st.slider("Min sessions to include", 1, 5, 1)
    show_heatmap = st.checkbox("Show team heatmap", value=True)

# ── Aggregate data ────────────────────────────────────────────────────────────
team_rows = []
for name in archers:
    profile  = load_archer_profile(name)
    sessions = [s for s in profile.get("sessions", []) if s.get("data")]
    if len(sessions) < min_sessions:
        continue
    all_dfs  = [pd.DataFrame(s["data"]) for s in sessions]
    combined = pd.concat(all_dfs, ignore_index=True)
    row = {"Archer": name, "Sessions": len(sessions), "Total frames": len(combined)}
    for col in ["Angle_Arm_Trunk", "Angle_Elbow", "Angle_Wrist",
                "DFL_Angle", "Elbow_Wrist_Vertical", "Shoulder_Tilt",
                "Stability_Arm_Trunk"]:
        if col in combined.columns:
            row[f"{col}_mean"] = combined[col].mean()
            row[f"{col}_std"]  = combined[col].std()
    # Consistency index
    all_series = [pd.DataFrame(s["data"])[rank_metric].dropna().values
                  for s in sessions
                  if s.get("data") and rank_metric in pd.DataFrame(s["data"]).columns]
    if len(all_series) > 1:
        row["CI"] = ConsistencyIndex.compute(all_series)
    elif len(all_series) == 1:
        row["CI"] = None
    team_rows.append(row)

if not team_rows:
    st.warning("Not enough session data to build the dashboard. Run more analyses first.")
    st.stop()

team_df = pd.DataFrame(team_rows)

# ── Team headline stats ───────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="team-stat"><div class="val">{len(team_rows)}</div><div class="lbl">Active Archers</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="team-stat"><div class="val">{int(team_df["Sessions"].sum())}</div><div class="lbl">Total Sessions</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="team-stat"><div class="val">{int(team_df["Total frames"].sum())}</div><div class="lbl">Frames Analysed</div></div>', unsafe_allow_html=True)
ci_vals = team_df["CI"].dropna()
ci_mean = ci_vals.mean() if not ci_vals.empty else 0
col4.markdown(f'<div class="team-stat"><div class="val">{ci_mean:.0f}</div><div class="lbl">Team Avg Consistency</div></div>', unsafe_allow_html=True)

st.divider()

# ── Consistency ranking ───────────────────────────────────────────────────────
st.markdown("### 🎯 Consistency Ranking")
st.caption(f"Ranked by Consistency Index on **{rank_metric.replace('_',' ')}**.")

ranked = team_df.dropna(subset=["CI"]).sort_values("CI", ascending=False).reset_index(drop=True)
if not ranked.empty:
    MEDALS = ["🥇", "🥈", "🥉"]
    for i, row in ranked.iterrows():
        medal = MEDALS[i] if i < 3 else f"#{i+1}"
        ci    = row["CI"]
        color = "#4CAF50" if ci >= 80 else ("#FFC107" if ci >= 55 else "#F44336")
        st.markdown(f"""
        <div class="rank-card">
            <span class="rank-num">{medal}</span>
            <span class="rank-name">{row['Archer']}</span>
            <span style="color:{color};font-weight:bold;margin-left:16px;">CI = {ci:.1f}</span>
            <div class="rank-meta">{int(row['Sessions'])} sessions · {int(row['Total frames'])} frames</div>
        </div>""", unsafe_allow_html=True)

# ── CI bar chart ──────────────────────────────────────────────────────────────
if not ranked.empty:
    fig_ci = go.Figure(go.Bar(
        x=ranked["Archer"], y=ranked["CI"],
        marker_color=["#4CAF50" if v >= 80 else "#FFC107" if v >= 55 else "#F44336"
                      for v in ranked["CI"]],
        text=[f"{v:.1f}" for v in ranked["CI"]],
        textposition="outside"
    ))
    fig_ci.update_layout(
        paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
        font_color="#E8F5E9", height=300,
        yaxis=dict(range=[0, 105], title="Consistency Index", gridcolor="#2E6B4F"),
        xaxis=dict(title="Archer"),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_ci, use_container_width=True)

st.divider()

# ── Team heatmap ──────────────────────────────────────────────────────────────
if show_heatmap:
    st.markdown("### 🌡️ Team Metric Heatmap")
    st.caption("Mean angle per archer and metric. Useful for spotting team-wide patterns or outliers.")

    mean_cols = [c for c in team_df.columns if c.endswith("_mean")]
    if mean_cols:
        heat_df = team_df.set_index("Archer")[mean_cols].rename(
            columns={c: c.replace("_mean","").replace("_"," ") for c in mean_cols}
        )
        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            colorscale="YlGn",
            text=np.round(heat_df.values, 1),
            texttemplate="%{text}°",
            showscale=True
        ))
        fig_heat.update_layout(
            paper_bgcolor="#0D1F16",
            font_color="#E8F5E9",
            height=max(300, len(team_rows) * 60),
            margin=dict(t=20, b=20, l=120)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Per-metric team comparison box ───────────────────────────────────────────
st.markdown("### 📦 Team Distribution per Metric")
angle_mean_cols = [c for c in team_df.columns if c.endswith("_mean") and "Stability" not in c]
if angle_mean_cols and len(team_df) > 1:
    fig_box = go.Figure()
    COLORS = ["#C9A84C","#4CAF50","#2196F3","#E91E63","#9C27B0","#FF9800"]
    for i, col in enumerate(angle_mean_cols):
        fig_box.add_trace(go.Box(
            y=team_df[col].dropna(),
            name=col.replace("_mean","").replace("_"," "),
            marker_color=COLORS[i % len(COLORS)],
            boxmean=True
        ))
    fig_box.update_layout(
        paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
        font_color="#E8F5E9", height=380,
        yaxis=dict(title="Mean angle (°)", gridcolor="#2E6B4F")
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("📋 Full team data table"):
    display_cols = ["Archer", "Sessions", "CI"] + angle_mean_cols
    display_cols = [c for c in display_cols if c in team_df.columns]
    st.dataframe(team_df[display_cols].round(2), use_container_width=True, hide_index=True)
