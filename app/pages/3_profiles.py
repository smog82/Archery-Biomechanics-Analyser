"""
Page 3 — Archer Profiles
Manage archer profiles, view session history and progression over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os, json
from datetime import datetime

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_APP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _APP)
from src.export.profile_manager import (
    list_archers, load_archer_profile, delete_session,
    get_progression_df, save_session
)
import _theme

st.set_page_config(page_title="Archer Profiles", layout="wide")
_theme.inject(st)
_theme.header(st, "Archer Profiles",
              "Manage profiles · Browse session history · Track technique evolution")

archers = list_archers()

# ── Sidebar: create / select ──────────────────────────────────────────────────
with st.sidebar:
    _theme.section(st, "New Archer")
    new_name = st.text_input("Archer name", key="new_archer_name")
    new_cat  = st.selectbox("Category", ["Senior", "Junior", "Cadet", "Recurve", "Compound", "Barebow"])
    new_hand = st.selectbox("Draw hand", ["Right", "Left"])
    if st.button("Create profile", type="primary"):
        if new_name:
            save_session(new_name, "profile_created", pd.DataFrame(), meta={"category": new_cat, "hand": new_hand})
            st.success(f"✅ Profile created for {new_name}")
            st.rerun()
        else:
            st.warning("Enter a name first.")
    st.divider()
    st.markdown("### 🔍 Select Archer")
    selected = st.selectbox("Choose archer", ["— Select —"] + archers)

# ── No archers yet ────────────────────────────────────────────────────────────
if not archers:
    st.info("No archer profiles yet. Create one in the sidebar, or run an analysis in **Upload & Analyse** with an archer name.")
    st.stop()

# ── Archer grid ───────────────────────────────────────────────────────────────
st.markdown("### 🏹 Registered Archers")
cols = st.columns(min(len(archers), 4))
for i, name in enumerate(archers):
    profile = load_archer_profile(name)
    n_sessions = len(profile.get("sessions", []))
    with cols[i % 4]:
        st.markdown(f"""
        <div class="archer-card">
            <div class="archer-name">🏹 {name}</div>
            <div class="archer-meta">{n_sessions} session{'s' if n_sessions != 1 else ''} recorded</div>
        </div>""", unsafe_allow_html=True)

# ── Selected archer detail ────────────────────────────────────────────────────
if selected != "— Select —":
    st.divider()
    profile = load_archer_profile(selected)
    sessions = profile.get("sessions", [])

    col_info, col_stats = st.columns([1, 2])

    with col_info:
        st.markdown(f"### {selected}")
        meta = profile.get("meta", {})
        st.markdown(f"**Category:** {meta.get('category', '—')}")
        st.markdown(f"**Draw hand:** {meta.get('hand', '—')}")
        st.markdown(f"**Sessions:** {len(sessions)}")
        if sessions:
            st.markdown(f"**First session:** {sessions[0].get('date','—')}")
            st.markdown(f"**Last session:** {sessions[-1].get('date','—')}")

        st.markdown("#### 📋 Sessions")
        for s in sessions:
            label = s.get("label", "unnamed")
            date  = s.get("date", "")
            n     = s.get("n_frames", 0)
            st.markdown(f'<span class="session-tag">📅 {date} · {label} · {n} frames</span>', unsafe_allow_html=True)

    with col_stats:
        prog_df = get_progression_df(selected)
        if not prog_df.empty:
            st.markdown("#### 📈 Metric Progression Across Sessions")
            angle_cols = [c for c in prog_df.columns if "Angle" in c or "DFL" in c or "Tilt" in c]
            if angle_cols:
                fig = go.Figure()
                COLORS = ["#C9A84C", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#FF9800"]
                for i, col in enumerate(angle_cols):
                    fig.add_trace(go.Scatter(
                        x=prog_df["session"], y=prog_df[col],
                        name=col.replace("_", " "),
                        line=dict(color=COLORS[i % len(COLORS)], width=2),
                        mode="lines+markers"
                    ))
                fig.update_layout(
                    paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                    font_color="#E8F5E9", height=350,
                    legend=dict(bgcolor="#1B3A2D", bordercolor="#2E6B4F")
                )
                fig.update_xaxes(title="Session", gridcolor="#2E6B4F")
                fig.update_yaxes(title="Mean angle (°)", gridcolor="#2E6B4F")
                st.plotly_chart(fig, width="stretch")

            # Stability trend
            if "Stability_Arm_Trunk" in prog_df.columns:
                st.markdown("#### 🎯 Stability Trend (lower = more consistent)")
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=prog_df["session"], y=prog_df["Stability_Arm_Trunk"],
                    marker_color="#C9A84C", name="Stability (std dev)"
                ))
                fig2.update_layout(
                    paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                    font_color="#E8F5E9", height=250
                )
                fig2.update_xaxes(gridcolor="#2E6B4F")
                fig2.update_yaxes(title="Std dev (°)", gridcolor="#2E6B4F")
                st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No session data with metrics yet for this archer.")

    # Session detail
    if sessions:
        st.divider()
        st.markdown("#### 🔍 Inspect Individual Session")
        session_labels = [f"{s.get('date','')} — {s.get('label','unnamed')}" for s in sessions]
        chosen_s = st.selectbox("Select session", session_labels)
        s_idx = session_labels.index(chosen_s)
        s_data = sessions[s_idx].get("data", {})
        if s_data:
            df_s = pd.DataFrame(s_data)
            st.dataframe(df_s.describe().round(2), width="stretch")
            angle_c = [c for c in df_s.columns if "Angle" in c or "DFL" in c]
            if angle_c:
                fig3 = go.Figure()
                COLORS = ["#C9A84C","#4CAF50","#2196F3","#E91E63","#9C27B0"]
                for i, c in enumerate(angle_c):
                    fig3.add_trace(go.Scatter(y=df_s[c], name=c.replace("_"," "),
                                              line=dict(color=COLORS[i%len(COLORS)])))
                fig3.update_layout(paper_bgcolor="#0D1F16", plot_bgcolor="#1B3A2D",
                                   font_color="#E8F5E9", height=320)
                fig3.update_xaxes(title="Frame", gridcolor="#2E6B4F")
                fig3.update_yaxes(title="Angle (°)", gridcolor="#2E6B4F")
                st.plotly_chart(fig3, width="stretch")
