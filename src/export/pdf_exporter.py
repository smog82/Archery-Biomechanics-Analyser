"""
src/export/pdf_exporter.py
Professional PDF report generator using reportlab.

Generates a 3-page report:
  Page 1 — Title + archer info + summary statistics table
  Page 2 — Angle time series chart (matplotlib)
  Page 3 — Phase timing table + shot segmentation summary
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

# ── reportlab imports ─────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Colour palette (matching the app theme) ───────────────────────────────────
_GREEN_DARK  = colors.HexColor("#1B3A2D")
_GREEN_MID   = colors.HexColor("#2E6B4F")
_GOLD        = colors.HexColor("#C9A84C")
_GOLD_LIGHT  = colors.HexColor("#E8C97A")
_TEXT_LIGHT  = colors.HexColor("#E8F5E9")
_BG_LIGHT    = colors.HexColor("#F5F9F6")

# Biomechanical reference ranges
_STATUS_RANGES: Dict[str, tuple] = {
    "Angle_Arm_Trunk":      (60,  130),
    "Angle_Elbow":          (100, 170),
    "Angle_Wrist":          (140, 180),
    "DFL_Angle":            (0,   30),
    "Elbow_Wrist_Vertical": (0,   45),
    "Bow_Cant_Angle":       (0,   20),
    "Shoulder_Tilt":        (0,   15),
    "Stability_Arm_Trunk":  (0,   10),
}


def _status(col: str, mean: float) -> str:
    lo, hi = _STATUS_RANGES.get(col, (None, None))
    if lo is None:
        return "—"
    return "✓ Normal" if lo <= mean <= hi else "⚠ Review"


def _status_color(col: str, mean: float):
    lo, hi = _STATUS_RANGES.get(col, (None, None))
    if lo is None:
        return colors.black
    return colors.HexColor("#1B5E20") if lo <= mean <= hi else colors.HexColor("#B71C1C")


def _chart_image(df: pd.DataFrame, width_cm: float = 16, height_cm: float = 10) -> Optional[io.BytesIO]:
    """Render angle time series as a PNG image (matplotlib) and return a BytesIO."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        angle_cols = [c for c in df.columns
                      if any(k in c for k in ("Angle", "DFL", "Tilt", "Cant", "Vertical"))
                      and df[c].dropna().count() > 3]
        if not angle_cols:
            return None

        palette = ["#C9A84C", "#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#FF9800"]
        fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
        fig.patch.set_facecolor("#1B3A2D")
        ax.set_facecolor("#1B3A2D")

        for i, col in enumerate(angle_cols):
            s = df[col].dropna()
            ax.plot(s.index, s.values, label=col.replace("_", " "),
                    color=palette[i % len(palette)], linewidth=1.4, alpha=0.9)

        ax.set_xlabel("Frame", color="#E8F5E9", fontsize=9)
        ax.set_ylabel("Angle (°)", color="#E8F5E9", fontsize=9)
        ax.tick_params(colors="#E8F5E9", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E6B4F")
        ax.grid(color="#2E6B4F", linewidth=0.5, alpha=0.6)
        leg = ax.legend(fontsize=7, loc="upper right",
                        facecolor="#1B3A2D", edgecolor="#2E6B4F", labelcolor="#E8F5E9")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def export_to_pdf(
    df: pd.DataFrame,
    archer_name: str = "",
    session_label: str = "",
    phase_timing: Optional[Dict[str, float]] = None,
    shot_count: int = 0,
    px_per_mm: Optional[float] = None,
    detection_rate_pct: float = 0.0,
) -> bytes:
    """
    Build a multi-page PDF report.

    Returns
    -------
    bytes — ready for st.download_button
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"Archery Biomechanics — {archer_name}",
        author="Archery Biomechanics Analyser",
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Style helpers ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ArchTitle", parent=styles["Title"],
        fontSize=22, textColor=_GOLD, alignment=TA_CENTER,
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "ArchSub", parent=styles["Normal"],
        fontSize=10, textColor=_GREEN_MID, alignment=TA_CENTER,
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "ArchH2", parent=styles["Heading2"],
        fontSize=13, textColor=_GREEN_DARK, spaceAfter=6, spaceBefore=14,
    )
    body_style = ParagraphStyle(
        "ArchBody", parent=styles["Normal"],
        fontSize=9.5, textColor=colors.black, spaceAfter=4,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: Title + info + summary
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("🏹 Archery Biomechanics Analyser", title_style))
    story.append(Paragraph("Computer Vision &amp; Biomechanics Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=_GOLD, spaceAfter=10))

    # Info table
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    info_rows = [
        ["Archer",        archer_name  or "—"],
        ["Session",       session_label or "—"],
        ["Generated",     now],
        ["Total frames",  str(len(df))],
        ["Detection rate",f"{detection_rate_pct:.1f}%"],
        ["Shots detected",str(shot_count)],
    ]
    if px_per_mm:
        info_rows.append(["Scale",  f"{px_per_mm:.3f} px/mm  ({1/px_per_mm:.2f} mm/px)"])

    info_tbl = Table(info_rows, colWidths=[4 * cm, 10 * cm])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), _GREEN_DARK),
        ("TEXTCOLOR",   (0, 0), (0, -1), _GOLD_LIGHT),
        ("TEXTCOLOR",   (1, 0), (1, -1), colors.black),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [_BG_LIGHT, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, _GREEN_MID),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=_GREEN_MID, spaceAfter=6))

    # Summary statistics table
    story.append(Paragraph("Biomechanical Summary", h2_style))

    angle_cols = [
        c for c in df.columns
        if c not in ("Frame", "Shot_Phase") and df[c].dropna().count() > 0
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    hdr = ["Metric", "Mean (°)", "Std Dev", "Min", "Max", "Median", "n", "Status"]
    rows = [hdr]
    for col in angle_cols:
        s   = df[col].dropna()
        if s.empty:
            continue
        mn = round(float(s.mean()), 2)
        st = round(float(s.std()),  2)
        mi = round(float(s.min()),  2)
        mx = round(float(s.max()),  2)
        md = round(float(s.median()), 2)
        rows.append([
            col.replace("_", " "),
            str(mn), str(st), str(mi), str(mx), str(md), str(len(s)),
            _status(col, mn),
        ])

    col_w = [4.5*cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.5*cm, 1.8*cm, 1.2*cm, 2.0*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    tbl_style = [
        ("BACKGROUND",   (0, 0), (-1, 0), _GREEN_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _GOLD),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("GRID",         (0, 0), (-1, -1), 0.4, _GREEN_MID),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    # Row shading
    for i in range(1, len(rows)):
        bg = _BG_LIGHT if i % 2 == 0 else colors.white
        tbl_style.append(("BACKGROUND", (0, i), (-1, i), bg))
    # Status column colour
    for i, col in enumerate(angle_cols, start=1):
        s = df[col].dropna()
        if s.empty:
            continue
        mn = float(s.mean())
        c  = _status_color(col, mn)
        tbl_style.append(("TEXTCOLOR", (7, i), (7, i), c))
        tbl_style.append(("FONTNAME",  (7, i), (7, i), "Helvetica-Bold"))

    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: Kinematic chart
    # ─────────────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Kinematic Time Series", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=_GREEN_MID, spaceAfter=8))

    chart_buf = _chart_image(df)
    if chart_buf:
        img = Image(chart_buf, width=16 * cm, height=10 * cm)
        story.append(img)
    else:
        story.append(Paragraph("(matplotlib not available for chart rendering)", body_style))

    # Angular velocity chart
    vel_cols = [c for c in df.columns if "Vel_" in c and df[c].dropna().count() > 3]
    if vel_cols:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Angular Velocity", h2_style))
        vel_buf = _chart_image(df[["Frame"] + vel_cols] if "Frame" in df.columns else df[vel_cols])
        if vel_buf:
            story.append(Image(vel_buf, width=16 * cm, height=7 * cm))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3: Phase timing + per-metric interpretation
    # ─────────────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Shot Phase Analysis", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=_GREEN_MID, spaceAfter=8))

    if phase_timing:
        pt_rows = [["Phase", "Duration (s)", "Notes"]]
        notes_map = {
            "pre_draw":       "Arm extension before draw begins",
            "draw":           "Active draw — elbow bending",
            "full_draw":      "Stable aim position",
            "release":        "Arrow release event",
            "follow_through": "Post-release arm movement",
        }
        for phase, secs in phase_timing.items():
            if secs > 0:
                pt_rows.append([phase.replace("_", " ").title(),
                                 f"{secs:.2f} s",
                                 notes_map.get(phase, "")])
        pt_tbl = Table(pt_rows, colWidths=[4*cm, 3.5*cm, 8.5*cm])
        pt_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _GREEN_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _GOLD),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.4, _GREEN_MID),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_BG_LIGHT, colors.white]),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pt_tbl)
        story.append(Spacer(1, 0.5 * cm))

    # Interpretation notes
    story.append(Paragraph("Metric Reference Ranges", h2_style))
    interp_rows = [["Metric", "Normal Range", "Interpretation"]]
    interp_data = [
        ("Arm Trunk Angle",      "60°–130°",  "Draw arm elevation relative to torso"),
        ("Elbow Angle",          "100°–170°", "Draw arm bend at full draw"),
        ("Wrist Angle",          "140°–180°", "Wrist alignment — deviation risks accuracy"),
        ("DFL Angle",            "0°–30°",    "Draw Force Line quality — bow arm alignment"),
        ("Elbow Wrist Vertical", "0°–45°",    "Bow arm cant at release"),
        ("Bow Cant Angle",       "0°–20°",    "Bow tilt relative to vertical"),
        ("Shoulder Tilt",        "0°–15°",    "Shoulder rotation from zenith view"),
    ]
    for m, r, i in interp_data:
        interp_rows.append([m, r, i])
    i_tbl = Table(interp_rows, colWidths=[4.5*cm, 3*cm, 8.5*cm])
    i_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _GREEN_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _GOLD),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.4, _GREEN_MID),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_BG_LIGHT, colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(i_tbl)

    # Footer
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=_GREEN_MID))
    story.append(Paragraph(
        f"Generated by Archery Biomechanics Analyser · {now} · "
        "Madrid Archery Team — Computer Vision &amp; Biomechanics Platform",
        ParagraphStyle("footer", parent=body_style, fontSize=7,
                       textColor=_GREEN_MID, alignment=TA_CENTER),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
