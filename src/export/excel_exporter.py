"""
src/export/excel_exporter.py
Exports analysis results to a professionally formatted Excel workbook.
Uses openpyxl for full formatting control.
"""

import io
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime


# ── Colour palette ────────────────────────────────────────────────────────────
GREEN_DARK  = "1B3A2D"
GREEN_MID   = "2E6B4F"
GOLD        = "C9A84C"
GOLD_LIGHT  = "E8C97A"
WHITE       = "FFFFFF"
LIGHT_GREEN = "E8F5E9"
GREY_LIGHT  = "F5F5F5"


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _font(bold=False, color=WHITE, size=11) -> Font:
    return Font(bold=bold, color=color, name="Calibri", size=size)


def _border() -> Border:
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)


def _center() -> Alignment:
    return Alignment(horizontal="center", vertical="center")


def export_to_excel(df: pd.DataFrame, archer_name: str = "", session_label: str = "") -> bytes:
    """
    Build a multi-sheet Excel workbook from the analysis DataFrame.

    Sheets:
      1. Summary        — key stats per metric
      2. Raw Data       — frame-by-frame data table
      3. Charts         — line charts of angle time series

    Returns bytes ready for st.download_button.
    """
    wb = Workbook()
    wb.remove(wb.active)   # remove default sheet

    _build_summary_sheet(wb, df, archer_name, session_label)
    _build_raw_sheet(wb, df)
    _build_charts_sheet(wb, df)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ── Sheet 1: Summary ──────────────────────────────────────────────────────────

def _build_summary_sheet(wb, df, archer_name, session_label):
    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False

    # Header banner
    ws.merge_cells("A1:H1")
    ws["A1"] = "🏹 ARCHERY BIOMECHANICS ANALYSER"
    ws["A1"].font      = Font(bold=True, color=GOLD, size=16, name="Calibri")
    ws["A1"].fill      = _fill(GREEN_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 36

    # Sub-header
    ws.merge_cells("A2:H2")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ws["A2"] = f"Archer: {archer_name or '—'}   |   Session: {session_label or '—'}   |   Generated: {now}"
    ws["A2"].font      = _font(color=GOLD_LIGHT, size=10)
    ws["A2"].fill      = _fill(GREEN_MID)
    ws["A2"].alignment = _center()

    # Column headers
    headers = ["Metric", "Mean (°)", "Std Dev (°)", "Min (°)", "Max (°)", "Median (°)", "Frames", "Status"]
    ws.append([""] * 8)     # spacer row
    ws.append(headers)
    hdr_row = 4
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=hdr_row, column=col, value=h)
        cell.font      = _font(bold=True, color=WHITE)
        cell.fill      = _fill(GREEN_MID)
        cell.alignment = _center()
        cell.border    = _border()

    # Metric rows
    angle_cols = [c for c in df.columns if c not in ["Frame"] and df[c].dropna().count() > 0]
    STATUS_RANGES = {
        "Angle_Arm_Trunk":       (60, 130),
        "Angle_Elbow":           (100, 170),
        "Angle_Wrist":           (140, 180),
        "DFL_Angle":             (0,  30),
        "Elbow_Wrist_Vertical":  (0,  45),
        "Shoulder_Tilt":         (0,  15),
        "Stability_Arm_Trunk":   (0,  10),   # lower = better
    }

    for i, col in enumerate(angle_cols):
        s = df[col].dropna()
        if s.empty:
            continue
        mean, std, mn, mx, med = s.mean(), s.std(), s.min(), s.max(), s.median()
        n = len(s)

        lo, hi = STATUS_RANGES.get(col, (None, None))
        if lo is not None:
            status = "✅ Normal" if lo <= mean <= hi else "⚠️ Review"
        else:
            status = "—"

        row_data = [col.replace("_", " "), round(mean, 2), round(std, 2),
                    round(mn, 2), round(mx, 2), round(med, 2), n, status]
        ws.append(row_data)
        r = hdr_row + 1 + i
        bg = LIGHT_GREEN if i % 2 == 0 else GREY_LIGHT
        for c_idx, val in enumerate(row_data, start=1):
            cell = ws.cell(row=r, column=c_idx, value=val)
            cell.fill      = _fill(bg)
            cell.font      = Font(name="Calibri", size=10)
            cell.alignment = _center()
            cell.border    = _border()
            if c_idx == 8 and "⚠️" in str(val):
                cell.font = Font(bold=True, color="C62828", name="Calibri", size=10)

    # Column widths
    widths = [28, 12, 12, 10, 10, 12, 10, 14]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ── Sheet 2: Raw Data ─────────────────────────────────────────────────────────

def _build_raw_sheet(wb, df):
    ws = wb.create_sheet("Raw Data")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:J1")
    ws["A1"] = "Frame-by-Frame Data"
    ws["A1"].font      = _font(bold=True, color=GOLD, size=13)
    ws["A1"].fill      = _fill(GREEN_DARK)
    ws["A1"].alignment = _center()
    ws.row_dimensions[1].height = 28

    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=2):
        ws.append(row)
        if r_idx == 2:
            for c_idx in range(1, len(row) + 1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.font      = _font(bold=True, color=WHITE)
                cell.fill      = _fill(GREEN_MID)
                cell.alignment = _center()
        else:
            bg = LIGHT_GREEN if r_idx % 2 == 0 else WHITE
            for c_idx in range(1, len(row) + 1):
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.fill      = _fill(bg)
                cell.font      = Font(name="Calibri", size=9)
                cell.alignment = _center()

    from openpyxl.utils import get_column_letter as _gcl
    for col_idx in range(1, len(df.columns) + 2):   # +1 for index col
        ws.column_dimensions[_gcl(col_idx)].width = 18


# ── Sheet 3: Charts ───────────────────────────────────────────────────────────

def _build_charts_sheet(wb, df):
    ws = wb.create_sheet("Charts")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:N1")
    ws["A1"] = "Kinematic Charts"
    ws["A1"].font      = _font(bold=True, color=GOLD, size=13)
    ws["A1"].fill      = _fill(GREEN_DARK)
    ws["A1"].alignment = _center()

    # Write a mini data table for the chart source
    angle_cols = [c for c in df.columns if c not in ["Frame"] and df[c].dropna().count() > 0]
    if not angle_cols or df.empty:
        ws.cell(row=3, column=1, value="No numeric metrics available to chart.")
        return
    start_row  = 3
    ws.cell(row=start_row, column=1, value="Frame")
    for c_idx, col in enumerate(angle_cols, start=2):
        ws.cell(row=start_row, column=c_idx, value=col)

    for r_idx, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        ws.cell(row=r_idx, column=1, value=row.get("Frame", r_idx - start_row))
        for c_idx, col in enumerate(angle_cols, start=2):
            val = row.get(col)
            ws.cell(row=r_idx, column=c_idx, value=round(val, 3) if val is not None and not np.isnan(val) else None)

    n_data = len(df)

    # Line chart: angles
    chart = LineChart()
    chart.title   = "Kinematic Angle Time Series"
    chart.style   = 10
    chart.y_axis.title = "Angle (°)"
    chart.x_axis.title = "Frame"
    chart.height  = 14
    chart.width   = 22

    for c_idx, col in enumerate(angle_cols, start=2):
        data = Reference(ws, min_col=c_idx, min_row=start_row, max_row=start_row + n_data)
        chart.add_data(data, titles_from_data=True)

    cats = Reference(ws, min_col=1, min_row=start_row + 1, max_row=start_row + n_data)
    chart.set_categories(cats)
    ws.add_chart(chart, "A3")

    # Bar chart: mean values
    means_row = start_row + n_data + 3
    ws.cell(row=means_row, column=1, value="Metric")
    ws.cell(row=means_row, column=2, value="Mean (°)")
    for i, col in enumerate(angle_cols):
        ws.cell(row=means_row + 1 + i, column=1, value=col.replace("_", " "))
        ws.cell(row=means_row + 1 + i, column=2, value=round(df[col].mean(), 2))

    bar = BarChart()
    bar.title   = "Mean Angle per Metric"
    bar.style   = 10
    bar.type    = "col"
    bar.y_axis.title = "Mean Angle (°)"
    bar.height  = 12; bar.width = 18
    data_ref = Reference(ws, min_col=2, min_row=means_row, max_row=means_row + len(angle_cols))
    cats_ref = Reference(ws, min_col=1, min_row=means_row + 1, max_row=means_row + len(angle_cols))
    bar.add_data(data_ref, titles_from_data=True)
    bar.set_categories(cats_ref)
    ws.add_chart(bar, "A35")
