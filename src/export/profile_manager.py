"""
src/export/profile_manager.py
JSON-based persistence layer for archer profiles and session data.
Profiles are stored in data/profiles/<archer_name>.json
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

PROFILES_DIR = Path(__file__).parent.parent.parent / "data" / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _archer_path(name: str) -> Path:
    safe = name.strip().replace(" ", "_").replace("/", "-")
    return PROFILES_DIR / f"{safe}.json"


def _df_to_records(df: pd.DataFrame) -> dict:
    """Convert DataFrame to JSON-serialisable dict, dropping NaN."""
    return {col: [None if pd.isna(v) else round(float(v), 4)
                  for v in df[col]]
            for col in df.columns}


def _clean(obj):
    """Recursively convert numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj


# ── Public API ────────────────────────────────────────────────────────────────

def list_archers() -> list:
    """Return sorted list of all archer names with saved profiles."""
    paths = sorted(PROFILES_DIR.glob("*.json"))
    return [p.stem.replace("_", " ") for p in paths]


def load_archer_profile(name: str) -> dict:
    path = _archer_path(name)
    if not path.exists():
        return {"name": name, "meta": {}, "sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Return a default profile if the file is corrupted or unreadable
        return {"name": name, "meta": {}, "sessions": []}


def save_session(name: str, label: str, df: pd.DataFrame, meta: dict = None):
    """
    Append a session to the archer's profile.
    Creates the profile if it does not exist.
    """
    profile = load_archer_profile(name)
    if meta:
        profile.setdefault("meta", {}).update(meta)
    profile["name"] = name

    summary = {}
    if not df.empty:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            s = df[col].dropna()
            if s.empty:
                continue
            mean_v = float(s.mean())
            std_v  = float(s.std()) if len(s) > 1 else 0.0
            summary[col] = {
                "mean": None if np.isnan(mean_v) else round(mean_v, 4),
                "std":  None if np.isnan(std_v) else round(std_v, 4),
            }

    session = {
        "label":    label,
        "date":     datetime.now().strftime("%Y-%m-%d"),
        "n_frames": len(df),
        "data":     _clean(_df_to_records(df)) if not df.empty else {},
        "summary":  _clean(summary),
    }
    profile.setdefault("sessions", []).append(session)

    path = _archer_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def delete_session(name: str, session_index: int):
    profile = load_archer_profile(name)
    sessions = profile.get("sessions", [])
    if 0 <= session_index < len(sessions):
        sessions.pop(session_index)
        profile["sessions"] = sessions
        with open(_archer_path(name), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)


def get_progression_df(name: str) -> pd.DataFrame:
    """
    Return a DataFrame with one row per session, columns = metric means.
    Useful for plotting progression over time.
    """
    profile = load_archer_profile(name)
    rows = []
    for s in profile.get("sessions", []):
        if not s.get("summary"):
            continue
        row = {"session": f"{s.get('date','')} {s.get('label','')}".strip()}
        for col, stats in s["summary"].items():
            row[col] = stats.get("mean")
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
