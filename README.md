# 🏹 Archery Biomechanics Analyser

**Computer-vision platform for biomechanical analysis of the archery shot — developed for and validated with the Madrid Archery Team.**

A two-plane (sagittal + zenithal) video analysis system that extracts frame-by-frame kinematic metrics of the draw and release, segments individual shots, scores technique consistency, and produces coach-ready Excel and PDF reports. Built on MediaPipe Pose, OpenCV and Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Biomechanical Metrics](#biomechanical-metrics)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Module Guide](#module-guide)
- [Calibration](#calibration)
- [Data & Privacy](#data--privacy)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

In recurve and compound archery, accuracy is decided in the final hundreds of milliseconds before release. Micro-adjustments of the wrist, forearm and draw elbow at full draw have an outsized effect on where the arrow lands. This project quantifies those movements from ordinary action-camera footage — no motion-capture suit, no physical markers — by running markerless pose estimation on two synchronised camera planes:

- **Sagittal (lateral) plane** — captures the draw arm, elbow flexion, wrist alignment and bow-arm cant.
- **Top / zenithal plane** — captures shoulder-line rotation and alignment from above.

The result is a longitudinal record of each archer's technique that coaches can review, compare across sessions, and benchmark across the team.

---

## Key Features

- **Dual-plane video analysis** with automatic temporal synchronisation between the two cameras (motion-energy cross-correlation, no audio track required).
- **Markerless pose estimation** using MediaPipe Pose (model complexity 2) with temporal smoothing and a Holistic fallback for the harder zenithal viewpoint.
- **Eight biomechanical metrics** computed per frame, each with a biomechanically grounded reference range.
- **Shot phase detection** — every frame is classified as `pre_draw`, `draw`, `full_draw`, `release` or `follow_through`, with per-phase timing. Supports multi-shot videos via an automatic re-arm mechanism.
- **Automatic shot segmentation** — detects individual shot cycles inside a longer clip.
- **Consistency Index (CI)** — a 0–100 score quantifying how repeatable an archer's movement pattern is across shots and sessions.
- **Real-time live camera mode** — webcam, USB action cam or RTSP stream, with optional dual-camera capture.
- **Pixel calibration** — ArUco-marker auto-detection or manual reference, converting normalised coordinates into real-world millimetres.
- **Per-archer profiles** — persistent session history, progression charts and metadata, stored as local JSON.
- **Coach Dashboard** — team-wide consistency ranking, metric heatmap and distribution plots.
- **Radar / spider charts** — instant visual technique fingerprint, single-archer evolution or multi-archer comparison.
- **Professional exports** — formatted multi-sheet Excel workbook, multi-page PDF report, annotated MP4 video, and raw CSV.

---

## Biomechanical Metrics

| Metric | Plane | Normal Range | What it measures |
|--------|-------|-------------|------------------|
| Arm–Trunk Angle | Sagittal | 60° – 130° | Draw-arm elevation relative to the torso |
| Elbow Angle | Sagittal | 100° – 170° | Draw-arm flexion at full draw |
| Wrist Angle | Sagittal | 140° – 180° | Wrist alignment — deviation risks accuracy |
| DFL Angle | Sagittal | 0° – 30° | Draw Force Line quality (bow-arm alignment) |
| Elbow–Wrist Vertical | Sagittal | 0° – 45° | Bow-forearm cant at release |
| Bow Cant Angle | Sagittal | 0° – 20° | Bow tilt relative to vertical |
| Shoulder Tilt | Top | 0° – 15° | Shoulder-line rotation from the zenith view |
| Stability (rolling σ) | Sagittal | Lower is better | Steadiness of the hold at full draw |

Two derived kinematic signals are also produced: **angular velocity** (°/s) for the elbow and arm–trunk angles, and **shoulder sway** (micro-movement dispersion during full draw).

> ⚠️ Reference ranges are practical guidance distilled from coaching literature and team data, not clinical thresholds. "Outside range" flags a movement worth a coach's attention — it is not a diagnosis.

---

## Architecture

```
                 ┌─────────────────┐   ┌─────────────────┐
   Sagittal ───► │  PoseEstimator  │   │  PoseEstimator  │ ◄─── Top view
     video       │    Sagittal     │   │      Top        │      video
                 └────────┬────────┘   └────────┬────────┘
                          │                     │
                          ▼                     ▼
                 ┌──────────────────────────────────────┐
                 │            Synchroniser               │
                 │  (motion-energy cross-correlation)    │
                 └────────────────┬─────────────────────┘
                                  ▼
        ┌──────────────┬──────────────────┬──────────────────┐
        ▼              ▼                  ▼                  ▼
   ┌─────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Angles  │   │ Kinematics │   │ Phase detect │   │ Shot segment │
   │ metrics │   │ vel / sway │   │ + timing     │   │              │
   └────┬────┘   └─────┬──────┘   └──────┬───────┘   └──────┬───────┘
        └──────────────┴────────┬────────┴──────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │  Per-frame results table  │
                  └────────────┬─────────────┘
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼
    ┌──────────────┐   ┌──────────────┐      ┌──────────────────┐
    │ Excel / PDF  │   │ Profile JSON │      │ Plotly dashboards │
    │ CSV / video  │   │ persistence  │      │ radar / coach     │
    └──────────────┘   └──────────────┘      └──────────────────┘
```

---

## Installation

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/archery-biomechanics-analyser.git
cd archery-biomechanics-analyser

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app/main_app.py
```

The app opens in your browser at `http://localhost:8501`.

> **MediaPipe note:** on first launch MediaPipe downloads its pose model (~30 MB). This requires an internet connection once; afterwards it runs fully offline.

---

## Usage

A typical analysis flow:

1. **(Optional) Calibrate** — open the *Calibration* page and set the pixel-to-mm scale with an ArUco marker or a known reference object. Required only if you want distances in real-world millimetres.
2. **Upload & Analyse** — drop in a paired sagittal and top-view clip, choose which metrics to compute, and run the analysis. Watch the live preview, then download the Excel/PDF/CSV report.
3. **Save the profile** — enter an archer name before running and the session is appended to that archer's history automatically.
4. **Review analytics** — use *Shot Comparison*, *Radar Charts* and the *Coach Dashboard* to track progress and benchmark against the team.

---

## Module Guide

| Page | Purpose |
|------|---------|
| **Home** | Platform overview, metric reference and getting-started guide |
| **Calibration** | Establish the pixel-to-mm scale (ArUco or manual) |
| **Upload & Analyse** | Core engine: dual-video upload, metric extraction, phase detection, exports |
| **Live Camera** | Real-time pose estimation from webcam / action cam / RTSP, single or dual camera |
| **Archer Profiles** | Create profiles, browse session history, view progression charts |
| **Shot Comparison** | Overlay normalised kinematic curves across sessions, with Consistency Index |
| **Radar Charts** | Spider-chart technique fingerprint — single-archer evolution or multi-archer |
| **Coach Dashboard** | Team-level consistency ranking, metric heatmap and distribution plots |

---

## Calibration

Real-world distance measurements require a known scale. Two methods are supported:

**ArUco marker (recommended).** Print any `DICT_4X4_50` marker, measure its printed black-square side in millimetres, place it flat in the camera's field of view, and upload the clip. The first 90 frames are scanned and the median pixel-per-mm scale is applied.

**Manual reference.** Measure a known object in a frame (an arrow ≈ 620–710 mm, a recurve riser ≈ 380 mm) in pixels, enter both pixel and real lengths, and the scale is computed directly.

Without calibration, distances are reported in MediaPipe's normalised 0–1 coordinates; all angle metrics work regardless of calibration.

---

## Data & Privacy

- All data stays **local**. Archer profiles are plain JSON files under `data/profiles/`. Nothing is uploaded anywhere.
- Uploaded videos are written to temporary files only for the duration of processing and deleted immediately afterwards.
- Pose estimation extracts joint coordinates only; raw video frames are never persisted unless you explicitly export an annotated video.

The repository ships with four demonstration profiles (`Ana_Garcia`, `Carlos_Lopez`, `Miguel_Torres`, `Test_Archer`) containing synthetic session data so the analytics pages are populated out of the box.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI framework | Streamlit |
| Pose estimation | MediaPipe Pose (complexity 2) + Holistic fallback |
| Computer vision | OpenCV (incl. ArUco) |
| Numerics | NumPy, pandas |
| Visualisation | Plotly |
| Excel export | openpyxl |
| PDF export | ReportLab + Matplotlib |

---

## Project Structure

```
archery-biomechanics-analyser/
├── app/
│   ├── main_app.py              # Home page / entry point
│   ├── _theme.py                # Centralised CSS theme & UI helpers
│   └── pages/
│       ├── 0_calibration.py     # Pixel-to-mm calibration
│       ├── 1_upload.py          # Core dual-video analysis engine
│       ├── 2_live_camera.py     # Real-time camera analysis
│       ├── 3_profiles.py        # Archer profile management
│       ├── 4_comparison.py      # Shot / session comparison
│       ├── 5_radar.py           # Radar technique fingerprints
│       └── 6_coach_dashboard.py # Team-level analytics
├── src/
│   ├── pose/                    # Sagittal & top-view pose estimators
│   ├── metrics/                 # Angles, kinematics, stability, consistency,
│   │                            #   shot-phase detection
│   ├── detection/               # Multi-shot segmentation
│   ├── sync/                    # Two-camera temporal synchronisation
│   ├── calibration/             # ArUco / manual pixel calibration
│   └── export/                  # Excel, PDF, annotated-video, profile JSON
├── data/profiles/               # Persistent archer profiles (JSON)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Roadmap

- Audio-spike synchronisation as an alternative to motion-energy alignment
- Per-shot (rather than per-session) consistency scoring inside a single clip
- Configurable, archer-specific reference ranges
- Release-timing analysis correlating wrist micro-movement with arrow grouping

---

## License

Released under the [MIT License](LICENSE).

---

*Developed by Alonso Martín Díez · Biomedical Engineering, Universidad Europea de Madrid · in collaboration with the Madrid Archery Team.*
