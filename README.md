Archery Biomechanics Analyser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00A67E?style=for-the-badge&logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Computer-vision platform for biomechanical analysis of the archery shot**

*Biomedical Engineering Portfolio · Alonso Martín Díez · Universidad Europea de Madrid 2026*

</div>

---

## 📋 Overview

Complete **markerless dual-plane biomechanical analysis system** for the archery shot. Two synchronised camera views (sagittal + zenithal) are processed through MediaPipe Pose to extract frame-by-frame kinematic metrics, classify shot phases, score technique consistency, and produce coach-ready reports — no motion-capture suit, no physical markers required.

Developed for and validated with the **Madrid Archery Team**.

### Platform at a Glance

| Capability | Detail |
|:-----------|:-------|
| 🎯 **8 biomechanical metrics** | Per-frame angles with reference ranges |
| 📹 **Dual-plane analysis** | Synchronised sagittal + top-view processing |
| 🤖 **Markerless pose** | MediaPipe Pose (complexity 2) + Holistic fallback |
| ⏱️ **5-phase shot detection** | pre_draw → draw → full_draw → release → follow_through |
| 🔄 **Multi-shot support** | Automatic re-arm for videos with multiple shots |
| 📊 **Consistency Index** | 0–100 repeatability score across shots/sessions |
| 📷 **Live camera mode** | Webcam / action cam / RTSP, single or dual |
| 📏 **Pixel calibration** | ArUco marker auto-detection or manual reference |
| 👤 **Archer profiles** | Persistent session history + progression tracking |
| 📄 **Professional exports** | Excel, PDF report, annotated MP4, CSV |

---

## 🏹 Biomechanical Metrics

Eight metrics are computed per frame from the MediaPipe 33-landmark skeleton:

```
Sagittal plane (lateral view)          Top plane (zenithal view)
──────────────────────────────         ──────────────────────────
Arm–Trunk Angle    60°–130°            Shoulder Tilt      0°–15°
Elbow Angle       100°–170°
Wrist Angle       140°–180°
DFL Angle           0°–30°
Elbow–Wrist Vert.   0°–45°
Bow Cant Angle      0°–20°
Stability (σ)     lower = better
```

| Metric | Plane | Normal Range | What it measures |
|:-------|:------|:-------------|:-----------------|
| Arm–Trunk Angle | Sagittal | 60° – 130° | Draw-arm elevation relative to the torso |
| Elbow Angle | Sagittal | 100° – 170° | Draw-arm flexion at full draw |
| Wrist Angle | Sagittal | 140° – 180° | Wrist alignment — deviation risks accuracy |
| DFL Angle | Sagittal | 0° – 30° | Draw Force Line quality (bow-arm alignment) |
| Elbow–Wrist Vertical | Sagittal | 0° – 45° | Bow-forearm cant at release |
| Bow Cant Angle | Sagittal | 0° – 20° | Bow tilt relative to vertical |
| Shoulder Tilt | Top | 0° – 15° | Shoulder-line rotation from zenith view |
| Stability (rolling σ) | Sagittal | Lower is better | Steadiness of the hold at full draw |

**Derived kinematics:** Angular velocity (°/s) for elbow and arm–trunk, and shoulder sway (micro-movement dispersion during the hold).

> ⚠️ Reference ranges are coaching guidance, not clinical thresholds. "Out of range" flags a movement worth attention — not a diagnosis.

---

## ⏱️ Shot Phase Detection

Each frame is classified from elbow-angle dynamics into five phases with per-phase timing:

```
┌──────────┐    ┌──────┐    ┌───────────┐    ┌─────────┐    ┌────────────────┐
│ PRE_DRAW │ →  │ DRAW │ →  │ FULL_DRAW │ →  │ RELEASE │ →  │ FOLLOW_THROUGH │
│ arm ext. │    │ bend │    │  stable   │    │ sudden  │    │   post-release │
│  >160°   │    │ 90-160°│   │  σ < 3°  │    │ Δ>15°/3f│    │   re-arm →     │
└──────────┘    └──────┘    └───────────┘    └─────────┘    └────────────────┘
                                                                     ↓
                                                            back to PRE_DRAW
                                                          (multi-shot support)
```

---

## 🏗️ Architecture

```
                 ┌─────────────────────┐   ┌─────────────────────┐
  Sagittal ────► │ PoseEstimatorSag.   │   │ PoseEstimatorTop    │ ◄──── Top view
    video        │ MediaPipe c2 + avg  │   │ Pose + Holistic EMA │       video
                 └─────────┬───────────┘   └─────────┬───────────┘
                           │                         │
                           ▼                         ▼
                 ┌─────────────────────────────────────────────┐
                 │              Synchroniser                    │
                 │   motion-energy normalised cross-correlation │
                 └──────────────────┬──────────────────────────┘
                                    ▼
          ┌──────────────┬──────────────────┬──────────────────┐
          ▼              ▼                  ▼                  ▼
    ┌───────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐
    │ 8 Angle   │  │ Kinematics │  │ Phase detect │  │    Shot      │
    │ metrics   │  │ vel · sway │  │ + timing     │  │ segmentation │
    └─────┬─────┘  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘
          └──────────────┴────────┬───────┴──────────────────┘
                                  ▼
                   ┌────────────────────────┐
                   │  Per-frame DataFrame   │
                   └───────────┬────────────┘
             ┌─────────────────┼──────────────────────┐
             ▼                 ▼                      ▼
     ┌──────────────┐  ┌──────────────┐     ┌──────────────────┐
     │ Excel · PDF  │  │ Profile JSON │     │ Plotly dashboards │
     │ CSV · MP4    │  │ persistence  │     │ radar · coach     │
     └──────────────┘  └──────────────┘     └──────────────────┘
```

---

## 📁 Repository Structure

```
archery-biomechanics-analyser/
│
├── 🖥️ app/
│   ├── main_app.py                    ← Home page / entry point
│   ├── _theme.py                      ← Centralised CSS theme & UI helpers
│   └── pages/
│       ├── 0_calibration.py           ← ArUco / manual pixel-to-mm calibration
│       ├── 1_upload.py                ← Core dual-video analysis engine
│       ├── 2_live_camera.py           ← Real-time webcam / RTSP analysis
│       ├── 3_profiles.py              ← Archer profile management
│       ├── 4_comparison.py            ← Shot / session kinematic overlay
│       ├── 5_radar.py                 ← Spider chart technique fingerprints
│       └── 6_coach_dashboard.py       ← Team-level analytics
│
├── 🔬 src/
│   ├── pose/
│   │   ├── sagittal.py                ← Sagittal pose estimator (c2 + moving avg)
│   │   └── top.py                     ← Top pose estimator (Pose + Holistic + EMA)
│   ├── metrics/
│   │   ├── angles.py                  ← 8 angle metrics + auto-detect side
│   │   ├── kinematics.py              ← Angular velocity + shoulder sway
│   │   ├── stability.py               ← Rolling σ stability tracker
│   │   ├── consistency.py             ← Consistency Index (0–100)
│   │   └── shot_phase_detector.py     ← 5-phase classification + re-arm
│   ├── detection/
│   │   └── shot_segmenter.py          ← Multi-shot cycle detection
│   ├── sync/
│   │   └── synchroniser.py            ← Cross-correlation temporal sync
│   ├── calibration/
│   │   └── aruco_calibrator.py        ← ArUco + manual scale
│   └── export/
│       ├── excel_exporter.py          ← Multi-sheet formatted workbook
│       ├── pdf_exporter.py            ← 3-page PDF report (ReportLab)
│       ├── video_exporter.py          ← Annotated MP4 with HUD overlay
│       └── profile_manager.py         ← JSON profile persistence
│
├── 📊 data/profiles/                  ← Persistent archer profiles (JSON)
│   ├── Ana_Garcia.json                ← Demo profile (3 sessions)
│   ├── Carlos_Lopez.json              ← Demo profile (3 sessions)
│   ├── Miguel_Torres.json             ← Demo profile (2 sessions)
│   └── Test_Archer.json               ← Demo profile (1 session)
│
├── 📄 docs/
│   └── Technical_Documentation.docx   ← Full 9-page technical docs
│
├── 🔧 requirements.txt
├── 📝 README.md
└── ⚖️ LICENSE (MIT)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/alonsomartindiez/archery-biomechanics-analyser.git
cd archery-biomechanics-analyser
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run app/main_app.py
```

Opens at `http://localhost:8501`.

> 💡 On first launch, MediaPipe downloads its pose model (~30 MB). Requires internet once; runs fully offline afterwards.

---

## 🎯 Usage Workflow

| Step | Page | Action |
|:-----|:-----|:-------|
| **01** | Calibration | Set pixel-to-mm scale (ArUco or manual reference) — optional |
| **02** | Upload & Analyse | Drop sagittal + top-view videos, select metrics, run analysis |
| **03** | Upload & Analyse | Download Excel / PDF / CSV report, optionally export annotated video |
| **04** | Archer Profiles | Session auto-saved to archer's profile (enter name before running) |
| **05** | Shot Comparison | Overlay normalised kinematic curves across sessions |
| **06** | Radar Charts | Spider chart technique fingerprint — single or multi-archer |
| **07** | Coach Dashboard | Team-wide consistency ranking, heatmap, distributions |

---

## 📏 Calibration

| Method | How it works | When to use |
|:-------|:-------------|:------------|
| **ArUco marker** (recommended) | Print DICT_4X4_50 marker, place in camera view, auto-detected in first 90 frames | Precise, reproducible measurements |
| **Manual reference** | Measure a known object in the frame (arrow ≈ 700 mm, riser ≈ 380 mm) | Quick setup, no marker needed |

Without calibration, distances are in normalised coordinates (0–1). All angle metrics work regardless.

---

## 🔧 Tech Stack

| Layer | Technology |
|:------|:-----------|
| UI framework | Streamlit |
| Pose estimation | MediaPipe Pose (complexity 2) + Holistic fallback |
| Computer vision | OpenCV (incl. ArUco module) |
| Numerics | NumPy · pandas |
| Visualisation | Plotly |
| Excel export | openpyxl |
| PDF export | ReportLab + Matplotlib |
| Video export | OpenCV VideoWriter (mp4v codec) |

---

## 🔒 Data & Privacy

- **All data stays local** — profiles are plain JSON under `data/profiles/`, nothing is uploaded externally
- **Videos are ephemeral** — written to temp files only during processing, deleted immediately after
- **No raw frames stored** — pose estimation extracts joint coordinates only; frames are never persisted unless you explicitly export an annotated video
- **4 synthetic demo profiles** ship with the repository so analytics pages work out of the box

---

## 🗺️ Roadmap

- [ ] Audio-spike synchronisation as an alternative to motion-energy alignment
- [ ] Per-shot (within a single clip) consistency scoring
- [ ] Configurable, archer-specific reference ranges
- [ ] Release-timing analysis correlating wrist micro-movement with arrow grouping
- [ ] 3D pose estimation with depth cameras

---

## 📚 References

1. Landers, D.M. et al. (1994). *Biomechanical analysis of archery performance*. Olympic Scientific Congress.
2. Ertan, H. et al. (2003). *Activation patterns in forearm muscles during archery shooting*. Human Movement Science, 22(1), 37-45.
3. Soylu, A.R. et al. (2006). *Archery performance level and repeatability of event-related EMG*. Human Movement Science, 25(6), 767-774.
4. Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.

---

## 👤 Author

**Alonso Martín Díez** · Biomedical Engineering · Universidad Europea de Madrid · 2026

In collaboration with the **Madrid Archery Team**.
