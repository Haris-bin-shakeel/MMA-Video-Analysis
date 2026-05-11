# FIGHT-IQ
## Autonomous Fight Intelligence Extraction System
### Computer Vision Pipeline for MMA Combat Analysis

> **Version:** 1.0.0 · **Engine:** OpenCV · **Runtime:** Local / Offline · **Output:** Structured JSON  
> **Repository:** [`mma_fighter_analysis/app/core`](https://github.com/Haris-bin-shakeel/MMA-Video-Analysis/tree/main/mma_fighter_analysis/app/core)

---

## Overview

**FIGHT-IQ** is a computer vision–based fight intelligence extraction system designed to transform raw MMA video footage into structured, queryable behavioral data. By combining manual fighter grounding with multi-signal tracking, the system produces per-second analytical profiles of each fighter — enabling evidence-based performance review, tactical dissection, and research-grade behavioral modeling without reliance on any external cloud service or machine learning model.

Rather than producing highlight reels or subjective commentary, FIGHT-IQ produces **machine-readable fight intelligence** — a structured record of what happened, when it happened, and how each fighter moved through the bout.

---

## Design Philosophy

Most fight analysis today is retrospective and qualitative: coaches rewind footage, annotate moments by hand, and communicate insights verbally. This creates a bottleneck between observation and structured knowledge.

FIGHT-IQ is built around three principles:

**1. Grounding Before Tracking**  
Identity confusion is the single largest failure mode in multi-object tracking pipelines applied to combat sports. FIGHT-IQ eliminates this by requiring manual fighter selection in the first frame. This is not a limitation — it is a deliberate architectural choice that guarantees identity consistency throughout the entire video timeline.

**2. Signals Over Pixels**  
Raw video is noise. FIGHT-IQ compresses video data into eight behaviorally meaningful signals per second per fighter, each of which carries direct tactical interpretation. The goal is not to reproduce what can be seen — it is to produce what cannot easily be seen.

**3. Local Sovereignty**  
The system runs entirely offline. No frames, no data, and no metadata leave the machine. This is a hard requirement for professional sports environments where competitive intelligence is sensitive.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Raw video file (.mp4 / .avi / .mov)                            │
│  Frame extraction · Resolution normalization · FPS calibration  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    PREPROCESSING LAYER                          │
│  Frame-by-frame greyscale conversion                            │
│  Gaussian blur (noise suppression)                              │
│  Background region isolation                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   MANUAL GROUNDING LAYER                        │
│  Frame-1 ROI selection — Fighter A / Fighter B                  │
│  Bounding box initialization (OpenCV ROI selector)              │
│  Identity anchor establishment                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     TRACKING ENGINE                             │
│  OpenCV CSRT / KCF tracker (per fighter)                        │
│  Frame-to-frame bounding box propagation                        │
│  Position, size, and centroid continuity                        │
│  Identity preservation across full video timeline               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                 SIGNAL EXTRACTION ENGINE                        │
│  8 behavioral signals computed per fighter per second           │
│  Centroid displacement · Bounding box area · Aspect ratio       │
│  Movement velocity · Relative positioning · Proximity index     │
│  Movement entropy · Positional quadrant mapping                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       OUTPUT LAYER                              │
│  Structured JSON export — one record per second per fighter     │
│  Fighter A profile · Fighter B profile · Relative metrics       │
│  Ready for downstream analysis, visualization, or ML ingestion  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fight Intelligence Signals

FIGHT-IQ extracts eight signals per fighter per second. Each signal is engineered to carry direct tactical meaning — not just geometric measurement.

| Signal | Technical Definition | Combat Interpretation |
|---|---|---|
| **Centroid Position** | (x, y) coordinate of bounding box center | Octagon positioning — corner, center, cage |
| **Bounding Box Area** | Width × Height of tracked region in pixels | Fighter's physical presence and posture compactness |
| **Aspect Ratio** | Height / Width of bounding box | Stance indicator — upright, crouched, or grounded |
| **Frame-to-Frame Displacement** | Euclidean distance between centroids across frames | Raw movement activity within the second |
| **Movement Velocity** | Displacement normalized to frame rate (pixels/sec) | Pace and urgency of fighter movement |
| **Inter-Fighter Proximity** | Centroid-to-centroid distance between both fighters | Engagement distance — clinch, striking, or retreating range |
| **Movement Entropy** | Variance in displacement vectors across the second | Predictability of movement — high entropy = unpredictable |
| **Positional Quadrant** | Canvas divided into 4 zones; fighter's dominant zone | Octagon control — which sector a fighter occupies |

### Signal Interpretation Reference

**Bounding Box Area** compresses when a fighter drops into a low wrestling stance or flattens against the cage, and expands when they are standing upright in a dominant striking posture. A sustained area reduction in Fighter B while Fighter A's area remains stable often signals a successful level change or cage control sequence.

**Aspect Ratio** is one of the most sensitive postural signals. An aspect ratio approaching 1.0 (square bounding box) indicates a fighter is hunched, shooting, or in a clinch. A tall ratio (height >> width) indicates an upright, technical stance. Rapid transitions in aspect ratio mark takedown attempts.

**Movement Entropy** is the system's most tactically sophisticated signal. A fighter with low entropy is predictable — moving in consistent directions at consistent speeds. High entropy fighters are difficult to read. Tracking entropy across rounds reveals how fighters adapt their movement patterns under fatigue or tactical pressure.

**Inter-Fighter Proximity** is the octagon control signal. When this value drops below a calibrated threshold, both fighters are in striking or grappling range. When it rises sharply, one fighter has broken distance — either to reset, recover, or bait. Sustained low proximity with low movement velocity indicates a clinch or ground-and-pound scenario.

---

## Output Format

All extracted intelligence is serialized to structured JSON. Each file represents one full fight video, with per-second records for both fighters.

```json
{
  "video_metadata": {
    "filename": "ufc_305_main_event.mp4",
    "duration_seconds": 298,
    "fps": 30,
    "resolution": "1920x1080"
  },
  "analysis": [
    {
      "second": 1,
      "fighter_a": {
        "centroid": [512, 380],
        "bbox_area": 14420,
        "aspect_ratio": 2.14,
        "velocity_px_per_sec": 38.7,
        "quadrant": "bottom-left",
        "movement_entropy": 0.41
      },
      "fighter_b": {
        "centroid": [1024, 390],
        "bbox_area": 15110,
        "aspect_ratio": 2.09,
        "velocity_px_per_sec": 22.1,
        "quadrant": "bottom-right",
        "movement_entropy": 0.19
      },
      "relative": {
        "inter_fighter_distance_px": 512.4,
        "engagement_zone": "striking_range"
      }
    }
  ]
}
```

---

## Installation

**Requirements**

```
Python      >= 3.8
OpenCV      >= 4.5.0
NumPy       >= 1.21
```

**Setup**

```bash
git clone https://github.com/Haris-bin-shakeel/MMA-Video-Analysis.git
cd MMA-Video-Analysis
pip install -r requirements.txt
```

**Run**

```bash
python main.py --video path/to/fight.mp4 --output results/
```

On launch, the first frame of the video will render in a selection window. Draw bounding boxes around Fighter A and Fighter B using the ROI selector. Press `Enter` to confirm selection and begin analysis. The system processes the video and writes a timestamped JSON file to the specified output directory upon completion.

---

## Use Cases

| User | Application |
|---|---|
| **Coaches & Cornermen** | Post-fight positional review · Identify rounds where fighter lost octagon control |
| **Performance Analysts** | Velocity and entropy trend analysis across rounds · Fatigue signature detection |
| **Scouts & Talent Evaluators** | Behavioral fingerprinting of fighters across multiple bouts |
| **Sports Scientists** | Research-grade movement data collection without wearable sensors |
| **Broadcast & Media** | Data-driven narratives · Tactical breakdowns backed by signal evidence |
| **ML Researchers** | Labeled behavioral time-series data for downstream model training |

---

## Limitations

Professional systems require honest documentation of their constraints. FIGHT-IQ has the following known limitations:

**Tracking Drift Under Occlusion**  
FIGHT-IQ uses frame-continuity tracking. When fighters are fully overlapping (e.g., deep clinch or ground-and-pound), bounding boxes may merge or drift. The manual grounding layer reduces but does not eliminate this in extended grappling sequences.

**No Pose or Joint Estimation**  
The system does not use skeletal or keypoint models. Signals are derived from bounding box geometry, which means granular limb-level actions (specific strikes, guard postures) are not directly captured.

**Camera Stability Dependency**  
Signal accuracy degrades on broadcast footage with significant camera panning, zooming, or rapid angle cuts. Best results are achieved on fixed-camera or minimally-panned recordings.

**Pixel-Space Metrics Only**  
All distance and velocity values are expressed in pixel units relative to the video resolution. Cross-video comparison requires normalization by resolution and octagon-to-pixel calibration.

**Single-Camera Input**  
The system processes one video stream. Multi-angle fusion is not currently supported.

---

## Future Improvements

The following capabilities are under consideration for subsequent versions:

- **Automatic Re-identification on Tracker Loss** — Recovery mechanism to re-acquire fighters after occlusion without manual intervention
- **Octagon Calibration Module** — Pixel-to-meter conversion using known octagon dimensions for absolute spatial metrics
- **Round Boundary Detection** — Automatic segmentation of analysis output by fight round based on pause/reset detection
- **Per-Round Summary Statistics** — Aggregated signal profiles per round for rapid round-by-round comparison
- **CSV Export Mode** — Flat tabular export format compatible with spreadsheet and statistical analysis tools
- **Multi-Video Fighter Profiling** — Cross-bout behavioral fingerprint aggregation for longitudinal fighter modeling

---

## Project Structure

```
mma_fighter_analysis/
├── app/
│   └── core/                  # Primary analysis pipeline
│       ├── tracker.py         # OpenCV tracker initialization and management
│       ├── signal_extractor.py # 8-signal behavioral extraction logic
│       ├── output_writer.py   # JSON serialization and file management
│       └── utils.py           # Frame processing and calibration helpers
├── main.py                    # Entry point and CLI argument handling
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Haris bin Shakeel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

You are free to use, modify, and distribute this project for any purpose — personal, academic, or commercial — with no restrictions beyond preserving the copyright notice.

---

*FIGHT-IQ — Computer Vision Pipeline for MMA Combat Analysis*  
*Built with OpenCV · Runs locally · No cloud dependency*
