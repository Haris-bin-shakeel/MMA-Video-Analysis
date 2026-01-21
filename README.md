# MMA Fighter Video Analysis System

## Overview

This system provides automated fighter tracking and behavioral analysis for MMA video footage. It uses pure computer vision techniques (no machine learning) to track manually selected fighters and extract meaningful movement signals.

### Key Features

- ✅ **Manual Fighter Selection** - You choose your fighter in the first frame
- ✅ **Identity Preservation** - Maintains fighter identity throughout the video
- ✅ **Presence Detection** - Outputs time intervals when each fighter is visible
- ✅ **Behavioral Signals** - Extracts 8 movement metrics per second
- ✅ **Pure OpenCV** - No ML models, no cloud services, runs locally
- ✅ **JSON Output** - Machine-readable results for downstream analysis

---

## System Requirements

### Hardware Requirements
- **CPU:** Intel i5 or equivalent (4+ cores recommended)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB free space (for code + dependencies)
- **Display:** Required for manual fighter selection

### Software Requirements
- **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python:** 3.8, 3.9, 3.10, or 3.11 (3.10 recommended)
- **Display Server:** Required (X11 on Linux, native on Windows/macOS)

---

## Installation Guide

Follow these steps **exactly** in order.

### Step 1: Verify Python Installation

Open a terminal/command prompt and check your Python version:

```bash
python --version
```

**Expected output:** `Python 3.8.x` through `Python 3.11.x`

**If Python is not installed or wrong version:**

- **Windows:** Download from [python.org](https://www.python.org/downloads/) and install with "Add to PATH" checked
- **macOS:** `brew install python@3.10` (requires Homebrew)
- **Linux:** `sudo apt update && sudo apt install python3.10 python3.10-venv`

### Step 2: Extract the Project

Extract the provided ZIP file to a location without spaces in the path:

**✅ Good paths:**
- `C:\Projects\mma_fighter_analysis`
- `/home/user/mma_fighter_analysis`
- `/Users/username/Desktop/mma_fighter_analysis`


```bash
# Navigate to the extracted folder
cd /path/to/mma_fighter_analysis
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verification:** Your prompt should now show `(venv)` at the beginning.

### Step 4: Install Dependencies

Copy the following into a file named `requirements.txt` in your project root:

```txt
numpy==1.26.4
opencv-contrib-python==4.11.0.86
matplotlib==3.10.8
pillow==12.1.0
python-dateutil==2.9.0.post0
six==1.17.0
packaging==25.0
pyparsing==3.3.1
cycler==0.12.1
kiwisolver==1.4.9
contourpy==1.3.3
```

Then install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** Should see "Successfully installed..." messages for all packages.

**Verification:**
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```
Expected: `OpenCV version: 4.11.0`

### Step 5: Verify Project Structure

Check that your project has this structure:

```
mma_fighter_analysis/
├── app/
│   └── core/
│       ├── __init__.py
│       ├── video_loader.py
│       ├── selector.py
│       ├── tracker.py
│       ├── presence_zones.py
│       └── behavioral_signals.py
├── tests/
│   └── test_signals.py
├── Videos/                    # Create this folder your vedios here
├── requirements.txt
└── README.md
```

**Create missing folders:**
```bash
mkdir Videos
mkdir output
```

### Step 6: Test Installation

Run the test script without arguments to see usage instructions:

```bash
python tests/test_signals.py
```

**Expected output:** Should display usage instructions without errors.

---

## Running the System

### Basic Workflow

1. Place your video file in the `Videos/` folder
2. Run the analysis script
3. Manually select fighters in the popup window
4. Wait for processing to complete
5. Find results in `output/` folder

### Step-by-Step Execution

#### Step 1: Prepare Your Video

Copy your MMA video to the `Videos/` folder:

```bash
# Example: Copy from Downloads
cp ~/Downloads/my_fight.mp4 Videos/
```

**Supported formats:** `.mp4`, `.mov`, `.webm`, `.avi`, `.mkv`

#### Step 2: Run the Analysis Script

```bash
python tests/test_signals.py Videos/my_fight.mp4
```

**What happens next:** The video will load and display metadata.

#### Step 3: Manual Fighter Selection

A window will open showing the first frame of your video.

**Selection Process:**

1. **Draw First Fighter:**
   - Click and hold left mouse button
   - Drag to create a rectangle around the first fighter
   - The fighter should be mostly inside the box
   - Release mouse button
   - Press **SPACEBAR** to confirm

2. **Draw Second Fighter:**
   - Click and drag around the second fighter
   - Press **SPACEBAR** to confirm

3. **Identify Your Fighter:**
   - Press **1** if Fighter 1 (first box) is YOUR fighter
   - Press **2** if Fighter 2 (second box) is YOUR fighter

**Tips for good selection:**
- Draw tight boxes around fighters (not too much empty space)
- Include the fighter's torso and head
- Don't worry about arms/legs extending beyond the box
- If you make a mistake, press **R** to reset the last box
- Press **ESC** to cancel (you'll need to restart)

#### Step 4: Processing

After selection, the system will:
- Display a confirmation screen (2 seconds)
- Begin frame-by-frame processing
- Show a live preview window with bounding boxes
- Display progress in the terminal

**Progress indicators:**
```
[████████████████████░░░░░░░░]  65.3%  Frame   2350/3600  My: ✓  Opp: ✓
```

**Controls during processing:**
- Press **Q** to stop early (results will be saved for processed frames)
- Press **Ctrl+C** to abort completely

**Processing time estimates:**
- 1-minute video (1080p): ~2-3 minutes
- 5-minute video (1080p): ~10-15 minutes
- 10-minute video (720p): ~12-18 minutes

#### Step 5: View Results

After processing completes, find your results in the `output/` folder:

```bash
output/
├── my_fight_presence_zones.json
└── my_fight_behavioral_signals.json
```

**Terminal output** will also display:
- Summary of presence zones
- Average signal values
- File save locations

---

## Output Files Explained

### File 1: Presence Zones (`*_presence_zones.json`)

This file contains time intervals when each fighter was visible.

**Example:**
```json
{
  "video_id": "my_fight",
  "my_fighter_presence": [
    { "start": 0.0, "end": 45.23 },
    { "start": 52.67, "end": 120.45 }
  ],
  "opponent_presence": [
    { "start": 0.0, "end": 48.91 },
    { "start": 50.12, "end": 122.89 }
  ],
  "metadata": {
    "fps": 30.0,
    "total_frames": 3600,
    "duration_seconds": 120.0,
    "my_fighter_stats": {
      "zone_count": 2,
      "total_frames_reliable": 2890,
      "visibility_percentage": 80.3
    },
    "opponent_stats": {
      "zone_count": 2,
      "total_frames_reliable": 3045,
      "visibility_percentage": 84.6
    }
  }
}
```

**How to use:**
- `start` and `end` are in **seconds**
- Gaps between zones indicate when fighter was off-screen or tracking was lost
- `visibility_percentage` shows what % of video each fighter was reliably tracked

### File 2: Behavioral Signals (`*_behavioral_signals.json`)

This file contains 8 movement metrics computed every second.

**Example:**
```json
{
  "video_id": "my_fight",
  "signals": [
    {
      "t": 0,
      "distance_delta": -0.12,
      "forward_velocity": 0.34,
      "lateral_displacement": 0.08,
      "vertical_level_change": 0.02,
      "contact_duration": 0.0,
      "control_overlap": 0.05,
      "recovery_latency": 0.0,
      "scramble_entropy": 0.15
    },
    {
      "t": 1,
      "distance_delta": -0.45,
      "forward_velocity": 0.78,
      "lateral_displacement": 0.12,
      "vertical_level_change": -0.05,
      "contact_duration": 0.23,
      "control_overlap": 0.32,
      "recovery_latency": 0.18,
      "scramble_entropy": 0.67
    }
  ],
  "metadata": {
    "fps": 30.0,
    "frame_width": 1920,
    "frame_height": 1080,
    "total_seconds": 120,
    "signal_descriptions": { ... }
  }
}
```

**Signal Definitions:**

| Signal | Range | Meaning |
|--------|-------|---------|
| **t** | 0 to N | Time in seconds |
| **distance_delta** | -1 to +1 | Change in fighter separation. **-1** = closing rapidly, **+1** = separating rapidly |
| **forward_velocity** | -1 to +1 | Movement direction. **+1** = moving toward opponent, **-1** = moving away |
| **lateral_displacement** | 0 to 1 | Sideways movement. **0** = no lateral motion, **1** = maximum sideways motion |
| **vertical_level_change** | -1 to +1 | Vertical position change. **+1** = standing up, **-1** = going to ground |
| **contact_duration** | 0 to 1 | Time in clinch range this second. **0.5** = 0.5 seconds of contact |
| **control_overlap** | 0 to 1 | Bounding box overlap. **0** = no overlap, **1** = complete overlap |
| **recovery_latency** | 0 to 1 | Time to stabilize after rapid movement. **0** = instant, **1** = full second to recover |
| **scramble_entropy** | 0 to 1 | Movement chaos. **0** = smooth/predictable, **1** = erratic/chaotic |

**Important Notes:**
- All values are **normalized and relative**
- They do NOT represent real-world units (meters, km/h, etc.)
- Compare values within the same video or across similar videos
- Values are computed from bounding box positions only

---

## Advanced Usage

### Custom Output Location

Specify a different output folder:

```bash
python tests/test_signals.py Videos/fight.mp4 my_custom_output
```

Results will be saved to `my_custom_output/` instead of `output/`.

### Adjust Processing Speed

Control the playback speed during processing (does not affect results):

```bash
# Normal speed (1x) - watch tracking in real-time
python tests/test_signals.py Videos/fight.mp4 output 1.0

# Fast playback (2x speed) - process faster
python tests/test_signals.py Videos/fight.mp4 output 2.0

# Maximum speed (no delay) - fastest processing
python tests/test_signals.py Videos/fight.mp4 output 0
```

**Note:** Higher speed means less visual feedback but faster completion.

### Using Absolute Paths

You can use videos from any location:

**Windows:**
```bash
python tests/test_signals.py C:\Users\username\Desktop\fight_video.mp4
```

**macOS/Linux:**
```bash
python tests/test_signals.py /Users/username/Desktop/fight_video.mp4
```

### Batch Processing Multiple Videos

Create a batch script:

**Windows (`batch_process.bat`):**
```batch
@echo off
python tests/test_signals.py Videos\fight1.mp4 output 0
python tests/test_signals.py Videos\fight2.mp4 output 0
python tests/test_signals.py Videos\fight3.mp4 output 0
echo All videos processed!
pause
```

**macOS/Linux (`batch_process.sh`):**
```bash
#!/bin/bash
python tests/test_signals.py Videos/fight1.mp4 output 0
python tests/test_signals.py Videos/fight2.mp4 output 0
python tests/test_signals.py Videos/fight3.mp4 output 0
echo "All videos processed!"
```

Run with:
```bash
# Windows
batch_process.bat

# macOS/Linux
chmod +x batch_process.sh
./batch_process.sh
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'cv2'"

**Cause:** OpenCV not installed correctly.

**Solution:**
```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.11.0.86
```

### Problem: "Video file not found"

**Cause:** Incorrect path or video not in correct location.

**Solution 1 - Check current directory:**
```bash
# Show where you are
pwd

# List videos folder
ls Videos/       # macOS/Linux
dir Videos\      # Windows
```

**Solution 2 - Use absolute path:**
```bash
python tests/test_signals.py C:\full\path\to\video.mp4
```

**Solution 3 - Check file extension:**
```bash
# Wrong (no extension)
python tests/test_signals.py Videos/fight

# Correct (with extension)
python tests/test_signals.py Videos/fight.mp4
```

### Problem: Selection window doesn't appear

**Cause:** Display issues or OpenCV GUI backend problem.

**Solution for Linux:**
```bash
# Install display dependencies
sudo apt-get install python3-tk
sudo apt-get install libgl1-mesa-glx

# Set display variable
export DISPLAY=:0

# Try again
python tests/test_signals.py Videos/video.mp4
```

**Solution for macOS (if using SSH):**
```bash
# OpenCV requires direct display access
# Do not run over SSH without X11 forwarding
# Run directly on the machine
```

**Solution for Windows:**
```bash
# Ensure no other program is blocking the display
# Close screen sharing software (TeamViewer, AnyDesk)
# Run directly on the machine, not via RDP
```

### Problem: "Segmentation fault" or crash during processing

**Cause:** Corrupted video file or incompatible codec.

**Solution 1 - Re-encode video:**
```bash
# Install ffmpeg (if not installed)
# Then re-encode:
ffmpeg -i original.mp4 -c:v libx264 -preset fast -crf 23 output.mp4
```

**Solution 2 - Try different video:**
```bash
# Test with a known-good video first
# If that works, original video has issues
```


**Tips for good selection:**

✅ **Good box size:**
- Should cover fighter's torso and head
- Can include upper legs
- Small amount of background is okay

 **Box too small:**
- Only covers head
- Too tight around body
- Will lose tracking easily

 **Box too large:**
- Includes other fighter
- Includes large background area
- Will confuse the tracker

**If you made a mistake:**
- Press **R** to remove the last box and try again
- Press **ESC** to cancel and restart the entire process

### Problem: Wrong fighter was selected

**Cause:** Pressed wrong number (1 or 2) during selection.

**Solution:** You must restart:
1. Press **Ctrl+C** to stop the current run
2. Run the script again: `python tests/test_signals.py Videos/video.mp4`
3. Draw the boxes again
4. Press the correct number this time

### Problem: "Permission denied" when saving output

**Cause:** Output folder is write-protected or open in another program.

**Solution 1 - Close programs:**
- Close any programs viewing the output folder
- Close JSON viewers or text editors with output files open

**Solution 2 - Change output location:**
```bash
python tests/test_signals.py Videos/video.mp4 C:\Temp\output
```

**Solution 3 - Check permissions:**
```bash
# Linux/macOS
chmod -R 755 output/

# Windows: Right-click output folder → Properties → Security → Ensure Write permission
```

### Problem: Virtual environment won't activate

**Windows - "Scripts\activate.bat" not found:**
```bash
# Recreate virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux - Permission denied:**
```bash
# Make activate script executable
chmod +x venv/bin/activate
source venv/bin/activate
```

---

## Understanding Results

### Interpreting Presence Zones

**Example scenario:**

```json
"my_fighter_presence": [
  { "start": 0.0, "end": 87.5 },
  { "start": 102.3, "end": 180.0 }
]
```

**What this means:**
- Fighter visible from 0:00 to 1:27.5 (87.5 seconds)
- Fighter NOT visible from 1:27.5 to 1:42.3 (14.8 seconds gap)
- Fighter visible again from 1:42.3 to 3:00 (77.7 seconds)

**Common reasons for gaps:**
- Fighter moved off-screen
- Camera cut to different angle
- Referee or corner blocked view
- Fighters went to ground (camera lost tracking)
- Extreme camera shake

### Interpreting Behavioral Signals

**Example high-activity moment:**

```json
{
  "t": 45,
  "distance_delta": -0.62,
  "forward_velocity": 0.85,
  "lateral_displacement": 0.23,
  "vertical_level_change": -0.15,
  "contact_duration": 0.78,
  "control_overlap": 0.45,
  "recovery_latency": 0.32,
  "scramble_entropy": 0.89
}
```

**Translation:**
- At second 45 of the video:
  - Fighters closing distance rapidly (`distance_delta: -0.62`)

