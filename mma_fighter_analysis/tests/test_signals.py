"""
Test script for video loading, fighter selection, and robust tracking.
Tests the complete pipeline: Load video → Select fighters → Track through video
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from mma_fighter_analysis.app.core.video_loader import VideoLoader
from mma_fighter_analysis.app.core.selector import quick_select_fighters


# ----------------------- TEST 1: VIDEO LOADING ----------------------- #
def test_video_loading(video_path: str):
    print("\n" + "=" * 60)
    print("TEST 1: VIDEO LOADING")
    print("=" * 60)

    try:
        video = VideoLoader(video_path)
        info = video.get_video_info()

        print("\n✅ Video loaded successfully!")
        print(f"  Filename:     {info['filename']}")
        print(f"  Duration:     {info['duration_formatted']}")
        print(f"  Resolution:   {info['width']}x{info['height']}")
        print(f"  FPS:          {info['fps']:.2f}")
        print(f"  Total Frames: {info['total_frames']}")

        video.close()
        return True
    except Exception as e:
        print(f"\n❌ Video loading failed: {e}")
        return False


# ------------------- TEST 2: FIRST FRAME EXTRACTION ------------------- #
def test_first_frame_extraction(video_path: str):
    print("\n" + "=" * 60)
    print("TEST 2: FIRST FRAME EXTRACTION")
    print("=" * 60)

    try:
        video = VideoLoader(video_path)
        first_frame = video.get_first_frame()
        print(f"\n✅ First frame extracted! Shape: {first_frame.shape}, Type: {first_frame.dtype}")
        video.close()
        return first_frame
    except Exception as e:
        print(f"\n❌ First frame extraction failed: {e}")
        return None


# ------------------- TEST 3: MANUAL FIGHTER SELECTION ------------------- #
def test_fighter_selection(video_path: str):
    print("\n" + "=" * 60)
    print("TEST 3: MANUAL FIGHTER SELECTION WITH ROLE CONFIRMATION")
    print("=" * 60)

    try:
        video = VideoLoader(video_path)
        first_frame = video.get_first_frame()

        print("\n🎯 Starting interactive selection...")
        print("   A window will open - follow on-screen instructions\n")

        fighters = quick_select_fighters(first_frame, num_fighters=2)

        if fighters.get("my_fighter") and fighters.get("opponent"):
            print("\n✅ Fighter selection successful!")
            x, y, w, h = fighters["my_fighter"]
            print(f"🔴 MY FIGHTER: x={x}, y={y}, w={w}, h={h}")
            x, y, w, h = fighters["opponent"]
            print(f"🔵 OPPONENT:   x={x}, y={y}, w={w}, h={h}")
            video.close()
            return fighters

        elif fighters.get("my_fighter"):
            print("\n⚠️ Single fighter selected")
            video.close()
            return fighters

        else:
            video.close()
            return None
    except Exception as e:
        print(f"\n❌ Fighter selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------- TEST 4: ROBUST FIGHTER TRACKING ------------------- #
class RobustTracker:
    """Tracks two fighters with maximum reliability using CSRT → KCF → MOSSE."""
    
    def __init__(self, frame, my_bbox, opp_bbox=None):
        self.frame = frame
        self.my_bbox = self._clamp_bbox(my_bbox, frame)
        self.opp_bbox = self._clamp_bbox(opp_bbox, frame) if opp_bbox else None

        self.my_tracker = self._init_tracker(self.my_bbox, frame)
        self.opp_tracker = self._init_tracker(self.opp_bbox, frame) if self.opp_bbox else None

        self.my_pos = self.my_bbox
        self.opp_pos = self.opp_bbox

    @staticmethod
    def _clamp_bbox(bbox, frame):
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(10, min(w, w_frame - x))
        h = max(10, min(h, h_frame - y))
        return (x, y, w, h)

    @staticmethod
    def _init_tracker(bbox, frame):
        if bbox is None:
            return None
        trackers = [cv2.legacy.TrackerCSRT_create, cv2.legacy.TrackerKCF_create, cv2.legacy.TrackerMOSSE_create]
        for t_func in trackers:
            tracker = t_func()
            ok = tracker.init(frame, bbox)
            if ok:
                return tracker
        print("❌ Tracker failed to initialize")
        return None

    def update(self, frame):
        # Update My Fighter
        if self.my_tracker:
            ok, bbox = self.my_tracker.update(frame)
            if ok:
                self.my_pos = self._ema(self.my_pos, bbox)
        # Update Opponent
        if self.opp_tracker:
            ok, bbox = self.opp_tracker.update(frame)
            if ok:
                self.opp_pos = self._ema(self.opp_pos, bbox)
        return self.my_pos, self.opp_pos

    @staticmethod
    def _ema(prev, curr, alpha=0.3):
        """Exponential moving average for smooth tracking"""
        x = int(prev[0] * (1 - alpha) + curr[0] * alpha)
        y = int(prev[1] * (1 - alpha) + curr[1] * alpha)
        w = int(prev[2] * (1 - alpha) + curr[2] * alpha)
        h = int(prev[3] * (1 - alpha) + curr[3] * alpha)
        return (x, y, w, h)


def test_tracking(video_path: str, fighters: dict):
    print("\n" + "=" * 60)
    print("TEST 4: ROBUST FIGHTER TRACKING")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = None

    first_frame_ret, first_frame = cap.read()
    if not first_frame_ret:
        print("❌ Failed to read first frame")
        return None

    robust_tracker = RobustTracker(first_frame, fighters["my_fighter"], fighters.get("opponent"))

    frame_no = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        my_bbox, opp_bbox = robust_tracker.update(frame)

        # Draw boxes
        cv2.rectangle(frame, (my_bbox[0], my_bbox[1]), (my_bbox[0]+my_bbox[2], my_bbox[1]+my_bbox[3]), (0,0,255), 2)
        if opp_bbox:
            cv2.rectangle(frame, (opp_bbox[0], opp_bbox[1]), (opp_bbox[0]+opp_bbox[2], opp_bbox[1]+opp_bbox[3]), (255,255,0), 2)

        # Display
        cv2.putText(frame, f"Frame {frame_no}/{total_frames}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(-1)  # pause until key press

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Tracking completed successfully!")
    return robust_tracker


# ------------------- COMPLETE PIPELINE ------------------- #
def test_complete_pipeline(video_path: str):
    print("\n\n" + "🚀" * 20)
    print("COMPLETE PIPELINE TEST")
    print("🚀" * 20)

    if not test_video_loading(video_path):
        return False

    first_frame = test_first_frame_extraction(video_path)
    if first_frame is None:
        return False

    fighters = test_fighter_selection(video_path)
    if fighters is None or not fighters.get("my_fighter"):
        return False

    tracker = test_tracking(video_path, fighters)
    if tracker is None:
        return False

    print("\n🎉 ALL TESTS PASSED!")
    return True


# ------------------- MAIN EXECUTION ------------------- #
def main():
    print("\n" + "═" * 60)
    print("🥊 MMA VIDEO ANALYSIS - MODULE TESTS")
    print("═" * 60)

    if len(sys.argv) < 2:
        print("\n❌ ERROR: No video path provided")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"\n❌ Video not found: {video_path}")
        sys.exit(1)

    success = test_complete_pipeline(video_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
