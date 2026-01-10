"""
Test script for video loading, fighter selection, tracking, and presence zones.
Tests the complete pipeline:
Load video → Select fighters → Track through video → Extract presence zones
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json

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


# ------------------- TEST 4: ADAPTIVE FIGHTER TRACKING ------------------- #
def test_tracking(video_path: str, fighters: dict):
    """
    Test 4: Adaptive fighter tracking through video.

    Args:
        video_path: Path to video file
        fighters: Dictionary with my_fighter and opponent bboxes
    """
    print("\n" + "=" * 60)
    print("TEST 4: ADAPTIVE FIGHTER TRACKING")
    print("=" * 60)

    try:
        from mma_fighter_analysis.app.core.tracker import track_video

        print("\n🎯 Tracking with adaptive multi-tracker ensemble...")
        print("   📺 Visual display enabled")
        print("   ⌨️  Press 'Q' to quit, 'P' to pause\n")

        tracker = track_video(
            video_path=video_path,
            my_fighter_bbox=fighters["my_fighter"],
            opponent_bbox=fighters.get("opponent"),
            progress_interval=100,
            show_video=True,
            save_video=False
        )

        print("\n✅ Tracking test successful!")
        return tracker

    except Exception as e:
        print(f"\n❌ Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------- TEST 5: PRESENCE ZONE EXTRACTION ------------------- #
def test_presence_zones(video_path: str, tracker):
    """
    Test 5: Extract presence zones from tracking results.

    Args:
        video_path: Path to video file
        tracker: DualFighterTracker instance
    """
    print("\n" + "=" * 60)
    print("TEST 5: PRESENCE ZONE EXTRACTION")
    print("=" * 60)

    try:
        from mma_fighter_analysis.app.core.presence_zones import (
            extract_presence_zones_from_tracker,
            save_presence_zones
        )

        # Get video info for duration
        video = VideoLoader(video_path)
        video_info = video.get_video_info()
        video.close()

        print("\n🎯 Extracting presence zones from tracking history...")

        # Extract zones
        zones_data = extract_presence_zones_from_tracker(
            tracker=tracker,
            video_id=Path(video_path).stem,
            video_duration=video_info["duration_seconds"],
            gap_threshold=2.0,   # Merge zones if gap < 2 seconds
            min_duration=1.0     # Minimum zone duration = 1 second
        )

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Save zones to JSON
        output_path = output_dir / f"{Path(video_path).stem}_presence_zones.json"
        save_presence_zones(zones_data, str(output_path))

        # Print JSON preview
        print("\n" + "=" * 60)
        print("📄 PRESENCE ZONES JSON OUTPUT (My Fighter)")
        print("=" * 60)
        print(json.dumps(zones_data["my_fighter"], indent=2))
        
        if zones_data["opponent"]["my_fighter_presence"]:
            print("\n" + "=" * 60)
            print("📄 PRESENCE ZONES JSON OUTPUT (Opponent)")
            print("=" * 60)
            print(json.dumps(zones_data["opponent"], indent=2))

        print("\n✅ Presence zone extraction successful!")
        return zones_data

    except Exception as e:
        print(f"\n❌ Presence zone extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------- TEST 6: BEHAVIORAL SIGNALS ------------------- #
def test_behavioral_signals(video_path: str, tracker):
    """
    Test 6: Extract behavioral signals from tracking results.

    Args:
        video_path: Path to video file
        tracker: DualFighterTracker instance
    """
    print("\n" + "=" * 60)
    print("TEST 6: BEHAVIORAL SIGNAL EXTRACTION")
    print("=" * 60)

    try:
        from mma_fighter_analysis.app.core.behavioral_signals import (
            extract_behavioral_signals_from_tracker,
            save_behavioral_signals
        )

        # Get video info
        video = VideoLoader(video_path)
        video_info = video.get_video_info()
        video.close()

        print("\n🎯 Extracting behavioral signals...")

        # Extract signals
        signals_data = extract_behavioral_signals_from_tracker(
            tracker=tracker,
            video_id=Path(video_path).stem,
            fps=video_info["fps"],
            frame_width=video_info["width"],
            frame_height=video_info["height"]
        )

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Save signals to JSON
        output_path = output_dir / f"{Path(video_path).stem}_behavioral_signals.json"
        save_behavioral_signals(signals_data, str(output_path))

        # Print sample signals
        if signals_data["signals"]:
            print("\n" + "=" * 60)
            print("📄 BEHAVIORAL SIGNALS SAMPLE (First 3 seconds)")
            print("=" * 60)
            sample = signals_data["signals"][:3]
            print(json.dumps({"video_id": signals_data["video_id"], "signals": sample}, indent=2))

        print("\n✅ Behavioral signal extraction successful!")
        return signals_data

    except Exception as e:
        print(f"\n❌ Behavioral signal extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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

    zones = test_presence_zones(video_path, tracker)
    if zones is None:
        return False

    signals = test_behavioral_signals(video_path, tracker)
    if signals is None:
        return False

    print("\n🎉 ALL TESTS PASSED!")
    print("\n" + "=" * 60)
    print("✅ DELIVERABLES GENERATED")
    print("=" * 60)
    print(f"📁 Presence Zones JSON:     output/{Path(video_path).stem}_presence_zones.json")
    print(f"📁 Behavioral Signals JSON: output/{Path(video_path).stem}_behavioral_signals.json")
    print("=" * 60)
    
    return True


# ------------------- MAIN EXECUTION ------------------- #
def main():
    print("\n" + "═" * 60)
    print("🥊 MMA VIDEO ANALYSIS - COMPLETE PIPELINE TEST")
    print("═" * 60)

    if len(sys.argv) < 2:
        print("\n❌ ERROR: No video path provided")
        print("\nUsage:")
        print("  python mma_fighter_analysis/tests/test_signals.py <video_path>")
        print("\nExample:")
        print("  python mma_fighter_analysis/tests/test_signals.py mma_fighter_analysis/Videos/testfight.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"\n❌ Video not found: {video_path}")
        sys.exit(1)

    success = test_complete_pipeline(video_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()