"""
MMA Fighter Analysis Pipeline - Test Runner
Updated: January 2026 - Guaranteed to work

Run from project root:
    cd mma_fighter_analysis
    python tests/test_signals.py Videos/VED.mp4
"""

import sys
import os
import cv2
import traceback
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# ROBUST PATH SETUP - Works regardless of how Python is called
# ══════════════════════════════════════════════════════════════════════════════

# Get absolute path to this script
script_path = Path(__file__).resolve()
tests_dir = script_path.parent
project_root = tests_dir.parent
core_dir = project_root / "app" / "core"

# Debug: Show paths
print("=" * 80)
print("PATH CONFIGURATION")
print("=" * 80)
print(f"Script:       {script_path}")
print(f"Tests dir:    {tests_dir}")
print(f"Project root: {project_root}")
print(f"Core dir:     {core_dir}")
print(f"Core exists:  {core_dir.exists()}")
print("=" * 80 + "\n")

# Add paths to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(core_dir))

# Change working directory to project root for relative paths
os.chdir(str(project_root))

# ══════════════════════════════════════════════════════════════════════════════
# TRY MULTIPLE IMPORT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

VideoLoader = None
quick_select_fighters = None
FighterTracker = None
PresenceZoneTracker = None

# Strategy 1: Try package imports
try:
    from ..app.core.video_loader import VideoLoader
    from ..app.core.selector import quick_select_fighters
    from ..app.core.tracker import FighterTracker
    from ..app.core.presence_zones import PresenceZoneTracker
    print("✓ Imports successful using: from ..app.core.module import Class\n")
except ImportError as e1:
    print(f"Strategy 1 failed: {e1}")
    
    # Strategy 2: Try direct imports (core in path)
    try:
        from ..app.core.video_loader import VideoLoader
        from ..app.core.selector import quick_select_fighters
        from ..app.core.tracker import FighterTracker
        from ..app.core.presence_zones import PresenceZoneTracker
        print("✓ Imports successful using: from ..app.core.module import Class\n")
    except ImportError as e2:
        print(f"Strategy 2 failed: {e2}")
        
        # Strategy 3: Manual file imports
        try:
            import importlib.util
            
            def load_module_from_file(module_name, file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            
            # Load each module
            video_loader_mod = load_module_from_file("video_loader", core_dir / "video_loader.py")
            selector_mod = load_module_from_file("selector", core_dir / "selector.py")
            tracker_mod = load_module_from_file("tracker", core_dir / "tracker.py")
            presence_mod = load_module_from_file("presence_zones", core_dir / "presence_zones.py")
            
            VideoLoader = video_loader_mod.VideoLoader
            quick_select_fighters = selector_mod.quick_select_fighters
            FighterTracker = tracker_mod.FighterTracker
            PresenceZoneTracker = presence_mod.PresenceZoneTracker
            
            print("✓ Imports successful using: manual file loading\n")
        except Exception as e3:
            print("\n" + "=" * 80)
            print("❌ ALL IMPORT STRATEGIES FAILED")
            print("=" * 80)
            print(f"\nStrategy 1 error: {e1}")
            print(f"Strategy 2 error: {e2}")
            print(f"Strategy 3 error: {e3}")
            print("\nPlease verify these files exist:")
            print(f"  {core_dir / 'video_loader.py'}")
            print(f"  {core_dir / 'selector.py'}")
            print(f"  {core_dir / 'tracker.py'}")
            print(f"  {core_dir / 'presence_zones.py'}")
            print("\nFiles found in core/:")
            try:
                for f in core_dir.glob("*.py"):
                    print(f"  - {f.name}")
            except:
                print("  (could not list files)")
            print("=" * 80 + "\n")
            sys.exit(1)

# Verify all imports succeeded
if not all([VideoLoader, quick_select_fighters, FighterTracker, PresenceZoneTracker]):
    print("❌ Some imports are None - this should not happen!")
    sys.exit(1)


def run_pipeline(video_path_str: str, output_folder: str = "output", playback_speed: float = 1.0):
    """
    Run the complete MMA fighter analysis pipeline.
    
    Args:
        video_path_str: Path to video file (relative or absolute)
        output_folder: Output directory for results
        playback_speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x speed, 0 = max speed)
    """
    # Handle both relative and absolute paths
    video_path = Path(video_path_str)
    
    # If relative path, resolve it from current working directory
    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path
    
    video_path = video_path.resolve()
    
    if not video_path.is_file():
        print(f"❌ Video file not found: {video_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Looking for: {video_path_str}")
        return

    video_id = video_path.stem
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 80)
    print(" MMA FIGHTER ANALYSIS PIPELINE ")
    print("═" * 80)
    print(f" Video:   {video_path.name}")
    print(f" Path:    {video_path}")
    print(f" Output:  {output_dir.resolve()}")
    print(f" Speed:   {playback_speed}x")
    print("═" * 80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Video
    # ═══════════════════════════════════════════════════════════════════════
    print("→ STEP 1: Loading video...")
    try:
        loader = VideoLoader(str(video_path))
    except Exception as e:
        print(f"❌ Failed to load video: {e}")
        traceback.print_exc()
        return

    info = loader.get_video_info()
    print(f"  • Resolution: {info['width']}×{info['height']}")
    print(f"  • FPS:        {info['fps']:.2f}")
    print(f"  • Duration:   {info['duration_formatted']} ({info['total_frames']} frames)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Manual Fighter Selection
    # ═══════════════════════════════════════════════════════════════════════
    print("→ STEP 2: Manual fighter selection")
    print("  Instructions:")
    print("    1. Click & drag to draw box around Fighter 1")
    print("    2. Press SPACE to confirm")
    print("    3. Draw box around Fighter 2")
    print("    4. Press SPACE to confirm")
    print("    5. Press 1 or 2 to select YOUR fighter\n")

    try:
        first_frame = loader.get_first_frame()
        roles = quick_select_fighters(first_frame, num_fighters=2)

        my_bbox = roles["my_fighter"]
        opp_bbox = roles.get("opponent")
        
        print(f"\n✅ Fighters selected:")
        print(f"   My Fighter:  {my_bbox}")
        print(f"   Opponent:    {opp_bbox}\n")
        
    except Exception as e:
        print(f"❌ Selection cancelled or failed: {e}")
        loader.close()
        return

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Initialize Trackers
    # ═══════════════════════════════════════════════════════════════════════
    print("→ STEP 3: Initializing trackers...")
    
    tracker = FighterTracker(
        frame_width=info['width'],
        frame_height=info['height'],
        fps=info['fps']
    )
    tracker.initialize(first_frame, my_bbox, opp_bbox)

    presence = PresenceZoneTracker(
        fps=info['fps'],
        merge_gap_seconds=1.5,
        presence_threshold=0.6  # Fix 6: Stricter threshold for presence JSON
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Process Video Frames
    # ═══════════════════════════════════════════════════════════════════════
    print("\n→ STEP 4: Processing video frames...")
    print("  (Press Ctrl+C or 'Q' key in video window to stop early)\n")
    
    loader.reset()
    frame_idx = 0
    total = info['total_frames']
    progress_interval = max(1, total // 50)  # Show ~50 progress updates

    try:
        while frame_idx < total:
            frame = loader.get_frame(frame_idx)
            if frame is None:
                print(f"\n⚠️  Warning: frame {frame_idx} could not be read")
                break

            # Update trackers
            tracker.update(frame)
            status = tracker.get_status()
            presence.update(status)

            # Draw bounding boxes on frame for visualization
            display_frame = frame.copy()
            if status['my_fighter']['bbox'] is not None:
                x, y, w, h = status['my_fighter']['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for my fighter
                cv2.putText(display_frame, "MY FIGHTER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if status['opponent']['bbox'] is not None:
                x, y, w, h = status['opponent']['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for opponent
                cv2.putText(display_frame, "OPPONENT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Add status info
            cv2.putText(display_frame, f"Frame: {frame_idx + 1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"My: {'✓' if status['my_fighter']['reliable'] else '✗'} Opp: {'✓' if status['opponent']['reliable'] else '✗'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize frame for better display (maintain aspect ratio)
            height, width = display_frame.shape[:2]
            max_display_width = 1280
            max_display_height = 720
            
            if width > max_display_width or height > max_display_height:
                # Calculate scaling factor to fit within max dimensions
                scale_w = max_display_width / width
                scale_h = max_display_height / height
                scale = min(scale_w, scale_h)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Show the frame
            cv2.imshow('MMA Fighter Tracking', display_frame)
            
            # Check for quit key (with adjustable delay for video playback)
            if playback_speed == 0:
                delay_ms = 1  # Max speed
            else:
                delay_ms = int(30 / playback_speed)  # Scale delay by speed factor
            
            key = cv2.waitKey(max(1, delay_ms)) & 0xFF  # Minimum 1ms delay
            if key == ord('q'):
                print("\n\n⚠️  Processing stopped by user (Q key)")
                break

            # Progress display
            if frame_idx % progress_interval == 0 or frame_idx == total - 1:
                perc = (frame_idx + 1) / total * 100
                my_ok = status['my_fighter']['reliable']
                opp_ok = status['opponent']['reliable']
                
                bar_length = 40
                filled = int(bar_length * (frame_idx + 1) / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f"  [{bar}] {perc:5.1f}%  "
                      f"Frame {frame_idx + 1:6d}/{total}  "
                      f"My: {'✓' if my_ok else '✗'}  Opp: {'✓' if opp_ok else '✗'}",
                      end='\r')

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing stopped by user (Ctrl+C)")

    print(f"\n\n✅ Processed {frame_idx} frames")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Finalize and Display Results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n→ STEP 5: Finalizing presence zones...")
    presence.finalize()
    presence.print_summary()

    # Export results
    json_path = output_dir / f"{video_id}_presence_zones.json"
    presence.export_json(video_id, str(json_path), include_metadata=True)
    
    print(f"\n📁 Results saved to: {json_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print(" PIPELINE COMPLETED SUCCESSFULLY ✓ ")
    print("═" * 80 + "\n")

    loader.close()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\n" + "═" * 80)
        print(" MMA FIGHTER ANALYSIS - USAGE ")
        print("═" * 80)
        print("\nRun from project root (mma_fighter_analysis/):\n")
        print("  python tests/test_signals.py Videos/VED.mp4")
        print("\nWith custom output folder:")
        print("  python tests/test_signals.py Videos/VED.mp4 custom_output")
        print("\nWith playback speed control:")
        print("  python tests/test_signals.py Videos/VED.mp4 output 1.0    # Normal speed")
        print("  python tests/test_signals.py Videos/VED.mp4 output 2.0    # 2x speed")
        print("  python tests/test_signals.py Videos/VED.mp4 output 0.5    # Half speed")
        print("  python tests/test_signals.py Videos/VED.mp4 output 0      # Max speed (no delay)")
        print("\nWith full path:")
        print("  python tests/test_signals.py C:/path/to/video.mp4")
        print("\n" + "═" * 80 + "\n")
        sys.exit(1)

    video_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) >= 3 else "output"
    
    # Parse speed argument (default 1.0 = normal speed)
    speed_arg = 1.0
    if len(sys.argv) >= 4:
        try:
            speed_arg = float(sys.argv[3])
            if speed_arg < 0:
                speed_arg = 0  # Max speed
        except ValueError:
            print(f"Warning: Invalid speed '{sys.argv[3]}', using default 1.0")
            speed_arg = 1.0

    try:
        run_pipeline(video_arg, output_arg, speed_arg)
    except Exception as e:
        print("\n" + "═" * 80)
        print(" PIPELINE ERROR ")
        print("═" * 80)
        traceback.print_exc()
        print("═" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()