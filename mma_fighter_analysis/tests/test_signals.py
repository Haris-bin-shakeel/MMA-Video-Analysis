
import sys
import os
import cv2
import traceback
from pathlib import Path


script_path = Path(__file__).resolve()
tests_dir = script_path.parent
project_root = tests_dir.parent
core_dir = project_root / "app" / "core"

print("=" * 80)
print("PATH CONFIGURATION")
print("=" * 80)
print(f"Script:       {script_path}")
print(f"Tests dir:    {tests_dir}")
print(f"Project root: {project_root}")
print(f"Core dir:     {core_dir}")
print(f"Core exists:  {core_dir.exists()}")
print("=" * 80 + "\n")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

# Change working directory to project root for relative paths
os.chdir(str(project_root))

print("Python path entries:")
for idx, p in enumerate(sys.path[:5]):
    print(f"  [{idx}] {p}")
print()


VideoLoader = None
quick_select_fighters = None
FighterTracker = None
PresenceZoneTracker = None
BehavioralSignalExtractor = None

import_success = False
import_errors = []

try:
    print("Attempting Strategy 1: from app.core import *")
    from app.core import (
        VideoLoader,
        quick_select_fighters,
        FighterTracker,
        PresenceZoneTracker,
        BehavioralSignalExtractor
    )
    print("✓ Strategy 1 successful!\n")
    import_success = True
except ImportError as e1:
    import_errors.append(("Strategy 1", str(e1)))
    print(f"✗ Strategy 1 failed: {e1}\n")

if not import_success:
    try:
        print("Attempting Strategy 2: from app.core.module import Class")
        from app.core.video_loader import VideoLoader
        from app.core.selector import quick_select_fighters
        from app.core.tracker import FighterTracker
        from app.core.presence_zones import PresenceZoneTracker
        from app.core.behavioral_signals import BehavioralSignalExtractor
        print("✓ Strategy 2 successful!\n")
        import_success = True
    except ImportError as e2:
        import_errors.append(("Strategy 2", str(e2)))
        print(f"✗ Strategy 2 failed: {e2}\n")

if not import_success:
    try:
        print("Attempting Strategy 3: from module import Class")
        from video_loader import VideoLoader
        from selector import quick_select_fighters
        from tracker import FighterTracker
        from presence_zones import PresenceZoneTracker
        from behavioral_signals import BehavioralSignalExtractor
        print("✓ Strategy 3 successful!\n")
        import_success = True
    except ImportError as e3:
        import_errors.append(("Strategy 3", str(e3)))
        print(f"✗ Strategy 3 failed: {e3}\n")

if not import_success:
    try:
        print("Attempting Strategy 4: Manual file loading with importlib")
        import importlib.util
        
        def load_module_from_file(module_name, file_path):
            """Load a module from a file path."""
            if not file_path.exists():
                raise FileNotFoundError(f"Module file not found: {file_path}")
            
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        
        print(f"  Loading video_loader from {core_dir / 'video_loader.py'}")
        video_loader_mod = load_module_from_file("video_loader", core_dir / "video_loader.py")
        
        print(f"  Loading selector from {core_dir / 'selector.py'}")
        selector_mod = load_module_from_file("selector", core_dir / "selector.py")
        
        print(f"  Loading tracker from {core_dir / 'tracker.py'}")
        tracker_mod = load_module_from_file("tracker", core_dir / "tracker.py")
        
        print(f"  Loading presence_zones from {core_dir / 'presence_zones.py'}")
        presence_mod = load_module_from_file("presence_zones", core_dir / "presence_zones.py")
        
        print(f"  Loading behavioral_signals from {core_dir / 'behavioral_signals.py'}")
        signals_mod = load_module_from_file("behavioral_signals", core_dir / "behavioral_signals.py")
        
        VideoLoader = video_loader_mod.VideoLoader
        quick_select_fighters = selector_mod.quick_select_fighters
        FighterTracker = tracker_mod.FighterTracker
        PresenceZoneTracker = presence_mod.PresenceZoneTracker
        BehavioralSignalExtractor = signals_mod.BehavioralSignalExtractor
        
        print("✓ Strategy 4 successful!\n")
        import_success = True
    except Exception as e4:
        import_errors.append(("Strategy 4", str(e4)))
        print(f"✗ Strategy 4 failed: {e4}\n")

if not import_success:
    print("\n" + "=" * 80)
    print("❌ ALL IMPORT STRATEGIES FAILED")
    print("=" * 80)
    
    for strategy_name, error in import_errors:
        print(f"\n{strategy_name} error:")
        print(f"  {error}")
    
    print("\n" + "-" * 80)
    print("DIAGNOSTIC INFORMATION:")
    print("-" * 80)
    
    print(f"\nProject structure check:")
    print(f"  Project root exists: {project_root.exists()}")
    print(f"  Core directory exists: {core_dir.exists()}")
    
    if core_dir.exists():
        print(f"\n  Files in core directory:")
        try:
            for f in sorted(core_dir.glob("*.py")):
                print(f"    ✓ {f.name}")
        except Exception as e:
            print(f"    ✗ Could not list files: {e}")
    else:
        print(f"\n  ✗ Core directory not found at: {core_dir}")
        print(f"  Looking for app/core/ structure...")
        
        app_dir = project_root / "app"
        if app_dir.exists():
            print(f"    ✓ Found app/ directory")
            for item in app_dir.iterdir():
                print(f"      - {item.name}")
        else:
            print(f"    ✗ No app/ directory found")
    
    print(f"\n  Required files:")
    required_files = [
        "video_loader.py",
        "selector.py", 
        "tracker.py",
        "presence_zones.py",
        "behavioral_signals.py"
    ]
    
    for filename in required_files:
        filepath = core_dir / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"    {exists} {filename}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS:")
    print("=" * 80)
    print("1. Verify you're running from the project root directory:")
    print(f"   cd {project_root}")
    print("2. Ensure the project structure is:")
    print("   mma_fighter_analysis/")
    print("   ├── app/")
    print("   │   └── core/")
    print("   │       ├── video_loader.py")
    print("   │       ├── selector.py")
    print("   │       ├── tracker.py")
    print("   │       ├── presence_zones.py")
    print("   │       └── behavioral_signals.py")
    print("   └── tests/")
    print("       └── test_signals.py")
    print("3. Check file permissions and ensure files are readable")
    print("=" * 80 + "\n")
    
    sys.exit(1)

if not all([VideoLoader, quick_select_fighters, FighterTracker, PresenceZoneTracker, BehavioralSignalExtractor]):
    print("=" * 80)
    print(" IMPORT VERIFICATION FAILED")
    print("=" * 80)
    print("Some imports are None after successful strategy!")
    print(f"  VideoLoader: {VideoLoader is not None}")
    print(f"  quick_select_fighters: {quick_select_fighters is not None}")
    print(f"  FighterTracker: {FighterTracker is not None}")
    print(f"  PresenceZoneTracker: {PresenceZoneTracker is not None}")
    print(f"  BehavioralSignalExtractor: {BehavioralSignalExtractor is not None}")
    print("=" * 80 + "\n")
    sys.exit(1)

print("=" * 80)
print("✓ ALL MODULES IMPORTED SUCCESSFULLY")
print("=" * 80 + "\n")


def run_pipeline(video_path_str: str, output_folder: str = "output", playback_speed: float = 1.0):
    """
    Run the complete MMA fighter analysis pipeline.
    
    Args:
        video_path_str: Path to video file (relative or absolute)
        output_folder: Output directory for results
        playback_speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x speed, 0 = max speed)
    """
    video_path = Path(video_path_str)
    
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

   
    print("→ STEP 1: Loading video...")
    try:
        loader = VideoLoader(str(video_path))
    except Exception as e:
        print(f" Failed to load video: {e}")
        traceback.print_exc()
        return

    info = loader.get_video_info()
    print(f"  • Resolution: {info['width']}×{info['height']}")
    print(f"  • FPS:        {info['fps']:.2f}")
    print(f"  • Duration:   {info['duration_formatted']} ({info['total_frames']} frames)\n")

   
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


    print("→ STEP 3: Initializing trackers and signal extractors...")
    
    tracker = FighterTracker(
        frame_width=info['width'],
        frame_height=info['height'],
        fps=info['fps']
    )
    tracker.initialize(first_frame, my_bbox, opp_bbox)

    presence = PresenceZoneTracker(
        fps=info['fps'],
        merge_gap_seconds=1.5,
        presence_threshold=0.6
    )
    
    signals = BehavioralSignalExtractor(
        frame_width=info['width'],
        frame_height=info['height'],
        fps=info['fps']
    )

  
    print("\n→ STEP 4: Processing video frames...")
    print("  (Press Ctrl+C or 'Q' key in video window to stop early)\n")
    
    loader.reset()
    frame_idx = 0
    total = info['total_frames']
    progress_interval = max(1, total // 50)

    try:
        while frame_idx < total:
            frame = loader.get_frame(frame_idx)
            if frame is None:
                print(f"\n⚠️  Warning: frame {frame_idx} could not be read")
                break

            # Update trackers and signal extractor
            tracker.update(frame)
            status = tracker.get_status()
            presence.update(status)
            signals.update(status)

           
            display_frame = frame.copy()
            if status['my_fighter']['bbox'] is not None:
                x, y, w, h = status['my_fighter']['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, "MY FIGHTER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if status['opponent']['bbox'] is not None:
                x, y, w, h = status['opponent']['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(display_frame, "OPPONENT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

           
            cv2.putText(display_frame, f"Frame: {frame_idx + 1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"My: {'✓' if status['my_fighter']['reliable'] else '✗'} Opp: {'✓' if status['opponent']['reliable'] else '✗'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize frame for better display
            height, width = display_frame.shape[:2]
            max_display_width = 1280
            max_display_height = 720
            
            if width > max_display_width or height > max_display_height:
                scale_w = max_display_width / width
                scale_h = max_display_height / height
                scale = min(scale_w, scale_h)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Show the frame
            cv2.imshow('MMA Fighter Tracking', display_frame)
            
            # Check for quit key
            if playback_speed == 0:
                delay_ms = 1
            else:
                delay_ms = int(30 / playback_speed)
            
            key = cv2.waitKey(max(1, delay_ms)) & 0xFF
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

    
    print("\n→ STEP 5: Finalizing results...")
    presence.finalize()
    signals.finalize()
    
   
    presence.print_summary()
    signals.print_summary()

    
    presence_json_path = output_dir / f"{video_id}_presence_zones.json"
    presence.export_json(video_id, str(presence_json_path), include_metadata=True)
    
    signals_json_path = output_dir / f"{video_id}_behavioral_signals.json"
    signals.export_json(video_id, str(signals_json_path), include_metadata=True)
    
    print(f"\n📁 Results saved:")
    print(f"   • Presence zones:      {presence_json_path}")
    print(f"   • Behavioral signals:  {signals_json_path}")

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
        print("  python tests/test_signals.py Videos/VED.mp4 output 0      # Max speed")
        print("\nWith full path:")
        print("  python tests/test_signals.py C:/path/to/video.mp4")
        print("\n" + "═" * 80 + "\n")
        sys.exit(1)

    video_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) >= 3 else "output"
    
    speed_arg = 1.0
    if len(sys.argv) >= 4:
        try:
            speed_arg = float(sys.argv[3])
            if speed_arg < 0:
                speed_arg = 0
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
