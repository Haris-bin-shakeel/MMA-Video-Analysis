# Basic unit tests (optional)

"""
Test script for video loading and fighter selection.
Tests the complete pipeline: Load video → Extract first frame → Manual selection
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import sys
from pathlib import Path

# Add parent directory to path to import app modules

from mma_fighter_analysis.app.core.video_loader import VideoLoader
from mma_fighter_analysis.app.core.selector import FighterSelector, quick_select_fighters



def test_video_loading(video_path: str):
    """
    Test 1: Video loading functionality.
    
    Args:
        video_path: Path to video file
    """
    print("\n" + "=" * 60)
    print("TEST 1: VIDEO LOADING")
    print("=" * 60)
    
    try:
        video = VideoLoader(video_path)
        info = video.get_video_info()
        
        print("\n✅ Video loaded successfully!")
        print(f"\nVideo Information:")
        print(f"  Filename:    {info['filename']}")
        print(f"  Duration:    {info['duration_formatted']}")
        print(f"  Resolution:  {info['width']}x{info['height']}")
        print(f"  FPS:         {info['fps']:.2f}")
        print(f"  Total Frames: {info['total_frames']}")
        
        video.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Video loading failed: {e}")
        return False


def test_first_frame_extraction(video_path: str):
    """
    Test 2: First frame extraction.
    
    Args:
        video_path: Path to video file
    """
    print("\n" + "=" * 60)
    print("TEST 2: FIRST FRAME EXTRACTION")
    print("=" * 60)
    
    try:
        video = VideoLoader(video_path)
        first_frame = video.get_first_frame()
        
        print(f"\n✅ First frame extracted!")
        print(f"  Frame shape: {first_frame.shape}")
        print(f"  Frame type:  {first_frame.dtype}")
        
        video.close()
        return first_frame
        
    except Exception as e:
        print(f"\n❌ First frame extraction failed: {e}")
        return None


def test_fighter_selection(video_path: str):
    """
    Test 3: Manual fighter selection.
    
    Args:
        video_path: Path to video file
    """
    print("\n" + "=" * 60)
    print("TEST 3: MANUAL FIGHTER SELECTION")
    print("=" * 60)
    
    try:
        video = VideoLoader(video_path)
        first_frame = video.get_first_frame()
        
        print("\n🎯 Starting interactive selection...")
        print("   A window will open - follow on-screen instructions\n")
        
        # Use the selector
        bboxes = quick_select_fighters(first_frame, num_fighters=2)
        
        if len(bboxes) >= 2:
            print("\n✅ Fighter selection successful!")
            print(f"\nSelected Fighters:")
            for idx, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                print(f"  Fighter {idx + 1}:")
                print(f"    Position: (x={x}, y={y})")
                print(f"    Size:     (width={w}, height={h})")
            
            video.close()
            return bboxes
        else:
            print(f"\n⚠️  Only {len(bboxes)} fighter(s) selected (expected 2)")
            video.close()
            return bboxes
            
    except Exception as e:
        print(f"\n❌ Fighter selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_complete_pipeline(video_path: str):
    """
    Complete integration test: All modules together.
    
    Args:
        video_path: Path to video file
    """
    print("\n\n" + "🚀 " * 20)
    print("COMPLETE PIPELINE TEST")
    print("🚀 " * 20)
    
    # Test 1: Video Loading
    if not test_video_loading(video_path):
        print("\n❌ Pipeline failed at video loading")
        return False
    
    # Test 2: First Frame Extraction
    first_frame = test_first_frame_extraction(video_path)
    if first_frame is None:
        print("\n❌ Pipeline failed at frame extraction")
        return False
    
    # Test 3: Fighter Selection
    bboxes = test_fighter_selection(video_path)
    if bboxes is None or len(bboxes) < 1:
        print("\n❌ Pipeline failed at fighter selection")
        return False
    
    # Success summary
    print("\n\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\n✅ Video Loading:        PASS")
    print("✅ Frame Extraction:     PASS")
    print("✅ Fighter Selection:    PASS")
    print("\n📊 Pipeline ready for tracking implementation!")
    print("=" * 60 + "\n")
    
    return True


def main():
    """Main test execution."""
    print("\n" + "═" * 60)
    print("🥊 MMA VIDEO ANALYSIS - MODULE TESTS")
    print("═" * 60)
    
    # Check for video path argument
    if len(sys.argv) < 2:
        print("\n❌ ERROR: No video path provided")
        print("\n📖 USAGE:")
        print("  python test_signals.py <path_to_video>")
        print("\n📝 EXAMPLES:")
        print("  python test_signals.py my_fight.mp4")
        print("  python test_signals.py C:/Videos/ufc_match.mp4")
        print("  python test_signals.py ../videos/fight.mov")
        print("\n💡 TIP: Make sure your video file is:")
        print("  - In .mp4, .mov, .webm, .avi, or .mkv format")
        print("  - Readable and not corrupted")
        print("  - Accessible from current directory")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Validate file exists
    if not Path(video_path).exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print("\n💡 Check:")
        print("  - File path is correct")
        print("  - File extension is included")
        print("  - You're in the right directory")
        sys.exit(1)
    
    # Run complete pipeline test
    success = test_complete_pipeline(video_path)
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🚀 Ready to implement tracking module next!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
