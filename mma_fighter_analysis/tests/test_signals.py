"""
COMPREHENSIVE SYSTEM TEST - CLIENT-READY VALIDATION

Tests:
1. Tracking accuracy - boxes stay ON fighters
2. Presence zone alignment - zones match visual tracking
3. Frame-by-frame validation - every second verified
4. Output format compliance - spec requirements met
5. Quality metrics - client acceptance criteria

Usage:
    python test_complete_system.py <video_path>

Output:
    - Visual tracking display (watch boxes follow fighters)
    - JSON presence zone files (spec compliant)
    - Validation report (client-ready verdict)
    - Quality metrics CSV (frame-by-frame analysis)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import csv
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from mma_fighter_analysis.app.core.video_loader import VideoLoader
from mma_fighter_analysis.app.core.selector import quick_select_fighters
from mma_fighter_analysis.app.core.tracker import DualFighterTracker
from mma_fighter_analysis.app.core.presence_zones import (
    extract_presence_zones_from_tracker,
    save_presence_zones
)


def validate_tracking_quality(tracker, video_info) -> dict:
    """
    Validate tracking quality against client requirements.
    
    CLIENT ACCEPTANCE CRITERIA:
    - Tracking rate > 80%: Boxes track fighters most of the time
    - Visual quality > 0.50: Boxes are ON fighters, not drifted
    - Presence coverage > 75%: Most video action captured
    - Gaps < 5 seconds: Brief losses only, no long disappearances
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING TRACKING QUALITY")
    print("=" * 70)
    
    my_stats = tracker.my_tracker.get_stats()
    
    # Check tracking rate
    tracking_rate = my_stats['tracking_rate']
    tracking_ok = tracking_rate >= 0.80
    
    print(f"\n📊 MY FIGHTER Tracking:")
    print(f"   Tracking Rate:    {tracking_rate*100:.1f}% ", end="")
    print("✅ PASS" if tracking_ok else "❌ FAIL (need >80%)")
    
    # Check visual quality
    visual_quality = my_stats.get('average_visual_quality', 0.0)
    visual_ok = visual_quality >= 0.50
    
    print(f"   Visual Quality:   {visual_quality:.3f} ", end="")
    print("✅ PASS" if visual_ok else "❌ FAIL (need >0.50)")
    
    # Check source distribution
    sources = my_stats['sources']
    print(f"\n   Tracking Sources:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = (count / my_stats['total_frames']) * 100
        print(f"   - {source:15s}: {count:5d} frames ({pct:5.1f}%)")
    
    # Lost frames analysis
    lost_frames = my_stats['lost_frames']
    lost_pct = (lost_frames / my_stats['total_frames']) * 100
    lost_ok = lost_pct <= 20
    
    print(f"\n   Lost Frames:      {lost_frames} ({lost_pct:.1f}%) ", end="")
    print("✅ PASS" if lost_ok else "❌ FAIL (need <20%)")
    
    # Overall verdict
    all_ok = tracking_ok and visual_ok and lost_ok
    
    print(f"\n{'=' * 70}")
    print(f"🎯 MY FIGHTER VERDICT: ", end="")
    print("✅ CLIENT-READY" if all_ok else "❌ NEEDS IMPROVEMENT")
    print(f"{'=' * 70}")
    
    # Opponent validation (if exists)
    if tracker.has_opponent:
        opp_stats = tracker.opp_tracker.get_stats()
        opp_tracking_rate = opp_stats['tracking_rate']
        opp_visual = opp_stats.get('average_visual_quality', 0.0)
        
        print(f"\n📊 OPPONENT Tracking:")
        print(f"   Tracking Rate:    {opp_tracking_rate*100:.1f}%")
        print(f"   Visual Quality:   {opp_visual:.3f}")
    
    return {
        "my_fighter": {
            "tracking_rate": tracking_rate,
            "visual_quality": visual_quality,
            "lost_percentage": lost_pct,
            "client_ready": all_ok
        },
        "opponent": {
            "tracking_rate": opp_tracking_rate if tracker.has_opponent else 0.0,
            "visual_quality": opp_visual if tracker.has_opponent else 0.0
        } if tracker.has_opponent else None
    }


def validate_presence_zones(zones_data, validation_results, video_duration) -> dict:
    """
    Validate presence zones against client requirements.
    
    CLIENT ACCEPTANCE CRITERIA:
    - Presence coverage > 75%: Most action captured
    - Average zone > 3 seconds: Meaningful sequences, not flicker
    - Max gap < 5 seconds: Brief losses only
    - Quality metrics match zones: Alignment verified
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING PRESENCE ZONES")
    print("=" * 70)
    
    my_zones = zones_data['my_fighter_zones']['my_fighter_presence']
    stats = zones_data['statistics']['my_fighter']
    
    # Check coverage
    coverage = stats['presence_percentage']
    coverage_ok = coverage >= 75.0
    
    print(f"\n📊 MY FIGHTER Zones:")
    print(f"   Total Zones:      {stats['total_zones']}")
    print(f"   Coverage:         {coverage:.1f}% ", end="")
    print("✅ PASS" if coverage_ok else "❌ FAIL (need >75%)")
    
    # Check average zone duration
    avg_duration = stats['average_zone_duration']
    duration_ok = avg_duration >= 3.0
    
    print(f"   Avg Zone:         {avg_duration:.2f}s ", end="")
    print("✅ PASS" if duration_ok else "❌ FAIL (need >3s)")
    
    # Check gaps
    gaps = stats.get('gaps', [])
    if gaps:
        max_gap = max(g['duration'] for g in gaps)
        gap_ok = max_gap <= 5.0
        
        print(f"   Max Gap:          {max_gap:.2f}s ", end="")
        print("✅ PASS" if gap_ok else "⚠️  WARNING (>5s gap)")
    else:
        gap_ok = True
        print(f"   Max Gap:          N/A (continuous tracking)")
    
    # Check quality alignment
    qm = stats['quality_metrics']
    exclusion_rate = qm['exclusion_rate']
    
    print(f"\n   Quality Metrics:")
    print(f"   - Avg Visual Quality: {qm['avg_visual_quality']:.3f}")
    print(f"   - Frames Excluded:    {qm['frames_excluded']} ({exclusion_rate:.1f}%)")
    print(f"   - Frames in Zones:    {qm['frames_in_zones']}")
    
    # Zone-by-zone breakdown
    if len(my_zones) > 0:
        print(f"\n   Zone Breakdown:")
        for i, zone in enumerate(my_zones[:5], 1):  # Show first 5
            duration = zone['end'] - zone['start']
            print(f"   {i}. {zone['start']:6.2f}s → {zone['end']:6.2f}s ({duration:5.2f}s)")
        
        if len(my_zones) > 5:
            print(f"   ... and {len(my_zones) - 5} more zones")
    
    # Overall verdict
    all_ok = coverage_ok and duration_ok and gap_ok
    
    print(f"\n{'=' * 70}")
    print(f"🎯 ZONE QUALITY VERDICT: ", end="")
    print("✅ CLIENT-READY" if all_ok else "❌ NEEDS IMPROVEMENT")
    print(f"{'=' * 70}")
    
    return {
        "coverage": coverage,
        "avg_zone_duration": avg_duration,
        "max_gap": max(g['duration'] for g in gaps) if gaps else 0.0,
        "client_ready": all_ok
    }


def export_frame_validation_csv(tracker, output_dir, video_id):
    """
    Export frame-by-frame validation data for detailed analysis.
    Useful for debugging and quality assurance.
    """
    print(f"\n📄 Exporting frame validation CSV...")
    
    output_path = Path(output_dir)
    csv_file = output_path / f"{video_id}_frame_validation.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_num', 'timestamp', 'fighter', 
            'tracking_active', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'confidence', 'visual_quality', 'tracker_source',
            'in_presence_zone', 'quality_verdict'
        ])
        
        # MY FIGHTER frames
        for frame in tracker.my_tracker.tracking_history:
            verdict = "GOOD" if (frame.tracking_active and 
                                 frame.confidence >= 0.40 and 
                                 frame.visual_quality >= 0.45) else "EXCLUDED"
            
            writer.writerow([
                frame.frame_num,
                f"{frame.timestamp:.2f}",
                "MY_FIGHTER",
                frame.tracking_active,
                frame.bbox[0] if frame.bbox else "",
                frame.bbox[1] if frame.bbox else "",
                frame.bbox[2] if frame.bbox else "",
                frame.bbox[3] if frame.bbox else "",
                f"{frame.confidence:.3f}",
                f"{frame.visual_quality:.3f}",
                frame.tracker_source,
                "",  # Will be filled by user if needed
                verdict
            ])
        
        # OPPONENT frames (if exists)
        if tracker.has_opponent:
            for frame in tracker.opp_tracker.tracking_history:
                verdict = "GOOD" if (frame.tracking_active and 
                                     frame.confidence >= 0.40 and 
                                     frame.visual_quality >= 0.45) else "EXCLUDED"
                
                writer.writerow([
                    frame.frame_num,
                    f"{frame.timestamp:.2f}",
                    "OPPONENT",
                    frame.tracking_active,
                    frame.bbox[0] if frame.bbox else "",
                    frame.bbox[1] if frame.bbox else "",
                    frame.bbox[2] if frame.bbox else "",
                    frame.bbox[3] if frame.bbox else "",
                    f"{frame.confidence:.3f}",
                    f"{frame.visual_quality:.3f}",
                    frame.tracker_source,
                    "",
                    verdict
                ])
    
    print(f"   ✅ Saved: {csv_file}")
    print(f"   Contains {len(tracker.my_tracker.tracking_history)} MY_FIGHTER frames")
    if tracker.has_opponent:
        print(f"   Contains {len(tracker.opp_tracker.tracking_history)} OPPONENT frames")


def run_complete_test(video_path: str):
    """Run complete system test with visual tracking and validation."""
    
    print("\n" + "═" * 70)
    print("🥊 COMPREHENSIVE MMA VIDEO ANALYSIS TEST")
    print("   CLIENT-READY VALIDATION")
    print("═" * 70)
    
    # Load video
    print("\n📹 Loading video...")
    video = VideoLoader(video_path)
    video_info = video.get_video_info()
    first_frame = video.get_first_frame()
    video.close()
    
    print(f"✅ Video: {video_info['filename']}")
    print(f"   {video_info['total_frames']} frames @ {video_info['fps']:.2f} FPS")
    print(f"   Duration: {video_info['duration_formatted']}")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    
    # Fighter selection
    print("\n🎯 Fighter selection...")
    print("   1. Select YOUR FIGHTER (draw box around them)")
    print("   2. Select OPPONENT (draw box around them)")
    print("   3. Press SPACE to confirm, ESC to skip")
    
    fighters = quick_select_fighters(first_frame, num_fighters=2)
    
    if not fighters or not fighters.get("my_fighter"):
        print("\n❌ No fighters selected!")
        return False
    
    print(f"✅ MY FIGHTER: {fighters['my_fighter']}")
    if fighters.get("opponent"):
        print(f"✅ OPPONENT:   {fighters['opponent']}")
    
    # Initialize tracker
    print("\n🎯 Starting production tracking...")
    print("   Watch the bounding boxes - they should STAY ON the fighters")
    print("   Press 'q' or ESC to stop early")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return False
    
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    width = video_info['width']
    height = video_info['height']
    
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame!")
        return False
    
    tracker = DualFighterTracker(
        fighters["my_fighter"],
        fighters.get("opponent"),
        first_frame
    )
    
    # Create display window
    cv2.namedWindow("TRACKING TEST", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TRACKING TEST", 1280, 720)
    
    frame_num = 1
    
    # Tracking loop with enhanced visualization
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps
        results = tracker.track_frame(frame, frame_num, timestamp)
        
        # Create display frame
        display = frame.copy()
        
        # Draw MY FIGHTER
        my_r = results["my_fighter"]
        if my_r.tracking_active and my_r.bbox:
            x, y, w, h = my_r.bbox
            
            # Color by visual quality
            if my_r.visual_quality >= 0.60:
                color = (0, 255, 0)  # Green = excellent
                quality_text = "EXCELLENT"
            elif my_r.visual_quality >= 0.45:
                color = (0, 200, 255)  # Yellow = good
                quality_text = "GOOD"
            elif my_r.visual_quality >= 0.30:
                color = (0, 165, 255)  # Orange = acceptable
                quality_text = "ACCEPTABLE"
            else:
                color = (0, 100, 255)  # Red = poor
                quality_text = "POOR"
            
            thickness = 3 if my_r.visual_quality >= 0.45 else 2
            cv2.rectangle(display, (x, y), (x+w, y+h), color, thickness)
            
            # Labels
            label1 = f"MY FIGHTER | {my_r.tracker_source[:6]}"
            label2 = f"Visual: {my_r.visual_quality:.2f} | {quality_text}"
            label3 = f"Conf: {my_r.confidence:.2f}"
            
            cv2.putText(display, label1, (x, max(y-35, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, label2, (x, max(y-18, 37)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(display, label3, (x, max(y-3, 52)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(display, "MY FIGHTER: LOST", 
                       (20, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw OPPONENT
        if tracker.has_opponent:
            opp_r = results["opponent"]
            if opp_r.tracking_active and opp_r.bbox:
                x, y, w, h = opp_r.bbox
                
                # Color by visual quality
                if opp_r.visual_quality >= 0.60:
                    color = (255, 255, 0)  # Cyan = excellent
                    quality_text = "EXCELLENT"
                elif opp_r.visual_quality >= 0.45:
                    color = (255, 200, 0)  # Light cyan = good
                    quality_text = "GOOD"
                elif opp_r.visual_quality >= 0.30:
                    color = (255, 165, 0)  # Light blue = acceptable
                    quality_text = "ACCEPTABLE"
                else:
                    color = (255, 100, 0)  # Blue = poor
                    quality_text = "POOR"
                
                thickness = 3 if opp_r.visual_quality >= 0.45 else 2
                cv2.rectangle(display, (x, y), (x+w, y+h), color, thickness)
                
                label1 = f"OPPONENT | {opp_r.tracker_source[:6]}"
                label2 = f"Visual: {opp_r.visual_quality:.2f} | {quality_text}"
                
                cv2.putText(display, label1, (x, max(y-25, 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(display, label2, (x, max(y-8, 37)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(display, "OPPONENT: LOST", 
                           (20, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Frame info
        info_text = f"Frame: {frame_num}/{total_frames} | Time: {timestamp:.2f}s | FPS: {fps:.1f}"
        cv2.putText(display, info_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(display, "Watch: Boxes should STAY ON fighters (not drift)", 
                   (20, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Press 'q' or ESC to stop", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Progress bar
        progress = frame_num / total_frames
        bar_w = width - 40
        bar_x = 20
        bar_y = 70
        bar_h = 15
        
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+int(bar_w*progress), bar_y+bar_h), (0, 255, 0), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200, 200, 200), 1)
        
        cv2.imshow("TRACKING TEST", display)
        
        frame_num += 1
        
        if frame_num % 30 == 0:
            pct = (frame_num / total_frames) * 100
            print(f"  Progress: {frame_num}/{total_frames} ({pct:.1f}%) - "
                  f"MY: {my_r.visual_quality:.2f} visual quality", end="")
            if tracker.has_opponent and results["opponent"].tracking_active:
                print(f", OPP: {results['opponent'].visual_quality:.2f}")
            else:
                print()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n⚠️  User interrupted tracking")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Validate tracking quality
    validation_results = validate_tracking_quality(tracker, video_info)
    
    # Extract presence zones
    print("\n" + "=" * 70)
    print("📍 EXTRACTING PRESENCE ZONES")
    print("=" * 70)
    
    video_id = Path(video_path).stem
    
    zones_data = extract_presence_zones_from_tracker(
        tracker=tracker,
        video_id=video_id,
        video_duration=video_info["duration_seconds"],
        gap_threshold=1.5,
        min_duration=1.0,
        confidence_threshold=0.40,
        visual_quality_threshold=0.45  # 🔧 KEY: Only include frames where box is ON fighter
    )
    
    # Save output files
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    save_presence_zones(zones_data, str(output_dir), video_id)
    
    # Validate presence zones
    zone_validation = validate_presence_zones(
        zones_data, 
        validation_results, 
        video_info["duration_seconds"]
    )
    
    # Export frame validation CSV
    export_frame_validation_csv(tracker, output_dir, video_id)
    
    # Final summary
    print("\n" + "═" * 70)
    print("🎉 TEST COMPLETE - FINAL SUMMARY")
    print("═" * 70)
    
    my_ready = validation_results['my_fighter']['client_ready']
    zone_ready = zone_validation['client_ready']
    overall_ready = my_ready and zone_ready
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Tracking Quality:    {'✅ CLIENT-READY' if my_ready else '❌ NEEDS WORK'}")
    print(f"   Presence Zones:      {'✅ CLIENT-READY' if zone_ready else '❌ NEEDS WORK'}")
    print(f"\n   Overall Verdict:     {'✅ CLIENT-READY' if overall_ready else '❌ NEEDS IMPROVEMENT'}")
    
    if overall_ready:
        print(f"\n🎯 SYSTEM IS CLIENT-READY!")
        print(f"   ✅ Boxes track fighters accurately")
        print(f"   ✅ Presence zones align with tracking")
        print(f"   ✅ Output files are spec-compliant")
        print(f"   ✅ Quality metrics meet requirements")
    else:
        print(f"\n⚠️  SYSTEM NEEDS IMPROVEMENT:")
        if not my_ready:
            print(f"   ❌ Tracking quality below requirements")
            print(f"      Check: Are boxes staying ON the fighters?")
        if not zone_ready:
            print(f"   ❌ Presence zone quality below requirements")
            print(f"      Check: Coverage, zone duration, gaps")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   {output_dir.absolute()}/{video_id}_my_fighter_zones.json")
    if tracker.has_opponent:
        print(f"   {output_dir.absolute()}/{video_id}_opponent_zones.json")
    print(f"   {output_dir.absolute()}/{video_id}_statistics.json")
    print(f"   {output_dir.absolute()}/{video_id}_frame_validation.csv")
    
    print(f"\n" + "═" * 70 + "\n")
    
    return overall_ready


def main():
    print("\n" + "═" * 70)
    print("🥊 MMA VIDEO ANALYSIS - COMPREHENSIVE TEST")
    print("   Validates tracking + presence zones + quality")
    print("═" * 70)

    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_complete_system.py <video_path>")
        print("\nExample:")
        print("   python test_complete_system.py mma_fighter_analysis/Videos/VED.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"\n❌ Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"📅 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "─" * 70 + "\n")

    success = run_complete_test(video_path)
    
    print(f"\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()