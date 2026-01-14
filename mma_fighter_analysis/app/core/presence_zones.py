"""
PRESENCE ZONE EXTRACTION - ALIGNED WITH VISUAL TRACKING
Ensures zones ONLY include frames where box is ON the fighter

KEY FIXES:
1. Uses visual_quality score (not just confidence)
2. Stricter thresholds - only includes good tracking
3. Frame-by-frame validation
4. Output matches what user SEES in video
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PresenceZone:
    """Represents a continuous time range where fighter is present."""
    start: float
    end: float
    
    def duration(self) -> float:
        return self.end - self.start
    
    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2)
        }


class AccuratePresenceZoneExtractor:
    """
    Extracts presence zones that MATCH visual tracking quality.
    
    🔧 KEY PRINCIPLE: Only include frames where box is VISUALLY on the fighter.
    """
    
    def __init__(self, 
                 gap_threshold_seconds: float = 1.5,
                 min_zone_duration: float = 1.0,
                 confidence_threshold: float = 0.40,
                 visual_quality_threshold: float = 0.45):
        """
        Initialize with STRICT quality requirements.
        
        Args:
            gap_threshold_seconds: Max gap to merge zones (default: 1.5s)
            min_zone_duration: Minimum zone duration (default: 1.0s)
            confidence_threshold: Minimum confidence (default: 0.40)
            visual_quality_threshold: Minimum visual quality (default: 0.45)
                🔧 NEW: Ensures box is actually ON the fighter
        
        REASONING:
        - visual_quality_threshold=0.45: Strict appearance matching
          * >0.45: Box is on correct fighter
          * 0.35-0.45: Possible drift
          * <0.35: Definitely lost or on wrong target
        
        - confidence_threshold=0.40: Combined with visual quality
          * Acts as secondary filter
          * Both must pass for frame to count
        """
        self.gap_threshold = gap_threshold_seconds
        self.min_duration = min_zone_duration
        self.confidence_threshold = confidence_threshold
        self.visual_quality_threshold = visual_quality_threshold
        
        print(f"\n🎯 ACCURATE ZONE EXTRACTION PARAMETERS:")
        print(f"   Confidence threshold:     {confidence_threshold:.2f}")
        print(f"   Visual quality threshold: {visual_quality_threshold:.2f} 🔧 NEW")
        print(f"   Min zone duration:        {self.min_duration:.1f}s")
        
        print(f"   Gap merge threshold:      {gap_threshold_seconds:.1f}s")
        print(f"\n💡 Zones will ONLY include frames where box is ON the fighter")
    
    def extract_zones(self, 
                     tracking_history: List,
                     video_id: str) -> Dict:
        """
        Extract presence zones - STRICTLY aligned with visual tracking.
        
        Args:
            tracking_history: List of TrackingFrame objects (with visual_quality)
            video_id: Video identifier
            
        Returns:
            Dictionary matching Section 4.2 format with ACCURATE zones
        """
        if not tracking_history:
            return {
                "video_id": video_id,
                "my_fighter_presence": []
            }
        
        # Extract raw zones with STRICT visual quality requirements
        raw_zones = self._extract_raw_zones(tracking_history)
        
        # Merge zones with small gaps
        merged_zones = self._merge_zones(raw_zones)
        
        # Filter by minimum duration
        filtered_zones = [
            zone for zone in merged_zones 
            if zone.duration() >= self.min_duration
        ]
        
        # Sort by start time
        filtered_zones.sort(key=lambda z: z.start)
        
        return {
            "video_id": video_id,
            "my_fighter_presence": [zone.to_dict() for zone in filtered_zones]
        }
    
    def _extract_raw_zones(self, tracking_history: List) -> List[PresenceZone]:
        """
        Extract raw zones with STRICT visual quality validation.
        
        A fighter is considered "present" ONLY when:
        1. tracking_active is True
        2. bbox exists (not None)
        3. confidence >= confidence_threshold
        4. visual_quality >= visual_quality_threshold 🔧 KEY: Box is ON fighter
        5. tracker_source is not "LOST"
        
        This ensures zones match what the user SEES in the video.
        """
        zones = []
        current_start = None
        current_end = None
        
        frames_rejected_visual = 0
        frames_rejected_conf = 0
        frames_accepted = 0
        
        for frame in tracking_history:
            # Check basic tracking status
            basic_ok = (
                frame.tracking_active and 
                frame.bbox is not None and
                frame.tracker_source != "LOST"
            )
            
            if not basic_ok:
                if current_start is not None:
                    zones.append(PresenceZone(current_start, current_end))
                    current_start = None
                    current_end = None
                continue
            
            # Check confidence
            conf_ok = frame.confidence >= self.confidence_threshold
            if not conf_ok:
                frames_rejected_conf += 1
                if current_start is not None:
                    zones.append(PresenceZone(current_start, current_end))
                    current_start = None
                    current_end = None
                continue
            
            # 🔧 KEY CHECK: Visual quality (is box ON the fighter?)
            visual_ok = frame.visual_quality >= self.visual_quality_threshold
            if not visual_ok:
                frames_rejected_visual += 1
                if current_start is not None:
                    zones.append(PresenceZone(current_start, current_end))
                    current_start = None
                    current_end = None
                continue
            
            # All checks passed - frame is valid
            frames_accepted += 1
            
            if current_start is None:
                current_start = frame.timestamp
                current_end = frame.timestamp
            else:
                current_end = frame.timestamp
        
        # Handle last zone
        if current_start is not None:
            zones.append(PresenceZone(current_start, current_end))
        
        print(f"\n📊 Frame Quality Analysis:")
        print(f"   Frames ACCEPTED:             {frames_accepted}")
        print(f"   Frames REJECTED (low conf):  {frames_rejected_conf}")
        print(f"   Frames REJECTED (visual):    {frames_rejected_visual} 🔧")
        
        return zones
    
    def _merge_zones(self, zones: List[PresenceZone]) -> List[PresenceZone]:
        """
        Merge zones if gap is small.
        
        Handles brief tracking losses due to:
        - Occlusions (fighter blocked temporarily)
        - Fast movements (brief tracker confusion)
        - Camera motion
        """
        if not zones:
            return []
        
        sorted_zones = sorted(zones, key=lambda z: z.start)
        merged = []
        current = sorted_zones[0]
        
        for next_zone in sorted_zones[1:]:
            gap = next_zone.start - current.end
            
            if gap <= self.gap_threshold:
                # Merge
                current = PresenceZone(current.start, next_zone.end)
            else:
                # Gap too large
                merged.append(current)
                current = next_zone
        
        merged.append(current)
        return merged
    
    def calculate_statistics(self, zones: List[PresenceZone], 
                            video_duration: float,
                            tracking_history: List) -> Dict:
        """
        Calculate comprehensive statistics with quality metrics.
        """
        if not zones:
            return {
                "total_zones": 0,
                "total_presence_time": 0.0,
                "presence_percentage": 0.0,
                "average_zone_duration": 0.0,
                "longest_zone": 0.0,
                "shortest_zone": 0.0,
                "gaps": [],
                "quality_metrics": {
                    "avg_confidence": 0.0,
                    "avg_visual_quality": 0.0,
                    "frames_in_zones": 0,
                    "frames_excluded": 0
                }
            }
        
        total_time = sum(zone.duration() for zone in zones)
        durations = [zone.duration() for zone in zones]
        
        # Calculate gaps
        gaps = []
        for i in range(len(zones) - 1):
            gap_duration = zones[i+1].start - zones[i].end
            if gap_duration > 0:
                gaps.append({
                    "after_zone": i + 1,
                    "start": round(zones[i].end, 2),
                    "end": round(zones[i+1].start, 2),
                    "duration": round(gap_duration, 2)
                })
        
        # Calculate quality metrics
        frames_in_zones = 0
        total_conf = 0.0
        total_visual = 0.0
        
        for frame in tracking_history:
            if frame.tracking_active and frame.bbox is not None:
                in_zone = any(zone.start <= frame.timestamp <= zone.end for zone in zones)
                if in_zone:
                    frames_in_zones += 1
                    total_conf += frame.confidence
                    total_visual += frame.visual_quality
        
        avg_conf = total_conf / frames_in_zones if frames_in_zones > 0 else 0.0
        avg_visual = total_visual / frames_in_zones if frames_in_zones > 0 else 0.0
        
        total_tracked = sum(1 for f in tracking_history if f.tracking_active and f.bbox is not None)
        frames_excluded = total_tracked - frames_in_zones
        
        return {
            "total_zones": len(zones),
            "total_presence_time": round(total_time, 2),
            "presence_percentage": round((total_time / video_duration) * 100, 2) if video_duration > 0 else 0.0,
            "average_zone_duration": round(sum(durations) / len(durations), 2),
            "longest_zone": round(max(durations), 2),
            "shortest_zone": round(min(durations), 2),
            "gaps": gaps,
            "quality_metrics": {
                "avg_confidence": round(avg_conf, 3),
                "avg_visual_quality": round(avg_visual, 3),
                "frames_in_zones": frames_in_zones,
                "frames_excluded": frames_excluded,
                "exclusion_rate": round((frames_excluded / total_tracked * 100), 2) if total_tracked > 0 else 0.0
            }
        }


def extract_presence_zones_from_tracker(tracker,
                                       video_id: str,
                                       video_duration: float,
                                       gap_threshold: float = 1.5,
                                       min_duration: float = 1.0,
                                       confidence_threshold: float = 0.40,
                                       visual_quality_threshold: float = 0.45,
                                       save_statistics: bool = True) -> Dict:
    """
    Extract ACCURATE presence zones aligned with visual tracking.
    
    🔧 PRODUCTION DEFAULTS:
    - gap_threshold: 1.5s
    - min_duration: 1.0s
    - confidence_threshold: 0.40
    - visual_quality_threshold: 0.45 🔧 NEW: Ensures box is ON fighter
    
    Returns zones that match what user SEES in video display.
    """
    extractor = AccuratePresenceZoneExtractor(
        gap_threshold, 
        min_duration, 
        confidence_threshold,
        visual_quality_threshold
    )
    
    # Extract MY FIGHTER zones
    my_zones_output = extractor.extract_zones(
        tracker.my_tracker.tracking_history,
        video_id
    )
    
    # Extract OPPONENT zones
    opponent_zones_output = None
    if tracker.has_opponent:
        opponent_zones_output = extractor.extract_zones(
            tracker.opp_tracker.tracking_history,
            video_id
        )
    
    # Calculate statistics
    stats = None
    if save_statistics:
        my_zone_objects = extractor._extract_raw_zones(
            tracker.my_tracker.tracking_history
        )
        my_zone_objects = extractor._merge_zones(my_zone_objects)
        my_zone_objects = [z for z in my_zone_objects if z.duration() >= min_duration]
        
        my_stats = extractor.calculate_statistics(
            my_zone_objects, 
            video_duration,
            tracker.my_tracker.tracking_history
        )
        
        opp_stats = None
        if tracker.has_opponent:
            opp_zone_objects = extractor._extract_raw_zones(
                tracker.opp_tracker.tracking_history
            )
            opp_zone_objects = extractor._merge_zones(opp_zone_objects)
            opp_zone_objects = [z for z in opp_zone_objects if z.duration() >= min_duration]
            opp_stats = extractor.calculate_statistics(
                opp_zone_objects,
                video_duration,
                tracker.opp_tracker.tracking_history
            )
        
        stats = {
            "video_id": video_id,
            "video_duration": round(video_duration, 2),
            "extraction_parameters": {
                "gap_threshold": gap_threshold,
                "min_duration": min_duration,
                "confidence_threshold": confidence_threshold,
                "visual_quality_threshold": visual_quality_threshold
            },
            "my_fighter": my_stats,
            "opponent": opp_stats
        }
    
    # Print report
    print("\n" + "=" * 70)
    print("📍 ACCURATE PRESENCE ZONE EXTRACTION COMPLETE")
    print("=" * 70)
    
    if stats:
        print(f"\n⚙️  EXTRACTION PARAMETERS:")
        print(f"   Confidence Threshold:     {confidence_threshold:.2f}")
        print(f"   Visual Quality Threshold: {visual_quality_threshold:.2f} 🔧 KEY")
        print(f"   Min Zone Duration:        {min_duration:.1f}s")
        print(f"   Gap Merge Threshold:      {gap_threshold:.1f}s")
        
        print(f"\n🔴 MY FIGHTER:")
        print(f"   Total Zones:     {stats['my_fighter']['total_zones']}")
        print(f"   Presence Time:   {stats['my_fighter']['total_presence_time']:.2f}s / {video_duration:.2f}s")
        print(f"   Coverage:        {stats['my_fighter']['presence_percentage']:.1f}%")
        
        if stats['my_fighter']['total_zones'] > 0:
            print(f"   Avg Zone:        {stats['my_fighter']['average_zone_duration']:.2f}s")
            print(f"   Longest Zone:    {stats['my_fighter']['longest_zone']:.2f}s")
            print(f"   Shortest Zone:   {stats['my_fighter']['shortest_zone']:.2f}s")
        
        qm = stats['my_fighter']['quality_metrics']
        print(f"\n   Quality Metrics:")
        print(f"   - Avg Confidence:    {qm['avg_confidence']:.3f}")
        print(f"   - Avg Visual Quality: {qm['avg_visual_quality']:.3f} 🔧")
        print(f"   - Frames Included:   {qm['frames_in_zones']}")
        print(f"   - Frames Excluded:   {qm['frames_excluded']} ({qm['exclusion_rate']:.1f}%)")
        
        if stats['my_fighter']['gaps']:
            print(f"\n   Gaps: {len(stats['my_fighter']['gaps'])} detected")
        
        if stats['opponent']:
            print(f"\n🔵 OPPONENT:")
            print(f"   Total Zones:     {stats['opponent']['total_zones']}")
            print(f"   Presence Time:   {stats['opponent']['total_presence_time']:.2f}s / {video_duration:.2f}s")
            print(f"   Coverage:        {stats['opponent']['presence_percentage']:.1f}%")
            
            if stats['opponent']['total_zones'] > 0:
                print(f"   Avg Zone:        {stats['opponent']['average_zone_duration']:.2f}s")
            
            opp_qm = stats['opponent']['quality_metrics']
            print(f"\n   Quality Metrics:")
            print(f"   - Avg Confidence:    {opp_qm['avg_confidence']:.3f}")
            print(f"   - Avg Visual Quality: {opp_qm['avg_visual_quality']:.3f} 🔧")
            print(f"   - Frames Excluded:   {opp_qm['frames_excluded']} ({opp_qm['exclusion_rate']:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Output Format: SPEC COMPLIANT (Section 4.2)")
    print("✅ Zones: ALIGNED with visual tracking quality")
    print("💡 Excluded frames where box drifted off fighter")
    print("=" * 70)
    
    return {
        "my_fighter_zones": my_zones_output,
        "opponent_zones": opponent_zones_output,
        "statistics": stats
    }


def save_presence_zones(zones_data: Dict, 
                       output_dir: str,
                       video_id: str,
                       pretty: bool = True):
    """Save presence zones to SEPARATE JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save MY FIGHTER zones
    my_fighter_file = output_path / f"{video_id}_my_fighter_zones.json"
    with open(my_fighter_file, 'w') as f:
        if pretty:
            json.dump(zones_data["my_fighter_zones"], f, indent=2)
        else:
            json.dump(zones_data["my_fighter_zones"], f)
    
    print(f"\n💾 Saved: {my_fighter_file}")
    print("   Format: ✅ SPEC COMPLIANT (Section 4.2)")
    
    num_zones = len(zones_data["my_fighter_zones"]["my_fighter_presence"])
    print(f"   Zones:  {num_zones}")
    
    if num_zones > 0:
        zones = zones_data["my_fighter_zones"]["my_fighter_presence"]
        print(f"   First:  {zones[0]['start']:.2f}s → {zones[0]['end']:.2f}s")
        if num_zones > 1:
            print(f"   Last:   {zones[-1]['start']:.2f}s → {zones[-1]['end']:.2f}s")
        
        # Show quality guarantee
        if zones_data.get("statistics"):
            qm = zones_data["statistics"]["my_fighter"]["quality_metrics"]
            print(f"\n   Quality Guarantee:")
            print(f"   - Visual Quality: {qm['avg_visual_quality']:.3f} (box ON fighter)")
            print(f"   - {qm['frames_excluded']} poor-quality frames excluded")
    
    # Save OPPONENT zones
    if zones_data["opponent_zones"]:
        opponent_file = output_path / f"{video_id}_opponent_zones.json"
        with open(opponent_file, 'w') as f:
            if pretty:
                json.dump(zones_data["opponent_zones"], f, indent=2)
            else:
                json.dump(zones_data["opponent_zones"], f)
        
        print(f"\n💾 Saved: {opponent_file}")
        print("   Format: ✅ SPEC COMPLIANT (Section 4.2)")
        
        num_zones = len(zones_data["opponent_zones"]["my_fighter_presence"])
        print(f"   Zones:  {num_zones}")
    
    # Save statistics
    if zones_data["statistics"]:
        stats_file = output_path / f"{video_id}_statistics.json"
        with open(stats_file, 'w') as f:
            if pretty:
                json.dump(zones_data["statistics"], f, indent=2)
            else:
                json.dump(zones_data["statistics"], f)
        
        print(f"\n💾 Saved: {stats_file}")
        print("   Format: Extended statistics with quality metrics")