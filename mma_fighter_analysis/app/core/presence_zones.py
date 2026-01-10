# Presence zone extraction & merging
"""
Presence Zone Extraction Module
Converts frame-by-frame tracking results into time-based presence zones.

Output format per requirements (Section 4.2):
{
    "video_id": "...",
    "my_fighter_presence": [
        { "start": 0, "end": 87 },
        { "start": 102, "end": 214 }
    ]
}
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PresenceZone:
    """Represents a continuous time range where fighter is present."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    
    def duration(self) -> float:
        """Calculate zone duration in seconds."""
        return self.end - self.start
    
    def to_dict(self) -> dict:
        """Convert to dictionary with rounded values."""
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2)
        }


class PresenceZoneExtractor:
    """
    Extracts presence zones from tracking history.
    
    Converts frame-by-frame tracking into continuous time ranges.
    Merges zones if gaps are small (configurable threshold).
    """
    
    def __init__(self, 
                 gap_threshold_seconds: float = 2.0,
                 min_zone_duration: float = 1.0):
        """
        Initialize zone extractor.
        
        Args:
            gap_threshold_seconds: Max gap to merge zones (default: 2.0s)
            min_zone_duration: Minimum zone duration to include (default: 1.0s)
        """
        self.gap_threshold = gap_threshold_seconds
        self.min_duration = min_zone_duration
    
    def extract_zones(self, 
                     tracking_history: List,
                     video_id: str) -> Dict:
        """
        Extract presence zones from tracking history.
        
        Args:
            tracking_history: List of TrackingFrame objects
            video_id: Video identifier
            
        Returns:
            Dictionary with format:
            {
                "video_id": "...",
                "my_fighter_presence": [
                    {"start": 0, "end": 87},
                    ...
                ]
            }
        """
        if not tracking_history:
            return {
                "video_id": video_id,
                "my_fighter_presence": []
            }
        
        # Extract raw zones (continuous tracking sequences)
        raw_zones = self._extract_raw_zones(tracking_history)
        
        # Merge zones with small gaps
        merged_zones = self._merge_zones(raw_zones)
        
        # Filter by minimum duration
        filtered_zones = [
            zone for zone in merged_zones 
            if zone.duration() >= self.min_duration
        ]
        
        # Convert to output format
        return {
            "video_id": video_id,
            "my_fighter_presence": [zone.to_dict() for zone in filtered_zones]
        }
    
    def _extract_raw_zones(self, tracking_history: List) -> List[PresenceZone]:
        """
        Extract raw presence zones (continuous tracked sequences).
        
        Args:
            tracking_history: List of TrackingFrame objects
            
        Returns:
            List of PresenceZone objects
        """
        zones = []
        current_start = None
        current_end = None
        
        for frame in tracking_history:
            if frame.tracking_active and frame.bbox is not None:
                # Fighter is present
                if current_start is None:
                    # Start new zone
                    current_start = frame.timestamp
                    current_end = frame.timestamp
                else:
                    # Extend current zone
                    current_end = frame.timestamp
            else:
                # Fighter is lost
                if current_start is not None:
                    # End current zone
                    zones.append(PresenceZone(current_start, current_end))
                    current_start = None
                    current_end = None
        
        # Handle last zone if video ends while tracking
        if current_start is not None:
            zones.append(PresenceZone(current_start, current_end))
        
        return zones
    
    def _merge_zones(self, zones: List[PresenceZone]) -> List[PresenceZone]:
        """
        Merge zones if gap between them is below threshold.
        
        Args:
            zones: List of PresenceZone objects
            
        Returns:
            List of merged PresenceZone objects
        """
        if not zones:
            return []
        
        # Sort zones by start time
        sorted_zones = sorted(zones, key=lambda z: z.start)
        
        merged = []
        current = sorted_zones[0]
        
        for next_zone in sorted_zones[1:]:
            gap = next_zone.start - current.end
            
            if gap <= self.gap_threshold:
                # Merge zones
                current = PresenceZone(current.start, next_zone.end)
            else:
                # Keep current zone, start new one
                merged.append(current)
                current = next_zone
        
        # Add last zone
        merged.append(current)
        
        return merged
    
    def get_statistics(self, zones: List[PresenceZone], 
                      video_duration: float) -> Dict:
        """
        Calculate presence statistics.
        
        Args:
            zones: List of PresenceZone objects
            video_duration: Total video duration in seconds
            
        Returns:
            Dictionary with statistics
        """
        if not zones:
            return {
                "total_zones": 0,
                "total_presence_time": 0.0,
                "presence_percentage": 0.0,
                "average_zone_duration": 0.0,
                "longest_zone": 0.0,
                "shortest_zone": 0.0
            }
        
        total_time = sum(zone.duration() for zone in zones)
        durations = [zone.duration() for zone in zones]
        
        return {
            "total_zones": len(zones),
            "total_presence_time": round(total_time, 2),
            "presence_percentage": round((total_time / video_duration) * 100, 2),
            "average_zone_duration": round(sum(durations) / len(durations), 2),
            "longest_zone": round(max(durations), 2),
            "shortest_zone": round(min(durations), 2)
        }


def extract_presence_zones_from_tracker(tracker,
                                       video_id: str,
                                       video_duration: float,
                                       gap_threshold: float = 2.0,
                                       min_duration: float = 1.0) -> Dict:
    """
    Extract presence zones from DualFighterTracker.
    
    Args:
        tracker: DualFighterTracker instance
        video_id: Video identifier
        video_duration: Total video duration in seconds
        gap_threshold: Max gap to merge zones (default: 2.0s)
        min_duration: Minimum zone duration (default: 1.0s)
        
    Returns:
        Dictionary with presence zones for both fighters
    """
    extractor = PresenceZoneExtractor(gap_threshold, min_duration)
    
    # Extract zones for my fighter
    my_zones = extractor.extract_zones(
        tracker.my_tracker.tracking_history,
        video_id
    )
    
    # Extract zones for opponent (if present)
    if tracker.has_opponent:
        opp_zones = extractor.extract_zones(
            tracker.opp_tracker.tracking_history,
            f"{video_id}_opponent"
        )
    else:
        opp_zones = {
            "video_id": f"{video_id}_opponent",
            "my_fighter_presence": []
        }
    
    # Calculate statistics
    my_zone_objects = extractor._extract_raw_zones(
        tracker.my_tracker.tracking_history
    )
    my_zone_objects = extractor._merge_zones(my_zone_objects)
    my_zone_objects = [z for z in my_zone_objects if z.duration() >= min_duration]
    
    my_stats = extractor.get_statistics(my_zone_objects, video_duration)
    
    if tracker.has_opponent:
        opp_zone_objects = extractor._extract_raw_zones(
            tracker.opp_tracker.tracking_history
        )
        opp_zone_objects = extractor._merge_zones(opp_zone_objects)
        opp_zone_objects = [z for z in opp_zone_objects if z.duration() >= min_duration]
        opp_stats = extractor.get_statistics(opp_zone_objects, video_duration)
    else:
        opp_stats = None
    
    # Print report
    print("\n" + "=" * 60)
    print("📍 PRESENCE ZONE EXTRACTION COMPLETE")
    print("=" * 60)
    
    print(f"\n🔴 MY FIGHTER:")
    print(f"   Total Zones:     {my_stats['total_zones']}")
    print(f"   Presence Time:   {my_stats['total_presence_time']:.2f}s / {video_duration:.2f}s")
    print(f"   Coverage:        {my_stats['presence_percentage']:.1f}%")
    print(f"   Avg Zone:        {my_stats['average_zone_duration']:.2f}s")
    print(f"   Longest Zone:    {my_stats['longest_zone']:.2f}s")
    print(f"   Shortest Zone:   {my_stats['shortest_zone']:.2f}s")
    
    if opp_stats:
        print(f"\n🔵 OPPONENT:")
        print(f"   Total Zones:     {opp_stats['total_zones']}")
        print(f"   Presence Time:   {opp_stats['total_presence_time']:.2f}s / {video_duration:.2f}s")
        print(f"   Coverage:        {opp_stats['presence_percentage']:.1f}%")
        print(f"   Avg Zone:        {opp_stats['average_zone_duration']:.2f}s")
        print(f"   Longest Zone:    {opp_stats['longest_zone']:.2f}s")
        print(f"   Shortest Zone:   {opp_stats['shortest_zone']:.2f}s")
    
    print("=" * 60)
    
    return {
        "my_fighter": my_zones,
        "opponent": opp_zones,
        "statistics": {
            "my_fighter": my_stats,
            "opponent": opp_stats
        }
    }


def save_presence_zones(zones_data: Dict, 
                       output_path: str,
                       pretty: bool = True):
    """
    Save presence zones to JSON file.
    
    Args:
        zones_data: Dictionary with zones and statistics
        output_path: Path to output JSON file
        pretty: Whether to pretty-print JSON (default: True)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(zones_data, f, indent=2)
        else:
            json.dump(zones_data, f)
    
    print(f"\n💾 Presence zones saved: {output_path}")


# ==================== EXAMPLE USAGE ==================== #

if __name__ == "__main__":
    """
    Example usage (for testing).
    In production, this is called after tracking completes.
    """
    
    # Mock tracking history for demonstration
    from dataclasses import dataclass
    
    @dataclass
    class MockTrackingFrame:
        frame_num: int
        timestamp: float
        bbox: Optional[Tuple[int, int, int, int]]
        tracking_active: bool
    
    # Simulate tracking history: present for 0-10s, lost 10-12s, present 12-20s
    mock_history = []
    
    # Zone 1: 0-10s (tracked)
    for i in range(300):  # 300 frames at 30fps = 10s
        mock_history.append(MockTrackingFrame(
            frame_num=i,
            timestamp=i/30.0,
            bbox=(100, 100, 200, 300),
            tracking_active=True
        ))
    
    # Gap: 10-12s (lost)
    for i in range(300, 360):  # 60 frames = 2s
        mock_history.append(MockTrackingFrame(
            frame_num=i,
            timestamp=i/30.0,
            bbox=None,
            tracking_active=False
        ))
    
    # Zone 2: 12-20s (tracked)
    for i in range(360, 600):  # 240 frames = 8s
        mock_history.append(MockTrackingFrame(
            frame_num=i,
            timestamp=i/30.0,
            bbox=(150, 120, 210, 310),
            tracking_active=True
        ))
    
    # Test extraction
    extractor = PresenceZoneExtractor(gap_threshold=2.0, min_duration=1.0)
    result = extractor.extract_zones(mock_history, "test_video")
    
    print("\n" + "=" * 60)
    print("TEST: PRESENCE ZONE EXTRACTION")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    
    # Calculate statistics
    zones = [PresenceZone(**z) for z in result["my_fighter_presence"]]
    stats = extractor.get_statistics(zones, 20.0)
    
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))