

import json
from typing import List, Dict, Optional, Tuple


class PresenceZoneTracker:
   
    
    def __init__(self, fps: float, merge_gap_seconds: float = 0.5, presence_threshold: float = 0.6):
       
        self.fps = fps
        self.merge_gap_frames = int(merge_gap_seconds * fps)
        self.presence_threshold = presence_threshold  # Fix 6: Separate presence threshold
        
        # Current state
        self.my_fighter_present = False
        self.opponent_present = False
        
        # Current zone start frames
        self.my_fighter_zone_start = None
        self.opponent_zone_start = None
        
        # Completed zones (frame numbers)
        self.my_fighter_zones = []
        self.opponent_zones = []
        
        # Frame counter
        self.current_frame = 0
        
        # Statistics
        self.my_fighter_total_frames_visible = 0
        self.opponent_total_frames_visible = 0
        self.my_fighter_temp_lost_count = 0
        self.opponent_temp_lost_count = 0
        self.my_fighter_unreliable_count = 0  # Tracked but validity too low
        self.opponent_unreliable_count = 0
        
        print(f"📊 Presence Tracker Initialized (V9 Aligned)")
        print(f"   FPS: {fps}")
        print(f"   Merge gaps < {merge_gap_seconds:.2f}s ({self.merge_gap_frames} frames)")
        print(f"   Using tracker 'reliable' field (validity > 0.4)")
    
    def update(self, tracker_status: Dict):
        
        my_state = tracker_status['my_fighter']['state']
        my_reliable = tracker_status['my_fighter']['reliable']
        my_bbox = tracker_status['my_fighter']['bbox']
        my_validity = tracker_status['my_fighter']['validity']
        
        
        opp_state = tracker_status['opponent']['state']
        opp_reliable = tracker_status['opponent']['reliable']
        opp_bbox = tracker_status['opponent']['bbox']
        opp_validity = tracker_status['opponent']['validity']
        
       
        my_visible = self._is_fighter_present(my_state, my_reliable, my_bbox, my_validity)
        
        if my_visible:
            self.my_fighter_total_frames_visible += 1
            
            if not self.my_fighter_present:
                # Transition: ABSENT → PRESENT
                self.my_fighter_zone_start = self.current_frame
                self.my_fighter_present = True
        else:
            
            if my_bbox is not None and not my_reliable and my_state == 'visible':
                self.my_fighter_unreliable_count += 1
            
            # TEMP_LOST now creates gaps (no grace period for presence JSON)
            if self.my_fighter_present:
                # Transition: PRESENT → ABSENT (gap created)
                zone = (self.my_fighter_zone_start, self.current_frame - 1)
                self.my_fighter_zones.append(zone)
                self.my_fighter_present = False
                self.my_fighter_zone_start = None
        
        opp_visible = self._is_fighter_present(opp_state, opp_reliable, opp_bbox, opp_validity)
        
        if opp_visible:
            self.opponent_total_frames_visible += 1
            
            if not self.opponent_present:
                # Transition: ABSENT → PRESENT
                self.opponent_zone_start = self.current_frame
                self.opponent_present = True
        else:
           
            if opp_bbox is not None and not opp_reliable and opp_state == 'visible':
                self.opponent_unreliable_count += 1
            
            # TEMP_LOST now creates gaps (no grace period for presence JSON)
            if self.opponent_present:
                # Transition: PRESENT → ABSENT (gap created)
                zone = (self.opponent_zone_start, self.current_frame - 1)
                self.opponent_zones.append(zone)
                self.opponent_present = False
                self.opponent_zone_start = None
        
        self.current_frame += 1
    
    def _is_fighter_present(self, state: str, reliable: bool, bbox, validity: float) -> bool:
       
        if bbox is None:
            return False
        
        
        if state != 'visible':
            return False
        
        if validity < self.presence_threshold:
            return False
        
        return True
    
    def finalize(self):
       
       
        if self.my_fighter_present and self.my_fighter_zone_start is not None:
            zone = (self.my_fighter_zone_start, self.current_frame - 1)
            self.my_fighter_zones.append(zone)
            self.my_fighter_present = False
        
        if self.opponent_present and self.opponent_zone_start is not None:
            zone = (self.opponent_zone_start, self.current_frame - 1)
            self.opponent_zones.append(zone)
            self.opponent_present = False
        
       
        self.my_fighter_zones = self._merge_small_gaps(self.my_fighter_zones)
        self.opponent_zones = self._merge_small_gaps(self.opponent_zones)
        
        print(f"✅ Presence tracking finalized")
        print(f"   My Fighter: {len(self.my_fighter_zones)} zones, {self.my_fighter_total_frames_visible} frames reliable")
        print(f"   Opponent: {len(self.opponent_zones)} zones, {self.opponent_total_frames_visible} frames reliable")
        print(f"   Grace periods: My={self.my_fighter_temp_lost_count}, Opp={self.opponent_temp_lost_count}")
        print(f"   Unreliable frames filtered: My={self.my_fighter_unreliable_count}, Opp={self.opponent_unreliable_count}")
    
    def _merge_small_gaps(self, zones: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
      
        if len(zones) <= 1:
            return zones
        
        # Sort by start frame
        zones = sorted(zones, key=lambda z: z[0])
        
        merged = []
        current_start, current_end = zones[0]
        
        for i in range(1, len(zones)):
            next_start, next_end = zones[i]
            gap = next_start - current_end - 1
            
            if gap <= self.merge_gap_frames:
                # Merge: extend current zone
                current_end = next_end
            else:
                # Gap too large: save current zone and start new one
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        # Add final zone
        merged.append((current_start, current_end))
        
        return merged
    
    def _frame_to_timestamp(self, frame: int) -> float:
        """Convert frame number to timestamp in seconds."""
        return frame / self.fps
    
    def get_zones_as_timestamps(self) -> Dict[str, List[Dict[str, float]]]:
       
        my_zones = [
            {
                "start": round(self._frame_to_timestamp(start), 2),
                "end": round(self._frame_to_timestamp(end), 2)
            }
            for start, end in self.my_fighter_zones
        ]
        
        opponent_zones = [
            {
                "start": round(self._frame_to_timestamp(start), 2),
                "end": round(self._frame_to_timestamp(end), 2)
            }
            for start, end in self.opponent_zones
        ]
        
        return {
            "my_fighter": my_zones,
            "opponent": opponent_zones
        }
    
    def get_zones_as_frames(self) -> Dict[str, List[Dict[str, int]]]:
       
        
        my_zones = [
            {"start": start, "end": end}
            for start, end in self.my_fighter_zones
        ]
        
        opponent_zones = [
            {"start": start, "end": end}
            for start, end in self.opponent_zones
        ]
        
        return {
            "my_fighter": my_zones,
            "opponent": opponent_zones
        }
    
    def export_json(self, video_name: str, output_path: str, include_metadata: bool = True):
       
        zones = self.get_zones_as_timestamps()
        
        output_data = {
            "video_id": video_name,
            "my_fighter_presence": zones["my_fighter"],
            "opponent_presence": zones["opponent"]
        }
        
        if include_metadata:
            output_data["metadata"] = {
                "fps": self.fps,
                "total_frames": self.current_frame,
                "duration_seconds": round(self.current_frame / self.fps, 2),
                "my_fighter_stats": {
                    "zone_count": len(self.my_fighter_zones),
                    "total_frames_reliable": self.my_fighter_total_frames_visible,
                    "frames_unreliable": self.my_fighter_unreliable_count,
                    "visibility_percentage": round(
                        100 * self.my_fighter_total_frames_visible / max(1, self.current_frame), 1
                    )
                },
                "opponent_stats": {
                    "zone_count": len(self.opponent_zones),
                    "total_frames_reliable": self.opponent_total_frames_visible,
                    "frames_unreliable": self.opponent_unreliable_count,
                    "visibility_percentage": round(
                        100 * self.opponent_total_frames_visible / max(1, self.current_frame), 1
                    )
                },
                "tracker_version": "v9_aligned",
                "uses_validity_filtering": True
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📁 Presence zones saved: {output_path}")
    
    def print_summary(self):
        """Print human-readable summary of presence zones."""
        zones = self.get_zones_as_timestamps()
        
        print("\n" + "=" * 60)
        print("📊 PRESENCE ZONE SUMMARY (V9 ALIGNED)")
        print("=" * 60)
        
        print(f"\n🔴 MY FIGHTER ({len(zones['my_fighter'])} zones):")
        total_my = 0
        for i, zone in enumerate(zones['my_fighter'], 1):
            duration = zone['end'] - zone['start']
            total_my += duration
            print(f"   {i}. {self._format_time(zone['start'])} → {self._format_time(zone['end'])} ({duration:.2f}s)")
        
        if len(zones['my_fighter']) == 0:
            print("   (No presence zones)")
        
        print(f"\n🔵 OPPONENT ({len(zones['opponent'])} zones):")
        total_opp = 0
        for i, zone in enumerate(zones['opponent'], 1):
            duration = zone['end'] - zone['start']
            total_opp += duration
            print(f"   {i}. {self._format_time(zone['start'])} → {self._format_time(zone['end'])} ({duration:.2f}s)")
        
        if len(zones['opponent']) == 0:
            print("   (No presence zones)")
        
        total_duration = self.current_frame / self.fps
        
        print(f"\n📈 TOTALS:")
        print(f"   Video duration: {total_duration:.2f}s ({self.current_frame} frames)")
        print(f"   My Fighter: {total_my:.2f}s ({100*total_my/max(1, total_duration):.1f}%)")
        print(f"   Opponent: {total_opp:.2f}s ({100*total_opp/max(1, total_duration):.1f}%)")
        print(f"   Grace periods: My={self.my_fighter_temp_lost_count}, Opp={self.opponent_temp_lost_count}")
        print(f"   Unreliable filtered: My={self.my_fighter_unreliable_count}, Opp={self.opponent_unreliable_count}")
        print("=" * 60 + "\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS.ms"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
    
    def get_current_status(self) -> Dict[str, any]:
       
        return {
            "frame": self.current_frame,
            "my_fighter_present": self.my_fighter_present,
            "opponent_present": self.opponent_present,
            "my_fighter_zones_count": len(self.my_fighter_zones),
            "opponent_zones_count": len(self.opponent_zones),
            "my_fighter_visibility": round(
                100 * self.my_fighter_total_frames_visible / max(1, self.current_frame), 1
            ),
            "opponent_visibility": round(
                100 * self.opponent_total_frames_visible / max(1, self.current_frame), 1
            ),
            "my_fighter_unreliable": self.my_fighter_unreliable_count,
            "opponent_unreliable": self.opponent_unreliable_count
        }
    
    def get_gaps(self) -> Dict[str, List[Dict[str, float]]]:
       
        def compute_gaps(zones, total_duration):
            if len(zones) == 0:
                return [{"start": 0.0, "end": total_duration}]
            
            gaps = []
            
           
            if zones[0]["start"] > 0:
                gaps.append({"start": 0.0, "end": zones[0]["start"]})
            
            
            for i in range(len(zones) - 1):
                gap_start = zones[i]["end"]
                gap_end = zones[i + 1]["start"]
                if gap_end > gap_start:
                    gaps.append({"start": gap_start, "end": gap_end})
            
            
            if zones[-1]["end"] < total_duration:
                gaps.append({"start": zones[-1]["end"], "end": total_duration})
            
            return gaps
        
        zones = self.get_zones_as_timestamps()
        total_duration = round(self.current_frame / self.fps, 2)
        
        return {
            "my_fighter": compute_gaps(zones["my_fighter"], total_duration),
            "opponent": compute_gaps(zones["opponent"], total_duration)
        }
