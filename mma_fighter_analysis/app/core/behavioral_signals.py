# Behavioral signal calculations
"""
Behavioral Signal Extraction Module
Extracts 8 behavioral movement signals from tracking data.

Output format per requirements (Section 5):
{
    "video_id": "...",
    "signals": [
        {
            "t": 12,
            "distance_delta": -0.34,
            "forward_velocity": 0.71,
            "lateral_displacement": 0.15,
            "vertical_level_change": -0.22,
            "contact_duration": 3.4,
            "control_overlap": 0.18,
            "recovery_latency": 1.9,
            "scramble_entropy": 0.62
        }
    ]
}

All values are relative/normalized (no real-world units).
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque


@dataclass
class SignalFrame:
    """Represents signals for a single timestamp."""
    t: float  # Timestamp in seconds
    distance_delta: float
    forward_velocity: float
    lateral_displacement: float
    vertical_level_change: float
    contact_duration: float
    control_overlap: float
    recovery_latency: float
    scramble_entropy: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary with rounded values."""
        return {
            "t": round(self.t, 2),
            "distance_delta": round(self.distance_delta, 3),
            "forward_velocity": round(self.forward_velocity, 3),
            "lateral_displacement": round(self.lateral_displacement, 3),
            "vertical_level_change": round(self.vertical_level_change, 3),
            "contact_duration": round(self.contact_duration, 2),
            "control_overlap": round(self.control_overlap, 3),
            "recovery_latency": round(self.recovery_latency, 2),
            "scramble_entropy": round(self.scramble_entropy, 3)
        }


class BehavioralSignalExtractor:
    """
    Extracts behavioral movement signals from dual fighter tracking.
    
    Computes signals per second based on bounding box positions.
    """
    
    # Thresholds (configurable)
    CLINCH_DISTANCE_THRESHOLD = 150  # pixels - fighters in contact range
    STABILIZATION_THRESHOLD = 20     # pixels - motion considered stable
    MOTION_HISTORY_WINDOW = 10       # frames for motion analysis
    
    def __init__(self, fps: float = 30.0, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialize signal extractor.
        
        Args:
            fps: Video frames per second
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Normalization factors
        self.max_distance = np.sqrt(frame_width**2 + frame_height**2)
        self.max_velocity = frame_width  # Max possible movement per frame
    
    def extract_signals(self, 
                       my_tracking_history: List,
                       opp_tracking_history: List,
                       video_id: str) -> Dict:
        """
        Extract behavioral signals from tracking histories.
        
        Args:
            my_tracking_history: Tracking history for my fighter
            opp_tracking_history: Tracking history for opponent
            video_id: Video identifier
            
        Returns:
            Dictionary with signals per second
        """
        # Group frames by second
        signals_by_second = {}
        
        # Calculate max timestamp
        max_time = max(
            max((f.timestamp for f in my_tracking_history), default=0),
            max((f.timestamp for f in opp_tracking_history), default=0)
        )
        
        # Process each second
        for second in range(int(max_time) + 1):
            # Get frames in this second
            my_frames = [f for f in my_tracking_history 
                        if int(f.timestamp) == second and f.tracking_active and f.bbox]
            opp_frames = [f for f in opp_tracking_history 
                         if int(f.timestamp) == second and f.tracking_active and f.bbox]
            
            if my_frames and opp_frames:
                # Calculate signals for this second
                signal = self._calculate_second_signals(
                    my_frames, 
                    opp_frames,
                    float(second),
                    my_tracking_history,
                    opp_tracking_history
                )
                signals_by_second[second] = signal
        
        # Convert to output format
        signals_list = [signals_by_second[s].to_dict() 
                       for s in sorted(signals_by_second.keys())]
        
        return {
            "video_id": video_id,
            "signals": signals_list
        }
    
    def _calculate_second_signals(self,
                                 my_frames: List,
                                 opp_frames: List,
                                 timestamp: float,
                                 my_full_history: List,
                                 opp_full_history: List) -> SignalFrame:
        """Calculate all 8 signals for a given second."""
        
        # Use middle frame of the second as representative
        my_frame = my_frames[len(my_frames) // 2]
        opp_frame = opp_frames[len(opp_frames) // 2]
        
        my_bbox = my_frame.bbox
        opp_bbox = opp_frame.bbox
        
        # Calculate centers
        my_center = self._get_center(my_bbox)
        opp_center = self._get_center(opp_bbox)
        
        # 1. Distance Delta
        distance_delta = self._calculate_distance_delta(
            my_frame, opp_frame, my_full_history, opp_full_history
        )
        
        # 2. Forward Velocity
        forward_velocity = self._calculate_forward_velocity(
            my_frame, my_center, opp_center, my_full_history
        )
        
        # 3. Lateral Displacement
        lateral_displacement = self._calculate_lateral_displacement(
            my_frame, my_center, opp_center, my_full_history
        )
        
        # 4. Vertical Level Change
        vertical_level_change = self._calculate_vertical_change(
            my_frame, my_full_history
        )
        
        # 5. Contact Duration
        contact_duration = self._calculate_contact_duration(
            my_frame, my_full_history, opp_full_history
        )
        
        # 6. Control Overlap
        control_overlap = self._calculate_overlap(my_bbox, opp_bbox)
        
        # 7. Recovery Latency
        recovery_latency = self._calculate_recovery_latency(
            my_frame, my_full_history
        )
        
        # 8. Scramble Entropy
        scramble_entropy = self._calculate_scramble_entropy(
            my_frame, opp_frame, my_full_history, opp_full_history
        )
        
        return SignalFrame(
            t=timestamp,
            distance_delta=distance_delta,
            forward_velocity=forward_velocity,
            lateral_displacement=lateral_displacement,
            vertical_level_change=vertical_level_change,
            contact_duration=contact_duration,
            control_overlap=control_overlap,
            recovery_latency=recovery_latency,
            scramble_entropy=scramble_entropy
        )
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def _get_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _calculate_distance_delta(self, my_frame, opp_frame, 
                                  my_history, opp_history) -> float:
        """
        Signal 1: Change in distance between fighters.
        Positive = moving apart, Negative = moving together.
        """
        my_center = self._get_center(my_frame.bbox)
        opp_center = self._get_center(opp_frame.bbox)
        current_distance = self._get_distance(my_center, opp_center)
        
        # Find previous frame (1 second ago)
        prev_time = my_frame.timestamp - 1.0
        my_prev = self._find_closest_frame(my_history, prev_time)
        opp_prev = self._find_closest_frame(opp_history, prev_time)
        
        if my_prev and opp_prev and my_prev.bbox and opp_prev.bbox:
            my_prev_center = self._get_center(my_prev.bbox)
            opp_prev_center = self._get_center(opp_prev.bbox)
            prev_distance = self._get_distance(my_prev_center, opp_prev_center)
            
            delta = (current_distance - prev_distance) / self.max_distance
            return np.clip(delta, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_forward_velocity(self, my_frame, my_center, 
                                    opp_center, my_history) -> float:
        """
        Signal 2: Forward/backward movement of my fighter.
        Positive = moving toward opponent, Negative = moving away.
        """
        prev_time = my_frame.timestamp - 1.0
        my_prev = self._find_closest_frame(my_history, prev_time)
        
        if my_prev and my_prev.bbox:
            my_prev_center = self._get_center(my_prev.bbox)
            
            # Vector toward opponent
            to_opponent = np.array([opp_center[0] - my_center[0], 
                                   opp_center[1] - my_center[1]])
            to_opponent_norm = np.linalg.norm(to_opponent)
            
            if to_opponent_norm > 0:
                to_opponent = to_opponent / to_opponent_norm
                
                # Movement vector
                movement = np.array([my_center[0] - my_prev_center[0],
                                    my_center[1] - my_prev_center[1]])
                
                # Project movement onto opponent direction
                forward_component = np.dot(movement, to_opponent)
                
                # Normalize
                velocity = forward_component / self.max_velocity
                return np.clip(velocity, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_lateral_displacement(self, my_frame, my_center,
                                       opp_center, my_history) -> float:
        """
        Signal 3: Sideways movement perpendicular to opponent direction.
        """
        prev_time = my_frame.timestamp - 1.0
        my_prev = self._find_closest_frame(my_history, prev_time)
        
        if my_prev and my_prev.bbox:
            my_prev_center = self._get_center(my_prev.bbox)
            
            # Vector toward opponent
            to_opponent = np.array([opp_center[0] - my_center[0],
                                   opp_center[1] - my_center[1]])
            to_opponent_norm = np.linalg.norm(to_opponent)
            
            if to_opponent_norm > 0:
                to_opponent = to_opponent / to_opponent_norm
                
                # Perpendicular vector (rotate 90 degrees)
                perpendicular = np.array([-to_opponent[1], to_opponent[0]])
                
                # Movement vector
                movement = np.array([my_center[0] - my_prev_center[0],
                                    my_center[1] - my_prev_center[1]])
                
                # Project onto perpendicular
                lateral = np.dot(movement, perpendicular)
                
                # Normalize
                displacement = lateral / self.max_velocity
                return np.clip(displacement, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_vertical_change(self, my_frame, my_history) -> float:
        """
        Signal 4: Change in vertical position (Y-axis).
        Positive = moving down, Negative = moving up.
        """
        my_center = self._get_center(my_frame.bbox)
        
        prev_time = my_frame.timestamp - 1.0
        my_prev = self._find_closest_frame(my_history, prev_time)
        
        if my_prev and my_prev.bbox:
            my_prev_center = self._get_center(my_prev.bbox)
            
            vertical_change = (my_center[1] - my_prev_center[1]) / self.frame_height
            return np.clip(vertical_change, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_contact_duration(self, my_frame, my_history, opp_history) -> float:
        """
        Signal 5: Time spent within clinch threshold.
        Returns seconds in contact in past window.
        """
        contact_count = 0
        window_frames = int(self.fps * 3)  # Look back 3 seconds
        
        start_idx = max(0, my_frame.frame_num - window_frames)
        
        for i in range(start_idx, my_frame.frame_num + 1):
            my_f = self._find_frame_by_num(my_history, i)
            opp_f = self._find_frame_by_num(opp_history, i)
            
            if my_f and opp_f and my_f.bbox and opp_f.bbox:
                my_c = self._get_center(my_f.bbox)
                opp_c = self._get_center(opp_f.bbox)
                distance = self._get_distance(my_c, opp_c)
                
                if distance < self.CLINCH_DISTANCE_THRESHOLD:
                    contact_count += 1
        
        contact_duration = contact_count / self.fps
        return min(contact_duration, 3.0)  # Cap at 3 seconds
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int],
                          bbox2: Tuple[int, int, int, int]) -> float:
        """
        Signal 6: Percentage of bounding box overlap.
        Indicates control/grappling.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        overlap = intersection_area / union_area
        return min(overlap, 1.0)
    
    def _calculate_recovery_latency(self, my_frame, my_history) -> float:
        """
        Signal 7: Time until motion stabilizes after rapid movement.
        Measures recovery from intense action.
        """
        # Look back for recent rapid movement
        window = int(self.fps * 2)  # 2 second window
        start_idx = max(0, my_frame.frame_num - window)
        
        rapid_movement_frame = None
        
        # Find most recent rapid movement
        for i in range(my_frame.frame_num, start_idx, -1):
            curr = self._find_frame_by_num(my_history, i)
            prev = self._find_frame_by_num(my_history, i - 1)
            
            if curr and prev and curr.bbox and prev.bbox:
                curr_c = self._get_center(curr.bbox)
                prev_c = self._get_center(prev.bbox)
                movement = self._get_distance(curr_c, prev_c)
                
                if movement > 50:  # Rapid movement threshold
                    rapid_movement_frame = i
                    break
        
        if rapid_movement_frame is None:
            return 0.0  # No recent rapid movement
        
        # Count frames until stabilization
        stabilized_frame = None
        for i in range(rapid_movement_frame + 1, min(my_frame.frame_num + 1, 
                                                      rapid_movement_frame + window)):
            curr = self._find_frame_by_num(my_history, i)
            prev = self._find_frame_by_num(my_history, i - 1)
            
            if curr and prev and curr.bbox and prev.bbox:
                curr_c = self._get_center(curr.bbox)
                prev_c = self._get_center(prev.bbox)
                movement = self._get_distance(curr_c, prev_c)
                
                if movement < self.STABILIZATION_THRESHOLD:
                    stabilized_frame = i
                    break
        
        if stabilized_frame:
            latency = (stabilized_frame - rapid_movement_frame) / self.fps
            return min(latency, 2.0)  # Cap at 2 seconds
        
        return 2.0  # Still recovering
    
    def _calculate_scramble_entropy(self, my_frame, opp_frame,
                                   my_history, opp_history) -> float:
        """
        Signal 8: Degree of chaotic/unpredictable movement.
        High entropy = scramble/chaotic, Low = controlled.
        """
        window = int(self.fps * 1)  # 1 second window
        start_idx = max(0, my_frame.frame_num - window)
        
        my_movements = []
        opp_movements = []
        
        # Collect movement vectors
        for i in range(start_idx, my_frame.frame_num):
            my_curr = self._find_frame_by_num(my_history, i + 1)
            my_prev = self._find_frame_by_num(my_history, i)
            opp_curr = self._find_frame_by_num(opp_history, i + 1)
            opp_prev = self._find_frame_by_num(opp_history, i)
            
            if my_curr and my_prev and my_curr.bbox and my_prev.bbox:
                my_c1 = self._get_center(my_curr.bbox)
                my_c0 = self._get_center(my_prev.bbox)
                my_movements.append(self._get_distance(my_c1, my_c0))
            
            if opp_curr and opp_prev and opp_curr.bbox and opp_prev.bbox:
                opp_c1 = self._get_center(opp_curr.bbox)
                opp_c0 = self._get_center(opp_prev.bbox)
                opp_movements.append(self._get_distance(opp_c1, opp_c0))
        
        if not my_movements or not opp_movements:
            return 0.0
        
        # Calculate variability (standard deviation)
        my_std = np.std(my_movements) if len(my_movements) > 1 else 0
        opp_std = np.std(opp_movements) if len(opp_movements) > 1 else 0
        
        # Normalize by max possible std
        max_std = self.max_velocity / 2
        
        combined_entropy = (my_std + opp_std) / (2 * max_std)
        return np.clip(combined_entropy, 0.0, 1.0)
    
    def _find_closest_frame(self, history: List, target_time: float):
        """Find frame closest to target timestamp."""
        if not history:
            return None
        
        closest = min(history, key=lambda f: abs(f.timestamp - target_time))
        
        # Only return if within 0.5 seconds
        if abs(closest.timestamp - target_time) < 0.5:
            return closest
        
        return None
    
    def _find_frame_by_num(self, history: List, frame_num: int):
        """Find frame by frame number."""
        for frame in history:
            if frame.frame_num == frame_num:
                return frame
        return None


def extract_behavioral_signals_from_tracker(tracker,
                                           video_id: str,
                                           fps: float,
                                           frame_width: int,
                                           frame_height: int) -> Dict:
    """
    Extract behavioral signals from DualFighterTracker.
    
    Args:
        tracker: DualFighterTracker instance
        video_id: Video identifier
        fps: Video FPS
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Dictionary with behavioral signals
    """
    if not tracker.has_opponent:
        print("\n⚠️  Cannot extract behavioral signals: no opponent tracked")
        return {
            "video_id": video_id,
            "signals": []
        }
    
    extractor = BehavioralSignalExtractor(fps, frame_width, frame_height)
    
    signals_data = extractor.extract_signals(
        tracker.my_tracker.tracking_history,
        tracker.opp_tracker.tracking_history,
        video_id
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BEHAVIORAL SIGNAL EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Video ID:        {video_id}")
    print(f"Total Signals:   {len(signals_data['signals'])} seconds")
    print(f"FPS:             {fps}")
    print(f"Resolution:      {frame_width}x{frame_height}")
    print("=" * 60)
    
    return signals_data


def save_behavioral_signals(signals_data: Dict,
                           output_path: str,
                           pretty: bool = True):
    """
    Save behavioral signals to JSON file.
    
    Args:
        signals_data: Dictionary with signals
        output_path: Path to output JSON file
        pretty: Whether to pretty-print JSON
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(signals_data, f, indent=2)
        else:
            json.dump(signals_data, f)
    
    print(f"\n💾 Behavioral signals saved: {output_path}")
