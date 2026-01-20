"""
Behavioral Signal Extraction for MMA Fighter Analysis
Extracts movement and positional signals from bounding box tracking data.

Signals Generated (per second):
- distance_delta: Change in distance between fighters
- forward_velocity: Forward/backward movement of selected fighter
- lateral_displacement: Sideways movement
- vertical_level_change: Change in vertical bbox center
- contact_duration: Time spent within clinch threshold
- control_overlap: % of bounding box overlap
- recovery_latency: Time until motion stabilizes
- scramble_entropy: Degree of chaotic movement

All values are normalized/relative (no real-world units).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class BehavioralSignalExtractor:
    """
    Extracts behavioral movement signals from fighter tracking data.
    
    Processes bounding box positions frame-by-frame and computes
    second-by-second behavioral metrics for analysis.
    """
    
    # === CONFIGURABLE THRESHOLDS ===
    CLINCH_DISTANCE_RATIO = 0.15  # Distance < 15% of frame width = clinch
    STABILIZATION_VELOCITY_THRESHOLD = 5.0  # Pixels/frame for "stable"
    SCRAMBLE_VELOCITY_THRESHOLD = 15.0  # Pixels/frame for "chaotic"
    HISTORY_WINDOW_SECONDS = 2.0  # Look-back window for signals
    
    def __init__(self, frame_width: int, frame_height: int, fps: float):
        """
        Initialize signal extractor.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Calculate normalization factors
        self.frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        self.clinch_distance_threshold = frame_width * self.CLINCH_DISTANCE_RATIO
        
        # Per-second aggregation window
        self.frames_per_second = int(fps)
        
        # Frame-by-frame history (for computing signals)
        self.frame_history = []  # List of dicts with frame data
        
        # Computed signals (per second)
        self.signals = []
        
        # Current second being processed
        self.current_second = 0
        
        print(f"📊 Behavioral Signal Extractor Initialized")
        print(f"   Resolution: {frame_width}x{frame_height}")
        print(f"   FPS: {fps}")
        print(f"   Clinch threshold: {self.clinch_distance_threshold:.1f} px")
    
    def update(self, tracker_status: Dict):
        """
        Process frame and update signal history.
        
        Args:
            tracker_status: Status dict from FighterTracker.get_status()
        """
        frame_num = tracker_status['frame']
        
        # Extract bounding boxes
        my_bbox = tracker_status['my_fighter']['bbox']
        opp_bbox = tracker_status['opponent']['bbox']
        
        # Extract reliability
        my_reliable = tracker_status['my_fighter']['reliable']
        opp_reliable = tracker_status['opponent']['reliable']
        
        # Store frame data
        frame_data = {
            'frame': frame_num,
            'timestamp': frame_num / self.fps,
            'my_bbox': my_bbox,
            'opp_bbox': opp_bbox,
            'my_reliable': my_reliable,
            'opp_reliable': opp_reliable,
            'in_clinch': tracker_status.get('in_clinch', False)
        }
        
        self.frame_history.append(frame_data)
        
        # Check if we've completed a second
        current_second = int(frame_num / self.fps)
        if current_second > self.current_second:
            # Compute signals for the completed second
            self._compute_signals_for_second(self.current_second)
            self.current_second = current_second
    
    def finalize(self):
        """
        Finalize signal extraction.
        Call after processing all frames to compute final second's signals.
        """
        # Compute signals for the last partial second
        if len(self.frame_history) > 0:
            last_frame = self.frame_history[-1]
            last_second = int(last_frame['timestamp'])
            if last_second >= self.current_second:
                self._compute_signals_for_second(last_second)
        
        print(f"✅ Signal extraction finalized")
        print(f"   Total signals computed: {len(self.signals)} seconds")
    
    def _compute_signals_for_second(self, second: int):
        """
        Compute all behavioral signals for a given second.
        
        Args:
            second: The second (0-indexed) to compute signals for
        """
        # Get frames for this second
        start_frame = second * self.frames_per_second
        end_frame = start_frame + self.frames_per_second
        
        second_frames = [
            f for f in self.frame_history
            if start_frame <= f['frame'] < end_frame
        ]
        
        if len(second_frames) == 0:
            return
        
        # Compute each signal
        signals = {
            't': second,
            'distance_delta': self._compute_distance_delta(second_frames),
            'forward_velocity': self._compute_forward_velocity(second_frames),
            'lateral_displacement': self._compute_lateral_displacement(second_frames),
            'vertical_level_change': self._compute_vertical_level_change(second_frames),
            'contact_duration': self._compute_contact_duration(second_frames),
            'control_overlap': self._compute_control_overlap(second_frames),
            'recovery_latency': self._compute_recovery_latency(second_frames),
            'scramble_entropy': self._compute_scramble_entropy(second_frames)
        }
        
        self.signals.append(signals)
    
    def _compute_distance_delta(self, frames: List[Dict]) -> float:
        """
        Compute change in distance between fighters over the second.
        
        Returns:
            Normalized distance change (-1 to 1, where negative = closing distance)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        # Get first and last distances
        first_dist = self._bbox_distance(valid_frames[0]['my_bbox'], valid_frames[0]['opp_bbox'])
        last_dist = self._bbox_distance(valid_frames[-1]['my_bbox'], valid_frames[-1]['opp_bbox'])
        
        # Normalize by frame diagonal
        delta = (last_dist - first_dist) / self.frame_diagonal
        
        return round(delta, 3)
    
    def _compute_forward_velocity(self, frames: List[Dict]) -> float:
        """
        Compute forward/backward velocity of my fighter.
        
        Forward = moving toward opponent (positive)
        Backward = moving away from opponent (negative)
        
        Returns:
            Normalized velocity (-1 to 1)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        # Calculate displacement toward/away from opponent
        total_forward = 0.0
        
        for i in range(1, len(valid_frames)):
            prev_frame = valid_frames[i-1]
            curr_frame = valid_frames[i]
            
            # Get centers
            my_center_prev = self._bbox_center(prev_frame['my_bbox'])
            my_center_curr = self._bbox_center(curr_frame['my_bbox'])
            opp_center = self._bbox_center(curr_frame['opp_bbox'])
            
            # Calculate distance change
            dist_prev = np.linalg.norm(np.array(my_center_prev) - np.array(opp_center))
            dist_curr = np.linalg.norm(np.array(my_center_curr) - np.array(opp_center))
            
            # Positive if moving toward opponent
            forward_movement = dist_prev - dist_curr
            total_forward += forward_movement
        
        # Normalize by diagonal and number of frames
        avg_forward = total_forward / len(valid_frames)
        normalized = avg_forward / (self.frame_diagonal / self.fps)
        
        return round(np.clip(normalized, -1, 1), 3)
    
    def _compute_lateral_displacement(self, frames: List[Dict]) -> float:
        """
        Compute sideways movement of my fighter (perpendicular to opponent).
        
        Returns:
            Normalized lateral displacement (0 to 1)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        # Calculate average lateral movement
        total_lateral = 0.0
        
        for i in range(1, len(valid_frames)):
            prev_frame = valid_frames[i-1]
            curr_frame = valid_frames[i]
            
            my_center_prev = np.array(self._bbox_center(prev_frame['my_bbox']))
            my_center_curr = np.array(self._bbox_center(curr_frame['my_bbox']))
            opp_center = np.array(self._bbox_center(curr_frame['opp_bbox']))
            
            # Vector from my fighter to opponent
            to_opponent = opp_center - my_center_curr
            to_opponent_norm = np.linalg.norm(to_opponent)
            
            if to_opponent_norm > 0:
                # Unit vector toward opponent
                forward_dir = to_opponent / to_opponent_norm
                
                # My fighter's displacement
                displacement = my_center_curr - my_center_prev
                
                # Project displacement onto perpendicular (lateral) direction
                # Perpendicular vector in 2D: (-y, x)
                lateral_dir = np.array([-forward_dir[1], forward_dir[0]])
                lateral_movement = abs(np.dot(displacement, lateral_dir))
                
                total_lateral += lateral_movement
        
        # Normalize
        avg_lateral = total_lateral / len(valid_frames)
        normalized = avg_lateral / (self.frame_diagonal / self.fps)
        
        return round(np.clip(normalized, 0, 1), 3)
    
    def _compute_vertical_level_change(self, frames: List[Dict]) -> float:
        """
        Compute change in vertical position of my fighter.
        
        Positive = moving up (standing up)
        Negative = moving down (going to ground)
        
        Returns:
            Normalized vertical change (-1 to 1)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        # Get first and last vertical centers
        first_y = self._bbox_center(valid_frames[0]['my_bbox'])[1]
        last_y = self._bbox_center(valid_frames[-1]['my_bbox'])[1]
        
        # Negative delta = moving up (y increases downward in image coords)
        delta_y = first_y - last_y
        
        # Normalize by frame height
        normalized = delta_y / self.frame_height
        
        return round(np.clip(normalized, -1, 1), 3)
    
    def _compute_contact_duration(self, frames: List[Dict]) -> float:
        """
        Compute total time spent in clinch range during this second.
        
        Returns:
            Duration in seconds (0 to 1.0)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) == 0:
            return 0.0
        
        # Count frames in clinch
        clinch_frames = sum(
            1 for f in valid_frames
            if self._bbox_distance(f['my_bbox'], f['opp_bbox']) < self.clinch_distance_threshold
        )
        
        # Convert to seconds
        duration = clinch_frames / self.fps
        
        return round(duration, 3)
    
    def _compute_control_overlap(self, frames: List[Dict]) -> float:
        """
        Compute average bounding box overlap (IoU) during this second.
        
        Returns:
            Average IoU (0 to 1)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) == 0:
            return 0.0
        
        # Calculate average IoU
        total_iou = sum(
            self._calculate_iou(f['my_bbox'], f['opp_bbox'])
            for f in valid_frames
        )
        
        avg_iou = total_iou / len(valid_frames)
        
        return round(avg_iou, 3)
    
    def _compute_recovery_latency(self, frames: List[Dict]) -> float:
        """
        Compute time until motion stabilizes after high-velocity movement.
        
        Returns:
            Latency in seconds (0 to 1.0)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 3:
            return 0.0
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(valid_frames)):
            prev_center = np.array(self._bbox_center(valid_frames[i-1]['my_bbox']))
            curr_center = np.array(self._bbox_center(valid_frames[i]['my_bbox']))
            velocity = np.linalg.norm(curr_center - prev_center)
            velocities.append(velocity)
        
        # Find if there was high velocity followed by stabilization
        recovery_frames = 0
        found_high_velocity = False
        
        for velocity in velocities:
            if velocity > self.SCRAMBLE_VELOCITY_THRESHOLD:
                found_high_velocity = True
                recovery_frames = 0
            elif found_high_velocity:
                if velocity < self.STABILIZATION_VELOCITY_THRESHOLD:
                    recovery_frames += 1
                    if recovery_frames > 5:  # Stabilized
                        break
                else:
                    recovery_frames += 1
        
        # Convert to seconds (capped at 1 second)
        latency = min(recovery_frames / self.fps, 1.0)
        
        return round(latency, 3)
    
    def _compute_scramble_entropy(self, frames: List[Dict]) -> float:
        """
        Compute degree of chaotic/unpredictable movement.
        
        High entropy = erratic, changing direction frequently
        Low entropy = smooth, predictable movement
        
        Returns:
            Entropy score (0 to 1)
        """
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 3:
            return 0.0
        
        # Calculate velocity vectors
        velocities = []
        for i in range(1, len(valid_frames)):
            prev_center = np.array(self._bbox_center(valid_frames[i-1]['my_bbox']))
            curr_center = np.array(self._bbox_center(valid_frames[i]['my_bbox']))
            velocity = curr_center - prev_center
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate direction changes
        direction_changes = 0
        total_velocity = 0
        
        for i in range(1, len(velocities)):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            
            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)
            
            total_velocity += curr_speed
            
            if prev_speed > 1 and curr_speed > 1:
                # Calculate angle between velocity vectors
                cos_angle = np.dot(prev_vel, curr_vel) / (prev_speed * curr_speed)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Significant direction change if angle > 45 degrees
                if angle > np.pi / 4:
                    direction_changes += 1
        
        # Entropy = frequency of direction changes weighted by total movement
        if len(velocities) > 0:
            change_rate = direction_changes / len(velocities)
            avg_velocity = total_velocity / len(velocities)
            
            # Normalize velocity component
            velocity_factor = min(avg_velocity / self.SCRAMBLE_VELOCITY_THRESHOLD, 1.0)
            
            entropy = change_rate * velocity_factor
        else:
            entropy = 0.0
        
        return round(np.clip(entropy, 0, 1), 3)
    
    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate center of bounding box."""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def _bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between bbox centers."""
        c1 = self._bbox_center(bbox1)
        c2 = self._bbox_center(bbox2)
        return np.linalg.norm(np.array(c1) - np.array(c2))
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_signals(self) -> List[Dict]:
        """
        Get all computed signals.
        
        Returns:
            List of signal dictionaries (one per second)
        """
        return self.signals
    
    def export_json(self, video_name: str, output_path: str, include_metadata: bool = True):
        """
        Export signals to JSON file.
        
        Args:
            video_name: Name/ID of the video
            output_path: Path to output JSON file
            include_metadata: Include extraction metadata
        """
        output_data = {
            "video_id": video_name,
            "signals": self.signals
        }
        
        if include_metadata:
            output_data["metadata"] = {
                "fps": self.fps,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "total_seconds": len(self.signals),
                "clinch_distance_threshold_px": round(self.clinch_distance_threshold, 1),
                "signal_descriptions": {
                    "distance_delta": "Change in distance between fighters (-1=closing, +1=separating)",
                    "forward_velocity": "Movement toward opponent (+) or away (-)",
                    "lateral_displacement": "Sideways movement (0=none, 1=max)",
                    "vertical_level_change": "Vertical movement (+=up, -=down)",
                    "contact_duration": "Time in clinch range (seconds)",
                    "control_overlap": "Bounding box overlap (0=none, 1=complete)",
                    "recovery_latency": "Time to stabilize after high movement (seconds)",
                    "scramble_entropy": "Movement chaos/unpredictability (0=smooth, 1=erratic)"
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📁 Behavioral signals saved: {output_path}")
    
    def print_summary(self):
        """Print human-readable summary of signals."""
        if len(self.signals) == 0:
            print("\n⚠️  No signals computed yet")
            return
        
        print("\n" + "=" * 60)
        print("📊 BEHAVIORAL SIGNALS SUMMARY")
        print("=" * 60)
        
        # Calculate averages
        avg_signals = {
            key: np.mean([s[key] for s in self.signals])
            for key in self.signals[0].keys() if key != 't'
        }
        
        print(f"\nTotal seconds analyzed: {len(self.signals)}")
        print(f"\nAverage Signal Values:")
        print(f"  • Distance Delta:        {avg_signals['distance_delta']:+.3f}")
        print(f"  • Forward Velocity:      {avg_signals['forward_velocity']:+.3f}")
        print(f"  • Lateral Displacement:  {avg_signals['lateral_displacement']:.3f}")
        print(f"  • Vertical Level Change: {avg_signals['vertical_level_change']:+.3f}")
        print(f"  • Contact Duration:      {avg_signals['contact_duration']:.3f}s")
        print(f"  • Control Overlap:       {avg_signals['control_overlap']:.3f}")
        print(f"  • Recovery Latency:      {avg_signals['recovery_latency']:.3f}s")
        print(f"  • Scramble Entropy:      {avg_signals['scramble_entropy']:.3f}")
        
        # Find most active seconds
        high_entropy = sorted(self.signals, key=lambda s: s['scramble_entropy'], reverse=True)[:3]
        high_contact = sorted(self.signals, key=lambda s: s['contact_duration'], reverse=True)[:3]
        
        print(f"\n🔥 Most Chaotic Moments (scramble_entropy):")
        for s in high_entropy:
            print(f"  • t={s['t']:3d}s: entropy={s['scramble_entropy']:.3f}")
        
        print(f"\n🤼 Most Contact Time:")
        for s in high_contact:
            print(f"  • t={s['t']:3d}s: duration={s['contact_duration']:.3f}s")
        
        print("=" * 60 + "\n")