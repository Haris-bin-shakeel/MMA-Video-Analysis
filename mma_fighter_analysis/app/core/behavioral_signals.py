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
   
    
    # === CONFIGURABLE THRESHOLDS ===
    CLINCH_DISTANCE_RATIO = 0.15  
    STABILIZATION_VELOCITY_THRESHOLD = 5.0  
    SCRAMBLE_VELOCITY_THRESHOLD = 15.0  
    HISTORY_WINDOW_SECONDS = 2.0  
    
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
        
       
        my_bbox = tracker_status['my_fighter']['bbox']
        opp_bbox = tracker_status['opponent']['bbox']
        
      
        my_reliable = tracker_status['my_fighter']['reliable']
        opp_reliable = tracker_status['opponent']['reliable']
        
      
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
        
       
        current_second = int(frame_num / self.fps)
        if current_second > self.current_second:
            self._compute_signals_for_second(self.current_second)
            self.current_second = current_second
    
    def finalize(self):
        """
        Finalize signal extraction.
        Call after processing all frames to compute final second's signals.
        """
        if len(self.frame_history) > 0:
            last_frame = self.frame_history[-1]
            last_second = int(last_frame['timestamp'])
            if last_second >= self.current_second:
                self._compute_signals_for_second(last_second)
        
        print(f"✅ Signal extraction finalized")
        print(f"   Total signals computed: {len(self.signals)} seconds")
    
    def _compute_signals_for_second(self, second: int):
       
        start_frame = second * self.frames_per_second
        end_frame = start_frame + self.frames_per_second
        
        second_frames = [
            f for f in self.frame_history
            if start_frame <= f['frame'] < end_frame
        ]
        
        if len(second_frames) == 0:
            return
        
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
       
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        first_dist = self._bbox_distance(valid_frames[0]['my_bbox'], valid_frames[0]['opp_bbox'])
        last_dist = self._bbox_distance(valid_frames[-1]['my_bbox'], valid_frames[-1]['opp_bbox'])
        
        delta = (last_dist - first_dist) / self.frame_diagonal
        
        return round(delta, 3)
    
    def _compute_forward_velocity(self, frames: List[Dict]) -> float:
       
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        total_forward = 0.0
        
        for i in range(1, len(valid_frames)):
            prev_frame = valid_frames[i-1]
            curr_frame = valid_frames[i]
            
            my_center_prev = self._bbox_center(prev_frame['my_bbox'])
            my_center_curr = self._bbox_center(curr_frame['my_bbox'])
            opp_center = self._bbox_center(curr_frame['opp_bbox'])
            
            dist_prev = np.linalg.norm(np.array(my_center_prev) - np.array(opp_center))
            dist_curr = np.linalg.norm(np.array(my_center_curr) - np.array(opp_center))
            
            forward_movement = dist_prev - dist_curr
            total_forward += forward_movement
        
        avg_forward = total_forward / len(valid_frames)
        normalized = avg_forward / (self.frame_diagonal / self.fps)
        
        return round(np.clip(normalized, -1, 1), 3)
    
    def _compute_lateral_displacement(self, frames: List[Dict]) -> float:
       
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        total_lateral = 0.0
        
        for i in range(1, len(valid_frames)):
            prev_frame = valid_frames[i-1]
            curr_frame = valid_frames[i]
            
            my_center_prev = np.array(self._bbox_center(prev_frame['my_bbox']))
            my_center_curr = np.array(self._bbox_center(curr_frame['my_bbox']))
            opp_center = np.array(self._bbox_center(curr_frame['opp_bbox']))
            
            to_opponent = opp_center - my_center_curr
            to_opponent_norm = np.linalg.norm(to_opponent)
            
            if to_opponent_norm > 0:
                forward_dir = to_opponent / to_opponent_norm
                
                displacement = my_center_curr - my_center_prev
                
                
                lateral_dir = np.array([-forward_dir[1], forward_dir[0]])
                lateral_movement = abs(np.dot(displacement, lateral_dir))
                
                total_lateral += lateral_movement
        
        # Normalize
        avg_lateral = total_lateral / len(valid_frames)
        normalized = avg_lateral / (self.frame_diagonal / self.fps)
        
        return round(np.clip(normalized, 0, 1), 3)
    
    def _compute_vertical_level_change(self, frames: List[Dict]) -> float:
        
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 2:
            return 0.0
        
        first_y = self._bbox_center(valid_frames[0]['my_bbox'])[1]
        last_y = self._bbox_center(valid_frames[-1]['my_bbox'])[1]
        
        delta_y = first_y - last_y
        
        normalized = delta_y / self.frame_height
        
        return round(np.clip(normalized, -1, 1), 3)
    
    def _compute_contact_duration(self, frames: List[Dict]) -> float:
       
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None and f['opp_bbox'] is not None
        ]
        
        if len(valid_frames) == 0:
            return 0.0
        
        clinch_frames = sum(
            1 for f in valid_frames
            if self._bbox_distance(f['my_bbox'], f['opp_bbox']) < self.clinch_distance_threshold
        )
        
        duration = clinch_frames / self.fps
        
        return round(duration, 3)
    
    def _compute_control_overlap(self, frames: List[Dict]) -> float:
       
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
        
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(valid_frames)):
            prev_center = np.array(self._bbox_center(valid_frames[i-1]['my_bbox']))
            curr_center = np.array(self._bbox_center(valid_frames[i]['my_bbox']))
            velocity = np.linalg.norm(curr_center - prev_center)
            velocities.append(velocity)
        
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
        
        latency = min(recovery_frames / self.fps, 1.0)
        
        return round(latency, 3)
    
    def _compute_scramble_entropy(self, frames: List[Dict]) -> float:
       
        valid_frames = [
            f for f in frames
            if f['my_bbox'] is not None
        ]
        
        if len(valid_frames) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(valid_frames)):
            prev_center = np.array(self._bbox_center(valid_frames[i-1]['my_bbox']))
            curr_center = np.array(self._bbox_center(valid_frames[i]['my_bbox']))
            velocity = curr_center - prev_center
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        direction_changes = 0
        total_velocity = 0
        
        for i in range(1, len(velocities)):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            
            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)
            
            total_velocity += curr_speed
            
            if prev_speed > 1 and curr_speed > 1:
                cos_angle = np.dot(prev_vel, curr_vel) / (prev_speed * curr_speed)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle > np.pi / 4:
                    direction_changes += 1
        
        if len(velocities) > 0:
            change_rate = direction_changes / len(velocities)
            avg_velocity = total_velocity / len(velocities)
            
            velocity_factor = min(avg_velocity / self.SCRAMBLE_VELOCITY_THRESHOLD, 1.0)
            
            entropy = change_rate * velocity_factor
        else:
            entropy = 0.0
        
        return round(np.clip(entropy, 0, 1), 3)
    
    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def _bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        
        c1 = self._bbox_center(bbox1)
        c2 = self._bbox_center(bbox2)
        return np.linalg.norm(np.array(c1) - np.array(c2))
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
       
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        
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
       
        return self.signals
    
    def export_json(self, video_name: str, output_path: str, include_metadata: bool = True):
      
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
