"""
Fighter Tracking Module - STABLE VERSION
Tracks selected fighters through video using OpenCV CSRT tracker.
NO machine learning. Pure OpenCV tracking.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrackingFrame:
    """Single frame tracking result."""
    frame_num: int
    timestamp: float
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    tracking_active: bool


class FighterTracker:
    """
    Tracks a single fighter through video using OpenCV CSRT tracker.
    """
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int], 
                 first_frame: np.ndarray,
                 tracker_type: str = "CSRT"):
        """
        Initialize tracker.
        
        Args:
            initial_bbox: Initial bounding box (x, y, w, h)
            first_frame: First frame of video (BGR format)
            tracker_type: "CSRT" or "KCF"
        """
        self.initial_bbox = initial_bbox
        self.tracker_type = tracker_type
        self.tracking_history: List[TrackingFrame] = []
        self.bbox_history = []  # For smoothing
        self.max_bbox_history = 5
        
        # Create tracker
        self.tracker = self._create_tracker(tracker_type)
        
        # Initialize with original frame (no preprocessing to avoid issues)
        success = self.tracker.init(first_frame, initial_bbox)
        if not success:
            raise ValueError(f"Failed to initialize {tracker_type} tracker")
        
        # Store first result
        self.tracking_history.append(TrackingFrame(
            frame_num=0,
            timestamp=0.0,
            bbox=initial_bbox,
            confidence=1.0,
            tracking_active=True
        ))
        self.bbox_history.append(initial_bbox)
        
        print(f"✅ {tracker_type} tracker initialized")
        print(f"   Initial bbox: {initial_bbox}")
    
    def _create_tracker(self, tracker_type: str):
        """Create OpenCV tracker."""
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
    
    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Smooth bounding box using exponential moving average."""
        if len(self.bbox_history) < 2:
            return bbox
        
        recent = self.bbox_history[-min(self.max_bbox_history, len(self.bbox_history)):]
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        weights /= weights.sum()
        
        x = int(sum(w * b[0] for w, b in zip(weights, recent)))
        y = int(sum(w * b[1] for w, b in zip(weights, recent)))
        w = int(sum(w * b[2] for w, b in zip(weights, recent)))
        h = int(sum(w * b[3] for w, b in zip(weights, recent)))
        
        return (x, y, w, h)
    
    def _validate_bbox(self, bbox: Tuple[int, int, int, int], 
                      frame_width: int, frame_height: int) -> bool:
        """Validate bbox is reasonable."""
        x, y, w, h = bbox
        
        # Check bounds
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            return False
        
        # Check size
        if w < 20 or h < 20:
            return False
        
        # Check drastic size change
        if self.bbox_history:
            last_w, last_h = self.bbox_history[-1][2], self.bbox_history[-1][3]
            if w > last_w * 3 or h > last_h * 3 or w < last_w / 3 or h < last_h / 3:
                return False
        
        return True
    
    def update(self, frame: np.ndarray, frame_num: int, timestamp: float) -> TrackingFrame:
        """Update tracker with new frame."""
        success, bbox = self.tracker.update(frame)
        
        if success:
            bbox = tuple(map(int, bbox))
            frame_height, frame_width = frame.shape[:2]
            
            if self._validate_bbox(bbox, frame_width, frame_height):
                self.bbox_history.append(bbox)
                if len(self.bbox_history) > self.max_bbox_history:
                    self.bbox_history.pop(0)
                
                smoothed = self._smooth_bbox(bbox)
                
                result = TrackingFrame(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    bbox=smoothed,
                    confidence=1.0,
                    tracking_active=True
                )
            else:
                success = False
        
        if not success:
            result = TrackingFrame(
                frame_num=frame_num,
                timestamp=timestamp,
                bbox=None,
                confidence=0.0,
                tracking_active=False
            )
        
        self.tracking_history.append(result)
        return result
    
    def get_tracking_history(self) -> List[TrackingFrame]:
        """Get complete tracking history."""
        return self.tracking_history


class DualFighterTracker:
    """Manages tracking for both fighters."""
    
    def __init__(self, 
                 my_fighter_bbox: Tuple[int, int, int, int],
                 opponent_bbox: Optional[Tuple[int, int, int, int]],
                 first_frame: np.ndarray,
                 tracker_type: str = "CSRT"):
        """Initialize dual tracker."""
        self.my_fighter_tracker = FighterTracker(my_fighter_bbox, first_frame, tracker_type)
        
        if opponent_bbox:
            self.opponent_tracker = FighterTracker(opponent_bbox, first_frame, tracker_type)
            self.has_opponent = True
        else:
            self.opponent_tracker = None
            self.has_opponent = False
            print("⚠️  Opponent not selected - tracking MY fighter only")
    
    def track_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Dict[str, TrackingFrame]:
        """Track both fighters in current frame."""
        my_result = self.my_fighter_tracker.update(frame, frame_num, timestamp)
        
        if self.has_opponent:
            opp_result = self.opponent_tracker.update(frame, frame_num, timestamp)
        else:
            opp_result = TrackingFrame(frame_num, timestamp, None, 0.0, False)
        
        return {"my_fighter": my_result, "opponent": opp_result}
    
    def get_tracking_summary(self) -> Dict:
        """Get tracking summary statistics."""
        my_history = self.my_fighter_tracker.get_tracking_history()
        my_tracked = sum(1 for h in my_history if h.tracking_active)
        
        summary = {
            "my_fighter": {
                "total_frames": len(my_history),
                "tracked_frames": my_tracked,
                "lost_frames": len(my_history) - my_tracked,
                "tracking_rate": my_tracked / len(my_history) if my_history else 0.0
            }
        }
        
        if self.has_opponent:
            opp_history = self.opponent_tracker.get_tracking_history()
            opp_tracked = sum(1 for h in opp_history if h.tracking_active)
            
            summary["opponent"] = {
                "total_frames": len(opp_history),
                "tracked_frames": opp_tracked,
                "lost_frames": len(opp_history) - opp_tracked,
                "tracking_rate": opp_tracked / len(opp_history) if opp_history else 0.0
            }
        
        return summary


def track_video(video_path: str,
                my_fighter_bbox: Tuple[int, int, int, int],
                opponent_bbox: Optional[Tuple[int, int, int, int]] = None,
                tracker_type: str = "CSRT",
                progress_interval: int = 100,
                show_video: bool = True,
                save_video: bool = False,
                output_path: str = "tracked_output.mp4") -> DualFighterTracker:
    """Track fighters through entire video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    print("\n" + "=" * 60)
    print("🎯 STARTING FIGHTER TRACKING")
    print("=" * 60)
    print(f"Tracker:      {tracker_type}")
    print(f"Total frames: {total_frames}")
    print(f"Duration:     {total_frames / fps:.2f} seconds")
    if show_video:
        print(f"Display:      ENABLED (Press 'Q' to quit, 'P' to pause)")
    print("=" * 60 + "\n")
    
    tracker = DualFighterTracker(my_fighter_bbox, opponent_bbox, first_frame, tracker_type)
    
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if show_video:
        window_name = "MMA Fighter Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    
    frame_num = 1
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_num / fps
            results = tracker.track_frame(frame, frame_num, timestamp)
            
            if show_video or save_video:
                display = frame.copy()
                
                # MY FIGHTER (RED)
                if results["my_fighter"].tracking_active and results["my_fighter"].bbox:
                    x, y, w, h = results["my_fighter"].bbox
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(display, "MY FIGHTER", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "MY FIGHTER: LOST", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # OPPONENT (CYAN)
                if results["opponent"].tracking_active and results["opponent"].bbox:
                    x, y, w, h = results["opponent"].bbox
                    cv2.rectangle(display, (x, y), (x+w, y+h), (255, 255, 0), 3)
                    cv2.putText(display, "OPPONENT", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif tracker.has_opponent:
                    cv2.putText(display, "OPPONENT: LOST", (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Info
                info = f"Frame: {frame_num}/{total_frames} | {timestamp:.2f}s"
                cv2.putText(display, info, (20, height-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Progress bar
                prog = frame_num / total_frames
                bar_w, bar_h = width-40, 20
                cv2.rectangle(display, (20, height-60), (20+bar_w, height-40), (50, 50, 50), -1)
                cv2.rectangle(display, (20, height-60), (20+int(bar_w*prog), height-40), (0, 255, 0), -1)
                cv2.putText(display, f"{prog*100:.1f}%", (30+bar_w, height-45),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if show_video:
                    cv2.imshow(window_name, display)
                if save_video and video_writer:
                    video_writer.write(display)
            
            frame_num += 1
        
        if show_video:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n⚠️  User quit")
                break
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                print("⏸️  PAUSED" if paused else "▶️  RESUMED")
        
        if frame_num % progress_interval == 0:
            print(f"  Progress: {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    if video_writer:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ TRACKING COMPLETE")
    print("=" * 60)
    
    summary = tracker.get_tracking_summary()
    print(f"\n🔴 MY FIGHTER: {summary['my_fighter']['tracked_frames']}/{summary['my_fighter']['total_frames']} " +
          f"({summary['my_fighter']['tracking_rate']*100:.1f}%)")
    
    if tracker.has_opponent:
        print(f"🔵 OPPONENT: {summary['opponent']['tracked_frames']}/{summary['opponent']['total_frames']} " +
              f"({summary['opponent']['tracking_rate']*100:.1f}%)")
    
    print("=" * 60 + "\n")
    
    return tracker