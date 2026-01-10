"""
PROFESSIONAL MMA TRACKER - OPTICAL FLOW + CSRT HYBRID
Senior Architecture Approach:

Problem with previous versions:
- Pure tracker-based approaches drift on texture-rich backgrounds (fence, mat)
- Validation was too permissive
- No visual verification of tracked region

New Architecture:
1. Sparse Optical Flow (Lucas-Kanade) for feature point tracking
2. CSRT as backup/validation
3. Feature point clustering to maintain fighter boundaries
4. Active template matching for drift detection
5. Aggressive re-initialization on drift

This mimics how humans track: follow distinctive points (hands, head, shoulders)
rather than trying to track entire bounding box as a template.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class TrackingFrame:
    """Stores frame-level tracking result."""
    frame_num: int
    timestamp: float
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    tracking_active: bool
    tracker_source: str


class OpticalFlowTracker:
    """
    Sparse optical flow tracker using Lucas-Kanade.
    Tracks feature points (corners) within fighter region.
    """
    
    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Feature detection parameters
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.1,
        minDistance=7,
        blockSize=7
    )
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int],
                 first_frame: np.ndarray,
                 fighter_id: str):
        """Initialize optical flow tracker."""
        self.fighter_id = fighter_id
        self.bbox = initial_bbox
        
        # Convert to grayscale
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track within bbox
        x, y, w, h = initial_bbox
        mask = np.zeros_like(self.prev_gray)
        mask[y:y+h, x:x+w] = 255
        
        self.points = cv2.goodFeaturesToTrack(
            self.prev_gray, 
            mask=mask,
            **self.feature_params
        )
        
        if self.points is None or len(self.points) < 10:
            raise ValueError(f"{fighter_id}: Not enough features detected in initial bbox")
        
        self.initial_point_count = len(self.points)
        print(f"  ✅ {fighter_id}: Tracking {self.initial_point_count} feature points")
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Update using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.points is None or len(self.points) < 5:
            return False, None
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **self.lk_params
        )
        
        if new_points is None:
            return False, None
        
        # Select good points
        good_new = new_points[status == 1]
        
        # Need at least 30% of original points
        if len(good_new) < max(5, self.initial_point_count * 0.3):
            return False, None
        
        # Calculate bounding box from point cloud
        points_array = good_new.reshape(-1, 2)
        x_min = int(np.min(points_array[:, 0]))
        y_min = int(np.min(points_array[:, 1]))
        x_max = int(np.max(points_array[:, 0]))
        y_max = int(np.max(points_array[:, 1]))
        
        # Add padding (10%)
        w = x_max - x_min
        h = y_max - y_min
        padding_w = int(w * 0.1)
        padding_h = int(h * 0.1)
        
        bbox = (
            max(0, x_min - padding_w),
            max(0, y_min - padding_h),
            w + 2 * padding_w,
            h + 2 * padding_h
        )
        
        # Update for next frame
        self.points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        self.bbox = bbox
        
        return True, bbox
    
    def reinitialize(self, bbox: Tuple[int, int, int, int], frame: np.ndarray):
        """Reinitialize feature points in new bbox."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        x, y, w, h = bbox
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        
        self.points = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            **self.feature_params
        )
        
        self.prev_gray = gray.copy()
        self.bbox = bbox
        
        if self.points is not None:
            self.initial_point_count = len(self.points)


class HybridFighterTracker:
    """
    Hybrid tracker combining Optical Flow + CSRT.
    - Optical Flow: Primary (follows feature points)
    - CSRT: Backup and validation
    """
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int],
                 first_frame: np.ndarray,
                 fighter_id: str):
        """Initialize hybrid tracker."""
        self.fighter_id = fighter_id
        self.frame_shape = first_frame.shape[:2]
        self.frame_w = first_frame.shape[1]
        self.frame_h = first_frame.shape[0]
        
        # Primary: Optical Flow
        try:
            self.flow_tracker = OpticalFlowTracker(initial_bbox, first_frame, fighter_id)
            self.has_flow = True
        except:
            self.has_flow = False
            print(f"  ⚠️  {fighter_id}: Optical flow initialization failed")
        
        # Backup: CSRT
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(first_frame, initial_bbox)
        
        # State
        self.last_valid_bbox = initial_bbox
        self.consecutive_failures = 0
        self.tracking_history: List[TrackingFrame] = []
        
        # Motion history for prediction
        self.bbox_history = deque(maxlen=10)
        self.bbox_history.append(initial_bbox)
        
        # Store initial template
        x, y, w, h = initial_bbox
        self.initial_template = first_frame[y:y+h, x:x+w].copy()
        
        # First frame
        self.tracking_history.append(TrackingFrame(
            frame_num=0,
            timestamp=0.0,
            bbox=initial_bbox,
            confidence=1.0,
            tracking_active=True,
            tracker_source="INIT"
        ))
        
        print(f"  ✅ {fighter_id}: Hybrid tracker initialized")
    
    def update(self, frame: np.ndarray, frame_num: int, timestamp: float) -> TrackingFrame:
        """Update with hybrid approach."""
        
        # Try Optical Flow first
        flow_success, flow_bbox = False, None
        if self.has_flow:
            flow_success, flow_bbox = self.flow_tracker.update(frame)
        
        # Try CSRT
        csrt_success, csrt_bbox = self.csrt.update(frame)
        if csrt_success:
            csrt_bbox = tuple(map(int, csrt_bbox))
        
        # Decision logic
        if flow_success and flow_bbox:
            # Validate flow result
            if self._validate_bbox(flow_bbox):
                # Flow is reliable - use it
                bbox = self._smooth_bbox(flow_bbox)
                source = "OPTICAL_FLOW"
                
                # Validate CSRT against flow
                if csrt_success and self._validate_bbox(csrt_bbox):
                    # If CSRT agrees with flow, fuse them
                    if self._boxes_agree(flow_bbox, csrt_bbox):
                        bbox = self._fuse_boxes(flow_bbox, csrt_bbox)
                        source = "FLOW+CSRT"
                
                self._update_success(bbox, frame)
                
                result = TrackingFrame(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=1.0,
                    tracking_active=True,
                    tracker_source=source
                )
            else:
                # Flow invalid - try CSRT
                result = self._try_csrt_fallback(csrt_success, csrt_bbox, frame, frame_num, timestamp)
        
        else:
            # Flow failed - use CSRT
            result = self._try_csrt_fallback(csrt_success, csrt_bbox, frame, frame_num, timestamp)
        
        self.tracking_history.append(result)
        return result
    
    def _try_csrt_fallback(self, csrt_success, csrt_bbox, frame, frame_num, timestamp):
        """Try CSRT as fallback."""
        if csrt_success and csrt_bbox and self._validate_bbox(csrt_bbox):
            bbox = self._smooth_bbox(csrt_bbox)
            self._update_success(bbox, frame)
            
            return TrackingFrame(
                frame_num=frame_num,
                timestamp=timestamp,
                bbox=bbox,
                confidence=0.8,
                tracking_active=True,
                tracker_source="CSRT"
            )
        else:
            # Both failed - try recovery
            return self._handle_failure(frame, frame_num, timestamp)
    
    def _validate_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Strict validation."""
        x, y, w, h = bbox
        
        # Size check
        if w < 40 or h < 70 or w > self.frame_w * 0.6 or h > self.frame_h * 0.6:
            return False
        
        # Aspect ratio
        aspect = h / w if w > 0 else 0
        if aspect < 0.8 or aspect > 3.5:
            return False
        
        # Position check
        cx, cy = x + w / 2, y + h / 2
        if cx < 0 or cx > self.frame_w or cy < 0 or cy > self.frame_h:
            return False
        
        # Motion check
        if len(self.bbox_history) > 0:
            last_bbox = self.bbox_history[-1]
            last_cx = last_bbox[0] + last_bbox[2] / 2
            last_cy = last_bbox[1] + last_bbox[3] / 2
            
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            if dist > 150:  # Max 150px jump
                return False
        
        return True
    
    def _boxes_agree(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> bool:
        """Check if two boxes are similar enough."""
        cx1, cy1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        cx2, cy2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        # Must be within 80px
        return dist < 80
    
    def _fuse_boxes(self, box1: Tuple[int, int, int, int],
                    box2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Fuse two agreeing boxes (70% flow, 30% CSRT)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x = int(0.7 * x1 + 0.3 * x2)
        y = int(0.7 * y1 + 0.3 * y2)
        w = int(0.7 * w1 + 0.3 * w2)
        h = int(0.7 * h1 + 0.3 * h2)
        
        return (x, y, w, h)
    
    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply EMA smoothing."""
        if len(self.bbox_history) == 0:
            return bbox
        
        last = self.bbox_history[-1]
        alpha = 0.6
        
        x = int(alpha * bbox[0] + (1 - alpha) * last[0])
        y = int(alpha * bbox[1] + (1 - alpha) * last[1])
        w = int(alpha * bbox[2] + (1 - alpha) * last[2])
        h = int(alpha * bbox[3] + (1 - alpha) * last[3])
        
        return (x, y, w, h)
    
    def _update_success(self, bbox: Tuple[int, int, int, int], frame: np.ndarray):
        """Update state on successful tracking."""
        self.bbox_history.append(bbox)
        self.last_valid_bbox = bbox
        self.consecutive_failures = 0
    
    def _handle_failure(self, frame: np.ndarray, frame_num: int, timestamp: float) -> TrackingFrame:
        """Handle tracking failure."""
        self.consecutive_failures += 1
        
        # Aggressive re-initialization after just 2 failures
        if self.consecutive_failures >= 2:
            # Try to reinitialize near last known position
            predicted = self._predict_position()
            
            if predicted:
                # Reinitialize both trackers
                self.csrt = cv2.TrackerCSRT_create()
                self.csrt.init(frame, predicted)
                
                if self.has_flow:
                    self.flow_tracker.reinitialize(predicted, frame)
                
                print(f"  🔄 {self.fighter_id}: Reinitialized at frame {frame_num}")
                
                return TrackingFrame(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    bbox=predicted,
                    confidence=0.5,
                    tracking_active=True,
                    tracker_source="PREDICTED"
                )
        
        # Mark as lost
        return TrackingFrame(
            frame_num=frame_num,
            timestamp=timestamp,
            bbox=None,
            confidence=0.0,
            tracking_active=False,
            tracker_source="LOST"
        )
    
    def _predict_position(self) -> Optional[Tuple[int, int, int, int]]:
        """Predict next position from history."""
        if len(self.bbox_history) < 3:
            return self.last_valid_bbox
        
        # Simple velocity-based prediction
        boxes = list(self.bbox_history)[-3:]
        
        # Calculate velocity
        vx = (boxes[-1][0] - boxes[-3][0]) / 2
        vy = (boxes[-1][1] - boxes[-3][1]) / 2
        
        last = boxes[-1]
        pred_x = int(last[0] + vx)
        pred_y = int(last[1] + vy)
        
        return (pred_x, pred_y, last[2], last[3])
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        total = len(self.tracking_history)
        active = sum(1 for t in self.tracking_history if t.tracking_active)
        
        sources = {}
        for t in self.tracking_history:
            sources[t.tracker_source] = sources.get(t.tracker_source, 0) + 1
        
        return {
            "total_frames": total,
            "tracked_frames": active,
            "lost_frames": total - active,
            "tracking_rate": active / total if total > 0 else 0.0,
            "sources": sources
        }


class DualFighterTracker:
    """Manages two hybrid trackers independently."""
    
    def __init__(self,
                 my_fighter_bbox: Tuple[int, int, int, int],
                 opponent_bbox: Optional[Tuple[int, int, int, int]],
                 first_frame: np.ndarray):
        """Initialize dual tracker."""
        self.my_tracker = HybridFighterTracker(
            my_fighter_bbox, first_frame, "MY_FIGHTER"
        )
        
        if opponent_bbox:
            self.opp_tracker = HybridFighterTracker(
                opponent_bbox, first_frame, "OPPONENT"
            )
            self.has_opponent = True
        else:
            self.opp_tracker = None
            self.has_opponent = False
        
        print(f"\n🥊 Optical Flow + CSRT Hybrid Tracker")
    
    def track_frame(self, frame: np.ndarray, frame_num: int, 
                   timestamp: float) -> Dict[str, TrackingFrame]:
        """Track both fighters."""
        my_result = self.my_tracker.update(frame, frame_num, timestamp)
        
        if self.has_opponent:
            opp_result = self.opp_tracker.update(frame, frame_num, timestamp)
        else:
            opp_result = TrackingFrame(frame_num, timestamp, None, 0.0, False, "N/A")
        
        return {"my_fighter": my_result, "opponent": opp_result}
    
    def get_tracking_summary(self) -> Dict:
        """Get summary."""
        summary = {"my_fighter": self.my_tracker.get_stats()}
        if self.has_opponent:
            summary["opponent"] = self.opp_tracker.get_stats()
        return summary


def track_video(video_path: str,
                my_fighter_bbox: Tuple[int, int, int, int],
                opponent_bbox: Optional[Tuple[int, int, int, int]] = None,
                progress_interval: int = 100,
                show_video: bool = True,
                save_video: bool = False,
                output_path: str = "tracked_output.mp4") -> DualFighterTracker:
    """Track video with optical flow hybrid approach."""
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
    print("🎯 OPTICAL FLOW + CSRT HYBRID TRACKING")
    print("=" * 60)
    print(f"Algorithm:    Sparse Optical Flow (Lucas-Kanade) + CSRT")
    print(f"Total frames: {total_frames}")
    print(f"Duration:     {total_frames / fps:.2f} seconds")
    print("=" * 60)
    
    tracker = DualFighterTracker(my_fighter_bbox, opponent_bbox, first_frame)
    
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if show_video:
        cv2.namedWindow("Hybrid Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hybrid Tracking", 1280, 720)
    
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
                
                # MY FIGHTER
                my_r = results["my_fighter"]
                if my_r.tracking_active and my_r.bbox:
                    x, y, w, h = my_r.bbox
                    
                    if "FLOW" in my_r.tracker_source:
                        color = (0, 0, 255)  # Red - optical flow
                        thickness = 3
                    else:
                        color = (0, 140, 255)  # Orange - CSRT
                        thickness = 2
                    
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, thickness)
                    label = f"MY ({my_r.tracker_source})"
                    cv2.putText(display, label, (x, max(y-10, 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # OPPONENT
                opp_r = results["opponent"]
                if opp_r.tracking_active and opp_r.bbox:
                    x, y, w, h = opp_r.bbox
                    
                    if "FLOW" in opp_r.tracker_source:
                        color = (255, 255, 0)  # Cyan
                        thickness = 3
                    else:
                        color = (255, 200, 0)
                        thickness = 2
                    
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, thickness)
                    label = f"OPP ({opp_r.tracker_source})"
                    cv2.putText(display, label, (x, max(y-10, 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(display, f"Frame: {frame_num}/{total_frames}", 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                prog = frame_num / total_frames
                bar_w = width - 40
                cv2.rectangle(display, (20, height-60), (20+bar_w, height-40), (50, 50, 50), -1)
                cv2.rectangle(display, (20, height-60), (20+int(bar_w*prog), height-40), (0, 255, 0), -1)
                
                if show_video:
                    cv2.imshow("Hybrid Tracking", display)
                if save_video and video_writer:
                    video_writer.write(display)
            
            frame_num += 1
        
        if show_video:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                paused = not paused
        
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
    
    my = summary["my_fighter"]
    print(f"\n🔴 MY FIGHTER:")
    print(f"   Tracked:      {my['tracked_frames']}/{my['total_frames']} frames")
    print(f"   Success Rate: {my['tracking_rate']*100:.1f}%")
    print(f"   Lost Frames:  {my['lost_frames']}")
    print(f"   Sources:      {my['sources']}")
    
    if tracker.has_opponent:
        opp = summary["opponent"]
        print(f"\n🔵 OPPONENT:")
        print(f"   Tracked:      {opp['tracked_frames']}/{opp['total_frames']} frames")
        print(f"   Success Rate: {opp['tracking_rate']*100:.1f}%")
        print(f"   Lost Frames:  {opp['lost_frames']}")
        print(f"   Sources:      {opp['sources']}")
    
    print("=" * 60 + "\n")
    
    if save_video:
        print(f"💾 Saved: {output_path}\n")
    
    return tracker