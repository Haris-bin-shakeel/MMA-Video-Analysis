

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from enum import Enum


class FighterState(Enum):
    """Fighter tracking state."""
    VISIBLE = "visible"
    OCCLUDED = "occluded"  
    TEMP_LOST = "temp_lost"
    LOST_CONFIRMED = "lost_confirmed"


class TrackingSource(Enum):
    """Source of tracking data."""
    CSRT = "CSRT"
    FLOW = "FLOW"
    HIST = "HIST"
    PRED = "PRED"
    FROZEN = "FROZEN"


class FighterTracker:
   
    
 
    MAX_DISPLACEMENT_RATIO = 0.25  
    MIN_BOX_SIZE = 30 
    MAX_SIZE_CHANGE_RATIO = 2.5  
    LOST_FRAME_THRESHOLD = 60 
    REAPPEAR_SEARCH_RADIUS_RATIO = 0.35 
    OVERLAP_IOU_THRESHOLD = 0.4  
    CLINCH_IOU_THRESHOLD = 0.7 
    CLINCH_EXIT_IOU = 0.5  
    HISTOGRAM_MATCH_THRESHOLD = 0.6 
    TRACKING_QUALITY_THRESHOLD = 3.0  
    VALIDITY_THRESHOLD = 0.4  
    VISIBILITY_THRESHOLD = 0.5  
    PRESENCE_THRESHOLD = 0.6  
    
    
    GROUND_MODE_Y_THRESHOLD = 0.7  
    GROUND_MODE_DURATION = 90 
    SCENE_CHANGE_THRESHOLD = 0.6  
    SINGLE_FIGHTER_MAX_SIZE_RATIO = 0.9 
    CAGE_EDGE_MARGIN = 50  
    TAKEDOWN_VELOCITY_THRESHOLD = 15 
    REENTRY_SEARCH_INTERVAL = 30  
    ADAPTIVE_DISPLACEMENT_FACTOR = 1.5  
    PARTIAL_FRAME_MARGIN = 100  
    HISTOGRAM_BLEND_RATE = 0.1 
    REFEREE_SIZE_RATIO = 0.3  
    REFEREE_MOTION_FACTOR = 2.0  
    
    def __init__(self, frame_width: int, frame_height: int, fps: float):
        """Initialize tracker with video properties."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        self.max_displacement = frame_diagonal * self.MAX_DISPLACEMENT_RATIO
        self.reappear_search_radius = frame_diagonal * self.REAPPEAR_SEARCH_RADIUS_RATIO
        
       
        self.my_tracker = None
        self.opponent_tracker = None
        
       
        self.my_fighter_bbox = None
        self.opponent_bbox = None
        
       
        self.my_fighter_state = FighterState.LOST_CONFIRMED
        self.opponent_state = FighterState.LOST_CONFIRMED
        
       
        self.my_fighter_source = None
        self.opponent_source = None
        
       
        self.my_fighter_confidence = 0.0
        self.opponent_confidence = 0.0
        
       
        self.my_fighter_validity = 0.0
        self.opponent_validity = 0.0
        
        
        self.my_fighter_history = []  
        self.opponent_history = []
        
        
        self.my_fighter_lost_count = 0
        self.opponent_lost_count = 0
        
        
        self.my_fighter_lost_since = None
        self.opponent_lost_since = None
        
       
        self.my_fighter_last_valid = None
        self.opponent_last_valid = None
        
       
        self.my_fighter_histogram = None
        self.opponent_histogram = None
        
        
        self.prev_gray = None
        
        
        self.frame_count = 0
        
       
        self.in_clinch = False
        self.clinch_frozen_fighter = None 
        self.clinch_start_frame = None 
        
        self.my_fighter_histogram_backup = None
        self.opponent_histogram_backup = None
        self.my_fighter_pre_clinch_pos = None
        self.opponent_pre_clinch_pos = None
        
        self.separation_cooldown = 0
        
        self.ground_mode_active = False
        self.ground_mode_start_frame = None
        
        self.prev_frame_histogram = None
        self.scene_change_detected = False
        
        self.my_fighter_recent_velocity = 0.0
        self.opponent_recent_velocity = 0.0
        
        self.my_fighter_reentry_search_counter = 0
        self.opponent_reentry_search_counter = 0
        
        self.takedown_in_progress = False
        self.takedown_start_frame = None
        
        self.potential_referee_bbox = None
        self.referee_detection_cooldown = 0
        
        self.my_fighter_texture_features = None
        self.opponent_texture_features = None
        
        self.prev_iou = 0.0
        self.separation_cooldown_frames = 0
        
        self.history_duration_seconds = 2.0
        self.max_history_frames = int(self.history_duration_seconds * fps)
        
        print("🎯 Fighter Tracker Initialized")
        print(f"   Resolution: {frame_width}x{frame_height}")
        print(f"   Max displacement/frame: {self.max_displacement:.1f} px")
        print(f"   Lost threshold: {self.LOST_FRAME_THRESHOLD} frames")
    
    def initialize(self, frame: np.ndarray, my_fighter_bbox: Tuple[int, int, int, int],
                   opponent_bbox: Tuple[int, int, int, int]):
        """Initialize tracker with first frame and bounding boxes."""
        self.my_fighter_bbox = my_fighter_bbox
        self.opponent_bbox = opponent_bbox
        
        # Initialize CSRT trackers
        self.my_tracker = cv2.TrackerCSRT_create()
        self.opponent_tracker = cv2.TrackerCSRT_create()
        
        self.my_tracker.init(frame, my_fighter_bbox)
        self.opponent_tracker.init(frame, opponent_bbox)
        
        # Store initial states
        self.my_fighter_last_valid = my_fighter_bbox
        self.opponent_last_valid = opponent_bbox
        
        # Set states
        self.my_fighter_state = FighterState.VISIBLE
        self.opponent_state = FighterState.VISIBLE
        
        # Set sources and confidence
        self.my_fighter_source = TrackingSource.CSRT
        self.opponent_source = TrackingSource.CSRT
        self.my_fighter_confidence = 1.0
        self.opponent_confidence = 1.0
        
        # FIX #8: Initialize validity scores
        self.my_fighter_validity = 1.0
        self.opponent_validity = 1.0
        
        # Initialize history
        cx, cy = self._bbox_center(my_fighter_bbox)
        self.my_fighter_history.append((cx, cy, my_fighter_bbox[2], my_fighter_bbox[3], 0))
        
        cx, cy = self._bbox_center(opponent_bbox)
        self.opponent_history.append((cx, cy, opponent_bbox[2], opponent_bbox[3], 0))
        
        # Build appearance models
        self.my_fighter_histogram = self._compute_histogram(frame, my_fighter_bbox)
        self.opponent_histogram = self._compute_histogram(frame, opponent_bbox)
        
        # Issue 9: Initialize texture features for appearance matching
        self.my_fighter_texture_features = self._compute_texture_features(frame, my_fighter_bbox)
        self.opponent_texture_features = self._compute_texture_features(frame, opponent_bbox)
        
        # Initialize optical flow
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.frame_count = 0
        print("✅ Tracker initialized with CSRT trackers")
    
    def update(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        self._detect_scene_change(frame)
        
        ground_mode = self._detect_ground_mode(self.my_fighter_bbox, self.opponent_bbox)
        
        motion_frame = cv2.absdiff(gray, self.prev_gray) if self.prev_gray is not None else gray
        self._detect_potential_referee(frame, motion_frame)
        
        global_motion = self._estimate_global_motion(self.prev_gray, gray) if self.prev_gray is not None else (0, 0)
        
        if np.linalg.norm(global_motion) > self.max_displacement * 0.5:
            global_motion = (0, 0)
        
        if self.my_fighter_bbox and self.opponent_bbox:
            iou = self._calculate_iou(self.my_fighter_bbox, self.opponent_bbox)
            
            if iou > 0.7:
                if self.my_fighter_state == FighterState.VISIBLE:
                    self.my_fighter_state = FighterState.OCCLUDED
                if self.opponent_state == FighterState.VISIBLE:
                    self.opponent_state = FighterState.OCCLUDED
            
            if self.in_clinch:
                if iou < self.CLINCH_EXIT_IOU:
                    self.in_clinch = False
                    self.clinch_frozen_fighter = None
                    self.clinch_start_frame = None
                    
                    self._reset_frozen_fighter_confidence()
                    
                    force_reinit_my = True
                    force_reinit_opp = True
                    self.separation_cooldown_frames = 10  # Issue 4: Disable swap check for 10 frames
                    
                    if self.my_fighter_histogram_backup is not None:
                        self.my_fighter_histogram = self._blend_histogram_after_clinch(
                            self.my_fighter_histogram, self.my_fighter_histogram_backup)
                    if self.opponent_histogram_backup is not None:
                        self.opponent_histogram = self._blend_histogram_after_clinch(
                            self.opponent_histogram, self.opponent_histogram_backup)
            else:
                if iou > self.CLINCH_IOU_THRESHOLD:
                    self.in_clinch = True
                    self.clinch_start_frame = self.frame_count
                    
                    if self.my_fighter_histogram is not None:
                        self.my_fighter_histogram_backup = self.my_fighter_histogram.copy()
                    if self.opponent_histogram is not None:
                        self.opponent_histogram_backup = self.opponent_histogram.copy()
                    
                    if self.my_fighter_bbox is not None:
                        self.my_fighter_pre_clinch_pos = self._bbox_center(self.my_fighter_bbox)
                    if self.opponent_bbox is not None:
                        self.opponent_pre_clinch_pos = self._bbox_center(self.opponent_bbox)
                    
                    if self.my_fighter_confidence < self.opponent_confidence:
                        self.clinch_frozen_fighter = 'my'
                    else:
                        self.clinch_frozen_fighter = 'opponent'
            
            if self.prev_iou > 0.7 and iou < 0.3:
                force_reinit_my = True
                force_reinit_opp = True
                self.separation_cooldown_frames = 10
            
            self.prev_iou = iou
        
        if self.separation_cooldown_frames > 0:
            self.separation_cooldown_frames -= 1
        
        clinch_duration = 0
        if self.in_clinch and self.clinch_start_frame is not None:
            clinch_duration = self.frame_count - self.clinch_start_frame
        
        my_bbox, my_source, my_conf = self._track_fighter(
            frame, gray, global_motion,
            self.my_tracker,
            self.my_fighter_last_valid,
            self.my_fighter_history,
            self.my_fighter_histogram,
            self.my_fighter_lost_count,
            "my_fighter",
            frozen=(self.in_clinch and self.clinch_frozen_fighter == 'my'),
            clinch_duration=clinch_duration  
        )
        
        opponent_bbox, opp_source, opp_conf = self._track_fighter(
            frame, gray, global_motion,
            self.opponent_tracker,
            self.opponent_last_valid,
            self.opponent_history,
            self.opponent_histogram,
            self.opponent_lost_count,
            "opponent",
            frozen=(self.in_clinch and self.clinch_frozen_fighter == 'opponent'),
            clinch_duration=clinch_duration  
        )
        
        if (not self.in_clinch and 
            self.separation_cooldown_frames == 0 and  
            my_bbox is not None and opponent_bbox is not None):
            my_bbox, opponent_bbox = self._prevent_identity_swap(my_bbox, opponent_bbox)
        
        reinit_my = False
        
        if my_bbox is not None:
            
            prev_last = self.my_fighter_last_valid
            
            self.my_fighter_bbox = my_bbox
            self.my_fighter_last_valid = my_bbox
            self.my_fighter_source = my_source
            self.my_fighter_confidence = my_conf
            
           
            self.my_fighter_validity = self._calculate_validity_score(
                my_bbox, prev_last, my_conf, my_source
            )
            
            
            if self._is_corner_stuck(self.my_fighter_bbox, self.my_fighter_history):
                self.my_fighter_bbox = None
                self.my_fighter_state = FighterState.TEMP_LOST
                self.my_fighter_lost_count += 1
                if self.my_fighter_lost_since is None:
                    self.my_fighter_lost_since = self.frame_count - 1
               
            else:
               
                prev_state = self.my_fighter_state
                self.my_fighter_state = FighterState.VISIBLE
                self.my_fighter_lost_count = 0
                self.my_fighter_lost_since = None
                
               
                if my_source in {TrackingSource.CSRT, TrackingSource.FLOW, TrackingSource.HIST}:
                    cx, cy = self._bbox_center(my_bbox)
                    self.my_fighter_history.append((cx, cy, my_bbox[2], my_bbox[3], self.frame_count))
                
                if my_source != TrackingSource.CSRT or prev_state != FighterState.VISIBLE:
                    reinit_my = True
                
                new_hist = self._compute_histogram(frame, my_bbox)
                if new_hist is not None:
                    current_iou = self._calculate_iou(my_bbox, opponent_bbox) if opponent_bbox else 0.0
                    if not self.in_clinch and current_iou < 0.5:
                        self.my_fighter_histogram = 0.9 * self.my_fighter_histogram + 0.1 * new_hist
        else:
            self.my_fighter_bbox = None
            self.my_fighter_lost_count += 1
            
            if self.my_fighter_lost_count < self.LOST_FRAME_THRESHOLD:
                prev_state = self.my_fighter_state
                self.my_fighter_state = FighterState.TEMP_LOST
                if self.my_fighter_lost_since is None:
                    self.my_fighter_lost_since = self.frame_count - 1
            else:
                prev_state = self.my_fighter_state
                self.my_fighter_state = FighterState.LOST_CONFIRMED
                if prev_state != FighterState.LOST_CONFIRMED:
                    self.my_tracker = None
            
            self.my_fighter_validity = max(0.0, self.my_fighter_validity - 0.05)
        
        reinit_opp = False
        
        if opponent_bbox is not None:
            prev_last = self.opponent_last_valid
            
            self.opponent_bbox = opponent_bbox
            self.opponent_last_valid = opponent_bbox
            self.opponent_source = opp_source
            self.opponent_confidence = opp_conf
            
            self.opponent_validity = self._calculate_validity_score(
                opponent_bbox, prev_last, opp_conf, opp_source
            )
            
            if self._is_corner_stuck(self.opponent_bbox, self.opponent_history):
                self.opponent_bbox = None
                self.opponent_state = FighterState.TEMP_LOST
                self.opponent_lost_count += 1
                if self.opponent_lost_since is None:
                    self.opponent_lost_since = self.frame_count - 1
            else:
                prev_state = self.opponent_state
                self.opponent_state = FighterState.VISIBLE
                self.opponent_lost_count = 0
                self.opponent_lost_since = None
                
                if opp_source in {TrackingSource.CSRT, TrackingSource.FLOW, TrackingSource.HIST}:
                    cx, cy = self._bbox_center(opponent_bbox)
                    self.opponent_history.append((cx, cy, opponent_bbox[2], opponent_bbox[3], self.frame_count))
                
                if opp_source != TrackingSource.CSRT or prev_state != FighterState.VISIBLE:
                    reinit_opp = True
                
                new_hist = self._compute_histogram(frame, opponent_bbox)
                if new_hist is not None:
                    current_iou = self._calculate_iou(my_bbox, opponent_bbox) if my_bbox else 0.0
                    if not self.in_clinch and current_iou < 0.5:
                        self.opponent_histogram = 0.9 * self.opponent_histogram + 0.1 * new_hist
        else:
            self.opponent_bbox = None
            self.opponent_lost_count += 1
            
            if self.opponent_lost_count < self.LOST_FRAME_THRESHOLD:
                prev_state = self.opponent_state
                self.opponent_state = FighterState.TEMP_LOST
                if self.opponent_lost_since is None:
                    self.opponent_lost_since = self.frame_count - 1
            else:
                prev_state = self.opponent_state
                self.opponent_state = FighterState.LOST_CONFIRMED
                if prev_state != FighterState.LOST_CONFIRMED:
                    self.opponent_tracker = None
            
            self.opponent_validity = max(0.0, self.opponent_validity - 0.05)
        
        if reinit_my and my_bbox is not None:
            self.my_tracker = cv2.TrackerCSRT_create()
            self.my_tracker.init(frame, my_bbox)
        
        if reinit_opp and opponent_bbox is not None:
            self.opponent_tracker = cv2.TrackerCSRT_create()
            self.opponent_tracker.init(frame, opponent_bbox)
        
        if (self.my_fighter_state == FighterState.VISIBLE and 
            self.opponent_state == FighterState.VISIBLE and
            my_bbox is not None and opponent_bbox is not None and
            self.separation_cooldown == 0):  
            
            iou = self._calculate_iou(my_bbox, opponent_bbox)
            if iou > 0.7:  
                self.my_fighter_state = FighterState.OCCLUDED
                self.opponent_state = FighterState.OCCLUDED
                
                if self.my_fighter_histogram is not None:
                    self.my_fighter_histogram_backup = self.my_fighter_histogram.copy()
                if self.opponent_histogram is not None:
                    self.opponent_histogram_backup = self.opponent_histogram.copy()
        
        elif (self.my_fighter_state == FighterState.OCCLUDED and 
              self.opponent_state == FighterState.OCCLUDED and
              my_bbox is not None and opponent_bbox is not None):
            
            iou = self._calculate_iou(my_bbox, opponent_bbox)
            if iou < 0.3:  
                self.my_fighter_state = FighterState.VISIBLE
                self.opponent_state = FighterState.VISIBLE
                
                self.separation_cooldown = 30
        
        if self.separation_cooldown > 0:
            self.separation_cooldown -= 1
        
        self.my_fighter_history = [
            h for h in self.my_fighter_history 
            if self.frame_count - h[4] <= self.max_history_frames
        ]
        self.opponent_history = [
            h for h in self.opponent_history 
            if self.frame_count - h[4] <= self.max_history_frames
        ]
        
        # Update previous frame
        self.prev_gray = gray
        
        visible_my_bbox = (self.my_fighter_bbox if self.my_fighter_confidence >= self.VISIBILITY_THRESHOLD 
                          else None)
        visible_opponent_bbox = (self.opponent_bbox if self.opponent_confidence >= self.VISIBILITY_THRESHOLD 
                                else None)
        
        return {
            "my_fighter": visible_my_bbox,
            "opponent": visible_opponent_bbox
        }
    
    def _estimate_global_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float]:
       
        try:
            
            corners = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, 
                minDistance=10, blockSize=3
            )
            
            if corners is None or len(corners) < 10:
                return (0, 0)
            
           
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None
            )
            
          
            good_old = corners[status == 1]
            good_new = next_pts[status == 1]
            
            if len(good_old) < 10:
                return (0, 0)
            
          
            displacement = good_new - good_old
            
           
            median_dx = np.median(displacement[:, 0])
            median_dy = np.median(displacement[:, 1])
            
            return (median_dx, median_dy)
            
        except Exception:
            return (0, 0)
    
    def _track_fighter(self, frame: np.ndarray, gray: np.ndarray, global_motion: Tuple[float, float],
                      tracker, last_valid: Optional[Tuple],
                      history: List[Tuple], histogram: np.ndarray,
                      lost_count: int, name: str, frozen: bool = False,
                      clinch_duration: int = 0  # FIX #1: Accept clinch duration
                      ) -> Tuple[Optional[Tuple[int, int, int, int]], TrackingSource, float]:
      
        if frozen and last_valid is not None:
            conf = max(0.2, 0.5 - 0.01 * clinch_duration)
            return (last_valid, TrackingSource.FROZEN, conf)
        
        takedown_active = False
        if last_valid is not None and len(history) >= 5:
            takedown_active = self._detect_takedown(history, last_valid)
        
        fighter_state = (self.my_fighter_state if name == "my_fighter" 
                        else self.opponent_state)
        reentry_counter = (self.my_fighter_reentry_search_counter if name == "my_fighter"
                          else self.opponent_reentry_search_counter)
        texture_features = (self.my_fighter_texture_features if name == "my_fighter"
                           else self.opponent_texture_features)
        
        if fighter_state == FighterState.LOST_CONFIRMED and reentry_counter % self.REENTRY_SEARCH_INTERVAL == 0:
            reentry_bbox = self._search_full_frame_for_reentry(frame, histogram, texture_features, name)
            
            if reentry_bbox is not None:
                if self._validate_bbox(reentry_bbox, None, name):  # No last_valid for re-entry
                    if self._validate_partial_frame_exit(reentry_bbox):
                        if name == "my_fighter":
                            self.my_fighter_state = FighterState.VISIBLE
                            self.my_fighter_reentry_search_counter = 0
                        else:
                            self.opponent_state = FighterState.VISIBLE
                            self.opponent_reentry_search_counter = 0
                        
                        reentry_bbox = self._apply_vertical_anchor_bias(reentry_bbox)
                        return (reentry_bbox, TrackingSource.HIST, 0.8)  # Good confidence for re-entry
        
        if name == "my_fighter":
            self.my_fighter_reentry_search_counter += 1
        else:
            self.opponent_reentry_search_counter += 1
        
        backup_histogram = (self.my_fighter_histogram_backup if name == "my_fighter"
                          else self.opponent_histogram_backup)
        
        if fighter_state == FighterState.OCCLUDED:
            if last_valid is not None and backup_histogram is not None:
                expand_search = True  # Wider search during occlusion
                search_bbox = self._search_by_appearance(
                    frame, last_valid, backup_histogram, history, expand_search, name
                )
                
                if search_bbox is not None and self._validate_bbox(search_bbox, last_valid, name):
                    search_bbox = self._apply_vertical_anchor_bias(search_bbox)
                    return (search_bbox, TrackingSource.HIST, 0.7)  # Lower confidence during occlusion
        
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                bbox = tuple(map(int, bbox))
                
                try:
                    quality = tracker.getResponse()  
                    if quality < self.TRACKING_QUALITY_THRESHOLD:
                        success = False  # Reject degraded CSRT tracking
                    else:
                        opponent_bbox_for_check = (self.opponent_bbox if name == "my_fighter" 
                                                 else self.my_fighter_bbox)
                        if opponent_bbox_for_check is not None:
                            iou_with_opponent = self._calculate_iou(bbox, opponent_bbox_for_check)
                            if iou_with_opponent > 0.5 and quality < 5.0:
                                success = False  # CSRT degraded during overlap
                except:
                    pass
                
                if success:
                    # Validate bbox
                    if self._validate_bbox(bbox, last_valid, name):
                        bbox = self._apply_vertical_anchor_bias(bbox)
                        return (bbox, TrackingSource.CSRT, 1.0)
        
        if last_valid is not None and self.prev_gray is not None:
            flow_bbox = self._track_with_dense_flow(
                self.prev_gray, gray, last_valid, global_motion
            )
            
            if flow_bbox is not None and self._validate_bbox(flow_bbox, last_valid, name):
                flow_bbox = self._apply_vertical_anchor_bias(flow_bbox)
                return (flow_bbox, TrackingSource.FLOW, 0.7)
        
        if last_valid is not None and histogram is not None:
            expand_search = (lost_count >= self.LOST_FRAME_THRESHOLD)
            
            search_bbox = self._search_by_appearance(
                frame, last_valid, histogram, history, expand_search, name
            )
            
            if search_bbox is not None and self._validate_bbox(search_bbox, last_valid, name):
                search_bbox = self._apply_vertical_anchor_bias(search_bbox)
                return (search_bbox, TrackingSource.HIST, 0.6)
        
        if len(history) >= 3 and lost_count < self.LOST_FRAME_THRESHOLD:
            predicted_bbox = self._predict_from_motion(history)
            
            if predicted_bbox is not None and self._validate_bbox(predicted_bbox, last_valid, name):
                predicted_bbox = self._apply_vertical_anchor_bias(predicted_bbox)
                conf = max(0.3, 0.8 - (lost_count / self.LOST_FRAME_THRESHOLD) * 0.5)
                return (predicted_bbox, TrackingSource.PRED, conf)
        
        return (None, None, 0.0)
    
    def _track_with_dense_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                               bbox: Tuple[int, int, int, int], 
                               global_motion: Tuple[float, float]) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = bbox
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI
        roi_prev = prev_gray[y1:y2, x1:x2]
        
        expand = int(max(w, h) * 0.25)
        search_x1 = max(0, x1 - expand)
        search_y1 = max(0, y1 - expand)
        search_x2 = min(self.frame_width, x2 + expand)
        search_y2 = min(self.frame_height, y2 + expand)
        
        roi_curr = curr_gray[search_y1:search_y2, search_x1:search_x2]
        roi_prev_expanded = prev_gray[search_y1:search_y2, search_x1:search_x2]
        
        if roi_prev.size == 0 or roi_curr.size == 0:
            return None
        
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                roi_prev_expanded, roi_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate median displacement in original ROI region
            roi_x = x1 - search_x1
            roi_y = y1 - search_y1
            flow_roi = flow[roi_y:roi_y+h, roi_x:roi_x+w]
            
            if flow_roi.size == 0:
                return None
            
            # Use median to be robust to outliers
            median_dx = np.median(flow_roi[:, :, 0])
            median_dy = np.median(flow_roi[:, :, 1])
            
            # Subtract global camera motion
            fighter_dx = median_dx - global_motion[0]
            fighter_dy = median_dy - global_motion[1]
            
            # Apply displacement
            new_x = int(x + fighter_dx)
            new_y = int(y + fighter_dy)
            
            return (new_x, new_y, w, h)
            
        except Exception:
            return None
    
    def _search_by_appearance(self, frame: np.ndarray, last_bbox: Tuple[int, int, int, int],
                             histogram: np.ndarray, history: List[Tuple],
                             expand_search: bool = False, name: str = "") -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = last_bbox
        cx, cy = x + w // 2, y + h // 2
        
        # Define search region around last known position
        search_radius = int(self.reappear_search_radius)
        if expand_search:
            search_radius = int(search_radius * 1.5)  # Expand for LOST_CONFIRMED
        
        search_x1 = max(0, cx - search_radius)
        search_y1 = max(0, cy - search_radius)
        search_x2 = min(self.frame_width, cx + search_radius)
        search_y2 = min(self.frame_height, cy + search_radius)
        
        best_match = None
        best_score = 0
        
        step = max(6, w // 6)
        
        for test_x in range(search_x1, search_x2 - w, step):
            for test_y in range(search_y1, search_y2 - h, step):
                test_bbox = (test_x, test_y, w, h)
                
                if self._overlaps_opponent(test_bbox, name):
                    continue
                
                test_hist = self._compute_histogram(frame, test_bbox)
                
                if test_hist is None:
                    continue
                
                score = cv2.compareHist(histogram, test_hist, cv2.HISTCMP_CORREL)
                
                if score > best_score:
                    best_score = score
                    best_match = test_bbox
        
        if best_score > self.HISTOGRAM_MATCH_THRESHOLD:
            return best_match
        
        return None
    
    def _overlaps_opponent(self, bbox: Tuple[int, int, int, int], current_fighter: str = "") -> bool:
      
        if current_fighter == "my_fighter":
            opponent_bbox = self.opponent_bbox
            opponent_state = self.opponent_state
        elif current_fighter == "opponent":
            opponent_bbox = self.my_fighter_bbox
            opponent_state = self.my_fighter_state
        else:
            opponent_bbox = self.opponent_bbox
            opponent_state = self.opponent_state
        
        if opponent_bbox is None:
            return False
        
        if opponent_state == FighterState.LOST_CONFIRMED:
            return False
        
        iou = self._calculate_iou(bbox, opponent_bbox)
        return iou > self.OVERLAP_IOU_THRESHOLD
    
    def _predict_from_motion(self, history: List[Tuple]) -> Optional[Tuple[int, int, int, int]]:
        if len(history) < 3:
            return None
        
        p1 = history[-3]
        p2 = history[-2]
        p3 = history[-1]
        
        vx1 = p2[0] - p1[0]
        vy1 = p2[1] - p1[1]
        vx2 = p3[0] - p2[0]
        vy2 = p3[1] - p2[1]
        
        vx = (vx1 + vx2) / 2
        vy = (vy1 + vy2) / 2
        
        next_cx = int(p3[0] + vx)
        next_cy = int(p3[1] + vy)
        
        w = p3[2]
        h = p3[3]
        
        next_x = next_cx - w // 2
        next_y = next_cy - h // 2
        
        return (next_x, next_y, w, h)
    
    def _validate_bbox(self, bbox: Optional[Tuple[int, int, int, int]],
                      last_valid: Optional[Tuple[int, int, int, int]],
                      name: str) -> bool:
        if bbox is None:
            return False
        
        x, y, w, h = bbox
        
        if w < self.MIN_BOX_SIZE or h < self.MIN_BOX_SIZE:
            return False
        
        if w > self.frame_width * 0.8 or h > self.frame_height * 0.8:
            return False
        
        cx, cy = x + w // 2, y + h // 2
        if cx < 0 or cx >= self.frame_width or cy < 0 or cy >= self.frame_height:
            return False
        
        if last_valid is not None:
            center_curr = self._bbox_center(bbox)
            center_prev = self._bbox_center(last_valid)
            displacement = np.sqrt((center_curr[0] - center_prev[0])**2 + 
                                 (center_curr[1] - center_prev[1])**2)
            
            if displacement > self.max_displacement:
                return False
            
            cx, cy = self._bbox_center(bbox)
            pcx, pcy = self._bbox_center(last_valid)
            motion = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            
            fighter_lost_count = (self.my_fighter_lost_count if name == "my_fighter" 
                                else self.opponent_lost_count)
            
            if motion < 2.0 and self.frame_count - fighter_lost_count < 10:
                return False
            
            prev_area = last_valid[2] * last_valid[3]
            curr_area = w * h
            
            if prev_area > 0:
                size_ratio = max(curr_area, prev_area) / min(curr_area, prev_area)
                if size_ratio > self.MAX_SIZE_CHANGE_RATIO:
                    return False
        
        fighter_confidence = (self.my_fighter_confidence if name == "my_fighter" 
                            else self.opponent_confidence)
        if self._near_cage(bbox) and fighter_confidence < 0.6:
            return False
        
        return True
    
    def _calculate_validity_score(self, bbox: Tuple[int, int, int, int],
                                   last_valid: Optional[Tuple[int, int, int, int]],
                                   confidence: float,
                                   source: TrackingSource) -> float:
     
        score = confidence
        
        if last_valid is not None:
            center_curr = self._bbox_center(bbox)
            center_prev = self._bbox_center(last_valid)
            displacement = np.sqrt((center_curr[0] - center_prev[0])**2 + 
                                 (center_curr[1] - center_prev[1])**2)
            
            displacement_ratio = displacement / self.max_displacement
            score *= (1.0 - 0.3 * min(1.0, displacement_ratio))
            
            prev_area = last_valid[2] * last_valid[3]
            curr_area = bbox[2] * bbox[3]
            
            if prev_area > 0:
                size_ratio = max(curr_area, prev_area) / min(curr_area, prev_area)
                size_penalty = min(1.0, (size_ratio - 1.0) / self.MAX_SIZE_CHANGE_RATIO)
                score *= (1.0 - 0.2 * size_penalty)
        
        return max(0.0, min(1.0, score))
    
    def _prevent_identity_swap(self, my_bbox: Tuple[int, int, int, int],
                               opponent_bbox: Tuple[int, int, int, int]
                               ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
       
        iou = self._calculate_iou(my_bbox, opponent_bbox)
        
        if iou < self.OVERLAP_IOU_THRESHOLD:
            return my_bbox, opponent_bbox
        
        confidence_correct = 0.0
        confidence_swapped = 0.0
        
        if len(self.my_fighter_history) >= 2 and len(self.opponent_history) >= 2:
            my_expected_cx = self.my_fighter_history[-1][0]
            my_expected_cy = self.my_fighter_history[-1][1]
            
            opp_expected_cx = self.opponent_history[-1][0]
            opp_expected_cy = self.opponent_history[-1][1]
            
            my_cx, my_cy = self._bbox_center(my_bbox)
            opp_cx, opp_cy = self._bbox_center(opponent_bbox)
            
            dist_my_to_my = np.sqrt((my_cx - my_expected_cx)**2 + (my_cy - my_expected_cy)**2)
            dist_my_to_opp = np.sqrt((my_cx - opp_expected_cx)**2 + (my_cy - opp_expected_cy)**2)
            dist_opp_to_my = np.sqrt((opp_cx - my_expected_cx)**2 + (opp_cy - my_expected_cy)**2)
            dist_opp_to_opp = np.sqrt((opp_cx - opp_expected_cx)**2 + (opp_cy - opp_expected_cy)**2)
            
            max_dist = self.max_displacement
            spatial_correct = 1.0 - min(1.0, (dist_my_to_my + dist_opp_to_opp) / (2 * max_dist))
            spatial_swapped = 1.0 - min(1.0, (dist_my_to_opp + dist_opp_to_my) / (2 * max_dist))
            
            confidence_correct += spatial_correct * 0.4  # 40% weight
            confidence_swapped += spatial_swapped * 0.4
        
        if (self.my_fighter_histogram is not None and self.opponent_histogram is not None and
            hasattr(self, 'prev_gray')):  # Only if we have current frame context
            
            current_frame = getattr(self, '_current_frame', None)
            if current_frame is not None:
                hist_my_current = self._compute_histogram(current_frame, my_bbox)
                hist_opp_current = self._compute_histogram(current_frame, opponent_bbox)
                
                if hist_my_current is not None and hist_opp_current is not None:
                    similarity_my_my = cv2.compareHist(self.my_fighter_histogram, hist_my_current, cv2.HISTCMP_CORREL)
                    similarity_my_opp = cv2.compareHist(self.opponent_histogram, hist_my_current, cv2.HISTCMP_CORREL)
                    
                    similarity_opp_my = cv2.compareHist(self.my_fighter_histogram, hist_opp_current, cv2.HISTCMP_CORREL)
                    similarity_opp_opp = cv2.compareHist(self.opponent_histogram, hist_opp_current, cv2.HISTCMP_CORREL)
                    
                    appearance_correct = (similarity_my_my + similarity_opp_opp) / 2
                    appearance_swapped = (similarity_my_opp + similarity_opp_my) / 2
                    
                    confidence_correct += appearance_correct * 0.3  # 30% weight
                    confidence_swapped += appearance_swapped * 0.3
        
        if len(self.my_fighter_history) >= 3 and len(self.opponent_history) >= 3:
            # Calculate recent velocity vectors
            my_vel_x = self.my_fighter_history[-1][0] - self.my_fighter_history[-2][0]
            my_vel_y = self.my_fighter_history[-1][1] - self.my_fighter_history[-2][1]
            
            opp_vel_x = self.opponent_history[-1][0] - self.opponent_history[-2][0]
            opp_vel_y = self.opponent_history[-1][1] - self.opponent_history[-2][1]
            
            # Current displacement vectors
            my_cx, my_cy = self._bbox_center(my_bbox)
            opp_cx, opp_cy = self._bbox_center(opponent_bbox)
            
            my_prev_cx, my_prev_cy = self.my_fighter_history[-1][0], self.my_fighter_history[-1][1]
            opp_prev_cx, opp_prev_cy = self.opponent_history[-1][0], self.opponent_history[-1][1]
            
            my_disp_x = my_cx - my_prev_cx
            my_disp_y = my_cy - my_prev_cy
            opp_disp_x = opp_cx - opp_prev_cx
            opp_disp_y = opp_cy - opp_prev_cy
            
            # Check velocity continuity (dot product similarity)
            def velocity_similarity(vel_x, vel_y, disp_x, disp_y):
                vel_mag = np.sqrt(vel_x**2 + vel_y**2)
                disp_mag = np.sqrt(disp_x**2 + disp_y**2)
                if vel_mag < 1 or disp_mag < 1:
                    return 0.5  # Neutral if too slow
                
                # Cosine similarity
                dot = vel_x * disp_x + vel_y * disp_y
                return max(0, min(1, dot / (vel_mag * disp_mag)))
            
            motion_correct = (velocity_similarity(my_vel_x, my_vel_y, my_disp_x, my_disp_y) + 
                            velocity_similarity(opp_vel_x, opp_vel_y, opp_disp_x, opp_disp_y)) / 2
            motion_swapped = (velocity_similarity(my_vel_x, my_vel_y, opp_disp_x, opp_disp_y) + 
                            velocity_similarity(opp_vel_x, opp_vel_y, my_disp_x, my_disp_y)) / 2
            
            confidence_correct += motion_correct * 0.3  # 30% weight
            confidence_swapped += motion_swapped * 0.3
        
        confidence_threshold = 0.1  
        
        if confidence_swapped > confidence_correct + confidence_threshold:
            return opponent_bbox, my_bbox
        
        return my_bbox, opponent_bbox
    
    def _compute_histogram(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Compute color histogram for appearance model."""
        x, y, w, h = bbox
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate center of bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _near_cage(self, bbox: Tuple[int, int, int, int], margin: int = 40) -> bool:
        x, y, w, h = bbox
        return (
            x < margin or
            y < margin or
            x + w > self.frame_width - margin or
            y + h > self.frame_height - margin
        )
    
    def _is_corner_stuck(self, bbox, history, frames=12):
        if bbox is None or len(history) < frames:
            return False

        recent = history[-frames:]
        xs = [h[0] for h in recent]
        ys = [h[1] for h in recent]

        if max(xs) - min(xs) < 3 and max(ys) - min(ys) < 3:
            x, y, w, h = bbox
            if (
                x < 40 or y < 40 or
                x + w > self.frame_width - 40 or
                y + h > self.frame_height - 40
            ):
                return True

        return False
    
    
    def _detect_scene_change(self, frame: np.ndarray) -> bool:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        if self.prev_frame_histogram is None:
            self.prev_frame_histogram = hist
            return False
        
       
        similarity = cv2.compareHist(self.prev_frame_histogram, hist, cv2.HISTCMP_CORREL)
        self.prev_frame_histogram = hist
        
        
        scene_changed = similarity < self.SCENE_CHANGE_THRESHOLD
        
        if scene_changed:
           
            self.my_tracker = None
            self.opponent_tracker = None
            self.my_fighter_state = FighterState.LOST_CONFIRMED
            self.opponent_state = FighterState.LOST_CONFIRMED
            self.scene_change_detected = True
            print("🎬 Scene change detected - resetting trackers")
        
        return scene_changed
    
    def _detect_ground_mode(self, my_bbox: Optional[Tuple], opp_bbox: Optional[Tuple]) -> bool:
    
        if my_bbox is None or opp_bbox is None:
            self.ground_mode_active = False
            self.ground_mode_start_frame = None
            return False
        
        _, my_y, _, my_h = my_bbox
        _, opp_y, _, opp_h = opp_bbox
        
        my_center_y = my_y + my_h / 2
        opp_center_y = opp_y + opp_h / 2
        
        frame_ground_level = self.frame_height * self.GROUND_MODE_Y_THRESHOLD
        
        both_on_ground = (my_center_y > frame_ground_level and 
                         opp_center_y > frame_ground_level)
        
        if both_on_ground:
            if self.ground_mode_start_frame is None:
                self.ground_mode_start_frame = self.frame_count
            elif (self.frame_count - self.ground_mode_start_frame) >= self.GROUND_MODE_DURATION:
                if not self.ground_mode_active:
                    self.ground_mode_active = True
                    print("🥋 Ground mode activated - adjusting tracking parameters")
        else:
            self.ground_mode_active = False
            self.ground_mode_start_frame = None
        
        return self.ground_mode_active
    
    def _detect_takedown(self, history: List[Tuple], current_bbox: Tuple) -> bool:
       
        if len(history) < 5:
            return False
        
        recent_positions = history[-5:]
        velocities = []
        
        for i in range(1, len(recent_positions)):
            prev_y = recent_positions[i-1][1]
            curr_y = recent_positions[i][1]
            velocity = curr_y - prev_y  
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        
        takedown_velocity = avg_velocity > self.TAKEDOWN_VELOCITY_THRESHOLD
        
        opponent_bbox = (self.opponent_bbox if current_bbox == self.my_fighter_bbox 
                        else self.my_fighter_bbox)
        
        high_overlap = False
        if opponent_bbox is not None:
            iou = self._calculate_iou(current_bbox, opponent_bbox)
            high_overlap = iou > 0.5
        
        takedown_detected = takedown_velocity and high_overlap
        
        if takedown_detected and not self.takedown_in_progress:
            self.takedown_in_progress = True
            self.takedown_start_frame = self.frame_count
            print("🤼 Takedown detected - reducing motion continuity weight")
        elif self.takedown_in_progress and self.frame_count - self.takedown_start_frame > 60:  # 2 seconds
            self.takedown_in_progress = False
        
        return self.takedown_in_progress
    
    def _compute_texture_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
      
        x, y, w, h = bbox
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(gray_roi, (64, 128), interpolation=cv2.INTER_LINEAR)
        
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        return features.flatten() if features is not None else None
    
    def _validate_partial_frame_exit(self, bbox: Tuple[int, int, int, int]) -> bool:
       
        x, y, w, h = bbox
        
        bbox_area = w * h
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        visible_width = max(0, x2 - x1)
        visible_height = max(0, y2 - y1)
        visible_area = visible_width * visible_height
        
        visibility_ratio = visible_area / bbox_area if bbox_area > 0 else 0
        
        cx, cy = self._bbox_center(bbox)
        center_outside = (cx < -self.PARTIAL_FRAME_MARGIN or 
                         cx > self.frame_width + self.PARTIAL_FRAME_MARGIN or
                         cy < -self.PARTIAL_FRAME_MARGIN or 
                         cy > self.frame_height + self.PARTIAL_FRAME_MARGIN)
        
        if center_outside:
            return visibility_ratio > 0.3 
        else:
            return visibility_ratio > 0.5 
    
    def _get_adaptive_max_displacement(self, name: str) -> float:
       
        recent_velocity = (self.my_fighter_recent_velocity if name == "my_fighter" 
                          else self.opponent_recent_velocity)
        
        base_displacement = self.max_displacement
        
        if recent_velocity > base_displacement * 0.5:
            adaptive_displacement = base_displacement * self.ADAPTIVE_DISPLACEMENT_FACTOR
            return min(adaptive_displacement, self.frame_width * 0.5)  # Cap at 50% of frame width
        
        return base_displacement
    
    def _search_full_frame_for_reentry(self, frame: np.ndarray, histogram: np.ndarray, 
                                     texture_features: Optional[np.ndarray], name: str) -> Optional[Tuple[int, int, int, int]]:
      
        if histogram is None:
            return None
        
        best_match = None
        best_score = 0
        
        step_size = 32
        min_size = self.MIN_BOX_SIZE
        max_size = min(self.frame_width // 3, self.frame_height // 3)  # Max 1/3 of frame
        
        for y in range(0, self.frame_height - min_size, step_size):
            for x in range(0, self.frame_width - min_size, step_size):
                for size in range(min_size, max_size + 1, step_size):
                    if x + size > self.frame_width or y + size > self.frame_height:
                        continue
                    
                    test_bbox = (x, y, size, size)
                    
                    test_hist = self._compute_histogram(frame, test_bbox)
                    if test_hist is None:
                        continue
                    
                    color_similarity = cv2.compareHist(histogram, test_hist, cv2.HISTCMP_CORREL)
                    
                    texture_similarity = 0.5  # Neutral score
                    if texture_features is not None:
                        test_texture = self._compute_texture_features(frame, test_bbox)
                        if test_texture is not None and len(test_texture) == len(texture_features):
                            dot_product = np.dot(texture_features, test_texture)
                            norm_a = np.linalg.norm(texture_features)
                            norm_b = np.linalg.norm(test_texture)
                            if norm_a > 0 and norm_b > 0:
                                texture_similarity = dot_product / (norm_a * norm_b)
                    
                    combined_score = 0.7 * color_similarity + 0.3 * texture_similarity
                    
                    if combined_score > best_score and combined_score > 0.4:  # Minimum threshold
                        best_score = combined_score
                        best_match = test_bbox
        
        return best_match
    
    def _detect_potential_referee(self, frame: np.ndarray, motion_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
       
        if self.referee_detection_cooldown > 0:
            self.referee_detection_cooldown -= 1
            return None
        
        thresh = cv2.threshold(motion_frame, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fighter_sizes = []
        if self.my_fighter_bbox is not None:
            _, _, w, h = self.my_fighter_bbox
            fighter_sizes.append(w * h)
        if self.opponent_bbox is not None:
            _, _, w, h = self.opponent_bbox
            fighter_sizes.append(w * h)
        
        avg_fighter_size = np.mean(fighter_sizes) if fighter_sizes else (self.frame_width * self.frame_height * 0.1)
        
        potential_referee = None
        max_motion_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < avg_fighter_size * self.REFEREE_SIZE_RATIO:
                continue  # Too small to be referee
            
            x, y, w, h = cv2.boundingRect(contour)
            
            referee_bbox = (x, y, w, h)
            overlaps_fighter = False
            
            for fighter_bbox in [self.my_fighter_bbox, self.opponent_bbox]:
                if fighter_bbox is not None:
                    if self._calculate_iou(referee_bbox, fighter_bbox) > 0.3:
                        overlaps_fighter = True
                        break
            
            if overlaps_fighter:
                continue
            
            motion_score = area  
            
            
            center_x, center_y = x + w//2, y + h//2
            
            
            if motion_score > max_motion_score:
                max_motion_score = motion_score
                potential_referee = referee_bbox
        
        if potential_referee is not None:
            self.referee_detection_cooldown = 60 
            print("👨‍⚖️ Potential referee detected - will avoid tracking")
        
        return potential_referee
    
    def _validate_bbox(self, bbox: Tuple[int, int, int, int], last_valid: Optional[Tuple], 
                      name: str) -> bool:
       
        x, y, w, h = bbox
        
        if w < self.MIN_BOX_SIZE or h < self.MIN_BOX_SIZE:
            return False
        
        max_size_ratio = self.MAX_SIZE_CHANGE_RATIO
        other_fighter_visible = ((name == "my_fighter" and self.opponent_state == FighterState.VISIBLE) or
                                (name == "opponent" and self.my_fighter_state == FighterState.VISIBLE))
        
        if not other_fighter_visible:
            max_size_ratio = self.SINGLE_FIGHTER_MAX_SIZE_RATIO
        
        if last_valid is not None:
            _, _, last_w, last_h = last_valid
            size_change_w = w / last_w if last_w > 0 else float('inf')
            size_change_h = h / last_h if last_h > 0 else float('inf')
            
            if size_change_w > max_size_ratio or size_change_w < 1/max_size_ratio:
                return False
            if size_change_h > max_size_ratio or size_change_h < 1/max_size_ratio:
                return False
        
        if last_valid is not None:
            curr_center = self._bbox_center(bbox)
            last_center = self._bbox_center(last_valid)
            
            displacement = np.linalg.norm(np.array(curr_center) - np.array(last_center))
            max_allowed = self._get_adaptive_max_displacement(name)
            
            if self.ground_mode_active:
                max_allowed *= 1.5 
            
            if displacement > max_allowed:
                return False
            
            if name == "my_fighter":
                self.my_fighter_recent_velocity = displacement
            else:
                self.opponent_recent_velocity = displacement
        
        if not self._validate_partial_frame_exit(bbox):
            return False
        
        if self.potential_referee_bbox is not None:
            referee_iou = self._calculate_iou(bbox, self.potential_referee_bbox)
            if referee_iou > 0.3: 
                return False
        
        return True
    
    def _apply_cage_aware_histogram_masking(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
       
        x, y, w, h = bbox
        
        # Extract ROI
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        roi = frame[y1:y2, x1:x2].copy()
        
        # Create mask for pixels near cage edges
        mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        
        bbox_near_left = x < self.CAGE_EDGE_MARGIN
        bbox_near_right = (x + w) > (self.frame_width - self.CAGE_EDGE_MARGIN)
        bbox_near_top = y < self.CAGE_EDGE_MARGIN
        bbox_near_bottom = (y + h) > (self.frame_height - self.CAGE_EDGE_MARGIN)
        
        if bbox_near_left or bbox_near_right or bbox_near_top or bbox_near_bottom:
            edge_margin = 10  # Pixels from bbox edge to mask
            
            if bbox_near_left:
                mask[:, :edge_margin] = 0
            if bbox_near_right:
                mask[:, -edge_margin:] = 0
            if bbox_near_top:
                mask[:edge_margin, :] = 0
            if bbox_near_bottom:
                mask[-edge_margin:, :] = 0
        
        return roi, mask
    
    def _blend_histogram_after_clinch(self, current_hist: np.ndarray, backup_hist: np.ndarray) -> np.ndarray:
       
        if backup_hist is None:
            return current_hist
        
        blended = (1 - self.HISTOGRAM_BLEND_RATE) * current_hist + self.HISTOGRAM_BLEND_RATE * backup_hist
        
        cv2.normalize(blended, blended, 0, 1, cv2.NORM_MINMAX)
        
        return blended
    
    def _reset_frozen_fighter_confidence(self):
       
        if self.clinch_frozen_fighter == "my":
            self.my_fighter_confidence = 0.7
        elif self.clinch_frozen_fighter == "opponent":
            self.opponent_confidence = 0.7
    
    def _compute_histogram(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        
        roi, mask = self._apply_cage_aware_histogram_masking(frame, bbox)
        
        if roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def _apply_vertical_anchor_bias(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
       
        x, y, w, h = bbox
        
        # Calculate current center
        cx = x + w // 2
        cy = y + h // 2
        
        
        bias_factor = 0.15
        new_cy = int(cy - h * bias_factor)
        
        new_cy = max(y, new_cy)
        
        new_x = cx - w // 2
        new_y = new_cy - h // 2
        
        new_x = max(0, min(new_x, self.frame_width - w))
        new_y = max(0, min(new_y, self.frame_height - h))
        
        return (new_x, new_y, w, h)
    
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
        
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_status(self) -> Dict[str, any]:
    = bbox is not None and validity > VALIDITY_THRESHOLD
       
        return {
            "frame": self.frame_count,
            "my_fighter": {
                "visible": self.my_fighter_bbox is not None,
                "bbox": self.my_fighter_bbox,
                "state": self.my_fighter_state.value,
                "lost_frames": self.my_fighter_lost_count,
                "lost_since": self.my_fighter_lost_since,
                "confidence": self.my_fighter_confidence,
                "validity": self.my_fighter_validity,
                "reliable": self.my_fighter_bbox is not None and self.my_fighter_validity > self.VALIDITY_THRESHOLD,  # FIX #5
                "source": self.my_fighter_source.value if self.my_fighter_source else None
            },
            "opponent": {
                "visible": self.opponent_bbox is not None,
                "bbox": self.opponent_bbox,
                "state": self.opponent_state.value,
                "lost_frames": self.opponent_lost_count,
                "lost_since": self.opponent_lost_since,
                "confidence": self.opponent_confidence,
                "validity": self.opponent_validity,
                "reliable": self.opponent_bbox is not None and self.opponent_validity > self.VALIDITY_THRESHOLD,  # FIX #5
                "source": self.opponent_source.value if self.opponent_source else None
            },
            "in_clinch": self.in_clinch
        }
