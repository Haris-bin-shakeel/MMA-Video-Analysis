"""
PRODUCTION TRACKER - FIXED FOR ACCURATE TRACKING
Ensures bounding boxes STAY ON FIGHTERS throughout video

KEY FIXES:
1. Enhanced motion validation - detects when box drifts off fighter
2. Stricter confidence scoring - only high when box is ON fighter
3. Better recovery mechanisms - quickly relocates lost fighters
4. Frame-by-frame quality checks - validates tracking every frame
5. Appearance consistency - verifies we're still tracking same fighter
6. 🔧 NEW: Equal strictness for both MY_FIGHTER and OPPONENT
7. 🔧 NEW: Cross-validation prevents box swapping between fighters
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
    visual_quality: float  # NEW: 0-1 score of how well box fits fighter


class EnhancedOpticalFlowTracker:
    """Enhanced Optical Flow with drift detection."""
    
    lk_params = dict(
        winSize=(35, 35),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    feature_params = dict(
        maxCorners=250,
        qualityLevel=0.02,
        minDistance=7,
        blockSize=7
    )
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int],
                 first_frame: np.ndarray,
                 fighter_id: str):
        self.fighter_id = fighter_id
        self.bbox = initial_bbox
        self.frame_count = 0
        
        x, y, w, h = initial_bbox
        self.initial_template = first_frame[y:y+h, x:x+w].copy()
        self.initial_size = (w, h)
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Store initial appearance for drift detection
        self.initial_hist = self._compute_color_hist(first_frame, initial_bbox)
        
        self._initialize_features(initial_bbox, self.prev_gray)
        
        self.feature_quality_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=10)
        self.consecutive_low_quality = 0
        self.size_consistency_score = 1.0
    
    def _compute_color_hist(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Compute color histogram for appearance validation."""
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 10 or h <= 10:
            return None
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def _initialize_features(self, bbox: Tuple[int, int, int, int], gray: np.ndarray):
        x, y, w, h = bbox
        
        # Focus on center (torso/head)
        center_margin = int(min(w, h) * 0.15)
        mask_center = np.zeros_like(gray)
        mask_center[y+center_margin:y+h-center_margin, 
                    x+center_margin:x+w-center_margin] = 255
        
        mask_full = np.zeros_like(gray)
        mask_full[y:y+h, x:x+w] = 255
        
        center_points = cv2.goodFeaturesToTrack(
            gray, mask=mask_center, maxCorners=150,
            qualityLevel=self.feature_params['qualityLevel'],
            minDistance=self.feature_params['minDistance'],
            blockSize=self.feature_params['blockSize']
        )
        
        edge_points = cv2.goodFeaturesToTrack(
            gray, mask=mask_full, maxCorners=100,
            qualityLevel=self.feature_params['qualityLevel'] * 1.5,
            minDistance=self.feature_params['minDistance'],
            blockSize=self.feature_params['blockSize']
        )
        
        if center_points is not None and edge_points is not None:
            self.points = np.vstack([center_points, edge_points])
        elif center_points is not None:
            self.points = center_points
        elif edge_points is not None:
            self.points = edge_points
        else:
            self.points = None
        
        if self.points is None or len(self.points) < 20:
            raise ValueError(f"{self.fighter_id}: Insufficient features")
        
        self.initial_point_count = len(self.points)
        print(f"  ✅ {self.fighter_id}: {self.initial_point_count} features initialized")
    
    def _check_drift(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Check if tracking has drifted off the fighter.
        Returns: (is_drifted, appearance_score)
        """
        # Size consistency check
        w, h = bbox[2], bbox[3]
        init_w, init_h = self.initial_size
        
        size_ratio_w = w / init_w if init_w > 0 else 1.0
        size_ratio_h = h / init_h if init_h > 0 else 1.0
        
        # If size changed >60%, likely drifted
        if size_ratio_w < 0.4 or size_ratio_w > 1.6 or size_ratio_h < 0.4 or size_ratio_h > 1.6:
            return True, 0.2
        
        # Appearance check
        current_hist = self._compute_color_hist(frame, bbox)
        if current_hist is None or self.initial_hist is None:
            return False, 0.5
        
        appearance_score = cv2.compareHist(self.initial_hist, current_hist, cv2.HISTCMP_CORREL)
        appearance_score = max(0.0, min(1.0, appearance_score))
        
        # 🔧 FIX #3: Stricter drift threshold (0.40 instead of 0.30)
        # Catches drift SOONER before box wanders too far
        if appearance_score < 0.40:  # Was 0.30
            return True, appearance_score
        
        return False, appearance_score
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]], float, float]:
        """
        Update tracker.
        Returns: (success, bbox, confidence, visual_quality)
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.points is None or len(self.points) < 15:
            return False, None, 0.0, 0.0
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **self.lk_params
        )
        
        if new_points is None or status is None:
            return False, None, 0.0, 0.0
        
        # Filter good points
        good_mask = status.flatten() == 1
        if np.sum(good_mask) == 0:
            return False, None, 0.0, 0.0
        
        good_errors = error[good_mask]
        if len(good_errors) > 8:
            error_threshold = np.percentile(good_errors, 70)
            error_mask = error.flatten() <= error_threshold
            final_mask = good_mask & error_mask
        else:
            final_mask = good_mask
        
        good_new = new_points[final_mask]
        good_old = self.points[final_mask]
        
        retention_rate = len(good_new) / self.initial_point_count
        self.feature_quality_history.append(retention_rate)
        
        min_features = max(15, int(self.initial_point_count * 0.25))
        
        if len(good_new) < min_features:
            self.consecutive_low_quality += 1
            if self.consecutive_low_quality >= 4:
                return False, None, 0.0, 0.0
        else:
            self.consecutive_low_quality = 0
        
        # Motion consistency filtering
        points_array = good_new.reshape(-1, 2)
        old_points_array = good_old.reshape(-1, 2)
        
        motion = points_array - old_points_array
        median_motion = np.median(motion, axis=0)
        motion_diff = np.linalg.norm(motion - median_motion, axis=1)
        motion_threshold = np.percentile(motion_diff, 85)
        consistent_mask = motion_diff <= motion_threshold
        
        filtered_points = points_array[consistent_mask]
        
        if len(filtered_points) < 12:
            filtered_points = points_array
        
        # Calculate bbox
        x_min = int(np.min(filtered_points[:, 0]))
        y_min = int(np.min(filtered_points[:, 1]))
        x_max = int(np.max(filtered_points[:, 0]))
        y_max = int(np.max(filtered_points[:, 1]))
        
        w = x_max - x_min
        h = y_max - y_min
        
        base_padding = 0.20
        quality_adjustment = 0.10 * (1 - retention_rate)
        padding_factor = base_padding + quality_adjustment
        
        padding_w = int(w * padding_factor)
        padding_h = int(h * padding_factor)
        
        bbox = (
            max(0, x_min - padding_w),
            max(0, y_min - padding_h),
            w + 2 * padding_w,
            h + 2 * padding_h
        )
        
        # 🔧 NEW: Check for drift
        is_drifted, appearance_score = self._check_drift(frame, bbox)
        
        if is_drifted:
            # Tracking drifted off fighter
            return False, None, 0.0, 0.0
        
        # Calculate confidence
        recent_quality = list(self.feature_quality_history)[-5:] if len(self.feature_quality_history) >= 5 else list(self.feature_quality_history)
        avg_quality = np.mean(recent_quality) if recent_quality else retention_rate
        
        base_confidence = min(1.0, avg_quality * 1.4)
        stability_bonus = 0.15 if self.consecutive_low_quality == 0 else 0.0
        motion_consistency = len(filtered_points) / len(points_array) if len(points_array) > 0 else 0.5
        motion_bonus = 0.10 * motion_consistency
        
        confidence = min(1.0, base_confidence + stability_bonus + motion_bonus)
        
        # 🔧 NEW: Visual quality score (combines appearance + confidence)
        visual_quality = (appearance_score * 0.6 + confidence * 0.4)
        
        self.confidence_history.append(confidence)
        
        # Update state
        self.points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        self.bbox = bbox
        
        return True, bbox, confidence, visual_quality
    
    def reinitialize(self, bbox: Tuple[int, int, int, int], frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            self.initial_size = (bbox[2], bbox[3])
            self.initial_hist = self._compute_color_hist(frame, bbox)
            self._initialize_features(bbox, gray)
            self.prev_gray = gray.copy()
            self.bbox = bbox
            self.feature_quality_history.clear()
            self.confidence_history.clear()
            self.consecutive_low_quality = 0
        except:
            self.points = None
            self.prev_gray = gray.copy()
            self.bbox = bbox


class ProductionHybridTracker:
    """Production tracker with strict quality validation."""
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int],
                 first_frame: np.ndarray,
                 fighter_id: str):
        self.fighter_id = fighter_id
        self.frame_shape = first_frame.shape[:2]
        self.frame_w = first_frame.shape[1]
        self.frame_h = first_frame.shape[0]
        
        # Initialize trackers
        try:
            self.flow_tracker = EnhancedOpticalFlowTracker(initial_bbox, first_frame, fighter_id)
            self.has_flow = True
        except Exception as e:
            self.has_flow = False
            print(f"  ⚠️  {fighter_id}: Optical flow initialization failed: {e}")
        
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(first_frame, initial_bbox)
        
        # State tracking
        self.last_valid_bbox = initial_bbox
        self.consecutive_failures = 0
        self.tracking_history: List[TrackingFrame] = []
        self.frame_count = 0
        
        # History
        self.bbox_history = deque(maxlen=25)
        self.velocity_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=15)
        self.visual_quality_history = deque(maxlen=15)
        self.bbox_history.append(initial_bbox)
        
        # Template for recovery
        x, y, w, h = initial_bbox
        self.initial_template = first_frame[y:y+h, x:x+w].copy()
        self.initial_hist = self._compute_histogram(first_frame, initial_bbox)
        
        # Validation
        self.frames_since_validation = 0
        self.validation_interval = 10  # More frequent validation
        
        # 🔧 FIX #1: STRICTER appearance drift threshold (0.45 instead of 0.35)
        # Now OPPONENT will be as strict as MY_FIGHTER
        self.appearance_drift_threshold = 0.45  # Was 0.35
        
        # Save initial frame
        self.tracking_history.append(TrackingFrame(
            frame_num=0,
            timestamp=0.0,
            bbox=initial_bbox,
            confidence=1.0,
            tracking_active=True,
            tracker_source="INIT",
            visual_quality=1.0
        ))
        
        print(f"  ✅ {fighter_id}: Production tracker initialized")
    
    def _compute_histogram(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, self.frame_w - x)
        h = min(h, self.frame_h - y)
        
        if w <= 10 or h <= 10:
            return None
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def _validate_appearance(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Validate appearance similarity."""
        current_hist = self._compute_histogram(frame, bbox)
        
        if current_hist is None or self.initial_hist is None:
            return 0.5
        
        similarity = cv2.compareHist(self.initial_hist, current_hist, cv2.HISTCMP_CORREL)
        return max(0.0, min(1.0, similarity))
    
    def _template_match_search(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                               search_radius: int = 100) -> Optional[Tuple[int, int, int, int]]:
        """Search for fighter using template matching."""
        x, y, w, h = bbox
        
        search_x1 = max(0, x - search_radius)
        search_y1 = max(0, y - search_radius)
        search_x2 = min(self.frame_w, x + w + search_radius)
        search_y2 = min(self.frame_h, y + h + search_radius)
        
        search_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            return None
        
        template_scale = 0.8
        template_resized = cv2.resize(self.initial_template, 
                                     (int(w * template_scale), int(h * template_scale)))
        
        try:
            result = cv2.matchTemplate(search_region, template_resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.45:  # Slightly higher threshold
                match_x = search_x1 + max_loc[0]
                match_y = search_y1 + max_loc[1]
                match_w = int(w * template_scale)
                match_h = int(h * template_scale)
                
                final_bbox = (match_x, match_y, int(match_w / template_scale), int(match_h / template_scale))
                
                if self._validate_bbox(final_bbox):
                    return final_bbox
        except:
            pass
        
        return None
    
    def update(self, frame: np.ndarray, frame_num: int, timestamp: float) -> TrackingFrame:
        """Update tracker with strict quality validation."""
        self.frame_count += 1
        self.frames_since_validation += 1
        
        # Try optical flow
        flow_success, flow_bbox, flow_conf, flow_visual = False, None, 0.0, 0.0
        if self.has_flow:
            flow_success, flow_bbox, flow_conf, flow_visual = self.flow_tracker.update(frame)
        
        # Try CSRT
        csrt_success, csrt_bbox = self.csrt.update(frame)
        if csrt_success:
            csrt_bbox = tuple(map(int, csrt_bbox))
        
        perform_validation = (self.frames_since_validation >= self.validation_interval)
        
        # Process optical flow result
        if flow_success and flow_bbox and self._validate_bbox(flow_bbox):
            appearance_score = 1.0
            
            if perform_validation:
                appearance_score = self._validate_appearance(frame, flow_bbox)
                self.frames_since_validation = 0
                
                # 🔧 STRICTER: If appearance drifted, reject immediately
                if appearance_score < self.appearance_drift_threshold:
                    print(f"  ⚠️  {self.fighter_id}: Appearance drift (score={appearance_score:.2f}) - REJECTING")
                    
                    # Try template search
                    corrected_bbox = self._template_match_search(frame, flow_bbox)
                    
                    if corrected_bbox:
                        flow_bbox = corrected_bbox
                        appearance_score = 0.65
                        flow_visual = 0.65
                        print(f"  ✅ {self.fighter_id}: Recovered with template matching")
                    else:
                        # Cannot recover - mark as lost
                        return self._handle_failure(frame, frame_num, timestamp)
            
            # Combined confidence with visual quality
            combined_conf = flow_conf * 0.60 + flow_visual * 0.40
            bbox = self._smooth_bbox(flow_bbox)
            source = "OPTICAL_FLOW"
            
            # Fusion with CSRT
            if csrt_success and self._validate_bbox(csrt_bbox):
                if self._boxes_agree(flow_bbox, csrt_bbox):
                    bbox = self._fuse_boxes(flow_bbox, csrt_bbox, 0.70, 0.30)
                    source = "FLOW+CSRT"
                    combined_conf = min(1.0, combined_conf * 1.10)
                    flow_visual = min(1.0, flow_visual * 1.10)
            
            self._update_success(bbox, frame)
            self.confidence_history.append(combined_conf)
            self.visual_quality_history.append(flow_visual)
            
            result = TrackingFrame(
                frame_num=frame_num,
                timestamp=timestamp,
                bbox=bbox,
                confidence=combined_conf,
                tracking_active=True,
                tracker_source=source,
                visual_quality=flow_visual
            )
            
            self.tracking_history.append(result)
            return result
        
        # Fallback to CSRT
        result = self._try_csrt_fallback(csrt_success, csrt_bbox, frame, frame_num, timestamp, perform_validation)
        self.tracking_history.append(result)
        return result
    
    def _try_csrt_fallback(self, csrt_success, csrt_bbox, frame, frame_num, timestamp, perform_validation):
        """CSRT fallback with strict validation."""
        if csrt_success and csrt_bbox and self._validate_bbox(csrt_bbox):
            appearance_score = self._validate_appearance(frame, csrt_bbox) if perform_validation else 0.75
            
            # 🔧 STRICTER: Only accept if appearance is good
            # Now uses self.appearance_drift_threshold (0.45) instead of 0.35
            if appearance_score >= self.appearance_drift_threshold:
                bbox = self._smooth_bbox(csrt_bbox)
                self._update_success(bbox, frame)
                
                base_conf = 0.75 * appearance_score
                conf = max(0.45, base_conf)
                
                if self.consecutive_failures == 0 and len(self.bbox_history) > 20:
                    conf = min(1.0, conf * 1.15)
                
                self.confidence_history.append(conf)
                
                # Visual quality for CSRT = appearance score
                visual_quality = appearance_score
                self.visual_quality_history.append(visual_quality)
                
                if len(self.confidence_history) >= 5:
                    recent_confs = list(self.confidence_history)[-5:]
                    conf = np.mean(recent_confs)
                
                return TrackingFrame(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=conf,
                    tracking_active=True,
                    tracker_source="CSRT",
                    visual_quality=visual_quality
                )
        
        return self._handle_failure(frame, frame_num, timestamp)
    
    def _validate_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Strict bbox validation."""
        x, y, w, h = bbox
        
        if w < 30 or h < 50:
            return False
        
        if w > self.frame_w * 0.70 or h > self.frame_h * 0.70:
            return False
        
        aspect = h / w if w > 0 else 0
        if aspect < 0.8 or aspect > 4.0:
            return False
        
        if not self._is_valid_position(bbox):
            return False
        
        # Motion validation
        if len(self.bbox_history) > 0:
            last_bbox = self.bbox_history[-1]
            last_cx = last_bbox[0] + last_bbox[2] / 2
            last_cy = last_bbox[1] + last_bbox[3] / 2
            
            cx = x + w / 2
            cy = y + h / 2
            
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            if dist > 150:  # Stricter jump detection
                return False
        
        return True
    
    def _is_valid_position(self, bbox: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        
        margin = 40
        if cx < -margin or cx > self.frame_w + margin:
            return False
        if cy < -margin or cy > self.frame_h + margin:
            return False
        
        return True
    
    def _boxes_agree(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int], threshold: int = 100) -> bool:
        cx1 = box1[0] + box1[2] / 2
        cy1 = box1[1] + box1[3] / 2
        cx2 = box2[0] + box2[2] / 2
        cy2 = box2[1] + box2[3] / 2
        
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        return dist < threshold
    
    def _fuse_boxes(self, box1: Tuple[int, int, int, int],
                    box2: Tuple[int, int, int, int],
                    w1: float = 0.7, w2: float = 0.3) -> Tuple[int, int, int, int]:
        x1, y1, w1_b, h1 = box1
        x2, y2, w2_b, h2 = box2
        
        x = int(w1 * x1 + w2 * x2)
        y = int(w1 * y1 + w2 * y2)
        w = int(w1 * w1_b + w2 * w2_b)
        h = int(w1 * h1 + w2 * h2)
        
        return (x, y, w, h)
    
    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if len(self.bbox_history) == 0:
            return bbox
        
        last = self.bbox_history[-1]
        
        if len(self.bbox_history) >= 2:
            prev = self.bbox_history[-2]
            recent_motion = np.sqrt((last[0] - prev[0])**2 + (last[1] - prev[1])**2)
            alpha = 0.60 if recent_motion > 50 else 0.75
        else:
            alpha = 0.70
        
        x = int(alpha * bbox[0] + (1 - alpha) * last[0])
        y = int(alpha * bbox[1] + (1 - alpha) * last[1])
        w = int(alpha * bbox[2] + (1 - alpha) * last[2])
        h = int(alpha * bbox[3] + (1 - alpha) * last[3])
        
        return (x, y, w, h)
    
    def _update_success(self, bbox: Tuple[int, int, int, int], frame: np.ndarray):
        self.bbox_history.append(bbox)
        self.last_valid_bbox = bbox
        self.consecutive_failures = 0
        
        if len(self.bbox_history) >= 2:
            prev = self.bbox_history[-2]
            curr = bbox
            vx = curr[0] - prev[0]
            vy = curr[1] - prev[1]
            self.velocity_history.append((vx, vy))
    
    def _handle_failure(self, frame: np.ndarray, frame_num: int, timestamp: float) -> TrackingFrame:
        self.consecutive_failures += 1
        
        # Attempt recovery after 2 failures (faster than before)
        if self.consecutive_failures >= 2:
            predicted = self._predict_position()
            
            if predicted:
                recovered_bbox = self._template_match_search(frame, predicted, search_radius=120)
                
                if recovered_bbox and self._is_valid_position(recovered_bbox):
                    self.csrt = cv2.TrackerCSRT_create()
                    self.csrt.init(frame, recovered_bbox)
                    
                    if self.has_flow:
                        try:
                            self.flow_tracker.reinitialize(recovered_bbox, frame)
                        except:
                            pass
                    
                    print(f"  🔄 {self.fighter_id}: Recovered at frame {frame_num}")
                    
                    recovery_conf = 0.50
                    recovery_visual = 0.50
                    self.confidence_history.append(recovery_conf)
                    self.visual_quality_history.append(recovery_visual)
                    
                    return TrackingFrame(
                        frame_num=frame_num,
                        timestamp=timestamp,
                        bbox=recovered_bbox,
                        confidence=recovery_conf,
                        tracking_active=True,
                        tracker_source="RECOVERED",
                        visual_quality=recovery_visual
                    )
        
        self.confidence_history.append(0.0)
        self.visual_quality_history.append(0.0)
        return TrackingFrame(
            frame_num=frame_num,
            timestamp=timestamp,
            bbox=None,
            confidence=0.0,
            tracking_active=False,
            tracker_source="LOST",
            visual_quality=0.0
        )
    
    def _predict_position(self) -> Optional[Tuple[int, int, int, int]]:
        if len(self.bbox_history) < 2:
            return self.last_valid_bbox
        
        if len(self.velocity_history) >= 3:
            recent_vels = list(self.velocity_history)[-5:]
            avg_vx = np.mean([v[0] for v in recent_vels])
            avg_vy = np.mean([v[1] for v in recent_vels])
            
            last = self.bbox_history[-1]
            pred_x = int(last[0] + avg_vx * 2.0)
            pred_y = int(last[1] + avg_vy * 2.0)
            
            return (pred_x, pred_y, last[2], last[3])
        
        return self.last_valid_bbox
    
    def get_stats(self) -> Dict:
        total = len(self.tracking_history)
        active = sum(1 for t in self.tracking_history if t.tracking_active)
        
        sources = {}
        for t in self.tracking_history:
            sources[t.tracker_source] = sources.get(t.tracker_source, 0) + 1
        
        avg_conf = np.mean([t.confidence for t in self.tracking_history if t.confidence > 0])
        avg_visual = np.mean([t.visual_quality for t in self.tracking_history if t.visual_quality > 0])
        
        return {
            "total_frames": total,
            "tracked_frames": active,
            "lost_frames": total - active,
            "tracking_rate": active / total if total > 0 else 0.0,
            "average_confidence": round(avg_conf, 3),
            "average_visual_quality": round(avg_visual, 3),
            "sources": sources
        }


class DualFighterTracker:
    """Tracks two fighters with strict quality validation and cross-validation."""
    
    def __init__(self, my_fighter_bbox: Tuple[int, int, int, int],
                 opponent_bbox: Optional[Tuple[int, int, int, int]],
                 first_frame: np.ndarray):
        print("\n" + "=" * 60)
        print("🥊 PRODUCTION DUAL FIGHTER TRACKER (ACCURATE)")
        print("=" * 60)
        
        self.my_tracker = ProductionHybridTracker(my_fighter_bbox, first_frame, "MY_FIGHTER")
        
        self.has_opponent = opponent_bbox is not None
        if self.has_opponent:
            self.opp_tracker = ProductionHybridTracker(opponent_bbox, first_frame, "OPPONENT")
        else:
            self.opp_tracker = None
        
        print("=" * 60 + "\n")
    
    # 🔧 FIX #2: NEW METHOD - Cross-validation to prevent box swapping
    def _check_box_overlap(self, bbox1: Optional[Tuple[int, int, int, int]], 
                          bbox2: Optional[Tuple[int, int, int, int]]) -> float:
        """
        Calculate overlap between two bounding boxes.
        Returns: overlap ratio (0.0 = no overlap, 1.0 = complete overlap)
        """
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        # IoU (Intersection over Union)
        overlap_ratio = intersection_area / union_area
        return overlap_ratio
    
    def track_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Dict:
        results = {}
        
        # Track both fighters
        my_result = self.my_tracker.update(frame, frame_num, timestamp)
        results["my_fighter"] = my_result
        
        if self.has_opponent:
            opp_result = self.opp_tracker.update(frame, frame_num, timestamp)
            
            # 🔧 FIX #2: Cross-validation - check if boxes are swapped/overlapping
            if my_result.tracking_active and opp_result.tracking_active:
                overlap = self._check_box_overlap(my_result.bbox, opp_result.bbox)
                
                # If overlap >50%, one tracker likely drifted to the other fighter
                if overlap > 0.50:
                    print(f"  ⚠️  Frame {frame_num}: Box overlap detected ({overlap:.2f}) - potential swap!")
                    
                    # Mark the one with LOWER visual quality as lost
                    if my_result.visual_quality < opp_result.visual_quality:
                        # MY_FIGHTER drifted to opponent
                        my_result = TrackingFrame(
                            frame_num=frame_num,
                            timestamp=timestamp,
                            bbox=None,
                            confidence=0.0,
                            tracking_active=False,
                            tracker_source="LOST_OVERLAP",
                            visual_quality=0.0
                        )
                        print(f"  🔴 MY_FIGHTER marked as lost (overlapping with opponent)")
                    else:
                        # OPPONENT drifted to my fighter
                        opp_result = TrackingFrame(
                            frame_num=frame_num,
                            timestamp=timestamp,
                            bbox=None,
                            confidence=0.0,
                            tracking_active=False,
                            tracker_source="LOST_OVERLAP",
                            visual_quality=0.0
                        )
                        print(f"  🔵 OPPONENT marked as lost (overlapping with MY_FIGHTER)")
                    
                    # Update results with corrected tracking
                    results["my_fighter"] = my_result
            
            results["opponent"] = opp_result
        
        return results
    
    def get_tracking_summary(self) -> Dict:
        summary = {
            "my_fighter": self.my_tracker.get_stats()
        }
        
        if self.has_opponent:
            summary["opponent"] = self.opp_tracker.get_stats()
        
        return summary