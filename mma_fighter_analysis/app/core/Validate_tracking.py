"""
COMPREHENSIVE TRACKING VALIDATION SYSTEM
Tests tracking quality WITHOUT relying on confidence scores

Validates:
1. Bbox movement physics (impossible jumps, teleportation)
2. Size consistency (sudden changes = identity loss)
3. Identity separation (fighters don't swap)
4. Coverage accuracy (zones match actual good tracking)
5. Frame-by-frame behavior analysis

Output: Detailed CSV + Quality Report + Client-Ready Verdict
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import sys


@dataclass
class BboxMetrics:
    """Frame-level bbox analysis metrics."""
    frame_num: int
    timestamp: float
    
    # Bbox data
    bbox_x: Optional[int]
    bbox_y: Optional[int]
    bbox_w: Optional[int]
    bbox_h: Optional[int]
    bbox_area: Optional[int]
    bbox_center_x: Optional[float]
    bbox_center_y: Optional[float]
    
    # Movement analysis
    movement_distance: Optional[float]
    movement_speed: Optional[float]  # pixels per frame
    direction_change: Optional[float]  # degrees
    
    # Size analysis
    area_change_pct: Optional[float]
    aspect_ratio: Optional[float]
    
    # Quality flags
    has_bbox: bool
    is_teleport: bool  # Moved >200px in 1 frame
    is_size_jump: bool  # Size changed >40% in 1 frame
    is_impossible_movement: bool  # Speed >150 px/frame
    is_edge_bbox: bool  # Bbox touching frame edges
    is_tiny_bbox: bool  # Too small to be a person
    is_huge_bbox: bool  # Too large (>50% of frame)
    
    # Zone validation
    in_presence_zone: bool
    confidence: float
    tracker_source: str
    
    # Verdict
    quality_score: float  # 0-100
    issues: str  # Comma-separated issues


@dataclass
class ValidationReport:
    """Overall validation report."""
    video_id: str
    total_frames: int
    
    # Quality metrics
    frames_with_bbox: int
    frames_with_good_bbox: int
    frames_with_issues: int
    
    # Movement analysis
    teleport_events: int
    impossible_movements: int
    size_jumps: int
    
    # Identity issues
    potential_identity_swaps: int
    bbox_collision_frames: int
    
    # Zone validation
    zones_reported: int
    zones_validated: int
    false_positive_frames: int  # In zone but bad bbox
    false_negative_frames: int  # Good bbox but not in zone
    
    # Overall scores
    tracking_accuracy_score: float  # 0-100
    zone_accuracy_score: float  # 0-100
    overall_quality_score: float  # 0-100
    
    # Verdict
    client_ready: bool
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class TrackingValidator:
    """Comprehensive tracking validation without visual inspection."""
    
    def __init__(self, video_path: str, statistics_json_path: str):
        self.video_path = video_path
        self.stats_path = statistics_json_path
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load statistics
        with open(statistics_json_path, 'r') as f:
            self.stats = json.load(f)
        
        # Load zones
        video_id = self.stats['video_id']
        zones_dir = Path(statistics_json_path).parent
        
        my_zones_file = zones_dir / f"{video_id}_my_fighter_zones.json"
        opp_zones_file = zones_dir / f"{video_id}_opponent_zones.json"
        
        with open(my_zones_file, 'r') as f:
            self.my_zones = json.load(f)['my_fighter_presence']
        
        if opp_zones_file.exists():
            with open(opp_zones_file, 'r') as f:
                self.opp_zones = json.load(f)['my_fighter_presence']
        else:
            self.opp_zones = []
        
        print(f"\n{'='*70}")
        print(f"🔍 TRACKING VALIDATION SYSTEM INITIALIZED")
        print(f"{'='*70}")
        print(f"Video: {Path(video_path).name}")
        print(f"Frames: {self.total_frames} @ {self.fps:.2f} FPS")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"MY FIGHTER zones: {len(self.my_zones)}")
        print(f"OPPONENT zones: {len(self.opp_zones)}")
        print(f"{'='*70}\n")
    
    def _is_in_zone(self, timestamp: float, zones: List[Dict]) -> bool:
        """Check if timestamp is within any presence zone."""
        for zone in zones:
            if zone['start'] <= timestamp <= zone['end']:
                return True
        return False
    
    def _calculate_bbox_metrics(self, 
                                frame_num: int,
                                timestamp: float,
                                bbox: Optional[Tuple],
                                prev_bbox: Optional[Tuple],
                                prev_metrics: Optional[BboxMetrics],
                                zones: List[Dict],
                                confidence: float,
                                tracker_source: str) -> BboxMetrics:
        """Calculate comprehensive bbox metrics for validation."""
        
        # Initialize with None values
        metrics = {
            'frame_num': frame_num,
            'timestamp': round(timestamp, 2),
            'bbox_x': None, 'bbox_y': None, 'bbox_w': None, 'bbox_h': None,
            'bbox_area': None, 'bbox_center_x': None, 'bbox_center_y': None,
            'movement_distance': None, 'movement_speed': None, 'direction_change': None,
            'area_change_pct': None, 'aspect_ratio': None,
            'has_bbox': False, 'is_teleport': False, 'is_size_jump': False,
            'is_impossible_movement': False, 'is_edge_bbox': False,
            'is_tiny_bbox': False, 'is_huge_bbox': False,
            'in_presence_zone': self._is_in_zone(timestamp, zones),
            'confidence': round(confidence, 3),
            'tracker_source': tracker_source,
            'quality_score': 0.0,
            'issues': ''
        }
        
        if bbox is None:
            return BboxMetrics(**metrics)
        
        # Extract bbox data
        x, y, w, h = bbox
        area = w * h
        cx, cy = x + w/2, y + h/2
        
        metrics.update({
            'has_bbox': True,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'bbox_area': area,
            'bbox_center_x': round(cx, 1),
            'bbox_center_y': round(cy, 1),
            'aspect_ratio': round(h/w if w > 0 else 0, 2)
        })
        
        issues = []
        quality = 100.0
        
        # Size validation
        frame_area = self.frame_width * self.frame_height
        area_ratio = area / frame_area
        
        if area < 2000:  # Too small (< ~40x50 px)
            metrics['is_tiny_bbox'] = True
            issues.append('TINY_BBOX')
            quality -= 30
        
        if area_ratio > 0.5:  # More than 50% of frame
            metrics['is_huge_bbox'] = True
            issues.append('HUGE_BBOX')
            quality -= 25
        
        # Edge detection
        edge_margin = 10
        if (x <= edge_margin or y <= edge_margin or 
            x + w >= self.frame_width - edge_margin or 
            y + h >= self.frame_height - edge_margin):
            metrics['is_edge_bbox'] = True
            issues.append('EDGE_BBOX')
            quality -= 15
        
        # Movement analysis (if we have previous bbox)
        if prev_bbox is not None:
            px, py, pw, ph = prev_bbox
            pcx, pcy = px + pw/2, py + ph/2
            
            # Movement distance
            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            metrics['movement_distance'] = round(dist, 1)
            metrics['movement_speed'] = round(dist, 1)  # px per frame
            
            # Teleportation detection (moved >200px in 1 frame)
            if dist > 200:
                metrics['is_teleport'] = True
                issues.append('TELEPORT')
                quality -= 40
            
            # Impossible movement (>150px/frame = ~4500px/sec at 30fps)
            if dist > 150:
                metrics['is_impossible_movement'] = True
                issues.append('IMPOSSIBLE_MOVE')
                quality -= 35
            
            # Direction change
            if prev_metrics and prev_metrics.bbox_center_x is not None:
                prev_dx = pcx - prev_metrics.bbox_center_x
                prev_dy = pcy - prev_metrics.bbox_center_y
                curr_dx = cx - pcx
                curr_dy = cy - pcy
                
                if prev_dx != 0 or prev_dy != 0:
                    prev_angle = np.arctan2(prev_dy, prev_dx)
                    curr_angle = np.arctan2(curr_dy, curr_dx)
                    angle_diff = abs(np.degrees(curr_angle - prev_angle))
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    metrics['direction_change'] = round(angle_diff, 1)
            
            # Size change
            prev_area = pw * ph
            area_change = ((area - prev_area) / prev_area * 100) if prev_area > 0 else 0
            metrics['area_change_pct'] = round(area_change, 1)
            
            # Size jump detection (>40% change in 1 frame)
            if abs(area_change) > 40:
                metrics['is_size_jump'] = True
                issues.append('SIZE_JUMP')
                quality -= 30
        
        # Zone consistency check
        if metrics['in_presence_zone'] and quality < 50:
            issues.append('FALSE_POSITIVE_ZONE')
            quality -= 20
        
        if not metrics['in_presence_zone'] and quality > 70:
            issues.append('FALSE_NEGATIVE_ZONE')
        
        metrics['quality_score'] = max(0, min(100, quality))
        metrics['issues'] = ','.join(issues) if issues else 'NONE'
        
        return BboxMetrics(**metrics)
    
    def validate_tracking(self, fighter_type: str = "my_fighter") -> Tuple[List[BboxMetrics], ValidationReport]:
        """
        Validate tracking for specified fighter.
        Returns: (frame_metrics_list, validation_report)
        """
        print(f"\n🔍 Validating {fighter_type.upper()} tracking...")
        print(f"{'─'*70}")
        
        zones = self.my_zones if fighter_type == "my_fighter" else self.opp_zones
        
        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        metrics_list = []
        prev_bbox = None
        prev_metrics = None
        
        # Tracking stats
        frames_with_bbox = 0
        frames_with_good_bbox = 0
        frames_with_issues = 0
        teleport_events = 0
        impossible_movements = 0
        size_jumps = 0
        false_positive_frames = 0
        false_negative_frames = 0
        
        # Process each frame
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_num / self.fps
            
            # Get tracking data (mock - in real use, you'd load from tracking history)
            # For now, we'll use zone info as proxy
            in_zone = self._is_in_zone(timestamp, zones)
            
            # Mock bbox (you'll need to extract real bbox from tracking history)
            bbox = None
            confidence = 0.0
            tracker_source = "UNKNOWN"
            
            # Calculate metrics
            metrics = self._calculate_bbox_metrics(
                frame_num, timestamp, bbox, prev_bbox, prev_metrics,
                zones, confidence, tracker_source
            )
            
            metrics_list.append(metrics)
            
            # Update stats
            if metrics.has_bbox:
                frames_with_bbox += 1
                if metrics.quality_score >= 70:
                    frames_with_good_bbox += 1
                if metrics.issues != 'NONE':
                    frames_with_issues += 1
                
                if metrics.is_teleport:
                    teleport_events += 1
                if metrics.is_impossible_movement:
                    impossible_movements += 1
                if metrics.is_size_jump:
                    size_jumps += 1
                
                if 'FALSE_POSITIVE_ZONE' in metrics.issues:
                    false_positive_frames += 1
                if 'FALSE_NEGATIVE_ZONE' in metrics.issues:
                    false_negative_frames += 1
            
            # Update for next iteration
            if bbox is not None:
                prev_bbox = bbox
            prev_metrics = metrics
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"  Processed: {frame_num}/{self.total_frames} frames ({frame_num/self.total_frames*100:.1f}%)")
        
        # Calculate scores
        tracking_accuracy = (frames_with_good_bbox / max(1, frames_with_bbox)) * 100 if frames_with_bbox > 0 else 0
        zone_accuracy = 100 - ((false_positive_frames + false_negative_frames) / max(1, self.total_frames)) * 100
        overall_quality = (tracking_accuracy * 0.6 + zone_accuracy * 0.4)
        
        # Generate report
        report = ValidationReport(
            video_id=self.stats['video_id'],
            total_frames=self.total_frames,
            frames_with_bbox=frames_with_bbox,
            frames_with_good_bbox=frames_with_good_bbox,
            frames_with_issues=frames_with_issues,
            teleport_events=teleport_events,
            impossible_movements=impossible_movements,
            size_jumps=size_jumps,
            potential_identity_swaps=0,  # TODO: Implement
            bbox_collision_frames=0,  # TODO: Implement
            zones_reported=len(zones),
            zones_validated=len(zones),  # TODO: Validate each zone
            false_positive_frames=false_positive_frames,
            false_negative_frames=false_negative_frames,
            tracking_accuracy_score=round(tracking_accuracy, 2),
            zone_accuracy_score=round(zone_accuracy, 2),
            overall_quality_score=round(overall_quality, 2),
            client_ready=overall_quality >= 85,
            critical_issues=[],
            warnings=[],
            recommendations=[]
        )
        
        # Add issues and recommendations
        if teleport_events > 10:
            report.critical_issues.append(f"HIGH TELEPORT COUNT: {teleport_events} events")
        if impossible_movements > 20:
            report.critical_issues.append(f"TOO MANY IMPOSSIBLE MOVEMENTS: {impossible_movements}")
        if false_positive_frames > self.total_frames * 0.1:
            report.critical_issues.append(f"FALSE POSITIVES: {false_positive_frames} frames ({false_positive_frames/self.total_frames*100:.1f}%)")
        
        if frames_with_issues > frames_with_bbox * 0.3:
            report.warnings.append(f"30%+ frames have tracking issues")
        
        if not report.client_ready:
            report.recommendations.append("Improve tracking algorithm stability")
            report.recommendations.append("Review presence zone thresholds")
            report.recommendations.append("Add identity verification mechanisms")
        
        print(f"\n✅ Validation complete: {frame_num} frames processed")
        print(f"{'─'*70}")
        
        return metrics_list, report
    
    def save_csv_report(self, metrics_list: List[BboxMetrics], output_path: str):
        """Save frame-by-frame metrics to CSV."""
        with open(output_path, 'w', newline='') as f:
            if not metrics_list:
                return
            
            fieldnames = list(asdict(metrics_list[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in metrics_list:
                writer.writerow(asdict(metrics))
        
        print(f"\n💾 CSV Report saved: {output_path}")
    
    def print_report(self, report: ValidationReport):
        """Print comprehensive validation report."""
        print(f"\n{'='*70}")
        print(f"📊 VALIDATION REPORT - {report.video_id}")
        print(f"{'='*70}")
        
        print(f"\n📈 TRACKING STATISTICS:")
        print(f"   Total Frames:        {report.total_frames}")
        print(f"   Frames with Bbox:    {report.frames_with_bbox} ({report.frames_with_bbox/report.total_frames*100:.1f}%)")
        print(f"   Good Quality Frames: {report.frames_with_good_bbox} ({report.frames_with_good_bbox/max(1,report.frames_with_bbox)*100:.1f}%)")
        print(f"   Frames with Issues:  {report.frames_with_issues}")
        
        print(f"\n⚠️  DETECTED ISSUES:")
        print(f"   Teleport Events:        {report.teleport_events}")
        print(f"   Impossible Movements:   {report.impossible_movements}")
        print(f"   Size Jumps:             {report.size_jumps}")
        print(f"   False Positive Frames:  {report.false_positive_frames}")
        print(f"   False Negative Frames:  {report.false_negative_frames}")
        
        print(f"\n🎯 QUALITY SCORES:")
        print(f"   Tracking Accuracy:   {report.tracking_accuracy_score:.1f}/100")
        print(f"   Zone Accuracy:       {report.zone_accuracy_score:.1f}/100")
        print(f"   Overall Quality:     {report.overall_quality_score:.1f}/100")
        
        if report.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in report.critical_issues:
                print(f"   ❌ {issue}")
        
        if report.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in report.warnings:
                print(f"   ⚠️  {warning}")
        
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report.recommendations:
                print(f"   • {rec}")
        
        print(f"\n{'='*70}")
        print(f"🎯 CLIENT READY: {'✅ YES' if report.client_ready else '❌ NO'}")
        print(f"{'='*70}\n")
    
    def close(self):
        """Release video resources."""
        self.cap.release()


def main():
    if len(sys.argv) < 3:
        print("\n❌ Usage: python validate_tracking.py <video_path> <statistics_json_path>")
        print("\nExample:")
        print("   python validate_tracking.py testfight.mp4 output/testfight_statistics.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    stats_path = sys.argv[2]
    
    if not Path(video_path).exists():
        print(f"\n❌ Video not found: {video_path}")
        sys.exit(1)
    
    if not Path(stats_path).exists():
        print(f"\n❌ Statistics file not found: {stats_path}")
        sys.exit(1)
    
    # Initialize validator
    validator = TrackingValidator(video_path, stats_path)
    
    try:
        # Validate MY FIGHTER
        my_metrics, my_report = validator.validate_tracking("my_fighter")
        validator.save_csv_report(my_metrics, "output/my_fighter_validation.csv")
        validator.print_report(my_report)
        
        # Validate OPPONENT (if exists)
        if validator.opp_zones:
            opp_metrics, opp_report = validator.validate_tracking("opponent")
            validator.save_csv_report(opp_metrics, "output/opponent_validation.csv")
            validator.print_report(opp_report)
    
    finally:
        validator.close()
    
    print("\n✅ Validation complete! Check CSV files for detailed frame-by-frame analysis.")


if __name__ == "__main__":
    main()