"""
Video Loading Module
Handles video file loading, frame extraction, and metadata retrieval.
Uses OpenCV for video processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoLoader:
    """
    Loads and manages video files for MMA analysis.
    
    Responsibilities:
    - Load video from file path
    - Extract video metadata (FPS, dimensions, total frames)
    - Provide frame-by-frame access
    - Handle corrupted/missing frames gracefully
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video loader.
        
        Args:
            video_path: Path to video file (.mp4, .mov, .webm)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)
        
        # Validate file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"❌ Video not found: {video_path}")
        
        # Validate file extension
        valid_extensions = ['.mp4', '.mov', '.webm', '.avi', '.mkv']
        if self.video_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"❌ Invalid video format. Use: {', '.join(valid_extensions)}")
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}\n"
                           f"   File may be corrupted or unsupported codec.")
        
        # Extract metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Print video info
        print("=" * 60)
        print("📹 VIDEO LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"File:         {self.video_path.name}")
        print(f"Resolution:   {self.width}x{self.height}")
        print(f"FPS:          {self.fps:.2f}")
        print(f"Total Frames: {self.total_frames}")
        print(f"Duration:     {self._format_duration(self.duration)}")
        print("=" * 60)
    
    def get_first_frame(self) -> np.ndarray:
        """
        Get the first frame of the video for manual selection.
        
        Returns:
            First frame as numpy array (BGR format)
            
        Raises:
            ValueError: If first frame cannot be read
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            raise ValueError("❌ Cannot read first frame from video")
        
        return frame
    
    def get_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_num: Frame index (0-based)
            
        Returns:
            Frame as numpy array, or None if frame cannot be read
        """
        if frame_num < 0 or frame_num >= self.total_frames:
            return None
        
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print(f"⚠️  Warning: Could not read frame {frame_num}")
                return None
            
            return frame
        
        except Exception as e:
            print(f"⚠️  Warning: Error reading frame {frame_num}: {e}")
            return None
    
    def get_timestamp(self, frame_num: int) -> float:
        """
        Convert frame number to timestamp in seconds.
        
        Args:
            frame_num: Frame index
            
        Returns:
            Timestamp in seconds
        """
        return frame_num / self.fps if self.fps > 0 else 0.0
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame at that timestamp, or None if invalid
        """
        frame_num = int(timestamp * self.fps)
        return self.get_frame(frame_num)
    
    def get_video_info(self) -> dict:
        """
        Get all video metadata as dictionary.
        
        Returns:
            Dictionary with video information
        """
        return {
            "path": str(self.video_path),
            "filename": self.video_path.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration,
            "duration_formatted": self._format_duration(self.duration)
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Release video resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
