

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoLoader:
    
    
    def __init__(self, video_path: str):
       
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
        
       
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
       
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
      
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            raise ValueError("❌ Cannot read first frame from video")
        
        return frame
    
    def get_frame(self, frame_num: int) -> Optional[np.ndarray]:
        
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
      
        return frame_num / self.fps if self.fps > 0 else 0.0
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
       
        frame_num = int(timestamp * self.fps)
        return self.get_frame(frame_num)
    
    def get_video_info(self) -> dict:
        
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
       
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def reset(self):
        
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
