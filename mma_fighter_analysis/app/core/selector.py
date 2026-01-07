"""
Manual Fighter Selection Module
Allows user to manually select fighters by drawing bounding boxes.
NO machine learning. NO automation. User decides.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FighterSelector:
    """
    Interactive bounding box selector for manual fighter identification.
    
    How it works:
    1. Display first frame
    2. User clicks and drags to draw rectangle around Fighter 1
    3. Press SPACE to confirm
    4. User draws rectangle around Fighter 2
    5. Press SPACE to confirm
    6. Returns both bounding boxes
    """
    
    def __init__(self, frame: np.ndarray):
        """
        Initialize selector with first frame.
        
        Args:
            frame: First frame of video (BGR numpy array)
        """
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.bboxes = []  # List of selected bounding boxes
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_bbox = None
        
        # Window name
        self.window_name = "MMA Fighter Selection"
        
        # Colors
        self.color_drawing = (255, 0, 0)      # Blue while drawing
        self.color_confirmed = (0, 255, 0)    # Green when confirmed
        self.color_text = (255, 255, 255)     # White text
        
    def select_fighters(self, num_fighters: int = 2) -> List[Tuple[int, int, int, int]]:
        """
        Interactive selection of fighters.
        
        Args:
            num_fighters: Number of fighters to select (default: 2)
            
        Returns:
            List of bounding boxes as [(x, y, w, h), ...]
            
        Instructions displayed to user:
        - Click and drag to draw box
        - Press SPACE to confirm current box
        - Press 'R' to reset last box
        - Press ESC to finish (if selected enough fighters)
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("🥊 MANUAL FIGHTER SELECTION")
        print("=" * 60)
        print(f"📌 Select {num_fighters} fighters")
        print("\nInstructions:")
        print("  1. Click and DRAG to draw a box around the fighter")
        print("  2. Press SPACE to confirm the box")
        print("  3. Repeat for next fighter")
        print("  4. Press ESC when done (or after selecting all fighters)")
        print("  5. Press 'R' to reset and start over")
        print("=" * 60 + "\n")
        
        while len(self.bboxes) < num_fighters:
            # Prepare display
            display = self.original_frame.copy()
            
            # Draw confirmed bboxes
            for idx, bbox in enumerate(self.bboxes):
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), 
                            self.color_confirmed, 3)
                
                # Label
                label = f"Fighter {idx + 1}"
                cv2.putText(display, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          self.color_confirmed, 2)
            
            # Draw current bbox being drawn
            if self.drawing and self.start_point and self.current_point:
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                cv2.rectangle(display, (x1, y1), (x2, y2), 
                            self.color_drawing, 2)
            
            # Draw temporary confirmed bbox (before SPACE pressed)
            elif self.temp_bbox is not None:
                x, y, w, h = self.temp_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), 
                            self.color_drawing, 2)
                
                # Instruction
                cv2.putText(display, "Press SPACE to confirm", 
                          (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          self.color_drawing, 2)
            
            # Status text
            status = f"Selected: {len(self.bboxes)}/{num_fighters}"
            cv2.putText(display, status, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                       self.color_text, 2)
            
            # Instructions on frame
            instructions = [
                "Click & Drag: Draw box",
                "SPACE: Confirm",
                "R: Reset",
                "ESC: Finish"
            ]
            y_offset = 80
            for instruction in instructions:
                cv2.putText(display, instruction, (20, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          self.color_text, 1)
                y_offset += 25
            
            # Show frame
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                if len(self.bboxes) >= 1:  # Allow finishing with at least 1
                    print(f"\n✅ Finished with {len(self.bboxes)} fighter(s)")
                    break
                else:
                    print("\n⚠️  Select at least 1 fighter before finishing")
            
            elif key == 32:  # SPACE
                if self.temp_bbox is not None and not self.drawing:
                    self.bboxes.append(self.temp_bbox)
                    self.temp_bbox = None
                    print(f"✅ Fighter {len(self.bboxes)} confirmed: {self.bboxes[-1]}")
            
            elif key == ord('r') or key == ord('R'):  # Reset
                if self.bboxes:
                    removed = self.bboxes.pop()
                    print(f"↩️  Removed last selection: {removed}")
                self.temp_bbox = None
                self.drawing = False
                self.start_point = None
                self.current_point = None
        
        cv2.destroyWindow(self.window_name)
        
        # Final confirmation
        print("\n" + "=" * 60)
        print("✅ SELECTION COMPLETE")
        print("=" * 60)
        for idx, bbox in enumerate(self.bboxes):
            x, y, w, h = bbox
            print(f"Fighter {idx + 1}: x={x}, y={y}, width={w}, height={h}")
        print("=" * 60 + "\n")
        
        return self.bboxes
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for drawing rectangles.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.temp_bbox = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update current point
                self.current_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Calculate bbox (top-left corner + width/height)
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # Validate bbox size (must be at least 10x10 pixels)
                if width > 10 and height > 10:
                    self.temp_bbox = (x_min, y_min, width, height)
                    print(f"📦 Box drawn: x={x_min}, y={y_min}, w={width}, h={height} (Press SPACE to confirm)")
                else:
                    print("⚠️  Box too small, draw a larger box")
                    self.temp_bbox = None
                
                self.start_point = None
                self.current_point = None


def quick_select_fighters(frame: np.ndarray, num_fighters: int = 2) -> List[Tuple[int, int, int, int]]:
    """
    Convenience function for quick fighter selection.
    
    Args:
        frame: First frame of video
        num_fighters: Number of fighters to select
        
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    selector = FighterSelector(frame)
    return selector.select_fighters(num_fighters)