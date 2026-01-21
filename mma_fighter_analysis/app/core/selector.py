

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class FighterSelector:
   
    
    def __init__(self, frame: np.ndarray):
       
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.bboxes = []  # List of selected bounding boxes
        
       
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_bbox = None
        
       
        self.window_name = "MMA Fighter Selection"
        
        
        self.color_drawing = (255, 0, 0)      # Blue while drawing
        self.color_confirmed = (0, 255, 0)    # Green when confirmed
        self.color_my_fighter = (0, 0, 255)   # Red for my fighter
        self.color_opponent = (255, 255, 0)   # Cyan for opponent
        self.color_text = (255, 255, 255)     # White text
        
    def select_fighters(self, num_fighters: int = 2) -> Dict[str, Tuple[int, int, int, int]]:
        
        self._draw_bounding_boxes(num_fighters)
        
        # Step 2: Confirm roles
        if len(self.bboxes) >= 2:
            roles = self._confirm_fighter_roles()
            return roles
        elif len(self.bboxes) == 1:
            print("\n⚠️  Only one fighter selected. Assuming single fighter tracking.")
            return {
                "my_fighter": self.bboxes[0],
                "opponent": None
            }
        else:
            raise ValueError("No fighters selected!")
    
    def _draw_bounding_boxes(self, num_fighters: int):
       
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("🥊 MANUAL FIGHTER SELECTION - STEP 1: DRAW BOXES")
        print("=" * 60)
        print(f"📌 Select {num_fighters} fighters")
        print("\nInstructions:")
        print("  1. Click and DRAG to draw a box around the first fighter")
        print("  2. Press SPACE to confirm the box")
        print("  3. Draw a box around the second fighter")
        print("  4. Press SPACE to confirm")
        print("  5. Press 'R' to reset last box if needed")
        print("=" * 60 + "\n")
        
        while len(self.bboxes) < num_fighters:
            
            display = self.original_frame.copy()
            
           
            for idx, bbox in enumerate(self.bboxes):
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), 
                            self.color_confirmed, 3)
                
               
                label = f"Fighter {idx + 1}"
                cv2.putText(display, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          self.color_confirmed, 2)
            
            
            if self.drawing and self.start_point and self.current_point:
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                cv2.rectangle(display, (x1, y1), (x2, y2), 
                            self.color_drawing, 2)
            
           
            elif self.temp_bbox is not None:
                x, y, w, h = self.temp_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), 
                            self.color_drawing, 2)
                
                
                cv2.putText(display, "Press SPACE to confirm", 
                          (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          self.color_drawing, 2)
            
          
            status = f"Selected: {len(self.bboxes)}/{num_fighters}"
            cv2.putText(display, status, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                       self.color_text, 2)
            
           
            instructions = [
                "Click & Drag: Draw box",
                "SPACE: Confirm",
                "R: Reset last",
                "ESC: Cancel"
            ]
            y_offset = 80
            for instruction in instructions:
                cv2.putText(display, instruction, (20, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          self.color_text, 1)
                y_offset += 25
            
           
            cv2.imshow(self.window_name, display)
            
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                if len(self.bboxes) >= 1:
                    print(f"\n✅ Finished with {len(self.bboxes)} fighter(s)")
                    break
                else:
                    print("\n⚠️  Selection cancelled")
                    cv2.destroyWindow(self.window_name)
                    raise ValueError("Selection cancelled by user")
            
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
    
    def _confirm_fighter_roles(self) -> Dict[str, Tuple[int, int, int, int]]:
       
        print("\n" + "=" * 60)
        print("🥊 STEP 2: CONFIRM FIGHTER ROLES")
        print("=" * 60)
        print("📌 Which fighter is YOURS?")
        print("\nPress:")
        print("  1 - Fighter 1 is MY fighter (Red box)")
        print("  2 - Fighter 2 is MY fighter (Red box)")
        print("=" * 60 + "\n")
        
        my_fighter_idx = None
        
        while my_fighter_idx is None:
            # Display both fighters with numbers
            display = self.original_frame.copy()
            
            for idx, bbox in enumerate(self.bboxes):
                x, y, w, h = bbox
                
                # Draw box
                color = self.color_confirmed
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
                
                # Large number label
                label = f"{idx + 1}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
                label_x = x + (w - label_size[0]) // 2
                label_y = y + (h + label_size[1]) // 2
                
                # Background for number
                cv2.rectangle(display, 
                            (label_x - 10, label_y - label_size[1] - 10),
                            (label_x + label_size[0] + 10, label_y + 10),
                            (0, 0, 0), -1)
                
                # Number
                cv2.putText(display, label, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                          self.color_text, 3)
                
                # Label at top
                top_label = f"Fighter {idx + 1}"
                cv2.putText(display, top_label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          color, 2)
            
            # Instruction text
            instruction = "Press 1 or 2 to select YOUR fighter"
            cv2.putText(display, instruction, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                       (0, 255, 255), 2)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                my_fighter_idx = 0
                print("✅ Fighter 1 selected as YOUR fighter")
            elif key == ord('2'):
                my_fighter_idx = 1
                print("✅ Fighter 2 selected as YOUR fighter")
            elif key == 27:  # ESC
                print("\n⚠️  Role selection cancelled")
                cv2.destroyWindow(self.window_name)
                raise ValueError("Role selection cancelled by user")
        
       
        self._show_final_confirmation(my_fighter_idx)
        
        cv2.destroyWindow(self.window_name)
        
        
        opponent_idx = 1 - my_fighter_idx  
        
        result = {
            "my_fighter": self.bboxes[my_fighter_idx],
            "opponent": self.bboxes[opponent_idx]
        }
        
       
        print("\n" + "=" * 60)
        print("✅ FIGHTER ROLES CONFIRMED")
        print("=" * 60)
        x, y, w, h = result["my_fighter"]
        print(f"🔴 MY FIGHTER:  x={x}, y={y}, width={w}, height={h}")
        x, y, w, h = result["opponent"]
        print(f"🔵 OPPONENT:    x={x}, y={y}, width={w}, height={h}")
        print("=" * 60 + "\n")
        
        return result
    
    def _show_final_confirmation(self, my_fighter_idx: int):
        
        display = self.original_frame.copy()
        
        opponent_idx = 1 - my_fighter_idx
        
        # Draw MY fighter (RED)
        x, y, w, h = self.bboxes[my_fighter_idx]
        cv2.rectangle(display, (x, y), (x + w, y + h), 
                     self.color_my_fighter, 4)
        cv2.putText(display, "MY FIGHTER", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                   self.color_my_fighter, 3)
        
        # Draw OPPONENT (CYAN)
        x, y, w, h = self.bboxes[opponent_idx]
        cv2.rectangle(display, (x, y), (x + w, y + h), 
                     self.color_opponent, 4)
        cv2.putText(display, "OPPONENT", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                   self.color_opponent, 3)
        
       
        cv2.putText(display, "Roles Confirmed! Starting analysis...", 
                   (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                   (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display)
        cv2.waitKey(2000)  
    
    def _mouse_callback(self, event, x, y, flags, param):
       
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.temp_bbox = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                
                self.current_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
               
                if width > 10 and height > 10:
                    self.temp_bbox = (x_min, y_min, width, height)
                    print(f"📦 Box drawn: x={x_min}, y={y_min}, w={width}, h={height} (Press SPACE to confirm)")
                else:
                    print("⚠️  Box too small, draw a larger box")
                    self.temp_bbox = None
                
                self.start_point = None
                self.current_point = None


def quick_select_fighters(frame: np.ndarray, num_fighters: int = 2) -> Dict[str, Tuple[int, int, int, int]]:
   
    selector = FighterSelector(frame)
    return selector.select_fighters(num_fighters)
