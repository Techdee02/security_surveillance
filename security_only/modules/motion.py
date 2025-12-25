"""
Motion detection module for security surveillance system
Optimized for Raspberry Pi 3 - lightweight background subtraction
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import time


class MotionDetector:
    """
    Lightweight motion detection using background subtraction
    Optimized for Pi 3's limited CPU resources
    """
    
    def __init__(self, 
                 sensitivity: float = 0.015,
                 min_area: int = 500,
                 blur_size: int = 21,
                 history: int = 500,
                 detect_shadows: bool = False):
        """
        Initialize motion detector
        
        Args:
            sensitivity: Motion sensitivity threshold (0-1), lower = more sensitive
            min_area: Minimum contour area to consider as motion (pixels)
            blur_size: Gaussian blur kernel size (must be odd)
            history: Number of frames for background model
            detect_shadows: Enable shadow detection (costs CPU)
        """
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        
        # Use MOG2 background subtractor (efficient on Pi 3)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=16,  # Lower = more sensitive
            detectShadows=detect_shadows
        )
        
        # Stats
        self.frame_count = 0
        self.motion_count = 0
        self.last_motion_time = None
        
        # Motion regions
        self.motion_contours = []
        self.motion_mask = None
        
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """
        Detect motion in frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            (has_motion, bounding_boxes) where bounding_boxes is list of (x, y, w, h)
        """
        self.frame_count += 1
        
        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)
        
        # Remove shadows (value 127) if shadow detection is on
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to fill gaps
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Store for visualization
        self.motion_mask = fg_mask
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and get bounding boxes
        motion_boxes = []
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
            valid_contours.append(contour)
        
        self.motion_contours = valid_contours
        
        # Check if motion detected
        has_motion = len(motion_boxes) > 0
        
        if has_motion:
            self.motion_count += 1
            self.last_motion_time = time.time()
        
        return has_motion, motion_boxes
    
    def draw_motion(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw motion bounding boxes on frame
        
        Args:
            frame: Input BGR frame
            boxes: List of (x, y, w, h) bounding boxes
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for x, y, w, h in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "MOTION", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add motion indicator
        if boxes:
            cv2.putText(annotated, f"Motion: {len(boxes)} region(s)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def get_motion_mask(self) -> Optional[np.ndarray]:
        """Get the current motion mask for visualization"""
        return self.motion_mask
    
    def get_stats(self) -> dict:
        """Get motion detection statistics"""
        motion_rate = self.motion_count / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'motion_events': self.motion_count,
            'motion_rate': motion_rate,
            'last_motion': self.last_motion_time,
            'seconds_since_motion': time.time() - self.last_motion_time if self.last_motion_time else None
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.frame_count = 0
        self.motion_count = 0
        self.last_motion_time = None
    
    def calibrate(self, frames: List[np.ndarray], warmup_frames: int = 30):
        """
        Calibrate background model with initial frames
        
        Args:
            frames: List of initial frames for calibration
            warmup_frames: Number of frames to use for warmup
        """
        print(f"Calibrating motion detector with {min(len(frames), warmup_frames)} frames...")
        
        for i, frame in enumerate(frames[:warmup_frames]):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            self.bg_subtractor.apply(blurred, learningRate=0.5)  # Fast learning during calibration
        
        print("✅ Calibration complete")


class SmartMotionFilter:
    """
    Smart motion filtering to reduce false positives
    Only trigger person detection when significant motion detected
    """
    
    def __init__(self, 
                 motion_threshold: float = 0.02,
                 cooldown_seconds: int = 2,
                 min_motion_frames: int = 2):
        """
        Initialize smart motion filter
        
        Args:
            motion_threshold: Minimum motion percentage to trigger detection
            cooldown_seconds: Seconds to wait after last motion before re-triggering
            min_motion_frames: Minimum consecutive frames with motion to trigger
        """
        self.motion_threshold = motion_threshold
        self.cooldown_seconds = cooldown_seconds
        self.min_motion_frames = min_motion_frames
        
        self.motion_frame_count = 0
        self.last_trigger_time = 0
        
    def should_run_detection(self, has_motion: bool, motion_boxes: List) -> bool:
        """
        Determine if person detection should run based on motion
        
        Args:
            has_motion: Whether motion was detected
            motion_boxes: List of motion bounding boxes
            
        Returns:
            True if person detection should run
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_trigger_time < self.cooldown_seconds:
            return False
        
        # Update motion frame counter
        if has_motion:
            self.motion_frame_count += 1
        else:
            self.motion_frame_count = 0
        
        # Trigger if enough consecutive motion frames
        if self.motion_frame_count >= self.min_motion_frames:
            self.last_trigger_time = current_time
            self.motion_frame_count = 0  # Reset
            return True
        
        return False
    
    def reset(self):
        """Reset filter state"""
        self.motion_frame_count = 0
        self.last_trigger_time = 0
