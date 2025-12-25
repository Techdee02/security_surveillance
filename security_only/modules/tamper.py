"""
Tamper Detection Module
Detects camera covering and camera movement
"""
import cv2
import numpy as np
from collections import deque
import time


class TamperDetector:
    """Detects camera tampering (covering, movement, obstruction)"""
    
    def __init__(self, 
                 brightness_threshold=20,
                 movement_threshold=0.15,
                 history_size=30,
                 check_interval=1.0):
        """
        Initialize tamper detector
        
        Args:
            brightness_threshold: Average brightness below this = covered (0-255)
            movement_threshold: Frame difference percentage indicating movement
            history_size: Number of frames to track for baseline
            check_interval: Seconds between tamper checks (reduce CPU usage)
        """
        self.brightness_threshold = brightness_threshold
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        self.check_interval = check_interval
        
        # Baseline tracking
        self.brightness_history = deque(maxlen=history_size)
        self.baseline_brightness = None
        self.baseline_frame = None
        self.baseline_established = False
        
        # State tracking
        self.last_check_time = 0
        self.is_covered = False
        self.is_moved = False
        self.tamper_events = []
        
        # Statistics
        self.total_checks = 0
        self.cover_detections = 0
        self.movement_detections = 0
        
        print(f"🛡️ TamperDetector initialized:")
        print(f"   Brightness threshold: {brightness_threshold}")
        print(f"   Movement threshold: {movement_threshold * 100:.1f}%")
        print(f"   Baseline history: {history_size} frames")
        print(f"   Check interval: {check_interval}s")
    
    def update_baseline(self, frame):
        """
        Update baseline brightness from normal frames
        
        Args:
            frame: Video frame to add to baseline
        """
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        self.brightness_history.append(avg_brightness)
        
        # Establish baseline after collecting enough samples
        if len(self.brightness_history) >= self.history_size and not self.baseline_established:
            self.baseline_brightness = np.median(list(self.brightness_history))
            self.baseline_frame = frame.copy()
            self.baseline_established = True
            print(f"✅ Tamper baseline established: avg brightness = {self.baseline_brightness:.1f}")
    
    def check_tampering(self, frame):
        """
        Check for camera tampering
        
        Args:
            frame: Current video frame
        
        Returns:
            Dict with tamper detection results
        """
        current_time = time.time()
        
        # Rate limiting - only check at intervals
        if current_time - self.last_check_time < self.check_interval:
            return {
                'checked': False,
                'covered': self.is_covered,
                'moved': self.is_moved,
                'tamper_detected': self.is_covered or self.is_moved
            }
        
        self.last_check_time = current_time
        self.total_checks += 1
        
        # Check 1: Camera covering (sudden darkness)
        covered = self._check_covering(frame)
        
        # Check 2: Camera movement (scene shift)
        moved = False
        if self.baseline_established and not covered:
            moved = self._check_movement(frame)
        
        # Update state
        prev_covered = self.is_covered
        prev_moved = self.is_moved
        
        self.is_covered = covered
        self.is_moved = moved
        
        # Log new tamper events
        if covered and not prev_covered:
            self.cover_detections += 1
            event = {
                'type': 'camera_covered',
                'timestamp': current_time,
                'brightness': np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            }
            self.tamper_events.append(event)
            print(f"🚨 TAMPER DETECTED: Camera covered!")
        
        if moved and not prev_moved:
            self.movement_detections += 1
            event = {
                'type': 'camera_moved',
                'timestamp': current_time,
                'difference': self._calculate_frame_difference(frame)
            }
            self.tamper_events.append(event)
            print(f"🚨 TAMPER DETECTED: Camera moved!")
        
        # Clear alerts if tampering stopped
        if not covered and prev_covered:
            print(f"✅ Camera uncovered")
        
        if not moved and prev_moved:
            print(f"✅ Camera position restored")
        
        return {
            'checked': True,
            'covered': covered,
            'moved': moved,
            'tamper_detected': covered or moved,
            'brightness': np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
            'baseline_brightness': self.baseline_brightness
        }
    
    def _check_covering(self, frame):
        """
        Check if camera is covered (very dark)
        
        Args:
            frame: Video frame
        
        Returns:
            True if camera appears to be covered
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Camera is covered if extremely dark
        if avg_brightness < self.brightness_threshold:
            return True
        
        # Also check against baseline (sudden darkness)
        if self.baseline_brightness is not None:
            brightness_drop = self.baseline_brightness - avg_brightness
            if brightness_drop > (self.baseline_brightness * 0.7):  # 70% drop
                return True
        
        return False
    
    def _check_movement(self, frame):
        """
        Check if camera has moved significantly
        
        Args:
            frame: Current video frame
        
        Returns:
            True if camera appears to have moved
        """
        if self.baseline_frame is None:
            return False
        
        difference = self._calculate_frame_difference(frame)
        
        # Significant movement detected
        return difference > self.movement_threshold
    
    def _calculate_frame_difference(self, frame):
        """
        Calculate difference between current frame and baseline
        
        Args:
            frame: Current frame
        
        Returns:
            Difference percentage (0.0 - 1.0)
        """
        if self.baseline_frame is None:
            return 0.0
        
        # Resize both frames to same size
        h, w = frame.shape[:2]
        baseline_resized = cv2.resize(self.baseline_frame, (w, h))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(baseline_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold to get significant changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        difference_pct = changed_pixels / total_pixels
        
        return difference_pct
    
    def reset_baseline(self, frame=None):
        """
        Reset baseline (e.g., after authorized camera adjustment)
        
        Args:
            frame: New baseline frame (optional)
        """
        self.brightness_history.clear()
        self.baseline_brightness = None
        self.baseline_frame = frame.copy() if frame is not None else None
        self.baseline_established = False
        self.is_covered = False
        self.is_moved = False
        
        print("🔄 Tamper baseline reset")
    
    def get_status(self):
        """Get current tamper detection status"""
        return {
            'baseline_established': self.baseline_established,
            'baseline_brightness': self.baseline_brightness,
            'currently_covered': self.is_covered,
            'currently_moved': self.is_moved,
            'tamper_detected': self.is_covered or self.is_moved,
            'total_checks': self.total_checks,
            'cover_detections': self.cover_detections,
            'movement_detections': self.movement_detections,
            'recent_events': self.tamper_events[-5:]  # Last 5 events
        }
    
    def draw_status(self, frame):
        """
        Draw tamper detection status on frame
        
        Args:
            frame: Frame to draw on
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Status text
        status_text = "TAMPER: "
        color = (0, 255, 0)  # Green
        
        if self.is_covered:
            status_text += "COVERED"
            color = (0, 0, 255)  # Red
        elif self.is_moved:
            status_text += "MOVED"
            color = (0, 0, 255)  # Red
        else:
            status_text += "OK"
        
        # Draw status banner
        cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(annotated, status_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw baseline info
        if self.baseline_established:
            info_text = f"Baseline: {self.baseline_brightness:.0f}"
            cv2.putText(annotated, info_text, (w - 200, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return annotated
