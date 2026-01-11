"""
Helmet Detection Module for Okada (Motorcycle) Riders
Detects whether motorcycle riders are wearing helmets.
Uses person detection + head region analysis.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelmetStatus(Enum):
    """Helmet detection status"""
    WEARING_HELMET = "wearing_helmet"
    NO_HELMET = "no_helmet"
    UNCERTAIN = "uncertain"


@dataclass
class RiderDetection:
    """Detection result for a motorcycle rider"""
    rider_id: str
    track_id: Optional[int]  # Associated vehicle track ID
    bbox: List[int]  # Person bounding box [x1, y1, x2, y2]
    head_bbox: Optional[List[int]]  # Head region bbox
    helmet_status: HelmetStatus
    confidence: float
    motorcycle_bbox: Optional[List[int]] = None
    timestamp: float = 0.0
    frame_number: int = 0
    
    @property
    def is_violation(self) -> bool:
        """Check if this is a helmet violation"""
        return self.helmet_status == HelmetStatus.NO_HELMET


class HelmetDetector:
    """
    Detects helmet usage on motorcycle riders.
    
    Detection Strategy:
    1. Identify motorcycles (okadas) from vehicle detections
    2. Find persons overlapping with motorcycles (riders)
    3. Analyze head region for helmet presence
    4. Use color and shape analysis to detect helmets
    """
    
    def __init__(
        self,
        helmet_confidence_threshold: float = 0.6,
        head_ratio: float = 0.25,  # Head is top 25% of person bbox
        min_helmet_area_ratio: float = 0.3,  # Minimum helmet coverage
        helmet_colors_hsv: List[Tuple] = None  # Common helmet colors in HSV
    ):
        """
        Initialize helmet detector.
        
        Args:
            helmet_confidence_threshold: Confidence threshold for helmet detection
            head_ratio: Portion of person bbox considered as head region
            min_helmet_area_ratio: Minimum ratio of head area that should be helmet
            helmet_colors_hsv: List of HSV color ranges for common helmets
        """
        self.helmet_confidence_threshold = helmet_confidence_threshold
        self.head_ratio = head_ratio
        self.min_helmet_area_ratio = min_helmet_area_ratio
        
        # Common helmet colors in HSV (lower, upper bounds)
        # Black, white, red, blue, yellow helmets
        self.helmet_colors_hsv = helmet_colors_hsv or [
            # Black helmets (low saturation, low value)
            ((0, 0, 0), (180, 50, 50)),
            # White helmets (low saturation, high value)
            ((0, 0, 200), (180, 30, 255)),
            # Red helmets
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255)),
            # Blue helmets
            ((100, 100, 100), (130, 255, 255)),
            # Yellow helmets
            ((20, 100, 100), (35, 255, 255)),
        ]
        
        # Track violations to avoid duplicate alerts
        self.violation_cooldown: Dict[int, float] = {}  # track_id -> last_violation_time
        self.cooldown_period = 30.0  # seconds between violations for same rider
        
        # Statistics
        self.total_riders_checked = 0
        self.helmets_detected = 0
        self.no_helmet_violations = 0
        
        logger.info(f"HelmetDetector initialized: confidence={helmet_confidence_threshold}, "
                   f"head_ratio={head_ratio}")
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_head_region(self, person_bbox: List[int]) -> List[int]:
        """Extract head region from person bounding box"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        head_height = int(height * self.head_ratio)
        
        return [x1, y1, x2, y1 + head_height]
    
    def _analyze_head_region(self, frame: np.ndarray, head_bbox: List[int]) -> Tuple[HelmetStatus, float]:
        """
        Analyze head region to detect helmet presence.
        
        Uses multiple heuristics:
        1. Color analysis - helmets often have distinct colors
        2. Edge detection - helmets have smooth, curved edges
        3. Texture analysis - helmets are typically uniform
        
        Args:
            frame: OpenCV image (BGR)
            head_bbox: Head region bounding box
            
        Returns:
            Tuple of (HelmetStatus, confidence)
        """
        x1, y1, x2, y2 = head_bbox
        
        # Ensure valid bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return HelmetStatus.UNCERTAIN, 0.0
        
        # Extract head region
        head_roi = frame[y1:y2, x1:x2]
        if head_roi.size == 0:
            return HelmetStatus.UNCERTAIN, 0.0
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        
        # Check for helmet colors
        helmet_pixel_count = 0
        total_pixels = head_roi.shape[0] * head_roi.shape[1]
        
        for lower, upper in self.helmet_colors_hsv:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            helmet_pixel_count += cv2.countNonZero(mask)
        
        # Avoid counting same pixel multiple times
        helmet_ratio = min(helmet_pixel_count / total_pixels, 1.0)
        
        # Edge detection for helmet shape
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / total_pixels
        
        # Helmets typically have:
        # - High color uniformity in certain regions
        # - Moderate edge density (curved surfaces)
        # - Distinct separation from face/hair
        
        # Calculate helmet confidence
        helmet_confidence = 0.0
        
        # Color score (0.4 weight)
        if helmet_ratio > 0.5:
            helmet_confidence += 0.4 * min(helmet_ratio, 1.0)
        
        # Edge score (0.3 weight) - moderate edges suggest helmet
        if 0.05 < edge_density < 0.25:
            helmet_confidence += 0.3 * (1.0 - abs(edge_density - 0.15) / 0.15)
        
        # Top region analysis (0.3 weight)
        # Upper half of head should be more uniform (helmet surface)
        upper_half = head_roi[:head_roi.shape[0]//2, :]
        if upper_half.size > 0:
            upper_std = np.std(cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY))
            # Lower std dev suggests uniform helmet surface
            if upper_std < 40:
                helmet_confidence += 0.3 * (1.0 - upper_std / 40)
        
        # Determine status
        if helmet_confidence >= self.helmet_confidence_threshold:
            return HelmetStatus.WEARING_HELMET, helmet_confidence
        elif helmet_confidence < 0.3:
            return HelmetStatus.NO_HELMET, 1.0 - helmet_confidence
        else:
            return HelmetStatus.UNCERTAIN, helmet_confidence
    
    def detect_riders(
        self,
        frame: np.ndarray,
        vehicle_detections: List[Dict],
        person_detections: List[Dict] = None,
        frame_number: int = 0
    ) -> List[RiderDetection]:
        """
        Detect motorcycle riders and their helmet status.
        
        Args:
            frame: OpenCV image (BGR)
            vehicle_detections: List of vehicle detections with 'vehicle_type', 'bbox', 'track_id'
            person_detections: Optional list of person detections with 'bbox'
            frame_number: Current frame number
            
        Returns:
            List of RiderDetection objects
        """
        timestamp = time.time()
        riders = []
        
        # Find okadas (motorcycles)
        okadas = [v for v in vehicle_detections if v.get('vehicle_type') == 'okada']
        
        if not okadas:
            return riders
        
        # If no person detections provided, we can only flag the motorcycle
        if not person_detections:
            for okada in okadas:
                # Create detection without person bbox
                rider = RiderDetection(
                    rider_id=str(uuid.uuid4())[:8],
                    track_id=okada.get('track_id'),
                    bbox=okada['bbox'],
                    head_bbox=None,
                    helmet_status=HelmetStatus.UNCERTAIN,
                    confidence=0.5,
                    motorcycle_bbox=okada['bbox'],
                    timestamp=timestamp,
                    frame_number=frame_number
                )
                riders.append(rider)
            return riders
        
        # Match persons to motorcycles
        for okada in okadas:
            okada_bbox = okada['bbox']
            track_id = okada.get('track_id')
            
            # Find persons overlapping with this motorcycle
            for person in person_detections:
                person_bbox = person['bbox']
                
                # Check overlap
                iou = self._calculate_iou(okada_bbox, person_bbox)
                
                # Also check if person center is within/near motorcycle bbox
                person_cx = (person_bbox[0] + person_bbox[2]) / 2
                person_cy = (person_bbox[1] + person_bbox[3]) / 2
                
                in_motorcycle = (
                    okada_bbox[0] - 50 <= person_cx <= okada_bbox[2] + 50 and
                    okada_bbox[1] - 100 <= person_cy <= okada_bbox[3] + 50
                )
                
                if iou > 0.1 or in_motorcycle:
                    # This person is likely a rider
                    self.total_riders_checked += 1
                    
                    # Get head region
                    head_bbox = self._get_head_region(person_bbox)
                    
                    # Analyze for helmet
                    helmet_status, confidence = self._analyze_head_region(frame, head_bbox)
                    
                    # Update statistics
                    if helmet_status == HelmetStatus.WEARING_HELMET:
                        self.helmets_detected += 1
                    elif helmet_status == HelmetStatus.NO_HELMET:
                        self.no_helmet_violations += 1
                    
                    rider = RiderDetection(
                        rider_id=str(uuid.uuid4())[:8],
                        track_id=track_id,
                        bbox=person_bbox,
                        head_bbox=head_bbox,
                        helmet_status=helmet_status,
                        confidence=confidence,
                        motorcycle_bbox=okada_bbox,
                        timestamp=timestamp,
                        frame_number=frame_number
                    )
                    riders.append(rider)
        
        return riders
    
    def get_violations(
        self,
        riders: List[RiderDetection],
        check_cooldown: bool = True
    ) -> List[RiderDetection]:
        """
        Get helmet violations from rider detections.
        
        Args:
            riders: List of RiderDetection objects
            check_cooldown: Whether to check violation cooldown
            
        Returns:
            List of violations (riders without helmets)
        """
        violations = []
        current_time = time.time()
        
        for rider in riders:
            if rider.helmet_status != HelmetStatus.NO_HELMET:
                continue
            
            # Check cooldown
            if check_cooldown and rider.track_id is not None:
                last_violation = self.violation_cooldown.get(rider.track_id, 0)
                if current_time - last_violation < self.cooldown_period:
                    continue
                
                # Update cooldown
                self.violation_cooldown[rider.track_id] = current_time
            
            violations.append(rider)
        
        return violations
    
    def get_stats(self) -> Dict:
        """Get helmet detection statistics"""
        return {
            'total_riders_checked': self.total_riders_checked,
            'helmets_detected': self.helmets_detected,
            'no_helmet_violations': self.no_helmet_violations,
            'helmet_rate': (
                self.helmets_detected / self.total_riders_checked 
                if self.total_riders_checked > 0 else 0.0
            ),
            'violation_rate': (
                self.no_helmet_violations / self.total_riders_checked
                if self.total_riders_checked > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.total_riders_checked = 0
        self.helmets_detected = 0
        self.no_helmet_violations = 0
        self.violation_cooldown.clear()
        logger.info("HelmetDetector statistics reset")


def create_helmet_violation(
    rider: RiderDetection,
    camera_id: str = "camera_1",
    location: str = "Lagos Traffic Camera"
) -> Dict:
    """
    Create a violation record from a rider detection.
    
    Args:
        rider: RiderDetection object
        camera_id: Camera identifier
        location: Location description
        
    Returns:
        Violation dict compatible with ViolationsDatabase
    """
    from modules.violations import ViolationType, ViolationSeverity
    
    return {
        'id': f"HELMET-{rider.rider_id}-{int(rider.timestamp)}",
        'violation_type': ViolationType.NO_HELMET.value,
        'severity': ViolationSeverity.HIGH.value,
        'camera_id': camera_id,
        'location': location,
        'timestamp': rider.timestamp,
        'frame_number': rider.frame_number,
        'track_id': rider.track_id,
        'vehicle_type': 'okada',
        'bbox': rider.bbox,
        'confidence': rider.confidence,
        'details': {
            'motorcycle_bbox': rider.motorcycle_bbox,
            'head_bbox': rider.head_bbox,
            'helmet_status': rider.helmet_status.value
        }
    }
