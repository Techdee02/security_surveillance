"""
Zone detection module for security surveillance
Monitors specific areas and triggers alerts based on detections in zones
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time


class Zone:
    """Represents a detection zone with boundaries and properties"""
    
    def __init__(self, 
                 name: str,
                 points: List[Tuple[int, int]],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 enabled: bool = True):
        """
        Initialize zone
        
        Args:
            name: Zone identifier (e.g., "entry", "perimeter")
            points: List of (x, y) points defining polygon
            color: RGB color for visualization
            enabled: Whether zone is active
        """
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color
        self.enabled = enabled
        
        # Stats
        self.detection_count = 0
        self.last_detection_time = None
        
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is inside zone
        
        Args:
            point: (x, y) coordinate
            
        Returns:
            True if point inside zone
        """
        if not self.enabled:
            return False
        
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0
    
    def contains_bbox(self, bbox: Tuple[int, int, int, int], 
                     overlap_threshold: float = 0.1) -> bool:
        """
        Check if bounding box overlaps with zone
        
        Args:
            bbox: (x, y, w, h) bounding box
            overlap_threshold: Minimum overlap ratio (0-1)
            
        Returns:
            True if bbox overlaps zone sufficiently
        """
        if not self.enabled:
            return False
        
        x, y, w, h = bbox
        
        # Check if center is in zone
        center = (x + w // 2, y + h // 2)
        if self.contains_point(center):
            return True
        
        # Check if corners overlap
        corners = [
            (x, y),           # top-left
            (x + w, y),       # top-right
            (x, y + h),       # bottom-left
            (x + w, y + h)    # bottom-right
        ]
        
        overlapping_corners = sum(1 for corner in corners if self.contains_point(corner))
        overlap_ratio = overlapping_corners / 4
        
        return overlap_ratio >= overlap_threshold
    
    def record_detection(self):
        """Record a detection in this zone"""
        self.detection_count += 1
        self.last_detection_time = time.time()
    
    def get_stats(self) -> Dict:
        """Get zone statistics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'detection_count': self.detection_count,
            'last_detection': self.last_detection_time,
            'seconds_since_detection': time.time() - self.last_detection_time 
                if self.last_detection_time else None
        }
    
    def draw(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Draw zone on frame
        
        Args:
            frame: Input frame
            alpha: Transparency (0-1)
            
        Returns:
            Frame with zone overlay
        """
        overlay = frame.copy()
        
        # Fill polygon
        cv2.fillPoly(overlay, [self.points], self.color)
        
        # Draw border
        cv2.polylines(overlay, [self.points], True, self.color, 2)
        
        # Blend with original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add label
        centroid = np.mean(self.points, axis=0).astype(int)
        label = f"{self.name}"
        if not self.enabled:
            label += " (OFF)"
        
        cv2.putText(frame, label, tuple(centroid),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        
        return frame


class ZoneMonitor:
    """Monitors multiple zones and tracks detections"""
    
    def __init__(self, zones: List[Zone]):
        """
        Initialize zone monitor
        
        Args:
            zones: List of Zone objects to monitor
        """
        self.zones = {zone.name: zone for zone in zones}
        self.total_detections = 0
        
    def check_detections(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Check which zones contain detections
        
        Args:
            detections: List of detection dicts with 'bbox' key
            
        Returns:
            Dict mapping zone names to list of detections in that zone
        """
        zone_detections = {name: [] for name in self.zones}
        
        for detection in detections:
            bbox = detection['bbox']
            
            for zone_name, zone in self.zones.items():
                if zone.contains_bbox(bbox):
                    zone_detections[zone_name].append(detection)
                    zone.record_detection()
                    self.total_detections += 1
        
        return zone_detections
    
    def get_zone(self, name: str) -> Optional[Zone]:
        """Get zone by name"""
        return self.zones.get(name)
    
    def enable_zone(self, name: str):
        """Enable a zone"""
        if name in self.zones:
            self.zones[name].enabled = True
    
    def disable_zone(self, name: str):
        """Disable a zone"""
        if name in self.zones:
            self.zones[name].enabled = False
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame"""
        for zone in self.zones.values():
            frame = zone.draw(frame)
        return frame
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'total_zones': len(self.zones),
            'enabled_zones': sum(1 for z in self.zones.values() if z.enabled),
            'total_detections': self.total_detections,
            'zone_stats': {name: zone.get_stats() for name, zone in self.zones.items()}
        }


def create_zones_from_config(config: Dict, frame_shape: Tuple[int, int]) -> List[Zone]:
    """
    Create zones from configuration
    
    Args:
        config: Configuration dict with zone definitions
        frame_shape: (height, width) of video frame
        
    Returns:
        List of Zone objects
    """
    height, width = frame_shape
    zones = []
    
    for zone_config in config.get('zones', []):
        name = zone_config['name']
        
        # Convert normalized coordinates to pixels
        points = []
        for x, y in zone_config['points']:
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            points.append((pixel_x, pixel_y))
        
        color = tuple(zone_config.get('color', [0, 255, 0]))
        enabled = zone_config.get('enabled', True)
        
        zone = Zone(name, points, color, enabled)
        zones.append(zone)
    
    return zones
