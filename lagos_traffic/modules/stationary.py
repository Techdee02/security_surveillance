"""
Stationary Vehicle Detection Module
Detects vehicles that remain stationary in traffic lanes for extended periods.
Flags potential obstructions and generates violations.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StationaryStatus(Enum):
    """Status of a stationary vehicle detection"""
    MOVING = "moving"
    SLOWING = "slowing"
    WARNING = "warning"  # > warning_threshold seconds
    VIOLATION = "violation"  # > violation_threshold seconds


@dataclass
class StationaryVehicle:
    """Information about a potentially stationary vehicle"""
    track_id: int
    vehicle_type: str
    first_position: Tuple[float, float]  # centroid (x, y)
    current_position: Tuple[float, float]
    first_seen: float  # timestamp
    last_seen: float
    last_bbox: List[int]
    stationary_since: Optional[float] = None
    total_stationary_time: float = 0.0
    status: StationaryStatus = StationaryStatus.MOVING
    violation_reported: bool = False
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    
    @property
    def stationary_duration(self) -> float:
        """Get current stationary duration in seconds"""
        if self.stationary_since is not None:
            return time.time() - self.stationary_since
        return self.total_stationary_time
    
    @property
    def bbox(self) -> List[int]:
        """Alias for last_bbox for compatibility"""
        return self.last_bbox


class StationaryVehicleDetector:
    """
    Detects vehicles that have been stationary for extended periods.
    Uses position tracking to identify vehicles blocking lanes.
    """
    
    def __init__(
        self,
        movement_threshold: float = 15.0,  # pixels - movement less than this is stationary
        warning_threshold: float = 30.0,    # seconds - time before warning
        violation_threshold: float = 120.0,  # seconds - time before violation (2 minutes)
        position_history_size: int = 30,     # frames of position history to keep
        fps: float = 10.0                    # frames per second for time calculations
    ):
        """
        Initialize stationary vehicle detector.
        
        Args:
            movement_threshold: Maximum pixel movement to be considered stationary
            warning_threshold: Seconds of no movement before warning
            violation_threshold: Seconds of no movement before violation
            position_history_size: Number of positions to keep in history
            fps: Frame rate for time calculations
        """
        self.movement_threshold = movement_threshold
        self.warning_threshold = warning_threshold
        self.violation_threshold = violation_threshold
        self.position_history_size = position_history_size
        self.fps = fps
        
        # Track stationary status for each vehicle
        self.vehicles: Dict[int, StationaryVehicle] = {}
        
        # Statistics
        self.total_warnings = 0
        self.total_violations = 0
        
        logger.info(
            f"StationaryVehicleDetector initialized: "
            f"movement_threshold={movement_threshold}px, "
            f"warning={warning_threshold}s, violation={violation_threshold}s"
        )
    
    def _calculate_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
    
    def update(
        self,
        tracked_vehicles: List[Dict],
        timestamp: Optional[float] = None
    ) -> Tuple[List[StationaryVehicle], List[StationaryVehicle], List[StationaryVehicle]]:
        """
        Update stationary detection with new tracked vehicles.
        
        Args:
            tracked_vehicles: List of tracked vehicle dicts with track_id, bbox, vehicle_type
            timestamp: Current timestamp (uses time.time() if not provided)
            
        Returns:
            Tuple of:
            - warnings: Vehicles that have been stationary > warning_threshold
            - new_violations: Vehicles that just crossed violation_threshold (not reported before)
            - all_stationary: All currently stationary vehicles
        """
        if timestamp is None:
            timestamp = time.time()
        
        current_track_ids = set()
        warnings = []
        new_violations = []
        all_stationary = []
        
        for vehicle in tracked_vehicles:
            track_id = vehicle.get('track_id')
            if track_id is None:
                continue
            
            current_track_ids.add(track_id)
            bbox = vehicle['bbox']
            vehicle_type = vehicle['vehicle_type']
            centroid = self._calculate_centroid(bbox)
            
            if track_id not in self.vehicles:
                # New vehicle - start tracking
                self.vehicles[track_id] = StationaryVehicle(
                    track_id=track_id,
                    vehicle_type=vehicle_type,
                    first_position=centroid,
                    current_position=centroid,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    last_bbox=bbox,
                    position_history=[(centroid[0], centroid[1], timestamp)]
                )
            else:
                # Update existing vehicle
                sv = self.vehicles[track_id]
                sv.last_seen = timestamp
                sv.last_bbox = bbox
                
                # Add to position history
                sv.position_history.append((centroid[0], centroid[1], timestamp))
                if len(sv.position_history) > self.position_history_size:
                    sv.position_history.pop(0)
                
                # Check movement
                distance = self._calculate_distance(sv.current_position, centroid)
                
                if distance < self.movement_threshold:
                    # Vehicle is stationary
                    if sv.stationary_since is None:
                        sv.stationary_since = timestamp
                    
                    stationary_duration = timestamp - sv.stationary_since
                    sv.total_stationary_time = stationary_duration
                    
                    # Update status based on duration
                    if stationary_duration >= self.violation_threshold:
                        sv.status = StationaryStatus.VIOLATION
                        if not sv.violation_reported:
                            sv.violation_reported = True
                            self.total_violations += 1
                            new_violations.append(sv)
                            logger.warning(
                                f"VIOLATION: {vehicle_type} (track {track_id}) "
                                f"stationary for {stationary_duration:.1f}s"
                            )
                    elif stationary_duration >= self.warning_threshold:
                        if sv.status != StationaryStatus.WARNING:
                            sv.status = StationaryStatus.WARNING
                            self.total_warnings += 1
                            logger.info(
                                f"WARNING: {vehicle_type} (track {track_id}) "
                                f"stationary for {stationary_duration:.1f}s"
                            )
                        warnings.append(sv)
                    else:
                        sv.status = StationaryStatus.SLOWING
                else:
                    # Vehicle is moving - reset stationary tracking
                    sv.current_position = centroid
                    sv.stationary_since = None
                    sv.status = StationaryStatus.MOVING
                    sv.violation_reported = False  # Reset so it can be reported again if stops
        
        # Clean up vehicles no longer being tracked
        missing_ids = set(self.vehicles.keys()) - current_track_ids
        for track_id in missing_ids:
            if timestamp - self.vehicles[track_id].last_seen > 5.0:  # 5 second grace period
                del self.vehicles[track_id]
        
        # Collect all stationary vehicles
        for sv in self.vehicles.values():
            if sv.status in [StationaryStatus.WARNING, StationaryStatus.VIOLATION, StationaryStatus.SLOWING]:
                if sv.stationary_since is not None:
                    all_stationary.append(sv)
        
        return warnings, new_violations, all_stationary
    
    def get_stationary_vehicles(
        self,
        min_duration: Optional[float] = None
    ) -> List[StationaryVehicle]:
        """
        Get all currently stationary vehicles.
        
        Args:
            min_duration: Minimum stationary duration in seconds (optional filter)
            
        Returns:
            List of stationary vehicles
        """
        result = []
        for sv in self.vehicles.values():
            if sv.stationary_since is not None:
                duration = time.time() - sv.stationary_since
                if min_duration is None or duration >= min_duration:
                    result.append(sv)
        return result
    
    def get_vehicle_status(self, track_id: int) -> Optional[StationaryVehicle]:
        """Get status of a specific vehicle by track ID"""
        return self.vehicles.get(track_id)
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        stationary_count = sum(
            1 for sv in self.vehicles.values() 
            if sv.status in [StationaryStatus.WARNING, StationaryStatus.VIOLATION]
        )
        
        return {
            'tracked_vehicles': len(self.vehicles),
            'stationary_count': stationary_count,
            'warning_count': sum(1 for sv in self.vehicles.values() if sv.status == StationaryStatus.WARNING),
            'violation_count': sum(1 for sv in self.vehicles.values() if sv.status == StationaryStatus.VIOLATION),
            'total_warnings_issued': self.total_warnings,
            'total_violations_issued': self.total_violations,
            'warning_threshold': self.warning_threshold,
            'violation_threshold': self.violation_threshold
        }
    
    def reset(self):
        """Reset all tracking state"""
        self.vehicles = {}
        self.total_warnings = 0
        self.total_violations = 0
        logger.info("StationaryVehicleDetector reset")


def create_stationary_violation(
    stationary_vehicle: StationaryVehicle,
    frame: Optional[any] = None,
    zone_id: Optional[str] = None
) -> Dict:
    """
    Create a violation record from a stationary vehicle detection.
    
    Args:
        stationary_vehicle: The StationaryVehicle object
        frame: Current video frame (for evidence capture)
        zone_id: Optional zone identifier
        
    Returns:
        Violation dictionary compatible with ViolationsDatabase
    """
    import uuid
    from datetime import datetime
    
    sv = stationary_vehicle
    
    # Determine severity based on duration
    if sv.total_stationary_time >= 300:  # 5+ minutes
        severity = "critical"
    elif sv.total_stationary_time >= 120:  # 2+ minutes
        severity = "high"
    elif sv.total_stationary_time >= 60:  # 1+ minute
        severity = "medium"
    else:
        severity = "low"
    
    # Generate violation ID
    violation_id = f"STA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    
    violation = {
        'id': violation_id,
        'violation_type': 'stationary_obstruction',
        'timestamp': datetime.now().isoformat(),
        'vehicle_type': sv.vehicle_type,
        'track_id': sv.track_id,
        'confidence': 0.95,  # High confidence for stationary detection
        'severity': severity,
        'bbox': sv.last_bbox,
        'zone_id': zone_id,
        'metadata': {
            'stationary_duration': sv.total_stationary_time,
            'first_position': sv.first_position,
            'current_position': sv.current_position,
            'first_seen': sv.first_seen,
            'status': sv.status.value
        }
    }
    
    return violation
