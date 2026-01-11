"""Core modules for Lagos traffic detection and analysis"""
from .detector import LagosVehicleDetector
from .tracker import VehicleTracker
from .database import TrafficDatabase
from .camera import VideoCamera, MultiVideoCamera
from .violations import (
    ViolationsDatabase, EvidenceCapture,
    ViolationType, ViolationStatus, ViolationSeverity,
    get_violations_db, get_evidence_capture
)
from .stationary import (
    StationaryVehicleDetector, StationaryVehicle,
    StationaryStatus, create_stationary_violation
)
from .helmet import (
    HelmetDetector, HelmetStatus, RiderDetection,
    create_helmet_violation
)

__all__ = [
    'LagosVehicleDetector',
    'VehicleTracker', 
    'TrafficDatabase',
    'VideoCamera',
    'MultiVideoCamera',
    'ViolationsDatabase',
    'EvidenceCapture',
    'ViolationType',
    'ViolationStatus',
    'ViolationSeverity',
    'get_violations_db',
    'get_evidence_capture',
    'StationaryVehicleDetector',
    'StationaryVehicle',
    'StationaryStatus',
    'create_stationary_violation',
    'HelmetDetector',
    'HelmetStatus',
    'RiderDetection',
    'create_helmet_violation'
]