"""
Lagos Traffic Analysis Dashboard - FastAPI Backend
Web API for real-time traffic monitoring and analytics
"""

import cv2
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import logging
import time
from typing import List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.detector import LagosVehicleDetector
from modules.camera import VideoCamera, MultiVideoCamera
from modules.database import TrafficDatabase
from modules.tracker import VehicleTracker
from modules.violations import (
    get_violations_db, get_evidence_capture,
    ViolationType, ViolationStatus, ViolationSeverity
)
from modules.stationary import (
    StationaryVehicleDetector, StationaryStatus, create_stationary_violation
)
from modules.helmet import (
    HelmetDetector, HelmetStatus, create_helmet_violation
)
from config import (
    TEST_VIDEOS_DIR, DASHBOARD_HOST, DASHBOARD_PORT,
    STREAM_FPS, DB_PATH, PROCESSING_WIDTH, STREAM_WIDTH,
    SKIP_FRAMES, JPEG_QUALITY, PROJECT_ROOT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Lagos Traffic Analysis Dashboard")

# Setup templates and static files
DASHBOARD_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(DASHBOARD_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")

# Global state
camera = None
detector = None
database = None
tracker = None  # Vehicle tracker for unique counting
stationary_detector = None  # Stationary vehicle detector
helmet_detector = None  # Helmet detection for okada riders
current_frame = None
frame_lock = asyncio.Lock()
processing_active = False

# WebSocket connections
active_connections: List[WebSocket] = []


async def broadcast_update(data: dict):
    """Broadcast update to all connected WebSocket clients"""
    if not active_connections:
        return
    
    message = json.dumps(data)
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


def initialize_system():
    """Initialize detector, camera, database, tracker, stationary detector, and helmet detector"""
    global camera, detector, database, tracker, stationary_detector, helmet_detector
    
    if detector is None:
        logger.info("Initializing vehicle detector...")
        detector = LagosVehicleDetector()
    
    if database is None:
        logger.info("Connecting to database...")
        database = TrafficDatabase(DB_PATH)
    
    if tracker is None:
        logger.info("Initializing vehicle tracker...")
        tracker = VehicleTracker(
            iou_threshold=0.3,      # Match if 30% overlap
            max_frames_missing=30,  # Remove track after 30 frames
            min_hits=3              # Count after 3 consecutive detections
        )
    
    if stationary_detector is None:
        logger.info("Initializing stationary vehicle detector...")
        stationary_detector = StationaryVehicleDetector(
            movement_threshold=15,    # Pixels movement threshold
            warning_threshold=30,     # Seconds before warning
            violation_threshold=120   # Seconds before violation (2 min)
        )
    
    if helmet_detector is None:
        logger.info("Initializing helmet detector...")
        helmet_detector = HelmetDetector(
            helmet_confidence_threshold=0.6,
            head_ratio=0.25
        )
    
    if camera is None:
        # Find test videos
        video_files = list(TEST_VIDEOS_DIR.glob('*.mp4'))
        video_files.extend(TEST_VIDEOS_DIR.glob('*.avi'))
        
        if video_files:
            logger.info(f"Found {len(video_files)} video files")
            if len(video_files) > 1:
                camera = MultiVideoCamera(
                    [str(v) for v in video_files],
                    loop_videos=False,  # Don't loop individual videos
                    loop_sequence=True  # Loop the entire sequence
                )
            else:
                camera = VideoCamera(str(video_files[0]), loop=True)
        else:
            logger.warning("No video files found in test_videos/")


async def process_frames():
    """Background task to process video frames"""
    global current_frame, processing_active
    
    initialize_system()
    
    if camera is None:
        logger.error("No camera available")
        return
    
    processing_active = True
    logger.info("Starting frame processing...")
    
    session_id = database.create_session("Dashboard Video Feed")
    frame_count = 0
    last_detections = []  # Keep last detections to draw on skipped frames
    
    # Use tracker's unique counts instead of per-frame counts
    last_broadcast_counts = {}
    
    # Performance tracking
    last_process_time = 0
    
    try:
        while processing_active:
            ret, frame = camera.read_frame()
            
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Resize frame for faster processing
            h, w = frame.shape[:2]
            if w > PROCESSING_WIDTH:
                scale = PROCESSING_WIDTH / w
                process_frame = cv2.resize(frame, (PROCESSING_WIDTH, int(h * scale)))
            else:
                process_frame = frame
            
            # Only run detection every N frames
            if frame_count % (SKIP_FRAMES + 1) == 0:
                # Run detection on resized frame
                detections = detector.detect(process_frame)
                
                # Scale bounding boxes back to original size if resized
                if w > PROCESSING_WIDTH:
                    scale_back = w / PROCESSING_WIDTH
                    for det in detections:
                        det['bbox'] = [int(c * scale_back) for c in det['bbox']]
                
                if detections:
                    # Update tracker - this handles unique counting
                    tracked_detections, new_vehicles = tracker.update(detections, frame_count)
                    last_detections = tracked_detections
                    
                    # Update stationary vehicle detector (pass None to use time.time())
                    warnings, new_violations, stationary_vehicles = stationary_detector.update(
                        tracked_detections, None
                    )
                    
                    # Log stationary vehicle violations
                    violations_db = get_violations_db()
                    evidence_capture = get_evidence_capture()
                    
                    for stat_vehicle in new_violations:
                        logger.warning(f"STATIONARY VIOLATION: {stat_vehicle.vehicle_type} (track {stat_vehicle.track_id}) "
                                      f"stationary for {stat_vehicle.total_stationary_time:.0f}s")
                        
                        # Create violation record
                        violation_data = create_stationary_violation(stat_vehicle, frame_count)
                        
                        # Capture evidence snapshot
                        snapshot_path = evidence_capture.capture_snapshot(
                            frame,
                            violation_data['id'],
                            stat_vehicle.bbox,
                            draw_bbox=True
                        )
                        
                        # Add evidence path to violation data
                        if snapshot_path:
                            violation_data['evidence_path'] = snapshot_path
                        
                        # Store violation
                        violations_db.add_violation(violation_data)
                    
                    # Log warnings (but don't create violations yet)
                    for stat_vehicle in warnings:
                        logger.info(f"Stationary warning: {stat_vehicle.vehicle_type} (track {stat_vehicle.track_id}) "
                                   f"stationary for {stat_vehicle.total_stationary_time:.0f}s")
                    
                    # Helmet detection for okada riders
                    # Note: We need person detections for full helmet analysis
                    # For now, detect based on tracked okadas
                    riders = helmet_detector.detect_riders(
                        frame,
                        tracked_detections,
                        person_detections=None,  # Would need person detection
                        frame_number=frame_count
                    )
                    
                    # Get helmet violations
                    helmet_violations = helmet_detector.get_violations(riders)
                    for rider in helmet_violations:
                        logger.warning(f"HELMET VIOLATION: Okada rider (track {rider.track_id}) without helmet")
                        
                        # Create violation record
                        violation_data = create_helmet_violation(rider)
                        
                        # Capture evidence
                        snapshot_path = evidence_capture.capture_snapshot(
                            frame,
                            violation_data['id'],
                            rider.bbox,
                            draw_bbox=True
                        )
                        
                        if snapshot_path:
                            violation_data['evidence_path'] = snapshot_path
                        
                        # Store violation
                        violations_db.add_violation(violation_data)
                    
                    # Only log NEW unique vehicles to database
                    if new_vehicles:
                        database.log_detections_batch(new_vehicles, frame_count)
                        logger.info(f"Frame {frame_count}: Logged {len(new_vehicles)} new unique vehicles")
                    
                    # Get unique counts from tracker
                    unique_counts = tracker.get_unique_counts()
                    
                    # Log detection progress every 50 frames
                    if frame_count % 50 == 0:
                        logger.info(f"Frame {frame_count}: {len(detections)} detections, {len(tracked_detections)} tracked, Unique counts: {unique_counts}")
                    
                    # Only broadcast if counts changed
                    if unique_counts != last_broadcast_counts:
                        last_broadcast_counts = unique_counts.copy()
                        
                        # Get stationary vehicle summary
                        stationary_summary = []
                        for v in stationary_vehicles:
                            if v.status in [StationaryStatus.WARNING, StationaryStatus.VIOLATION]:
                                stationary_summary.append({
                                    'track_id': v.track_id,
                                    'type': v.vehicle_type,
                                    'duration': round(v.stationary_duration, 0),
                                    'status': v.status.value
                                })
                        
                        # Broadcast update via WebSocket
                        await broadcast_update({
                            'type': 'detection',
                            'frame': frame_count,
                            'active_tracks': tracker.get_active_tracks(),
                            'new_vehicles': len(new_vehicles),
                            'counts': unique_counts,
                            'total_unique': sum(unique_counts.values()),
                            'stationary_vehicles': stationary_summary,
                            'stationary_count': len(stationary_summary)
                        })
                else:
                    # No detections - still update tracker to age tracks
                    tracker.update([], frame_count)
            
            # Always draw the last known detections on every frame
            if last_detections:
                frame = detector.draw_detections(frame, last_detections)
            
            # Resize frame for streaming (reduce bandwidth)
            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                scale = STREAM_WIDTH / w
                stream_frame = cv2.resize(frame, (STREAM_WIDTH, int(h * scale)))
            else:
                stream_frame = frame
            
            # Store current frame for streaming
            async with frame_lock:
                current_frame = stream_frame
            
            # Minimal sleep to allow other tasks
            await asyncio.sleep(0.01)
    
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
    finally:
        processing_active = False
        if session_id:
            database.close_session(session_id)
        logger.info("Frame processing stopped")


@app.on_event("startup")
async def startup_event():
    """Start background processing on startup"""
    logger.info("Starting Lagos Traffic Dashboard...")
    asyncio.create_task(process_frames())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global processing_active, camera, database
    
    logger.info("Shutting down...")
    processing_active = False
    
    if camera:
        camera.release()
    
    if database:
        database.close()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get system status"""
    initialize_system()
    
    camera_info = camera.get_info() if camera else {}
    
    return {
        "status": "running" if processing_active else "stopped",
        "detector_loaded": detector is not None,
        "camera_active": camera is not None,
        "database_connected": database is not None,
        "camera_info": camera_info
    }


@app.get("/api/vehicle_counts")
async def get_vehicle_counts(time_window: int = 3600):
    """
    Get unique vehicle counts (each vehicle counted once)
    
    Args:
        time_window: Time window in seconds (default: 3600 = 1 hour)
    """
    initialize_system()
    
    if tracker is None:
        return {"error": "Tracker not initialized"}
    
    # Get unique counts from tracker (real-time session)
    unique_counts = tracker.get_unique_counts()
    tracker_stats = tracker.get_stats()
    
    # Get historical counts from database
    db_counts = database.get_vehicle_counts(time_window) if database else {}
    total_db_counts = database.get_total_counts() if database else {}
    
    return {
        "time_window": time_window,
        "unique_counts": unique_counts,  # Current session unique counts
        "active_tracks": tracker_stats['active_tracks'],
        "total_unique_vehicles": tracker_stats['total_unique_vehicles'],
        "counts": unique_counts,  # For backward compatibility
        "total_counts": total_db_counts  # Historical from DB
    }


@app.get("/api/recent_detections")
async def get_recent_detections(limit: int = 50):
    """
    Get recent vehicle detections
    
    Args:
        limit: Maximum number of detections to return
    """
    initialize_system()
    
    if database is None:
        return {"error": "Database not initialized"}
    
    detections = database.get_recent_detections(limit)
    
    return {
        "count": len(detections),
        "detections": detections
    }


@app.get("/api/stats")
async def get_stats():
    """Get tracker and database statistics"""
    initialize_system()
    
    stats = {}
    
    if tracker:
        stats['tracker'] = tracker.get_stats()
    
    if database:
        stats['database'] = database.get_stats()
    
    return stats


@app.get("/api/hourly_counts")
async def get_hourly_counts(hours: int = 24):
    """
    Get vehicle counts grouped by hour
    
    Args:
        hours: Number of hours to look back
    """
    initialize_system()
    
    if database is None:
        return {"error": "Database not initialized"}
    
    hourly_data = database.get_counts_by_hour(hours)
    
    return {
        "hours": hours,
        "data": hourly_data
    }


@app.get("/stream")
async def video_stream():
    """Stream video feed with detections"""
    
    def generate_frames():
        """Generate MJPEG stream"""
        global current_frame
        
        while True:
            if current_frame is not None:
                # Encode frame as JPEG (lower quality = faster)
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1.0 / max(STREAM_FPS, 15))  # Cap at 15fps for smoother streaming
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_connections)}")


def run_dashboard(host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT):
    """Run the dashboard server"""
    import uvicorn
    
    logger.info(f"Starting dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# =============================================================================
# VIOLATION REVIEW API ENDPOINTS
# =============================================================================

@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    """Violation review page"""
    return templates.TemplateResponse("review.html", {"request": request})


@app.get("/api/violations/pending")
async def get_pending_violations(
    limit: int = 50,
    violation_type: str = None,
    severity: str = None
):
    """Get pending violations for review"""
    violations_db = get_violations_db()
    violations = violations_db.get_pending_violations(limit, violation_type, severity)
    
    return {
        "count": len(violations),
        "violations": violations
    }


@app.get("/api/violations/stats")
async def get_violation_stats():
    """Get violation statistics"""
    violations_db = get_violations_db()
    return violations_db.get_violation_stats()


@app.get("/api/violations/{violation_id}")
async def get_violation(violation_id: str):
    """Get violation details by ID"""
    violations_db = get_violations_db()
    violation = violations_db.get_violation(violation_id)
    
    if not violation:
        return {"error": "Violation not found"}
    
    # Get review history
    review_log = violations_db.get_review_log(violation_id)
    violation['review_log'] = review_log
    
    return violation


@app.post("/api/violations/{violation_id}/approve")
async def approve_violation(violation_id: str, reviewer: str = "operator", notes: str = None):
    """Approve a violation"""
    violations_db = get_violations_db()
    success = violations_db.approve_violation(violation_id, reviewer, notes)
    
    if success:
        return {"status": "success", "message": f"Violation {violation_id} approved"}
    return {"status": "error", "message": "Failed to approve violation"}


@app.post("/api/violations/{violation_id}/reject")
async def reject_violation(violation_id: str, reviewer: str = "operator", notes: str = None):
    """Reject a violation"""
    violations_db = get_violations_db()
    success = violations_db.reject_violation(violation_id, reviewer, notes)
    
    if success:
        return {"status": "success", "message": f"Violation {violation_id} rejected"}
    return {"status": "error", "message": "Failed to reject violation"}


@app.post("/api/violations/{violation_id}/escalate")
async def escalate_violation(violation_id: str, reviewer: str = "operator", notes: str = None):
    """Escalate a violation for supervisor review"""
    violations_db = get_violations_db()
    success = violations_db.escalate_violation(violation_id, reviewer, notes)
    
    if success:
        return {"status": "success", "message": f"Violation {violation_id} escalated"}
    return {"status": "error", "message": "Failed to escalate violation"}


@app.get("/api/violations/by_status/{status}")
async def get_violations_by_status(status: str, limit: int = 100):
    """Get violations by status"""
    violations_db = get_violations_db()
    
    try:
        status_enum = ViolationStatus(status)
    except ValueError:
        return {"error": f"Invalid status: {status}"}
    
    violations = violations_db.get_violations_by_status(status_enum, limit)
    
    return {
        "status": status,
        "count": len(violations),
        "violations": violations
    }


@app.get("/api/violation_types")
async def get_violation_types():
    """Get list of violation types with their severities and fines"""
    from modules.violations import VIOLATION_SEVERITY, VIOLATION_FINES
    
    types = []
    for vtype in ViolationType:
        types.append({
            "type": vtype.value,
            "name": vtype.value.replace("_", " ").title(),
            "severity": VIOLATION_SEVERITY.get(vtype, ViolationSeverity.MEDIUM).value,
            "fine": VIOLATION_FINES.get(vtype, 0)
        })
    
    return {"violation_types": types}


# =============================================================================
# STATIONARY VEHICLE DETECTION API ENDPOINTS
# =============================================================================

@app.get("/api/stationary/vehicles")
async def get_stationary_vehicles():
    """Get currently stationary vehicles"""
    initialize_system()
    
    if stationary_detector is None:
        return {"error": "Stationary detector not initialized", "vehicles": []}
    
    stationary = stationary_detector.get_stationary_vehicles()
    
    return {
        "count": len(stationary),
        "vehicles": [
            {
                "track_id": v.track_id,
                "vehicle_type": v.vehicle_type,
                "bbox": v.bbox,
                "stationary_duration": round(v.stationary_duration, 1),
                "status": v.status.value,
                "first_seen": v.first_seen,
                "stationary_since": v.stationary_since,
                "violation_reported": v.violation_reported
            }
            for v in stationary
        ]
    }


@app.get("/api/stationary/stats")
async def get_stationary_stats():
    """Get stationary vehicle detection statistics"""
    initialize_system()
    
    if stationary_detector is None:
        return {"error": "Stationary detector not initialized"}
    
    stats = stationary_detector.get_stats()
    
    return {
        "total_tracked": stats['tracked_vehicles'],
        "currently_stationary": stats['stationary_count'],
        "warnings_issued": stats['total_warnings_issued'],
        "violations_detected": stats['total_violations_issued'],
        "by_status": {
            "warning": stats['warning_count'],
            "violation": stats['violation_count']
        }
    }


@app.get("/api/stationary/config")
async def get_stationary_config():
    """Get stationary detection configuration"""
    initialize_system()
    
    if stationary_detector is None:
        return {"error": "Stationary detector not initialized"}
    
    return {
        "movement_threshold_px": stationary_detector.movement_threshold,
        "warning_threshold_sec": stationary_detector.warning_threshold,
        "violation_threshold_sec": stationary_detector.violation_threshold
    }


# =============================================================================
# HELMET DETECTION API ENDPOINTS
# =============================================================================

@app.get("/api/helmet/stats")
async def get_helmet_stats():
    """Get helmet detection statistics"""
    initialize_system()
    
    if helmet_detector is None:
        return {"error": "Helmet detector not initialized"}
    
    return helmet_detector.get_stats()


@app.get("/api/helmet/config")
async def get_helmet_config():
    """Get helmet detection configuration"""
    initialize_system()
    
    if helmet_detector is None:
        return {"error": "Helmet detector not initialized"}
    
    return {
        "confidence_threshold": helmet_detector.helmet_confidence_threshold,
        "head_ratio": helmet_detector.head_ratio,
        "min_helmet_area_ratio": helmet_detector.min_helmet_area_ratio,
        "cooldown_period_sec": helmet_detector.cooldown_period
    }


# Serve evidence files
app.mount("/evidence", StaticFiles(directory=str(PROJECT_ROOT / "evidence")), name="evidence")


if __name__ == "__main__":
    run_dashboard()
