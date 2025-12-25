"""
FastAPI Dashboard Application
Security & Surveillance Dashboard (Security Mode Only)
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import cv2
import time
import asyncio
from pathlib import Path
from typing import Optional
import json

# Get the dashboard directory path
DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

# Create FastAPI app
app = FastAPI(
    title="Security Surveillance Dashboard",
    description="Real-time security monitoring and surveillance dashboard",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global state to store system references (will be set by launcher)
class AppState:
    def __init__(self):
        self.surveillance_system = None
        self.security_db = None
        self.camera = None
        self.alert_queue = []  # Thread-safe list for alerts
        
app.state.app_state = AppState()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Security Surveillance Dashboard"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Security Surveillance",
        "version": "1.0.0"
    }


# ===========================
# SECURITY SURVEILLANCE APIs
# ===========================

@app.get("/api/security/status")
async def get_security_status():
    """Get current security system status"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {
            "running": False,
            "message": "System not initialized"
        }
    
    system = app_state.surveillance_system
    
    return {
        "running": system.running,
        "frame_count": system.frame_count,
        "uptime_seconds": time.time() - getattr(system, 'start_time', time.time()),
        "fps": getattr(system, 'current_fps', 0),
        "detections_today": getattr(system, 'detections_today', 0),
        "alerts_active": getattr(system, 'active_alerts', 0),
        "recording": getattr(system, 'is_recording', False)
    }


@app.get("/api/security/detections")
async def get_recent_detections(limit: int = 10):
    """Get recent person detections"""
    app_state = app.state.app_state
    
    if not app_state.security_db:
        return {"detections": []}
    
    try:
        detections = app_state.security_db.get_recent_detections(limit=limit)
        return {"detections": detections}
    except Exception as e:
        return {"detections": [], "error": str(e)}


@app.get("/api/security/zones")
async def get_zone_status():
    """Get status of all monitored zones"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {"zones": []}
    
    system = app_state.surveillance_system
    
    if not hasattr(system, 'zone_monitor') or not system.zone_monitor:
        return {"zones": []}
    
    zones = []
    zone_list = system.zone_monitor.zones if isinstance(system.zone_monitor.zones, list) else system.zone_monitor.zones.values()
    for zone in zone_list:
        zones.append({
            "name": zone.name,
            "enabled": zone.enabled,
            "detection_count": zone.detection_count,
            "last_detection": zone.last_detection_time,
            "color": zone.color
        })
    
    return {"zones": zones}


@app.get("/api/security/alerts")
async def get_active_alerts():
    """Get currently active alerts"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {"alerts": []}
    
    system = app_state.surveillance_system
    
    if not hasattr(system, 'alert_manager'):
        return {"alerts": []}
    
    # Get recent alerts from database
    if app_state.security_db:
        try:
            alerts = app_state.security_db.get_recent_events(limit=20, event_type='alert')
            return {"alerts": alerts}
        except Exception as e:
            return {"alerts": [], "error": str(e)}
    
    return {"alerts": []}


@app.get("/api/security/recordings")
async def get_recordings(limit: int = 20):
    """Get list of recorded video clips"""
    recordings_dir = Path("data/recordings")
    
    if not recordings_dir.exists():
        return {"recordings": []}
    
    recordings = []
    for file_path in sorted(recordings_dir.glob("*.mp4"), reverse=True)[:limit]:
        stat = file_path.stat()
        recordings.append({
            "filename": file_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": stat.st_mtime,
            "url": f"/api/security/recordings/{file_path.name}"
        })
    
    return {"recordings": recordings}


@app.get("/api/security/behavior")
async def get_behavior_profile():
    """Get learned behavior patterns"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {"behavior": None}
    
    system = app_state.surveillance_system
    
    if not hasattr(system, 'behavior_learner') or not system.behavior_learner:
        return {"behavior": None}
    
    try:
        profile = system.behavior_learner.get_profile()
        return {"behavior": profile}
    except Exception as e:
        return {"behavior": None, "error": str(e)}


@app.get("/api/security/statistics")
async def get_statistics(days: int = 7):
    """Get surveillance statistics"""
    app_state = app.state.app_state
    
    if not app_state.security_db:
        return {"statistics": {}}
    
    try:
        stats = app_state.security_db.get_statistics(days=days)
        return {"statistics": stats}
    except Exception as e:
        return {"statistics": {}, "error": str(e)}


@app.post("/api/security/start")
async def start_surveillance():
    """Start surveillance system"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {"success": False, "error": "System not initialized"}
    
    try:
        app_state.surveillance_system.start()
        return {"success": True, "message": "Surveillance started"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/security/stop")
async def stop_surveillance():
    """Stop surveillance system"""
    app_state = app.state.app_state
    
    if not app_state.surveillance_system:
        return {"success": False, "error": "System not initialized"}
    
    try:
        app_state.surveillance_system.stop()
        return {"success": True, "message": "Surveillance stopped"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_video_stream():
    """Generator function for MJPEG video stream"""
    app_state = app.state.app_state
    
    while True:
        if app_state.surveillance_system and app_state.surveillance_system.latest_annotated_frame is not None:
            frame = app_state.surveillance_system.latest_annotated_frame
            
            # Resize frame for web streaming (reduce bandwidth)
            frame_resized = cv2.resize(frame, (640, 360))
            
            # Encode frame as JPEG with lower quality for faster streaming
            ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.066)  # ~15 FPS for web streaming


@app.get("/api/security/video_feed")
async def video_feed():
    """Live video stream endpoint (MJPEG)"""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===========================
# WEBSOCKET SUPPORT
# ===========================

from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/security")
async def websocket_security(websocket: WebSocket):
    """WebSocket endpoint for real-time security updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Check for pending alerts and broadcast them FIRST
            app_state = app.state.app_state
            
            if app_state.alert_queue:
                # Send all pending alerts
                while app_state.alert_queue:
                    alert_data = app_state.alert_queue.pop(0)
                    print(f"📢 Broadcasting alert via WebSocket: {alert_data}")
                    await manager.broadcast(alert_data)
            
            # Send status update
            if app_state.surveillance_system:
                system = app_state.surveillance_system
                
                data = {
                    "type": "status",
                    "running": system.running,
                    "frame_count": system.frame_count,
                    "detections": getattr(system, 'detections_today', 0),
                    "timestamp": time.time()
                }
                
                await websocket.send_json(data)
            
            # Short sleep instead of blocking on receive
            await asyncio.sleep(0.5)  # Check every 500ms
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ===========================
# STARTUP/SHUTDOWN
# ===========================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("🚀 Security Surveillance Dashboard starting...")
    print(f"📁 Static files: {STATIC_DIR}")
    print(f"📄 Templates: {TEMPLATES_DIR}")
    print("✅ Dashboard ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down dashboard...")
    
    app_state = app.state.app_state
    if app_state.surveillance_system and app_state.surveillance_system.running:
        app_state.surveillance_system.stop()
