# Security Surveillance System

**AI-powered security monitoring with real-time person detection and zone-based alerting**  
Complete surveillance system with web dashboard, video recording, and intelligent alert management.

---

## 🌟 Features

- 🎯 **Real-time Person Detection** - YOLOv8n model with optimized confidence threshold (0.25) for multi-screen surveillance
- 🗺️ **Zone-Based Monitoring** - Multiple configurable detection zones (entrance, restricted) with customizable alert levels
- 📹 **Event Recording** - Automatic video recording with pre/post buffers when detections occur
- 🌐 **Web Dashboard** - FastAPI-powered real-time monitoring interface with live video feed
- 🔔 **Alert System** - In-app popup notifications with sound alerts for zone breaches
- 📊 **Real-time Statistics** - FPS tracking, detection counts, active alerts, and recording status
- 🗄️ **Event Database** - SQLite database logging all detections and system events
- 🎥 **Multi-Video Support** - Sequential looping of multiple surveillance video sources
- ⚡ **Performance Optimized** - Frame skipping, resolution scaling, and efficient processing

---

## 📁 Project Structure

```
security_only/
├── modules/              # Core system modules
│   ├── camera.py         # Camera capture
│   ├── detector.py       # Person detection (YOLOv8n)
│   ├── motion.py         # Motion detection
│   ├── zones.py          # Zone management
│   ├── alerts.py         # Alert system
│   ├── recorder.py       # Video recording
│   ├── tamper.py         # Tamper detection
│   ├── behavior.py       # Pattern learning
│   ├── database.py       # SQLite event logging
│   └── performance.py    # Performance monitoring
├── dashboard/            # Web dashboard
│   ├── app.py           # FastAPI application
│   ├── static/          # CSS, JavaScript
│   └── templates/       # HTML templates
├── config/
│   └── config.yaml      # System configuration
├── data/
│   ├── models/          # AI models storage
│   ├── recordings/      # Video clips
│   └── logs/            # System logs & database
├── main.py              # Main application
├── download_model.py    # YOLOv8n model downloader
├── requirements.txt     # Dependencies
└── test_*.py           # Test scripts
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd security_only
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download YOLOv8n Model

```bash
python download_model.py
```

### 3. Add Test Videos

Place surveillance videos in `data/test_videos/` or configure camera source in `config/config.yaml`

### 4. Run the Integrated System

```bash
# Run surveillance with web dashboard
python run_integrated.py
```

### 5. Access Dashboard

Open browser: `http://localhost:8080`

**Dashboard Features:**
- Live video feed with detection overlays
- Real-time statistics (FPS, detections, alerts, recordings)
- Recent detections list with timestamps
- Zone status monitoring
- In-app popup alert notifications with sound

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

**Camera:**
```yaml
camera:
  source:  # Can be list of videos for sequential looping
    - "data/test_videos/video1.mp4"
    - "data/test_videos/video2.mp4"
  width: 640
  height: 480
  fps: 20
```

**Detection:**
```yaml
detection:
  confidence_threshold: 0.25  # Lower for multi-screen surveillance
  input_size: 640  # Larger for better small object detection
  frame_skip: 5  # Process every Nth frame
```

**Zones:**
```yaml
zones:
  enabled: true
  definitions:
    - name: "entrance"
      x1: 100
      y1: 100
      x2: 300
      y2: 300
      alert_level: "low"
    - name: "restricted"
      x1: 400
      y1: 200
      x2: 600
      y2: 400
      alert_level: "high"
```

**Recording:**
```yaml
recording:
  enabled: true
  output_dir: "data/recordings"
  fps: 10
  duration: 10  # seconds per clip
  max_storage_mb: 1000
```

---

## 💻 System Requirements

**Development/Testing:**
- Python 3.8+
- 4GB+ RAM
- OpenCV with video codecs
- Works with video files or camera streams

**Performance:**
- Detection: ~20-25 FPS with frame_skip=5
- Stream: 15 FPS @ 640x360 resolution
- RAM usage: ~500-800 MB
- Supports 1920x1080 HD video sources

---

## 🧪 Testing

Run individual component tests:

```bash
# Test camera capture
python test_camera_sim.py

# Test person detection
python test_detection.py

# Test zones & alerts
python test_zones_alerts.py

```

---

## 📊 System Components

### Core Modules

1. **CameraCapture** - Multi-video source support with sequential looping
2. **PersonDetector** - YOLOv8n inference with configurable confidence threshold
3. **MotionDetector** - Background subtraction for motion detection
4. **ZoneMonitor** - Polygon-based zone detection with customizable areas
5. **ZoneAlertManager** - Alert triggering with cooldown management
6. **VideoRecorder** - Event-triggered recording with pre/post buffers
7. **EventDatabase** - SQLite logging for detections and system events
8. **PerformanceMonitor** - FPS tracking and performance metrics

---

## 🌐 Web Dashboard

The integrated FastAPI dashboard provides:

- 📹 **Live Video Feed** - MJPEG stream with detection overlays
- 📊 **Real-time Statistics** - FPS, detection count, active alerts, recording status
- 🔔 **Alert Notifications** - In-app popup alerts with sound for zone breaches
- 📋 **Recent Detections** - Scrollable list of recent person detections with timestamps
- 🗺️ **Zone Status** - Current status of all monitored zones
- 🔄 **Auto-refresh** - Dashboard updates every 10 seconds

### API Endpoints

- `GET /` - Dashboard UI
- `GET /api/security/status` - System status (running, FPS, detections, alerts, recording)
- `GET /api/security/detections?limit=N` - Recent detection events
- `GET /api/security/zones` - Zone status and configuration
- `GET /video_feed` - Live MJPEG video stream
- `WS /ws/security` - WebSocket for real-time alert notifications

---

## 📁 Data Storage

**Database:** `data/logs/events.db`
- Detection events with timestamps, zones, confidence scores
- System events and alerts
- Daily statistics summaries

**Recordings:** `data/recordings/`
- AVI format video clips
- Named with zone and timestamp (e.g., `zone_restricted_20251225_143056.avi`)
- Automatic cleanup when storage limit reached

**Logs:** `data/logs/system.log`
- System events and errors
- Performance metrics
- Debug information

---

## 🎯 Detection Optimization

**For Multi-Screen Surveillance:**
- Lower confidence threshold (0.25) to catch smaller people
- Increase input size (640) for better small object detection
- Frame skipping (5) for better performance

**For Single Camera:**
- Higher confidence threshold (0.5) to reduce false positives
- Standard input size (416) for faster processing
- Lower frame skip (2) for more frequent detection

---

## 🔧 Troubleshooting

**No detections appearing:**
- Check video source is accessible
- Verify YOLOv8n model is downloaded (`data/models/yolov8n.pt`)
- Lower confidence threshold in config
- Check zone definitions don't exclude detection areas

**Video recording issues:**
- Ensure `data/recordings/` directory exists and is writable
- Check sufficient disk space available
- Verify OpenCV is installed with video codec support

**Dashboard not accessible:**
- Check server is running (`python run_integrated.py`)
- Verify port 8080 is not in use
- Check firewall settings

**High CPU usage:**
- Increase frame_skip value
- Reduce input_size
- Lower video resolution
- Disable unnecessary features

---

## 📝 License

This project is open source and available for personal and educational use.

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Components are well-tested
- Documentation is updated
- Performance impact is considered
- `GET /api/security/behavior` - Behavior profile
- `GET /api/security/video_feed` - Live stream
- `POST /api/security/start` - Start surveillance
- `POST /api/security/stop` - Stop surveillance
- `WS /ws/security` - WebSocket updates

---

## 🔒 Privacy & Security

- ✅ All AI processing runs locally on-device
- ✅ No internet connection required
- ✅ No cloud uploads or external data transmission
- ✅ All data stored locally in SQLite database
- ✅ Video recordings saved locally only


---

## 🤝 Support

For issues or questions, please refer to the original repository documentation.

---

**Built with:** Python • OpenCV • YOLOv8 • FastAPI • PyTorch
