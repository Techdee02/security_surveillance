# EcoGuard - Deployment Guide

## 🎯 Overview

Unified web dashboard for **Security & Surveillance** + **Smart Agriculture** systems running on Raspberry Pi 3.

## ✨ Features

### Security Dashboard
- 📹 **Live Video Feed** - MJPEG streaming with quality control (low/medium/high)
- 👥 **Person Detection** - Real-time detection stats and history
- 🗺️ **Zone Monitoring** - Zone activity charts and statistics
- 📹 **Video Recordings** - Browse, play, and download recorded clips
- 📊 **System Stats** - Total detections, events, storage usage
- 🔔 **Real-time Updates** - WebSocket live feed

### Agriculture Dashboard
- 🌡️ **Sensor Monitoring** - Soil moisture, temperature, humidity, light
- 📈 **Historical Charts** - Sensor trends over time (6-48 hours)
- 💧 **Irrigation Control** - Manual/automatic control with duration settings
- 🚨 **System Alerts** - Sensor threshold alerts and system notifications
- 📊 **Statistics** - Daily irrigation totals and system metrics

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd security_surveillance
pip install fastapi uvicorn python-multipart aiofiles websockets
```

### 2. Launch Dashboard Only (Testing)

```bash
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```

Access dashboard at: **http://localhost:8080**

### 3. Launch Integrated System (Production)

```bash
python launch_integrated.py
```

This starts:
- Surveillance system (person detection, recording, etc.)
- Web dashboard on port 8080
- Both systems run in parallel with shared state

## 📡 API Endpoints

### Security Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/security/stats` | GET | Overall system statistics |
| `/api/security/detections` | GET | Recent detection events (limit, zone filter) |
| `/api/security/zones` | GET | Zone activity statistics (days parameter) |
| `/api/security/recordings` | GET | List recorded videos (sort: newest/oldest/largest) |
| `/api/security/recording/{filename}` | GET | Stream/download specific recording |
| `/api/security/stream` | GET | Live MJPEG video feed (quality parameter) |
| `/api/security/snapshot` | GET | Single frame capture |
| `/api/security/system/status` | GET | Current operational status |
| `/api/security/people/count` | GET | Current people count |

### Agriculture Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agriculture/sensors` | GET | All current sensor readings |
| `/api/agriculture/sensor/{name}` | GET | Specific sensor reading |
| `/api/agriculture/history` | GET | Historical sensor data (sensor, hours) |
| `/api/agriculture/irrigation/status` | GET | Irrigation system status |
| `/api/agriculture/irrigation/control` | POST | Control irrigation (action, duration) |
| `/api/agriculture/stats` | GET | System statistics (days parameter) |
| `/api/agriculture/alerts` | GET | Recent system alerts (limit parameter) |
| `/api/agriculture/system/status` | GET | Overall system status |

### WebSocket Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws/live` | WebSocket | Main live updates (all events) |
| `/ws/security` | WebSocket | Security-specific updates |
| `/ws/agriculture` | WebSocket | Agriculture-specific updates |

## 📱 Dashboard Usage

### Security Mode
1. View live camera feed with quality selection
2. Monitor recent detections in real-time
3. Check zone activity charts (adjustable time period)
4. Browse and download video recordings
5. Take snapshots from live feed

### Agriculture Mode
1. View current sensor readings with status indicators
2. Analyze sensor history charts (selectable sensor, time period)
3. Control irrigation system:
   - **Start Manual**: Run irrigation for specified duration
   - **Stop**: Immediately stop irrigation
   - **Auto Mode**: Enable rule-based automation
4. Monitor system alerts and recommendations

### Mode Switching
Click the **Security** or **Agriculture** buttons in the header to switch between dashboards.

## 🎨 UI Features

- **Dark Theme** - Professional dark UI optimized for 24/7 monitoring
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Real-time Updates** - WebSocket-powered live data (5-second refresh)
- **Chart.js Integration** - Interactive charts for zone activity and sensor trends
- **System Status** - Live connection indicator with online/offline status
- **Quality Controls** - Adjustable video quality and data refresh periods

## 🔧 Configuration

### Video Stream Quality

```javascript
// Low: 320x240, 50% JPEG quality
// Medium: 640x480, 60% JPEG quality (default)
// High: 1280x720, 80% JPEG quality
```

### WebSocket Connection

The dashboard automatically connects to WebSocket for live updates. Auto-reconnect with exponential backoff (max 5 attempts).

### API Rate Limiting

Default refresh interval: 5 seconds for all data sources.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           FastAPI Dashboard (Port 8080)         │
├─────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌──────────────────┐   │
│  │  Security APIs  │   │ Agriculture APIs │   │
│  │  (8 endpoints)  │   │  (8 endpoints)   │   │
│  └─────────────────┘   └──────────────────┘   │
│  ┌──────────────────────────────────────────┐  │
│  │      WebSocket (3 channels)              │  │
│  │  /ws/live, /ws/security, /ws/agriculture │  │
│  └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│            Shared AppState                      │
│  - surveillance_system                          │
│  - camera                                       │
│  - security_db (EventDatabase)                  │
│  - agriculture_db (future)                      │
└─────────────────────────────────────────────────┘
         ▲                          ▲
         │                          │
    ┌────┴────┐              ┌─────┴──────┐
    │ Camera  │              │  ESP32+    │
    │ RTSP    │              │  MQTT      │
    └─────────┘              └────────────┘
```

## 📊 Technology Stack

- **Backend**: FastAPI 0.123+, Python 3.12
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Real-time**: WebSocket
- **Charts**: Chart.js 4.4.0
- **Icons**: Font Awesome 6.4.0
- **Video**: MJPEG streaming (OpenCV)
- **Database**: SQLite (via EventDatabase)

## 🔐 Security Considerations

- **CORS**: Currently allows all origins (development mode)
  - For production, set specific origins in `dashboard/app.py`
- **Authentication**: Not implemented (add middleware for production)
- **HTTPS**: Use reverse proxy (nginx) with SSL for production
- **Rate Limiting**: Consider adding rate limiting for API endpoints

## 📦 File Structure

```
dashboard/
├── app.py                    # FastAPI main application
├── routes/
│   ├── security.py          # Security API endpoints
│   ├── agriculture.py       # Agriculture API endpoints
│   └── websocket.py         # WebSocket handlers
├── static/
│   ├── css/
│   │   └── style.css        # Dashboard styling (850+ lines)
│   └── js/
│       └── dashboard.js     # Dashboard logic (650+ lines)
└── templates/
    └── index.html           # Main HTML template
```

## 🐛 Troubleshooting

### Video Stream Shows "Camera Unavailable"
- Check camera connection and RTSP URL
- Ensure `app.state.app_state.camera` is set in `launch_integrated.py`
- Verify camera is started before accessing `/api/security/stream`

### WebSocket Disconnects Frequently
- Check network stability
- Verify firewall settings allow WebSocket connections
- Increase keep-alive interval in `dashboard.js` if needed

### No Detection Data
- Ensure surveillance system is running (`launch_integrated.py`)
- Check database file exists: `data/logs/events.db`
- Verify `app.state.app_state.security_db` is set

### Agriculture Sensors Show No Data
- Agriculture endpoints use placeholder data until ESP32 integration
- Implement MQTT client and update `app.state.app_state.agriculture_db`

## 🚀 Deployment on Raspberry Pi 3

### 1. Clone Repository

```bash
git clone https://github.com/Techdee1/EdgeAI-IoT.git
cd EdgeAI-IoT/security_surveillance
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip
```

### 3. Install Python Dependencies

```bash
pip3 install -r requirements.txt
pip3 install fastapi uvicorn websockets aiofiles python-multipart
```

### 4. Configure Camera

Edit `config/config.yaml`:
```yaml
camera:
  source: "rtsp://YOUR_CAMERA_IP:554/1"
  width: 640
  height: 480
```

### 5. Launch System

```bash
python3 launch_integrated.py
```

### 6. Access Dashboard

From any device on the same network:
```
http://RASPBERRY_PI_IP:8080
```

## 🔄 Auto-Start on Boot (systemd)

Create `/etc/systemd/system/edgeai.service`:

```ini
[Unit]
Description=Edge AI Unified System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/EdgeAI-IoT/security_surveillance
ExecStart=/usr/bin/python3 launch_integrated.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable edgeai
sudo systemctl start edgeai
sudo systemctl status edgeai
```

## 📚 API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8080/api/docs
- **ReDoc**: http://localhost:8080/api/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See main repository LICENSE file.

## 🔗 Links

- **GitHub**: https://github.com/Techdee1/EdgeAI-IoT
- **API Docs**: http://localhost:8080/api/docs
- **Dashboard**: http://localhost:8080

---

**Built with ❤️ for Raspberry Pi 3 | Edge AI | Privacy-First**
