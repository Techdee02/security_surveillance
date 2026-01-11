# Lagos Traffic Analysis System - Feature Roadmap

## Overview

This document outlines the planned features for enhancing the Lagos Traffic Analysis System with advanced violation detection, safety monitoring, and enforcement capabilities.

---

## 🎯 Feature Goals

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Traffic Rules Violation Detection | Detect lane violations, wrong-way driving, illegal stops | 📋 Planned |
| 2 | Helmet Detection for Okada Riders | Identify motorcycle riders without safety helmets | 📋 Planned |
| 3 | Pothole / Road Hazard Detection | Detect road surface damage and hazards | 📋 Planned |
| 4 | Stationary Vehicle Detection | Identify vehicles blocking lanes or causing obstruction | 📋 Planned |
| 5 | ALPR (License Plate Recognition) | Capture and read vehicle plate numbers | 📋 Planned |
| 6 | Human-in-the-Loop Verification | Review queue for violations before enforcement | ✅ **Completed** |

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Stream Input                        │
│              (CCTV / Traffic Cameras / Drones)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Vehicle Detection Layer (Existing)              │
│         Okada, Danfo, BRT, Keke Napep, Cars, Trucks         │
│                    [YOLOv8 + Custom Classification]          │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│  Helmet   │  │ Stationary│  │  Traffic  │  │  Pothole  │
│ Detection │  │  Vehicle  │  │   Rules   │  │ Detection │
│  Module   │  │  Module   │  │  Module   │  │  Module   │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      └──────────────┴──────┬───────┴──────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Violation Detected?   │
              └───────────┬─────────────┘
                          │ Yes
                          ▼
              ┌─────────────────────────┐
              │   ALPR + Snapshot       │
              │   Capture Evidence      │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Review Queue (DB)     │
              │   Status: PENDING       │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Human Review UI       │
              │   Approve / Reject      │
              └───────────┬─────────────┘
                          │ Approved
                          ▼
              ┌─────────────────────────┐
              │   Enforcement Action    │
              │   (Alert/Fine/Report)   │
              └─────────────────────────┘
```

---

## 🔧 Feature Specifications

### 1. Traffic Rules Violation Detection

#### Description
Detect various traffic rule violations using zone-based analysis and vehicle trajectory tracking.

#### Violation Types
| Violation | Detection Method | Priority |
|-----------|------------------|----------|
| Wrong-way driving | Trajectory direction analysis | High |
| Lane violation | Crossing lane boundary polygons | High |
| Running red light | Traffic light zone monitoring | High |
| Illegal U-turn | Trajectory pattern matching | Medium |
| No-parking zone violation | Stationary detection in marked zones | Medium |
| Bus lane violation | Non-BRT vehicles in BRT lanes | Medium |

#### Technical Approach
```python
# Zone-based violation detection
class TrafficZone:
    zone_id: str
    zone_type: str  # "lane", "intersection", "no_parking", "brt_only"
    polygon: List[Tuple[int, int]]  # Boundary coordinates
    rules: List[str]  # Applicable rules
    allowed_vehicles: List[str]  # Vehicle types allowed
    direction: Optional[str]  # Expected traffic direction
```

#### Data Requirements
- Zone configuration (polygon coordinates for each monitored area)
- Traffic light status integration (optional, for red light detection)
- Direction rules per lane

#### Output
```json
{
  "violation_id": "VIO-2026-001234",
  "violation_type": "wrong_way_driving",
  "timestamp": "2026-01-10T14:32:15Z",
  "location": {"zone_id": "ZONE-A1", "coordinates": [6.5244, 3.3792]},
  "vehicle": {
    "type": "okada",
    "plate_number": "LAG-123-ABC",
    "confidence": 0.87
  },
  "evidence": {
    "snapshot_url": "/evidence/VIO-2026-001234.jpg",
    "video_clip_url": "/evidence/VIO-2026-001234.mp4"
  },
  "status": "pending_review"
}
```

---

### 2. Helmet Detection for Okada Riders

#### Description
Identify motorcycle (okada) riders who are not wearing safety helmets, a common safety violation in Lagos.

#### Technical Approach

**Option A: Two-Stage Detection (Recommended)**
1. Detect motorcycle using existing detector
2. Crop rider region (upper body + head)
3. Run helmet classifier on cropped region

**Option B: End-to-End Detection**
- Train YOLOv8 model with classes: `helmet`, `no_helmet`, `motorcycle`

#### Model Options
| Model | Source | Pros | Cons |
|-------|--------|------|------|
| Custom YOLOv8 | Train on helmet dataset | Best accuracy for Lagos | Requires labeled data |
| Pre-trained Helmet Model | Roboflow/HuggingFace | Quick deployment | May need fine-tuning |
| Classification Model | ResNet/EfficientNet | Lightweight | Requires good crop |

#### Datasets Available
- Indian Traffic Helmet Dataset (similar traffic patterns)
- Vietnam Motorcycle Safety Dataset
- Custom Lagos dataset (to be collected)

#### Implementation
```python
class HelmetDetector:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def detect_helmet(self, frame, motorcycle_bbox) -> dict:
        """
        Check if rider has helmet
        
        Returns:
            {
                "has_helmet": bool,
                "confidence": float,
                "rider_bbox": [x1, y1, x2, y2],
                "head_region": [x1, y1, x2, y2]
            }
        """
        # Crop rider region (upper portion of motorcycle bbox)
        rider_region = self._extract_rider_region(frame, motorcycle_bbox)
        
        # Run helmet detection
        result = self.model(rider_region)
        
        return {
            "has_helmet": result.class_name == "helmet",
            "confidence": result.confidence
        }
```

#### Output
```json
{
  "violation_id": "HEL-2026-000456",
  "violation_type": "no_helmet",
  "timestamp": "2026-01-10T14:35:22Z",
  "vehicle": {
    "type": "okada",
    "track_id": 1234
  },
  "detection": {
    "helmet_detected": false,
    "confidence": 0.92,
    "rider_count": 2,
    "riders_without_helmet": 2
  },
  "evidence": {
    "snapshot_url": "/evidence/HEL-2026-000456.jpg"
  },
  "status": "pending_review"
}
```

---

### 3. Pothole / Road Hazard Detection

#### Description
Detect road surface hazards including potholes, cracks, debris, and water puddles that pose risks to vehicles and pedestrians.

#### Hazard Types
| Hazard | Visual Characteristics | Risk Level |
|--------|----------------------|------------|
| Pothole | Dark, irregular shape, shadow | High |
| Large crack | Linear dark patterns | Medium |
| Water puddle | Reflective surface | Medium |
| Debris | Objects on road surface | High |
| Road damage | Texture anomalies | Medium |

#### Technical Approach

**Option A: Object Detection**
- Train YOLOv8 on pothole detection dataset
- Real-time detection with bounding boxes

**Option B: Semantic Segmentation**
- Segment road surface vs damaged areas
- More accurate boundary detection

#### Datasets Available
- Pothole Detection Dataset (Kaggle)
- Road Damage Detection Challenge Dataset
- Custom Lagos road dataset (to be collected)

#### Implementation
```python
class RoadHazardDetector:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.hazard_types = ["pothole", "crack", "debris", "puddle"]
    
    def detect_hazards(self, frame) -> List[dict]:
        """
        Detect road hazards in frame
        
        Returns:
            List of hazard detections with location and severity
        """
        results = self.model(frame)
        
        hazards = []
        for detection in results:
            hazards.append({
                "type": detection.class_name,
                "bbox": detection.bbox,
                "confidence": detection.confidence,
                "severity": self._estimate_severity(detection),
                "location": self._estimate_real_world_location(detection.bbox)
            })
        
        return hazards
```

#### Output
```json
{
  "hazard_id": "HAZ-2026-000789",
  "hazard_type": "pothole",
  "timestamp": "2026-01-10T14:40:00Z",
  "location": {
    "pixel_coords": [450, 620, 520, 680],
    "gps_coords": [6.5244, 3.3792],
    "road_name": "Lekki-Epe Expressway"
  },
  "severity": "high",
  "estimated_size": "0.8m x 0.5m",
  "confidence": 0.85,
  "evidence": {
    "snapshot_url": "/evidence/HAZ-2026-000789.jpg"
  },
  "status": "reported",
  "reported_to": "Lagos State Public Works"
}
```

---

### 4. Stationary Vehicle / Obstruction Detection

#### Description
Detect vehicles that remain stationary in non-parking zones, blocking traffic lanes or causing obstruction.

#### Detection Criteria
| Scenario | Threshold | Action |
|----------|-----------|--------|
| Vehicle stopped in lane | > 30 seconds | Warning |
| Vehicle stopped in lane | > 2 minutes | Violation |
| Vehicle in no-parking zone | > 1 minute | Violation |
| Breakdown (hazard lights) | Any duration | Alert (assistance) |
| Double parking | > 30 seconds | Violation |

#### Technical Approach
Extend existing `VehicleTracker` to track dwell time:

```python
class StationaryVehicleDetector:
    def __init__(self, tracker: VehicleTracker):
        self.tracker = tracker
        self.stationary_threshold = 30  # seconds
        self.violation_threshold = 120  # seconds
        self.vehicle_positions = {}  # track_id -> position history
    
    def update(self, tracked_vehicles: List[dict], timestamp: float):
        """
        Update stationary status for all tracked vehicles
        """
        for vehicle in tracked_vehicles:
            track_id = vehicle["track_id"]
            current_pos = vehicle["center"]
            
            if track_id not in self.vehicle_positions:
                self.vehicle_positions[track_id] = {
                    "first_seen": timestamp,
                    "last_position": current_pos,
                    "stationary_since": None
                }
            else:
                # Check if vehicle has moved
                last_pos = self.vehicle_positions[track_id]["last_position"]
                distance = self._calculate_distance(current_pos, last_pos)
                
                if distance < 10:  # pixels - considered stationary
                    if self.vehicle_positions[track_id]["stationary_since"] is None:
                        self.vehicle_positions[track_id]["stationary_since"] = timestamp
                else:
                    self.vehicle_positions[track_id]["stationary_since"] = None
                    self.vehicle_positions[track_id]["last_position"] = current_pos
    
    def get_stationary_vehicles(self, current_time: float) -> List[dict]:
        """
        Get list of vehicles that have been stationary beyond threshold
        """
        stationary = []
        for track_id, data in self.vehicle_positions.items():
            if data["stationary_since"] is not None:
                duration = current_time - data["stationary_since"]
                if duration > self.stationary_threshold:
                    stationary.append({
                        "track_id": track_id,
                        "duration": duration,
                        "is_violation": duration > self.violation_threshold
                    })
        return stationary
```

#### Output
```json
{
  "violation_id": "STA-2026-000321",
  "violation_type": "stationary_obstruction",
  "timestamp": "2026-01-10T14:45:30Z",
  "vehicle": {
    "type": "private_car",
    "track_id": 567,
    "plate_number": "KJA-456-XY"
  },
  "obstruction": {
    "duration_seconds": 185,
    "zone": "main_lane",
    "blocking_lanes": 1,
    "traffic_impact": "moderate"
  },
  "evidence": {
    "snapshot_url": "/evidence/STA-2026-000321.jpg",
    "video_clip_url": "/evidence/STA-2026-000321.mp4"
  },
  "status": "pending_review"
}
```

---

### 5. ALPR (Automatic License Plate Recognition)

#### Description
Capture high-quality snapshots of vehicles and extract license plate numbers for identification and enforcement.

#### Nigerian Plate Formats
| Type | Format | Example |
|------|--------|---------|
| Private (Old) | ABC-123-XY | LAG-456-AB |
| Private (New) | STATE ABC 123 XY | LAGOS KJA 234 BX |
| Commercial | ABC-123 XY | BDG-789 KT |
| Government | STATE GOV 123 | LAGOS GOV 001 |
| Diplomatic | 123 CD 456 | 001 CD 234 |

#### Technical Approach

**Two-Stage Pipeline:**
1. **Plate Detection:** Detect license plate region
2. **OCR:** Read characters from plate

```python
class ALPRSystem:
    def __init__(self, detector_model: str, ocr_engine: str = "paddleocr"):
        self.plate_detector = load_model(detector_model)
        self.ocr = self._init_ocr(ocr_engine)
    
    def recognize_plate(self, frame, vehicle_bbox) -> dict:
        """
        Detect and read license plate from vehicle region
        
        Returns:
            {
                "plate_detected": bool,
                "plate_bbox": [x1, y1, x2, y2],
                "plate_text": str,
                "confidence": float,
                "plate_type": str
            }
        """
        # Crop vehicle region with margin
        vehicle_region = self._crop_with_margin(frame, vehicle_bbox)
        
        # Detect plate
        plate_detection = self.plate_detector(vehicle_region)
        
        if not plate_detection:
            return {"plate_detected": False}
        
        # Crop plate region
        plate_region = self._crop_plate(vehicle_region, plate_detection.bbox)
        
        # Preprocess for OCR
        processed_plate = self._preprocess_plate(plate_region)
        
        # Run OCR
        ocr_result = self.ocr(processed_plate)
        
        # Post-process and validate
        plate_text = self._postprocess_plate_text(ocr_result)
        plate_type = self._identify_plate_type(plate_text)
        
        return {
            "plate_detected": True,
            "plate_bbox": plate_detection.bbox,
            "plate_text": plate_text,
            "confidence": ocr_result.confidence,
            "plate_type": plate_type
        }
    
    def _preprocess_plate(self, plate_image):
        """
        Enhance plate image for better OCR
        - Resize to standard height
        - Convert to grayscale
        - Apply adaptive thresholding
        - Deskew if needed
        """
        pass
    
    def _postprocess_plate_text(self, raw_text: str) -> str:
        """
        Clean and validate plate text
        - Remove special characters
        - Fix common OCR errors (0 vs O, 1 vs I)
        - Validate against Nigerian plate patterns
        """
        pass
```

#### OCR Engine Options
| Engine | Pros | Cons |
|--------|------|------|
| PaddleOCR | Good accuracy, free | Larger model |
| EasyOCR | Easy to use, free | Moderate accuracy |
| Tesseract | Lightweight, free | Needs tuning |
| Google Vision API | High accuracy | Paid, needs internet |
| Plate Recognizer | Optimized for plates | Paid API |

#### Output
```json
{
  "plate_id": "PLT-2026-001234",
  "timestamp": "2026-01-10T14:50:00Z",
  "vehicle": {
    "type": "okada",
    "track_id": 890
  },
  "plate": {
    "text": "LAG-234-XY",
    "confidence": 0.94,
    "type": "private_old",
    "state": "Lagos"
  },
  "image": {
    "full_frame_url": "/plates/PLT-2026-001234_full.jpg",
    "plate_crop_url": "/plates/PLT-2026-001234_plate.jpg"
  },
  "associated_violations": ["HEL-2026-000456"]
}
```

---

### 6. Human-in-the-Loop Verification

#### Description
Provide a review interface for human operators to verify AI-detected violations before any enforcement action is taken.

#### Workflow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Violation  │───▶│   Review    │───▶│   Human     │───▶│ Enforcement │
│  Detected   │    │   Queue     │    │   Review    │    │   Action    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │  Decision:    │
                                    │  - Approve    │
                                    │  - Reject     │
                                    │  - Escalate   │
                                    └───────────────┘
```

#### Database Schema
```sql
-- Violations table
CREATE TABLE violations (
    id TEXT PRIMARY KEY,
    violation_type TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    vehicle_type TEXT,
    plate_number TEXT,
    confidence FLOAT,
    status TEXT DEFAULT 'pending',  -- pending, approved, rejected, escalated
    
    -- Evidence
    snapshot_path TEXT,
    video_clip_path TEXT,
    
    -- Review info
    reviewed_by TEXT,
    reviewed_at DATETIME,
    review_notes TEXT,
    
    -- Enforcement
    enforcement_action TEXT,
    fine_amount FLOAT,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Review audit log
CREATE TABLE review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id TEXT REFERENCES violations(id),
    action TEXT NOT NULL,  -- viewed, approved, rejected, escalated
    reviewer TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);
```

#### Review Interface Features
| Feature | Description |
|---------|-------------|
| Violation Queue | List of pending violations sorted by priority |
| Evidence Viewer | Display snapshot + video clip with zoom |
| Quick Actions | One-click approve/reject buttons |
| Bulk Review | Select multiple similar violations |
| Escalation | Flag for supervisor review |
| Statistics | Reviewer performance metrics |

#### API Endpoints
```python
# Review Queue Endpoints
GET  /api/violations/pending          # Get pending violations
GET  /api/violations/{id}             # Get violation details
POST /api/violations/{id}/approve     # Approve violation
POST /api/violations/{id}/reject      # Reject violation
POST /api/violations/{id}/escalate    # Escalate to supervisor
GET  /api/violations/stats            # Review statistics

# Evidence Endpoints
GET  /api/evidence/{violation_id}/snapshot    # Get snapshot image
GET  /api/evidence/{violation_id}/video       # Get video clip
```

#### Review UI Mockup
```
┌────────────────────────────────────────────────────────────────────┐
│  🚦 Lagos Traffic Violation Review System                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐  │
│  │     📷 Evidence Viewer      │  │     📋 Violation Details   │  │
│  │                             │  │                            │  │
│  │   [Snapshot Image]          │  │  ID: HEL-2026-000456       │  │
│  │                             │  │  Type: No Helmet           │  │
│  │   ▶️ [Play Video Clip]      │  │  Time: 14:35:22            │  │
│  │                             │  │  Vehicle: Okada            │  │
│  │   🔍 Zoom: [+] [-]          │  │  Plate: LAG-234-XY         │  │
│  │                             │  │  Confidence: 92%           │  │
│  └─────────────────────────────┘  │                            │  │
│                                   │  AI Detection:             │  │
│  ┌─────────────────────────────┐  │  ☑️ Motorcycle detected    │  │
│  │     📝 Review Notes         │  │  ☑️ Rider visible          │  │
│  │  ┌───────────────────────┐  │  │  ❌ No helmet detected     │  │
│  │  │                       │  │  │                            │  │
│  │  │ (Optional notes...)   │  │  └────────────────────────────┘  │
│  │  │                       │  │                                   │
│  │  └───────────────────────┘  │  ┌────────────────────────────┐  │
│  └─────────────────────────────┘  │     ⚡ Quick Actions        │  │
│                                   │                            │  │
│  ┌─────────────────────────────┐  │  [✅ Approve] [❌ Reject]  │  │
│  │     📊 Queue Stats          │  │  [⚠️ Escalate] [⏭️ Skip]   │  │
│  │  Pending: 23 | Today: 156   │  │                            │  │
│  │  Your Reviews: 45           │  └────────────────────────────┘  │
│  └─────────────────────────────┘                                   │
│                                                                     │
│  ◀️ Previous                                            Next ▶️    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📅 Implementation Priority

| Phase | Features | Timeline | Dependencies |
|-------|----------|----------|--------------|
| **Phase 1** | Human-in-the-Loop UI + Stationary Detection | Week 1-2 | Database schema |
| **Phase 2** | Helmet Detection | Week 3-4 | Helmet model |
| **Phase 3** | ALPR Integration | Week 5-6 | OCR setup |
| **Phase 4** | Traffic Zone Violations | Week 7-8 | Zone config UI |
| **Phase 5** | Pothole Detection | Week 9-10 | Road hazard model |

---

## 🔌 Integration Points

### External Systems
| System | Integration | Purpose |
|--------|-------------|---------|
| LASTMA | API/Database | Violation reporting |
| VIO (Lagos State) | API | Fine processing |
| FRSC | API | Federal road violations |
| Lagos State CCTV | RTSP streams | Camera feeds |
| SMS Gateway | API | Violation notifications |

### Data Flow
```
Camera Feed → Detection → Violation → Review Queue → Approval → Enforcement
                                          ↓
                                    Evidence Storage
                                          ↓
                                    Analytics Dashboard
```

---

## 📊 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Detection Accuracy | > 90% | Validated against human review |
| False Positive Rate | < 10% | Rejected violations / Total |
| Review Time | < 30 sec/violation | Average time per review |
| System Uptime | > 99% | Monitoring |
| Processing Latency | < 500ms | End-to-end detection time |

---

## 🛡️ Security & Privacy

### Data Protection
- All evidence encrypted at rest
- Access logging for all reviews
- Role-based access control
- Data retention policy (configurable)

### Privacy Considerations
- Face blurring option for non-violators
- License plate masking in public displays
- GDPR-compliant data handling
- Audit trail for all access

---

## 📚 References

### Datasets
- [Helmet Detection Dataset](https://universe.roboflow.com/search?q=helmet)
- [Pothole Detection Dataset](https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset)
- [License Plate Datasets](https://github.com/openalpr/benchmarks)

### Models
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

### Related Projects
- [OpenALPR](https://github.com/openalpr/openalpr)
- [Deep SORT Tracking](https://github.com/nwojke/deep_sort)

---

## 📝 Notes

### Lagos-Specific Considerations
1. **Okada Density:** Very high motorcycle traffic requires robust detection
2. **Yellow Vehicles:** Danfo buses share color with Keke Napep - need shape differentiation
3. **Road Conditions:** Variable road quality affects pothole detection
4. **Plate Conditions:** Many plates damaged, dirty, or non-standard

### Technical Constraints
1. **Processing Power:** CPU-only deployment may limit real-time performance
2. **Network:** Variable connectivity for cloud-based OCR
3. **Storage:** High volume of evidence requires efficient storage strategy
4. **Lighting:** 24/7 operation needs low-light capable models

---

*Document Version: 1.0*  
*Last Updated: January 10, 2026*  
*Author: Lagos Traffic Analysis System Team*
