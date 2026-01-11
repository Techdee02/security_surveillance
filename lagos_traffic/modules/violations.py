"""
Traffic Violations Module
Manages violation detection, evidence capture, and review workflow
"""

import sqlite3
import json
import uuid
import cv2
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional
from enum import Enum

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evidence storage directory
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class ViolationType(str, Enum):
    """Types of traffic violations"""
    NO_HELMET = "no_helmet"
    STATIONARY_OBSTRUCTION = "stationary_obstruction"
    WRONG_WAY = "wrong_way"
    LANE_VIOLATION = "lane_violation"
    NO_PARKING_ZONE = "no_parking_zone"
    BRT_LANE_VIOLATION = "brt_lane_violation"
    RED_LIGHT = "red_light"
    SPEEDING = "speeding"


class ViolationStatus(str, Enum):
    """Status of violation in review workflow"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class ViolationSeverity(str, Enum):
    """Severity level of violation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Violation type to severity mapping
VIOLATION_SEVERITY = {
    ViolationType.NO_HELMET: ViolationSeverity.HIGH,
    ViolationType.STATIONARY_OBSTRUCTION: ViolationSeverity.MEDIUM,
    ViolationType.WRONG_WAY: ViolationSeverity.CRITICAL,
    ViolationType.LANE_VIOLATION: ViolationSeverity.MEDIUM,
    ViolationType.NO_PARKING_ZONE: ViolationSeverity.LOW,
    ViolationType.BRT_LANE_VIOLATION: ViolationSeverity.MEDIUM,
    ViolationType.RED_LIGHT: ViolationSeverity.HIGH,
    ViolationType.SPEEDING: ViolationSeverity.HIGH,
}

# Fine amounts in Naira
VIOLATION_FINES = {
    ViolationType.NO_HELMET: 10000,
    ViolationType.STATIONARY_OBSTRUCTION: 25000,
    ViolationType.WRONG_WAY: 50000,
    ViolationType.LANE_VIOLATION: 20000,
    ViolationType.NO_PARKING_ZONE: 15000,
    ViolationType.BRT_LANE_VIOLATION: 30000,
    ViolationType.RED_LIGHT: 50000,
    ViolationType.SPEEDING: 30000,
}


class ViolationsDatabase:
    """
    Manages violation storage, review workflow, and evidence
    """
    
    def __init__(self, db_path=None):
        """Initialize violations database"""
        self.db_path = db_path or DB_PATH
        self.conn = None
        self.connect()
        self.create_tables()
        logger.info("Violations database initialized")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def create_tables(self):
        """Create violations tables"""
        cursor = self.conn.cursor()
        
        # Violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Vehicle info
                vehicle_type TEXT,
                vehicle_track_id INTEGER,
                plate_number TEXT,
                
                -- Detection info
                confidence REAL,
                bbox TEXT,
                
                -- Location
                zone_id TEXT,
                camera_id TEXT,
                
                -- Evidence paths
                snapshot_path TEXT,
                video_clip_path TEXT,
                
                -- Review workflow
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                reviewed_at DATETIME,
                review_notes TEXT,
                
                -- Enforcement
                fine_amount REAL,
                enforcement_status TEXT,
                
                -- Metadata
                extra_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Review audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violation_review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                previous_status TEXT,
                new_status TEXT,
                FOREIGN KEY (violation_id) REFERENCES violations(id)
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_status 
            ON violations(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_type 
            ON violations(violation_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
            ON violations(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_plate 
            ON violations(plate_number)
        """)
        
        self.conn.commit()
        logger.info("Violations tables created/verified")
    
    def generate_violation_id(self, violation_type: ViolationType) -> str:
        """Generate unique violation ID"""
        prefix = {
            ViolationType.NO_HELMET: "HEL",
            ViolationType.STATIONARY_OBSTRUCTION: "STA",
            ViolationType.WRONG_WAY: "WWD",
            ViolationType.LANE_VIOLATION: "LAN",
            ViolationType.NO_PARKING_ZONE: "NPZ",
            ViolationType.BRT_LANE_VIOLATION: "BRT",
            ViolationType.RED_LIGHT: "RLT",
            ViolationType.SPEEDING: "SPD",
        }.get(violation_type, "VIO")
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:6].upper()
        return f"{prefix}-{timestamp}-{unique}"
    
    def create_violation(
        self,
        violation_type: ViolationType,
        vehicle_type: str = None,
        vehicle_track_id: int = None,
        plate_number: str = None,
        confidence: float = None,
        bbox: List[int] = None,
        zone_id: str = None,
        camera_id: str = None,
        snapshot_path: str = None,
        video_clip_path: str = None,
        extra_data: Dict = None
    ) -> Optional[str]:
        """
        Create a new violation record
        
        Returns:
            violation_id if successful, None otherwise
        """
        try:
            violation_id = self.generate_violation_id(violation_type)
            severity = VIOLATION_SEVERITY.get(violation_type, ViolationSeverity.MEDIUM)
            fine_amount = VIOLATION_FINES.get(violation_type, 0)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO violations (
                    id, violation_type, severity, vehicle_type, vehicle_track_id,
                    plate_number, confidence, bbox, zone_id, camera_id,
                    snapshot_path, video_clip_path, fine_amount, extra_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation_id,
                violation_type.value if isinstance(violation_type, ViolationType) else violation_type,
                severity.value if isinstance(severity, ViolationSeverity) else severity,
                vehicle_type,
                vehicle_track_id,
                plate_number,
                confidence,
                json.dumps(bbox) if bbox else None,
                zone_id,
                camera_id,
                snapshot_path,
                video_clip_path,
                fine_amount,
                json.dumps(extra_data) if extra_data else None
            ))
            
            self.conn.commit()
            logger.info(f"Created violation: {violation_id}")
            return violation_id
            
        except sqlite3.Error as e:
            logger.error(f"Error creating violation: {e}")
            return None
    
    def get_violation(self, violation_id: str) -> Optional[Dict]:
        """Get violation by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM violations WHERE id = ?", (violation_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting violation: {e}")
            return None
    
    def get_pending_violations(
        self,
        limit: int = 50,
        violation_type: str = None,
        severity: str = None
    ) -> List[Dict]:
        """Get pending violations for review"""
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM violations WHERE status = 'pending'"
            params = []
            
            if violation_type:
                query += " AND violation_type = ?"
                params.append(violation_type)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error getting pending violations: {e}")
            return []
    
    def get_violations_by_status(
        self,
        status: ViolationStatus,
        limit: int = 100
    ) -> List[Dict]:
        """Get violations by status"""
        try:
            cursor = self.conn.cursor()
            status_value = status.value if isinstance(status, ViolationStatus) else status
            
            cursor.execute("""
                SELECT * FROM violations 
                WHERE status = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (status_value, limit))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error getting violations by status: {e}")
            return []
    
    def update_violation_status(
        self,
        violation_id: str,
        new_status: ViolationStatus,
        reviewer: str,
        notes: str = None
    ) -> bool:
        """Update violation status and log the review action"""
        try:
            cursor = self.conn.cursor()
            
            # Get current status
            cursor.execute("SELECT status FROM violations WHERE id = ?", (violation_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            previous_status = row['status']
            new_status_value = new_status.value if isinstance(new_status, ViolationStatus) else new_status
            
            # Update violation
            cursor.execute("""
                UPDATE violations
                SET status = ?, reviewed_by = ?, reviewed_at = ?, review_notes = ?, updated_at = ?
                WHERE id = ?
            """, (
                new_status_value,
                reviewer,
                datetime.now().isoformat(),
                notes,
                datetime.now().isoformat(),
                violation_id
            ))
            
            # Log review action
            cursor.execute("""
                INSERT INTO violation_review_log
                (violation_id, action, reviewer, notes, previous_status, new_status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                violation_id,
                new_status_value,
                reviewer,
                notes,
                previous_status,
                new_status_value
            ))
            
            self.conn.commit()
            logger.info(f"Updated violation {violation_id} status to {new_status_value}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error updating violation status: {e}")
            return False
    
    def approve_violation(self, violation_id: str, reviewer: str, notes: str = None) -> bool:
        """Approve a violation"""
        return self.update_violation_status(
            violation_id, ViolationStatus.APPROVED, reviewer, notes
        )
    
    def reject_violation(self, violation_id: str, reviewer: str, notes: str = None) -> bool:
        """Reject a violation"""
        return self.update_violation_status(
            violation_id, ViolationStatus.REJECTED, reviewer, notes
        )
    
    def escalate_violation(self, violation_id: str, reviewer: str, notes: str = None) -> bool:
        """Escalate a violation for supervisor review"""
        return self.update_violation_status(
            violation_id, ViolationStatus.ESCALATED, reviewer, notes
        )
    
    def get_violation_stats(self) -> Dict:
        """Get violation statistics"""
        try:
            cursor = self.conn.cursor()
            
            stats = {
                'total': 0,
                'pending': 0,
                'approved': 0,
                'rejected': 0,
                'escalated': 0,
                'by_type': {},
                'by_severity': {},
                'today': 0
            }
            
            # Total counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count FROM violations GROUP BY status
            """)
            for row in cursor.fetchall():
                stats[row['status']] = row['count']
                stats['total'] += row['count']
            
            # By violation type
            cursor.execute("""
                SELECT violation_type, COUNT(*) as count FROM violations GROUP BY violation_type
            """)
            for row in cursor.fetchall():
                stats['by_type'][row['violation_type']] = row['count']
            
            # By severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count FROM violations GROUP BY severity
            """)
            for row in cursor.fetchall():
                stats['by_severity'][row['severity']] = row['count']
            
            # Today's count
            cursor.execute("""
                SELECT COUNT(*) as count FROM violations 
                WHERE DATE(timestamp) = DATE('now')
            """)
            stats['today'] = cursor.fetchone()['count']
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error getting violation stats: {e}")
            return {}
    
    def get_review_log(self, violation_id: str) -> List[Dict]:
        """Get review history for a violation"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM violation_review_log
                WHERE violation_id = ?
                ORDER BY timestamp DESC
            """, (violation_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error getting review log: {e}")
            return []
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary"""
        d = dict(row)
        
        # Parse JSON fields
        if d.get('bbox'):
            try:
                d['bbox'] = json.loads(d['bbox'])
            except:
                pass
        
        if d.get('extra_data'):
            try:
                d['extra_data'] = json.loads(d['extra_data'])
            except:
                pass
        
        return d
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Violations database connection closed")


class EvidenceCapture:
    """
    Captures and stores violation evidence (snapshots and video clips)
    """
    
    def __init__(self, evidence_dir: Path = None):
        self.evidence_dir = evidence_dir or EVIDENCE_DIR
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.snapshots_dir = self.evidence_dir / "snapshots"
        self.videos_dir = self.evidence_dir / "videos"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.videos_dir.mkdir(exist_ok=True)
    
    def capture_snapshot(
        self,
        frame,
        violation_id: str,
        bbox: List[int] = None,
        draw_bbox: bool = True
    ) -> Optional[str]:
        """
        Capture and save a snapshot of the violation
        
        Args:
            frame: OpenCV image
            violation_id: Violation ID for naming
            bbox: Bounding box to highlight [x1, y1, x2, y2]
            draw_bbox: Whether to draw bounding box on image
            
        Returns:
            Path to saved snapshot
        """
        try:
            # Create a copy to avoid modifying original
            snapshot = frame.copy()
            
            # Draw bounding box if provided
            if bbox and draw_bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add label
                cv2.putText(
                    snapshot, "VIOLATION",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                snapshot, timestamp,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Save snapshot
            filename = f"{violation_id}.jpg"
            filepath = self.snapshots_dir / filename
            cv2.imwrite(str(filepath), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Saved snapshot: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return None
    
    def get_snapshot_url(self, violation_id: str) -> str:
        """Get URL path for snapshot"""
        return f"/evidence/snapshots/{violation_id}.jpg"
    
    def get_video_url(self, violation_id: str) -> str:
        """Get URL path for video clip"""
        return f"/evidence/videos/{violation_id}.mp4"


# Singleton instances
_violations_db = None
_evidence_capture = None


def get_violations_db() -> ViolationsDatabase:
    """Get singleton violations database instance"""
    global _violations_db
    if _violations_db is None:
        _violations_db = ViolationsDatabase()
    return _violations_db


def get_evidence_capture() -> EvidenceCapture:
    """Get singleton evidence capture instance"""
    global _evidence_capture
    if _evidence_capture is None:
        _evidence_capture = EvidenceCapture()
    return _evidence_capture
