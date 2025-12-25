"""
Event Database Module
SQLite database for logging detection events
"""
import sqlite3
import os
from datetime import datetime
import json


class EventDatabase:
    """Manages SQLite database for detection events"""
    
    def __init__(self, db_path='data/logs/events.db'):
        """
        Initialize event database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"🗄️ EventDatabase initialized: {db_path}")
    
    def _init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Detection events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                zone_name TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                recording_file TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System events table (alerts, errors, status changes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Statistics table (daily summaries)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_detections INTEGER DEFAULT 0,
                total_alerts INTEGER DEFAULT 0,
                total_recordings INTEGER DEFAULT 0,
                zones_triggered TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detection_timestamp 
            ON detection_events(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detection_zone 
            ON detection_events(zone_name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_system_timestamp 
            ON system_events(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_detection(self, zone_name=None, confidence=0.0, 
                     bbox=None, recording_file=None, metadata=None):
        """
        Log a person detection event
        
        Args:
            zone_name: Name of zone where detection occurred
            confidence: Detection confidence score
            bbox: Bounding box [x1, y1, x2, y2]
            recording_file: Path to associated recording
            metadata: Additional metadata as dict
        
        Returns:
            Event ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Parse bounding box (handle numpy arrays)
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = None, None, None, None
        
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO detection_events 
            (timestamp, event_type, zone_name, confidence, 
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, recording_file, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, 'person_detected', zone_name, confidence,
              x1, y1, x2, y2, recording_file, metadata_json))
        
        event_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return event_id
    
    def log_system_event(self, event_type, severity='info', 
                        message=None, metadata=None):
        """
        Log a system event (alert, error, status change)
        
        Args:
            event_type: Type of system event
            severity: Event severity (info, warning, error, critical)
            message: Human-readable message
            metadata: Additional metadata as dict
        
        Returns:
            Event ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO system_events 
            (timestamp, event_type, severity, message, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, event_type, severity, message, metadata_json))
        
        event_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return event_id
    
    def update_daily_stats(self, date=None, detections=0, 
                          alerts=0, recordings=0, zones=None):
        """
        Update daily statistics
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            detections: Number of detections to add
            alerts: Number of alerts to add
            recordings: Number of recordings to add
            zones: List of zone names triggered
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        zones_json = json.dumps(zones) if zones else None
        
        cursor.execute('''
            INSERT INTO daily_stats 
            (date, total_detections, total_alerts, total_recordings, zones_triggered)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_detections = total_detections + ?,
                total_alerts = total_alerts + ?,
                total_recordings = total_recordings + ?,
                zones_triggered = ?,
                updated_at = CURRENT_TIMESTAMP
        ''', (date, detections, alerts, recordings, zones_json,
              detections, alerts, recordings, zones_json))
        
        conn.commit()
        conn.close()
    
    def get_recent_detections(self, limit=10, zone_name=None):
        """
        Get recent detection events
        
        Args:
            limit: Maximum number of events to return
            zone_name: Filter by zone (optional)
        
        Returns:
            List of detection events
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if zone_name:
            cursor.execute('''
                SELECT * FROM detection_events 
                WHERE zone_name = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (zone_name, limit))
        else:
            cursor.execute('''
                SELECT * FROM detection_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        events = []
        for row in rows:
            event = dict(row)
            # Parse JSON metadata if it exists
            if event.get('metadata'):
                try:
                    if isinstance(event['metadata'], str):
                        event['metadata'] = json.loads(event['metadata'])
                    elif isinstance(event['metadata'], bytes):
                        event['metadata'] = json.loads(event['metadata'].decode('utf-8'))
                except:
                    event['metadata'] = {}
            # Handle bbox coordinates stored as various types
            for coord_key in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']:
                if coord_key in event and event[coord_key] is not None:
                    try:
                        if isinstance(event[coord_key], bytes):
                            # Convert binary blob to float using struct
                            import struct
                            event[coord_key] = struct.unpack('f', event[coord_key])[0]
                        else:
                            event[coord_key] = float(event[coord_key])
                    except:
                        event[coord_key] = None
            events.append(event)
        
        conn.close()
        return events
    
    def get_detections_by_timerange(self, start_time, end_time):
        """
        Get detections within time range
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
        
        Returns:
            List of detection events
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detection_events 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        ''', (start_time, end_time))
        
        rows = cursor.fetchall()
        events = [dict(row) for row in rows]
        
        conn.close()
        return events
    
    def get_zone_statistics(self, days=7):
        """
        Get detection statistics by zone
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dict with zone statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                zone_name,
                COUNT(*) as detection_count,
                AVG(confidence) as avg_confidence,
                MAX(confidence) as max_confidence,
                MIN(timestamp) as first_detection,
                MAX(timestamp) as last_detection
            FROM detection_events 
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY zone_name
            ORDER BY detection_count DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        stats = {}
        
        for row in rows:
            zone_name = row[0] if row[0] else 'unknown'
            stats[zone_name] = {
                'count': row[1],
                'avg_confidence': row[2],
                'max_confidence': row[3],
                'first_detection': row[4],
                'last_detection': row[5]
            }
        
        conn.close()
        return stats
    
    def get_daily_summary(self, date=None):
        """
        Get daily statistics summary
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        
        Returns:
            Dict with daily statistics
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_stats 
            WHERE date = ?
        ''', (date,))
        
        row = cursor.fetchone()
        
        if row:
            summary = dict(row)
            if summary.get('zones_triggered'):
                summary['zones_triggered'] = json.loads(summary['zones_triggered'])
        else:
            summary = None
        
        conn.close()
        return summary
    
    def cleanup_old_events(self, days=30):
        """
        Delete events older than specified days
        
        Args:
            days: Number of days to keep
        
        Returns:
            Number of deleted events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old detection events
        cursor.execute('''
            DELETE FROM detection_events 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        detections_deleted = cursor.rowcount
        
        # Delete old system events
        cursor.execute('''
            DELETE FROM system_events 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        system_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        total_deleted = detections_deleted + system_deleted
        
        if total_deleted > 0:
            print(f"🗑️ Database cleanup: Deleted {total_deleted} old events")
        
        return total_deleted
    
    def get_total_events(self):
        """Get total number of events in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM detection_events')
        detections = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM system_events')
        system = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'detections': detections,
            'system_events': system,
            'total': detections + system
        }


class HealthDatabase:
    """Manages SQLite database for crop health detection events"""
    
    def __init__(self, db_path='data/logs/health_events.db'):
        """
        Initialize health database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"🌱 HealthDatabase initialized: {db_path}")
    
    def _init_database(self):
        """Create health detection tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Health detection events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                disease_class TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_healthy INTEGER NOT NULL,
                severity TEXT,
                symptoms TEXT,
                organic_treatment TEXT,
                chemical_treatment TEXT,
                prevention TEXT,
                image_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Disease statistics table (aggregated data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disease_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_class TEXT NOT NULL UNIQUE,
                total_detections INTEGER DEFAULT 0,
                last_detected TEXT,
                avg_confidence REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Crop monitoring table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crop_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_type TEXT NOT NULL UNIQUE,
                total_scans INTEGER DEFAULT 0,
                healthy_count INTEGER DEFAULT 0,
                disease_count INTEGER DEFAULT 0,
                last_scan TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_health_timestamp 
            ON health_detections(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_health_crop 
            ON health_detections(crop_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_health_disease 
            ON health_detections(disease_class)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_detection(self, detection: dict, image_path: str = None):
        """
        Log a health detection event
        
        Args:
            detection: Detection dictionary from CropDiseaseDetector
            image_path: Optional path to saved image
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        recommendations = detection.get('recommendations', {})
        
        # Insert detection event
        cursor.execute('''
            INSERT INTO health_detections (
                timestamp, crop_type, disease_class, disease_name,
                confidence, is_healthy, severity,
                symptoms, organic_treatment, chemical_treatment, prevention,
                image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            detection['crop_type'],
            detection['disease_class'],
            detection['disease_name'],
            detection['confidence'],
            1 if detection['is_healthy'] else 0,
            recommendations.get('severity', 'unknown'),
            json.dumps(recommendations.get('symptoms', [])),
            json.dumps(recommendations.get('organic_treatment', [])),
            json.dumps(recommendations.get('chemical_treatment', [])),
            json.dumps(recommendations.get('prevention', [])),
            image_path
        ))
        
        # Update disease statistics
        if not detection['is_healthy']:
            cursor.execute('''
                INSERT INTO disease_stats (disease_class, total_detections, last_detected, avg_confidence)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(disease_class) DO UPDATE SET
                    total_detections = total_detections + 1,
                    last_detected = ?,
                    avg_confidence = (avg_confidence * (total_detections - 1) + ?) / total_detections,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                detection['disease_class'],
                datetime.now().isoformat(),
                detection['confidence'],
                datetime.now().isoformat(),
                detection['confidence']
            ))
        
        # Update crop monitoring
        cursor.execute('''
            INSERT INTO crop_monitoring (crop_type, total_scans, healthy_count, disease_count, last_scan)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(crop_type) DO UPDATE SET
                total_scans = total_scans + 1,
                healthy_count = healthy_count + ?,
                disease_count = disease_count + ?,
                last_scan = ?,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            detection['crop_type'],
            1 if detection['is_healthy'] else 0,
            0 if detection['is_healthy'] else 1,
            datetime.now().isoformat(),
            1 if detection['is_healthy'] else 0,
            0 if detection['is_healthy'] else 1,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_detections(self, limit: int = 10, crop_type: str = None):
        """
        Get recent health detection events
        
        Args:
            limit: Maximum number of records to return
            crop_type: Optional filter by crop type
            
        Returns:
            List of detection records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if crop_type:
            cursor.execute('''
                SELECT * FROM health_detections 
                WHERE crop_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (crop_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM health_detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            # Parse JSON fields
            if record.get('symptoms'):
                record['symptoms'] = json.loads(record['symptoms'])
            if record.get('organic_treatment'):
                record['organic_treatment'] = json.loads(record['organic_treatment'])
            if record.get('chemical_treatment'):
                record['chemical_treatment'] = json.loads(record['chemical_treatment'])
            if record.get('prevention'):
                record['prevention'] = json.loads(record['prevention'])
            results.append(record)
        
        conn.close()
        return results
    
    def get_disease_statistics(self, limit: int = None):
        """
        Get disease detection statistics
        
        Args:
            limit: Optional limit for top diseases
            
        Returns:
            List of disease statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM disease_stats 
            ORDER BY total_detections DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_crop_statistics(self):
        """
        Get crop monitoring statistics
        
        Returns:
            List of crop statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM crop_monitoring 
            ORDER BY total_scans DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_health_summary(self):
        """
        Get overall health monitoring summary
        
        Returns:
            Dictionary with summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM health_detections')
        total_detections = cursor.fetchone()[0]
        
        # Healthy vs disease
        cursor.execute('SELECT COUNT(*) FROM health_detections WHERE is_healthy = 1')
        healthy_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM health_detections WHERE is_healthy = 0')
        disease_count = cursor.fetchone()[0]
        
        # Unique diseases detected
        cursor.execute('SELECT COUNT(DISTINCT disease_class) FROM disease_stats')
        unique_diseases = cursor.fetchone()[0]
        
        # Crops monitored
        cursor.execute('SELECT COUNT(*) FROM crop_monitoring')
        crops_monitored = cursor.fetchone()[0]
        
        # Most common disease
        cursor.execute('''
            SELECT disease_class, total_detections 
            FROM disease_stats 
            ORDER BY total_detections DESC 
            LIMIT 1
        ''')
        top_disease = cursor.fetchone()
        
        # Most scanned crop
        cursor.execute('''
            SELECT crop_type, total_scans 
            FROM crop_monitoring 
            ORDER BY total_scans DESC 
            LIMIT 1
        ''')
        top_crop = cursor.fetchone()
        
        conn.close()
        
        summary = {
            'total_detections': total_detections,
            'healthy_count': healthy_count,
            'disease_count': disease_count,
            'health_rate': (healthy_count / total_detections * 100) if total_detections > 0 else 0,
            'unique_diseases': unique_diseases,
            'crops_monitored': crops_monitored,
            'most_common_disease': top_disease[0] if top_disease else None,
            'most_common_disease_count': top_disease[1] if top_disease else 0,
            'most_scanned_crop': top_crop[0] if top_crop else None,
            'most_scanned_crop_count': top_crop[1] if top_crop else 0
        }
        
        return summary
    
    def get_detections_by_date(self, start_date: str, end_date: str = None):
        """
        Get detections within a date range
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format), defaults to now
            
        Returns:
            List of detection records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if end_date:
            cursor.execute('''
                SELECT * FROM health_detections 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))
        else:
            cursor.execute('''
                SELECT * FROM health_detections 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (start_date,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def cleanup_old_records(self, days: int = 30):
        """
        Delete detection records older than specified days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM health_detections 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted > 0:
            print(f"🗑️ Health database cleanup: Deleted {deleted} old records")
        
        return deleted
    
    def export_to_csv(self, output_path: str, start_date: str = None):
        """
        Export detection records to CSV file
        
        Args:
            output_path: Path to output CSV file
            start_date: Optional start date filter
        """
        import csv
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if start_date:
            cursor.execute('''
                SELECT timestamp, crop_type, disease_class, disease_name, 
                       confidence, is_healthy, severity 
                FROM health_detections 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (start_date,))
        else:
            cursor.execute('''
                SELECT timestamp, crop_type, disease_class, disease_name, 
                       confidence, is_healthy, severity 
                FROM health_detections 
                ORDER BY timestamp DESC
            ''')
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Crop', 'Disease Class', 'Disease Name', 
                           'Confidence', 'Is Healthy', 'Severity'])
            writer.writerows(cursor.fetchall())
        
        conn.close()
        print(f"📄 Exported health records to: {output_path}")
