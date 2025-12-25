"""
Behavioral Pattern Learning Module
Learns normal activity patterns and detects anomalies
"""
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os


class BehaviorLearner:
    """Learns normal activity patterns and detects anomalies"""
    
    def __init__(self, 
                 learning_period_days=7,
                 min_samples=10,
                 anomaly_threshold=2.5,
                 time_bucket_minutes=30,
                 save_path='data/logs/behavior_profile.json'):
        """
        Initialize behavior learner
        
        Args:
            learning_period_days: Days of data to consider for learning
            min_samples: Minimum samples needed before anomaly detection
            anomaly_threshold: Standard deviations for anomaly (higher = stricter)
            time_bucket_minutes: Bucket size for time-based patterns
            save_path: Path to save/load learned patterns
        """
        self.learning_period_days = learning_period_days
        self.min_samples = min_samples
        self.anomaly_threshold = anomaly_threshold
        self.time_bucket_minutes = time_bucket_minutes
        self.save_path = save_path
        
        # Pattern storage
        # Format: {zone: {hour_bucket: [detection_counts]}}
        self.patterns = defaultdict(lambda: defaultdict(list))
        
        # Statistics per zone
        self.zone_stats = defaultdict(lambda: {
            'total_detections': 0,
            'first_seen': None,
            'last_seen': None,
            'hourly_average': {},
            'anomaly_count': 0
        })
        
        # Anomaly events
        self.anomalies = []
        
        # Load existing patterns if available
        self._load_patterns()
        
        print(f"🧠 BehaviorLearner initialized:")
        print(f"   Learning period: {learning_period_days} days")
        print(f"   Time buckets: {time_bucket_minutes} minutes")
        print(f"   Anomaly threshold: {anomaly_threshold} σ")
        print(f"   Min samples: {min_samples}")
    
    def learn_detection(self, zone_name, timestamp=None, confidence=0.0):
        """
        Learn from a detection event
        
        Args:
            zone_name: Zone where detection occurred
            timestamp: Detection timestamp (datetime object or ISO string)
            confidence: Detection confidence score
        """
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Get time bucket (e.g., hour 0-23)
        hour = timestamp.hour
        minute_bucket = (timestamp.minute // self.time_bucket_minutes) * self.time_bucket_minutes
        time_key = f"{hour:02d}:{minute_bucket:02d}"
        
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = timestamp.weekday()
        
        # Store pattern: day_of_week + time_bucket
        pattern_key = f"{day_of_week}_{time_key}"
        
        # Record detection
        self.patterns[zone_name][pattern_key].append({
            'timestamp': timestamp.isoformat(),
            'confidence': confidence,
            'hour': hour,
            'day': day_of_week
        })
        
        # Update zone statistics
        stats = self.zone_stats[zone_name]
        stats['total_detections'] += 1
        
        if stats['first_seen'] is None:
            stats['first_seen'] = timestamp.isoformat()
        stats['last_seen'] = timestamp.isoformat()
        
        # Update hourly averages
        if time_key not in stats['hourly_average']:
            stats['hourly_average'][time_key] = []
        stats['hourly_average'][time_key].append(confidence)
    
    def check_anomaly(self, zone_name, timestamp=None):
        """
        Check if current detection is anomalous
        
        Args:
            zone_name: Zone to check
            timestamp: Current timestamp
        
        Returns:
            Dict with anomaly detection result
        """
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        hour = timestamp.hour
        minute_bucket = (timestamp.minute // self.time_bucket_minutes) * self.time_bucket_minutes
        time_key = f"{hour:02d}:{minute_bucket:02d}"
        day_of_week = timestamp.weekday()
        pattern_key = f"{day_of_week}_{time_key}"
        
        # Check if we have enough data
        if zone_name not in self.patterns:
            return {
                'is_anomaly': False,
                'reason': 'no_data',
                'confidence': 0.0,
                'expected_frequency': 0,
                'learning': True
            }
        
        zone_patterns = self.patterns[zone_name]
        
        # Get historical frequency for this time slot
        historical_data = zone_patterns.get(pattern_key, [])
        
        if len(historical_data) < self.min_samples:
            return {
                'is_anomaly': False,
                'reason': 'insufficient_samples',
                'confidence': 0.0,
                'samples': len(historical_data),
                'required': self.min_samples,
                'learning': True
            }
        
        # Calculate statistics for this time slot across all weeks
        # Group by week and count detections per week
        week_counts = defaultdict(int)
        for event in historical_data:
            event_time = datetime.fromisoformat(event['timestamp'])
            week_key = event_time.strftime('%Y-%W')  # Year-Week
            week_counts[week_key] += 1
        
        counts = list(week_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts) if len(counts) > 1 else 1.0
        
        # Check 1: Unusual time detection
        # If this time slot typically has 0-1 detections per week
        unusual_time = mean_count < 1.0
        
        # Check 2: Unusually high frequency
        # Get recent detections in last hour
        recent_threshold = timestamp - timedelta(hours=1)
        recent_count = sum(1 for event in historical_data 
                          if datetime.fromisoformat(event['timestamp']) > recent_threshold)
        
        unusual_frequency = False
        if std_count > 0:
            z_score = (recent_count - mean_count) / std_count
            unusual_frequency = z_score > self.anomaly_threshold
        
        # Determine if anomaly
        is_anomaly = unusual_time or unusual_frequency
        
        result = {
            'is_anomaly': is_anomaly,
            'unusual_time': unusual_time,
            'unusual_frequency': unusual_frequency,
            'expected_frequency': mean_count,
            'actual_frequency': recent_count,
            'std_dev': std_count,
            'time_slot': time_key,
            'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week],
            'learning': False
        }
        
        # Log anomaly
        if is_anomaly:
            anomaly_event = {
                'zone': zone_name,
                'timestamp': timestamp.isoformat(),
                'reason': 'unusual_time' if unusual_time else 'unusual_frequency',
                'details': result
            }
            self.anomalies.append(anomaly_event)
            self.zone_stats[zone_name]['anomaly_count'] += 1
            
            reason = "unusual time" if unusual_time else "unusual frequency"
            print(f"🚨 ANOMALY DETECTED: {zone_name} - {reason} at {time_key}")
        
        return result
    
    def get_zone_profile(self, zone_name):
        """
        Get learned behavioral profile for a zone
        
        Args:
            zone_name: Zone to get profile for
        
        Returns:
            Dict with zone behavioral profile
        """
        if zone_name not in self.patterns:
            return None
        
        stats = self.zone_stats[zone_name]
        patterns = self.patterns[zone_name]
        
        # Calculate activity by hour
        hourly_activity = defaultdict(list)
        for pattern_key, events in patterns.items():
            day, time = pattern_key.split('_')
            hour = int(time.split(':')[0])
            hourly_activity[hour].extend(events)
        
        # Summarize hourly patterns
        hourly_summary = {}
        for hour, events in hourly_activity.items():
            hourly_summary[hour] = {
                'detections': len(events),
                'avg_confidence': np.mean([e['confidence'] for e in events]) if events else 0
            }
        
        # Find peak activity hours
        sorted_hours = sorted(hourly_summary.items(), 
                            key=lambda x: x[1]['detections'], 
                            reverse=True)
        peak_hours = [h for h, _ in sorted_hours[:3]]
        
        return {
            'zone': zone_name,
            'total_detections': stats['total_detections'],
            'first_seen': stats['first_seen'],
            'last_seen': stats['last_seen'],
            'anomaly_count': stats['anomaly_count'],
            'hourly_activity': hourly_summary,
            'peak_hours': peak_hours,
            'pattern_count': len(patterns),
            'learned_patterns': len([p for p in patterns.values() if len(p) >= self.min_samples])
        }
    
    def get_all_profiles(self):
        """Get behavioral profiles for all zones"""
        profiles = {}
        for zone_name in self.patterns.keys():
            profiles[zone_name] = self.get_zone_profile(zone_name)
        return profiles
    
    def get_recent_anomalies(self, limit=10):
        """Get recent anomaly events"""
        return self.anomalies[-limit:] if self.anomalies else []
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            data = {
                'patterns': dict(self.patterns),
                'zone_stats': dict(self.zone_stats),
                'anomalies': self.anomalies[-100:],  # Keep last 100 anomalies
                'metadata': {
                    'learning_period_days': self.learning_period_days,
                    'anomaly_threshold': self.anomaly_threshold,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Failed to save patterns: {e}")
            return False
    
    def _load_patterns(self):
        """Load learned patterns from disk"""
        if not os.path.exists(self.save_path):
            print("   No existing behavior patterns found")
            return False
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            # Convert defaultdict structure
            self.patterns = defaultdict(lambda: defaultdict(list))
            for zone, patterns in data.get('patterns', {}).items():
                for key, events in patterns.items():
                    self.patterns[zone][key] = events
            
            self.zone_stats = defaultdict(lambda: {
                'total_detections': 0,
                'first_seen': None,
                'last_seen': None,
                'hourly_average': {},
                'anomaly_count': 0
            })
            for zone, stats in data.get('zone_stats', {}).items():
                self.zone_stats[zone] = stats
            
            self.anomalies = data.get('anomalies', [])
            
            print(f"   ✅ Loaded patterns for {len(self.patterns)} zones")
            return True
        except Exception as e:
            print(f"   ⚠️ Failed to load patterns: {e}")
            return False
    
    def save(self):
        """Manually save patterns"""
        return self._save_patterns()
    
    def cleanup_old_data(self):
        """Remove data older than learning period"""
        cutoff = datetime.now() - timedelta(days=self.learning_period_days)
        removed_count = 0
        
        for zone in self.patterns:
            for pattern_key in list(self.patterns[zone].keys()):
                events = self.patterns[zone][pattern_key]
                filtered_events = [
                    e for e in events 
                    if datetime.fromisoformat(e['timestamp']) > cutoff
                ]
                
                removed = len(events) - len(filtered_events)
                removed_count += removed
                
                if filtered_events:
                    self.patterns[zone][pattern_key] = filtered_events
                else:
                    del self.patterns[zone][pattern_key]
        
        if removed_count > 0:
            print(f"🗑️ Behavior cleanup: Removed {removed_count} old events")
        
        return removed_count
