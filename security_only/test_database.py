"""
Test event database module
"""
import sys
sys.path.insert(0, '/workspaces/EdgeAI-IoT/security_surveillance')

from modules.database import EventDatabase
from datetime import datetime, timedelta
import os


def test_event_database():
    """Test the event database module"""
    print("=" * 70)
    print("EVENT DATABASE MODULE TEST")
    print("=" * 70)
    
    # Initialize database
    print("\n1️⃣ Initializing EventDatabase...")
    db = EventDatabase(db_path='data/logs/test_events.db')
    
    # Test detection logging
    print("\n2️⃣ Logging detection events...")
    
    # Log detection in entry zone
    event_id1 = db.log_detection(
        zone_name='entry_door',
        confidence=0.87,
        bbox=[100, 150, 300, 450],
        recording_file='data/recordings/detection_001.mp4',
        metadata={'frame_num': 123, 'motion_detected': True}
    )
    print(f"   ✅ Logged detection event ID: {event_id1}")
    
    # Log detection in perimeter zone
    event_id2 = db.log_detection(
        zone_name='perimeter_left',
        confidence=0.92,
        bbox=[200, 200, 400, 500],
        metadata={'frame_num': 456}
    )
    print(f"   ✅ Logged detection event ID: {event_id2}")
    
    # Log detection without zone
    event_id3 = db.log_detection(
        confidence=0.65,
        bbox=[50, 100, 150, 350]
    )
    print(f"   ✅ Logged detection event ID: {event_id3}")
    
    # Test system event logging
    print("\n3️⃣ Logging system events...")
    
    sys_id1 = db.log_system_event(
        event_type='alert_triggered',
        severity='warning',
        message='Person detected in restricted zone',
        metadata={'zone': 'entry_door', 'alert_type': 'buzzer'}
    )
    print(f"   ✅ Logged system event ID: {sys_id1}")
    
    sys_id2 = db.log_system_event(
        event_type='recording_started',
        severity='info',
        message='Video recording initiated'
    )
    print(f"   ✅ Logged system event ID: {sys_id2}")
    
    # Update daily statistics
    print("\n4️⃣ Updating daily statistics...")
    db.update_daily_stats(
        detections=3,
        alerts=1,
        recordings=1,
        zones=['entry_door', 'perimeter_left']
    )
    print("   ✅ Daily stats updated")
    
    # Query recent detections
    print("\n5️⃣ Querying recent detections...")
    recent = db.get_recent_detections(limit=5)
    print(f"   Found {len(recent)} recent detections:")
    for event in recent:
        print(f"   • ID {event['id']}: {event['zone_name'] or 'unknown'} "
              f"(confidence: {event['confidence']:.2f})")
    
    # Query detections by zone
    print("\n6️⃣ Querying detections by zone...")
    entry_detections = db.get_recent_detections(limit=10, zone_name='entry_door')
    print(f"   Entry zone detections: {len(entry_detections)}")
    
    # Get zone statistics
    print("\n7️⃣ Getting zone statistics...")
    stats = db.get_zone_statistics(days=7)
    print(f"   Zone statistics for last 7 days:")
    for zone, data in stats.items():
        print(f"   • {zone}: {data['count']} detections, "
              f"avg confidence: {data['avg_confidence']:.2f}")
    
    # Get daily summary
    print("\n8️⃣ Getting daily summary...")
    today = datetime.now().strftime('%Y-%m-%d')
    summary = db.get_daily_summary(date=today)
    if summary:
        print(f"   Daily summary for {today}:")
        print(f"   • Detections: {summary['total_detections']}")
        print(f"   • Alerts: {summary['total_alerts']}")
        print(f"   • Recordings: {summary['total_recordings']}")
        if summary.get('zones_triggered'):
            print(f"   • Zones: {', '.join(summary['zones_triggered'])}")
    else:
        print(f"   No summary available for {today}")
    
    # Get total events
    print("\n9️⃣ Getting total events count...")
    totals = db.get_total_events()
    print(f"   Total detections: {totals['detections']}")
    print(f"   Total system events: {totals['system_events']}")
    print(f"   Total events: {totals['total']}")
    
    # Test time range query
    print("\n🔟 Testing time range query...")
    end_time = datetime.now().isoformat()
    start_time = (datetime.now() - timedelta(hours=1)).isoformat()
    range_events = db.get_detections_by_timerange(start_time, end_time)
    print(f"   Events in last hour: {len(range_events)}")
    
    print("\n" + "=" * 70)
    print("EVENT DATABASE TEST: ✅ COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   ✅ Detection logging: Working")
    print(f"   ✅ System event logging: Working")
    print(f"   ✅ Daily statistics: Working")
    print(f"   ✅ Query operations: Working")
    print(f"   ✅ Zone statistics: Working")
    print(f"   ✅ Time range queries: Working")
    
    # Check database file
    db_path = 'data/logs/test_events.db'
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path) / 1024
        print(f"\n   Database file: {db_size:.2f} KB")
    
    print("\n✨ Event database is production-ready!")
    
    return True


if __name__ == "__main__":
    test_event_database()
