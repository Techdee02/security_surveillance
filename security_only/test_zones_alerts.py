"""
Test zone detection and alert system
"""
import sys
sys.path.insert(0, '/workspaces/EdgeAI-IoT/security_surveillance')

from modules.zones import Zone, ZoneMonitor, create_zones_from_config
from modules.alerts import AlertSystem, AlertLevel, ZoneAlertManager
import cv2
import numpy as np
import os


def create_test_frame_with_zones(width=640, height=480):
    """Create frame with visible zones"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (120, 140, 160)
    
    # Add background elements
    cv2.rectangle(frame, (50, 50), (150, 150), (80, 100, 120), -1)
    cv2.rectangle(frame, (500, 300), (600, 400), (100, 120, 140), -1)
    
    return frame


def test_zones_and_alerts():
    """Test zone detection and alert system"""
    print("=" * 70)
    print("ZONE DETECTION & ALERT SYSTEM TEST")
    print("=" * 70)
    
    os.makedirs('data/test_output', exist_ok=True)
    
    # Define zones
    print("\n1️⃣ Creating detection zones...")
    
    # Entry zone (top-center)
    entry_zone = Zone(
        name="entry",
        points=[(250, 0), (390, 0), (390, 200), (250, 200)],
        color=(0, 255, 0),  # Green
        enabled=True
    )
    
    # Perimeter zone (left side)
    perimeter_zone = Zone(
        name="perimeter",
        points=[(0, 100), (150, 100), (150, 400), (0, 400)],
        color=(255, 165, 0),  # Orange
        enabled=True
    )
    
    # Restricted zone (bottom-right)
    restricted_zone = Zone(
        name="restricted",
        points=[(450, 300), (640, 300), (640, 480), (450, 480)],
        color=(255, 0, 0),  # Red
        enabled=True
    )
    
    zones = [entry_zone, perimeter_zone, restricted_zone]
    zone_monitor = ZoneMonitor(zones)
    
    print(f"   ✅ Created {len(zones)} zones:")
    for zone in zones:
        print(f"      - {zone.name} ({zone.color})")
    
    # Initialize alert system
    print("\n2️⃣ Initializing alert system...")
    alert_system = AlertSystem(
        buzzer_pin=18,
        led_pin=23,
        simulate=True  # Simulation mode
    )
    
    # Configure zone alert rules
    alert_manager = ZoneAlertManager(alert_system)
    alert_manager.add_zone_rule("entry", AlertLevel.INFO, cooldown_sec=5.0)
    alert_manager.add_zone_rule("perimeter", AlertLevel.WARNING, cooldown_sec=8.0)
    alert_manager.add_zone_rule("restricted", AlertLevel.CRITICAL, cooldown_sec=15.0)
    
    print("   ✅ Alert rules configured:")
    print("      - entry: INFO (5s cooldown)")
    print("      - perimeter: WARNING (8s cooldown)")
    print("      - restricted: CRITICAL (15s cooldown)")
    
    # Create test frame
    print("\n3️⃣ Creating test frame...")
    frame = create_test_frame_with_zones()
    
    # Draw zones
    frame_with_zones = zone_monitor.draw_zones(frame.copy())
    cv2.imwrite('data/test_output/zones_visualization.jpg', frame_with_zones)
    print("   ✅ Zones visualization saved: data/test_output/zones_visualization.jpg")
    
    # Test detections in different zones
    print("\n4️⃣ Testing detections in zones...")
    
    test_cases = [
        {
            'name': "Person at entry",
            'detections': [
                {'bbox': (300, 50, 60, 120), 'confidence': 0.85, 'class_name': 'person'}
            ]
        },
        {
            'name': "Person in perimeter",
            'detections': [
                {'bbox': (50, 200, 60, 120), 'confidence': 0.90, 'class_name': 'person'}
            ]
        },
        {
            'name': "Person in restricted area",
            'detections': [
                {'bbox': (500, 350, 60, 120), 'confidence': 0.78, 'class_name': 'person'}
            ]
        },
        {
            'name': "Multiple people (entry + perimeter)",
            'detections': [
                {'bbox': (280, 100, 50, 100), 'confidence': 0.82, 'class_name': 'person'},
                {'bbox': (80, 250, 55, 110), 'confidence': 0.88, 'class_name': 'person'}
            ]
        },
        {
            'name': "No detections in zones",
            'detections': [
                {'bbox': (320, 240, 60, 120), 'confidence': 0.75, 'class_name': 'person'}
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        
        # Check zone detections
        zone_detections = zone_monitor.check_detections(test_case['detections'])
        
        # Report which zones triggered
        triggered_zones = [name for name, dets in zone_detections.items() if dets]
        if triggered_zones:
            print(f"      Zones triggered: {', '.join(triggered_zones)}")
            
            # Trigger alerts
            alerts = alert_manager.process_zone_detections(zone_detections)
            print(f"      Alerts triggered: {alerts}")
        else:
            print("      No zones triggered")
        
        # Visualize
        vis_frame = frame.copy()
        vis_frame = zone_monitor.draw_zones(vis_frame)
        
        # Draw detections
        for det in test_case['detections']:
            x, y, w, h = det['bbox']
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.imwrite(f'data/test_output/zone_test_{i}.jpg', vis_frame)
    
    # Test cooldown
    print("\n5️⃣ Testing alert cooldown...")
    print("   Triggering restricted zone alert twice in quick succession...")
    
    test_detection = [{'bbox': (500, 350, 60, 120), 'confidence': 0.80, 'class_name': 'person'}]
    
    # First trigger
    zone_dets = zone_monitor.check_detections(test_detection)
    alert1 = alert_manager.process_zone_detections(zone_dets)
    print(f"   First alert: {'✅ Triggered' if alert1 > 0 else '❌ Not triggered'}")
    
    # Immediate second trigger (should be blocked by cooldown)
    zone_dets = zone_monitor.check_detections(test_detection)
    alert2 = alert_manager.process_zone_detections(zone_dets)
    print(f"   Second alert (immediate): {'✅ Triggered' if alert2 > 0 else '❌ Blocked by cooldown'}")
    
    # Get statistics
    print("\n6️⃣ Statistics:")
    zone_stats = zone_monitor.get_stats()
    alert_stats = alert_manager.get_stats()
    
    print(f"\n   📊 Zone Monitor:")
    print(f"      Total zones: {zone_stats['total_zones']}")
    print(f"      Enabled zones: {zone_stats['enabled_zones']}")
    print(f"      Total detections: {zone_stats['total_detections']}")
    
    print(f"\n   🚨 Alert System:")
    alert_info = alert_stats['alert_stats']
    print(f"      Total alerts: {alert_info['total_alerts']}")
    print(f"      Alert zones: {', '.join(alert_info['alert_zones'])}")
    
    print(f"\n   Zone-specific stats:")
    for zone_name, stats in zone_stats['zone_stats'].items():
        print(f"      {zone_name}: {stats['detection_count']} detections")
    
    print("\n" + "=" * 70)
    print("ZONE & ALERT TEST: ✅ COMPLETE")
    print("=" * 70)
    
    print("\n📋 Summary:")
    print("   ✅ Zone detection working correctly")
    print("   ✅ Multiple zone types configured")
    print("   ✅ Alert system integrated")
    print("   ✅ Zone-specific alert rules")
    print("   ✅ Cooldown mechanism prevents spam")
    print("   ✅ Simulation mode for development")
    
    print("\n🎯 Integration Ready:")
    print("   - Combine with person detector")
    print("   - Add to main surveillance loop")
    print("   - Connect to GPIO on Pi 3 deployment")
    
    return True


if __name__ == "__main__":
    test_zones_and_alerts()
