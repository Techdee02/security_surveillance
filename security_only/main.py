"""
Main Surveillance Application
Integrates all modules into a complete security system
"""
import sys
import yaml
import time
import signal
import cv2
from datetime import datetime

# Import all modules
from modules.camera import CameraCapture
from modules.motion import MotionDetector
from modules.detector import PersonDetector
from modules.zones import Zone, ZoneMonitor
from modules.alerts import ZoneAlertManager
from modules.recorder import VideoRecorder, StorageManager
from modules.database import EventDatabase
from modules.tamper import TamperDetector
from modules.behavior import BehaviorLearner
from modules.performance import PerformanceMonitor


class SurveillanceSystem:
    """Main surveillance system integrating all components"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize surveillance system
        
        Args:
            config_path: Path to configuration file
        """
        print("=" * 70)
        print("🔒 EDGE AI SECURITY & SURVEILLANCE SYSTEM")
        print("=" * 70)
        print(f"Starting system at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # System state
        self.running = False
        self.frame_count = 0
        self.latest_frame = None  # Store latest frame for streaming
        self.latest_annotated_frame = None  # Store annotated frame with detections
        self.detections_today = 0  # Counter for detections
        self.active_alerts = 0  # Counter for active alerts
        self.is_recording = False  # Recording state
        self.start_time = time.time()  # System start time
        self.current_fps = 0.0  # Current FPS
        self.last_fps_time = time.time()  # Last FPS calculation time
        self.fps_frame_count = 0  # Frame count for FPS calculation
        
        # Initialize all components
        self._init_components()
        
        # Set up signal handlers (only if in main thread)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Running in a thread, skip signal handlers
            pass
        
        print("\n✅ System initialization complete!")
        print("=" * 70)
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        print(f"📄 Loading configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("   ✅ Configuration loaded")
            return config
        except Exception as e:
            print(f"   ❌ Failed to load config: {e}")
            print("   Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration if file not found"""
        return {
            'camera': {'source': 0, 'width': 640, 'height': 480, 'fps': 10},
            'detection': {'confidence': 0.5, 'input_size': 416, 'frame_skip': 2},
            'motion': {'threshold': 1.0, 'learning_rate': 0.001},
            'zones': {'enabled': []},
            'alerts': {'cooldown_seconds': 60, 'buzzer_pin': 17, 'led_pin': 27},
            'recording': {'enabled': True, 'pre_buffer': 5, 'post_buffer': 10},
            'tamper': {'enabled': True, 'brightness_threshold': 20, 'movement_threshold': 0.15},
            'behavior': {'enabled': True, 'learning_period_days': 7}
        }
    
    def _init_components(self):
        """Initialize all system components"""
        print("\n🔧 Initializing components...")
        
        # 1. Camera
        cam_config = self.config['camera']
        self.camera = CameraCapture(
            source=cam_config.get('source', 0),
            width=cam_config.get('width', 640),
            height=cam_config.get('height', 480)
        )
        
        # 2. Motion Detector
        motion_config = self.config.get('motion', {})
        self.motion_detector = MotionDetector(
            sensitivity=motion_config.get('threshold', 0.015)
        )
        
        # 3. Person Detector
        det_config = self.config['detection']
        self.person_detector = PersonDetector(
            model_path='data/models/yolov8n.pt',
            conf_threshold=det_config.get('confidence_threshold', 0.5),
            input_size=det_config.get('input_size', 416)
        )
        self.person_detector.load_model()
        
        # 4. Zone Monitor
        zones_config = self.config.get('zones', {})
        zone_list = []
        if zones_config.get('enabled', False) and 'definitions' in zones_config:
            for zone_data in zones_config['definitions']:
                # Convert rectangle to polygon points
                points = [
                    (zone_data['x1'], zone_data['y1']),
                    (zone_data['x2'], zone_data['y1']),
                    (zone_data['x2'], zone_data['y2']),
                    (zone_data['x1'], zone_data['y2'])
                ]
                zone = Zone(
                    name=zone_data['name'],
                    points=points,
                    color=tuple(zone_data.get('color', [0, 255, 0]))
                )
                zone_list.append(zone)
        self.zone_monitor = ZoneMonitor(zones=zone_list)
        
        # 5. Alert Manager
        alert_config = self.config.get('alerts', {})
        from modules.alerts import AlertSystem
        alert_system = AlertSystem(
            buzzer_pin=alert_config.get('buzzer_pin', 18),
            led_pin=alert_config.get('led_red_pin', 23),
            simulate=not alert_config.get('gpio_enabled', False)
        )
        self.alert_manager = ZoneAlertManager(alert_system=alert_system)
        
        # 6. Video Recorder
        rec_config = self.config.get('recording', {})
        
        # Get actual frame size from camera
        ret, sample_frame = self.camera.read_frame()
        if ret and sample_frame is not None:
            actual_height, actual_width = sample_frame.shape[:2]
            print(f"📹 Using actual frame size for recorder: {actual_width}x{actual_height}")
        else:
            actual_width = cam_config.get('width', 640)
            actual_height = cam_config.get('height', 480)
            print(f"⚠️  Could not read frame, using config size: {actual_width}x{actual_height}")
        
        self.recorder = VideoRecorder(
            output_dir='data/recordings',
            pre_buffer_seconds=rec_config.get('pre_buffer', 5),
            post_buffer_seconds=rec_config.get('post_buffer', 10),
            fps=rec_config.get('fps', 10),
            resolution=(actual_width, actual_height)
        ) if rec_config.get('enabled', True) else None
        
        # 7. Storage Manager
        self.storage_manager = StorageManager(
            recordings_dir='data/recordings',
            max_storage_mb=1000
        ) if self.recorder else None
        
        # 8. Event Database
        self.database = EventDatabase(db_path='data/logs/events.db')
        
        # 9. Tamper Detector
        tamper_config = self.config.get('tamper', {})
        self.tamper_detector = TamperDetector(
            brightness_threshold=tamper_config.get('brightness_threshold', 20),
            movement_threshold=tamper_config.get('movement_threshold', 0.15)
        ) if tamper_config.get('enabled', True) else None
        
        # 10. Behavior Learner
        behavior_config = self.config.get('behavior', {})
        self.behavior_learner = BehaviorLearner(
            learning_period_days=behavior_config.get('learning_period_days', 7),
            save_path='data/logs/behavior_profile.json'
        ) if behavior_config.get('enabled', True) else None
        
        # 11. Performance Monitor
        self.performance = PerformanceMonitor()
        
        print("   ✅ All components initialized")
    
    def start(self):
        """Start the surveillance system"""
        print("\n🚀 Starting surveillance system...")
        
        # Start camera
        if not self.camera.start():
            print("❌ Failed to start camera")
            return False
        
        self.running = True
        self.frame_count = 0
        
        print("✅ System is now monitoring...")
        print("\nPress Ctrl+C to stop\n")
        print("-" * 70)
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n\n⚠️  Keyboard interrupt received")
        except Exception as e:
            print(f"\n\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
        
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        det_config = self.config['detection']
        frame_skip = det_config.get('frame_skip', 2)
        
        while self.running:
            # Performance tracking - FPS calculation
            self.fps_frame_count += 1
            if self.fps_frame_count >= 30:  # Calculate FPS every 30 frames
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                self.current_fps = self.fps_frame_count / elapsed if elapsed > 0 else 0
                self.fps_frame_count = 0
                self.last_fps_time = current_time
            
            # Decay active alerts every 150 frames (~5 seconds)
            if self.frame_count % 150 == 0:
                if self.active_alerts > 0:
                    self.active_alerts = max(0, self.active_alerts - 1)
            
            # Read frame
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                print("⚠️  Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # Always add frame to recorder buffer
            if self.recorder:
                self.recorder.add_frame(frame)
            
            # Always update tamper baseline
            if self.tamper_detector and self.frame_count <= 30:
                self.tamper_detector.update_baseline(frame)
            
            # Check for tampering
            tamper_result = None
            if self.tamper_detector:
                tamper_result = self.tamper_detector.check_tampering(frame)
                if tamper_result.get('tamper_detected'):
                    self._handle_tamper(tamper_result)
            
            # Skip frame processing if needed
            if self.frame_count % frame_skip != 0:
                continue
            
            # Motion detection
            has_motion, motion_areas = self.motion_detector.detect(frame)
            
            # Person detection (run on every processed frame)
            detections, annotated_frame = self.person_detector.detect_persons(
                frame, 
                draw_boxes=True
            )
            
            # Store annotated frame for streaming
            self.latest_frame = frame
            self.latest_annotated_frame = annotated_frame
            
            if detections:
                self._handle_detections(frame, detections, annotated_frame)
            
            # Update recording with detections
            if self.recorder:
                self.recorder.update_recording(annotated_frame, has_detection=len(detections) > 0)
                self.is_recording = self.recorder.is_recording
            
            # Periodic status update
            if self.frame_count % 100 == 0:
                self._print_status()
            
            # Periodic storage cleanup
            if self.storage_manager and self.frame_count % 1000 == 0:
                if self.storage_manager.should_cleanup():
                    self.storage_manager.cleanup_old_recordings()
    
    def _handle_detections(self, frame, detections, annotated_frame):
        """Handle person detections"""
        # Log ALL detections to database first (not just zone-based)
        for detection in detections:
            self.database.log_detection(
                zone_name='all',  # General detection, not zone-specific
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                metadata={'frame': self.frame_count}
            )
            self.detections_today += 1  # Increment counter
        
        # Check zones
        zone_detections = self.zone_monitor.check_detections(detections)
        
        # Log zone-specific detections to database
        for zone_name, zone_dets in zone_detections.items():
            if not zone_dets:
                continue
            
            for detection in zone_dets:
                # Log to database
                event_id = self.database.log_detection(
                    zone_name=zone_name,
                    confidence=detection['confidence'],
                    bbox=detection['bbox'],
                    metadata={'frame': self.frame_count, 'in_zone': True}
                )
        
        # Process zone detections and trigger alerts
        alerts_triggered = self.alert_manager.process_zone_detections(zone_detections)
        self.active_alerts += alerts_triggered  # Accumulate active alerts
        
        if alerts_triggered > 0:
            # Log alert to database
            self.database.log_system_event(
                event_type='alert_triggered',
                severity='warning',
                message=f"Person detected - {alerts_triggered} alerts triggered",
                metadata={'alerts_count': alerts_triggered}
            )
            
            # Broadcast alert via WebSocket if dashboard is available
            try:
                from dashboard.app import app
                app_state = app.state.app_state
                alert_zones = [k for k, v in zone_detections.items() if v]
                alert_data = {
                    "type": "alert",
                    "message": f"Person detected in {', '.join(alert_zones) if alert_zones else 'zone'}",
                    "severity": "warning",
                    "timestamp": time.time(),
                    "zone_detections": {k: len(v) for k, v in zone_detections.items() if v}
                }
                # Add to thread-safe queue instead of using asyncio.run()
                app_state.alert_queue.append(alert_data)
                print(f"🔔 Alert queued for WebSocket: {alert_data['message']}")
            except Exception as e:
                print(f"⚠️ Failed to queue alert: {e}")
                pass  # Dashboard not available
            
            # Start recording if we have zone detections
            if self.recorder:
                for zone_name, zone_dets in zone_detections.items():
                    if zone_dets:
                        detection = zone_dets[0]  # Use first detection
                        filename = self.recorder.start_recording(
                            event_type=f'zone_{zone_name}',
                            metadata={'zone': zone_name, 'confidence': detection['confidence']}
                        )
                        
                        if filename:
                            # Update detection with recording file
                            self.database.log_system_event(
                                event_type='recording_started',
                                severity='info',
                                message=f"Recording started: {filename}"
                            )
                        break  # Only start one recording
        
        # Behavioral learning
        if self.behavior_learner:
            for zone_name, zone_dets in zone_detections.items():
                for detection in zone_dets:
                    self.behavior_learner.learn_detection(
                        zone_name=zone_name,
                        confidence=detection['confidence']
                    )
                    
                    # Check for anomaly
                    anomaly_result = self.behavior_learner.check_anomaly(zone_name)
                    if anomaly_result.get('is_anomaly'):
                        self.database.log_system_event(
                            event_type='anomaly_detected',
                            severity='warning',
                            message=f"Anomalous activity in {zone_name}",
                            metadata=anomaly_result
                        )
                        print(f"🚨 ANOMALY: Unusual activity in {zone_name}")
        
        # Update daily stats with ALL detections (not just zone-based)
        total_detections = len(detections)
        alerts_count = sum(len(dets) for dets in zone_detections.values()) if zone_detections else 0
        all_zones = list(zone_detections.keys()) if zone_detections else ['all']
        self.database.update_daily_stats(
            detections=total_detections,
            alerts=alerts_triggered,
            recordings=1 if alerts_triggered > 0 else 0,
            zones=all_zones
        )
    
    def _handle_tamper(self, tamper_result):
        """Handle tampering detection"""
        if tamper_result.get('covered'):
            self.database.log_system_event(
                event_type='tamper_detected',
                severity='critical',
                message='Camera covering detected',
                metadata=tamper_result
            )
            # Trigger alert through the alert system
            if hasattr(self.alert_manager, 'alert_system'):
                from modules.alerts import AlertLevel
                self.alert_manager.alert_system.trigger_alert(
                    zone_name='system',
                    level=AlertLevel.CRITICAL,
                    duration_sec=5.0
                )
        
        if tamper_result.get('moved'):
            self.database.log_system_event(
                event_type='tamper_detected',
                severity='critical',
                message='Camera movement detected',
                metadata=tamper_result
            )
            # Trigger alert through the alert system
            if hasattr(self.alert_manager, 'alert_system'):
                from modules.alerts import AlertLevel
                self.alert_manager.alert_system.trigger_alert(
                    zone_name='system',
                    level=AlertLevel.CRITICAL,
                    duration_sec=5.0
                )
    
    def _print_status(self):
        """Print system status"""
        recorder_status = self.recorder.get_status() if self.recorder else {'recording': False}
        
        print(f"\n📊 Status [Frame {self.frame_count}]:")
        print(f"   Recording: {'🔴 YES' if recorder_status['recording'] else '⚪ NO'}")
        if recorder_status['recording']:
            print(f"   Duration: {recorder_status['duration']:.1f}s | Frames: {recorder_status['frames']}")
        
        # Database stats
        totals = self.database.get_total_events()
        print(f"   Events: {totals['detections']} detections, {totals['system_events']} system")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️  Received signal {signum}")
        self.running = False
    
    def stop(self):
        """Stop the surveillance system"""
        print("\n🛑 Stopping surveillance system...")
        
        self.running = False
        
        # Stop camera
        if self.camera:
            self.camera.release()
        
        # Stop recording
        if self.recorder:
            self.recorder.stop_recording()
            self.recorder.cleanup()
        
        # Stop alerts
        if self.alert_manager:
            pass  # Alert system will cleanup automatically
        
        # Save behavioral patterns
        if self.behavior_learner:
            self.behavior_learner.save()
        
        # Final statistics
        print("\n📊 Final Statistics:")
        totals = self.database.get_total_events()
        print(f"   Total frames: {self.frame_count}")
        print(f"   Detection events: {totals['detections']}")
        print(f"   System events: {totals['system_events']}")
        
        if self.storage_manager:
            usage = self.storage_manager.get_storage_usage()
            print(f"   Storage used: {usage['total_mb']:.1f} MB ({usage['file_count']} files)")
        
        print("\n✅ System stopped gracefully")
        print("=" * 70)


def main():
    """Main entry point"""
    # Create and start surveillance system
    system = SurveillanceSystem(config_path='config/config.yaml')
    system.start()


if __name__ == "__main__":
    main()
