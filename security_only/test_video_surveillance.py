"""
Test surveillance system with video file instead of camera
"""
import sys
import yaml
import cv2
import time
import argparse
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.camera import CameraCapture
from modules.detector import PersonDetector
from modules.motion import MotionDetector
from modules.zones import Zone, ZoneMonitor


def test_video_surveillance(video_path, show_preview=True):
    """
    Test surveillance system with a video file
    
    Args:
        video_path: Path to video file
        show_preview: Whether to display video preview
    """
    print("=" * 70)
    print("🎥 VIDEO-BASED SURVEILLANCE TEST")
    print("=" * 70)
    print(f"Video: {video_path}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        print("\nPlease download a CCTV video and place it in data/test_videos/")
        print("See DOWNLOAD_CCTV_VIDEO.md for instructions")
        return
    
    # Initialize components
    print("📹 Initializing camera with video file...")
    camera = CameraCapture(source=video_path)
    
    print("🤖 Loading YOLOv8n model...")
    detector = PersonDetector(
        model_path="data/models/yolov8n.pt",
        conf_threshold=0.5,
        input_size=416
    )
    
    if not detector.load_model():
        print("❌ Failed to load model")
        return
    
    print("🏃 Initializing motion detector...")
    motion_detector = MotionDetector(sensitivity=0.015, min_area=500)
    
    print("🗺️ Setting up detection zones...")
    zones = [
        Zone(
            name="entrance",
            points=[(100, 100), (300, 100), (300, 300), (100, 300)],
            color=(0, 255, 0),
            enabled=True
        ),
        Zone(
            name="restricted",
            points=[(400, 200), (600, 200), (600, 400), (400, 400)],
            color=(0, 0, 255),
            enabled=True
        )
    ]
    zone_monitor = ZoneMonitor(zones)
    
    print("\n" + "=" * 70)
    print("▶️  STARTING VIDEO PROCESSING")
    print("=" * 70)
    print("Press 'q' to quit, 'p' to pause, SPACE to continue")
    print()
    
    # Statistics
    stats = {
        'total_frames': 0,
        'motion_frames': 0,
        'detection_frames': 0,
        'total_detections': 0,
        'zone_detections': {}
    }
    
    paused = False
    frame_skip = 2  # Process every Nth frame
    fps_start = time.time()
    fps_count = 0
    
    # Get total frames in video for proper end detection
    import cv2
    temp_cap = cv2.VideoCapture(video_path)
    total_video_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()
    
    print(f"Video info: {total_video_frames} total frames")
    print()
    
    with camera:
        while True:
            # Read frame
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("\n✅ Video ended - Processing complete")
                break
            
            stats['total_frames'] += 1
            
            # Stop if we've processed all frames in the video
            if stats['total_frames'] >= total_video_frames:
                print("\n✅ Reached end of video - Processing complete")
                break
            
            # Skip frames for performance
            if stats['total_frames'] % frame_skip != 0:
                continue
            
            frame_start = time.time()
            
            # Motion detection
            has_motion, motion_boxes = motion_detector.detect(frame)
            
            if has_motion:
                stats['motion_frames'] += 1
                
                # Person detection on motion frames
                detections, annotated = detector.detect_persons(frame, draw_boxes=True)
                
                if len(detections) > 0:
                    stats['detection_frames'] += 1
                    stats['total_detections'] += len(detections)
                    
                    # Check zones
                    zone_detections = zone_monitor.check_detections(detections)
                    
                    for zone_name, zone_dets in zone_detections.items():
                        if len(zone_dets) > 0:
                            stats['zone_detections'][zone_name] = \
                                stats['zone_detections'].get(zone_name, 0) + len(zone_dets)
                else:
                    annotated = frame.copy()
                
                # Draw zones
                annotated = zone_monitor.draw_zones(annotated)
                
                # Calculate FPS
                fps_count += 1
                elapsed = time.time() - fps_start
                current_fps = fps_count / elapsed if elapsed > 0 else 0
                
                # Add info overlay
                info_text = [
                    f"Frame: {stats['total_frames']}",
                    f"Detections: {len(detections)}",
                    f"FPS: {current_fps:.1f}",
                    f"Motion: {'YES' if has_motion else 'NO'}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(annotated, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
                
                # Show preview
                if show_preview:
                    cv2.imshow('Video Surveillance Test', annotated)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️  User stopped processing")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
                    elif key == ord(' ') and paused:
                        paused = False
                        print("\n▶️  RESUMED")
                
                while paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p') or key == ord(' '):
                        paused = False
                        print("\n▶️  RESUMED")
                        break
                    elif key == ord('q'):
                        print("\n⏹️  User stopped processing")
                        break
            
            frame_time = time.time() - frame_start
            
            # Print progress every 30 frames
            if stats['total_frames'] % 30 == 0:
                fps_calc = fps_count / elapsed if elapsed > 0 else 0
                print(f"📊 Processed {stats['total_frames']} frames | "
                      f"Detections: {stats['total_detections']} | "
                      f"FPS: {fps_calc:.1f}")
    
    # Close windows
    if show_preview:
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)
    print(f"Total Frames Processed: {stats['total_frames']}")
    print(f"Frames with Motion: {stats['motion_frames']}")
    print(f"Frames with Detections: {stats['detection_frames']}")
    print(f"Total Person Detections: {stats['total_detections']}")
    
    total_time = time.time() - fps_start
    avg_fps = fps_count / total_time if total_time > 0 else 0
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total Processing Time: {total_time:.1f}s")
    
    if stats['zone_detections']:
        print(f"\n🗺️ Zone Detections:")
        for zone_name, count in stats['zone_detections'].items():
            print(f"  - {zone_name}: {count} detections")
    
    print("=" * 70)
    print("✅ Test complete!")


def main():
    parser = argparse.ArgumentParser(description='Test surveillance with video file')
    parser.add_argument('--video', '-v', 
                       default='data/test_videos/cctv_footage.mp4',
                       help='Path to video file')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable video preview window')
    
    args = parser.parse_args()
    
    test_video_surveillance(args.video, show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
