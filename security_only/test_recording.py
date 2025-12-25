"""
Test video recording and storage management
"""
import sys
sys.path.insert(0, '/workspaces/EdgeAI-IoT/security_surveillance')

from modules.recorder import VideoRecorder, StorageManager
import cv2
import numpy as np
import time
import os


def create_test_frame(width=640, height=480, frame_num=0, has_person=False):
    """Create a test frame with frame number and optional person"""
    # Create background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (120, 140, 160)
    
    # Add frame number
    cv2.putText(img, f"Frame {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add timestamp
    cv2.putText(img, f"Time: {time.time():.2f}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Add person if requested
    if has_person:
        person_x, person_y = width // 2, height // 2
        cv2.circle(img, (person_x, person_y - 80), 40, (220, 180, 140), -1)
        cv2.rectangle(img, (person_x - 50, person_y - 40),
                     (person_x + 50, person_y + 80), (180, 150, 120), -1)
        cv2.putText(img, "PERSON DETECTED", (width//2 - 150, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return img


def test_recording_module():
    """Test the video recording module"""
    print("=" * 70)
    print("VIDEO RECORDING MODULE TEST")
    print("=" * 70)
    
    # Initialize recorder
    print("\n1️⃣ Initializing VideoRecorder...")
    recorder = VideoRecorder(
        output_dir='data/recordings',
        pre_buffer_seconds=2,
        post_buffer_seconds=3,
        fps=10,
        resolution=(640, 480)
    )
    
    # Simulate continuous frame capture
    print("\n2️⃣ Simulating video stream...")
    print("   Buffering frames (no recording yet)...")
    
    frame_num = 0
    
    # Phase 1: Buffer frames (no detection)
    for i in range(30):
        frame = create_test_frame(frame_num=frame_num, has_person=False)
        recorder.add_frame(frame)
        frame_num += 1
        time.sleep(0.05)  # Simulate 20 FPS capture
    
    print(f"   ✅ Buffered {frame_num} frames")
    
    # Phase 2: Detection event triggers recording
    print("\n3️⃣ Detection event - Starting recording...")
    filename = recorder.start_recording(
        event_type='person_detected',
        metadata={'zone': 'entry_door', 'confidence': 0.87}
    )
    
    if not filename:
        print("❌ Failed to start recording")
        return False
    
    # Phase 3: Continue recording with detections
    print("\n4️⃣ Recording with detections...")
    detection_frames = 0
    for i in range(30):
        has_person = (i % 5 < 3)  # Person in 3 out of 5 frames
        frame = create_test_frame(frame_num=frame_num, has_person=has_person)
        recorder.add_frame(frame)
        recorder.update_recording(frame, has_detection=has_person)
        
        if has_person:
            detection_frames += 1
        
        frame_num += 1
        time.sleep(0.05)
        
        # Show status every 10 frames
        if i % 10 == 0:
            status = recorder.get_status()
            print(f"   Recording: {status['duration']:.1f}s, "
                  f"{status['frames']} frames")
    
    print(f"   ✅ Recorded {detection_frames} frames with detections")
    
    # Phase 4: No more detections, recording should stop after post-buffer
    print("\n5️⃣ No more detections - Post-buffer countdown...")
    for i in range(40):
        frame = create_test_frame(frame_num=frame_num, has_person=False)
        recorder.add_frame(frame)
        
        still_recording = recorder.update_recording(frame, has_detection=False)
        frame_num += 1
        time.sleep(0.05)
        
        if not still_recording:
            print(f"   ✅ Recording stopped automatically after post-buffer")
            break
    
    # Get final status
    status = recorder.get_status()
    print(f"\n   Final status: {status}")
    
    # Check if file was created
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024
        print(f"\n   ✅ Recording saved: {os.path.basename(filename)}")
        print(f"   File size: {file_size:.1f} KB")
    else:
        print(f"\n   ❌ Recording file not found: {filename}")
    
    # Cleanup
    recorder.cleanup()
    
    print("\n" + "=" * 70)
    print("RECORDING MODULE TEST: ✅ COMPLETE")
    print("=" * 70)
    
    return True


def test_storage_management():
    """Test storage management and cleanup"""
    print("\n" + "=" * 70)
    print("STORAGE MANAGEMENT TEST")
    print("=" * 70)
    
    # Initialize storage manager
    print("\n1️⃣ Initializing StorageManager...")
    storage = StorageManager(
        recordings_dir='data/recordings',
        max_storage_mb=1,  # Low limit for testing
        min_free_space_mb=0.1
    )
    
    # Check current usage
    print("\n2️⃣ Checking current storage usage...")
    usage = storage.get_storage_usage()
    print(f"   Current usage: {usage['total_mb']:.2f} MB")
    print(f"   Files: {usage['file_count']}")
    print(f"   Max allowed: {usage['max_mb']:.2f} MB")
    
    # Check if cleanup needed
    print("\n3️⃣ Checking if cleanup needed...")
    if storage.should_cleanup():
        print("   ⚠️ Storage limit exceeded, cleanup needed")
        deleted = storage.cleanup_old_recordings()
        print(f"   ✅ Cleanup complete: {deleted} files deleted")
    else:
        print("   ✅ Storage within limits, no cleanup needed")
    
    # Check usage after cleanup
    usage_after = storage.get_storage_usage()
    print(f"\n4️⃣ Storage after cleanup:")
    print(f"   Current usage: {usage_after['total_mb']:.2f} MB")
    print(f"   Files: {usage_after['file_count']}")
    
    print("\n" + "=" * 70)
    print("STORAGE MANAGEMENT TEST: ✅ COMPLETE")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    # Test recording
    test_recording_module()
    
    # Test storage management
    test_storage_management()
    
    print("\n✨ All recording tests complete!")
