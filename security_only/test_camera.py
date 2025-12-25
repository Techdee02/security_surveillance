"""
Test script for camera module (headless mode)
"""
import sys
sys.path.insert(0, '/workspaces/EdgeAI-IoT/security_surveillance')

from modules.camera import CameraCapture
import cv2
import os
import yaml

def test_camera():
    """Test camera capture and save sample frames"""
    print("Testing camera capture (headless mode)...")
    print("Will capture 10 frames and save a test image...")
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_source = config['camera'].get('source', 0)
    camera_width = config['camera'].get('width', 640)
    camera_height = config['camera'].get('height', 480)
    
    print(f"\n📹 Camera Configuration:")
    print(f"   Source: {camera_source}")
    print(f"   Resolution: {camera_width}x{camera_height}")
    print()
    
    # Create test output directory
    os.makedirs('data/test_output', exist_ok=True)
    
    with CameraCapture(source=camera_source, width=camera_width, height=camera_height) as camera:
        if not camera.is_open:
            print("❌ Failed to open camera!")
            print("Note: Running in a dev container without camera access.")
            print("This is expected - the code is ready for when you have camera access.")
            return False
        
        print("✅ Camera opened successfully!")
        
        frame_count = 0
        max_frames = 10
        
        while frame_count < max_frames:
            ret, frame = camera.read_frame()
            
            if not ret or frame is None:
                print(f"❌ Failed to capture frame {frame_count + 1}")
                break
            
            frame_count += 1
            
            # Add frame counter overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save first frame as test image
            if frame_count == 1:
                test_img_path = 'data/test_output/camera_test.jpg'
                cv2.imwrite(test_img_path, frame)
                print(f"✅ Saved test frame to: {test_img_path}")
            
            print(f"  Frame {frame_count}/{max_frames} captured - Shape: {frame.shape}")
        
        print(f"\n✅ Successfully captured {frame_count} frames!")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    print("\n✅ Camera test complete!")

if __name__ == "__main__":
    test_camera()
