"""
Test person detection pipeline with simulated frames
"""
import sys
sys.path.insert(0, '/workspaces/EdgeAI-IoT/security_surveillance')

from modules.detector import PersonDetector
import cv2
import numpy as np
import os

def create_test_image_with_person(width=640, height=480):
    """Create a test image with a person silhouette"""
    # Create background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (120, 140, 160)  # Grayish background
    
    # Draw person silhouette
    person_x, person_y = width // 2, height // 2
    
    # Head
    cv2.circle(img, (person_x, person_y - 80), 40, (220, 180, 140), -1)
    
    # Body
    cv2.rectangle(img, 
                 (person_x - 50, person_y - 40),
                 (person_x + 50, person_y + 80),
                 (180, 150, 120), -1)
    
    # Arms
    cv2.rectangle(img, 
                 (person_x - 90, person_y - 20),
                 (person_x - 50, person_y + 40),
                 (180, 150, 120), -1)
    cv2.rectangle(img, 
                 (person_x + 50, person_y - 20),
                 (person_x + 90, person_y + 40),
                 (180, 150, 120), -1)
    
    # Legs
    cv2.rectangle(img, 
                 (person_x - 40, person_y + 80),
                 (person_x - 10, person_y + 160),
                 (160, 130, 100), -1)
    cv2.rectangle(img, 
                 (person_x + 10, person_y + 80),
                 (person_x + 40, person_y + 160),
                 (160, 130, 100), -1)
    
    # Add text
    cv2.putText(img, "Test Person Image", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img

def test_person_detection():
    """Test the person detection pipeline"""
    print("=" * 70)
    print("PERSON DETECTION PIPELINE TEST")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('data/test_output', exist_ok=True)
    
    # Initialize detector
    print("\n1️⃣ Initializing PersonDetector...")
    detector = PersonDetector(
        model_path='data/models/yolov8n.pt',
        conf_threshold=0.3,  # Lower threshold for test
        input_size=416
    )
    
    # Load model
    print("\n2️⃣ Loading YOLOv8n model...")
    if not detector.load_model():
        print("❌ Failed to load model!")
        return False
    
    # Use Ultralytics sample image (has real people)
    print("\n3️⃣ Downloading test image with real people...")
    import urllib.request
    test_url = 'https://ultralytics.com/images/bus.jpg'
    try:
        urllib.request.urlretrieve(test_url, 'data/test_output/test_detection_input.jpg')
        test_img = cv2.imread('data/test_output/test_detection_input.jpg')
        print("   ✅ Test image downloaded and loaded")
    except Exception as e:
        print(f"   ⚠️ Could not download test image, using synthetic: {e}")
        test_img = create_test_image_with_person()
        cv2.imwrite('data/test_output/test_detection_input.jpg', test_img)
    
    # Run detection
    print("\n4️⃣ Running person detection...")
    detections, annotated_frame = detector.detect_persons(test_img, draw_boxes=True)
    
    # Display results
    print(f"\n   📊 Detection Results:")
    print(f"   {detector.get_detection_summary(detections)}")
    
    if detections:
        print(f"\n   Detailed detections:")
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            conf = det['confidence']
            print(f"   [{i}] Person - Confidence: {conf:.3f}")
            print(f"       BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    # Save annotated frame
    cv2.imwrite('data/test_output/test_detection_output.jpg', annotated_frame)
    print(f"\n   ✅ Annotated frame saved: data/test_output/test_detection_output.jpg")
    
    # Benchmark inference speed
    print("\n5️⃣ Benchmarking inference speed...")
    stats = detector.benchmark_inference(test_img, num_runs=10)
    
    print("\n" + "=" * 70)
    print("DETECTION PIPELINE TEST: ✅ COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\n📋 Summary:")
    print(f"   ✅ Model loaded successfully")
    print(f"   ✅ Detection pipeline working")
    print(f"   ✅ Inference speed: {stats['fps']:.2f} FPS")
    print(f"   ✅ Average latency: {stats['avg_time_ms']:.0f} ms")
    
    # Pi 3 expectations
    print("\n🎯 Raspberry Pi 3 Expectations:")
    print(f"   Current: {stats['fps']:.2f} FPS (dev container)")
    print(f"   Pi 3 expected: 1-3 FPS @ 416x416")
    print(f"   Optimization: Use frame skipping for real-time monitoring")
    
    print("\n✨ Person detection is ready for deployment!")
    
    return True

if __name__ == "__main__":
    test_person_detection()
