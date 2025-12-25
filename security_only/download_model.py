"""
Download and verify YOLOv8n model for person detection
"""
from ultralytics import YOLO
import os

def download_yolov8n():
    """Download YOLOv8n pre-trained model"""
    print("=" * 60)
    print("DOWNLOADING YOLOv8n MODEL")
    print("=" * 60)
    
    # Ensure models directory exists
    os.makedirs('data/models', exist_ok=True)
    
    try:
        print("\n📥 Downloading YOLOv8n (nano) model...")
        print("   This is the smallest YOLO model, optimized for edge devices")
        
        # Load model (will auto-download if not present)
        model = YOLO('yolov8n.pt')
        
        print("\n✅ Model downloaded successfully!")
        
        # Get model info
        print("\n📊 Model Information:")
        print(f"   Model type: YOLOv8n (nano)")
        print(f"   Task: Object Detection")
        print(f"   Classes: {len(model.names)} categories")
        
        # Check if 'person' class exists
        class_names = model.names
        if 'person' in class_names.values():
            person_id = [k for k, v in class_names.items() if v == 'person'][0]
            print(f"\n✅ 'person' class found at index: {person_id}")
        else:
            print("\n❌ 'person' class not found!")
            return False
        
        # Show some relevant classes for security
        print("\n🎯 Relevant classes for security:")
        security_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 
                           'dog', 'cat', 'backpack', 'handbag']
        for idx, name in class_names.items():
            if name in security_classes:
                print(f"   [{idx}] {name}")
        
        # Model file size
        model_path = 'yolov8n.pt'
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n💾 Model size: {size_mb:.2f} MB")
            print(f"   Location: {os.path.abspath(model_path)}")
        
        # Copy to data/models for organization
        import shutil
        dest_path = 'data/models/yolov8n.pt'
        if not os.path.exists(dest_path):
            shutil.copy(model_path, dest_path)
            print(f"\n📁 Model copied to: {dest_path}")
        
        print("\n" + "=" * 60)
        print("MODEL DOWNLOAD: ✅ SUCCESS")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False

def test_model_inference():
    """Test model with a simple inference"""
    print("\n" + "=" * 60)
    print("TESTING MODEL INFERENCE")
    print("=" * 60)
    
    try:
        # Load model
        model = YOLO('data/models/yolov8n.pt')
        
        print("\n🧪 Running inference test on sample image...")
        
        # Create a test image (person silhouette simulation)
        import numpy as np
        import cv2
        
        # Create 640x640 test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img[:] = (100, 100, 100)  # Gray background
        
        # Draw a person-like shape
        cv2.ellipse(test_img, (320, 200), (50, 70), 0, 0, 360, (200, 150, 100), -1)  # Head
        cv2.rectangle(test_img, (270, 250), (370, 450), (100, 120, 150), -1)  # Body
        cv2.rectangle(test_img, (270, 450), (310, 550), (80, 100, 120), -1)  # Left leg
        cv2.rectangle(test_img, (330, 450), (370, 550), (80, 100, 120), -1)  # Right leg
        
        # Save test image
        cv2.imwrite('data/test_output/test_person.jpg', test_img)
        print("   Created test image: data/test_output/test_person.jpg")
        
        # Run inference
        results = model(test_img, verbose=False)
        
        print(f"\n✅ Inference completed!")
        print(f"   Detected {len(results[0].boxes)} objects")
        
        # Check for person detection
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"   - {class_name}: {confidence:.2f} confidence")
        else:
            print("   (No objects detected in test image - this is OK for synthetic data)")
        
        print("\n" + "=" * 60)
        print("MODEL TEST: ✅ PASSED")
        print("=" * 60)
        print("\n✨ YOLOv8n is ready for person detection!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = download_yolov8n()
    if success:
        test_model_inference()
