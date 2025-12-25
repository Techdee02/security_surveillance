"""
Person Detection Module - YOLOv8n inference pipeline
"""
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time


class PersonDetector:
    """Handles person detection using YOLOv8n"""
    
    def __init__(self, model_path: str = "data/models/yolov8n.pt", 
                 conf_threshold: float = 0.5,
                 input_size: int = 416):
        """
        Initialize person detector
        
        Args:
            model_path: Path to YOLOv8 model file
            conf_threshold: Confidence threshold for detections (0-1)
            input_size: Input image size (320, 416, or 640)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.model = None
        self.person_class_id = 0  # 'person' is class 0 in COCO
        
    def load_model(self) -> bool:
        """Load YOLOv8 model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"   Confidence threshold: {self.conf_threshold}")
            print(f"   Input size: {self.input_size}x{self.input_size}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_persons(self, frame: np.ndarray, 
                       draw_boxes: bool = True) -> Tuple[List[dict], np.ndarray]:
        """
        Detect persons in a frame
        
        Args:
            frame: Input image (BGR format)
            draw_boxes: Whether to draw bounding boxes on frame
            
        Returns:
            Tuple of (detections list, annotated frame)
            Each detection dict contains: bbox, confidence, class_name
        """
        if self.model is None:
            print("⚠️ Model not loaded! Call load_model() first.")
            return [], frame
        
        # Run inference
        results = self.model(frame, 
                           imgsz=self.input_size,
                           conf=self.conf_threshold,
                           verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        # Process results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Filter for person class only
                if class_id == self.person_class_id:
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person'
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    if draw_boxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Person {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                                        0.5, 2)
                        
                        # Background for label
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 5),
                                    (x1 + label_size[0], y1),
                                    (0, 255, 0), -1)
                        
                        # Label text
                        cv2.putText(annotated_frame, label, 
                                  (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 0), 2)
        
        return detections, annotated_frame
    
    def benchmark_inference(self, frame: np.ndarray, num_runs: int = 10) -> dict:
        """
        Benchmark inference speed
        
        Args:
            frame: Test frame
            num_runs: Number of inference runs
            
        Returns:
            Dict with timing statistics
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        print(f"\n🔬 Benchmarking inference ({num_runs} runs)...")
        
        times = []
        for i in range(num_runs):
            start = time.time()
            _ = self.model(frame, imgsz=self.input_size, 
                          conf=self.conf_threshold, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        stats = {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'fps': fps
        }
        
        print(f"   Average time: {stats['avg_time_ms']:.1f} ms")
        print(f"   Min time: {stats['min_time_ms']:.1f} ms")
        print(f"   Max time: {stats['max_time_ms']:.1f} ms")
        print(f"   Estimated FPS: {stats['fps']:.2f}")
        
        return stats
    
    def get_detection_summary(self, detections: List[dict]) -> str:
        """Get human-readable summary of detections"""
        if not detections:
            return "No persons detected"
        
        count = len(detections)
        avg_conf = np.mean([d['confidence'] for d in detections])
        
        return f"{count} person(s) detected (avg conf: {avg_conf:.2f})"
