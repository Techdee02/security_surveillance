"""
Performance optimization module for Raspberry Pi 3
Threading, frame buffering, and smart inference management
"""
import threading
import queue
import time
import numpy as np
from typing import Optional, Callable, Dict, Any
import cv2


class ThreadedCamera:
    """
    Threaded camera capture for non-blocking frame reading
    Keeps fresh frames available while inference runs in parallel
    """
    
    def __init__(self, camera, buffer_size: int = 2):
        """
        Initialize threaded camera
        
        Args:
            camera: CameraCapture instance
            buffer_size: Maximum frames to buffer (small = more current)
        """
        self.camera = camera
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = 0
        
    def start(self):
        """Start capture thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ Threaded camera capture started")
        
    def _capture_loop(self):
        """Internal capture loop running in thread"""
        while self.running:
            ret, frame = self.camera.read_frame()
            
            if ret:
                self.frames_captured += 1
                self.last_frame_time = time.time()
                
                try:
                    # Non-blocking put, drops old frames if buffer full
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    self.frames_dropped += 1
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                    except:
                        pass
            else:
                time.sleep(0.01)  # Brief pause on read failure
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read latest frame from buffer
        
        Returns:
            Latest frame or None if buffer empty
        """
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✅ Threaded camera capture stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / self.frames_captured if self.frames_captured > 0 else 0,
            'buffer_size': self.frame_queue.qsize(),
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time > 0 else None
        }


class FrameSkipper:
    """
    Smart frame skipping for efficient inference
    Skips frames based on configurable interval
    """
    
    def __init__(self, skip_frames: int = 2):
        """
        Initialize frame skipper
        
        Args:
            skip_frames: Process every Nth frame (0 = no skip, 2 = every 3rd frame)
        """
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        
    def should_process(self) -> bool:
        """
        Check if current frame should be processed
        
        Returns:
            True if frame should be processed
        """
        self.frame_count += 1
        
        if self.skip_frames == 0 or (self.frame_count - 1) % (self.skip_frames + 1) == 0:
            self.processed_count += 1
            return True
        else:
            self.skipped_count += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get skipping statistics"""
        return {
            'total_frames': self.frame_count,
            'processed': self.processed_count,
            'skipped': self.skipped_count,
            'skip_rate': self.skipped_count / self.frame_count if self.frame_count > 0 else 0
        }
    
    def reset(self):
        """Reset counters"""
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0


class InferenceThrottler:
    """
    Throttles inference to target FPS for consistent performance
    Prevents CPU overload on Pi 3
    """
    
    def __init__(self, target_fps: float = 2.0):
        """
        Initialize throttler
        
        Args:
            target_fps: Target inference rate (FPS)
        """
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self.last_inference_time = 0
        self.inference_count = 0
        
    def should_infer(self) -> bool:
        """
        Check if enough time passed for next inference
        
        Returns:
            True if inference should run
        """
        current_time = time.time()
        elapsed = current_time - self.last_inference_time
        
        if elapsed >= self.min_interval:
            self.last_inference_time = current_time
            self.inference_count += 1
            return True
        
        return False
    
    def wait_if_needed(self):
        """Sleep if running too fast to maintain target FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_inference_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttling statistics"""
        elapsed = time.time() - self.last_inference_time if self.inference_count > 0 else 0
        
        return {
            'inference_count': self.inference_count,
            'target_fps': self.target_fps,
            'actual_fps': 1.0 / elapsed if elapsed > 0 else 0,
            'min_interval_ms': self.min_interval * 1000
        }


class PerformanceMonitor:
    """
    Monitors system performance and provides optimization recommendations
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of samples for rolling average
        """
        self.window_size = window_size
        
        # Timing data
        self.capture_times = []
        self.motion_times = []
        self.inference_times = []
        self.total_times = []
        
        # Counters
        self.frames_processed = 0
        self.start_time = time.time()
        
    def record_capture(self, duration_ms: float):
        """Record frame capture time"""
        self._add_sample(self.capture_times, duration_ms)
    
    def record_motion(self, duration_ms: float):
        """Record motion detection time"""
        self._add_sample(self.motion_times, duration_ms)
    
    def record_inference(self, duration_ms: float):
        """Record inference time"""
        self._add_sample(self.inference_times, duration_ms)
    
    def record_total(self, duration_ms: float):
        """Record total frame processing time"""
        self._add_sample(self.total_times, duration_ms)
        self.frames_processed += 1
    
    def _add_sample(self, samples: list, value: float):
        """Add sample and maintain window size"""
        samples.append(value)
        if len(samples) > self.window_size:
            samples.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        runtime = time.time() - self.start_time
        
        stats = {
            'runtime_seconds': runtime,
            'frames_processed': self.frames_processed,
            'avg_fps': self.frames_processed / runtime if runtime > 0 else 0,
        }
        
        # Add timing stats for each component
        if self.capture_times:
            stats['capture_avg_ms'] = np.mean(self.capture_times)
            stats['capture_max_ms'] = np.max(self.capture_times)
        
        if self.motion_times:
            stats['motion_avg_ms'] = np.mean(self.motion_times)
            stats['motion_max_ms'] = np.max(self.motion_times)
        
        if self.inference_times:
            stats['inference_avg_ms'] = np.mean(self.inference_times)
            stats['inference_max_ms'] = np.max(self.inference_times)
            stats['inference_count'] = len(self.inference_times)
        
        if self.total_times:
            stats['total_avg_ms'] = np.mean(self.total_times)
            stats['total_max_ms'] = np.max(self.total_times)
        
        return stats
    
    def print_report(self):
        """Print formatted performance report"""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        
        print(f"\n⏱️  Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"📊 Frames Processed: {stats['frames_processed']}")
        print(f"🎬 Average FPS: {stats['avg_fps']:.2f}")
        
        if 'capture_avg_ms' in stats:
            print(f"\n📷 Camera Capture:")
            print(f"   Average: {stats['capture_avg_ms']:.1f} ms")
            print(f"   Max: {stats['capture_max_ms']:.1f} ms")
        
        if 'motion_avg_ms' in stats:
            print(f"\n🏃 Motion Detection:")
            print(f"   Average: {stats['motion_avg_ms']:.1f} ms")
            print(f"   Max: {stats['motion_max_ms']:.1f} ms")
        
        if 'inference_avg_ms' in stats:
            print(f"\n🤖 Person Detection:")
            print(f"   Average: {stats['inference_avg_ms']:.1f} ms")
            print(f"   Max: {stats['inference_max_ms']:.1f} ms")
            print(f"   Count: {stats['inference_count']}")
        
        if 'total_avg_ms' in stats:
            print(f"\n⚡ Total Pipeline:")
            print(f"   Average: {stats['total_avg_ms']:.1f} ms")
            print(f"   Max: {stats['total_max_ms']:.1f} ms")
            
            # Calculate theoretical max FPS
            max_fps = 1000 / stats['total_avg_ms']
            print(f"   Theoretical Max FPS: {max_fps:.2f}")
        
        print("\n" + "=" * 70)
    
    def get_recommendations(self) -> list:
        """Get optimization recommendations based on performance"""
        stats = self.get_stats()
        recommendations = []
        
        if 'inference_avg_ms' in stats and stats['inference_avg_ms'] > 500:
            recommendations.append("⚠️  Inference is slow - consider reducing input size or using ONNX")
        
        if 'motion_avg_ms' in stats and stats['motion_avg_ms'] > 100:
            recommendations.append("⚠️  Motion detection is slow - reduce resolution or blur size")
        
        if stats['avg_fps'] < 1:
            recommendations.append("⚠️  Overall FPS too low - increase frame skipping or reduce inference")
        
        if not recommendations:
            recommendations.append("✅ Performance is good for Pi 3!")
        
        return recommendations
