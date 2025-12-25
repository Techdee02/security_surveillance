"""
Camera Module - Handles video capture from device camera or IP camera
"""
import cv2
import time
import threading
from typing import Optional, Tuple
import numpy as np


class CameraCapture:
    """Handles camera initialization and frame capture with thread safety"""
    
    def __init__(self, source: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera capture
        
        Args:
            source: Camera source (0 for device camera, RTSP URL, or list of video files)
            width: Frame width
            height: Frame height
        """
        # Support multiple video sources
        if isinstance(source, list):
            self.sources = source
            self.current_source_index = 0
            self.source = self.sources[0]
        else:
            self.sources = [source]
            self.current_source_index = 0
            self.source = source
            
        self.width = width
        self.height = height
        self.cap = None
        self.is_open = False
        self.lock = threading.Lock()  # Thread-safe frame reading
        self.last_frame = None
        self.last_ret = False
        self.consecutive_failures = 0
        self.max_failures = 5  # Reconnect after 5 consecutive failures
        self.reconnect_delay = 2  # seconds
        
    def _switch_to_next_source(self):
        """Switch to the next video source in the list"""
        if len(self.sources) > 1:
            self.current_source_index = (self.current_source_index + 1) % len(self.sources)
            self.source = self.sources[self.current_source_index]
            print(f"📹 Switching to video {self.current_source_index + 1}/{len(self.sources)}: {self.source}")
            return True
        return False
    
    def start(self) -> bool:
        """Initialize camera connection"""
        try:
            print(f"\n📹 Connecting to camera: {self.source}")
            
            # For RTSP streams, use CAP_FFMPEG backend with timeout
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                # Set environment variable to prevent FFmpeg threading issues
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
                
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                # Set connection timeout (in milliseconds)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                # Reduce buffering to prevent threading issues
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera source {self.source}")
                print(f"   Please verify:")
                print(f"   1. Camera is online and accessible")
                print(f"   2. RTSP URL is correct")
                print(f"   3. You're on the same network as the camera")
                print(f"   4. No firewall blocking the connection")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Test read a frame to verify connection
            ret, test_frame = self.cap.read()
            if not ret:
                print(f"❌ Error: Camera opened but cannot read frames")
                self.cap.release()
                return False
            
            self.is_open = True
            print(f"✅ Camera connected successfully: {self.width}x{self.height}")
            print(f"   Actual frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame (thread-safe with auto-reconnect)
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_open or self.cap is None:
            return False, None
        
        with self.lock:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Successful read - reset failure counter
                    self.consecutive_failures = 0
                    self.last_frame = frame.copy()
                    self.last_ret = True
                    return ret, frame
                else:
                    # Failed to read frame - check if we should switch to next video
                    if len(self.sources) > 1:
                        # Switch to next video in the list
                        self.current_source_index = (self.current_source_index + 1) % len(self.sources)
                        self.source = self.sources[self.current_source_index]
                        print(f"📹 Switching to video {self.current_source_index + 1}/{len(self.sources)}: {self.source}")
                        # Release current capture before opening new one
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        if self.start():
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                self.consecutive_failures = 0
                                self.last_frame = frame.copy()
                                self.last_ret = True
                                return ret, frame
                    
                    # Failed to read frame
                    self.consecutive_failures += 1
                    
                    # Attempt reconnection after consecutive failures
                    if self.consecutive_failures >= self.max_failures:
                        print(f"\n⚠️ {self.consecutive_failures} consecutive frame failures. Attempting reconnection...")
                        self._reconnect()
                    
                    # Return cached frame if available
                    if self.last_frame is not None:
                        return True, self.last_frame
                    return False, None
                    
            except Exception as e:
                self.consecutive_failures += 1
                print(f"❌ Error reading frame: {e}")
                
                # Attempt reconnection after consecutive failures
                if self.consecutive_failures >= self.max_failures:
                    print(f"\n⚠️ Connection issues detected. Attempting reconnection...")
                    self._reconnect()
                
                # Return cached frame if available
                if self.last_frame is not None:
                    return True, self.last_frame
                return False, None
    
    def _reconnect(self):
        """Internal method to reconnect to camera"""
        try:
            print(f"🔄 Reconnecting to camera...")
            self.consecutive_failures = 0
            
            # Release existing connection
            if self.cap is not None:
                self.cap.release()
            
            time.sleep(self.reconnect_delay)
            
            # Attempt to reconnect
            if isinstance(self.source, str) and self.source.startswith('rtsp'):
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if self.cap.isOpened():
                # Test frame read
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"✅ Camera reconnected successfully!")
                    self.is_open = True
                else:
                    print(f"⚠️ Camera opened but cannot read frames")
            else:
                print(f"⚠️ Failed to reconnect to camera")
                self.is_open = False
                
        except Exception as e:
            print(f"❌ Reconnection error: {e}")
            self.is_open = False
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
