"""
Video Recording Module
Handles event-triggered video recording with pre/post buffers
"""
import cv2
import os
from datetime import datetime
from collections import deque
import threading
import time


class VideoRecorder:
    """Manages event-triggered video recording"""
    
    def __init__(self, output_dir='data/recordings', 
                 pre_buffer_seconds=5, 
                 post_buffer_seconds=10,
                 fps=10,
                 resolution=(640, 480),
                 codec='mp4v',
                 max_recording_duration=300):
        """
        Initialize video recorder
        
        Args:
            output_dir: Directory to save recordings
            pre_buffer_seconds: Seconds of video before event
            post_buffer_seconds: Seconds after last detection
            fps: Frames per second for output video
            resolution: Video resolution (width, height)
            codec: Video codec (mp4v, avc1, etc.)
            max_recording_duration: Maximum recording length in seconds
        """
        self.output_dir = output_dir
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.max_recording_duration = max_recording_duration
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Pre-event frame buffer (circular buffer)
        buffer_size = int(fps * pre_buffer_seconds)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Recording state
        self.is_recording = False
        self.current_writer = None
        self.current_filename = None
        self.recording_start_time = None
        self.last_detection_time = None
        self.frame_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"📹 VideoRecorder initialized:")
        print(f"   Output: {output_dir}")
        print(f"   Pre-buffer: {pre_buffer_seconds}s ({buffer_size} frames)")
        print(f"   Post-buffer: {post_buffer_seconds}s")
        print(f"   Resolution: {resolution[0]}x{resolution[1]} @ {fps} FPS")
    
    def add_frame(self, frame):
        """
        Add frame to circular buffer (always running)
        
        Args:
            frame: Video frame to buffer
        """
        with self.lock:
            self.frame_buffer.append(frame.copy())
    
    def start_recording(self, event_type='detection', metadata=None):
        """
        Start a new recording (if not already recording)
        
        Args:
            event_type: Type of event triggering recording
            metadata: Additional metadata (zone info, etc.)
        
        Returns:
            Filename if recording started, None otherwise
        """
        with self.lock:
            if self.is_recording:
                # Already recording, update last detection time
                self.last_detection_time = time.time()
                return None
            
            # Generate filename with timestamp (.avi works better with mp4v)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = os.path.join(
                self.output_dir, 
                f"{event_type}_{timestamp}.avi"
            )
            
            # Create video writer
            # Use mp4v codec which is well supported
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.current_writer = cv2.VideoWriter(
                    self.current_filename,
                    fourcc,
                    self.fps,
                    self.resolution
                )
                
                if not self.current_writer.isOpened():
                    print(f"❌ Failed to create video writer: {self.current_filename}")
                    self.current_writer.release()
                    self.current_writer = None
                    return None
                    
                print(f"✅ Video writer created: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")
            except Exception as e:
                print(f"❌ Error creating video writer: {e}")
                return None
            
            # Write pre-buffered frames
            frames_written = 0
            for buffered_frame in self.frame_buffer:
                if frames_written == 0:
                    print(f"   First buffered frame: {buffered_frame.shape[1]}x{buffered_frame.shape[0]}")
                    print(f"   Writer expects: {self.resolution[0]}x{self.resolution[1]}")
                self.current_writer.write(buffered_frame)
                frames_written += 1
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.last_detection_time = time.time()
            self.frame_count = frames_written
            
            print(f"🎬 Recording started: {os.path.basename(self.current_filename)}")
            print(f"   Pre-buffered frames written: {frames_written}")
            if metadata:
                print(f"   Metadata: {metadata}")
            
            return self.current_filename
    
    def update_recording(self, frame, has_detection=False):
        """
        Update ongoing recording with new frame
        
        Args:
            frame: Current video frame
            has_detection: Whether current frame has detections
        
        Returns:
            True if recording updated, False if not recording
        """
        with self.lock:
            if not self.is_recording:
                return False
            
            # Write frame to video
            self.current_writer.write(frame)
            self.frame_count += 1
            
            # Update last detection time
            if has_detection:
                self.last_detection_time = time.time()
            
            # Check if we should stop recording
            current_time = time.time()
            time_since_detection = current_time - self.last_detection_time
            recording_duration = current_time - self.recording_start_time
            
            # Stop conditions:
            # 1. Post-buffer time exceeded
            # 2. Max recording duration reached
            if (time_since_detection >= self.post_buffer_seconds or 
                recording_duration >= self.max_recording_duration):
                self._stop_recording()
                return False
            
            return True
    
    def _stop_recording(self):
        """Internal method to stop recording (must hold lock)"""
        if self.current_writer:
            self.current_writer.release()
            self.current_writer = None
            
            duration = time.time() - self.recording_start_time
            print(f"⏹️  Recording stopped: {os.path.basename(self.current_filename)}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Frames: {self.frame_count}")
            print(f"   Size: {os.path.getsize(self.current_filename) / 1024:.1f} KB")
        
        self.is_recording = False
        self.current_filename = None
        self.recording_start_time = None
        self.last_detection_time = None
        self.frame_count = 0
    
    def stop_recording(self):
        """Manually stop current recording"""
        with self.lock:
            if self.is_recording:
                self._stop_recording()
                return True
            return False
    
    def get_status(self):
        """Get current recording status"""
        with self.lock:
            if not self.is_recording:
                return {
                    'recording': False,
                    'buffer_size': len(self.frame_buffer)
                }
            
            current_time = time.time()
            return {
                'recording': True,
                'filename': os.path.basename(self.current_filename),
                'duration': current_time - self.recording_start_time,
                'frames': self.frame_count,
                'time_since_detection': current_time - self.last_detection_time,
                'buffer_size': len(self.frame_buffer)
            }
    
    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            if self.is_recording:
                self._stop_recording()
            self.frame_buffer.clear()
        print("📹 VideoRecorder cleaned up")


class StorageManager:
    """Manages storage space for recordings"""
    
    def __init__(self, recordings_dir='data/recordings',
                 max_storage_mb=1000,
                 min_free_space_mb=100):
        """
        Initialize storage manager
        
        Args:
            recordings_dir: Directory containing recordings
            max_storage_mb: Maximum storage for recordings (MB)
            min_free_space_mb: Minimum free space to maintain (MB)
        """
        self.recordings_dir = recordings_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.min_free_space_bytes = min_free_space_mb * 1024 * 1024
        
        os.makedirs(recordings_dir, exist_ok=True)
        
        print(f"💾 StorageManager initialized:")
        print(f"   Max storage: {max_storage_mb} MB")
        print(f"   Min free space: {min_free_space_mb} MB")
    
    def get_storage_usage(self):
        """Get current storage usage"""
        total_size = 0
        file_count = 0
        
        if os.path.exists(self.recordings_dir):
            for filename in os.listdir(self.recordings_dir):
                filepath = os.path.join(self.recordings_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.mp4'):
                    total_size += os.path.getsize(filepath)
                    file_count += 1
        
        return {
            'total_bytes': total_size,
            'total_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'max_mb': self.max_storage_bytes / (1024 * 1024)
        }
    
    def cleanup_old_recordings(self, force=False):
        """
        Delete oldest recordings if storage limit exceeded
        
        Args:
            force: Force cleanup even if under limit
        
        Returns:
            Number of files deleted
        """
        usage = self.get_storage_usage()
        
        if not force and usage['total_bytes'] < self.max_storage_bytes:
            return 0
        
        # Get all recording files with timestamps
        files = []
        for filename in os.listdir(self.recordings_dir):
            filepath = os.path.join(self.recordings_dir, filename)
            if os.path.isfile(filepath) and filename.endswith('.mp4'):
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                files.append((filepath, mtime, size))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Delete oldest files until under limit
        deleted_count = 0
        freed_space = 0
        
        for filepath, mtime, size in files:
            if usage['total_bytes'] - freed_space < self.max_storage_bytes * 0.8:
                break  # Keep deleting until 80% of max
            
            try:
                os.remove(filepath)
                freed_space += size
                deleted_count += 1
                print(f"🗑️  Deleted old recording: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"❌ Failed to delete {filepath}: {e}")
        
        if deleted_count > 0:
            print(f"💾 Storage cleanup: Deleted {deleted_count} files, freed {freed_space / (1024*1024):.1f} MB")
        
        return deleted_count
    
    def should_cleanup(self):
        """Check if cleanup is needed"""
        usage = self.get_storage_usage()
        return usage['total_bytes'] >= self.max_storage_bytes
    
    def get_oldest_recording(self):
        """Get path to oldest recording"""
        files = []
        for filename in os.listdir(self.recordings_dir):
            filepath = os.path.join(self.recordings_dir, filename)
            if os.path.isfile(filepath) and filename.endswith('.mp4'):
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime))
        
        if not files:
            return None
        
        files.sort(key=lambda x: x[1])
        return files[0][0]
