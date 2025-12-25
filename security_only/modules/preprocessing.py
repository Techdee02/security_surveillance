"""
Image Preprocessing Module
Shared utilities for image processing, enhancement, and input handling
Supports both security and health monitoring systems
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import base64
from io import BytesIO
from PIL import Image


class ImagePreprocessor:
    """
    Shared image preprocessing utilities for both security and health systems
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        
    def resize_frame(self, frame: np.ndarray, 
                     size: Optional[Tuple[int, int]] = None,
                     maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame to target size
        
        Args:
            frame: Input frame (BGR or RGB)
            size: Target size (width, height), uses self.target_size if None
            maintain_aspect: Whether to maintain aspect ratio with padding
            
        Returns:
            Resized frame
        """
        if size is None:
            size = self.target_size
            
        if not maintain_aspect:
            return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        
        # Maintain aspect ratio with padding
        h, w = frame.shape[:2]
        target_w, target_h = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized frame on canvas
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray,
                       method: str = 'standard') -> np.ndarray:
        """
        Normalize image for model input
        
        Args:
            image: Input image
            method: Normalization method
                - 'standard': Scale to [0, 1]
                - 'imagenet': ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                - 'minmax': Min-max scaling
                
        Returns:
            Normalized image
        """
        if method == 'standard':
            return image.astype(np.float32) / 255.0
            
        elif method == 'imagenet':
            # Convert to float and scale to [0, 1]
            img = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            img = (img - mean) / std
            return img
            
        elif method == 'minmax':
            img = image.astype(np.float32)
            img_min = img.min()
            img_max = img.max()
            
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            
            return img
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_clahe(self, frame: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Improves contrast and visibility in low-light conditions
        
        Args:
            frame: Input frame (BGR)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)
        
        # Merge channels
        lab_clahe = cv2.merge([l_channel_clahe, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_gaussian_blur(self, frame: np.ndarray,
                           kernel_size: Tuple[int, int] = (5, 5),
                           sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise
        
        Args:
            frame: Input frame
            kernel_size: Gaussian kernel size (must be odd)
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Blurred frame
        """
        return cv2.GaussianBlur(frame, kernel_size, sigma)
    
    def apply_sharpening(self, frame: np.ndarray,
                        strength: float = 1.0) -> np.ndarray:
        """
        Apply sharpening filter to enhance edges
        
        Args:
            frame: Input frame
            strength: Sharpening strength (0-2)
            
        Returns:
            Sharpened frame
        """
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9 + strength, -1],
                          [-1, -1, -1]]) / (1 + strength)
        
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened
    
    def adjust_brightness_contrast(self, frame: np.ndarray,
                                   brightness: int = 0,
                                   contrast: int = 0) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            frame: Input frame
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (-100 to 100)
            
        Returns:
            Adjusted frame
        """
        # Convert to float for processing
        img = frame.astype(np.float32)
        
        # Apply brightness
        if brightness != 0:
            img = img + brightness
        
        # Apply contrast
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128
        
        # Clip values and convert back
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def auto_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            White-balanced frame
        """
        # Calculate channel means
        b, g, r = cv2.split(frame)
        
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        
        # Calculate average of means
        avg_mean = (b_mean + g_mean + r_mean) / 3
        
        # Calculate scaling factors
        b_scale = avg_mean / b_mean if b_mean > 0 else 1.0
        g_scale = avg_mean / g_mean if g_mean > 0 else 1.0
        r_scale = avg_mean / r_mean if r_mean > 0 else 1.0
        
        # Apply scaling
        b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
        r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
        
        # Merge channels
        balanced = cv2.merge([b, g, r])
        
        return balanced


class ImageInputHandler:
    """
    Handle various image input sources: camera, file upload, URL
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
    
    def load_from_file(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            file_path: Path to image file
            
        Returns:
            Loaded image (BGR) or None if failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"Failed to load image: {file_path}")
                return None
            
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def load_from_base64(self, base64_string: str) -> Optional[np.ndarray]:
        """
        Load image from base64 string
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            Decoded image (BGR) or None if failed
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(img_bytes))
            
            # Convert to numpy array (RGB)
            image_rgb = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_rgb.shape) == 3:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_rgb
            
            return image_bgr
            
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    def load_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load image from raw bytes
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Decoded image (BGR) or None if failed
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Failed to decode image from bytes")
                return None
            
            return image
            
        except Exception as e:
            print(f"Error loading image from bytes: {e}")
            return None
    
    def encode_to_base64(self, frame: np.ndarray, 
                        format: str = 'JPEG',
                        quality: int = 90) -> str:
        """
        Encode frame to base64 string
        
        Args:
            frame: Input frame (BGR)
            format: Image format (JPEG, PNG)
            quality: JPEG quality (0-100)
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Encode to bytes
            buffer = BytesIO()
            if format.upper() == 'JPEG':
                pil_image.save(buffer, format='JPEG', quality=quality)
            else:
                pil_image.save(buffer, format=format)
            
            # Encode to base64
            img_bytes = buffer.getvalue()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            
            # Add data URL prefix
            mime_type = f"image/{format.lower()}"
            data_url = f"data:{mime_type};base64,{base64_string}"
            
            return data_url
            
        except Exception as e:
            print(f"Error encoding image to base64: {e}")
            return ""
    
    def save_to_file(self, frame: np.ndarray, 
                    file_path: Union[str, Path],
                    quality: int = 95) -> bool:
        """
        Save frame to file
        
        Args:
            frame: Input frame (BGR)
            file_path: Output file path
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set compression parameters
            if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif file_path.suffix.lower() == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                params = []
            
            success = cv2.imwrite(str(file_path), frame, params)
            
            if success:
                print(f"Image saved to: {file_path}")
            else:
                print(f"Failed to save image to: {file_path}")
            
            return success
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False


class CameraManager:
    """
    Manage camera sources and capture
    """
    
    def __init__(self):
        self.current_source = None
        self.camera = None
    
    def connect_camera(self, source: Union[int, str]) -> bool:
        """
        Connect to camera source
        
        Args:
            source: Camera source
                - int: USB camera index (0, 1, 2, ...)
                - str: RTSP URL or video file path
                
        Returns:
            True if connected, False otherwise
        """
        try:
            # Release existing camera
            self.disconnect_camera()
            
            # Open new camera
            self.camera = cv2.VideoCapture(source)
            
            if not self.camera.isOpened():
                print(f"Failed to open camera source: {source}")
                return False
            
            self.current_source = source
            print(f"Connected to camera: {source}")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False
    
    def disconnect_camera(self):
        """Disconnect from current camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.current_source = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture frame from camera
        
        Returns:
            Captured frame (BGR) or None if failed
        """
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        
        if not ret:
            return None
        
        return frame
    
    def get_camera_info(self) -> dict:
        """
        Get camera information
        
        Returns:
            Dictionary with camera properties
        """
        if self.camera is None or not self.camera.isOpened():
            return {}
        
        info = {
            'source': self.current_source,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'codec': int(self.camera.get(cv2.CAP_PROP_FOURCC))
        }
        
        return info
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            True if successful, False otherwise
        """
        if self.camera is None or not self.camera.isOpened():
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        return True
    
    def list_available_cameras(self) -> list:
        """
        List available USB cameras
        
        Returns:
            List of available camera indices
        """
        available = []
        
        # Test indices 0-10
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        return available
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect_camera()


# Convenience functions
def preprocess_for_detection(frame: np.ndarray,
                             target_size: Tuple[int, int] = (640, 640),
                             normalize: bool = True,
                             enhance: bool = False) -> np.ndarray:
    """
    Quick preprocessing for object detection
    
    Args:
        frame: Input frame
        target_size: Target size
        normalize: Whether to normalize
        enhance: Whether to apply CLAHE enhancement
        
    Returns:
        Preprocessed frame
    """
    preprocessor = ImagePreprocessor(target_size)
    
    # Resize
    processed = preprocessor.resize_frame(frame, size=target_size)
    
    # Enhance if requested
    if enhance:
        processed = preprocessor.apply_clahe(processed)
    
    # Normalize if requested
    if normalize:
        processed = preprocessor.normalize_image(processed, method='standard')
    
    return processed


def preprocess_for_classification(frame: np.ndarray,
                                  target_size: Tuple[int, int] = (224, 224),
                                  normalize_method: str = 'imagenet') -> np.ndarray:
    """
    Quick preprocessing for image classification
    
    Args:
        frame: Input frame
        target_size: Target size
        normalize_method: Normalization method
        
    Returns:
        Preprocessed frame
    """
    preprocessor = ImagePreprocessor(target_size)
    
    # Resize
    processed = preprocessor.resize_frame(frame, size=target_size, maintain_aspect=False)
    
    # Normalize
    processed = preprocessor.normalize_image(processed, method=normalize_method)
    
    return processed
