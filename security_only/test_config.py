"""
Test Configuration Loader
Tests for config.yaml loading and access
"""
import unittest
from pathlib import Path
from modules.config_loader import ConfigLoader, get_config, get_value


class TestConfigLoader(unittest.TestCase):
    """Test ConfigLoader class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.config = ConfigLoader("config/config.yaml")
    
    def test_load_config(self):
        """Test configuration loading"""
        self.assertIsNotNone(self.config.config)
        self.assertIsInstance(self.config.config, dict)
    
    def test_get_health_model_config(self):
        """Test getting health model configuration"""
        model_config = self.config.get_health_model_config()
        
        self.assertIsInstance(model_config, dict)
        self.assertIn('keras_path', model_config)
        self.assertIn('tflite_path', model_config)
        self.assertIn('classes_path', model_config)
        self.assertIn('use_tflite', model_config)
        self.assertIn('input_size', model_config)
    
    def test_get_health_detection_config(self):
        """Test getting health detection configuration"""
        detection_config = self.config.get_health_detection_config()
        
        self.assertIsInstance(detection_config, dict)
        self.assertIn('interval', detection_config)
        self.assertIn('confidence_threshold', detection_config)
        self.assertIn('save_images', detection_config)
    
    def test_get_with_dot_notation(self):
        """Test getting values with dot notation"""
        camera_source = self.config.get('camera.source')
        self.assertIsNotNone(camera_source)
        
        health_interval = self.config.get('health_system.detection.interval')
        self.assertIsNotNone(health_interval)


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIG LOADER TESTS")
    print("=" * 70)
    unittest.main(verbosity=2)
