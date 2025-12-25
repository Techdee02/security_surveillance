"""
Configuration Loader Module
Loads and provides access to system configuration from config.yaml
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage system configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_health_model_config(self) -> Dict[str, Any]:
        return self.get('health_system.model', {})
    
    def get_health_detection_config(self) -> Dict[str, Any]:
        return self.get('health_system.detection', {})
    
    def get_health_preprocessing_config(self) -> Dict[str, Any]:
        return self.get('health_system.preprocessing', {})
    
    def get_health_upload_config(self) -> Dict[str, Any]:
        return self.get('health_system.upload', {})


_config_instance = None

def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance

def get_value(key_path: str, default: Any = None) -> Any:
    config = get_config()
    return config.get(key_path, default)
