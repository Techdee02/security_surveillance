"""
Alert system module for security surveillance
GPIO control for buzzer and LEDs (with simulation mode)
"""
import time
from typing import Dict, Optional
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


class AlertSystem:
    """
    Manages alerts via GPIO (buzzer, LEDs)
    Supports simulation mode for development without hardware
    """
    
    def __init__(self,
                 buzzer_pin: int = 18,
                 led_pin: int = 23,
                 simulate: bool = True):
        """
        Initialize alert system
        
        Args:
            buzzer_pin: GPIO pin for buzzer
            led_pin: GPIO pin for LED indicator
            simulate: If True, simulate GPIO without hardware
        """
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.simulate = simulate
        
        # Alert state
        self.active_alerts = {}
        self.alert_history = []
        self.cooldown_timers = {}
        
        # GPIO setup
        if not simulate:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(buzzer_pin, GPIO.OUT)
                GPIO.setup(led_pin, GPIO.OUT)
                GPIO.output(buzzer_pin, GPIO.LOW)
                GPIO.output(led_pin, GPIO.LOW)
                print(f"✅ GPIO initialized: Buzzer={buzzer_pin}, LED={led_pin}")
            except Exception as e:
                print(f"⚠️  GPIO initialization failed: {e}")
                print("   Falling back to simulation mode")
                self.simulate = True
        else:
            print("🔧 Alert system running in SIMULATION mode")
    
    def trigger_alert(self,
                     zone_name: str,
                     level: AlertLevel = AlertLevel.WARNING,
                     duration_sec: float = 2.0,
                     cooldown_sec: float = 10.0) -> bool:
        """
        Trigger an alert
        
        Args:
            zone_name: Name of zone triggering alert
            level: Alert severity level
            duration_sec: Duration of alert (buzzer/LED on time)
            cooldown_sec: Minimum time before same zone can trigger again
            
        Returns:
            True if alert triggered, False if in cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if zone_name in self.cooldown_timers:
            last_alert_time = self.cooldown_timers[zone_name]
            if current_time - last_alert_time < cooldown_sec:
                remaining = cooldown_sec - (current_time - last_alert_time)
                if self.simulate:
                    print(f"   ⏳ Alert cooldown for '{zone_name}': {remaining:.1f}s remaining")
                return False
        
        # Trigger alert
        alert_data = {
            'zone': zone_name,
            'level': level,
            'timestamp': current_time,
            'duration': duration_sec
        }
        
        self.active_alerts[zone_name] = alert_data
        self.alert_history.append(alert_data)
        self.cooldown_timers[zone_name] = current_time
        
        if self.simulate:
            self._simulate_alert(zone_name, level, duration_sec)
        else:
            self._hardware_alert(level, duration_sec)
        
        return True
    
    def _simulate_alert(self, zone_name: str, level: AlertLevel, duration: float):
        """Simulate alert without hardware"""
        emoji = "🔵" if level == AlertLevel.INFO else "🟡" if level == AlertLevel.WARNING else "🔴"
        
        print(f"\n   {emoji} ALERT TRIGGERED {emoji}")
        print(f"   Zone: {zone_name}")
        print(f"   Level: {level.name}")
        print(f"   Duration: {duration:.1f}s")
        
        if level == AlertLevel.CRITICAL:
            print("   🔊 BUZZER: BEEP BEEP BEEP!")
            print("   💡 LED: FLASHING RED")
        elif level == AlertLevel.WARNING:
            print("   🔊 BUZZER: BEEP!")
            print("   💡 LED: SOLID YELLOW")
        else:
            print("   💡 LED: SOLID BLUE")
    
    def _hardware_alert(self, level: AlertLevel, duration: float):
        """Trigger actual hardware alert"""
        if level == AlertLevel.CRITICAL:
            # Fast beeping + LED flashing
            end_time = time.time() + duration
            while time.time() < end_time:
                self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
                self.GPIO.output(self.led_pin, self.GPIO.HIGH)
                time.sleep(0.2)
                self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
                self.GPIO.output(self.led_pin, self.GPIO.LOW)
                time.sleep(0.2)
        
        elif level == AlertLevel.WARNING:
            # Single beep + LED on
            self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
            self.GPIO.output(self.led_pin, self.GPIO.HIGH)
            time.sleep(duration)
            self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
            self.GPIO.output(self.led_pin, self.GPIO.LOW)
        
        else:  # INFO
            # Just LED on
            self.GPIO.output(self.led_pin, self.GPIO.HIGH)
            time.sleep(duration)
            self.GPIO.output(self.led_pin, self.GPIO.LOW)
    
    def clear_alert(self, zone_name: str):
        """Clear active alert for a zone"""
        if zone_name in self.active_alerts:
            del self.active_alerts[zone_name]
    
    def clear_all_alerts(self):
        """Clear all active alerts"""
        self.active_alerts.clear()
        
        if not self.simulate:
            self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
            self.GPIO.output(self.led_pin, self.GPIO.LOW)
    
    def get_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts': len(self.alert_history),
            'alert_zones': list(set(a['zone'] for a in self.alert_history)),
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        if not self.simulate:
            self.GPIO.cleanup([self.buzzer_pin, self.led_pin])
            print("✅ GPIO cleanup complete")


class ZoneAlertManager:
    """
    Manages alerts for specific zones with custom rules
    Integrates zone monitoring with alert system
    """
    
    def __init__(self, alert_system: AlertSystem):
        """
        Initialize zone alert manager
        
        Args:
            alert_system: AlertSystem instance
        """
        self.alert_system = alert_system
        self.zone_rules = {}
    
    def add_zone_rule(self,
                     zone_name: str,
                     level: AlertLevel = AlertLevel.WARNING,
                     cooldown_sec: float = 10.0):
        """
        Add alert rule for a zone
        
        Args:
            zone_name: Zone to monitor
            level: Alert level for this zone
            cooldown_sec: Cooldown period
        """
        self.zone_rules[zone_name] = {
            'level': level,
            'cooldown': cooldown_sec
        }
    
    def process_zone_detections(self, zone_detections: Dict[str, list]) -> int:
        """
        Process zone detections and trigger appropriate alerts
        
        Args:
            zone_detections: Dict mapping zone names to detection lists
            
        Returns:
            Number of alerts triggered
        """
        alerts_triggered = 0
        
        for zone_name, detections in zone_detections.items():
            if not detections:
                continue
            
            # Get zone-specific rules or use defaults
            rule = self.zone_rules.get(zone_name, {
                'level': AlertLevel.WARNING,
                'cooldown': 10.0
            })
            
            # Trigger alert
            triggered = self.alert_system.trigger_alert(
                zone_name=zone_name,
                level=rule['level'],
                duration_sec=2.0,
                cooldown_sec=rule['cooldown']
            )
            
            if triggered:
                alerts_triggered += 1
        
        return alerts_triggered
    
    def get_stats(self) -> Dict:
        """Get combined statistics"""
        return {
            'zone_rules': len(self.zone_rules),
            'alert_stats': self.alert_system.get_stats()
        }
