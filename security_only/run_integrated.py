#!/usr/bin/env python3
"""
Integrated launcher for Security Surveillance System
Runs both surveillance and dashboard in a single process
"""
import sys
import threading
import time
from pathlib import Path
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import SurveillanceSystem
from dashboard.app import app

def run_surveillance(system):
    """Run surveillance system in background thread"""
    try:
        print("🎥 Starting surveillance system...")
        system.start()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping surveillance...")
        system.stop()
    except Exception as e:
        print(f"❌ Surveillance error: {e}")

def main():
    print("=" * 70)
    print("🚀 INTEGRATED SECURITY SURVEILLANCE SYSTEM")
    print("=" * 70)
    print("Initializing system...")
    print()
    
    # Initialize surveillance system
    config_path = Path(__file__).parent / "config" / "config.yaml"
    system = SurveillanceSystem(str(config_path))
    
    # Set reference in dashboard app state
    app.state.app_state.surveillance_system = system
    app.state.app_state.security_db = system.database
    app.state.app_state.camera = system.camera
    
    print("✅ System initialized")
    print()
    
    # Start surveillance in background thread
    surveillance_thread = threading.Thread(target=run_surveillance, args=(system,), daemon=True)
    surveillance_thread.start()
    
    # Give surveillance a moment to start
    time.sleep(2)
    
    # Start dashboard
    print("=" * 70)
    print("🌐 DASHBOARD")
    print("=" * 70)
    print("🔗 Dashboard URL: http://localhost:8080")
    print("📚 API Docs: http://localhost:8080/api/docs")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Run uvicorn in main thread
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        system.stop()
        print("✅ Stopped successfully")

if __name__ == "__main__":
    main()
