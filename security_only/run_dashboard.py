"""
Launch surveillance system with web dashboard
"""
import sys
import threading
import time
import uvicorn

# Import surveillance system
from main import SurveillanceSystem

# Import dashboard app
from dashboard.app import app


def run_surveillance():
    """Run surveillance system in separate thread"""
    print("\n🔒 Starting Surveillance System with Video...")
    
    # Create system and override config to use video
    surveillance_system = SurveillanceSystem(config_path='config/config.yaml')
    
    # Set up reference for dashboard
    app.state.app_state.surveillance_system = surveillance_system
    
    # Start surveillance
    try:
        surveillance_system.start()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping surveillance...")
        surveillance_system.stop()
    except Exception as e:
        print(f"⚠️  Surveillance system error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 SECURITY SURVEILLANCE DASHBOARD")
    print("=" * 70)
    print("Using video: data/test_videos/4417762-hd_1080_1920_30fps.mp4")
    print()
    
    # Start surveillance in background thread
    surveillance_thread = threading.Thread(target=run_surveillance, daemon=True)
    surveillance_thread.start()
    
    # Give system time to initialize
    time.sleep(2)
    
    print("\n" + "=" * 70)
    print("🌐 DASHBOARD READY")
    print("=" * 70)
    print("URL: http://localhost:8080")
    print("Press CTRL+C to stop")
    print("=" * 70 + "\n")
    
    # Run dashboard
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="warning"
        )
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")


if __name__ == "__main__":
    main()
