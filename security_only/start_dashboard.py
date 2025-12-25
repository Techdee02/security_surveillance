"""
Simple dashboard launcher - Security Surveillance Only
"""
import uvicorn
from dashboard.app import app

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SECURITY SURVEILLANCE DASHBOARD")
    print("=" * 70)
    print("Dashboard URL: http://localhost:8080")
    print("API Docs: http://localhost:8080/api/docs")
    print("=" * 70)
    print("\nNote: Use main.py to start the surveillance system")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
