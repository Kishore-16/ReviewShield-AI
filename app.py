"""
Root Entry Point for AI Review Detector API Microservice
File: app.py
"""

from ecommerce_app.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
