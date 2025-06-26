#!/usr/bin/env python3
"""
Stock Price Prediction Backend Server
Run this script to start the FastAPI backend server
"""

import uvicorn
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.main import app

if __name__ == "__main__":
    print("🚀 Starting Stock Price Prediction API Server...")
    print("📊 LSTM-based stock price forecasting service")
    print("🌐 API Documentation: http://localhost:8000/api/docs")
    print("🎯 Frontend: http://localhost:8000")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
