#!/usr/bin/env python3
"""
Simple server to run the stock prediction system without training first
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend application"""
    return FileResponse("frontend/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database_status": "healthy",
        "model_status": "not_loaded"
    }

@app.get("/api/v1/health")
async def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database_status": "healthy",
        "model_status": "not_loaded"
    }

@app.post("/api/v1/predict/single")
async def predict_single():
    """Mock single prediction endpoint"""
    return {
        "symbol": "AAPL",
        "current_price": 150.25,
        "predicted_price": 152.30,
        "prediction_date": datetime.utcnow().isoformat(),
        "target_date": (datetime.utcnow()).isoformat(),
        "confidence_score": 0.85
    }

@app.post("/api/v1/predict/future")
async def predict_future():
    """Mock future predictions endpoint"""
    base_price = 150.0
    predictions = []
    
    for i in range(30):
        price = base_price + (i * 0.5) + ((-1) ** i * 2)  # Simple mock pattern
        predictions.append({
            "date": (datetime.utcnow()).isoformat(),
            "price": round(price, 2)
        })
    
    return {
        "symbol": "AAPL",
        "predictions": predictions,
        "prediction_date": datetime.utcnow().isoformat(),
        "days_predicted": 30
    }

@app.get("/api/v1/stocks")
async def get_stocks():
    """Mock stocks endpoint"""
    return {
        "stocks": [
            {
                "id": 1,
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            },
            {
                "id": 2,
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "sector": "Technology",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        ],
        "total": 2
    }

if __name__ == "__main__":
    print("🚀 Starting Simple Stock Price Prediction Server...")
    print("📊 Mock API for demonstration")
    print("🌐 Frontend: http://localhost:8000")
    print("🔧 API Docs: http://localhost:8000/docs")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
