from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging
import os

from .api import router
from .database import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction service with historical data and future forecasting",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Stock Prediction"])

# Serve static files (frontend)
if os.path.exists("frontend/dist"):
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend application"""
        return FileResponse("frontend/dist/index.html")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Stock Price Prediction API...")
    
    # Initialize database
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Stock Price Prediction API...")

@app.get("/api/health")
async def root_health():
    """Root health check"""
    return {
        "status": "healthy",
        "message": "Stock Price Prediction API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
