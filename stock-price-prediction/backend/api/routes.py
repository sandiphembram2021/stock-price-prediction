from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from ..database import get_db, StockCRUD, StockDataCRUD, PredictionCRUD, ModelMetricsCRUD, UserPreferencesCRUD
from ..models import StockPredictor
from .schemas import *

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Global model instance
predictor = StockPredictor()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    model_status = "loaded" if predictor.is_trained else "not_loaded"
    
    return HealthCheckResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database_status=db_status,
        model_status=model_status
    )

@router.get("/stocks", response_model=StockListResponse)
async def get_stocks(db: Session = Depends(get_db)):
    """Get all stocks in the database"""
    try:
        stocks = StockCRUD.get_all_stocks(db)
        return StockListResponse(stocks=stocks, total=len(stocks))
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stocks")

@router.post("/predict/single", response_model=SinglePredictionResponse)
async def predict_single_price(request: PredictionRequest, db: Session = Depends(get_db)):
    """Predict next day's stock price"""
    try:
        stock = StockCRUD.get_or_create_stock(db, request.symbol)
        predicted_price = predictor.predict_single(request.symbol)
        
        if predicted_price is None:
            raise HTTPException(status_code=500, detail="Failed to generate prediction")
        
        latest_data = StockDataCRUD.get_latest_stock_data(db, stock.id)
        current_price = latest_data.close_price if latest_data else None
        
        target_date = datetime.utcnow() + timedelta(days=1)
        prediction = PredictionCRUD.create_prediction(
            db, stock.id, target_date, predicted_price,
            model_version="LSTM_v1", prediction_type="single"
        )
        
        return SinglePredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            prediction_date=prediction.prediction_date,
            target_date=target_date,
            confidence_score=prediction.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Error predicting price for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/future", response_model=FuturePredictionsResponse)
async def predict_future_prices(request: PredictionRequest, db: Session = Depends(get_db)):
    """Predict future stock prices for multiple days"""
    try:
        stock = StockCRUD.get_or_create_stock(db, request.symbol)
        future_prices = predictor.predict_future(request.symbol, request.days)
        
        if not future_prices:
            raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        predictions = []
        base_date = datetime.utcnow()
        
        for i, price in enumerate(future_prices):
            target_date = base_date + timedelta(days=i+1)
            predictions.append({
                "date": target_date.isoformat(),
                "price": float(price)
            })
            
            PredictionCRUD.create_prediction(
                db, stock.id, target_date, float(price),
                model_version="LSTM_v1", prediction_type="future_series"
            )
        
        return FuturePredictionsResponse(
            symbol=request.symbol,
            predictions=predictions,
            prediction_date=base_date,
            days_predicted=len(future_prices)
        )
        
    except Exception as e:
        logger.error(f"Error predicting future prices for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Train the LSTM model on historical data"""
    try:
        stock = StockCRUD.get_or_create_stock(db, request.symbol)
        
        # Train model in background
        def train_background():
            try:
                metrics = predictor.train(
                    request.symbol, 
                    request.start_date, 
                    request.end_date,
                    request.epochs, 
                    request.batch_size
                )
                
                # Save metrics to database
                ModelMetricsCRUD.save_model_metrics(
                    db, "LSTM_v1", request.symbol,
                    datetime.fromisoformat(request.start_date),
                    datetime.fromisoformat(request.end_date),
                    metrics
                )
                
            except Exception as e:
                logger.error(f"Background training failed: {e}")
        
        background_tasks.add_task(train_background)
        
        return TrainingResponse(
            success=True,
            message=f"Training started for {request.symbol}",
            metrics=None,
            model_path=predictor.model_path
        )
        
    except Exception as e:
        logger.error(f"Error starting training for {request.symbol}: {e}")
        return TrainingResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            metrics=None,
            model_path=None
        )

@router.get("/model/info")
async def get_model_info():
    """Get model information and status"""
    try:
        model_info = predictor.get_model_summary()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.post("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(request: UserPreferencesRequest, db: Session = Depends(get_db)):
    """Update user preferences"""
    try:
        prefs = UserPreferencesCRUD.update_favorite_stocks(db, request.user_id, request.favorite_stocks)
        
        favorite_stocks = UserPreferencesCRUD.get_favorite_stocks(db, request.user_id)
        
        return UserPreferencesResponse(
            user_id=request.user_id,
            favorite_stocks=favorite_stocks,
            default_prediction_days=prefs.default_prediction_days,
            created_at=prefs.created_at,
            updated_at=prefs.updated_at
        )
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")
