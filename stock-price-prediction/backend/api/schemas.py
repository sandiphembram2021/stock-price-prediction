from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class PredictionType(str, Enum):
    SINGLE = "single"
    FUTURE_SERIES = "future_series"

# Request schemas
class StockSymbolRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, GOOGL)")

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    days: int = Field(default=1, ge=1, le=365, description="Number of days to predict")

class TrainingRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to train on")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training end date (YYYY-MM-DD)")
    epochs: int = Field(default=50, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Training batch size")

class UserPreferencesRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    favorite_stocks: List[str] = Field(default=[], description="List of favorite stock symbols")
    default_prediction_days: int = Field(default=30, ge=1, le=365)

# Response schemas
class StockInfo(BaseModel):
    id: int
    symbol: str
    name: Optional[str]
    sector: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class StockDataPoint(BaseModel):
    id: int
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adj_close: Optional[float]

    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    id: int
    stock_symbol: str
    prediction_date: datetime
    target_date: datetime
    predicted_price: float
    actual_price: Optional[float]
    confidence_score: Optional[float]
    model_version: Optional[str]
    prediction_type: str

    class Config:
        from_attributes = True

class SinglePredictionResponse(BaseModel):
    symbol: str
    current_price: Optional[float]
    predicted_price: float
    prediction_date: datetime
    target_date: datetime
    confidence_score: Optional[float]

class FuturePredictionsResponse(BaseModel):
    symbol: str
    predictions: List[Dict[str, Any]]  # List of {date: str, price: float}
    prediction_date: datetime
    days_predicted: int

class TrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]]
    model_path: Optional[str]

class ModelMetricsResponse(BaseModel):
    id: int
    model_name: str
    stock_symbol: str
    training_start_date: datetime
    training_end_date: datetime
    train_loss: Optional[float]
    val_loss: Optional[float]
    epochs: Optional[int]
    training_samples: Optional[int]
    test_samples: Optional[int]
    window_size: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class StockDataResponse(BaseModel):
    symbol: str
    data: List[StockDataPoint]
    total_records: int
    date_range: Dict[str, str]  # start_date, end_date

class UserPreferencesResponse(BaseModel):
    user_id: str
    favorite_stocks: List[str]
    default_prediction_days: int
    created_at: datetime
    updated_at: datetime

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_status: str
    model_status: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime

class StockListResponse(BaseModel):
    stocks: List[StockInfo]
    total: int

class PredictionListResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    symbol: str
