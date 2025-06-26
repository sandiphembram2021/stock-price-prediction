from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Stock(Base):
    """Stock information table"""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    sector = Column(String(50), nullable=True)
    market_cap = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_data = relationship("StockData", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")

class StockData(Base):
    """Historical stock price data"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adj_close = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="stock_data")

class Prediction(Base):
    """Stock price predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    prediction_date = Column(DateTime, nullable=False, index=True)
    target_date = Column(DateTime, nullable=False, index=True)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    prediction_type = Column(String(20), default='single')  # 'single', 'future_series'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")

class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    stock_symbol = Column(String(10), nullable=False)
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    epochs = Column(Integer, nullable=True)
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    window_size = Column(Integer, nullable=True)
    model_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPreferences(Base):
    """User preferences and settings"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)  # Can be session ID or user ID
    favorite_stocks = Column(Text, nullable=True)  # JSON string of stock symbols
    default_prediction_days = Column(Integer, default=30)
    chart_preferences = Column(Text, nullable=True)  # JSON string of chart settings
    notification_settings = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
