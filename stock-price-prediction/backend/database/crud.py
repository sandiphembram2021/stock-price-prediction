from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from .models import Stock, StockData, Prediction, ModelMetrics, UserPreferences
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class StockCRUD:
    """CRUD operations for Stock model"""
    
    @staticmethod
    def create_stock(db: Session, symbol: str, name: str = None, sector: str = None) -> Stock:
        """Create a new stock entry"""
        stock = Stock(symbol=symbol.upper(), name=name, sector=sector)
        db.add(stock)
        db.commit()
        db.refresh(stock)
        return stock
    
    @staticmethod
    def get_stock_by_symbol(db: Session, symbol: str) -> Optional[Stock]:
        """Get stock by symbol"""
        return db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    
    @staticmethod
    def get_or_create_stock(db: Session, symbol: str, name: str = None) -> Stock:
        """Get existing stock or create new one"""
        stock = StockCRUD.get_stock_by_symbol(db, symbol)
        if not stock:
            stock = StockCRUD.create_stock(db, symbol, name)
        return stock
    
    @staticmethod
    def get_all_stocks(db: Session) -> List[Stock]:
        """Get all stocks"""
        return db.query(Stock).all()

class StockDataCRUD:
    """CRUD operations for StockData model"""
    
    @staticmethod
    def add_stock_data(db: Session, stock_id: int, date: datetime, 
                      open_price: float, high_price: float, low_price: float,
                      close_price: float, volume: int, adj_close: float = None) -> StockData:
        """Add stock price data"""
        stock_data = StockData(
            stock_id=stock_id,
            date=date,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            adj_close=adj_close
        )
        db.add(stock_data)
        db.commit()
        db.refresh(stock_data)
        return stock_data
    
    @staticmethod
    def get_stock_data(db: Session, stock_id: int, start_date: datetime = None, 
                      end_date: datetime = None, limit: int = None) -> List[StockData]:
        """Get stock data with optional date filtering"""
        query = db.query(StockData).filter(StockData.stock_id == stock_id)
        
        if start_date:
            query = query.filter(StockData.date >= start_date)
        if end_date:
            query = query.filter(StockData.date <= end_date)
        
        query = query.order_by(desc(StockData.date))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def get_latest_stock_data(db: Session, stock_id: int) -> Optional[StockData]:
        """Get the most recent stock data entry"""
        return db.query(StockData).filter(StockData.stock_id == stock_id)\
                 .order_by(desc(StockData.date)).first()

class PredictionCRUD:
    """CRUD operations for Prediction model"""
    
    @staticmethod
    def create_prediction(db: Session, stock_id: int, target_date: datetime,
                         predicted_price: float, confidence_score: float = None,
                         model_version: str = None, prediction_type: str = 'single') -> Prediction:
        """Create a new prediction"""
        prediction = Prediction(
            stock_id=stock_id,
            prediction_date=datetime.utcnow(),
            target_date=target_date,
            predicted_price=predicted_price,
            confidence_score=confidence_score,
            model_version=model_version,
            prediction_type=prediction_type
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    
    @staticmethod
    def get_predictions(db: Session, stock_id: int, start_date: datetime = None,
                       end_date: datetime = None, limit: int = None) -> List[Prediction]:
        """Get predictions with optional filtering"""
        query = db.query(Prediction).filter(Prediction.stock_id == stock_id)
        
        if start_date:
            query = query.filter(Prediction.target_date >= start_date)
        if end_date:
            query = query.filter(Prediction.target_date <= end_date)
        
        query = query.order_by(desc(Prediction.prediction_date))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()

class ModelMetricsCRUD:
    """CRUD operations for ModelMetrics model"""
    
    @staticmethod
    def save_model_metrics(db: Session, model_name: str, stock_symbol: str,
                          training_start_date: datetime, training_end_date: datetime,
                          metrics: Dict[str, Any]) -> ModelMetrics:
        """Save model training metrics"""
        model_metrics = ModelMetrics(
            model_name=model_name,
            stock_symbol=stock_symbol.upper(),
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            train_loss=metrics.get('loss'),
            val_loss=metrics.get('val_loss'),
            epochs=metrics.get('epochs'),
            training_samples=metrics.get('training_samples'),
            test_samples=metrics.get('test_samples'),
            window_size=metrics.get('window_size'),
            model_path=metrics.get('model_path')
        )
        db.add(model_metrics)
        db.commit()
        db.refresh(model_metrics)
        return model_metrics

class UserPreferencesCRUD:
    """CRUD operations for UserPreferences model"""
    
    @staticmethod
    def get_or_create_preferences(db: Session, user_id: str) -> UserPreferences:
        """Get or create user preferences"""
        prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        if not prefs:
            prefs = UserPreferences(user_id=user_id)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
        return prefs
    
    @staticmethod
    def update_favorite_stocks(db: Session, user_id: str, stocks: List[str]):
        """Update user's favorite stocks"""
        prefs = UserPreferencesCRUD.get_or_create_preferences(db, user_id)
        prefs.favorite_stocks = json.dumps(stocks)
        prefs.updated_at = datetime.utcnow()
        db.commit()
        return prefs
    
    @staticmethod
    def get_favorite_stocks(db: Session, user_id: str) -> List[str]:
        """Get user's favorite stocks"""
        prefs = UserPreferencesCRUD.get_or_create_preferences(db, user_id)
        if prefs.favorite_stocks:
            return json.loads(prefs.favorite_stocks)
        return []
