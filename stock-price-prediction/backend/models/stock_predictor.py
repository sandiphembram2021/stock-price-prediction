import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    """
    LSTM-based stock price prediction model - Based on roni.ipynb implementation
    """

    def __init__(self, window_size: int = 10, model_path: str = None):
        # Using window_size=10 as in your original model
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_path = model_path or 'backend/models/saved/lstm_stock_model.keras'
        self.is_trained = False
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        """
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing
        """
        # Remove NaN values
        data = data.dropna()
        
        # Split into train and test
        train_size = int(len(data) * train_ratio)
        data_train = pd.DataFrame(data['Close'][:train_size])
        data_test = pd.DataFrame(data['Close'][train_size:])
        
        # Scale the training data
        data_train_scaled = self.scaler.fit_transform(data_train)
        
        # Create sequences for training
        x_train, y_train = self._create_sequences(data_train_scaled, self.window_size)
        
        # Prepare test data
        past_days = data_train.tail(self.window_size)
        data_test_combined = pd.concat([past_days, data_test], ignore_index=True)
        data_test_scaled = self.scaler.transform(data_test_combined)
        
        x_test, y_test = self._create_sequences(data_test_scaled, self.window_size)
        
        return x_train, y_train, x_test, y_test
    
    def _create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        """
        x, y = [], []
        
        for i in range(window_size, len(data)):
            x.append(data[i-window_size:i])
            y.append(data[i, 0])
        
        x = np.array(x)
        y = np.array(y)
        
        # Reshape for LSTM input
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
            
        return x, y
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture - Based on roni.ipynb
        """
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))

        # Third LSTM layer
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))

        # Fourth LSTM layer
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, symbol: str, start_date: str, end_date: str, epochs: int = 50, batch_size: int = 16) -> dict:
        """
        Train the LSTM model
        """
        logger.info(f"Starting training for {symbol}")
        
        # Fetch and prepare data
        data = self.fetch_stock_data(symbol, start_date, end_date)
        x_train, y_train, x_test, y_test = self.prepare_data(data)
        
        if len(x_train) == 0:
            raise ValueError("Not enough data to create training sequences")
        
        # Build model
        self.model = self.build_model((x_train.shape[1], x_train.shape[2]))
        
        # Train model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        # Save scaler for future use
        try:
            import joblib
            scaler_path = self.model_path.replace('.keras', '_scaler.pkl').replace('lstm_stock_model', 'scaler')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        except Exception as e:
            logger.warning(f"Could not save scaler: {e}")

        self.is_trained = True

        logger.info(f"Model trained and saved to {self.model_path}")

        return {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
            'epochs': epochs,
            'training_samples': len(x_train),
            'test_samples': len(x_test)
        }
    
    def load_model(self) -> bool:
        """
        Load pre-trained model and scaler
        """
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)

                # Try to load the scaler if it exists
                scaler_path = self.model_path.replace('.keras', '_scaler.pkl').replace('lstm_stock_model', 'scaler')
                if os.path.exists(scaler_path):
                    import joblib
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Scaler loaded from {scaler_path}")

                self.is_trained = True
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_future(self, symbol: str, days: int = 30) -> List[float]:
        """
        Predict future stock prices - Based on roni.ipynb implementation
        """
        if not self.is_trained:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")

        # Get recent data (last 365 days to ensure we have enough for window)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        data = self.fetch_stock_data(symbol, start_date, end_date)

        # Get last window_size days and scale them
        last_days = data['Close'].tail(self.window_size).values
        last_days_scaled = self.scaler.transform(last_days.reshape(-1, 1))

        # Predict future prices using the same approach as in roni.ipynb
        future_predictions = []
        current_input = last_days_scaled

        for _ in range(days):
            # Reshape for LSTM input: (batch_size, timesteps, features)
            prediction = self.model.predict(current_input.reshape(1, self.window_size, 1), verbose=0)
            future_predictions.append(prediction[0, 0])

            # Update input for next prediction (sliding window)
            current_input = np.append(current_input[1:], prediction, axis=0)

        # Inverse transform to get actual price values
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        return future_predictions.flatten().tolist()
    
    def predict_single(self, symbol: str) -> float:
        """
        Predict next day's price
        """
        predictions = self.predict_future(symbol, days=1)
        return predictions[0] if predictions else None
    
    def get_model_summary(self) -> dict:
        """
        Get model information
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "window_size": self.window_size,
            "model_path": self.model_path,
            "is_trained": self.is_trained,
            "total_params": self.model.count_params(),
            "layers": len(self.model.layers)
        }
