#!/usr/bin/env python3
"""
Convert the Jupyter notebook model (roni.ipynb) to work with the backend system
This script extracts the trained model and makes it compatible with the backend API
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotebookModelConverter:
    """
    Converts the roni.ipynb model to backend-compatible format
    """
    
    def __init__(self):
        self.window_size = 10  # As used in the notebook
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance - same as notebook"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data exactly as in the notebook"""
        # Remove NaN values
        data = data.dropna()
        
        # Split into train and test (80/20 split)
        train_size = int(len(data) * 0.80)
        data_train = pd.DataFrame(data['Close'][:train_size])
        data_test = pd.DataFrame(data['Close'][train_size:])
        
        # Scale the training data
        data_train_scale = self.scaler.fit_transform(data_train)
        
        # Create sequences using the notebook's approach
        x = []
        y = []
        
        for i in range(self.window_size, data_train_scale.shape[0]):
            x.append(data_train_scale[i-self.window_size:i])
            y.append(data_train_scale[i, 0])
        
        x = np.array(x)
        y = np.array(y)
        
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        
        logger.info(f"Training data shape: x={x.shape}, y={y.shape}")
        return x, y, data_train, data_test
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """Build the exact model from the notebook"""
        model = Sequential()
        
        # LSTM layers exactly as in notebook
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, symbol: str = 'SBI', start_date: str = '2025-02-01', 
                   end_date: str = '2025-06-08', epochs: int = 50, batch_size: int = 16):
        """Train the model using the notebook's approach"""
        logger.info(f"Training model for {symbol} from {start_date} to {end_date}")
        
        # Fetch data
        data = self.fetch_stock_data(symbol, start_date, end_date)
        
        # Prepare training data
        x_train, y_train, data_train, data_test = self.prepare_training_data(data)
        
        if len(x_train) == 0:
            raise ValueError("Not enough data to create training sequences")
        
        # Build model
        self.model = self.build_model((x_train.shape[1], x_train.shape[2]))
        
        logger.info("Starting model training...")
        
        # Train model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save model
        model_dir = 'backend/models/saved'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lstm_stock_model.keras')
        
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler for future use
        import joblib
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return {
            'loss': history.history['loss'][-1],
            'epochs': epochs,
            'training_samples': len(x_train),
            'model_path': model_path,
            'scaler_path': scaler_path
        }
    
    def test_prediction(self, symbol: str = 'SBI', days: int = 30):
        """Test the model prediction"""
        if not self.model:
            logger.error("Model not trained yet!")
            return None
        
        # Get recent data for prediction
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        data = self.fetch_stock_data(symbol, start_date, end_date)
        
        # Get last window_size days
        last_days = data['Close'].tail(self.window_size).values
        last_days_scaled = self.scaler.transform(last_days.reshape(-1, 1))
        
        # Predict future prices
        future_predictions = []
        current_input = last_days_scaled
        
        for _ in range(days):
            prediction = self.model.predict(current_input.reshape(1, self.window_size, 1), verbose=0)
            future_predictions.append(prediction[0, 0])
            current_input = np.append(current_input[1:], prediction, axis=0)
        
        # Inverse transform predictions
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        logger.info(f"Generated {days} day predictions for {symbol}")
        return future_predictions.flatten().tolist()

def main():
    """Main function to convert and train the model"""
    print("🧠 Converting Jupyter Notebook Model to Backend Format")
    print("=" * 60)
    
    converter = NotebookModelConverter()
    
    try:
        # Train the model using the same parameters as the notebook
        print("📊 Training model with notebook parameters...")
        metrics = converter.train_model(
            symbol='SBI',  # Same as in notebook
            start_date='2025-02-01',  # Same as in notebook
            end_date='2025-06-08',    # Same as in notebook
            epochs=50,
            batch_size=16
        )
        
        print(f"✅ Training completed!")
        print(f"   Final Loss: {metrics['loss']:.6f}")
        print(f"   Training Samples: {metrics['training_samples']}")
        print(f"   Model Path: {metrics['model_path']}")
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        predictions = converter.test_prediction('SBI', days=10)
        
        if predictions:
            print(f"✅ Generated 10-day predictions:")
            for i, price in enumerate(predictions[:5], 1):
                print(f"   Day {i}: ${price:.2f}")
            print(f"   ... and {len(predictions)-5} more days")
        
        print("\n🎉 Model conversion completed successfully!")
        print("💡 You can now use this model with the backend API")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
