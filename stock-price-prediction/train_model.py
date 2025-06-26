#!/usr/bin/env python3
"""
Train the LSTM model using your roni.ipynb approach
Run this script to train a model that can be used by the backend API
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models.stock_predictor import StockPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train the model using the notebook approach"""
    print("🧠 Training LSTM Model for Stock Price Prediction")
    print("=" * 55)
    
    # Initialize predictor with window_size=10 (as in your notebook)
    predictor = StockPredictor(window_size=10)
    
    try:
        print("📊 Starting model training...")
        print("   Symbol: SBI (as in your notebook)")
        print("   Date Range: 2025-02-01 to 2025-06-08")
        print("   Window Size: 10 days")
        print("   Architecture: 4 LSTM layers (50, 60, 80, 120 units)")
        print()
        
        # Train using the same parameters as your notebook
        metrics = predictor.train(
            symbol='SBI',
            start_date='2025-02-01',
            end_date='2025-06-08',
            epochs=50,
            batch_size=16
        )
        
        print("✅ Training completed successfully!")
        print(f"   Final Loss: {metrics['loss']:.6f}")
        print(f"   Training Samples: {metrics['training_samples']}")
        print(f"   Test Samples: {metrics['test_samples']}")
        print()
        
        # Test a prediction
        print("🔮 Testing prediction...")
        try:
            prediction = predictor.predict_single('SBI')
            if prediction:
                print(f"   Next day prediction for SBI: ${prediction:.2f}")
            
            future_predictions = predictor.predict_future('SBI', days=7)
            if future_predictions:
                print(f"   7-day predictions generated: {len(future_predictions)} values")
                print(f"   Sample predictions: ${future_predictions[0]:.2f}, ${future_predictions[1]:.2f}, ${future_predictions[2]:.2f}...")
        except Exception as e:
            print(f"   Warning: Could not test prediction: {e}")
        
        print()
        print("🎉 Model training completed!")
        print("💡 You can now start the backend server and use the API")
        print("🚀 Run: python run_backend.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.exception("Training error details:")
        return 1

if __name__ == "__main__":
    exit(main())
