# 📈 Stock Price Prediction System

A full-stack AI-powered application for predicting stock prices using LSTM neural networks. This system provides both single-day and multi-day stock price forecasting with an intuitive web interface.

## 🌟 Features

- **LSTM Neural Network**: Deep learning model for accurate stock price prediction
- **Real-time Data**: Fetches live stock data from Yahoo Finance
- **Interactive Charts**: Beautiful visualizations using Chart.js
- **Multiple Prediction Types**: Single day and future series predictions
- **Model Training**: Train custom models with historical data
- **Lightweight Database**: SQLite for storing predictions and metrics
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Modern Frontend**: Responsive web interface with real-time updates

## 🏗️ Architecture

```
stock-price-prediction/
├── backend/                 # FastAPI Backend
│   ├── api/                # API routes and schemas
│   ├── database/           # SQLite database models and CRUD
│   ├── models/             # LSTM model implementation
│   └── main.py            # FastAPI application
├── frontend/               # Web Frontend
│   ├── index.html         # Main HTML file
│   ├── styles.css         # CSS styling
│   ├── app.js            # JavaScript application
│   └── config.js         # Configuration
├── model/                  # Original Jupyter notebook
└── run_backend.py         # Server startup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-price-prediction
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Start the backend server**
   ```bash
   python run_backend.py
   ```

4. **Open the application**
   - Frontend: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

## 📊 Usage

### Making Predictions

1. **Single Day Prediction**
   - Enter a stock symbol (e.g., AAPL, GOOGL)
   - Select "1 Day" from the prediction dropdown
   - Click "Predict" button

2. **Future Series Prediction**
   - Enter a stock symbol
   - Select desired prediction period (1 week to 6 months)
   - Click "Predict" button

### Training Custom Models

1. Click the "Train Model" button
2. Set training parameters:
   - Start Date: Beginning of training data
   - End Date: End of training data
   - Epochs: Number of training iterations
   - Batch Size: Training batch size
3. Click "Start Training"

### Viewing Results

- **Prediction Summary**: Current vs predicted prices with change indicators
- **Interactive Chart**: Visual representation of predictions
- **History Table**: Past predictions with accuracy metrics

## 🔧 API Endpoints

### Health & Status
- `GET /api/health` - API health check
- `GET /api/v1/health` - Detailed health status

### Predictions
- `POST /api/v1/predict/single` - Single day prediction
- `POST /api/v1/predict/future` - Multi-day predictions
- `GET /api/v1/predictions/{symbol}` - Prediction history

### Model Management
- `POST /api/v1/train` - Train model
- `GET /api/v1/model/info` - Model information

### Data Management
- `GET /api/v1/stocks` - List all stocks
- `GET /api/v1/stocks/{symbol}` - Stock information
- `GET /api/v1/stocks/{symbol}/data` - Historical data

## 🗄️ Database Schema

The application uses SQLite with the following main tables:

- **stocks**: Stock information and metadata
- **stock_data**: Historical price data
- **predictions**: Model predictions and results
- **model_metrics**: Training performance metrics
- **user_preferences**: User settings and favorites

## 🧠 Model Details

### LSTM Architecture
- **Input Layer**: 100-day price sequences
- **LSTM Layers**: 4 layers with dropout (50, 60, 80, 120 units)
- **Output Layer**: Single price prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

### Data Processing
- **Normalization**: MinMax scaling (0-1 range)
- **Sequence Length**: 100 days
- **Train/Test Split**: 80/20
- **Features**: Close price (primary)

## 🎨 Frontend Features

### User Interface
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Live status indicators
- **Interactive Charts**: Zoom, pan, hover details
- **Toast Notifications**: User feedback system

### Visualization
- **Chart.js Integration**: Professional charts
- **Color Coding**: Positive/negative indicators
- **Multiple Views**: Single and series predictions
- **Historical Comparison**: Accuracy tracking

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Train Your Model** (Using your roni.ipynb approach)
   ```bash
   python train_model.py
   ```

3. **Run the Application**
   ```bash
   python run_backend.py
   ```

4. **Access the Interface**
   - Web App: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs

## 📓 Using Your Jupyter Notebook Model

This system is built to work with your existing `model/roni.ipynb` implementation:

- **Same Architecture**: 4 LSTM layers (50, 60, 80, 120 units) with dropout
- **Same Parameters**: Window size of 10 days, Adam optimizer
- **Same Data**: Uses SBI stock data from 2025-02-01 to 2025-06-08
- **Compatible**: Your trained model can be directly used by the backend API

## 🛠️ Development

### Project Structure
- `backend/`: FastAPI server and ML models
- `frontend/`: Web interface
- `model/`: Original Jupyter notebook research

### Key Technologies
- **Backend**: FastAPI, SQLAlchemy, TensorFlow
- **Frontend**: Vanilla JavaScript, Chart.js
- **Database**: SQLite
- **ML**: LSTM Neural Networks

## 📈 Performance & Accuracy

The LSTM model provides competitive accuracy for stock price prediction, though results vary by market conditions and stock volatility. The system is designed for educational and research purposes.

---

**Built with ❤️ using FastAPI, TensorFlow, and modern web technologies**
