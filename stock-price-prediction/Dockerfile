# Stock Price Prediction System Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY run_backend.py /app/
COPY .env.example /app/.env

# Create necessary directories
RUN mkdir -p /app/backend/database \
    && mkdir -p /app/backend/models/saved \
    && mkdir -p /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///app/backend/database/stock_prediction.db
ENV MODEL_PATH=/app/backend/models/saved/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["python", "run_backend.py"]
