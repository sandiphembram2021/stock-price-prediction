#!/usr/bin/env python3
"""
Setup script for Stock Price Prediction System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "backend/database",
        "backend/models/saved",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully!")

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def setup_environment():
    """Setup environment file"""
    print("⚙️ Setting up environment...")
    
    env_content = """# Stock Price Prediction Environment Configuration
# Database
DATABASE_URL=sqlite:///backend/database/stock_prediction.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Settings
MODEL_PATH=backend/models/saved/
WINDOW_SIZE=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Environment file created!")

def main():
    """Main setup function"""
    print("🚀 Stock Price Prediction System Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Setup environment
    setup_environment()

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Train your model: python train_model.py")
    print("2. Run the server: python run_backend.py")
    print("3. Open browser: http://localhost:8000")
    print("4. Start predicting stock prices!")
    print("\n📚 Documentation: README.md")
    print("🔧 API Docs: http://localhost:8000/api/docs")
    print("📓 Your model: model/roni.ipynb")

if __name__ == "__main__":
    main()
