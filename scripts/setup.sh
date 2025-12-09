#!/bin/bash
# Setup script for Voice Phishing Detection System

set -e

echo "🔒 AI Voice Phishing Detection System - Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/audio data/legal_docs data/vectors
mkdir -p models/checkpoints models/configs
mkdir -p logs

# Copy .env.example to .env if not exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys!"
fi

# Download sample data (optional)
# echo "📥 Downloading sample data..."
# python scripts/download_data.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI and Langfuse API keys"
echo "2. Run the API server: uvicorn src.api.main:app --reload"
echo "3. Run the Streamlit app: streamlit run frontend/app.py"
echo ""
