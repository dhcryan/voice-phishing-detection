#!/bin/bash
# Run both API server and Streamlit frontend

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start API server in background
echo "🚀 Starting API server on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit
echo "🎨 Starting Streamlit on port 8501..."
streamlit run frontend/app.py --server.port 8501 &
STREAMLIT_PID=$!

echo ""
echo "✅ Services started!"
echo "   API Server: http://localhost:8000"
echo "   Streamlit:  http://localhost:8501"
echo "   API Docs:   http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "👋 Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
