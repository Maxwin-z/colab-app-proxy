#!/bin/bash

# Function to handle script termination
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p)
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Default to starting ngrok
START_NGROK=true

# Parse arguments
for arg in "$@"
do
    if [ "$arg" == "--local" ]; then
        START_NGROK=false
    fi
done

# Start App Proxy (Current Directory)
echo "Starting App Proxy on port 8000..."
(uv sync && source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000) &

# Start Ngrok
if [ "$START_NGROK" = true ]; then
    echo "Starting Ngrok..."
    (uv sync && source .venv/bin/activate && python start_ngrok.py) &
else
    echo "Skipping Ngrok startup (--local specified)..."
fi

# Wait for all background processes
wait
