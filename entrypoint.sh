#!/bin/bash

# Start API server in background
echo "Starting CoverHunter API server..."
python api.py --port 8080 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API..."
until python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" 2>/dev/null; do
    sleep 2
done
echo "API ready. Starting crawler..."

# Start crawler (non-critical, runs to completion)
python crawl_songs.py --api http://localhost:8080 --delay 2 &

# Wait for API (container lives/dies with the API)
wait $API_PID
