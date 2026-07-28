#!/usr/bin/env bash
# Start AI Image Editor (Linux/macOS)
set -e

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Starting backend on http://localhost:8000 ..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

echo "Starting frontend on http://localhost:5173 ..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "  AI Image Editor is running!"
echo "  Open: http://localhost:5173"
echo "  Press Ctrl+C to stop."
echo "============================================"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
