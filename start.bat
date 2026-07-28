@echo off
REM Start AI Image Editor (Windows)

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Starting backend on http://localhost:8000 ...
start "AI Editor Backend" cmd /k "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting frontend on http://localhost:5173 ...
start "AI Editor Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================
echo   AI Image Editor is running!
echo   Open: http://localhost:5173
echo ============================================
