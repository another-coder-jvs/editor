@echo off
REM One-click installer for AI Image Editor (Windows)
echo ============================================
echo   AI Image Editor - One-Click Installer
echo ============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11+ is required. Please install from python.org
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js 18+ is required. Please install from nodejs.org
    pause
    exit /b 1
)

echo [1/5] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/5] Installing backend dependencies...
pip install --upgrade pip
pip install -r backend\requirements.txt

echo [3/5] Installing SAM2 (optional)...
pip install git+https://github.com/facebookresearch/segment-anything-2.git
if errorlevel 1 echo SAM2 not available, will use SAM fallback.

echo [4/5] Installing frontend dependencies...
cd frontend
npm install
cd ..

echo [5/5] Creating directories...
if not exist weights mkdir weights
if not exist projects mkdir projects
if not exist outputs mkdir outputs
if not exist temp mkdir temp

echo.
echo ============================================
echo   Installation complete!
echo   Run start.bat to launch the application.
echo ============================================
pause
