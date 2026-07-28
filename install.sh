#!/usr/bin/env bash
# One-click installer for AI Image Editor (Linux/macOS)
set -e

echo "============================================"
echo "  AI Image Editor – One-Click Installer"
echo "============================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3.11+ is required. Please install it first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Check Node.js
if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js 18+ is required. Please install it first."
    exit 1
fi

echo "Node version: $(node --version)"

# Create virtual environment
echo ""
echo "[1/5] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
echo ""
echo "[2/5] Installing backend dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Install SAM2 (optional, falls back to SAM)
echo ""
echo "[3/5] Installing SAM2 (optional)..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git || \
    echo "SAM2 not available, will use SAM fallback."

# Install frontend dependencies
echo ""
echo "[4/5] Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create required directories
echo ""
echo "[5/5] Creating directories..."
mkdir -p weights projects outputs temp

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  To start the application:"
echo "    Linux/macOS: ./start.sh"
echo "    Windows:     start.bat"
echo "============================================"
