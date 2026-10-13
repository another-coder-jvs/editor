"""
Colab Start — Run this cell to start Ollama + the backend server.
Run setup_colab.py first if you haven't installed Ollama yet.
"""
import subprocess
import os
import sys
import time

# ── 1. Ensure Ollama is running ──────────────────────────────────────────────
print("Checking Ollama...")

# Start ollama serve if not running
subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3)

import requests
try:
    r = requests.get("http://127.0.0.1:11434/api/version", timeout=5)
    if r.status_code == 200:
        print(f"✅ Ollama running — version: {r.json().get('version')}")
    else:
        print("⚠️  Ollama may not be responding")
except Exception:
    print("⚠️  Ollama not available — vision detection will use fallback prompt")

# ── 2. Start backend ─────────────────────────────────────────────────────────
print("\nStarting backend server...")

os.environ["ENV"] = "server"  # use Google Drive paths

sys.path.insert(0, "backend")
os.chdir("backend")

from fastapi import FastAPI
from pyngrok import ngrok
import nest_asyncio
import uvicorn

nest_asyncio.apply()

# Import the actual app
from main import app

# ── 3. Start ngrok + uvicorn ─────────────────────────────────────────────────
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
if NGROK_AUTH_TOKEN:
    from pyngrok import ngrok as ng
    ng.set_auth_token(NGROK_AUTH_TOKEN)
    ng.kill()
    public_url = ng.connect(8000)
    print("\n" + "=" * 60)
    print("Public URL:", public_url.public_url)
    print("Swagger Docs:", public_url.public_url + "/docs")
    print("=" * 60)
else:
    print("\n⚠️  No NGROK_AUTH_TOKEN set. Server will run on localhost:8000")

config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
await server.serve()
