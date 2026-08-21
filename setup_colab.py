"""
Colab Setup — Run this cell first to install Ollama + start the backend.
"""
import subprocess
import os
import sys

# ── 1. Install Ollama ─────────────────────────────────────────────────────────
print("=" * 60)
print("Installing Ollama...")
print("=" * 60)

subprocess.run(
    "curl -fsSL https://ollama.com/install.sh | sh",
    shell=True, check=False,
)

# ── 2. Start Ollama server in background ──────────────────────────────────────
print("\nStarting Ollama server...")

# Kill any existing ollama process
subprocess.run("pkill -f 'ollama serve' || true", shell=True, check=False)

# Start ollama serve in background
subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

import time
time.sleep(3)

# ── 3. Pull vision model ─────────────────────────────────────────────────────
print("Pulling minicpm-v4.6 vision model (this may take a few minutes)...")
subprocess.run("ollama pull minicpm-v4.6", shell=True, check=False)

# ── 4. Verify ────────────────────────────────────────────────────────────────
import requests
try:
    r = requests.get("http://127.0.0.1:11434/api/version", timeout=5)
    if r.status_code == 200:
        print(f"\n✅ Ollama running — version: {r.json().get('version')}")
    else:
        print("\n⚠️  Ollama started but may not be responding")
except Exception as e:
    print(f"\n⚠️  Ollama check failed: {e}")

print("\n" + "=" * 60)
print("Setup complete! Now run the backend.")
print("=" * 60)
