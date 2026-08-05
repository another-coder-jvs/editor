from pathlib import Path
import os

_BASE = Path(__file__).resolve().parents[2]

# Set ENV=local to use local paths, defaults to server (Google Drive) paths
_ENV = os.getenv("ENV", "server")

if _ENV == "local":
    TEMP_DIR     = _BASE / "temp"
    OUTPUT_DIR   = _BASE / "outputs"
    PROJECTS_DIR = _BASE / "projects"
    WEIGHT_DIR   = _BASE / "weights"
else:
    _DRIVE = Path("/content/drive/MyDrive/project_folders")
    TEMP_DIR     = _DRIVE / "temp"
    OUTPUT_DIR   = _DRIVE / "outputs"
    PROJECTS_DIR = _DRIVE / "projects"
    WEIGHT_DIR   = Path("/content/editor/weights")
