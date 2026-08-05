"""Router: POST /project/save and POST /project/load"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from schemas import SaveRequest, LoadRequest

router = APIRouter()
logger = logging.getLogger(__name__)

from utils import config
PROJECTS_DIR = config.PROJECTS_DIR
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/save")
async def save_project(req: SaveRequest):
    project_path = PROJECTS_DIR / f"{req.project_name}.json"
    logger.info(f"[project/save] name={req.project_name} layers={len(req.layers)} → {project_path}")
    try:
        project_path.write_text(json.dumps(req.model_dump(), indent=2))
    except Exception as e:
        logger.exception(f"[project/save] failed: {e}")
        raise
    logger.info(f"[project/save] saved OK")
    return {"status": "saved", "path": str(project_path)}


@router.get("/list")
async def list_projects():
    projects = [p.stem for p in sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
    return {"projects": projects}


@router.delete("/{project_name}")
async def delete_project(project_name: str):
    project_path = PROJECTS_DIR / f"{project_name}.json"
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    project_path.unlink()
    return {"status": "deleted"}
async def load_project(req: LoadRequest):
    project_path = PROJECTS_DIR / f"{req.project_name}.json"
    logger.info(f"[project/load] name={req.project_name} path={project_path}")
    if not project_path.exists():
        logger.error(f"[project/load] not found: {project_path}")
        raise HTTPException(status_code=404, detail="Project not found")
    data = json.loads(project_path.read_text())
    logger.info(f"[project/load] loaded {len(data.get('layers', []))} layers")
    return data
