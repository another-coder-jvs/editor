"""Router: GET /progress/{session_id}"""
from __future__ import annotations

from fastapi import APIRouter
from schemas import ProgressResponse
from services.progress_store import get_progress

router = APIRouter()


@router.get("/{session_id}", response_model=ProgressResponse)
async def progress(session_id: str):
    return get_progress(session_id)
