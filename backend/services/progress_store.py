"""
Progress tracking – simple in-memory store for task progress.
"""
from __future__ import annotations

from typing import Dict
from schemas import ProgressResponse

_store: Dict[str, ProgressResponse] = {}


def set_progress(session_id: str, task: str, progress: float, message: str, done: bool = False):
    _store[session_id] = ProgressResponse(
        session_id=session_id,
        task=task,
        progress=progress,
        message=message,
        done=done,
    )


def get_progress(session_id: str) -> ProgressResponse:
    return _store.get(
        session_id,
        ProgressResponse(session_id=session_id, task="idle", progress=0.0, message="", done=True),
    )
