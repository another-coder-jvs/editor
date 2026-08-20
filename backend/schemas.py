"""
Pydantic schemas for the AI Image Editor.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class LayerData(BaseModel):
    id: str
    name: str
    mask_path: str
    png_path: str
    bbox: BoundingBox
    z_index: int = 0
    visible: bool = True
    opacity: float = 1.0
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    scale: Dict[str, float] = Field(default_factory=lambda: {"x": 1.0, "y": 1.0})
    rotation: float = 0.0
    history: List[str] = Field(default_factory=list)
    locked: bool = False


class DetectRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = None  # custom detection prompt; None = auto-detect all


class DetectResponse(BaseModel):
    objects: List[Dict[str, Any]]  # [{label, bbox, score}]
    session_id: str
    image_path: str  # path to the saved image on disk


class SegmentRequest(BaseModel):
    session_id: str
    image_path: str
    objects: List[Dict[str, Any]]


class SegmentResponse(BaseModel):
    session_id: str
    layers: List[LayerData]


class EditRequest(BaseModel):
    session_id: str
    layer_id: str
    prompt: str
    image_path: str
    strength: float = 0.75
    guidance_scale: float = 7.5
    steps: int = 20
    edit_type: str | None = None
    edit_params: dict | None = None


class EditResponse(BaseModel):
    layer_id: str
    edited_png_path: str
    session_id: str


class MergeRequest(BaseModel):
    session_id: str
    layers: List[LayerData]
    canvas_width: int
    canvas_height: int
    output_format: str = "png"


class MergeResponse(BaseModel):
    output_path: str
    session_id: str


class SaveRequest(BaseModel):
    session_id: str
    project_name: str
    original_image_path: str
    layers: List[LayerData]
    canvas_width: int
    canvas_height: int
    canvas_position: Dict[str, float] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)
    prompts: List[str] = Field(default_factory=list)


class LoadRequest(BaseModel):
    project_name: str


class ExportRequest(BaseModel):
    session_id: str
    layers: List[LayerData]
    canvas_width: int
    canvas_height: int
    format: str = "png"
    upscale: bool = False
    upscale_factor: int = 2


class ProgressResponse(BaseModel):
    session_id: str
    task: str
    progress: float  # 0.0 – 1.0
    message: str
    done: bool
