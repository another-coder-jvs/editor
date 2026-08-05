
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Project started.. ! importing files !")


import os
logger.info(f"Imported : os !")
from contextlib import asynccontextmanager
logger.info(f"Imported : contextlib !")
import uvicorn
logger.info(f"Imported : uvicorn !")
from fastapi import FastAPI
logger.info(f"Imported : fastapi !")
from fastapi.middleware.cors import CORSMiddleware
logger.info(f"Imported : CORSMiddleware !")
from fastapi.staticfiles import StaticFiles
logger.info(f"Imported : StaticFiles !")

from routers import detect, segment, layers, edit, merge, project, export, progress, session
from services.model_manager import ModelManager
from utils import config
logger.info("Files imported !")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Image Editor backend...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.PROJECTS_DIR, exist_ok=True)
    os.makedirs(config.WEIGHT_DIR, exist_ok=True)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Image Editor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/detect", tags=["detect"])
app.include_router(segment.router, prefix="/segment", tags=["segment"])
app.include_router(layers.router, prefix="/layers", tags=["layers"])
app.include_router(edit.router, prefix="/edit", tags=["edit"])
app.include_router(merge.router, prefix="/merge", tags=["merge"])
app.include_router(project.router, prefix="/project", tags=["project"])
app.include_router(export.router, prefix="/export", tags=["export"])
app.include_router(progress.router, prefix="/progress", tags=["progress"])
app.include_router(session.router, prefix="/session", tags=["session"])

if os.path.exists("../outputs"):
    app.mount("/outputs", StaticFiles(directory="../outputs"), name="outputs")
if os.path.exists("../drive/MyDrive/project_folders/temp"):
    app.mount("/temp", StaticFiles(directory="../temp"), name="temp")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0" , port=8000)