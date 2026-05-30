from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import os
import shutil
import uuid
from pathlib import Path

from inference import (
    set_georasters,
    detect_on_image,
    annotate_and_measure
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
ORTH_PATH = UPLOAD_DIR / "orthomosaic.tif"
DSM_PATH = UPLOAD_DIR / "dsm.tif"
DTM_PATH = UPLOAD_DIR / "dtm.tif"

IMAGES_DIR = UPLOAD_DIR / "images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Juliflora Height Estimation (DSM-DTM) Backend"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Static Files
# --------------------------------------------------

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)

# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# --------------------------------------------------
# Orthomosaic Upload
# --------------------------------------------------

@app.post("/upload/orthomosaic")
async def upload_orthomosaic(
    file: UploadFile = File(...)
):
    dest = str(ORTH_PATH)

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "ok",
        "path": dest
    }

# --------------------------------------------------
# DSM Upload
# --------------------------------------------------

@app.post("/upload/dsm")
async def upload_dsm(
    file: UploadFile = File(...)
):
    dest = str(DSM_PATH)

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "ok",
        "path": dest
    }

# --------------------------------------------------
# DTM Upload
# --------------------------------------------------

@app.post("/upload/dtm")
async def upload_dtm(
    file: UploadFile = File(...)
):
    dest = str(DTM_PATH)

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "ok",
        "path": dest
    }

# --------------------------------------------------
# Prepare Rasters
# --------------------------------------------------

@app.post("/prepare_rasters")
async def prepare_rasters():

    if not (
        ORTH_PATH.exists()
        and DSM_PATH.exists()
        and DTM_PATH.exists()
    ):
        raise HTTPException(
            status_code=400,
            detail="Orthomosaic, DSM or DTM missing."
        )

    set_georasters(
        str(ORTH_PATH),
        str(DSM_PATH),
        str(DTM_PATH)
    )

    return {
        "status": "ok",
        "message": "Rasters loaded into memory."
    }

# --------------------------------------------------
# Image Upload & Detection
# --------------------------------------------------

@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...)
):

    ext = Path(file.filename).suffix.lower()

    allowed = [
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png"
    ]

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type."
        )

    unique_name = f"{uuid.uuid4().hex}{ext}"

    dest = IMAGES_DIR / unique_name

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = detect_on_image(str(dest))

    if not detections:
        return {
            "status": "ok",
            "boundary_count": 0,
            "detections": [],
            "annotated_image": None
        }

    enriched, annotated_path = annotate_and_measure(
        str(dest),
        detections
    )

    return {
        "status": "ok",
        "boundary_count": len(enriched),
        "detections": enriched,
        "annotated_image": annotated_path.replace("\\", "/")
    }

# --------------------------------------------------
# Download Annotated Image
# --------------------------------------------------

@app.get("/download/annotated")
def download_annotated(path: str):

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=os.path.basename(path)
    )