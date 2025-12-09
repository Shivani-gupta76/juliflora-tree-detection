import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from inference import set_georasters, detect_on_image, annotate_and_measure
from pathlib import Path
import uuid

UPLOAD_DIR = "uploads"
ORTH_PATH = os.path.join(UPLOAD_DIR, "orthomosaic.tif")
DSM_PATH = os.path.join(UPLOAD_DIR, "dsm.tif")
DTM_PATH = os.path.join(UPLOAD_DIR, "dtm.tif")
IMAGES_DIR = os.path.join(UPLOAD_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Juliflora Height Estimation (DSM-DTM) Backend")
# Allow frontend (HTML/JS) to call this API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"

# Serve /uploads/* URLs as static files (images, annotated outputs)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
def root():
    return {"message": "Juliflora DSM/DTM backend running. Upload orthomosaic + dsm + dtm before inferring."}

@app.post("/upload/orthomosaic")
async def upload_orthomosaic(file: UploadFile = File(...)):
    dest = ORTH_PATH
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "ok", "path": dest}

@app.post("/upload/dsm")
async def upload_dsm(file: UploadFile = File(...)):
    dest = DSM_PATH
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "ok", "path": dest}

@app.post("/upload/dtm")
async def upload_dtm(file: UploadFile = File(...)):
    dest = DTM_PATH
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "ok", "path": dest}

@app.post("/prepare_rasters")
async def prepare_rasters():
    if not (os.path.exists(ORTH_PATH) and os.path.exists(DSM_PATH) and os.path.exists(DTM_PATH)):
        raise HTTPException(status_code=400, detail="Orthomosaic, DSM or DTM missing.")
    set_georasters(ORTH_PATH, DSM_PATH, DTM_PATH)
    return {"status": "ok", "message": "Rasters loaded into memory."}

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    allowed = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(IMAGES_DIR, unique_name)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = detect_on_image(dest)
    if not detections:
        return {"status": "ok", "detections": [], "annotated_image": None}

    enriched, annotated_path = annotate_and_measure(dest, detections)
    return {
        "status": "ok",
        "tree_count": len(enriched),   # 👈 number of Juliflora trees detected
        "detections": enriched,
        "annotated_image": annotated_path,
    }


@app.get("/download/annotated")
def download_annotated(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/jpeg", filename=os.path.basename(path))
