# juliflora_backend/inference.py
#
# Improved inference utilities:
# - Better filtering of false vegetation
# - Smaller and cleaner individual boundaries
# - Confidence filtering
# - Area estimation
# - Cleaner annotated outputs

import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8_juliflora.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

yolo_model = YOLO(str(MODEL_PATH))


# ---------------------------------------------------------------------
# Dummy georaster hook
# ---------------------------------------------------------------------
def set_georasters(*args, **kwargs):
    print("Geospatial rasters ignored in simplified inference.")


# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------
def calculate_area(x1, y1, x2, y2):
    return int((x2 - x1) * (y2 - y1))


# ---------------------------------------------------------------------
# YOLO Detection
# ---------------------------------------------------------------------
def detect_on_image(image_path: str) -> List[Dict]:

    results = yolo_model(
    image_path,
    conf=0.12
)[0]

    detections: List[Dict] = []

    names = yolo_model.names

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        conf = float(box.conf[0])

        cls_id = int(box.cls[0])

        cls_name = names[cls_id]

        width = x2 - x1
        height = y2 - y1

        area = calculate_area(x1, y1, x2, y2)

        aspect_ratio = width / max(height, 1)

        # -------------------------------------------------------------
        # FILTERING SECTION
        # -------------------------------------------------------------

        # Ignore weak detections
        if conf < 0.45:
            continue

        # Remove huge merged vegetation
        if width > 350 or height > 350:
            continue

        # Remove tiny noisy detections
        if area < 800:
            continue

        # Remove unrealistic shapes
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            continue

        detections.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "estimated_area": area,
            }
        )

    return detections


# ---------------------------------------------------------------------
# Annotation + Enrichment
# ---------------------------------------------------------------------
def annotate_and_measure(
    image_path: str,
    detections=None
) -> Tuple[List[Dict], str]:

    if detections is None:
        detections = detect_on_image(image_path)

    img = cv2.imread(image_path)

    if img is None:
        raise RuntimeError(f"Could not read image at {image_path}")

    annotated = img.copy()

    enriched: List[Dict] = []

    for idx, det in enumerate(detections, start=1):

        x1, y1, x2, y2 = det["bbox_xyxy"]

        conf = det.get("confidence", 0.0)

        class_name = det.get("class_name", "juliflora")

        estimated_area = det.get("estimated_area", 0)

        # -------------------------------------------------------------
        # Draw bounding box
        # -------------------------------------------------------------
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        # -------------------------------------------------------------
        # Label text
        # -------------------------------------------------------------
        label = (
            f"{class_name} "
            f"{conf:.2f}"
        )

        cv2.putText(
            annotated,
            label,
            (int(x1), max(int(y1) - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # Center marker
        # -------------------------------------------------------------
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cv2.circle(
            annotated,
            (center_x, center_y),
            3,
            (0, 0, 255),
            -1
        )

        # -------------------------------------------------------------
        # Enriched output
        # -------------------------------------------------------------
        enriched.append(
            {
                **det,
                "boundary_id": idx,
                "height_m": None,
                "width_m": None,
            }
        )

    # -----------------------------------------------------------------
    # Save annotated image
    # -----------------------------------------------------------------
    img_path = Path(image_path)

    out_dir = img_path.parent / "annotated"

    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / img_path.name

    cv2.imwrite(str(out_path), annotated)

    return enriched, str(out_path)