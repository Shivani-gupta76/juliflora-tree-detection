# juliflora_backend/inference.py
#
# Simplified inference utilities:
# - Uses YOLO model to detect Juliflora in an image
# - Draws bounding boxes
# - Returns detections + path to annotated image
# - Does NOT require DSM/DTM or any geospatial rasters

import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
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
# Dummy georaster hook (for compatibility with app.py)
# ---------------------------------------------------------------------
def set_georasters(*args, **kwargs):
    """
    Placeholder to keep compatibility with previous design.
    We are not using DSM/DTM here, so this does nothing.
    """
    print("set_georasters() called, but geospatial rasters are ignored in this simplified version.")


# ---------------------------------------------------------------------
# Run YOLO on an image and return simple detection list
# ---------------------------------------------------------------------
def detect_on_image(image_path: str) -> List[Dict]:
    """
    Run YOLO on the image and return a list of detections.
    Each detection dict contains:
      - bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
      - confidence: float
      - class_id: int
      - class_name: str
    """
    results = yolo_model(image_path)[0]
    detections: List[Dict] = []
    names = yolo_model.names

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        detections.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
            }
        )

    return detections


# ---------------------------------------------------------------------
# Annotate image with boxes and prepare JSON-friendly enriched data
# ---------------------------------------------------------------------
def annotate_and_measure(image_path: str, detections=None) -> Tuple[List[Dict], str]:
    """
    Draw bounding boxes on the image and return:
      - enriched_dets: list of dicts with bbox + confidence (+ dummy height/width)
      - annotated_path: path to saved annotated image

    In this simplified version we:
      - use pixel coordinates only
      - set height_m and width_m to None (no DSM/DTM)
    """
    # If detections not passed, run YOLO here
    if detections is None:
        detections = detect_on_image(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image at {image_path}")

    annotated = img.copy()
    enriched: List[Dict] = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det.get("confidence", 0.0)
        class_name = det.get("class_name", "juliflora")

        # Draw rectangle
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        # Put label text
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (int(x1), max(int(y1) - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Add dummy real-world measures (you can extend later)
        enriched.append(
            {
                **det,
                "height_m": None,
                "width_m": None,
            }
        )

    # Save annotated image next to original in an "annotated" folder
    img_path = Path(image_path)
    out_dir = img_path.parent / "annotated"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / img_path.name

    cv2.imwrite(str(out_path), annotated)

    return enriched, str(out_path)
