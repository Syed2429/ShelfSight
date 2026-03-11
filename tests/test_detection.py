"""
Standalone detection smoke-test (SAHI edition).

Usage
-----
python tests/test_detection.py <path/to/image.jpg>

What it does
------------
1. Loads the image at <path> as raw bytes.
2. Instantiates Detector (SAHI+YOLOv8s) and calls detect().
3. Draws bboxes with label text and confidence score, using a unique
   colour per detected class label.
4. Prints a summary: total detections, unique labels found, min/max bbox area.
5. Saves the annotated result to outputs/test_detection_sahi.jpg.
"""

import os
import sys

# Allow running from the repo root: resolve the detection service package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETECTION_DIR = os.path.join(_REPO_ROOT, "services", "detection")
sys.path.insert(0, _DETECTION_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from detector import Detector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All detections are labelled "product" — use a single green colour.
_GREEN = (0, 200, 0)


def draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Overlay bounding boxes and confidence scores onto *image_bgr* (in-place)."""
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICKNESS = 2

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        text = f"{conf:.2f}"
        color = _GREEN

        # Bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, THICKNESS)

        # Label background strip
        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            image_bgr,
            (x1, label_y1),
            (x1 + tw + 4, label_y1 + th + baseline + 4),
            color,
            cv2.FILLED,
        )

        # Label text (white for contrast)
        cv2.putText(
            image_bgr,
            text,
            (x1 + 2, label_y1 + th + 2),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            THICKNESS,
            cv2.LINE_AA,
        )

    return image_bgr


def print_summary(detections: list, image_dims: list) -> None:
    """Print total products detected, avg confidence, and bbox area range."""
    count = len(detections)
    w, h = image_dims
    print(f"\n{'='*50}")
    print(f"  Image dimensions       : {w} x {h} px")
    print(f"  Total products detected: {count}")

    if count == 0:
        print("  No products detected above threshold.")
        print("="*50)
        return

    areas = [d["area"] for d in detections]
    avg_conf = sum(d["confidence"] for d in detections) / count

    print(f"  Avg confidence         : {avg_conf:.2f}")
    print(f"  Bbox area range        : {min(areas):,} - {max(areas):,} px²")
    print("="*50)


def ensure_outputs_dir(repo_root: str) -> str:
    out_dir = os.path.join(repo_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_detection.py <path/to/image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1].strip()
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    # 1. Load raw bytes
    with open(image_path, "rb") as fh:
        image_bytes = fh.read()
    print(f"[test] Loaded {len(image_bytes):,} bytes from {image_path}")

    # 2. Run detection
    print("[test] Initialising Detector (SAHI+YOLOv8s) …")
    detector = Detector()

    print("[test] Running detect() …")
    result = detector.detect(image_bytes)

    detections = result["detections"]
    image_dims = result["image_dims"]

    # 3. Print summary
    print_summary(detections, image_dims)

    # 4. Draw bboxes with per-class colours and confidence scores
    img_bgr = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if img_bgr is None:
        print("[ERROR] Could not decode image for visualisation.")
        sys.exit(1)

    annotated = draw_detections(img_bgr, detections)

    # 5. Save result
    out_dir = ensure_outputs_dir(_REPO_ROOT)
    out_path = os.path.join(out_dir, "test_detection_sahi.jpg")
    success = cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if success:
        print(f"\n[test] Annotated image saved to: {out_path}")
    else:
        print(f"[ERROR] Failed to write output image to {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
