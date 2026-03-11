"""
Standalone grouping smoke-test.

Usage
-----
python tests/test_grouping.py <path/to/image.jpg>

Pipeline
--------
1. Load image bytes from <path>.
2. Run Detector (SAHI+YOLOv8s) → detections + image_dims.
3. Base64-encode the raw image and pass to Grouper.
4. Print: num_groups, products per group, group_id assignments.
5. Draw bboxes coloured by group_id with "G{id}" label.
6. Save annotated result to outputs/test_grouping.jpg.
"""

import os
import sys
import base64

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETECTION_DIR = os.path.join(_REPO_ROOT, "services", "detection")
_GROUPING_DIR = os.path.join(_REPO_ROOT, "services", "grouping")
sys.path.insert(0, _DETECTION_DIR)
sys.path.insert(0, _GROUPING_DIR)

import cv2                          # noqa: E402
import numpy as np                  # noqa: E402
from detector import Detector       # noqa: E402
from grouper import Grouper         # noqa: E402


# ---------------------------------------------------------------------------
# Colour palette — one BGR colour per group id
# ---------------------------------------------------------------------------
_PALETTE = [
    (0,   200,   0),   # green
    (255,  80,   0),   # blue
    (0,    80, 255),   # red
    (0,   200, 200),   # yellow
    (200,   0, 200),   # magenta
    (0,   160, 255),   # orange
    (255, 200,   0),   # cyan
    (128,   0, 255),   # purple
    (0,   128, 128),   # teal
    (200, 200,   0),   # light-blue
    (180, 105, 255),   # pink
    (0,   255, 191),   # mint
]


def _group_color(group_id: int) -> tuple:
    return _PALETTE[group_id % len(_PALETTE)]


def draw_groups(image_bgr: np.ndarray, grouped_products: list) -> np.ndarray:
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.60
    THICKNESS  = 2

    for prod in grouped_products:
        x1, y1, x2, y2 = prod["bbox"]
        gid   = prod["group_id"]
        conf  = prod["confidence"]
        text  = f"G{gid} {conf:.2f}"
        color = _group_color(gid)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, THICKNESS)

        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            image_bgr,
            (x1, label_y1),
            (x1 + tw + 4, label_y1 + th + baseline + 4),
            color,
            cv2.FILLED,
        )
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


def print_summary(grouped_products: list, num_groups: int, image_dims: list) -> None:
    w, h = image_dims
    print(f"\n{'='*55}")
    print(f"  Image dimensions      : {w} x {h} px")
    print(f"  Total products        : {len(grouped_products)}")
    print(f"  Number of groups      : {num_groups}")

    if not grouped_products:
        print("="*55)
        return

    # Products per group
    from collections import defaultdict
    per_group = defaultdict(list)
    for p in grouped_products:
        per_group[p["group_id"]].append(p)

    print(f"\n  {'Group':<8} {'Count':<8} {'Avg conf':<12} {'Bbox ids'}")
    print(f"  {'-'*50}")
    for gid in sorted(per_group):
        members  = per_group[gid]
        avg_conf = sum(m["confidence"] for m in members) / len(members)
        ids      = [m["product_id"] for m in members]
        print(f"  G{gid:<7} {len(members):<8} {avg_conf:<12.2f} {ids}")

    print("="*55)


def ensure_outputs_dir() -> str:
    out_dir = os.path.join(_REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_grouping.py <path/to/image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1].strip()
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    # 1. Load raw bytes
    with open(image_path, "rb") as fh:
        image_bytes = fh.read()
    print(f"[test] Loaded {len(image_bytes):,} bytes from {image_path}")

    # 2. Detect
    print("\n[test] Initialising Detector (SAHI+YOLOv8s)...")
    detector = Detector()
    print("[test] Running detect()...")
    det_result   = detector.detect(image_bytes)
    detections   = det_result["detections"]
    image_dims   = det_result["image_dims"]
    print(f"[test] {len(detections)} products detected")

    # 3. Group
    print("\n[test] Initialising Grouper (ResNet50+DBSCAN)...")
    grouper = Grouper()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    print("[test] Running group_products()...")
    grp_result       = grouper.group_products(image_base64, detections, image_dims)
    grouped_products = grp_result["grouped_products"]
    num_groups       = grp_result["num_groups"]

    # 4. Print summary
    print_summary(grouped_products, num_groups, image_dims)

    # 5. Annotate
    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("[ERROR] Could not decode image for visualisation.")
        sys.exit(1)

    annotated = draw_groups(img_bgr, grouped_products)

    # 6. Save
    out_dir  = ensure_outputs_dir()
    out_path = os.path.join(out_dir, "test_grouping.jpg")
    ok = cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        print(f"\n[test] Annotated image saved to: {out_path}")
    else:
        print(f"[ERROR] Failed to write output image to {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
