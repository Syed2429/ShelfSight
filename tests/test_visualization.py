"""
End-to-end pipeline test: detection → grouping → visualization.
Usage:
    python tests/test_visualization.py "<path_to_image>"
"""

import sys
import os
import base64
import shutil

# ── service paths ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for svc in ("detection", "grouping", "visualization"):
    sys.path.insert(0, os.path.join(BASE, "services", svc))

from detector    import Detector
from grouper     import Grouper
from visualizer  import Visualizer

# ── helpers ────────────────────────────────────────────────────────────────
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_visualization.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    print(f"[Test] Image: {os.path.basename(img_path)}  ({len(image_bytes)//1024} KB)")

    # ── Step 1: Detection ──────────────────────────────────────────────────
    print("\n[Step 1] Running SAHI detector…")
    detector = Detector()
    det_result = detector.detect(image_bytes)
    detections  = det_result["detections"]
    image_dims  = det_result["image_dims"]
    print(f"  detected: {len(detections)} products  |  image: {image_dims}")

    if not detections:
        print("[WARN] No detections — aborting pipeline.")
        sys.exit(0)

    # ── Step 2: Grouping ───────────────────────────────────────────────────
    print("\n[Step 2] Running spatial+color grouper…")
    grouper = Grouper()
    grp_result       = grouper.group(detections, image_bytes)
    # Flatten groups list to a flat grouped_products list
    grouped_products = [p for g in grp_result["groups"] for p in g["products"]]
    num_groups       = grp_result["num_groups"]
    print(f"  groups:   {num_groups}")

    counts = {}
    for p in grouped_products:
        counts[p["group_id"]] = counts.get(p["group_id"], 0) + 1
    for gid in sorted(counts):
        print(f"    G{gid}: {counts[gid]} products")

    # ── Step 3: Visualization ──────────────────────────────────────────────
    print("\n[Step 3] Running visualizer…")
    visualizer = Visualizer()
    vis_bytes, default_path = visualizer.draw_groups(image_bytes, grouped_products)

    out_path = os.path.join(OUTPUTS, "test_full_pipeline.jpg")
    with open(out_path, "wb") as f:
        f.write(vis_bytes)
    print(f"  saved: {out_path}  ({len(vis_bytes)//1024} KB)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Pipeline Summary ──────────────────────────────────────────")
    print(f"  Products detected : {len(detections)}")
    print(f"  Groups formed     : {num_groups}")
    print(f"  Output image      : {out_path}")
    print("  PASS")


if __name__ == "__main__":
    main()
