import cv2
import numpy as np
import torch
import torchvision.ops as ops
import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        service_dir = os.path.dirname(os.path.abspath(__file__))
        local_weights = os.path.join(service_dir, "best.pt")

        print(f"[Detector] Looking for weights at: {local_weights}")
        print(f"[Detector] File exists: {os.path.exists(local_weights)}")

        if os.path.exists(local_weights):
            print("[Detector] Loading SKU-110K YOLOv10 weights via SAHI...")
            model_path = local_weights
        else:
            print("[Detector] Local weights not found, using HuggingFace fallback...")
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="foduucom/product-detection-in-shelf-yolov8",
                filename="best.pt"
            )

        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.30,
            device=self.device
        )
        print(f"[Detector] SAHI+YOLOv10 ready on {self.device}")

    def _apply_nms(self, detections: list,
                iou_threshold: float = 0.40) -> list:
        if len(detections) < 2:
            return detections
        boxes = torch.tensor(
            [[d["bbox"][0], d["bbox"][1],
            d["bbox"][2], d["bbox"][3]]
            for d in detections], dtype=torch.float32)
        scores = torch.tensor(
            [d["confidence"] for d in detections],
            dtype=torch.float32)
        keep = ops.nms(boxes, scores, iou_threshold)
        kept = [detections[i] for i in keep.tolist()]
        for idx, det in enumerate(kept):
            det["product_id"] = idx
        return kept

    def detect(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")

        h, w = image.shape[:2]
        print(f"[Detection] Image {w}x{h}, running SAHI+YOLOv10...")

        result1 = get_sliced_prediction(
            image, self.model,
            slice_height=256, slice_width=256,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            verbose=0
        )

        result2 = get_sliced_prediction(
            image, self.model,
            slice_height=480, slice_width=480,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )

        all_preds = (result1.object_prediction_list +
                    result2.object_prediction_list)

        EDGE_MARGIN = 8  # pixels — bbox touching edge = partial product

        detections = []
        for i, obj in enumerate(all_preds):
            bbox = obj.bbox
            x1 = int(bbox.minx)
            y1 = int(bbox.miny)
            x2 = int(bbox.maxx)
            y2 = int(bbox.maxy)
            area = (x2 - x1) * (y2 - y1)

            # Products cut off at the image boundary are partially visible —
            # apply looser thresholds so they aren't dropped
            touches_edge = (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or
                            x2 >= w - EDGE_MARGIN or y2 >= h - EDGE_MARGIN)
            min_area = 600 if touches_edge else 1500
            min_side = 15 if touches_edge else 25

            if area < min_area:
                continue
            bw, bh = x2 - x1, y2 - y1
            if bh == 0:
                continue
            # drop detections too thin/short in either dimension (price tags, shelf labels)
            if bw < min_side or bh < min_side:
                continue
            if not (0.15 < bw / bh < 5.0):
                continue

            detections.append({
                "product_id": i,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(obj.score.value), 4),
                "label": "product",
                "area": area
            })

        detections = self._apply_nms(detections)
        print(f"[Detection] Found {len(detections)} products")

        return {
            "detections": detections,
            "count": len(detections),
            "image_dims": [w, h]
        }
