import cv2
import numpy as np
import os


_PALETTE = [
    (0,   200,   0),
    (255,  80,   0),
    (0,    80, 255),
    (0,   200, 200),
    (200,   0, 200),
    (0,   160, 255),
    (255, 200,   0),
    (128,   0, 255),
    (0,   128, 128),
    (200, 200,   0),
    (180, 105, 255),
    (0,   255, 191),
]

_OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs"
)


class Visualizer:
    def __init__(self):
        os.makedirs(_OUTPUTS_DIR, exist_ok=True)
        print("[Visualizer] Ready")

    def draw_groups(self, image_bytes: bytes,
                    grouped_products: list) -> tuple:
        """
        Draw group-coloured bounding boxes with confidence labels.

        Returns (jpeg_bytes, output_path).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        FONT       = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.55
        THICKNESS  = 2

        for prod in grouped_products:
            x1, y1, x2, y2 = prod["bbox"]
            gid   = prod["group_id"]
            text  = f"G{gid}"
            color = _PALETTE[gid % len(_PALETTE)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)

            (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            label_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                img,
                (x1, label_y1),
                (x1 + tw + 4, label_y1 + th + baseline + 4),
                color, cv2.FILLED,
            )
            cv2.putText(
                img, text,
                (x1 + 2, label_y1 + th + 2),
                FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA,
            )

        out_path = os.path.join(_OUTPUTS_DIR, "visualization_output.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Failed to encode annotated image")

        return bytes(buf), out_path
