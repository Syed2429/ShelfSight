"""
Detection service — Flask app that wraps the YOLOv8 Detector.

Endpoints
---------
GET  /health   — liveness probe (returns 200 when model is loaded)
POST /detect   — accepts raw image bytes (multipart or application/octet-stream)
               or a JSON body {"image": "<base64-encoded image>"}
               returns {count, detections: [{bbox, label, confidence}, ...]}
"""

import base64
import io
import os

from flask import Flask, request, jsonify

from detector import Detector

# ---------------------------------------------------------------------------
# App + model initialisation
# ---------------------------------------------------------------------------

app = Flask(__name__)
_detector: Detector | None = None  # lazy-initialised once on first request
_model_ready = False


def get_detector() -> Detector:
    """Return the singleton Detector, initialising it on first call."""
    global _detector, _model_ready
    if _detector is None:
        _detector = Detector()
        _model_ready = True
    return _detector


# Pre-load at startup (not lazily) so the healthcheck is accurate.
with app.app_context():
    get_detector()

PORT = int(os.getenv("PORT", 5001))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness + readiness probe."""
    return jsonify({
        "status": "ok",
        "service": "detection",
        "model_ready": _model_ready,
        "device": get_detector().device,
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accept an image and return YOLOv8 detections.

    Supported request formats (in priority order):
      1. multipart/form-data  — field name ``image``
      2. application/octet-stream — raw bytes in body
      3. application/json    — {"image": "<base64-string>"}
    """
    image_bytes = _extract_image_bytes(request)
    if image_bytes is None:
        return jsonify({"error": "No image provided. Send multipart field 'image', "
                                 "raw bytes as octet-stream, or JSON {\"image\": \"<base64>\"}."}), 400

    try:
        result = get_detector().detect(image_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Detection failed")
        return jsonify({"error": "Internal inference error.", "detail": str(exc)}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_image_bytes(req) -> bytes | None:
    """Pull raw image bytes out of the request regardless of encoding."""
    # 1. multipart/form-data
    if "image" in req.files:
        return req.files["image"].read()

    # 2. raw binary body
    content_type = req.content_type or ""
    if "octet-stream" in content_type:
        data = req.get_data()
        return data if data else None

    # 3. JSON with base64-encoded image
    if "json" in content_type:
        payload = req.get_json(silent=True) or {}
        b64 = payload.get("image")
        if b64:
            try:
                return base64.b64decode(b64)
            except Exception:
                return None

    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
