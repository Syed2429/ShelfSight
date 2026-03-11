from flask import Flask, request, jsonify
from grouper import Grouper
import os

app = Flask(__name__)
grouper = Grouper()
PORT = int(os.getenv("PORT", 5002))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "grouping",
                    "model": "agglomerative-cosine"})


@app.route("/group", methods=["POST"])
def group():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    missing = [f for f in ("image_base64", "detections", "image_dims") if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        import base64
        image_bytes = base64.b64decode(data["image_base64"])
        result = grouper.group(data["detections"], image_bytes)

        # Flatten groups → grouped_products for downstream compatibility
        grouped_products = []
        for g in result["groups"]:
            grouped_products.extend(g["products"])

        return jsonify({
            "grouped_products": grouped_products,
            "num_groups": result["num_groups"]
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
