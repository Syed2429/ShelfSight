from flask import Flask, request, jsonify
from visualizer import Visualizer
import base64
import os

app = Flask(__name__)
visualizer = Visualizer()
PORT = int(os.getenv("PORT", 5003))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "visualization"})


@app.route("/visualize", methods=["POST"])
def visualize():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400
        if "image_base64" not in data:
            return jsonify({"error": "Missing image_base64"}), 400
        if "grouped_products" not in data:
            return jsonify({"error": "Missing grouped_products"}), 400

        image_bytes = base64.b64decode(data["image_base64"])
        grouped_products = data["grouped_products"]

        vis_bytes, output_path = visualizer.draw_groups(image_bytes, grouped_products)
        vis_base64 = base64.b64encode(vis_bytes).decode("utf-8")

        return jsonify({
            "visualization_base64": vis_base64,
            "output_path": output_path,
            "num_groups": len(set(p["group_id"] for p in grouped_products)),
            "total_products": len(grouped_products)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
