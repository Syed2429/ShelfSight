from flask import Flask, request, jsonify, render_template_string
import requests
import base64
import time
import uuid
import os

app = Flask(__name__)

DETECTION_URL     = os.getenv("DETECTION_URL",     "http://localhost:5001")
GROUPING_URL      = os.getenv("GROUPING_URL",      "http://localhost:5002")
VISUALIZATION_URL = os.getenv("VISUALIZATION_URL", "http://localhost:5003")
PORT              = int(os.getenv("PORT", 5000))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Retail AI Pipeline</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .upload-box { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
        input[type=file] { margin: 10px 0; }
        button { background: #0066cc; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0052a3; }
        button:disabled { background: #aaa; cursor: not-allowed; }
        #result { display: none; }
        #vis-image { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        #json-output { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; overflow-x: auto; font-family: monospace; font-size: 13px; max-height: 400px; overflow-y: auto; }
        .stats { display: flex; gap: 20px; margin: 15px 0; }
        .stat-card { background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .stat-card h3 { margin: 0; font-size: 28px; color: #0066cc; }
        .stat-card p { margin: 5px 0 0; color: #666; font-size: 13px; }
        #loading { display: none; color: #666; font-style: italic; margin: 10px 0; }
        .error { color: red; background: #fff0f0; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>&#x1F6D2; Retail Shelf AI Pipeline</h1>
    <div class="upload-box">
        <h2>Upload Shelf Image</h2>
        <input type="file" id="imageInput" accept="image/*">
        <br>
        <button onclick="analyzeImage()" id="analyzeBtn">Analyze</button>
        <div id="loading">&#x23F3; Processing... (this may take 30-60 seconds on CPU)</div>
    </div>

    <div id="result">
        <div class="stats">
            <div class="stat-card"><h3 id="stat-products">-</h3><p>Products Detected</p></div>
            <div class="stat-card"><h3 id="stat-groups">-</h3><p>Brand Groups</p></div>
            <div class="stat-card"><h3 id="stat-time">-</h3><p>Processing Time</p></div>
        </div>
        <img id="vis-image" src="" alt="Visualization">
        <h3>JSON Response</h3>
        <pre id="json-output"></pre>
    </div>

    <script>
        async function analyzeImage() {
            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files[0]) { alert('Please select an image first'); return; }

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.error) {
                    document.getElementById('json-output').innerHTML =
                        '<span class="error">Error: ' + data.error + '</span>';
                    document.getElementById('result').style.display = 'block';
                    return;
                }

                document.getElementById('stat-products').textContent = data.products_detected;
                document.getElementById('stat-groups').textContent = data.num_groups;
                document.getElementById('stat-time').textContent = data.processing_time_ms + 'ms';
                document.getElementById('vis-image').src =
                    'data:image/jpeg;base64,' + data.visualization_base64;

                const display = {...data};
                display.visualization_base64 = '[base64 image data]';
                document.getElementById('json-output').textContent =
                    JSON.stringify(display, null, 2);
                document.getElementById('result').style.display = 'block';
            } catch(e) {
                alert('Request failed: ' + e.message);
            } finally {
                btn.disabled = false;
                document.getElementById('loading').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/health", methods=["GET"])
def health():
    services = {}
    for name, url in [("detection", DETECTION_URL),
                      ("grouping", GROUPING_URL),
                      ("visualization", VISUALIZATION_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            services[name] = r.json()
        except Exception as e:
            services[name] = {"status": "unreachable", "error": str(e)}
    return jsonify({"status": "ok", "services": services})


@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()
    image_id = uuid.uuid4().hex

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return jsonify({"error": f"Invalid file type: {ext}"}), 400

    image_bytes = file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        return jsonify({"error": "Image too large (max 15MB)"}), 400

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # Step 1: Detection
        t1 = time.time()
        det_response = requests.post(
            f"{DETECTION_URL}/detect",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=120
        )
        det_response.raise_for_status()
        detection_result = det_response.json()
        detection_time = round((time.time() - t1) * 1000)

        detections  = detection_result["detections"]
        image_dims  = detection_result["image_dims"]
        print(f"[Gateway] Detection: {len(detections)} products in {detection_time}ms")

        # Step 2: Grouping
        t2 = time.time()
        grp_response = requests.post(
            f"{GROUPING_URL}/group",
            json={"image_base64": image_base64,
                  "detections":   detections,
                  "image_dims":   image_dims},
            timeout=120
        )
        grp_response.raise_for_status()
        grouping_result = grp_response.json()
        grouping_time   = round((time.time() - t2) * 1000)

        grouped_products = grouping_result["grouped_products"]
        num_groups       = grouping_result["num_groups"]
        print(f"[Gateway] Grouping: {num_groups} groups in {grouping_time}ms")

        # Step 3: Visualization
        t3 = time.time()
        vis_response = requests.post(
            f"{VISUALIZATION_URL}/visualize",
            json={"image_base64":    image_base64,
                  "grouped_products": grouped_products},
            timeout=60
        )
        vis_response.raise_for_status()
        vis_result = vis_response.json()
        vis_time   = round((time.time() - t3) * 1000)

        total_time = round((time.time() - start_time) * 1000)
        print(f"[Gateway] Total: {total_time}ms")

        groups_summary = {}
        for p in grouped_products:
            gid = p["group_id"]
            if gid not in groups_summary:
                groups_summary[gid] = []
            groups_summary[gid].append({
                "product_id": p["product_id"],
                "bbox":       p["bbox"],
                "confidence": p["confidence"]
            })

        return jsonify({
            "status":               "success",
            "image_id":             image_id,
            "products_detected":    len(grouped_products),
            "num_groups":           num_groups,
            "groups": [
                {"group_id": gid, "products": prods}
                for gid, prods in sorted(groups_summary.items())
            ],
            "visualization_base64": vis_result["visualization_base64"],
            "output_path":          vis_result["output_path"],
            "timing": {
                "detection_ms":     detection_time,
                "grouping_ms":      grouping_time,
                "visualization_ms": vis_time,
                "total_ms":         total_time
            },
            "processing_time_ms": total_time
        })

    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Service timeout"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
