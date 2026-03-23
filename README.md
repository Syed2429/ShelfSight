# Retail Shelf AI Pipeline

An end-to-end AI pipeline that detects products on retail shelf images, groups them by brand/visual similarity, and returns annotated visualizations with a structured JSON response.

---
---

## What is this?

ShelfSight is a production-grade retail shelf analysis pipeline that 
takes a photo of a retail shelf and automatically:

- **Detects every product** on the shelf using YOLOv10 trained on 
  the SKU-110K retail dataset, with SAHI sliced inference for dense 
  detection
- **Groups products by brand** using color + spatial clustering — 
  same brand products get the same color bounding box
- **Returns a visual result** — an annotated image with color-coded 
  bounding boxes and group labels
- **Returns structured JSON** — product count, group assignments, 
  bounding box coordinates, and timing breakdown

### Example Output

| Image | Products Detected | Brand Groups | Time |
|---|---|---|---|
| Salad dressing shelf | 225 | 26 | 22s |
| Hair care shelf | 42 | 10 | 15s |
| Biscuit shelf | 125 | 19 | 18s |

### Use Cases
- **Planogram compliance** — verify products are stocked correctly
- **Out-of-stock detection** — identify empty shelf gaps
- **Retail analytics** — count SKUs and brand distribution per shelf
- **Inventory auditing** — automated shelf scanning without manual counting

---

## Quick Start

```bash
# Clone the repo
git clone <your-repo-url>
cd retail-ai-pipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Add your trained weights (optional)
# Place best.pt in services/detection/best.pt
# Without it, falls back to HuggingFace retail model automatically

# Start all 4 services in separate terminals
.venv\Scripts\python services\detection\app.py     # Terminal 1 — port 5001
.venv\Scripts\python services\grouping\app.py      # Terminal 2 — port 5002
.venv\Scripts\python services\visualization\app.py # Terminal 3 — port 5003
.venv\Scripts\python gateway\app.py                # Terminal 4 — port 5000

# Open browser
# http://localhost:5000
```

---

## Architecture

```
Browser / Client
      │  POST /analyze (multipart image)
      ▼
┌─────────────┐  /detect  ┌───────────────┐
│   Gateway   │ ────────► │   Detection   │  (YOLOv10 + SAHI)
│  port 5000  │ ◄──────── │   port 5001   │
│             │  /group   ├───────────────┤
│             │ ────────► │   Grouping    │  (AgglomerativeClustering)
│             │ ◄──────── │   port 5002   │
│             │ /visualize├───────────────┤
│             │ ────────► │ Visualization │  (OpenCV color bboxes)
│             │ ◄──────── │   port 5003   │
└─────────────┘           └───────────────┘
      │  JSON response + visualization image
      ▼
   Browser
```

| Service       | Port | Responsibility                                    |
|---------------|------|---------------------------------------------------|
| gateway       | 5000 | Web UI, orchestrates detection → grouping → viz   |
| detection     | 5001 | SAHI dual-pass YOLOv10 product detection          |
| grouping      | 5002 | Agglomerative clustering by spatial+color features |
| visualization | 5003 | Draws color-coded bboxes, saves output images     |

---

## Setup & Run (Local — no Docker)

### Prerequisites

- Python 3.10 or 3.12
- Windows PowerShell or Linux/macOS terminal

### 1. Create virtual environment

```powershell
cd "D:\Projects -exp\Retail_pipeline"
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r retail-ai-pipeline\requirements.txt
```

### 3. Place model weights (optional)

If you have a custom-trained `best.pt`, copy it to:
```
retail-ai-pipeline\services\detection\best.pt
```
If not present, the detection service automatically downloads `foduucom/product-detection-in-shelf-yolov8` from HuggingFace on first run.

### 4. Start the 4 services (4 separate terminals)

From `D:\Projects -exp\Retail_pipeline`:

```powershell
# Terminal 1 — Detection
.venv\Scripts\python retail-ai-pipeline\services\detection\app.py

# Terminal 2 — Grouping
.venv\Scripts\python retail-ai-pipeline\services\grouping\app.py

# Terminal 3 — Visualization
.venv\Scripts\python retail-ai-pipeline\services\visualization\app.py

# Terminal 4 — Gateway
.venv\Scripts\python retail-ai-pipeline\gateway\app.py
```

### 5. Open the web UI

Navigate to **http://localhost:5000** in your browser, upload a shelf image, and click **Analyze**.

---

## Setup & Run (Docker)

```bash
cd retail-ai-pipeline
docker-compose up --build
```

Then open **http://localhost:5000**.

---

## Run End-to-End Test (no services needed)

```powershell
cd "D:\Projects -exp\Retail_pipeline"
.venv\Scripts\python retail-ai-pipeline\tests\test_visualization.py "retail-ai-pipeline\testing images\sample_images\<your_image.jpg>"
```

Output is saved to `retail-ai-pipeline\outputs\test_full_pipeline.jpg`.

---

## API Reference

### Gateway — `POST /analyze`

**Request:** `multipart/form-data`, field `image` (JPEG/PNG, max 15 MB)

**Response:**
```json
{
  "status": "success",
  "image_id": "uuid-string",
  "products_detected": 39,
  "num_groups": 8,
  "groups": [
    {
      "group_id": 0,
      "products": [
        { "product_id": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.87 },
        { "product_id": 3, "bbox": [x1, y1, x2, y2], "confidence": 0.72 }
      ]
    }
  ],
  "visualization_base64": "<base64-encoded JPEG>",
  "output_path": "/app/outputs/visualization_output.jpg",
  "timing": {
    "detection_ms": 4200,
    "grouping_ms": 120,
    "visualization_ms": 80,
    "total_ms": 4400
  }
}
```

---

### Detection Service — `POST /detect`

**Request formats (any one):**
- `multipart/form-data` with field `image`
- `application/octet-stream` raw bytes
- `application/json` → `{ "image": "<base64>" }`

**Response:**
```json
{
  "count": 39,
  "image_dims": [1920, 1080],
  "detections": [
    {
      "product_id": 0,
      "bbox": [120, 45, 210, 180],
      "confidence": 0.87,
      "label": "product",
      "area": 12150
    }
  ]
}
```

---

### Grouping Service — `POST /group`

**Request:**
```json
{
  "image_base64": "<base64>",
  "detections": [ { "product_id": 0, "bbox": [...], "confidence": 0.87, ... } ],
  "image_dims": [1920, 1080]
}
```

**Response:**
```json
{
  "num_groups": 8,
  "grouped_products": [
    {
      "product_id": 0,
      "bbox": [120, 45, 210, 180],
      "confidence": 0.87,
      "label": "product",
      "group_id": 2
    }
  ]
}
```

---

### Visualization Service — `POST /visualize`

**Request:**
```json
{
  "image_base64": "<base64>",
  "grouped_products": [ { "product_id": 0, "bbox": [...], "group_id": 2 } ]
}
```

**Response:**
```json
{
  "visualization_base64": "<base64 JPEG>",
  "output_path": "/app/outputs/visualization_output.jpg",
  "num_groups": 8,
  "total_products": 39
}
```

---

## Technical Approach

### Detection — YOLOv10 + SAHI

- **Model:** Custom YOLOv10 weights trained on SKU-110K retail shelf dataset (`services/detection/best.pt`)
- **SAHI sliced inference:** Two passes — 256×256 tiles (overlap 0.3) + 480×480 tiles (overlap 0.2) — catches both small and large products on high-res shelf images
- **Post-processing:** NMS (IoU=0.45), area filter (>500 px²), aspect ratio filter (0.15 < w/h < 6.0)
- **Fallback:** If `best.pt` is absent, downloads `foduucom/product-detection-in-shelf-yolov8` from HuggingFace automatically
- **Why SAHI with YOLOv10?** SAHI uses the ultralytics interface (`model_type="yolov8"`) which is compatible with YOLOv10 weights — no code changes needed to benefit from SAHI's sliced inference

**Why SAHI?** High-res shelf images with densely packed small products benefit greatly from sliced inference — a single full-image pass misses ~40% of products.

**Alternatives considered:**
- Generic COCO YOLOv8 (`yolov8s.pt`) — detects products but with many false positives on shelf labels and price tags
- Two-stage R-CNN (Faster-RCNN) — higher accuracy but too slow for CPU inference
- EfficientDet — competitive, but SAHI+YOLOv10 ecosystem is better supported

### Grouping — Agglomerative Clustering (Spatial + Color)

- **Feature extraction:** 132-dim vector per product:
  - 128-dim HSV histogram from center 60% of bbox (hue×2 weight, saturation, value) — normalized
  - 4 spatial features: `x_center×1.5`, `y_center×0.5`, `w_ratio`, `h_ratio` — x weighted higher because same-brand products are stocked together **horizontally**
- **Algorithm:** `AgglomerativeClustering(distance_threshold=0.25, metric='cosine', linkage='average')` — no need to specify k
- **Post-processing:** Singletons merged into nearest non-singleton cluster using cosine similarity

**Why spatial+color?** Pure color histograms fail when different brands share similar packaging colors (e.g., two orange-labeled products). Adding x-position as a feature leverages the shelf planogram — products of the same brand are placed adjacent to each other.

**Tuning:** Raise `distance_threshold` to 0.35–0.45 for fewer, coarser groups; lower to 0.15–0.20 for finer brand splits.

**Alternatives considered:**
- Pure color histogram — groups by color only, splits brands with similar packaging
- KMeans with elbow method — over-splits (often k=2)
- KMeans with silhouette score — better but still requires k estimation
- DBSCAN — parameter-sensitive, creates too many noise points on dense shelves

### Visualization — OpenCV

- 12-color palette, each group gets a unique color
- Filled label backgrounds with white `G{id}` text for readability
- Saves to `outputs/visualization_output.jpg` (JPEG quality 92)

### Scalability

- Each service is independently scalable — add more detection workers behind a load balancer
- Docker Compose `healthcheck` ensures services are ready before gateway starts
- Service URLs are environment-variable-driven (`DETECTION_URL`, `GROUPING_URL`, `VISUALIZATION_URL`) — drop-in for Kubernetes or cloud deployments
- All inter-service communication is base64 JSON over HTTP — stateless, horizontally scalable

---

## Project Structure

```
retail-ai-pipeline/
├── docker-compose.yml
├── requirements.txt
├── README.md
├── gateway/
│   ├── app.py              # Web UI + /analyze orchestrator (port 5000)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── utils/image_utils.py
├── services/
│   ├── detection/
│   │   ├── app.py          # Flask wrapper (port 5001)
│   │   ├── detector.py     # SAHI + YOLOv10 inference
│   │   ├── best.pt         # Custom weights (optional, auto-downloaded if absent)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── grouping/
│   │   ├── app.py          # Flask wrapper (port 5002)
│   │   ├── grouper.py      # AgglomerativeClustering grouper
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── visualization/
│       ├── app.py          # Flask wrapper (port 5003)
│       ├── visualizer.py   # OpenCV bbox drawing
│       ├── Dockerfile
│       └── requirements.txt
├── outputs/                # Saved visualization images
├── tests/
│   └── test_visualization.py  # End-to-end pipeline test
└── testing images/
    └── sample_images/      # Input test images
```

