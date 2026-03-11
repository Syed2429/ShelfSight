import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


class Grouper:
    def __init__(self):
        print("[Grouper] Spatial+color grouper ready")

    def _extract_features(self, image: np.ndarray,
                          bbox: list, img_w: int, img_h: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox

        # Crop upper 65% of height, trim 8% from each side — captures label area
        margin_x = int((x2 - x1) * 0.08)
        cx1 = max(0, x1 + margin_x)
        cx2 = min(image.shape[1], x2 - margin_x)
        cy1 = y1
        cy2 = min(image.shape[0], y1 + int((y2 - y1) * 0.65))

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(132)

        crop = cv2.resize(crop, (48, 48))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        h_hist = h_hist / (h_hist.sum() + 1e-6)
        s_hist = s_hist / (s_hist.sum() + 1e-6)
        v_hist = v_hist / (v_hist.sum() + 1e-6)

        color_feat = np.concatenate([h_hist * 2.5, s_hist, v_hist])  # 128-dim

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        w_ratio = (x2 - x1) / img_w
        h_ratio = (y2 - y1) / img_h

        spatial_feat = np.array([
            x_center * 0.4,   # moderate x influence — same brand stocked horizontally
            y_center * 0.15,  # small y influence
            w_ratio,
            h_ratio
        ])

        return np.concatenate([color_feat, spatial_feat])  # 132-dim

    def group(self, detections: list, image_bytes: bytes) -> dict:
        if not detections:
            return {"groups": [], "num_groups": 0}

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]

        print(f"[Grouper] Extracting features for {len(detections)} products...")

        if len(detections) == 1:
            detections[0]["group_id"] = 0
            return {"groups": [{"group_id": 0, "products": detections}],
                    "num_groups": 1}

        features = []
        for det in detections:
            feat = self._extract_features(image, det["bbox"], img_w, img_h)
            features.append(feat)

        X = normalize(np.array(features), norm='l2')

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.22,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(X)

        # Merge singleton groups into nearest neighbour
        from collections import Counter
        counts = Counter(labels)
        singletons = {lbl for lbl, cnt in counts.items() if cnt == 1}

        if singletons and len(detections) > 3:
            non_singleton_mask = np.array(
                [labels[i] not in singletons for i in range(len(labels))])

            if non_singleton_mask.sum() > 0:
                non_singleton_X = X[non_singleton_mask]
                non_singleton_labels = labels[non_singleton_mask]

                for i, lbl in enumerate(labels):
                    if lbl in singletons:
                        dists = np.dot(non_singleton_X, X[i])
                        nearest = np.argmax(dists)
                        # Only merge if actually similar — don't force different brands together
                        if dists[nearest] >= 0.75:
                            labels[i] = non_singleton_labels[nearest]

        # Remap to sequential IDs
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = [remap[l] for l in labels]

        groups_dict = {}
        for det, gid in zip(detections, labels):
            det["group_id"] = int(gid)
            if gid not in groups_dict:
                groups_dict[gid] = []
            groups_dict[gid].append(det)

        groups = [{"group_id": gid, "products": prods}
                  for gid, prods in sorted(groups_dict.items())]

        num_groups = len(groups)
        print(f"[Grouper] {len(detections)} products \u2192 {num_groups} groups")
        for g in groups:
            print(f"  G{g['group_id']}: {len(g['products'])} products")

        return {"groups": groups, "num_groups": num_groups}
