import os
import cv2
import numpy as np
import time
import gc
from collections import OrderedDict

from core.db import query_db

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None


class CVEngine:
    def __init__(self, weights_path='yolov8n.pt'):
        self.weights_path = weights_path
        self.weights_paths = []
        self.model = None
        self.models = []
        self.model_specs = []
        self.model_cache = OrderedDict()
        self.max_cached_models = 1
        self.scan_zone = {
            'x1': 0.08,
            'y1': 0.08,
            'x2': 0.92,
            'y2': 0.94,
        }
        self.model_confidence = 0.20
        self.accept_confidence = 0.30
        self.product_match_confidence = 0.30
        self.min_box_area = 0.003
        self.max_box_area = 0.75
        self.min_zone_overlap = 0.45
        self.reload_models(default_path=weights_path)

    def _resolve_model_specs(self, default_path):
        specs = []
        seen_paths = set()
        seen_products = set()

        def add_spec(path, product_id=None, product_name=None, model_class_name=None):
            normalized = os.path.abspath(path) if path != 'yolov8n.pt' else path
            if normalized in seen_paths:
                return
            seen_paths.add(normalized)
            specs.append({
                'path': normalized,
                'product_id': product_id,
                'product_name': product_name,
                'model_class_name': model_class_name,
            })

        try:
            versions = query_db(
                """
                SELECT
                    mv.weights_path,
                    tj.product_id,
                    p.product_name,
                    p.model_class_name
                FROM model_versions mv
                JOIN training_jobs tj ON tj.job_id = mv.job_id
                JOIN products p ON p.product_id = tj.product_id
                WHERE mv.weights_path IS NOT NULL
                  AND mv.weights_path != ''
                  AND tj.product_id IS NOT NULL
                  AND tj.status = 'success'
                ORDER BY mv.version_id DESC
                """
            )
            for version in versions:
                product_id = version['product_id']
                path = version['weights_path']
                if not product_id or product_id in seen_products:
                    continue
                if path and os.path.exists(path):
                    add_spec(
                        path,
                        product_id=product_id,
                        product_name=version['product_name'],
                        model_class_name=version['model_class_name'],
                    )
                    seen_products.add(product_id)
        except Exception:
            pass

        if not specs:
            fallback_paths = [
                os.path.join('runs', 'detect', 'train', 'weights', 'best.pt'),
                os.path.join('runs', 'detect', 'train_job_1', 'weights', 'best.pt'),
                os.path.join('runs', 'detect', 'runs', 'detect', 'train_job_2', 'weights', 'best.pt'),
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    add_spec(path)

        if not specs:
            add_spec(default_path)

        return specs

    def _resolve_weights_path(self, default_path):
        specs = self._resolve_model_specs(default_path)
        return specs[0]['path']

    def load_models(self, specs):
        if not YOLO:
            print("WARNING: ultralytics is not installed. Inference will be mocked.")
            self.model_specs = list(specs)
            self.weights_paths = [spec['path'] for spec in specs]
            self.weights_path = self.weights_paths[0] if self.weights_paths else 'yolov8n.pt'
            return

        self._clear_model_cache()
        loaded_paths = []
        loaded_specs = []

        for spec in specs:
            path = spec['path']
            try:
                if not os.path.exists(path) and path != 'yolov8n.pt':
                    print(f"Weights {path} not found. Skipping.")
                    continue

                model_path = path if os.path.exists(path) else 'yolov8n.pt'
                loaded_paths.append(model_path)
                loaded_specs.append({
                    **spec,
                    'path': model_path,
                })
                print(f"Model {model_path} registered successfully.")
            except Exception as e:
                print(f"Error loading YOLO model {path}: {e}")

        self.model_specs = loaded_specs
        self.weights_paths = loaded_paths or [spec['path'] for spec in specs]
        self.weights_path = self.weights_paths[0] if self.weights_paths else 'yolov8n.pt'
        self.model = self._get_model(self.weights_path) if self.model_specs else None
        self.models = [self.model] if self.model else []

    def load_model(self, path):
        self.load_models([{'path': path}])

    def reload_models(self, default_path='yolov8n.pt'):
        specs = self._resolve_model_specs(default_path)
        self.load_models(specs)

    def get_scan_zone(self):
        return dict(self.scan_zone)

    def _clear_model_cache(self):
        self.model_cache.clear()
        self.model = None
        self.models = []
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model(self, path):
        if not YOLO:
            return None

        cached_model = self.model_cache.pop(path, None)
        if cached_model is not None:
            self.model_cache[path] = cached_model
            self.model = cached_model
            self.models = list(self.model_cache.values())
            return cached_model

        try:
            model = YOLO(path)
        except Exception as e:
            print(f"Error loading YOLO model {path}: {e}")
            return None

        while len(self.model_cache) >= self.max_cached_models:
            _, evicted_model = self.model_cache.popitem(last=False)
            del evicted_model
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model_cache[path] = model
        self.model = model
        self.models = list(self.model_cache.values())
        print(f"Model {path} loaded successfully.")
        return model

    def _crop_to_scan_zone(self, img):
        height, width = img.shape[:2]
        zone = self.scan_zone
        x1 = max(0, min(width - 1, int(round(zone['x1'] * width))))
        y1 = max(0, min(height - 1, int(round(zone['y1'] * height))))
        x2 = max(x1 + 1, min(width, int(round(zone['x2'] * width))))
        y2 = max(y1 + 1, min(height, int(round(zone['y2'] * height))))
        return img[y1:y2, x1:x2], {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    def _intersection_area(self, a, b):
        left = max(a['x1'], b['x1'])
        top = max(a['y1'], b['y1'])
        right = min(a['x2'], b['x2'])
        bottom = min(a['y2'], b['y2'])
        if right <= left or bottom <= top:
            return 0.0
        return (right - left) * (bottom - top)

    def _accept_candidate(self, candidate):
        bbox = candidate.get('bbox') or {}
        width = max(0.0, bbox.get('x2', 0.0) - bbox.get('x1', 0.0))
        height = max(0.0, bbox.get('y2', 0.0) - bbox.get('y1', 0.0))
        area = width * height
        if area <= 0:
            return False

        center_x = bbox.get('x1', 0.0) + width / 2
        center_y = bbox.get('y1', 0.0) + height / 2
        if not (self.scan_zone['x1'] <= center_x <= self.scan_zone['x2']):
            return False
        if not (self.scan_zone['y1'] <= center_y <= self.scan_zone['y2']):
            return False

        if candidate['confidence'] < self.accept_confidence:
            return False
        if candidate.get('product_id') and candidate['confidence'] < self.product_match_confidence:
            return False
        if area < self.min_box_area or area > self.max_box_area:
            return False

        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio < 0.10 or aspect_ratio > 8.0:
            return False

        zone_overlap = self._intersection_area(bbox, self.scan_zone) / area
        if zone_overlap < self.min_zone_overlap:
            return False

        return True

    def _resolved_detection_class_name(self, spec, raw_class_name):
        preferred_name = spec.get('model_class_name') or spec.get('product_name')
        if preferred_name:
            return preferred_name
        return raw_class_name

    def _bbox_iou(self, a, b):
        intersection = self._intersection_area(a, b)
        if intersection <= 0:
            return 0.0

        area_a = max(0.0, a['x2'] - a['x1']) * max(0.0, a['y2'] - a['y1'])
        area_b = max(0.0, b['x2'] - b['x1']) * max(0.0, b['y2'] - b['y1'])
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _suppress_overlapping_detections(self, detections, iou_threshold=0.45):
        kept = []
        for item in sorted(detections, key=lambda entry: entry['confidence'], reverse=True):
            bbox = item.get('bbox') or {}
            if any(self._bbox_iou(bbox, kept_item.get('bbox') or {}) >= iou_threshold for kept_item in kept):
                continue
            kept.append(item)
        return kept

    def infer(self, image_bytes):
        """
        Runs YOLO inference on an image bytes array.
        Returns a list of dicts: [{'class_name': str, 'confidence': float}]
        """
        if not self.model_specs:
            # Note: For prototype testing without a trained model or if ultralytics fails,
            # we can inject a mock detection if the user uploads any image.
            # In a real app we simply return []
            return [{"class_name": "mock_product", "confidence": 0.99}]

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []

        roi, roi_bounds = self._crop_to_scan_zone(img)
        if roi.size == 0:
            return []

        detected = []
        orig_h, orig_w = img.shape[:2]

        for spec in self.model_specs:
            model = self._get_model(spec['path'])
            if model is None:
                continue

            results = model.predict(roi, conf=self.model_confidence, verbose=False)
            best_detection = None
            for r in results:
                boxes = sorted(r.boxes, key=lambda box: float(box.conf[0]), reverse=True) if r.boxes is not None else []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    raw_class_name = model.names[cls_id]
                    class_name = self._resolved_detection_class_name(spec, raw_class_name)
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    full_x1 = roi_bounds['x1'] + x1
                    full_y1 = roi_bounds['y1'] + y1
                    full_x2 = roi_bounds['x1'] + x2
                    full_y2 = roi_bounds['y1'] + y2
                    candidate = {
                        'class_name': class_name,
                        'confidence': conf,
                        'product_id': spec.get('product_id'),
                        'product_name': spec.get('product_name'),
                        'model_class_name': spec.get('model_class_name'),
                        'raw_class_name': raw_class_name,
                        'bbox': {
                            'x1': max(0.0, min(1.0, full_x1 / orig_w)) if orig_w else 0.0,
                            'y1': max(0.0, min(1.0, full_y1 / orig_h)) if orig_h else 0.0,
                            'x2': max(0.0, min(1.0, full_x2 / orig_w)) if orig_w else 0.0,
                            'y2': max(0.0, min(1.0, full_y2 / orig_h)) if orig_h else 0.0,
                        }
                    }
                    if not self._accept_candidate(candidate):
                        continue
                    if not best_detection or conf > best_detection['confidence']:
                        best_detection = candidate
            if best_detection:
                detected.append(best_detection)

        detected = self._suppress_overlapping_detections(detected)

        by_product = {}
        by_class = {}
        for item in detected:
            product_id = item.get('product_id')
            if product_id:
                previous = by_product.get(product_id)
                if not previous or item['confidence'] > previous['confidence']:
                    by_product[product_id] = item
            else:
                class_key = item['class_name']
                previous = by_class.get(class_key)
                if not previous or item['confidence'] > previous['confidence']:
                    by_class[class_key] = item

        return list(by_product.values()) + list(by_class.values())

cv_engine = CVEngine()
