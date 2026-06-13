import base64
import io
import math
import os
import sys
import uuid

import cv2
import numpy as np
from PIL import Image

SAM_AVAILABLE = False

MODEL_CACHE = {}


def _encode_mask_png(mask):
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _fallback_segments(image_np):
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    min_area = max(200, (h * w) // 2000)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 15 or bh < 15:
            continue
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] else x + bw // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] else y + bh // 2
        
        orientation = 0.0
        if bw and bh:
            orientation = float(math.degrees(math.atan2(bh, bw)))
        
        segments.append({
            'object_id': f'obj_{uuid.uuid4().hex[:8]}',
            'mask': mask,
            'mask_base64': _encode_mask_png(mask),
            'centroid': (cx, cy),
            'bbox': (x, y, bw, bh),
            'area': int(area),
            'orientation': round(orientation, 2)
        })
    
    # Ensure we have at least one segment
    if not segments:
        mask = np.ones((h, w), dtype=np.uint8)
        segments = [{
            'object_id': f'obj_{uuid.uuid4().hex[:8]}',
            'mask': mask,
            'mask_base64': _encode_mask_png(mask),
            'centroid': (w // 2, h // 2),
            'bbox': (0, 0, w, h),
            'area': w * h,
            'orientation': 0.0
        }]
    
    return segments[:30]


def load_sam_generator(model_type='vit_b', checkpoint_path=None):
    if not SAM_AVAILABLE:
        return None
    key = (model_type, checkpoint_path)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if checkpoint_path and os.path.exists(checkpoint_path):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    else:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    MODEL_CACHE[key] = mask_generator
    return mask_generator


def segment_image(image_np):
    if SAM_AVAILABLE:
        generator = load_sam_generator()
        if generator is not None:
            sam_results = generator.generate(image_np)
            objects = []
            for item in sam_results:
                mask = item['segmentation'].astype(np.uint8)
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                x0, y0, x1, y1 = item['bbox']
                area = int(np.count_nonzero(mask))
                centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
                orientation = 0.0
                objects.append({
                    'object_id': f'obj_{uuid.uuid4().hex[:8]}',
                    'mask': mask,
                    'mask_base64': _encode_mask_png(mask),
                    'centroid': centroid,
                    'bbox': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                    'area': area,
                    'orientation': orientation
                })
            return objects
    return _fallback_segments(image_np)
