import numpy as np
from PIL import Image
import cv2

LABELS = [
    'car', 'traffic signal', 'pedestrian', 'motorcycle', 'bus', 'truck',
    'bicycle', 'person', 'chair', 'table', 'laptop', 'bottle', 'tree', 'pole', 'building'
]

def classify_object(image_np, mask):
    """Fast classification using shape and color analysis."""
    return _smart_fallback_classify(image_np, mask)

def extract_object_region(image_np, mask):
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return Image.fromarray(image_np)
    x0, x1 = x_coords.min(), x_coords.max()
    y0, y1 = y_coords.min(), y_coords.max()
    crop = image_np[y0:y1 + 1, x0:x1 + 1]
    return Image.fromarray(crop)

def _smart_fallback_classify_impl(region_img):
    """Classify using shape and color features."""
    region = np.array(region_img)
    if region.size == 0:
        return 'object', 0.50
    
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return 'object', 0.50
    
    aspect = w / (h + 1e-5)
    
    # Analyze edges
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-5) if edges.size > 0 else 0
    
    # Color analysis
    avg_val = np.mean(gray)
    if len(region.shape) == 3:
        red_mask = (region[:,:,0] > 150).sum()
        green_mask = (region[:,:,1] > 150).sum()
        total_pixels = region.size / 3
        red_ratio = red_mask / (total_pixels + 1e-5)
        green_ratio = green_mask / (total_pixels + 1e-5)
    else:
        red_ratio = 0
        green_ratio = 0
    
    # Decision logic
    if red_ratio > 0.25 and green_ratio < 0.15:
        return 'traffic signal', 0.75
    if aspect < 0.4 and h > w and edge_density > 0.15:
        return 'person', 0.70
    if 0.8 < aspect < 1.5 and edge_density < 0.2:
        return 'car', 0.75
    if aspect > 1.8 and edge_density < 0.18:
        return 'bus', 0.68
    if aspect < 0.7 and edge_density > 0.22:
        return 'motorcycle', 0.72
    if green_ratio > 0.35:
        return 'tree', 0.65
    if edge_density > 0.25:
        return 'bicycle', 0.62
    if avg_val > 200:
        return 'pole', 0.60
    
    return 'object', 0.55

def _smart_fallback_classify(image_np, mask):
    """Classify object from image and mask."""
    try:
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return 'object', 0.50
        
        x0, x1 = x_coords.min(), x_coords.max()
        y0, y1 = y_coords.min(), y_coords.max()
        crop = image_np[y0:y1 + 1, x0:x1 + 1]
        region = Image.fromarray(crop)
        return _smart_fallback_classify_impl(region)
    except Exception as e:
        print(f"Classification error: {e}")
        return 'object', 0.50
