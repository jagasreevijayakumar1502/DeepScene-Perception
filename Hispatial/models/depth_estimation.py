import cv2
import numpy as np

MIDAS_AVAILABLE = False


def _init_midas_model():
    global _midas, _midas_transform
    if _midas is not None:
        return
    import torch
    import torchvision.transforms as transforms
    _midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    _midas.eval()
    _midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform


def estimate_depth_map(image_np):
    if MIDAS_AVAILABLE:
        try:
            if _midas is None or _midas_transform is None:
                _init_midas_model()
            import torch
            input_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            input_image = np.float32(input_image) / 255.0
            input_tensor = _midas_transform(input_image).unsqueeze(0)
            with torch.no_grad():
                prediction = _midas(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=image_np.shape[:2], mode='bicubic', align_corners=False
                ).squeeze()
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth
        except Exception:
            pass
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    depth = cv2.GaussianBlur(gray, (21, 21), 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return 1.0 - depth


def estimate_object_depth(depth_map, mask):
    masked = depth_map[mask > 0]
    if masked.size == 0:
        return float(depth_map.mean())
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) > 0:
        centroid_y = np.mean(y_coords)
        bottom_ratio = centroid_y / (mask.shape[0] + 1e-5)
        base_depth = float(masked.mean() * 25.0 + 2.0)
        proximity_boost = (1.0 - bottom_ratio) * 5.0
        return base_depth + proximity_boost
    return float(masked.mean() * 25.0 + 2.0)
