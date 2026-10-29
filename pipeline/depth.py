"""Monocular depth on CPU (Depth Anything V2 Small, with MiDaS fallback)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from pipeline.exceptions import PipelineError

_MODEL_CACHE: Dict[str, Any] = {}


def estimate_depth(image_rgb: np.ndarray) -> np.ndarray:
    """Return a relative depth map (H, W) float32, larger = farther from camera."""
    try:
        return _depth_anything(image_rgb)
    except Exception as primary:
        try:
            return _midas_small(image_rgb)
        except Exception as fallback:
            raise PipelineError(
                "Could not run a depth model. Install torch and transformers "
                "(Depth Anything) or a working torch.hub MiDaS download. "
                f"Primary error: {primary}. Fallback error: {fallback}"
            ) from fallback


def _depth_anything(image_rgb: np.ndarray) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    key = "depth_anything"
    if key not in _MODEL_CACHE:
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.eval()
        _MODEL_CACHE[key] = (processor, model)

    processor, model = _MODEL_CACHE[key]
    image = Image.fromarray(image_rgb)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)
    return _normalize_depth(depth)


def _midas_small(image_rgb: np.ndarray) -> np.ndarray:
    import torch

    key = "midas_small"
    if key not in _MODEL_CACHE:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _MODEL_CACHE[key] = (model, transforms.small_transform)

    model, transform = _MODEL_CACHE[key]
    sample = transform(image_rgb)
    if sample.ndim == 3:
        sample = sample.unsqueeze(0)
    with torch.no_grad():
        prediction = model(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy().astype(np.float32)
    # MiDaS disparity: larger = closer. Convert to relative distance.
    disp = np.clip(depth, 1e-6, None)
    distance = 1.0 / disp
    return _normalize_depth(distance)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise PipelineError("Depth model returned an empty map.")
    lo, hi = np.percentile(finite, (1, 99))
    scaled = (depth - lo) / max(hi - lo, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    # Map to a plausible camera-distance range in metres before hand scaling.
    return (0.35 + scaled * 0.45).astype(np.float32)
