"""MediaPipe Hands landmarks and a simple hand mask."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from pipeline.exceptions import HandNotDetectedError, PipelineError

LANDMARK_NAMES: Tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

WRIST_INDEX = 0
MIDDLE_TIP_INDEX = 12
MCP_INDICES: Tuple[int, ...] = (5, 9, 13, 17)
PIP_INDICES: Tuple[int, ...] = (6, 10, 14, 18)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass
class HandLandmarks:
    """2D landmarks in pixel coordinates plus a binary hand mask."""

    image_rgb: np.ndarray
    points_px: np.ndarray  # (21, 2) float32
    normalized: np.ndarray  # (21, 3) x, y, z from MediaPipe
    mask: np.ndarray  # uint8 0/255
    handedness: str = "Unknown"
    confidence: Optional[float] = None
    names: Tuple[str, ...] = field(default_factory=lambda: LANDMARK_NAMES)

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    def as_json(self) -> Dict[str, Any]:
        points = []
        for i, name in enumerate(self.names):
            x, y = self.points_px[i]
            nx, ny, nz = self.normalized[i]
            points.append(
                {
                    "id": i,
                    "name": name,
                    "pixel": {"x": float(x), "y": float(y)},
                    "normalized": {"x": float(nx), "y": float(ny), "z": float(nz)},
                }
            )
        return {
            "handedness": self.handedness,
            "confidence": self.confidence,
            "image_size": {"width": self.width, "height": self.height},
            "landmarks": points,
        }


def load_rgb_image(path: str) -> np.ndarray:
    """Load an image as RGB uint8, applying EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        return np.asarray(rgb)


def extract_landmarks(image_path: str) -> HandLandmarks:
    """Run MediaPipe Hands and build a dilated convex-hull mask."""
    image_rgb = load_rgb_image(image_path)
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise PipelineError(
            "Could not import MediaPipe. Check its installation and dependencies: "
            f"{exc}"
        ) from exc

    if hasattr(mp, "solutions"):
        points_px, normalized, handedness, confidence = _extract_solutions(mp, image_rgb)
    else:
        points_px, normalized, handedness, confidence = _extract_tasks(mp, image_rgb)

    mask = _mask_from_landmarks(points_px, image_rgb.shape[0], image_rgb.shape[1])
    return HandLandmarks(
        image_rgb=image_rgb,
        points_px=points_px,
        normalized=normalized,
        mask=mask,
        handedness=handedness,
        confidence=confidence,
    )


def _extract_solutions(
    mp: Any, image_rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str, Optional[float]]:
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as hands:
        result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        raise HandNotDetectedError(_NO_HAND_MSG)
    hand_lms = result.multi_hand_landmarks[0]
    h, w = image_rgb.shape[:2]
    points_px = np.zeros((21, 2), dtype=np.float32)
    normalized = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(hand_lms.landmark):
        points_px[i, 0] = lm.x * w
        points_px[i, 1] = lm.y * h
        normalized[i] = (lm.x, lm.y, lm.z)
    handedness = "Unknown"
    confidence = None
    if result.multi_handedness:
        classification = result.multi_handedness[0].classification[0]
        handedness = classification.label
        confidence = float(classification.score)
    return points_px, normalized, handedness, confidence


def _ensure_task_model() -> Path:
    if _MODEL_PATH.is_file() and _MODEL_PATH.stat().st_size > 1000:
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    except Exception as exc:
        raise PipelineError(
            "Could not download the MediaPipe hand_landmarker.task model. "
            "Check your network connection."
        ) from exc
    return _MODEL_PATH


def _extract_tasks(
    mp: Any, image_rgb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str, Optional[float]]:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    model_path = _ensure_task_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=RunningMode.IMAGE,
    )
    rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        raise HandNotDetectedError(_NO_HAND_MSG)

    h, w = image_rgb.shape[:2]
    points_px = np.zeros((21, 2), dtype=np.float32)
    normalized = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(result.hand_landmarks[0]):
        points_px[i, 0] = lm.x * w
        points_px[i, 1] = lm.y * h
        normalized[i] = (lm.x, lm.y, lm.z)
    handedness = "Unknown"
    confidence = None
    if result.handedness:
        cats = result.handedness[0]
        if cats:
            handedness = cats[0].category_name
            confidence = float(cats[0].score)
    return points_px, normalized, handedness, confidence


def _mask_from_landmarks(points_px: np.ndarray, height: int, width: int) -> np.ndarray:
    hull = cv2.convexHull(points_px.astype(np.float32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    radius = max(8, int(0.06 * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


_NO_HAND_MSG = (
    "No hand detected. Please upload a clear photo of a single hand, "
    "with the palm or back of the hand facing the camera."
)
