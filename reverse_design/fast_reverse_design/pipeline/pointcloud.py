"""Back-project a masked depth map, scale by hand length, and clean with Open3D."""

from __future__ import annotations

from typing import Any, Tuple, Dict

import numpy as np
import open3d as o3d

from pipeline.exceptions import PipelineError
from pipeline.landmarks import MIDDLE_TIP_INDEX, WRIST_INDEX, HandLandmarks


def assumed_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """Pinhole camera guess for a phone-style hand photo."""
    focal = 1.2 * float(max(width, height))
    cx = width * 0.5
    cy = height * 0.5
    return focal, focal, cx, cy


def backproject(
    depth: np.ndarray,
    mask: np.ndarray,
    image_rgb: np.ndarray,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N, 3) points in camera space and (N, 3) RGB colors in 0..1."""
    h, w = mask.shape[:2]
    fx, fy, cx, cy = assumed_intrinsics(w, h)
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    valid = mask[ys, xs] > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = depth[ys.astype(int), xs.astype(int)]
    keep = np.isfinite(z) & (z > 1e-4)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if xs.size < 50:
        raise PipelineError("Too few masked pixels to build a point cloud.")
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = image_rgb[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
    return points, colors


def landmarks_to_3d(hand: HandLandmarks, depth: np.ndarray) -> np.ndarray:
    """Back-project each 2D landmark using a small median depth window."""
    h, w = depth.shape[:2]
    fx, fy, cx, cy = assumed_intrinsics(w, h)
    pts = np.zeros((21, 3), dtype=np.float32)
    radius = max(2, int(0.01 * max(h, w)))
    for i, (u, v) in enumerate(hand.points_px):
        ui, vi = int(round(u)), int(round(v))
        y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
        x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
        window = depth[y0:y1, x0:x1]
        z = float(np.median(window[np.isfinite(window)])) if window.size else 0.5
        pts[i, 0] = (u - cx) * z / fx
        pts[i, 1] = (v - cy) * z / fy
        pts[i, 2] = z
    return pts


def scale_to_hand_length(
    points: np.ndarray,
    landmarks_3d: np.ndarray,
    hand_length_m: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale so wrist → middle fingertip matches the measured hand length."""
    measured = float(
        np.linalg.norm(landmarks_3d[MIDDLE_TIP_INDEX] - landmarks_3d[WRIST_INDEX])
    )
    if measured < 1e-6:
        raise PipelineError("Could not measure 3D hand length from landmarks.")
    scale = hand_length_m / measured
    return points * scale, landmarks_3d * scale, scale


def clean_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.002,
) -> o3d.geometry.PointCloud:
    """Voxel downsample, statistical outlier removal, and normal estimation."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) < 30:
        raise PipelineError("Point cloud is too sparse after cleaning.")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    return pcd


def export_ply(pcd: o3d.geometry.PointCloud, path: str) -> None:
    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False)
    if not ok:
        raise PipelineError(f"Failed to write PLY: {path}")


def build_point_cloud(
    hand: HandLandmarks,
    depth: np.ndarray,
    hand_length_cm: float = 18.0,
) -> Dict[str, Any]:
    """Full point-cloud stage: back-project, scale, clean."""
    points, colors = backproject(depth, hand.mask, hand.image_rgb)
    lm3d = landmarks_to_3d(hand, depth)
    points, lm3d, scale = scale_to_hand_length(points, lm3d, hand_length_cm / 100.0)
    pcd = clean_point_cloud(points, colors)
    return {"pcd": pcd, "landmarks_3d": lm3d, "scale": scale}
