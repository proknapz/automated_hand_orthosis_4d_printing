"""Deterministic voxel encoding for normalized hand geometry."""

from math import ceil
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


DEFAULT_RESOLUTION = 96
DEFAULT_BOUNDS = (-1.0, 1.0)
SURFACE_CHANNEL_COUNT = 1
UNLABELED_CHANNEL_COUNT = 2
FREIHAND_KEYPOINT_NAMES = (
    "wrist",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
    "thumb_tip",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "index_tip",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "middle_tip",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "ring_tip",
    "pinky_joint_1",
    "pinky_joint_2",
    "pinky_joint_3",
    "pinky_tip",
)


def _validated_points(
    points: NDArray, name: str, bounds: Tuple[float, float]
) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("{} must have shape (N, 3) with N > 0".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    if np.any(array < bounds[0]) or np.any(array > bounds[1]):
        raise ValueError(
            "{} lie outside the configured bounds [{}, {}]; "
            "normalize or reconfigure bounds rather than clipping".format(
                name, bounds[0], bounds[1]
            )
        )
    return array


def _validate_bounds(bounds: Tuple[float, float]) -> Tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain (minimum, maximum)")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("bounds must be finite and minimum must be less than maximum")
    return lower, upper


def _voxel_indices(
    points: NDArray[np.float64],
    resolution: int,
    bounds: Tuple[float, float],
) -> NDArray[np.intp]:
    scaled = (points - bounds[0]) * (resolution / (bounds[1] - bounds[0]))
    indices = np.floor(scaled).astype(np.intp)
    # The upper bound belongs to the final cell, not a cell beyond the grid.
    return np.clip(indices, 0, resolution - 1)


def _sample_triangles(
    vertices: NDArray[np.float64],
    faces: NDArray[np.intp],
    cell_width: float,
) -> NDArray[np.float64]:
    samples = [vertices]
    for face in faces:
        triangle = vertices[face]
        longest_edge = max(
            np.linalg.norm(triangle[0] - triangle[1]),
            np.linalg.norm(triangle[1] - triangle[2]),
            np.linalg.norm(triangle[2] - triangle[0]),
        )
        steps = max(1, int(ceil(longest_edge / (cell_width * 0.5))))
        barycentric_points = []
        for first in range(steps + 1):
            for second in range(steps + 1 - first):
                third = steps - first - second
                barycentric_points.append(
                    (first * triangle[0] + second * triangle[1] + third * triangle[2])
                    / steps
                )
        samples.append(np.asarray(barycentric_points, dtype=np.float64))
    return np.concatenate(samples, axis=0)


def encode_hand_geometry(
    vertices: NDArray,
    faces: Optional[NDArray] = None,
    keypoints: Optional[NDArray] = None,
    keypoint_names: Optional[Sequence[str]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    bounds: Tuple[float, float] = DEFAULT_BOUNDS,
) -> NDArray[np.uint8]:
    """Encode normalized XYZ geometry as a surface/keypoint occupancy grid.

    Input coordinates must already be in the agreed upstream normalized frame
    and within the inclusive cubic bounds. ``faces`` uses zero-based triangle
    indices into ``vertices``. Mesh triangles are deterministically sampled at
    no more than half-cell spacing along their longest edge. Without faces,
    only supplied vertices can contribute to the surface channel.

    Without ``keypoint_names``, output has two channels: surface occupancy and
    union keypoint occupancy. When names are supplied, output has one surface
    channel followed by one channel per keypoint, in the given order. Each
    named channel retains landmark identity even if two landmarks occupy the
    same voxel. Names should follow ``FREIHAND_KEYPOINT_NAMES`` for FreiHAND
    annotations. The function does not apply normalization, rescale input, or
    silently clip geometry.
    """
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    bounds = _validate_bounds(bounds)
    points = _validated_points(vertices, "vertices", bounds)

    if faces is None:
        face_indices = np.empty((0, 3), dtype=np.intp)
    else:
        raw_faces = np.asarray(faces)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer vertex indices")
        if np.any(raw_faces < 0) or np.any(raw_faces >= points.shape[0]):
            raise ValueError("faces contain an index outside the vertices array")
        face_indices = raw_faces.astype(np.intp, copy=False)

    if keypoints is None:
        landmarks = None
    else:
        landmarks = _validated_points(keypoints, "keypoints", bounds)

    cell_width = (bounds[1] - bounds[0]) / resolution
    surface_points = (
        _sample_triangles(points, face_indices, cell_width)
        if face_indices.shape[0]
        else points
    )
    if keypoint_names is not None:
        if landmarks is None:
            raise ValueError("keypoint_names require keypoints")
        names = tuple(keypoint_names)
        if len(names) != landmarks.shape[0]:
            raise ValueError(
                "keypoint_names must contain one name per keypoint; got {} names "
                "for {} keypoints".format(len(names), landmarks.shape[0])
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("keypoint_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("keypoint_names must be unique")
        channel_count = SURFACE_CHANNEL_COUNT + len(names)
    else:
        names = None
        channel_count = UNLABELED_CHANNEL_COUNT

    representation = np.zeros(
        (resolution, resolution, resolution, channel_count), dtype=np.uint8
    )

    surface_indices = _voxel_indices(surface_points, resolution, bounds)
    representation[
        surface_indices[:, 0],
        surface_indices[:, 1],
        surface_indices[:, 2],
        0,
    ] = 1

    if landmarks is not None:
        landmark_indices = _voxel_indices(landmarks, resolution, bounds)
        if names is None:
            representation[
                landmark_indices[:, 0],
                landmark_indices[:, 1],
                landmark_indices[:, 2],
                1,
            ] = 1
        else:
            channels = np.arange(len(names), dtype=np.intp) + SURFACE_CHANNEL_COUNT
            representation[
                landmark_indices[:, 0],
                landmark_indices[:, 1],
                landmark_indices[:, 2],
                channels,
            ] = 1
    return representation


def decode_hand_geometry(
    representation: NDArray,
    bounds: Tuple[float, float] = DEFAULT_BOUNDS,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return approximate surface and keypoint coordinates at occupied cell centers.

    This is a lossy diagnostic decoder, not an inverse mesh reconstruction.
    """
    bounds = _validate_bounds(bounds)
    grid = np.asarray(representation)
    if (
        grid.ndim != 4
        or grid.shape[0] != grid.shape[1]
        or grid.shape[1] != grid.shape[2]
    ):
        raise ValueError("representation must have shape (R, R, R, C)")
    if grid.shape[3] < UNLABELED_CHANNEL_COUNT or grid.shape[0] < 2:
        raise ValueError("representation must have shape (R, R, R, C), C >= 2, R >= 2")
    if grid.dtype != np.uint8 or np.any((grid != 0) & (grid != 1)):
        raise ValueError("representation must contain uint8 values 0 or 1")

    resolution = grid.shape[0]
    cell_width = (bounds[1] - bounds[0]) / resolution
    centers = bounds[0] + (np.argwhere(grid[..., 0] == 1) + 0.5) * cell_width
    keypoint_occupancy = np.any(grid[..., 1:] == 1, axis=-1)
    keypoint_centers = bounds[0] + (
        np.argwhere(keypoint_occupancy) + 0.5
    ) * cell_width
    return centers.astype(np.float64), keypoint_centers.astype(np.float64)


def decode_named_keypoints(
    representation: NDArray,
    keypoint_names: Sequence[str],
    bounds: Tuple[float, float] = DEFAULT_BOUNDS,
) -> dict:
    """Decode each named landmark channel to its occupied cell center(s)."""
    bounds = _validate_bounds(bounds)
    grid = np.asarray(representation)
    names = tuple(keypoint_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("keypoint_names must be non-empty strings")
    if (
        grid.ndim != 4
        or grid.shape[0] != grid.shape[1]
        or grid.shape[1] != grid.shape[2]
        or grid.shape[0] < 2
    ):
        raise ValueError("representation must have shape (R, R, R, C)")
    if grid.shape[3] != SURFACE_CHANNEL_COUNT + len(names):
        raise ValueError(
            "representation channel count must equal one surface channel plus "
            "one channel per name"
        )
    if grid.dtype != np.uint8 or np.any((grid != 0) & (grid != 1)):
        raise ValueError("representation must contain uint8 values 0 or 1")
    if len(set(names)) != len(names):
        raise ValueError("keypoint_names must be unique, non-empty strings")

    cell_width = (bounds[1] - bounds[0]) / grid.shape[0]
    return {
        name: (
            bounds[0]
            + (np.argwhere(grid[..., channel + SURFACE_CHANNEL_COUNT] == 1) + 0.5)
            * cell_width
        ).astype(np.float64)
        for channel, name in enumerate(names)
    }
