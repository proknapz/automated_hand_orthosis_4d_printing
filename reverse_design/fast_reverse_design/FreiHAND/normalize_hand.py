"""Invertible translation and uniform-scale normalization for hand geometry."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from .data_loader import HandGeometry, validate_geometry
except ImportError:
    from data_loader import HandGeometry, validate_geometry


DEFAULT_BOUNDS = (-1.0, 1.0)


@dataclass
class GeometryTransform:
    """Parameters needed to reproduce and invert a normalization operation.

    With the current axis-preserving policy, rotation is the identity matrix:
    ``normalized = (raw - center) / scale + bounds_midpoint``. ``scale`` is in
    source-units per normalized coordinate unit, so absolute units remain
    recoverable when the source unit is known.
    """

    center: NDArray[np.float64]
    scale: float
    rotation: NDArray[np.float64]
    output_bounds: Tuple[float, float]
    source_units: str


def _validated_bounds(bounds: Tuple[float, float]) -> Tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain (minimum, maximum)")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("bounds must be finite and minimum must be less than maximum")
    return lower, upper


def _transform_points(
    points: Optional[NDArray[np.float64]],
    center: NDArray[np.float64],
    scale: float,
    rotation: NDArray[np.float64],
    output_midpoint: float,
) -> Optional[NDArray[np.float64]]:
    if points is None:
        return None
    return ((points - center) @ rotation.T) / scale + output_midpoint


def normalize_hand(
    geometry: HandGeometry,
    output_bounds: Tuple[float, float] = DEFAULT_BOUNDS,
) -> Tuple[HandGeometry, GeometryTransform]:
    """Center the geometry's bounding box and uniformly scale to output bounds.

    Bounds are computed across both vertices and keypoints, so neither can be
    unexpectedly excluded. The transform does not rotate or mirror the input;
    input axes and pose orientation are preserved. Uniform scale preserves
    proportions, while physical size is retained in the returned transform.
    """
    geometry = validate_geometry(geometry)
    lower, upper = _validated_bounds(output_bounds)
    point_arrays = [
        points
        for points in (geometry.vertices, geometry.keypoints)
        if points is not None
    ]
    all_points = np.concatenate(point_arrays, axis=0)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    center = (minimum + maximum) / 2.0
    largest_extent = float(np.max(maximum - minimum))
    if not np.isfinite(largest_extent) or largest_extent <= 0.0:
        raise ValueError("geometry must have a positive spatial extent")

    bounds_midpoint = (lower + upper) / 2.0
    bounds_half_width = (upper - lower) / 2.0
    scale = largest_extent / (2.0 * bounds_half_width)
    rotation = np.eye(3, dtype=np.float64)
    normalized = HandGeometry(
        sample_id=geometry.sample_id,
        vertices=_transform_points(
            geometry.vertices, center, scale, rotation, bounds_midpoint
        ),
        keypoints=_transform_points(
            geometry.keypoints, center, scale, rotation, bounds_midpoint
        ),
        faces=None if geometry.faces is None else geometry.faces.copy(),
        units="normalized",
        metadata=dict(geometry.metadata),
    )
    transform = GeometryTransform(
        center=center.copy(),
        scale=scale,
        rotation=rotation,
        output_bounds=(lower, upper),
        source_units=geometry.units,
    )
    return normalized, transform


def denormalize_hand(
    normalized: HandGeometry,
    transform: GeometryTransform,
) -> HandGeometry:
    """Invert ``normalize_hand`` using its recorded transform."""
    normalized = validate_geometry(normalized)
    if not np.isfinite(transform.center).all() or transform.center.shape != (3,):
        raise ValueError("transform.center must be a finite array of shape (3,)")
    if not np.isfinite(transform.scale) or transform.scale <= 0.0:
        raise ValueError("transform.scale must be finite and positive")
    if transform.rotation.shape != (3, 3) or not np.isfinite(transform.rotation).all():
        raise ValueError("transform.rotation must be a finite array of shape (3, 3)")
    if not np.allclose(transform.rotation.T @ transform.rotation, np.eye(3)):
        raise ValueError("transform.rotation must be an orthonormal matrix")

    lower, upper = _validated_bounds(transform.output_bounds)
    midpoint = (lower + upper) / 2.0

    def inverse(points: Optional[NDArray[np.float64]]) -> Optional[NDArray[np.float64]]:
        if points is None:
            return None
        return ((points - midpoint) * transform.scale) @ transform.rotation + transform.center

    return HandGeometry(
        sample_id=normalized.sample_id,
        vertices=inverse(normalized.vertices),
        keypoints=inverse(normalized.keypoints),
        faces=None if normalized.faces is None else normalized.faces.copy(),
        units=transform.source_units,
        metadata=dict(normalized.metadata),
    )
