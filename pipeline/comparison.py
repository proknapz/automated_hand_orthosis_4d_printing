"""Measurements extracted from exported orthosis STL components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from pipeline.exceptions import PipelineError


def inner_ring_circumference_mm(path: str | Path) -> float:
    """Measure the inner loop of a ring STL at its mid-band cross-section."""
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except (OSError, ValueError, TypeError) as exc:
        raise PipelineError("Could not read the ring STL.") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 12 or len(mesh.faces) == 0:
        raise PipelineError("The ring STL does not contain a valid mesh.")
    if not np.isfinite(mesh.vertices).all():
        raise PipelineError("The ring STL contains invalid coordinates.")
    if not mesh.is_watertight:
        raise PipelineError("The ring STL is not watertight.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) != 3 or singular_values[-1] <= 1e-8:
        raise PipelineError("Could not determine the ring band direction.")
    cross_sections = []
    for axis in axes:
        section = mesh.section(plane_origin=vertices.mean(axis=0), plane_normal=axis)
        if section is None:
            continue
        loop_lengths = []
        for loop in section.discrete:
            if len(loop) < 4 or not np.allclose(loop[0], loop[-1], atol=1e-5):
                continue
            length = float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())
            if np.isfinite(length) and length > 0:
                loop_lengths.append(length)
        if len(loop_lengths) == 2:
            cross_sections.append(loop_lengths)
    if not cross_sections:
        raise PipelineError("Could not identify the inner ring opening.")
    loop_lengths = max(cross_sections, key=sum)
    # STL coordinates in this project are exported in millimetres.
    return min(loop_lengths)
