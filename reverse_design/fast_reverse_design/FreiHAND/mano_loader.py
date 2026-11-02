"""MANO hand model loader for face topology integration with FreiHAND.

This module loads face connectivity from MANO hand models and provides
validation to ensure compatibility with FreiHAND's 778-vertex representation.

MANO model access: https://mano.is.tue.mpg.de/
License: See MANO website for terms. This code does not ship MANO models.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


MANO_EXPECTED_VERTEX_COUNT = 778
MANO_EXPECTED_FACE_COUNT = 1538


def load_mano_faces(
    mano_pkl_path: Path,
) -> NDArray[np.int64]:
    """Load face topology from a MANO .pkl model file.

    Args:
        mano_pkl_path: Path to MANO_RIGHT.pkl or MANO_LEFT.pkl

    Returns:
        Face array of shape (F, 3) with zero-based vertex indices.

    Raises:
        FileNotFoundError: If the MANO model file does not exist.
        ValueError: If the file is not a valid MANO model or face count mismatch.
        TypeError: If MANO model structure is unexpected.
    """
    mano_pkl_path = Path(mano_pkl_path)
    if not mano_pkl_path.is_file():
        raise FileNotFoundError(
            f"MANO model not found at {mano_pkl_path}. "
            "Download from https://mano.is.tue.mpg.de/"
        )

    try:
        with open(mano_pkl_path, "rb") as f:
            mano_data = pickle.load(f, encoding="latin1")
    except (pickle.PickleError, EOFError) as e:
        raise ValueError(
            f"Failed to load MANO model from {mano_pkl_path}: {e}"
        ) from e

    if not isinstance(mano_data, dict):
        raise TypeError(
            f"MANO model is not a dictionary; got {type(mano_data).__name__}"
        )

    if "f" not in mano_data:
        raise ValueError(
            "MANO model does not have face topology ('f' key). "
            "Ensure this is a valid MANO .pkl file."
        )

    faces = mano_data["f"]
    if not isinstance(faces, np.ndarray):
        faces = np.asarray(faces, dtype=np.int64)
    else:
        faces = faces.astype(np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Face topology must have shape (F, 3); got {faces.shape}"
        )

    if faces.shape[0] != MANO_EXPECTED_FACE_COUNT:
        raise ValueError(
            f"MANO model has {faces.shape[0]} faces; expected {MANO_EXPECTED_FACE_COUNT}. "
            "Verify MANO model version compatibility."
        )

    if faces.min() < 0:
        raise ValueError(
            f"Face indices contain negative values (min={faces.min()}). "
            "Face indices must be non-negative."
        )

    max_vertex_index = faces.max()
    if max_vertex_index != MANO_EXPECTED_VERTEX_COUNT - 1:
        raise ValueError(
            f"Face topology references vertex indices up to {max_vertex_index}; "
            f"expected maximum {MANO_EXPECTED_VERTEX_COUNT - 1}. "
            "MANO model may not be compatible with FreiHAND's 778-vertex layout."
        )

    return faces


def validate_faces_with_vertices(
    faces: NDArray[np.int64],
    vertices: NDArray[np.float64],
    sample_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validate that face topology is compatible with a vertex array.

    Args:
        faces: Face array of shape (F, 3) with zero-based vertex indices.
        vertices: Vertex array of shape (V, 3).
        sample_id: Optional sample ID for error messages.

    Returns:
        Tuple of (is_valid, message). Message is informative/empty string if valid.
    """
    if faces.ndim != 2 or faces.shape[1] != 3:
        return False, f"Faces must have shape (F, 3); got {faces.shape}"

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return False, f"Vertices must have shape (V, 3); got {vertices.shape}"

    num_vertices = vertices.shape[0]
    max_face_index = faces.max()

    if max_face_index >= num_vertices:
        sample_str = f" (sample {sample_id})" if sample_id else ""
        return (
            False,
            f"Face topology references vertex {max_face_index} "
            f"but only {num_vertices} vertices provided{sample_str}",
        )

    if faces.min() < 0:
        sample_str = f" (sample {sample_id})" if sample_id else ""
        return False, f"Face indices contain negative values{sample_str}"

    return True, ""


def get_mano_status() -> str:
    """Return a summary of MANO integration readiness."""
    return (
        "MANO face topology loader ready. "
        "To enable:\n"
        "  1. Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/\n"
        "  2. Pass mano_pkl_path to FreiHANDDataset or load_geometry_npz\n"
        "  3. Faces will be loaded automatically and attached to HandGeometry"
    )
