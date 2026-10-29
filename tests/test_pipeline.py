"""Minimal pipeline tests: geometry plus a sample image with no hand."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from pipeline.exceptions import HandNotDetectedError
from pipeline.landmarks import extract_landmarks
from pipeline.pointcloud import scale_to_hand_length
from pipeline.splint import generate_splint

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_hand.png"


def _synthetic_hand() -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """A curved patch and 21 landmarks spanning 18 cm wrist-to-tip."""
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[0] = (0.0, 0.0, 0.0)
    xs = np.linspace(-0.03, 0.03, 4)
    for i, mcp in enumerate((5, 9, 13, 17)):
        landmarks[mcp] = (xs[i], 0.09, 0.01)
        landmarks[mcp + 1] = (xs[i], 0.12, 0.012)
        landmarks[mcp + 2] = (xs[i], 0.15, 0.011)
        landmarks[mcp + 3] = (xs[i], 0.18, 0.01)
    landmarks[1] = (-0.04, 0.02, 0.008)
    landmarks[2] = (-0.045, 0.05, 0.01)
    landmarks[3] = (-0.05, 0.07, 0.011)
    landmarks[4] = (-0.055, 0.09, 0.01)

    uu = np.linspace(-0.045, 0.045, 40)
    vv = np.linspace(-0.03, 0.16, 60)
    pts = []
    for v in vv:
        for u in uu:
            z = 0.012 + 0.008 * np.exp(-((u / 0.04) ** 2)) - 0.15 * (v**2)
            pts.append((u, v, z))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.tile((0.8, 0.6, 0.5), (len(pts), 1)))
    pcd.estimate_normals()
    return pcd, landmarks


def test_sample_image_no_hand_detected(sample_image: Path) -> None:
    """The bundled sample is a plain PNG, so MediaPipe should report no hand."""
    assert sample_image.is_file()
    with pytest.raises(HandNotDetectedError, match="No hand detected"):
        extract_landmarks(str(sample_image))


def test_scale_to_hand_length() -> None:
    landmarks = np.zeros((21, 3))
    landmarks[0] = (0, 0, 0)
    landmarks[12] = (0, 2.0, 0)
    points = np.array([[0.0, 1.0, 0.0]])
    scaled, lm, factor = scale_to_hand_length(points, landmarks, 0.18)
    assert factor == pytest.approx(0.09)
    assert scaled[0, 1] == pytest.approx(0.09)
    assert np.linalg.norm(lm[12] - lm[0]) == pytest.approx(0.18)


def test_generate_splint_has_faces() -> None:
    pcd, landmarks = _synthetic_hand()
    mesh = generate_splint(pcd, landmarks, side="dorsal", finger_extension=False)
    assert mesh.faces.shape[0] > 50
    assert mesh.vertices.shape[0] > 50
    assert mesh.area > 0


def test_generate_splint_finger_extension_is_longer() -> None:
    pcd, landmarks = _synthetic_hand()
    short = generate_splint(pcd, landmarks, side="dorsal", finger_extension=False)
    long = generate_splint(pcd, landmarks, side="dorsal", finger_extension=True)
    assert long.extents[1] >= short.extents[1] * 0.9


def test_upload_rejects_non_image(client) -> None:
    response = client.post(
        "/upload",
        data={"photo": (io.BytesIO(b"not an image"), "notes.txt"), "hand_length_cm": "18"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body


def test_upload_sample_image_returns_no_hand_message(client, sample_image: Path) -> None:
    data = sample_image.read_bytes()
    response = client.post(
        "/upload",
        data={
            "photo": (io.BytesIO(data), "sample_hand.png"),
            "hand_length_cm": "18",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "No hand detected" in response.get_json()["error"]
