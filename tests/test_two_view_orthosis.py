"""Tests for landmark alignment and mesh export from paired hand views."""

from __future__ import annotations

import json
import io

import numpy as np
import pytest
import trimesh

from pipeline.exceptions import PipelineError
from pipeline.orthosis import canonicalize_landmarks, generate_two_view_orthosis


def _hand_points() -> np.ndarray:
    points = np.zeros((21, 2), dtype=float)
    points[0] = (200, 600)
    points[1:5] = [(320, 500), (350, 450), (390, 400), (420, 350)]
    points[5:9] = [(260, 400), (265, 300), (270, 220), (275, 150)]
    points[9:13] = [(220, 390), (220, 270), (220, 180), (220, 100)]
    points[13:17] = [(180, 400), (175, 300), (170, 220), (165, 150)]
    points[17:21] = [(140, 420), (130, 340), (120, 280), (115, 220)]
    return points


def test_canonicalize_two_views_returns_same_landmark_shape() -> None:
    points = _hand_points()
    canonical, summary = canonicalize_landmarks(points, points.copy(), 18.0)
    assert canonical.shape == (21, 2)
    assert np.linalg.norm(canonical[12] - canonical[0]) == pytest.approx(0.18)
    assert summary["mean_view_disagreement_mm"] == pytest.approx(0.0)


def test_two_view_mesh_exports_closed_rings_and_connected_thumb(tmp_path) -> None:
    points = _hand_points()
    result = generate_two_view_orthosis(
        points,
        points.copy(),
        str(tmp_path),
        {"hand_length_cm": 18.0},
    )
    assert (tmp_path / "orthosis.stl").is_file()
    assert (tmp_path / "hand_preview.ply").is_file()
    assert (tmp_path / "index_ring_mcp.stl").is_file()
    assert (tmp_path / "thumb_palm_bridge.stl").is_file()
    assert (tmp_path / "orthosis_parts.zip").is_file()
    mesh = trimesh.load(str(tmp_path / "orthosis.stl"), force="mesh")
    assert len(mesh.faces) > 100
    assert np.max(mesh.extents) > 100  # 18 cm hand exports at hand-sized mm scale
    metadata = json.loads((tmp_path / "landmarks.json").read_text(encoding="utf-8"))
    assert metadata["mesh"]["stl_units"] == "mm"
    assert metadata["mesh"]["component_shells_watertight"]
    assert metadata["mesh"]["closed_part_count"] >= 15
    assert metadata["mesh"]["watertight"] == mesh.is_watertight
    for part_name in metadata["parts"]:
        part = trimesh.load(str(tmp_path / part_name), force="mesh")
        assert part.is_watertight, part_name
    closed_ring = trimesh.load(str(tmp_path / "index_ring_mcp.stl"), force="mesh")
    assert closed_ring.euler_number == 0  # one continuous loop around a finger hole
    palm = trimesh.load(str(tmp_path / "palm_plate.stl"), force="mesh")
    thumb_bridge = trimesh.load(str(tmp_path / "thumb_palm_bridge.stl"), force="mesh")
    assert np.all(thumb_bridge.bounds[0] <= palm.bounds[1]) and np.all(
        palm.bounds[0] <= thumb_bridge.bounds[1]
    )
    assert any("not stereo" in assumption for assumption in metadata["assumptions"])


def test_two_view_generation_rejects_bad_landmark_count(tmp_path) -> None:
    with pytest.raises(PipelineError, match="21 finite landmark"):
        generate_two_view_orthosis([], [], str(tmp_path), {})


def test_analyze_route_requires_both_photos(client, sample_image) -> None:
    response = client.post(
        "/analyze",
        data={
            "dorsal_photo": (io.BytesIO(sample_image.read_bytes()), "dorsal.png"),
            "hand_length_cm": "18",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "choose a JPG or PNG" in response.get_json()["error"]


def test_analyze_route_returns_landmark_review_data(client, sample_image, tmp_path, monkeypatch) -> None:
    import app as flask_app

    class FakeHand:
        handedness = "Right"
        confidence = 0.94

        def as_json(self):
            return {
                "handedness": self.handedness,
                "confidence": self.confidence,
                "image_size": {"width": 500, "height": 700},
                "landmarks": [
                    {
                        "id": index,
                        "name": "landmark_%d" % index,
                        "pixel": {"x": float(index + 10), "y": float(index + 20)},
                        "normalized": {"x": 0.1, "y": 0.2, "z": 0.0},
                    }
                    for index in range(21)
                ],
            }

    uploads = tmp_path / "uploads"
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(flask_app, "extract_landmarks", lambda _path: FakeHand())
    image_bytes = sample_image.read_bytes()
    response = client.post(
        "/analyze",
        data={
            "dorsal_photo": (io.BytesIO(image_bytes), "dorsal.png"),
            "palmar_photo": (io.BytesIO(image_bytes), "palmar.png"),
            "hand_length_cm": "18",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["dorsal"]["landmarks"]) == 21
    assert payload["dorsal"]["confidence"] == pytest.approx(0.94)
    assert client.get(payload["dorsal_image_url"]).data == image_bytes


def test_generate_route_returns_assembled_and_part_urls(client, tmp_path, monkeypatch) -> None:
    import app as flask_app

    job_id = "a" * 32
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    job_uploads = uploads / job_id
    job_uploads.mkdir(parents=True)
    (job_uploads / "dorsal.png").write_bytes(b"test")
    (job_uploads / "palmar.png").write_bytes(b"test")
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(flask_app, "OUTPUT_DIR", outputs)
    points = _hand_points().tolist()
    response = client.post(
        "/generate",
        data={
            "job_id": job_id,
            "hand_length_cm": "18",
            "corrections": json.dumps({"dorsal_points": points, "palmar_points": points}),
            "finger_index_circumference_mm": "52",
            "reference_index_mcp_circumference_mm": "52",
            "reference_source": "test caliper measurement",
        },
    )
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert payload["orthosis_url"].endswith("/orthosis.stl")
    assert payload["parts_zip_url"].endswith("/orthosis_parts.zip")
    assert payload["part_urls"]
    bridge_url = next(url for url in payload["part_urls"] if url.endswith("/thumb_palm_bridge.stl"))
    assert client.get(bridge_url).status_code == 200
    job_record = json.loads((outputs / job_id / "job.json").read_text(encoding="utf-8"))
    assert job_record["reference_measurements_mm"] == {"index": 52.0}
    assert job_record["reference_source"] == "test caliper measurement"

    listed = client.get("/api/comparison/jobs")
    assert listed.status_code == 200
    job = next(item for item in listed.get_json()["jobs"] if item["job_id"] == job_id)
    assert job["eligible"]
    assert len(job["photos"]) == 2

    compared = client.post(
        "/api/comparison/compare",
        json={"job_ids": [job_id]},
    )
    assert compared.status_code == 200, compared.get_json()
    comparison = compared.get_json()
    assert comparison["eligible_job_count"] == 1
    assert comparison["statistics"]["index"]["count"] == 1
    measurement = comparison["results"][0]["measurements"][0]
    assert measurement["reference_circumference_mm"] == 52
    assert measurement["stl_inner_circumference_mm"] > 52
    assert measurement["reference_also_used_for_sizing"]


def test_comparison_rejects_invalid_job_id(client) -> None:
    response = client.post("/api/comparison/compare", json={"job_ids": ["../outside"]})
    assert response.status_code == 400
    assert "Invalid processing job" in response.get_json()["error"]


def test_comparison_reports_legacy_jobs_without_references(client, tmp_path, monkeypatch) -> None:
    import app as flask_app

    job_id = "b" * 32
    outputs = tmp_path / "outputs"
    job_folder = outputs / job_id
    job_folder.mkdir(parents=True)
    (job_folder / "orthosis.stl").write_bytes(b"legacy mesh")
    monkeypatch.setattr(flask_app, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", tmp_path / "uploads")

    listed = client.get("/api/comparison/jobs")
    assert listed.status_code == 200
    job = listed.get_json()["jobs"][0]
    assert not job["eligible"]

    compared = client.post("/api/comparison/compare", json={"job_ids": [job_id]})
    assert compared.status_code == 200
    result = compared.get_json()["results"][0]
    assert not result["eligible"]
    assert "No independent reference" in result["reason"]


def test_generation_requires_a_source_for_reference_measurements(
    client, tmp_path, monkeypatch
) -> None:
    import app as flask_app

    job_id = "c" * 32
    upload_folder = tmp_path / "uploads" / job_id
    upload_folder.mkdir(parents=True)
    (upload_folder / "dorsal.png").write_bytes(b"placeholder")
    (upload_folder / "palmar.png").write_bytes(b"placeholder")
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(flask_app, "OUTPUT_DIR", tmp_path / "outputs")

    response = client.post(
        "/generate",
        data={
            "job_id": job_id,
            "reference_index_mcp_circumference_mm": "52",
        },
    )
    assert response.status_code == 400
    assert "measurement source" in response.get_json()["error"]


def test_delete_saved_job_removes_only_its_outputs_and_photos(
    client, tmp_path, monkeypatch
) -> None:
    import app as flask_app

    job_id = "d" * 32
    other_id = "e" * 32
    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    for current_id in (job_id, other_id):
        (outputs / current_id).mkdir(parents=True)
        (outputs / current_id / "orthosis.stl").write_bytes(b"stl")
        (uploads / current_id).mkdir(parents=True)
        (uploads / current_id / "dorsal.png").write_bytes(b"photo")
    (outputs / "unrelated.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(flask_app, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", uploads)

    response = client.delete("/api/comparison/jobs/%s" % job_id)

    assert response.status_code == 200
    assert response.get_json()["deleted"] == [job_id]
    assert not (outputs / job_id).exists()
    assert not (uploads / job_id).exists()
    assert (outputs / other_id / "orthosis.stl").is_file()
    assert (uploads / other_id / "dorsal.png").is_file()
    assert (outputs / "unrelated.txt").is_file()


def test_delete_all_removes_only_saved_orthosis_jobs(client, tmp_path, monkeypatch) -> None:
    import app as flask_app

    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    job_ids = ("f" * 32, "1" * 32)
    for job_id in job_ids:
        (outputs / job_id).mkdir(parents=True)
        (outputs / job_id / "orthosis.stl").write_bytes(b"stl")
        (uploads / job_id).mkdir(parents=True)
        (uploads / job_id / "palmar.png").write_bytes(b"photo")
    ignored_id = "2" * 32
    (outputs / ignored_id).mkdir(parents=True)
    (outputs / ignored_id / "splint.stl").write_bytes(b"legacy splint")
    monkeypatch.setattr(flask_app, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", uploads)

    response = client.delete("/api/comparison/jobs")

    assert response.status_code == 200
    assert set(response.get_json()["deleted"]) == set(job_ids)
    for job_id in job_ids:
        assert not (outputs / job_id).exists()
        assert not (uploads / job_id).exists()
    assert (outputs / ignored_id / "splint.stl").is_file()


def test_delete_route_rejects_unknown_job(client) -> None:
    response = client.delete("/api/comparison/jobs/not-a-job")
    assert response.status_code == 404
