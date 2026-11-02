import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from FreiHAND.data_loader import (
    FREIHAND_KEYPOINT_COUNT,
    FreiHANDDataset,
    HandGeometry,
    load_geometry_npz,
    validate_geometry,
)
from FreiHAND.normalize_hand import denormalize_hand, normalize_hand
from FreiHAND.validate_geometry import validate_indices


def synthetic_geometry(sample_id="synthetic-0", offset=0.0):
    vertices = np.array(
        [
            [-2.0, -1.0, -0.5],
            [2.0, -1.0, -0.5],
            [0.0, 3.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    keypoints = np.array(
        [[-1.5, -0.5, 0.0], [1.5, 0.5, 0.25]], dtype=np.float64
    )
    vertices[:, 0] += offset
    keypoints[:, 0] += offset
    return HandGeometry(
        sample_id=sample_id,
        vertices=vertices,
        keypoints=keypoints,
        faces=np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64),
        units="synthetic_mm",
    )


def write_synthetic_annotations(root: Path, count=3):
    camera, mano, keypoints = [], [], []
    for sample_index in range(count):
        camera.append(np.eye(3).tolist())
        mano.append(
            (np.arange(61, dtype=float).reshape(1, 61) + sample_index).tolist()
        )
        keypoints.append(
            (np.arange(FREIHAND_KEYPOINT_COUNT * 3, dtype=float).reshape(21, 3)
             + sample_index).tolist()
        )
    for name, payload in (
        ("training_K.json", camera),
        ("training_mano.json", mano),
        ("training_xyz.json", keypoints),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


class GeometryPipelineTests(unittest.TestCase):
    def test_validation_canonicalizes_numeric_arrays_and_accepts_keypoints_only(self):
        geometry = HandGeometry(
            sample_id="keypoints-only",
            keypoints=[[0, 0, 0], [1, 0, 0]],
        )

        validated = validate_geometry(geometry)

        self.assertEqual(validated.keypoints.shape, (2, 3))
        self.assertEqual(validated.keypoints.dtype, np.float64)
        self.assertIsNone(validated.vertices)

    def test_normalization_preserves_axes_proportions_and_records_inverse(self):
        source = synthetic_geometry()
        normalized, transform = normalize_hand(source)
        restored = denormalize_hand(normalized, transform)

        self.assertEqual(normalized.units, "normalized")
        self.assertEqual(normalized.vertices.shape, source.vertices.shape)
        self.assertEqual(normalized.faces.tolist(), source.faces.tolist())
        np.testing.assert_allclose(transform.rotation, np.eye(3))
        self.assertAlmostEqual(float(np.max(normalized.vertices[:, 0])), 1.0)
        self.assertTrue(np.all(normalized.vertices >= -1.0))
        self.assertTrue(np.all(normalized.vertices <= 1.0))
        np.testing.assert_allclose(restored.vertices, source.vertices, atol=1e-12)
        np.testing.assert_allclose(restored.keypoints, source.keypoints, atol=1e-12)
        self.assertEqual(restored.units, source.units)

    def test_normalization_is_deterministic_for_multiple_synthetic_samples(self):
        for sample in (synthetic_geometry("sample-a"), synthetic_geometry("sample-b", 17.0)):
            first, first_transform = normalize_hand(sample)
            second, second_transform = normalize_hand(sample)
            np.testing.assert_array_equal(first.vertices, second.vertices)
            np.testing.assert_array_equal(first.keypoints, second.keypoints)
            np.testing.assert_array_equal(first_transform.center, second_transform.center)
            self.assertEqual(first_transform.scale, second_transform.scale)

    def test_invalid_geometry_is_rejected(self):
        invalid = [
            HandGeometry(sample_id="empty"),
            HandGeometry(sample_id="empty-vertices", vertices=np.empty((0, 3))),
            HandGeometry(sample_id="wrong-shape", vertices=np.zeros((2, 2))),
            HandGeometry(sample_id="nan", vertices=np.array([[np.nan, 0.0, 0.0]])),
            HandGeometry(
                sample_id="bad-faces",
                vertices=np.zeros((3, 3)),
                faces=np.array([[0, 1, 3]]),
            ),
        ]
        for geometry in invalid:
            with self.subTest(sample_id=geometry.sample_id):
                with self.assertRaises(ValueError):
                    validate_geometry(geometry)

    def test_degenerate_geometry_cannot_be_scaled(self):
        geometry = HandGeometry(
            sample_id="point-only",
            keypoints=np.zeros((1, 3)),
        )
        with self.assertRaisesRegex(ValueError, "positive spatial extent"):
            normalize_hand(geometry)

    def test_generic_npz_loader_and_explicit_sample_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "geometry.npz"
            geometry = synthetic_geometry()
            np.savez(
                str(archive),
                vertices=geometry.vertices,
                keypoints=geometry.keypoints,
                faces=geometry.faces,
            )

            loaded = load_geometry_npz(archive, sample_id="patient-sample-17", units="mm")

            self.assertEqual(loaded.sample_id, "patient-sample-17")
            self.assertEqual(loaded.units, "mm")
            np.testing.assert_array_equal(loaded.vertices, geometry.vertices)
            np.testing.assert_array_equal(loaded.faces, geometry.faces)

    def test_synthetic_official_annotation_reader_preserves_alignment_and_image_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root)
            dataset = FreiHANDDataset(root, image_version="hom")

            sample = dataset.load_sample(1)

            self.assertEqual(dataset.sample_count, 3)
            self.assertEqual(sample.sample_id, "training:00000001")
            self.assertEqual(sample.keypoints.shape, (21, 3))
            self.assertEqual(sample.units, "m")
            self.assertIn(
                "camera-relative",
                sample.metadata["source_coordinate_frame_note"].lower(),
            )
            self.assertIn(
                "meters",
                sample.metadata["source_coordinate_units_note"],
            )
            self.assertIsNone(sample.vertices)
            self.assertIsNone(sample.faces)
            self.assertEqual(sample.metadata["annotation_index"], 1)
            self.assertEqual(sample.metadata["image_index"], 32561)
            self.assertEqual(
                sample.metadata["image_relative_path"],
                "training/rgb/00032561.jpg",
            )
            self.assertEqual(sample.metadata["camera_intrinsics"].shape, (3, 3))

    def test_official_reader_loads_aligned_vertex_and_scale_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=2)
            vertices = [
                np.arange(778 * 3, dtype=float).reshape(778, 3).tolist(),
                (np.arange(778 * 3, dtype=float).reshape(778, 3) + 1).tolist(),
            ]
            (root / "training_verts.json").write_text(
                json.dumps(vertices), encoding="utf-8"
            )
            (root / "training_scale.json").write_text(
                json.dumps([0.031, 0.029]), encoding="utf-8"
            )

            dataset = FreiHANDDataset(root)
            first = dataset.load_sample(0)
            second = dataset.load_sample(1)

            self.assertEqual(first.vertices.shape, (778, 3))
            self.assertEqual(first.vertices.dtype, np.float64)
            self.assertEqual(second.vertices.shape, (778, 3))
            self.assertAlmostEqual(first.metadata["hand_scale"], 0.031)
            self.assertAlmostEqual(second.metadata["hand_scale"], 0.029)
            self.assertIsNone(first.faces)

    def test_annotation_mismatch_and_nonfinite_data_raise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=2)
            (root / "training_xyz.json").write_text(
                json.dumps([[[float("nan"), 0, 0]] * 21]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "mismatched lengths"):
                FreiHANDDataset(root).sample_count

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=1)
            (root / "training_xyz.json").write_text(
                json.dumps([[[float("nan"), 0, 0]] * 21]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                FreiHANDDataset(root).load_sample(0)

    def test_mano_annotation_shape_matches_reference_helper_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=1)
            (root / "training_mano.json").write_text(
                json.dumps([[0.0] * 61]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, r"shape \(1, P\), P >= 61"):
                FreiHANDDataset(root).load_sample(0)

    def test_sample_index_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=2)
            dataset = FreiHANDDataset(root)
            with self.assertRaises(IndexError):
                dataset.load_sample(2)
            with self.assertRaises(IndexError):
                dataset.load_sample(-1)

    def test_multi_index_validation_reports_synthetic_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_annotations(root, count=3)

            failures = validate_indices(root, [0, 1, 2])

            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
