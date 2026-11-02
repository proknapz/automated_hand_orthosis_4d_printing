import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from FreiHAND.data_loader import FreiHANDDataset
from FreiHAND.export_reconstruction_samples import (
    export_reconstruction_sample,
    split_sample_indices,
)


def write_export_fixture(root: Path, keypoint_shift: float = 0.0) -> None:
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[:, 0] = np.linspace(-0.03125, 0.03125, 21)
    keypoints[:, 1] = np.linspace(0.03125, -0.03125, 21)
    keypoints[:, 2] = np.linspace(-0.03125, 0.03125, 21)
    keypoints += keypoint_shift
    vertices = np.column_stack(
        (
            np.linspace(-0.0625, 0.0625, 778),
            np.sin(np.linspace(0.0, np.pi * 2.0, 778)) * 0.03125,
            np.linspace(-0.0625, 0.0625, 778),
        )
    )
    payloads = {
        "training_K.json": [np.eye(3).tolist()],
        "training_mano.json": [np.zeros((1, 61)).tolist()],
        "training_xyz.json": [keypoints.tolist()],
        "training_verts.json": [vertices.tolist()],
        "training_scale.json": [float(np.linalg.norm(keypoints[9] - keypoints[10]))],
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


class ReconstructionBenchmarkTests(unittest.TestCase):
    def test_split_is_deterministic_and_partitions_samples(self):
        indices = list(range(20))

        first = split_sample_indices(indices, seed=7)
        second = split_sample_indices(indices, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})
        flattened = [index for values in first.values() for index in values]
        self.assertEqual(len(flattened), len(indices))
        self.assertEqual(set(flattened), set(indices))
        self.assertEqual(len(first["train"]), 14)
        self.assertEqual(len(first["validation"]), 3)
        self.assertEqual(len(first["test"]), 3)

    def test_export_has_occupancy_only_inputs_and_ordered_vertex_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            write_export_fixture(dataset_root)

            path = export_reconstruction_sample(
                FreiHANDDataset(dataset_root),
                0,
                root / "benchmark",
                resolution=8,
            )
            with np.load(str(path), allow_pickle=False) as archive:
                occupancy = archive["occupancy"]
                targets = archive["target_vertices"]
                metadata = json.loads(str(archive["metadata_json"]))

            self.assertEqual(path.name, "training_00000000.npz")
            self.assertEqual(occupancy.shape, (8, 8, 8, 1))
            self.assertEqual(occupancy.dtype, np.uint8)
            self.assertTrue(set(np.unique(occupancy)).issubset({0, 1}))
            self.assertEqual(targets.shape, (778, 3))
            self.assertEqual(targets.dtype, np.float32)
            self.assertTrue(np.isfinite(targets).all())
            self.assertGreaterEqual(float(targets.min()), -1.0)
            self.assertLessEqual(float(targets.max()), 1.0)
            self.assertEqual(metadata["input_channels"], ["vertex_occupancy"])
            self.assertTrue(metadata["keypoint_channels_excluded"])
            self.assertEqual(metadata["normalization_source"], "vertices only")
            self.assertEqual(metadata["target_vertex_order"], "source FreiHAND vertex array order")
            self.assertEqual(metadata["target_shape"], [778, 3])
            self.assertEqual(metadata["source_units"], "m")

    def test_keypoint_values_do_not_change_reconstruction_input_or_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            write_export_fixture(first_root)
            write_export_fixture(second_root, keypoint_shift=0.001)

            first = export_reconstruction_sample(
                FreiHANDDataset(first_root), 0, root / "first_output", resolution=8
            )
            second = export_reconstruction_sample(
                FreiHANDDataset(second_root), 0, root / "second_output", resolution=8
            )
            with np.load(str(first), allow_pickle=False) as first_archive:
                first_occupancy = first_archive["occupancy"]
                first_targets = first_archive["target_vertices"]
            with np.load(str(second), allow_pickle=False) as second_archive:
                second_occupancy = second_archive["occupancy"]
                second_targets = second_archive["target_vertices"]

            np.testing.assert_array_equal(first_occupancy, second_occupancy)
            np.testing.assert_array_equal(first_targets, second_targets)

    def test_split_rejects_duplicate_or_too_few_indices(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            split_sample_indices([1, 1, 2])
        with self.assertRaisesRegex(ValueError, "at least three"):
            split_sample_indices([1, 2])

if __name__ == "__main__":
    unittest.main()
