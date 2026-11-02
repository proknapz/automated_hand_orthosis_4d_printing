import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from FreiHAND.data_loader import FreiHANDDataset
from FreiHAND.encode_hand import FREIHAND_KEYPOINT_NAMES
from FreiHAND.export_encoded_samples import export_sample


def write_export_fixture(root: Path) -> None:
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[:, 0] = np.linspace(-0.03125, 0.03125, 21)
    keypoints[:, 1] = np.linspace(0.03125, -0.03125, 21)
    keypoints[:, 2] = np.linspace(-0.03125, 0.03125, 21)
    vertices = np.column_stack(
        (
            np.linspace(-0.0625, 0.0625, 778),
            np.sin(np.linspace(0.0, np.pi * 2.0, 778)) * 0.03125,
            np.linspace(-0.0625, 0.0625, 778),
        )
    )
    reference_scale = float(np.linalg.norm(keypoints[9] - keypoints[10]))
    payloads = {
        "training_K.json": [np.eye(3).tolist()],
        "training_mano.json": [np.zeros((1, 61)).tolist()],
        "training_xyz.json": [keypoints.tolist()],
        "training_verts.json": [vertices.tolist()],
        "training_scale.json": [reference_scale],
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


class ExportEncodedSamplesTests(unittest.TestCase):
    def test_export_contains_tensor_channel_names_and_inverse_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            output_dir = root / "export"
            dataset_root.mkdir()
            write_export_fixture(dataset_root)

            output_path = export_sample(
                FreiHANDDataset(dataset_root), 0, output_dir, resolution=8
            )

            with np.load(str(output_path), allow_pickle=False) as archive:
                representation = archive["representation"]
                metadata = json.loads(str(archive["metadata_json"]))
                keypoint_names = archive["keypoint_names"].tolist()

            self.assertEqual(output_path.name, "training_00000000.npz")
            self.assertEqual(representation.shape, (8, 8, 8, 22))
            self.assertEqual(representation.dtype, np.uint8)
            self.assertTrue(set(np.unique(representation)).issubset({0, 1}))
            self.assertEqual(keypoint_names, list(FREIHAND_KEYPOINT_NAMES))
            self.assertEqual(metadata["sample_id"], "training:00000000")
            self.assertEqual(metadata["source_units"], "m")
            self.assertEqual(metadata["representation_shape"], [8, 8, 8, 22])
            self.assertGreater(
                metadata["normalization_scale_source_units_per_normalized_unit"], 0
            )
            self.assertEqual(metadata["normalized_bounds"], [-1.0, 1.0])

    def test_export_refuses_overwrite_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            output_dir = root / "export"
            dataset_root.mkdir()
            write_export_fixture(dataset_root)
            dataset = FreiHANDDataset(dataset_root)

            output_path = export_sample(dataset, 0, output_dir, resolution=8)
            with self.assertRaises(FileExistsError):
                export_sample(dataset, 0, output_dir, resolution=8)

            overwritten = export_sample(
                dataset, 0, output_dir, resolution=8, overwrite=True
            )
            self.assertEqual(overwritten, output_path)
            with np.load(str(overwritten), allow_pickle=False) as archive:
                self.assertEqual(archive["representation"].shape, (8, 8, 8, 22))


if __name__ == "__main__":
    unittest.main()
