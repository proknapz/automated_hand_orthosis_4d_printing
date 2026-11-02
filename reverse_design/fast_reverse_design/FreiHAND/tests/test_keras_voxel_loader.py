import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import tensorflow as tf

    from FreiHAND.encode_hand import FREIHAND_KEYPOINT_NAMES
    from FreiHAND.keras_voxel_loader import (
        FreiHANDReconstructionSequence,
        FreiHANDVoxelSequence,
    )
except ModuleNotFoundError as error:
    if error.name != "tensorflow":
        raise
    tf = None
    FreiHANDReconstructionSequence = None
    FreiHANDVoxelSequence = None
    FREIHAND_KEYPOINT_NAMES = ()


def write_archive(path: Path, sample_number: int = 0) -> np.ndarray:
    shape = (4, 4, 4, 22)
    representation = np.zeros(shape, dtype=np.uint8)
    representation[sample_number, 1, 2, sample_number] = 1
    channel_order = ["vertex_occupancy", *FREIHAND_KEYPOINT_NAMES]
    metadata: Dict[str, Any] = {
        "sample_id": "training:{:08d}".format(sample_number),
        "source_units": "m",
        "representation_shape": list(shape),
        "representation_dtype": "uint8",
        "channel_order": channel_order,
    }
    np.savez_compressed(
        str(path),
        representation=representation,
        metadata_json=np.asarray(json.dumps(metadata)),
        keypoint_names=np.asarray(FREIHAND_KEYPOINT_NAMES, dtype=np.str_),
    )
    return representation


def write_reconstruction_archive(path: Path) -> None:
    occupancy = np.zeros((4, 4, 4, 1), dtype=np.uint8)
    occupancy[1, 2, 3, 0] = 1
    target = np.zeros((778, 3), dtype=np.float32)
    metadata = {
        "sample_id": "training:00000000",
        "source_units": "m",
        "input_channels": ["vertex_occupancy"],
        "input_shape": [4, 4, 4, 1],
        "input_dtype": "uint8",
        "keypoint_channels_excluded": True,
        "normalization_source": "vertices only",
        "target_shape": [778, 3],
        "target_dtype": "float32",
        "target_vertex_order": "source FreiHAND vertex array order",
    }
    np.savez_compressed(
        str(path),
        occupancy=occupancy,
        target_vertices=target,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


@unittest.skipIf(tf is None, "TensorFlow is optional and is not installed")
class KerasVoxelSequenceTests(unittest.TestCase):
    def test_returns_channels_last_float32_features_and_separate_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = [
                write_archive(root / "sample_0.npz", sample_number=0),
                write_archive(root / "sample_1.npz", sample_number=1),
            ]
            sequence = FreiHANDVoxelSequence(
                [root / "sample_0.npz", root / "sample_1.npz"],
                batch_size=2,
                expected_shape=(4, 4, 4, 22),
            )

            batch = sequence[0]

            self.assertIsInstance(batch, tf.Tensor)
            self.assertEqual(batch.shape, (2, 4, 4, 4, 22))
            self.assertEqual(batch.dtype, tf.float32)
            np.testing.assert_array_equal(batch.numpy(), np.stack(expected).astype(np.float32))
            self.assertEqual(
                [item["sample_id"] for item in sequence.metadata_for_batch(0)],
                ["training:00000000", "training:00000001"],
            )
            self.assertEqual(len(sequence), 1)

    def test_includes_partial_final_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_archive(root / "sample_0.npz")
            write_archive(root / "sample_1.npz", sample_number=1)
            write_archive(root / "sample_2.npz", sample_number=2)
            sequence = FreiHANDVoxelSequence(
                [root / "sample_0.npz", root / "sample_1.npz", root / "sample_2.npz"],
                batch_size=2,
                expected_shape=(4, 4, 4, 22),
            )

            self.assertEqual(len(sequence), 2)
            self.assertEqual(sequence[1].shape, (1, 4, 4, 4, 22))
            self.assertEqual(
                sequence.metadata_for_batch(1)[0]["sample_id"], "training:00000002"
            )

    def test_rejects_wrong_shape_and_invalid_batch_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "sample.npz"
            write_archive(archive_path)
            sequence = FreiHANDVoxelSequence([archive_path])

            with self.assertRaisesRegex(ValueError, "representation shape"):
                sequence[0]
            with self.assertRaises(IndexError):
                sequence[1]

    def test_rejects_non_binary_tensor_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "sample.npz"
            representation = write_archive(archive_path)
            representation[0, 0, 0, 0] = 2
            with np.load(str(archive_path), allow_pickle=False) as archive:
                metadata_text = archive["metadata_json"]
                keypoint_names = archive["keypoint_names"]
            np.savez_compressed(
                str(archive_path),
                representation=representation,
                metadata_json=metadata_text,
                keypoint_names=keypoint_names,
            )
            sequence = FreiHANDVoxelSequence(
                [archive_path], expected_shape=(4, 4, 4, 22)
            )

            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                sequence[0]

    def test_requires_non_empty_archive_list(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            FreiHANDVoxelSequence([])

    def test_keras_predict_accepts_feature_only_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_archive(root / "sample.npz")
            sequence = FreiHANDVoxelSequence(
                [root / "sample.npz"], expected_shape=(4, 4, 4, 22)
            )
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(4, 4, 4, 22)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1),
                ]
            )

            predictions = model.predict(sequence, verbose=0)

            self.assertEqual(predictions.shape, (1, 1))

    def test_requires_the_contract_channel_count(self):
        with self.assertRaisesRegex(ValueError, "include all 22 channels"):
            FreiHANDVoxelSequence([], expected_shape=(4, 4, 4, 21))


@unittest.skipIf(tf is None, "TensorFlow is optional and is not installed")
class KerasReconstructionSequenceTests(unittest.TestCase):
    def test_returns_occupancy_inputs_and_vertex_targets_without_landmark_channels(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "reconstruction.npz"
            write_reconstruction_archive(archive_path)
            sequence = FreiHANDReconstructionSequence(
                [archive_path],
                input_shape=(4, 4, 4, 1),
            )

            inputs, targets = sequence[0]

            self.assertEqual(inputs.shape, (1, 4, 4, 4, 1))
            self.assertEqual(inputs.dtype, tf.float32)
            self.assertEqual(targets.shape, (1, 778, 3))
            self.assertEqual(targets.dtype, tf.float32)
            self.assertEqual(float(tf.reduce_max(inputs)), 1.0)
            self.assertEqual(
                sequence.metadata_for_batch(0)[0]["keypoint_channels_excluded"], True
            )


if __name__ == "__main__":
    unittest.main()
