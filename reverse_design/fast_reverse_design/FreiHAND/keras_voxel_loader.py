"""Keras input-only loader for exported FreiHAND voxel archives."""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

try:
    from .encode_hand import FREIHAND_KEYPOINT_NAMES
except ImportError:
    from encode_hand import FREIHAND_KEYPOINT_NAMES


DEFAULT_SHAPE = (96, 96, 96, 22)
DEFAULT_CHANNEL_NAMES = ("vertex_occupancy",) + FREIHAND_KEYPOINT_NAMES
RECONSTRUCTION_INPUT_SHAPE = (96, 96, 96, 1)
RECONSTRUCTION_TARGET_SHAPE = (778, 3)


class FreiHANDVoxelSequence(tf.keras.utils.Sequence):
    """Yield channels-last float32 feature batches from exported NPZ samples.

    The archives contain no supervised target labels. Each batch therefore
    contains features only and is suitable for model prediction or as the input
    side of a custom training loop. Metadata is available separately through
    :meth:`metadata_for_batch`.
    """

    def __init__(
        self,
        archive_paths: Sequence[Path],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        expected_shape: Tuple[int, int, int, int] = DEFAULT_SHAPE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            len(expected_shape) != 4
            or any(
                isinstance(size, bool) or not isinstance(size, int)
                for size in expected_shape
            )
            or any(size < 1 for size in expected_shape)
        ):
            raise ValueError("expected_shape must contain four positive integers")
        if expected_shape[-1] != len(DEFAULT_CHANNEL_NAMES):
            raise ValueError(
                "expected_shape must include all {} channels".format(
                    len(DEFAULT_CHANNEL_NAMES)
                )
            )

        paths = [Path(path) for path in archive_paths]
        if not paths:
            raise ValueError("at least one NPZ archive path is required")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "voxel archive(s) do not exist: {}".format(", ".join(missing))
            )
        if any(path.suffix.lower() != ".npz" for path in paths):
            raise ValueError("all voxel archives must use the .npz format")

        self.archive_paths = paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.expected_shape = tuple(expected_shape)
        self._rng = np.random.RandomState(seed)
        self._order = np.arange(len(paths), dtype=np.int64)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        if self.shuffle:
            self._rng.shuffle(self._order)

    def __len__(self) -> int:
        return (len(self.archive_paths) + self.batch_size - 1) // self.batch_size

    def _batch_paths(self, batch_index: int) -> List[Path]:
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise TypeError("batch index must be an integer")
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("batch index {} is out of range".format(batch_index))
        start = batch_index * self.batch_size
        sample_indices = self._order[start : start + self.batch_size]
        return [self.archive_paths[int(sample_index)] for sample_index in sample_indices]

    def _read_archive(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        required_fields = {"representation", "metadata_json", "keypoint_names"}
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                missing = required_fields.difference(archive.files)
                if missing:
                    raise ValueError(
                        "{} is missing required fields: {}".format(
                            path, ", ".join(sorted(missing))
                        )
                    )
                representation = archive["representation"]
                metadata_text = archive["metadata_json"]
                keypoint_names = archive["keypoint_names"]
        except (OSError, ValueError) as error:
            raise ValueError(
                "could not read voxel archive {}: {}".format(path, error)
            ) from error

        if representation.shape != self.expected_shape:
            raise ValueError(
                "{} has representation shape {}; expected {}".format(
                    path, representation.shape, self.expected_shape
                )
            )
        if representation.dtype != np.uint8:
            raise ValueError(
                "{} representation dtype is {}; expected uint8".format(
                    path, representation.dtype
                )
            )
        if not np.isin(representation, (0, 1)).all():
            raise ValueError("{} representation must contain only 0 and 1".format(path))
        metadata = self._parse_metadata(path, metadata_text, keypoint_names)
        return representation, metadata

    def _read_metadata(self, path: Path) -> Dict[str, Any]:
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                required_fields = {"metadata_json", "keypoint_names"}
                missing = required_fields.difference(archive.files)
                if missing:
                    raise ValueError(
                        "{} is missing required fields: {}".format(
                            path, ", ".join(sorted(missing))
                        )
                    )
                metadata_text = archive["metadata_json"]
                keypoint_names = archive["keypoint_names"]
        except (OSError, ValueError) as error:
            raise ValueError(
                "could not read voxel archive metadata {}: {}".format(path, error)
            ) from error
        return self._parse_metadata(path, metadata_text, keypoint_names)

    def _parse_metadata(
        self, path: Path, metadata_text: np.ndarray, keypoint_names: np.ndarray
    ) -> Dict[str, Any]:
        if metadata_text.shape != ():
            raise ValueError("{} metadata_json must be a scalar string".format(path))
        try:
            metadata = json.loads(str(metadata_text.item()))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("{} contains invalid metadata_json".format(path)) from error
        if not isinstance(metadata, dict):
            raise ValueError("{} metadata_json must contain a JSON object".format(path))
        if metadata.get("representation_shape") != list(self.expected_shape):
            raise ValueError(
                "{} metadata shape does not match the configured tensor shape".format(
                    path
                )
            )
        if metadata.get("representation_dtype") != "uint8":
            raise ValueError("{} metadata dtype must be uint8".format(path))
        if metadata.get("source_units") != "m":
            raise ValueError(
                "{} source_units must be 'm' for exported FreiHAND data".format(path)
            )
        if not isinstance(metadata.get("sample_id"), str) or not metadata["sample_id"]:
            raise ValueError("{} metadata must contain a non-empty sample_id".format(path))
        if keypoint_names.tolist() != list(FREIHAND_KEYPOINT_NAMES):
            raise ValueError(
                "{} keypoint_names do not match the official FreiHAND channel order".format(
                    path
                )
            )
        if metadata.get("channel_order") != list(DEFAULT_CHANNEL_NAMES):
            raise ValueError(
                "{} metadata channel_order does not match the tensor channels".format(
                    path
                )
            )
        return metadata

    def __getitem__(self, batch_index: int) -> tf.Tensor:
        features = []
        for path in self._batch_paths(batch_index):
            representation, metadata = self._read_archive(path)
            self._metadata[str(path)] = metadata
            features.append(representation)
        batch = np.stack(features, axis=0)
        return tf.convert_to_tensor(batch, dtype=tf.float32)

    def metadata_for_batch(self, batch_index: int) -> List[Dict[str, Any]]:
        """Return metadata corresponding to a batch in its current sample order."""
        metadata_items = []
        for path in self._batch_paths(batch_index):
            key = str(path)
            if key not in self._metadata:
                self._metadata[key] = self._read_metadata(path)
            metadata_items.append(dict(self._metadata[key]))
        return metadata_items

    def on_epoch_end(self) -> None:
        """Shuffle sample order between epochs when configured to do so."""
        if self.shuffle:
            self._rng.shuffle(self._order)


class FreiHANDReconstructionSequence(tf.keras.utils.Sequence):
    """Yield occupancy-only inputs and ordered normalized-vertex targets."""

    def __init__(
        self,
        archive_paths: Sequence[Path],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        input_shape: Tuple[int, int, int, int] = RECONSTRUCTION_INPUT_SHAPE,
        target_shape: Tuple[int, int] = RECONSTRUCTION_TARGET_SHAPE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            len(input_shape) != 4
            or any(
                isinstance(size, bool) or not isinstance(size, int)
                for size in input_shape
            )
            or any(size < 1 for size in input_shape)
            or input_shape[-1] != 1
        ):
            raise ValueError("input_shape must contain positive dimensions and 1 channel")
        if (
            tuple(target_shape) != RECONSTRUCTION_TARGET_SHAPE
        ):
            raise ValueError(
                "target_shape must match the FreiHAND ordered vertex contract {}".format(
                    RECONSTRUCTION_TARGET_SHAPE
                )
            )
        paths = [Path(path) for path in archive_paths]
        if not paths:
            raise ValueError("at least one reconstruction archive path is required")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "reconstruction archive(s) do not exist: {}".format(
                    ", ".join(missing)
                )
            )
        if any(path.suffix.lower() != ".npz" for path in paths):
            raise ValueError("all reconstruction archives must use the .npz format")

        self.archive_paths = paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = tuple(input_shape)
        self._rng = np.random.RandomState(seed)
        self._order = np.arange(len(paths), dtype=np.int64)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        if self.shuffle:
            self._rng.shuffle(self._order)

    def __len__(self) -> int:
        return (len(self.archive_paths) + self.batch_size - 1) // self.batch_size

    def _batch_paths(self, batch_index: int) -> List[Path]:
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise TypeError("batch index must be an integer")
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("batch index {} is out of range".format(batch_index))
        start = batch_index * self.batch_size
        sample_indices = self._order[start : start + self.batch_size]
        return [self.archive_paths[int(sample_index)] for sample_index in sample_indices]

    def _read_archive(self, path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                required = {"occupancy", "target_vertices", "metadata_json"}
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(
                        "{} is missing required fields: {}".format(
                            path, ", ".join(sorted(missing))
                        )
                    )
                occupancy = archive["occupancy"]
                target = archive["target_vertices"]
                metadata_text = archive["metadata_json"]
        except (OSError, ValueError) as error:
            raise ValueError(
                "could not read reconstruction archive {}: {}".format(path, error)
            ) from error

        if occupancy.shape != self.input_shape:
            raise ValueError(
                "{} occupancy shape is {}; expected {}".format(
                    path, occupancy.shape, self.input_shape
                )
            )
        if occupancy.dtype != np.uint8 or not np.isin(occupancy, (0, 1)).all():
            raise ValueError("{} occupancy must be binary uint8".format(path))
        if target.shape != RECONSTRUCTION_TARGET_SHAPE:
            raise ValueError(
                "{} target shape is {}; expected {}".format(
                    path, target.shape, RECONSTRUCTION_TARGET_SHAPE
                )
            )
        if target.dtype != np.float32 or not np.isfinite(target).all():
            raise ValueError("{} targets must be finite float32 vertices".format(path))
        if np.any(target < -1.0) or np.any(target > 1.0):
            raise ValueError("{} targets must lie within normalized bounds [-1, 1]".format(path))
        if metadata_text.shape != ():
            raise ValueError("{} metadata_json must be a scalar string".format(path))
        try:
            metadata = json.loads(str(metadata_text.item()))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("{} contains invalid metadata_json".format(path)) from error
        if not isinstance(metadata, dict):
            raise ValueError("{} metadata_json must contain a JSON object".format(path))
        if metadata.get("input_shape") != list(self.input_shape):
            raise ValueError("{} metadata input shape does not match".format(path))
        if metadata.get("input_dtype") != "uint8":
            raise ValueError("{} metadata input dtype must be uint8".format(path))
        if metadata.get("target_shape") != list(RECONSTRUCTION_TARGET_SHAPE):
            raise ValueError("{} metadata target shape does not match".format(path))
        if metadata.get("target_dtype") != "float32":
            raise ValueError("{} metadata target dtype must be float32".format(path))
        if metadata.get("input_channels") != ["vertex_occupancy"]:
            raise ValueError("{} must contain only vertex-occupancy input".format(path))
        if metadata.get("keypoint_channels_excluded") is not True:
            raise ValueError("{} must confirm keypoint-channel exclusion".format(path))
        if metadata.get("normalization_source") != "vertices only":
            raise ValueError("{} normalization must use vertices only".format(path))
        if metadata.get("target_vertex_order") != "source FreiHAND vertex array order":
            raise ValueError("{} target vertex order does not match".format(path))
        if metadata.get("source_units") != "m":
            raise ValueError("{} source_units must be 'm'".format(path))
        if not isinstance(metadata.get("sample_id"), str) or not metadata["sample_id"]:
            raise ValueError("{} metadata must contain a non-empty sample_id".format(path))
        return occupancy, target, metadata

    def __getitem__(self, batch_index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = []
        targets = []
        for path in self._batch_paths(batch_index):
            occupancy, target, metadata = self._read_archive(path)
            inputs.append(occupancy)
            targets.append(target)
            self._metadata[str(path)] = metadata
        return (
            tf.convert_to_tensor(np.stack(inputs, axis=0), dtype=tf.float32),
            tf.convert_to_tensor(np.stack(targets, axis=0), dtype=tf.float32),
        )

    def metadata_for_batch(self, batch_index: int) -> List[Dict[str, Any]]:
        metadata_items = []
        for path in self._batch_paths(batch_index):
            key = str(path)
            if key not in self._metadata:
                _, _, metadata = self._read_archive(path)
                self._metadata[key] = metadata
            metadata_items.append(dict(self._metadata[key]))
        return metadata_items

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._order)
