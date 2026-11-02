"""Export selected FreiHAND samples as framework-neutral voxel tensors."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    from .data_loader import FreiHANDDataset
    from .encode_hand import FREIHAND_KEYPOINT_NAMES, encode_hand_geometry
    from .normalize_hand import normalize_hand
except ImportError:
    from data_loader import FreiHANDDataset
    from encode_hand import FREIHAND_KEYPOINT_NAMES, encode_hand_geometry
    from normalize_hand import normalize_hand


def export_sample(
    dataset: FreiHANDDataset,
    index: int,
    output_dir: Path,
    resolution: int = 96,
    overwrite: bool = False,
) -> Path:
    """Encode and save one sample with its identity and inverse transform."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("index must be an integer")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2")

    source = dataset.load_sample(index)
    if source.vertices is None:
        raise ValueError(
            "training vertices are required to export {}".format(source.sample_id)
        )
    if source.keypoints is None or source.keypoints.shape != (
        len(FREIHAND_KEYPOINT_NAMES),
        3,
    ):
        raise ValueError(
            "expected {} FreiHAND keypoints for {}".format(
                len(FREIHAND_KEYPOINT_NAMES), source.sample_id
            )
        )

    normalized, transform = normalize_hand(source)
    representation = encode_hand_geometry(
        normalized.vertices,
        faces=normalized.faces,
        keypoints=normalized.keypoints,
        keypoint_names=FREIHAND_KEYPOINT_NAMES,
        resolution=resolution,
        bounds=transform.output_bounds,
    )
    expected_shape = (resolution, resolution, resolution, 22)
    if representation.shape != expected_shape or representation.dtype != np.uint8:
        raise RuntimeError(
            "encoder returned {}, {}; expected {}, uint8".format(
                representation.shape, representation.dtype, expected_shape
            )
        )
    if not np.isin(representation, (0, 1)).all():
        raise RuntimeError("encoder output must contain only binary values")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_{:08d}.npz".format(index)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it".format(output_path)
        )

    metadata = {
        "sample_id": source.sample_id,
        "sample_index": index,
        "split": dataset.split,
        "image_version": dataset.image_version,
        "source_units": source.units,
        "reference_bone_scale_m": source.metadata.get("hand_scale"),
        "normalization_center_source_units": transform.center.tolist(),
        "normalization_scale_source_units_per_normalized_unit": transform.scale,
        "normalization_rotation": transform.rotation.tolist(),
        "normalized_bounds": list(transform.output_bounds),
        "coordinate_frame_note": source.metadata.get(
            "source_coordinate_frame_note", ""
        ),
        "geometry_note": (
            "FreiHAND annotation vertices; no faces supplied, so channel 0 "
            "marks vertices only."
        ),
        "representation_shape": list(representation.shape),
        "representation_dtype": str(representation.dtype),
        "channel_order": [
            "vertex_occupancy",
            *FREIHAND_KEYPOINT_NAMES,
        ],
    }

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=".training_{:08d}_".format(index),
            dir=str(output_dir),
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                representation=representation,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                keypoint_names=np.asarray(FREIHAND_KEYPOINT_NAMES, dtype=np.str_),
            )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                "{} already exists; pass --overwrite to replace it".format(output_path)
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def export_samples(
    dataset_root: Path,
    indices: Iterable[int],
    output_dir: Path,
    resolution: int = 96,
    overwrite: bool = False,
) -> List[Path]:
    """Export selected training samples, failing explicitly on any bad sample."""
    dataset = FreiHANDDataset(
        Path(dataset_root), split="training", image_version="gs", units="m"
    )
    return [
        export_sample(dataset, index, output_dir, resolution, overwrite)
        for index in indices
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "Datasets" / "FreiHAND",
        help="Directory containing training_K.json and aligned annotations",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 100, 1000, 32000],
        help="Unique training annotation indices to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination (default: <dataset-root>/encoded_prototype)",
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-sample exports",
    )
    args = parser.parse_args(argv)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.dataset_root / "encoded_prototype"
    )
    try:
        output_paths = export_samples(
            args.dataset_root,
            args.indices,
            output_dir,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, IndexError) as error:
        parser.error(str(error))
    for output_path in output_paths:
        print("Exported {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
