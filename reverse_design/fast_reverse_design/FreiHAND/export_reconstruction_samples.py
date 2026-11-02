"""Prepare an input-only-occupancy to normalized-vertices benchmark."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .data_loader import FreiHANDDataset, HandGeometry
    from .encode_hand import DEFAULT_RESOLUTION, encode_hand_geometry
    from .normalize_hand import normalize_hand
except ImportError:
    from data_loader import FreiHANDDataset, HandGeometry
    from encode_hand import DEFAULT_RESOLUTION, encode_hand_geometry
    from normalize_hand import normalize_hand


VERTEX_COUNT = 778
SPLIT_NAMES = ("train", "validation", "test")


def split_sample_indices(
    indices: Sequence[int], seed: int = 42
) -> Dict[str, List[int]]:
    """Create a deterministic 70/15/15 split of unique sample indices."""
    ordered = list(indices)
    if len(ordered) < 3:
        raise ValueError("at least three unique sample indices are required")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in ordered):
        raise ValueError("sample indices must be integers")
    if len(set(ordered)) != len(ordered):
        raise ValueError("sample indices must be unique")
    if any(index < 0 for index in ordered):
        raise ValueError("sample indices cannot be negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    permutation = np.random.RandomState(seed).permutation(ordered).tolist()
    train_count = int(len(ordered) * 0.70)
    validation_count = int(len(ordered) * 0.15)
    test_count = len(ordered) - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("each train/validation/test split must contain a sample")
    return {
        "train": permutation[:train_count],
        "validation": permutation[train_count : train_count + validation_count],
        "test": permutation[train_count + validation_count :],
    }


def export_reconstruction_sample(
    dataset: FreiHANDDataset,
    index: int,
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    overwrite: bool = False,
) -> Path:
    """Save channel-0 occupancy input and ordered normalized-vertex targets."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 2
    ):
        raise ValueError("resolution must be an integer of at least 2")

    source = dataset.load_sample(index)
    if source.vertices is None or source.vertices.shape != (VERTEX_COUNT, 3):
        actual_shape = None if source.vertices is None else source.vertices.shape
        raise ValueError(
            "expected {} ordered vertices for {}; got {}".format(
                VERTEX_COUNT, source.sample_id, actual_shape
            )
        )
    vertex_geometry = HandGeometry(
        sample_id=source.sample_id,
        vertices=source.vertices,
        faces=source.faces,
        units=source.units,
        metadata=dict(source.metadata),
    )
    normalized, transform = normalize_hand(vertex_geometry)
    if normalized.vertices is None:
        raise RuntimeError("normalization unexpectedly removed the vertex array")
    encoded = encode_hand_geometry(
        normalized.vertices,
        resolution=resolution,
        bounds=transform.output_bounds,
    )
    occupancy = encoded[..., :1].copy()
    expected_occupancy_shape = (resolution, resolution, resolution, 1)
    if occupancy.shape != expected_occupancy_shape or occupancy.dtype != np.uint8:
        raise RuntimeError(
            "occupancy encoder returned {}, {}; expected {}, uint8".format(
                occupancy.shape, occupancy.dtype, expected_occupancy_shape
            )
        )
    if not np.isin(occupancy, (0, 1)).all():
        raise RuntimeError("occupancy input must contain only binary values")

    targets = np.asarray(normalized.vertices, dtype=np.float32)
    if targets.shape != (VERTEX_COUNT, 3) or not np.isfinite(targets).all():
        raise RuntimeError("normalized vertex targets must be finite with shape (778, 3)")
    if np.any(targets < -1.0) or np.any(targets > 1.0):
        raise RuntimeError("normalized vertex targets must lie within [-1, 1]")

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
        "source_units": source.units,
        "normalization_center_source_units": transform.center.tolist(),
        "normalization_scale_source_units_per_normalized_unit": transform.scale,
        "normalization_rotation": transform.rotation.tolist(),
        "normalized_bounds": list(transform.output_bounds),
        "normalization_source": "vertices only",
        "input_channels": ["vertex_occupancy"],
        "input_shape": list(occupancy.shape),
        "input_dtype": str(occupancy.dtype),
        "target_name": "ordered_normalized_vertices",
        "target_shape": list(targets.shape),
        "target_dtype": str(targets.dtype),
        "target_vertex_order": "source FreiHAND vertex array order",
        "vertex_correspondence_assumption": (
            "vertex indices have consistent correspondence across samples; "
            "verify before interpreting per-index errors anatomically"
        ),
        "keypoint_channels_excluded": True,
    }

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=".reconstruction_{:08d}_".format(index),
            dir=str(output_dir),
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                occupancy=occupancy,
                target_vertices=targets,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
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


def prepare_benchmark(
    dataset_root: Path,
    sample_count: int = 200,
    output_dir: Optional[Path] = None,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[Dict[str, List[int]], List[Path]]:
    """Export an evenly sampled subset and write its reproducible split manifest."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    dataset = FreiHANDDataset(
        Path(dataset_root), split="training", image_version="gs", units="m"
    )
    available_count = dataset.sample_count
    if sample_count < 3 or sample_count > available_count:
        raise ValueError(
            "sample_count must be between 3 and the {} available samples".format(
                available_count
            )
        )
    indices = np.linspace(
        0, available_count - 1, num=sample_count, dtype=np.int64
    ).tolist()
    if len(set(indices)) != sample_count:
        raise ValueError("sample_count is too high for unique evenly spaced indices")

    splits = split_sample_indices(indices, seed=seed)
    if output_dir is None:
        output_dir = Path(dataset_root) / "reconstruction_prototype"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace it".format(manifest_path)
        )
    output_paths = [
        export_reconstruction_sample(
            dataset, index, output_dir, resolution=resolution, overwrite=overwrite
        )
        for index in indices
    ]

    manifest = {
        "benchmark": "FreiHAND occupancy-to-ordered-vertices reconstruction",
        "status": "prototype representation benchmark; not orthosis or clinical validation",
        "seed": seed,
        "sample_count": sample_count,
        "sample_selection": "evenly spaced across FreiHAND training indices",
        "resolution": resolution,
        "input": {
            "name": "vertex_occupancy",
            "channels": 1,
            "keypoint_channels_excluded": True,
        },
        "target": {
            "name": "ordered_normalized_vertices",
            "shape": [VERTEX_COUNT, 3],
        },
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": splits,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return splits, output_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "Datasets" / "FreiHAND",
        help="Directory containing training_K.json and aligned annotations",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=200,
        help="Number of evenly spaced samples for the prototype benchmark",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sample archives and split manifest",
    )
    args = parser.parse_args(argv)
    try:
        splits, paths = prepare_benchmark(
            args.dataset_root,
            sample_count=args.sample_count,
            output_dir=args.output_dir,
            resolution=args.resolution,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, IndexError) as error:
        parser.error(str(error))
    print("Exported {} reconstruction samples.".format(len(paths)))
    for split_name in SPLIT_NAMES:
        print("{}: {} samples".format(split_name, len(splits[split_name])))
    print(
        "Input: one vertex-occupancy channel. Target: ordered normalized vertices."
    )
    if args.output_dir is None:
        print("Output directory: {}".format(args.dataset_root / "reconstruction_prototype"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
