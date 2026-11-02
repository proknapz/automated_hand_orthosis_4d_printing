"""Visualize and validate real FreiHAND annotations already on disk."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from .data_loader import FreiHANDDataset
    from .download_dataset import find_dataset_root
    from .validate_geometry import validate_indices
    from .visualize_hand import visualize
except ImportError:
    from data_loader import FreiHANDDataset
    from download_dataset import find_dataset_root
    from validate_geometry import validate_indices
    from visualize_hand import visualize


DEFAULT_DATA_DIR = Path.home() / "Datasets" / "FreiHAND"
DEFAULT_INDICES = (0, 100, 1000, 32000)


def run_real_validation(
    data_dir: Path,
    indices: Sequence[int] = DEFAULT_INDICES,
    sample_index: int = 0,
    output_path: Optional[Path] = None,
    image_version: str = "gs",
) -> List[str]:
    """Find the dataset, save a real sample plot, and validate selected indices."""
    dataset_root = find_dataset_root(Path(data_dir).expanduser())
    print("Using FreiHAND dataset root: {}".format(dataset_root))

    dataset = FreiHANDDataset(
        dataset_root,
        split="training",
        image_version=image_version,
    )
    sample = dataset.load_sample(sample_index)
    if output_path is None:
        output_path = (
            Path(__file__).parent
            / "visualizations"
            / "01_training_{:08d}.png".format(sample_index)
        )
    visualize(
        sample,
        output_path,
        label="FreiHAND {} (vertices + keypoints; faces unavailable)".format(
            sample.sample_id
        ),
    )

    failures = validate_indices(
        dataset_root,
        list(indices),
        split="training",
        image_version=image_version,
    )
    if failures:
        print("{} of {} validation sample(s) failed.".format(len(failures), len(indices)))
    else:
        print("Validated {} real FreiHAND sample(s).".format(len(indices)))
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset extraction folder (default: ~/Datasets/FreiHAND)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_INDICES),
        help="Training annotation indices to validate",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Training annotation index to visualize",
    )
    parser.add_argument(
        "--image-version",
        choices=("gs", "hom", "sample", "auto"),
        default="gs",
        help="Training image version recorded with the sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Visualization output path (default: FreiHAND/visualizations/01_training_<index>.png)",
    )
    args = parser.parse_args(argv)

    try:
        failures = run_real_validation(
            data_dir=args.data_dir,
            indices=args.indices,
            sample_index=args.sample_index,
            output_path=args.output,
            image_version=args.image_version,
        )
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        parser.error(str(error))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
