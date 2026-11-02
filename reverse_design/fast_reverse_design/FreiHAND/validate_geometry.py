"""Validate normalization on selected FreiHAND annotation indices."""

import argparse
from pathlib import Path
from typing import List

import numpy as np

try:
    from .data_loader import FreiHANDDataset
    from .normalize_hand import denormalize_hand, normalize_hand
except ImportError:
    from data_loader import FreiHANDDataset
    from normalize_hand import denormalize_hand, normalize_hand


def _arrays_equal(left, right, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.allclose(left, right, rtol=0.0, atol=tolerance))


def validate_indices(
    dataset_root: Path,
    indices: List[int],
    split: str = "training",
    image_version: str = "gs",
) -> List[str]:
    """Run load, deterministic-normalization, and inverse-round-trip checks."""
    dataset = FreiHANDDataset(
        root=dataset_root, split=split, image_version=image_version
    )
    failures = []
    for index in indices:
        try:
            source = dataset.load_sample(index)
            normalized, transform = normalize_hand(source)
            repeated, repeated_transform = normalize_hand(source)
            restored = denormalize_hand(normalized, transform)
            for name in ("vertices", "keypoints", "faces"):
                if not _arrays_equal(
                    getattr(normalized, name), getattr(repeated, name), 1e-12
                ):
                    raise AssertionError("normalization is not deterministic ({})".format(name))
                if not _arrays_equal(
                    getattr(source, name), getattr(restored, name), 1e-10
                ):
                    raise AssertionError("inverse transform mismatch ({})".format(name))
            if not np.allclose(transform.center, repeated_transform.center, atol=1e-12):
                raise AssertionError("translation is not deterministic")
            if not np.isclose(transform.scale, repeated_transform.scale, atol=1e-12):
                raise AssertionError("scale is not deterministic")
            for name in ("vertices", "keypoints"):
                points = getattr(normalized, name)
                if points is not None and (
                    np.any(points < -1.0 - 1e-12) or np.any(points > 1.0 + 1e-12)
                ):
                    raise AssertionError(
                        "normalized {} fall outside [-1, 1]".format(name)
                    )
            print(
                "PASS {} vertices={} keypoints={} units={} scale={:.8g}".format(
                    source.sample_id,
                    None if source.vertices is None else source.vertices.shape,
                    None if source.keypoints is None else source.keypoints.shape,
                    source.units,
                    transform.scale,
                )
            )
        except (OSError, ValueError, IndexError, AssertionError) as error:
            message = "FAIL {}: {}".format(index, error)
            failures.append(message)
            print(message)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--split", choices=("training", "evaluation"), default="training")
    parser.add_argument(
        "--image-version",
        choices=("gs", "hom", "sample", "auto"),
        default="gs",
    )
    args = parser.parse_args()
    failures = validate_indices(
        args.dataset_root, args.indices, args.split, args.image_version
    )
    if failures:
        raise SystemExit("{} of {} samples failed".format(len(failures), len(args.indices)))
    print("Validated {} selected sample(s).".format(len(args.indices)))


if __name__ == "__main__":
    main()
