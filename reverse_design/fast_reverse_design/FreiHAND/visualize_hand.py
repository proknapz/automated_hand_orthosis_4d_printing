"""Plot a hand point cloud/keypoints before and after normalization."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from .data_loader import (
        FreiHANDDataset,
        HandGeometry,
        load_geometry_npz,
    )
    from .normalize_hand import normalize_hand
except ImportError:
    from data_loader import FreiHANDDataset, HandGeometry, load_geometry_npz
    from normalize_hand import normalize_hand


def make_synthetic_geometry() -> HandGeometry:
    """Return a visibly synthetic point-cloud hand for preliminary testing."""
    points = []
    for x in np.linspace(-45.0, 45.0, 18):
        for y in np.linspace(-28.0, 25.0, 12):
            points.append((x, y, 0.0))
    for x, length in zip((-36.0, -12.0, 12.0, 36.0), (72.0, 94.0, 88.0, 72.0)):
        for y in np.linspace(25.0, length, 14):
            for z in (-3.0, 0.0, 3.0):
                points.append((x, y, z))
    for fraction in np.linspace(0.0, 1.0, 16):
        points.append((-45.0 - 33.0 * fraction, 18.0 - 50.0 * fraction, 2.0 * fraction))
    return HandGeometry(
        sample_id="synthetic-hand-example",
        vertices=np.asarray(points, dtype=np.float64),
        keypoints=np.asarray(
            [
                [0.0, -28.0, 0.0],
                [-36.0, 25.0, 0.0],
                [-12.0, 25.0, 0.0],
                [12.0, 25.0, 0.0],
                [36.0, 25.0, 0.0],
                [-78.0, -32.0, 0.0],
            ],
            dtype=np.float64,
        ),
        units="synthetic_units",
    )


def _plot_geometry(axis, geometry: HandGeometry, title: str) -> None:
    points = geometry.vertices if geometry.vertices is not None else geometry.keypoints
    label = "vertices" if geometry.vertices is not None else "keypoints (no mesh vertices)"
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, alpha=0.5, label=label)
    if geometry.keypoints is not None:
        axis.scatter(
            geometry.keypoints[:, 0],
            geometry.keypoints[:, 1],
            geometry.keypoints[:, 2],
            s=24,
            marker="x",
            label="3D keypoints",
        )
    axis.set_title(title)
    axis.set_xlabel("X ({})".format(geometry.units))
    axis.set_ylabel("Y ({})".format(geometry.units))
    axis.set_zlabel("Z ({})".format(geometry.units))
    axis.legend(loc="upper left")


def visualize(
    geometry: HandGeometry,
    output_path: Path,
    label: Optional[str] = None,
) -> Path:
    normalized, transform = normalize_hand(geometry)
    figure = plt.figure(figsize=(12, 5))
    raw_axis = figure.add_subplot(1, 2, 1, projection="3d")
    normalized_axis = figure.add_subplot(1, 2, 2, projection="3d")
    sample_label = label or geometry.sample_id
    _plot_geometry(raw_axis, geometry, "Raw input coordinates")
    _plot_geometry(normalized_axis, normalized, "Normalized coordinates (no rotation)")
    figure.suptitle(sample_label)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(
        "Saved {} (center={}, scale={:.8g} source-units/normalized-unit)".format(
            output_path, transform.center.tolist(), transform.scale
        )
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Generic .npz geometry archive")
    source.add_argument("--dataset-root", type=Path, help="Unpacked official FreiHAND root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", choices=("training", "evaluation"), default="training")
    parser.add_argument(
        "--image-version", choices=("gs", "hom", "sample", "auto"), default="gs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "visualizations"
        / "01_raw_and_normalized_synthetic.png",
    )
    args = parser.parse_args()

    if args.input is not None:
        geometry = load_geometry_npz(args.input)
        label = "Geometry archive: {}".format(geometry.sample_id)
    elif args.dataset_root is not None:
        geometry = FreiHANDDataset(
            args.dataset_root, split=args.split, image_version=args.image_version
        ).load_sample(args.index)
        label = "FreiHAND annotation {} (keypoints; mesh unavailable)".format(
            geometry.sample_id
        )
    else:
        geometry = make_synthetic_geometry()
        label = "Synthetic example (not FreiHAND data)"

    visualize(geometry, args.output, label=label)
    if args.dataset_root is None and args.input is None:
        print("Synthetic geometry only; no FreiHAND sample was processed.")


if __name__ == "__main__":
    main()
