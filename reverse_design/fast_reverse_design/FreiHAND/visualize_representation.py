"""Visualize normalized geometry alongside its voxel representation."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from .encode_hand import (
        FREIHAND_KEYPOINT_NAMES,
        decode_hand_geometry,
        decode_named_keypoints,
        encode_hand_geometry,
    )
    from .data_loader import FreiHANDDataset
    from .normalize_hand import normalize_hand
except ImportError:
    from encode_hand import (
        FREIHAND_KEYPOINT_NAMES,
        decode_hand_geometry,
        decode_named_keypoints,
        encode_hand_geometry,
    )
    from data_loader import FreiHANDDataset
    from normalize_hand import normalize_hand


def make_synthetic_hand() -> np.ndarray:
    """Create a simple, clearly synthetic hand-shaped point cloud."""
    points = []
    for x in np.linspace(-0.48, 0.48, 17):
        for y in np.linspace(-0.28, 0.28, 13):
            points.append((x, y, 0.0))
    finger_x = [-0.38, -0.13, 0.13, 0.38]
    finger_lengths = [0.67, 0.88, 0.82, 0.67]
    for x, length in zip(finger_x, finger_lengths):
        for y in np.linspace(0.28, length, 18):
            for z in (-0.035, 0.0, 0.035):
                points.append((x, y, z))
    for t in np.linspace(0.0, 1.0, 20):
        points.append((-0.48 - 0.33 * t, 0.20 - 0.48 * t, 0.02 * t))
    return np.asarray(points, dtype=np.float64)


def visualize(
    vertices: np.ndarray,
    output_path: Path,
    faces: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    synthetic: bool = False,
    resolution: int = 96,
    label: Optional[str] = None,
    keypoint_names: Optional[tuple] = None,
) -> Path:
    grid = encode_hand_geometry(
        vertices,
        faces=faces,
        keypoints=keypoints,
        keypoint_names=keypoint_names,
        resolution=resolution,
    )
    surface_centers, keypoint_centers = decode_hand_geometry(grid)
    named_keypoints = (
        None
        if keypoint_names is None
        else decode_named_keypoints(grid, keypoint_names)
    )

    figure = plt.figure(figsize=(12, 5))
    source_axis = figure.add_subplot(1, 2, 1, projection="3d")
    source_axis.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], s=5, alpha=0.7, label="vertices"
    )
    if keypoints is not None:
        source_axis.scatter(
            keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
            s=24, marker="x", label="keypoints",
        )
    source_axis.set_title("Normalized input geometry")
    source_axis.set_xlabel("X")
    source_axis.set_ylabel("Y")
    source_axis.set_zlabel("Z")
    source_axis.set_xlim(-1, 1)
    source_axis.set_ylim(-1, 1)
    source_axis.set_zlim(-1, 1)
    source_axis.legend(loc="upper left")

    grid_axis = figure.add_subplot(1, 2, 2, projection="3d")
    if surface_centers.size:
        grid_axis.scatter(
            surface_centers[:, 0], surface_centers[:, 1], surface_centers[:, 2],
            s=7,
            alpha=0.45,
            label="vertex voxels" if faces is None else "surface voxels",
        )
    if keypoint_centers.size:
        if named_keypoints is None:
            grid_axis.scatter(
                keypoint_centers[:, 0],
                keypoint_centers[:, 1],
                keypoint_centers[:, 2],
                s=30,
                marker="x",
                label="keypoint voxels",
            )
        else:
            for name, centers in named_keypoints.items():
                if centers.size:
                    grid_axis.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        centers[:, 2],
                        s=28,
                        marker="x",
                    )
                    grid_axis.text(*centers[0], name, fontsize=6)
    grid_axis.set_title("Encoded occupancy cell centers ({})".format(resolution))
    grid_axis.set_xlabel("X")
    grid_axis.set_ylabel("Y")
    grid_axis.set_zlabel("Z")
    grid_axis.set_xlim(-1, 1)
    grid_axis.set_ylim(-1, 1)
    grid_axis.set_zlim(-1, 1)
    grid_axis.legend(loc="upper left")

    title = (
        "Synthetic demonstration (not FreiHAND data)"
        if synthetic
        else (label or "Normalized geometry input")
    )
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional .npz with normalized vertices and optional faces/keypoints arrays",
    )
    parser.add_argument("--dataset-root", type=Path, help="Official FreiHAND training root")
    parser.add_argument("--index", type=int, default=0, help="Training annotation index")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=96)
    args = parser.parse_args()

    if args.input is not None and args.dataset_root is not None:
        parser.error("choose either --input or --dataset-root")

    if args.dataset_root is not None:
        sample = FreiHANDDataset(args.dataset_root).load_sample(args.index)
        normalized, _ = normalize_hand(sample)
        if normalized.vertices is None:
            parser.error("selected FreiHAND sample has no vertex annotations")
        vertices = normalized.vertices
        faces = sample.faces
        keypoints = normalized.keypoints
        keypoint_names = FREIHAND_KEYPOINT_NAMES
        synthetic = False
        label = "FreiHAND training:{:08d}".format(args.index)
        default_output = (
            Path(__file__).parent
            / "visualizations"
            / "03_spatial_representation_training_{:08d}.png".format(args.index)
        )
    elif args.input is None:
        vertices = make_synthetic_hand()
        faces = None
        keypoints = None
        keypoint_names = None
        synthetic = True
        label = None
        default_output = (
            Path(__file__).parent
            / "visualizations"
            / "03_spatial_representation_synthetic.png"
        )
    else:
        with np.load(str(args.input), allow_pickle=False) as sample:
            if "vertices" not in sample:
                parser.error("input .npz must contain a 'vertices' array")
            vertices = sample["vertices"]
            faces = sample["faces"] if "faces" in sample else None
            keypoints = sample["keypoints"] if "keypoints" in sample else None
        keypoint_names = None
        synthetic = False
        label = "Normalized geometry from {}".format(args.input.name)
        default_output = (
            Path(__file__).parent
            / "visualizations"
            / "03_spatial_representation_from_npz.png"
        )

    saved = visualize(
        vertices,
        args.output or default_output,
        faces=faces,
        keypoints=keypoints,
        synthetic=synthetic,
        resolution=args.resolution,
        label=label,
        keypoint_names=keypoint_names,
    )
    print("Saved visualization to {}".format(saved))
    if synthetic:
        print("Input is synthetic demonstration geometry; no FreiHAND data was processed.")


if __name__ == "__main__":
    main()
