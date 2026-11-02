"""Inspect real FreiHAND samples and exercise Person 2 encoding."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    from .data_loader import FreiHANDDataset
    from .encode_hand import (
        FREIHAND_KEYPOINT_NAMES,
        decode_hand_geometry,
        decode_named_keypoints,
        encode_hand_geometry,
    )
    from .normalize_hand import denormalize_hand, normalize_hand
except ImportError:
    from data_loader import FreiHANDDataset
    from encode_hand import (
        FREIHAND_KEYPOINT_NAMES,
        decode_hand_geometry,
        decode_named_keypoints,
        encode_hand_geometry,
    )
    from normalize_hand import denormalize_hand, normalize_hand


def sample_indices(sample_count: int, count: int) -> List[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if sample_count < 1:
        raise ValueError("dataset must contain at least one sample")
    return sorted(
        set(np.linspace(0, sample_count - 1, min(sample_count, count), dtype=int).tolist())
    )


def inspect_samples(
    dataset_root: Path,
    count: int = 50,
    resolution: int = 96,
    output_path: Optional[Path] = None,
) -> Path:
    """Check normalization/voxel encoding and write measured sample statistics."""
    dataset = FreiHANDDataset(Path(dataset_root), split="training", image_version="gs")
    indices = sample_indices(dataset.sample_count, count)
    all_keypoints = np.asarray(dataset._load_annotations()[2], dtype=np.float64)
    if all_keypoints.ndim != 3 or all_keypoints.shape[1:] != (21, 3):
        raise ValueError(
            "training keypoint annotations must have shape (N, 21, 3), got {}".format(
                all_keypoints.shape
            )
        )
    scale_annotations = dataset._load_scales(len(all_keypoints))
    if scale_annotations is None:
        raise FileNotFoundError(
            "training_scale.json is required for the reference-bone consistency check"
        )
    all_reference_scales = np.asarray(
        scale_annotations, dtype=np.float64
    ).reshape(-1)
    if all_reference_scales.shape != (len(all_keypoints),):
        raise ValueError("training reference-bone scales must contain one value per sample")
    reference_lengths = np.linalg.norm(
        all_keypoints[:, 9, :] - all_keypoints[:, 10, :], axis=1
    )
    reference_scale_max_error = float(
        np.max(np.abs(all_reference_scales - reference_lengths))
    )
    if not np.array_equal(all_reference_scales, reference_lengths):
        raise ValueError(
            "reference-bone scales do not exactly match keypoint distances "
            "between indices 9 and 10 (maximum absolute difference {})".format(
                reference_scale_max_error
            )
        )

    records = []
    failures = []

    for index in indices:
        try:
            source = dataset.load_sample(index)
            if source.vertices is None:
                raise ValueError("training vertex annotation was not loaded")
            if source.keypoints is None or source.metadata.get("hand_scale") is None:
                raise ValueError("keypoints or metric reference-bone scale is missing")

            normalized, transform = normalize_hand(source)
            repeated, repeated_transform = normalize_hand(source)
            restored = denormalize_hand(normalized, transform)
            if not np.array_equal(normalized.vertices, repeated.vertices):
                raise AssertionError("vertex normalization was not deterministic")
            if not np.array_equal(normalized.keypoints, repeated.keypoints):
                raise AssertionError("keypoint normalization was not deterministic")
            if transform.scale != repeated_transform.scale:
                raise AssertionError("normalization scale was not deterministic")
            if not np.allclose(restored.vertices, source.vertices, rtol=0.0, atol=1e-10):
                raise AssertionError("vertex inverse-transform check failed")
            if not np.allclose(restored.keypoints, source.keypoints, rtol=0.0, atol=1e-10):
                raise AssertionError("keypoint inverse-transform check failed")

            tensor = encode_hand_geometry(
                normalized.vertices,
                faces=source.faces,
                keypoints=normalized.keypoints,
                keypoint_names=FREIHAND_KEYPOINT_NAMES,
                resolution=resolution,
            )
            expected_channels = 1 + len(FREIHAND_KEYPOINT_NAMES)
            if tensor.shape != (
                resolution,
                resolution,
                resolution,
                expected_channels,
            ):
                raise AssertionError("encoder returned unexpected shape {}".format(tensor.shape))
            if tensor.dtype != np.uint8 or not np.isin(tensor, (0, 1)).all():
                raise AssertionError("encoder returned an invalid dtype or value")
            decoded_vertices, decoded_keypoints = decode_hand_geometry(tensor)
            decoded_named_keypoints = decode_named_keypoints(
                tensor, FREIHAND_KEYPOINT_NAMES
            )

            def quantization_errors(points):
                indices_3d = np.floor((points + 1.0) * (resolution / 2.0)).astype(int)
                indices_3d = np.clip(indices_3d, 0, resolution - 1)
                centers = -1.0 + (indices_3d + 0.5) * (2.0 / resolution)
                error = np.linalg.norm(points - centers, axis=1)
                return float(error.mean()), float(error.max())

            vertex_mean_error, vertex_max_error = quantization_errors(normalized.vertices)
            keypoint_mean_error, keypoint_max_error = quantization_errors(
                normalized.keypoints
            )
            normalized_fingertips = normalized.keypoints[[4, 8, 12, 16, 20]]
            encoded_fingertips = np.asarray(
                [
                    decoded_named_keypoints[name][0]
                    for name in (
                        "thumb_tip",
                        "index_tip",
                        "middle_tip",
                        "ring_tip",
                        "pinky_tip",
                    )
                ]
            )
            source_tip_gaps = np.linalg.norm(
                normalized_fingertips[1:] - normalized_fingertips[:-1], axis=1
            )
            encoded_tip_gaps = np.linalg.norm(
                encoded_fingertips[1:] - encoded_fingertips[:-1], axis=1
            )
            fingertip_gap_errors = np.abs(encoded_tip_gaps - source_tip_gaps)
            keypoint_voxel_indices = np.floor(
                (normalized.keypoints + 1.0) * (resolution / 2.0)
            ).astype(int)
            keypoint_voxel_indices = np.clip(
                keypoint_voxel_indices, 0, resolution - 1
            )
            distinct_landmark_cells = len(
                {tuple(index) for index in keypoint_voxel_indices}
            )
            raw_min = source.vertices.min(axis=0)
            raw_max = source.vertices.max(axis=0)
            raw_keypoint_min = source.keypoints.min(axis=0)
            raw_keypoint_max = source.keypoints.max(axis=0)
            projected_keypoints = (
                source.metadata["camera_intrinsics"] @ source.keypoints.T
            ).T
            projected_keypoints = (
                projected_keypoints[:, :2] / projected_keypoints[:, 2:3]
            )
            records.append(
                {
                    "index": index,
                    "vertices": int(source.vertices.shape[0]),
                    "keypoints": int(source.keypoints.shape[0]),
                    "scale": float(transform.scale),
                    "reference_bone_scale": float(source.metadata["hand_scale"]),
                    "raw_min": raw_min,
                    "raw_max": raw_max,
                    "raw_keypoint_min": raw_keypoint_min,
                    "raw_keypoint_max": raw_keypoint_max,
                    "projected_keypoint_min": projected_keypoints.min(axis=0),
                    "projected_keypoint_max": projected_keypoints.max(axis=0),
                    "surface_voxels": int(tensor[..., 0].sum()),
                    "keypoint_voxels": int(tensor[..., 1].sum()),
                    "decoded_surface_points": int(decoded_vertices.shape[0]),
                    "decoded_keypoint_points": int(decoded_keypoints.shape[0]),
                    "vertex_mean_error": vertex_mean_error,
                    "vertex_max_error": vertex_max_error,
                    "keypoint_mean_error": keypoint_mean_error,
                    "keypoint_max_error": keypoint_max_error,
                    "distinct_landmark_cells": distinct_landmark_cells,
                    "fingertip_gap_mean_error": float(
                        fingertip_gap_errors.mean()
                    ),
                    "fingertip_gap_max_error": float(
                        fingertip_gap_errors.max()
                    ),
                }
            )
            print(
                "PASS {} V={} K={} tensor={} surface_cells={} scale={:.8g} "
                "ref_bone={:.8g}".format(
                    source.sample_id,
                    source.vertices.shape[0],
                    source.keypoints.shape[0],
                    tensor.shape,
                    int(tensor[..., 0].sum()),
                    transform.scale,
                    float(source.metadata["hand_scale"]),
                )
            )
        except (OSError, ValueError, IndexError, AssertionError) as error:
            failures.append((index, str(error)))
            print("FAIL training:{:08d}: {}".format(index, error))

    report_path = output_path or (
        Path(__file__).parent / "results" / "real_sample_inspection.md"
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report_path,
        indices,
        records,
        failures,
        resolution,
        reference_scale_max_error,
        len(all_keypoints),
    )
    print("Report saved to {}".format(report_path))
    if failures:
        raise RuntimeError(
            "{} of {} selected samples failed; see {}".format(
                len(failures), len(indices), report_path
            )
        )
    return report_path


def _write_report(
    output_path: Path,
    indices: Sequence[int],
    records: Sequence[dict],
    failures: Sequence[tuple],
    resolution: int,
    reference_scale_max_error: float,
    reference_scale_sample_count: int,
) -> None:
    lines = [
        "# Real FreiHAND sample inspection",
        "",
        "Generated by `FreiHAND/inspect_real_dataset.py`.",
        "",
        "- Requested samples: {}".format(len(indices)),
        "- Passed: {}".format(len(records)),
        "- Failed: {}".format(len(failures)),
        "- Indices: {}".format(", ".join(str(index) for index in indices)),
        "- Encoder tensor: `({}, {}, {}, {})`, `uint8`, binary; channel 0 is vertex occupancy and channels 1-21 preserve named FreiHAND keypoint identity.".format(
            resolution,
            resolution,
            resolution,
            1 + len(FREIHAND_KEYPOINT_NAMES),
        ),
        "- Coordinate units: meters, supported by the official evaluator reporting raw XYZ errors in centimeters after multiplying by 100.",
        "- Coordinate frame: camera-relative projection frame is inferred from the official `K` projection; formal origin, axis names/signs, and handedness are unspecified.",
        "- Full-training reference-bone check: `training_scale` matches `norm(xyz[9] - xyz[10])` for {} samples; maximum absolute difference `{:.8g}`.".format(
            reference_scale_sample_count, reference_scale_max_error
        ),
        "- Validation uses normalized vertices and keypoints; faces are not supplied.",
        "- Surface occupancy therefore marks the dataset's sampled vertices only; it is not a dense mesh-surface encoding.",
        "",
    ]
    if records:
        scales = np.asarray([record["scale"] for record in records])
        reference_scales = np.asarray(
            [record["reference_bone_scale"] for record in records]
        )
        raw_mins = np.stack([record["raw_min"] for record in records])
        raw_maxs = np.stack([record["raw_max"] for record in records])
        raw_keypoint_mins = np.stack(
            [record["raw_keypoint_min"] for record in records]
        )
        raw_keypoint_maxs = np.stack(
            [record["raw_keypoint_max"] for record in records]
        )
        projected_mins = np.stack(
            [record["projected_keypoint_min"] for record in records]
        )
        projected_maxs = np.stack(
            [record["projected_keypoint_max"] for record in records]
        )
        lines.extend(
            [
                "## Measured summary",
                "",
                "- Vertex count set: `{}`".format(
                    sorted(set(record["vertices"] for record in records))
                ),
                "- Keypoint count set: `{}`".format(
                    sorted(set(record["keypoints"] for record in records))
                ),
                "- Raw vertex coordinate minima by axis, across sample bounds: `{}`".format(
                    np.min(raw_mins, axis=0).tolist()
                ),
                "- Raw vertex coordinate maxima by axis, across sample bounds: `{}`".format(
                    np.max(raw_maxs, axis=0).tolist()
                ),
                "- Raw keypoint coordinate minima by axis: `{}`".format(
                    np.min(raw_keypoint_mins, axis=0).tolist()
                ),
                "- Raw keypoint coordinate maxima by axis: `{}`".format(
                    np.max(raw_keypoint_maxs, axis=0).tolist()
                ),
                "- Projected keypoint pixel bounds (u,v) across samples: min `{}`, max `{}` (reference uses 224 x 224 crops).".format(
                    np.min(projected_mins, axis=0).tolist(),
                    np.max(projected_maxs, axis=0).tolist(),
                ),
                "- Bounding-box normalization scale median / min / max (source units per normalized unit): `{:.8g}` / `{:.8g}` / `{:.8g}`".format(
                    float(np.median(scales)), float(np.min(scales)), float(np.max(scales))
                ),
                "- Dataset reference-bone scale median / min / max (meters): `{:.8g}` / `{:.8g}` / `{:.8g}`".format(
                    float(np.median(reference_scales)),
                    float(np.min(reference_scales)),
                    float(np.max(reference_scales)),
                ),
                "- Per-vertex quantization error in normalized coordinates: mean over samples `{:.8g}`, worst sample maximum `{:.8g}`.".format(
                    float(np.mean([record["vertex_mean_error"] for record in records])),
                    float(np.max([record["vertex_max_error"] for record in records])),
                ),
                "- Per-keypoint quantization error in normalized coordinates: mean over samples `{:.8g}`, worst sample maximum `{:.8g}`.".format(
                    float(np.mean([record["keypoint_mean_error"] for record in records])),
                    float(np.max([record["keypoint_max_error"] for record in records])),
                ),
                "- Distinct landmark cells: minimum across samples `{}` of 21 (identity is retained in separate channels even if positions share a cell).".format(
                    min(record["distinct_landmark_cells"] for record in records)
                ),
                "- Adjacent fingertip landmark-separation error in normalized coordinates: mean `{:.8g}`, worst pair/sample `{:.8g}`. This is a landmark-distance proxy, not surface clearance.".format(
                    float(
                        np.mean(
                            [
                                record["fingertip_gap_mean_error"]
                                for record in records
                            ]
                        )
                    ),
                    float(
                        np.max(
                            [
                                record["fingertip_gap_max_error"]
                                for record in records
                            ]
                        )
                    ),
                ),
                "",
                "## Per-sample results",
                "",
                "| Index | Vertices | Keypoints | Ref-bone scale | Normalizer scale | Surface voxels | Distinct landmark cells | Vertex max quant. error | Keypoint max quant. error | Max fingertip-gap error |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in records:
            lines.append(
                "| {index} | {vertices} | {keypoints} | {reference_bone_scale:.8g} | {scale:.8g} | {surface_voxels} | {distinct_landmark_cells} | {vertex_max_error:.8g} | {keypoint_max_error:.8g} | {fingertip_gap_max_error:.8g} |".format(
                    **record
                )
            )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend("- `training:{:08d}`: {}".format(index, message) for index, message in failures)
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "These measurements describe only the selected samples and the current prototype normalization/encoder. They do not establish clinical tolerances, a canonical anatomical orientation, or orthosis fit. The encoder's surface channel is vertex occupancy because no face topology was supplied; point quantization errors do not measure distance to the continuous hand surface.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "Datasets" / "FreiHAND",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        inspect_samples(
            args.dataset_root,
            count=args.count,
            resolution=args.resolution,
            output_path=args.output,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
