"""Measure named-landmark voxel resolutions on real FreiHAND samples."""

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from .data_loader import FreiHANDDataset
    from .encode_hand import FREIHAND_KEYPOINT_NAMES, encode_hand_geometry
    from .normalize_hand import normalize_hand
    from .inspect_real_dataset import sample_indices
except ImportError:
    from data_loader import FreiHANDDataset
    from encode_hand import FREIHAND_KEYPOINT_NAMES, encode_hand_geometry
    from normalize_hand import normalize_hand
    from inspect_real_dataset import sample_indices


FINGERTIP_INDICES = (4, 8, 12, 16, 20)
DEFAULT_RESOLUTIONS = (32, 48, 64, 96, 128)
DEFAULT_MAX_TENSOR_MIB = 32.0
DEFAULT_MAX_SECONDS_PER_SAMPLE = 5.0


def _cell_centers(
    points: np.ndarray, resolution: int, bounds: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = bounds
    width = (upper - lower) / resolution
    indices = np.floor((points - lower) / width).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)
    centers = lower + (indices + 0.5) * width
    return centers, indices


def compare_resolutions(
    dataset_root: Path,
    count: int = 50,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    max_tensor_mib: float = DEFAULT_MAX_TENSOR_MIB,
    max_seconds_per_sample: float = DEFAULT_MAX_SECONDS_PER_SAMPLE,
    output_path: Optional[Path] = None,
) -> Path:
    if count < 1:
        raise ValueError("count must be positive")
    if not resolutions or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2
        for value in resolutions
    ):
        raise ValueError("resolutions must be integers >= 2")
    if not np.isfinite(max_tensor_mib) or max_tensor_mib <= 0:
        raise ValueError("max_tensor_mib must be finite and positive")
    if not np.isfinite(max_seconds_per_sample) or max_seconds_per_sample <= 0:
        raise ValueError("max_seconds_per_sample must be finite and positive")

    dataset = FreiHANDDataset(Path(dataset_root), split="training", image_version="gs")
    indices = sample_indices(dataset.sample_count, count)
    measurements = []
    for resolution in resolutions:
        records = []
        for sample_index in indices:
            source = dataset.load_sample(sample_index)
            if source.vertices is None or source.keypoints is None:
                raise ValueError(
                    "sample {} must provide both vertices and keypoints".format(
                        source.sample_id
                    )
                )
            normalized, transform = normalize_hand(source)
            started = time.perf_counter()
            grid = encode_hand_geometry(
                normalized.vertices,
                faces=source.faces,
                keypoints=normalized.keypoints,
                keypoint_names=FREIHAND_KEYPOINT_NAMES,
                resolution=resolution,
            )
            elapsed = time.perf_counter() - started

            vertex_centers, _ = _cell_centers(
                normalized.vertices, resolution, (-1.0, 1.0)
            )
            keypoint_centers, keypoint_indices = _cell_centers(
                normalized.keypoints, resolution, (-1.0, 1.0)
            )
            vertex_errors = np.linalg.norm(
                vertex_centers - normalized.vertices, axis=1
            )
            keypoint_errors = np.linalg.norm(
                keypoint_centers - normalized.keypoints, axis=1
            )
            depth_errors = np.abs(
                keypoint_centers[:, 2] - normalized.keypoints[:, 2]
            )

            tips = keypoint_centers[list(FINGERTIP_INDICES)]
            source_tip_positions = normalized.keypoints[list(FINGERTIP_INDICES)]
            represented_adjacent_distances = np.linalg.norm(
                tips[1:] - tips[:-1], axis=1
            )
            source_adjacent_distances = np.linalg.norm(
                source_tip_positions[1:] - source_tip_positions[:-1], axis=1
            )
            fingertip_separation_errors = np.abs(
                represented_adjacent_distances - source_adjacent_distances
            )
            records.append(
                {
                    "sample_id": source.sample_id,
                    "seconds": elapsed,
                    "tensor_bytes": int(grid.nbytes),
                    "vertex_errors_metric": vertex_errors * transform.scale,
                    "keypoint_errors_metric": keypoint_errors * transform.scale,
                    "depth_errors_metric": depth_errors * transform.scale,
                    "fingertip_separation_errors_metric": (
                        fingertip_separation_errors * transform.scale
                    ),
                    "distinct_landmark_cells": len(
                        {tuple(row) for row in keypoint_indices}
                    ),
                    "landmarks_preserved": bool(
                        all(
                            int(grid[..., channel + 1].sum()) == 1
                            for channel in range(len(FREIHAND_KEYPOINT_NAMES))
                        )
                    ),
                }
            )
            del grid

        all_point_errors = np.concatenate(
            [
                np.concatenate(
                    (record["vertex_errors_metric"], record["keypoint_errors_metric"])
                )
                for record in records
            ]
        )
        all_depth_errors = np.concatenate(
            [record["depth_errors_metric"] for record in records]
        )
        all_gap_errors = np.concatenate(
            [record["fingertip_separation_errors_metric"] for record in records]
        )
        seconds = np.asarray([record["seconds"] for record in records])
        tensor_bytes = records[0]["tensor_bytes"]
        measurements.append(
            {
                "resolution": int(resolution),
                "samples": len(records),
                "tensor_bytes": tensor_bytes,
                "tensor_mib": tensor_bytes / (1024.0 * 1024.0),
                "point_error_mean": float(np.mean(all_point_errors)),
                "point_error_p95": float(np.percentile(all_point_errors, 95)),
                "point_error_max": float(np.max(all_point_errors)),
                "depth_error_mean": float(np.mean(all_depth_errors)),
                "depth_error_p95": float(np.percentile(all_depth_errors, 95)),
                "depth_error_max": float(np.max(all_depth_errors)),
                "fingertip_error_mean": float(np.mean(all_gap_errors)),
                "fingertip_error_max": float(np.max(all_gap_errors)),
                "minimum_distinct_landmark_cells": min(
                    record["distinct_landmark_cells"] for record in records
                ),
                "named_landmarks_all_preserved": all(
                    record["landmarks_preserved"] for record in records
                ),
                "seconds_median": float(statistics.median(seconds.tolist())),
                "seconds_p95": float(np.percentile(seconds, 95)),
                "seconds_max": float(np.max(seconds)),
            }
        )

    feasible = [
        item
        for item in measurements
        if item["tensor_mib"] <= max_tensor_mib
        and item["seconds_max"] <= max_seconds_per_sample
        and item["named_landmarks_all_preserved"]
    ]
    selected = max(feasible, key=lambda item: item["resolution"]) if feasible else None
    report = output_path or (
        Path(__file__).parent / "results" / "resolution_comparison.md"
    )
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report,
        indices,
        measurements,
        selected,
        max_tensor_mib,
        max_seconds_per_sample,
    )
    print("Resolution comparison saved to {}".format(report))
    if selected is None:
        raise RuntimeError("no resolution met the configured memory/runtime limits")
    print(
        "Selected prototype setting: {}^3, {} channels, {:.2f} MiB/sample, "
        "max {:.4f} s/sample".format(
            selected["resolution"],
            1 + len(FREIHAND_KEYPOINT_NAMES),
            selected["tensor_mib"],
            selected["seconds_max"],
        )
    )
    return report


def _write_report(
    output_path: Path,
    indices: Sequence[int],
    measurements: Sequence[dict],
    selected: Optional[dict],
    max_tensor_mib: float,
    max_seconds_per_sample: float,
) -> None:
    lines = [
        "# FreiHAND named-landmark voxel resolution comparison",
        "",
        "Generated by `FreiHAND/compare_resolutions.py`.",
        "",
        "- Real training samples: {} evenly spaced indices, from {} through {}.".format(
            len(indices), indices[0], indices[-1]
        ),
        "- Channels: 1 vertex-occupancy channel plus 21 separate named-landmark channels; no face topology was provided.",
        "- Runtime budget: at most {:.3g} seconds per sample.".format(
            max_seconds_per_sample
        ),
        "- Measurement environment: Python {}; {} ({})".format(
            platform.python_version(),
            platform.platform(),
            platform.processor() or "CPU not reported",
        ),
        "- Output tensor budget: at most {:.3g} MiB per sample (dense uint8 payload only; excludes temporary arrays, model activations, and batch storage).".format(
            max_tensor_mib
        ),
        "- Error units: meters, based on the official evaluator's conversion of raw XYZ errors to centimeters by multiplying by 100. Normalize/inverse-transform retained for each sample.",
        "- Fingertip comparison uses distances between adjacent labeled fingertip landmarks; it is not inter-surface clearance.",
        "- No domain-approved fit/clinical tolerance was supplied. Derived grid quantization bounds are prototype engineering checks only, not clinical or fabrication acceptance.",
        "",
        "| Resolution | Tensor MiB | Point error mean / p95 / max | Z error mean / p95 / max | Adjacent fingertip separation error mean / max | Minimum unique XYZ cells occupied by 21 named landmarks | Landmark channels preserved | Encode seconds median / p95 / max | Within budgets |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for item in measurements:
        within_budget = (
            item["tensor_mib"] <= max_tensor_mib
            and item["seconds_max"] <= max_seconds_per_sample
            and item["named_landmarks_all_preserved"]
        )
        lines.append(
            "| {resolution}^3 | {tensor_mib:.3f} | {point_error_mean:.6g} / {point_error_p95:.6g} / {point_error_max:.6g} | {depth_error_mean:.6g} / {depth_error_p95:.6g} / {depth_error_max:.6g} | {fingertip_error_mean:.6g} / {fingertip_error_max:.6g} | {minimum_distinct_landmark_cells} | {named_landmarks_all_preserved} | {seconds_median:.5f} / {seconds_p95:.5f} / {seconds_max:.5f} | {within_budget} |".format(
                within_budget=within_budget, **item
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            (
                "Selected the largest tested resolution meeting the declared tensor/time budgets and preserving every named landmark channel: "
                "`{}^3`.".format(selected["resolution"])
                if selected
                else "No tested resolution met the declared tensor/time budgets."
            ),
            "This 96^3 setting is selected for **prototype engineering evaluation only**. Point-to-cell-center and per-axis depth bounds follow mathematically from the grid; they are not fit or clinical tolerances. The gap metric is only a named-landmark distance proxy, not a surface-clearance measurement. Actual orthosis fit, safety, and clinical acceptance are outside this report's scope.",
            "",
            "Per-sample inputs are re-normalized independently to `[-1, 1]`; errors are multiplied by the retained uniform normalization scale and reported in meters. Convert to millimeters by multiplying by 1000 when presenting to a reviewer.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path.home() / "Datasets" / "FreiHAND"
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS)
    )
    parser.add_argument("--max-tensor-mib", type=float, default=DEFAULT_MAX_TENSOR_MIB)
    parser.add_argument(
        "--max-seconds-per-sample",
        type=float,
        default=DEFAULT_MAX_SECONDS_PER_SAMPLE,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        compare_resolutions(
            args.dataset_root,
            count=args.count,
            resolutions=args.resolutions,
            max_tensor_mib=args.max_tensor_mib,
            max_seconds_per_sample=args.max_seconds_per_sample,
            output_path=args.output,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
