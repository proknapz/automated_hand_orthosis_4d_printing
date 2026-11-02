"""Train and evaluate a small 3D-CNN reconstruction prototype."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

try:
    from .keras_voxel_loader import FreiHANDReconstructionSequence
except ImportError:
    from keras_voxel_loader import FreiHANDReconstructionSequence


def build_reconstruction_model(
    input_shape: Tuple[int, int, int, int] = (96, 96, 96, 1),
    vertex_count: int = 778,
) -> tf.keras.Model:
    """Build a compact channels-last CNN that predicts ordered XYZ vertices."""
    if len(input_shape) != 4 or input_shape[-1] != 1:
        raise ValueError("input_shape must be a 4D spatial tensor with one channel")
    if any(size < 1 for size in input_shape):
        raise ValueError("input_shape dimensions must be positive")
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, int) or vertex_count < 1:
        raise ValueError("vertex_count must be a positive integer")

    inputs = tf.keras.Input(shape=input_shape, name="vertex_occupancy")
    x = tf.keras.layers.Conv3D(
        4, kernel_size=3, strides=2, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.Conv3D(
        8, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv3D(
        16, kernel_size=3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        vertex_count * 3, name="normalized_vertex_coordinates"
    )(x)
    outputs = tf.keras.layers.Reshape(
        (vertex_count, 3), name="ordered_vertices"
    )(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


def _load_split(
    dataset_dir: Path,
    split_name: str,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
) -> Tuple[List[int], FreiHANDReconstructionSequence]:
    manifest_path = dataset_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing benchmark split manifest: {}".format(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not read split manifest: {}".format(error)) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        raise ValueError("split manifest must contain a splits object")
    indices = manifest["splits"].get(split_name)
    if not isinstance(indices, list) or not indices:
        raise ValueError("split {!r} must contain at least one sample".format(split_name))
    paths = [dataset_dir / "training_{:08d}.npz".format(index) for index in indices]
    return indices, FreiHANDReconstructionSequence(
        paths, batch_size=batch_size, shuffle=shuffle, seed=seed
    )


def _mean_training_target(sequence: FreiHANDReconstructionSequence) -> np.ndarray:
    targets = []
    for batch_index in range(len(sequence)):
        _, batch_targets = sequence[batch_index]
        targets.append(batch_targets.numpy())
    return np.mean(np.concatenate(targets, axis=0), axis=0).astype(np.float32)


def evaluate_model(
    model: tf.keras.Model,
    sequence: FreiHANDReconstructionSequence,
    mean_target: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate vertex error in normalized coordinates and source millimeters."""
    sample_metrics = []
    for batch_index in range(len(sequence)):
        inputs, target_batch = sequence[batch_index]
        targets = target_batch.numpy()
        predictions = model.predict_on_batch(inputs).astype(np.float64)
        metadata_items = sequence.metadata_for_batch(batch_index)
        for target, prediction, metadata in zip(targets, predictions, metadata_items):
            scale = metadata.get(
                "normalization_scale_source_units_per_normalized_unit"
            )
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not np.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError("sample metadata has an invalid normalization scale")
            baseline = np.broadcast_to(mean_target, target.shape).astype(np.float64)
            model_error = np.linalg.norm(prediction - target, axis=1)
            baseline_error = np.linalg.norm(baseline - target, axis=1)
            sample_metrics.append(
                {
                    "sample_id": metadata["sample_id"],
                    "model_mean_vertex_error_normalized": float(
                        np.mean(model_error)
                    ),
                    "model_mean_vertex_error_mm": float(
                        np.mean(model_error) * scale * 1000.0
                    ),
                    "baseline_mean_vertex_error_normalized": float(
                        np.mean(baseline_error)
                    ),
                    "baseline_mean_vertex_error_mm": float(
                        np.mean(baseline_error) * scale * 1000.0
                    ),
                }
            )

    def aggregate(key: str) -> Dict[str, float]:
        values = np.asarray([item[key] for item in sample_metrics], dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "sample_count": len(sample_metrics),
        "model_mean_vertex_error_normalized": aggregate(
            "model_mean_vertex_error_normalized"
        ),
        "model_mean_vertex_error_mm": aggregate("model_mean_vertex_error_mm"),
        "training_mean_shape_baseline_error_normalized": aggregate(
            "baseline_mean_vertex_error_normalized"
        ),
        "training_mean_shape_baseline_error_mm": aggregate(
            "baseline_mean_vertex_error_mm"
        ),
        "per_sample": sample_metrics,
    }


def train(
    dataset_dir: Path,
    epochs: int = 10,
    batch_size: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit on the manifest training split and report held-out test metrics."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dataset_dir = Path(dataset_dir)
    train_indices, train_sequence = _load_split(
        dataset_dir, "train", batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_indices, validation_sequence = _load_split(
        dataset_dir, "validation", batch_size=batch_size
    )
    test_indices, test_sequence = _load_split(
        dataset_dir, "test", batch_size=batch_size
    )
    if set(train_indices) & set(validation_indices) or set(train_indices) & set(
        test_indices
    ) or set(validation_indices) & set(test_indices):
        raise ValueError("train, validation, and test splits must be disjoint")

    model = build_reconstruction_model()
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    mean_target = _mean_training_target(train_sequence)
    test_metrics = evaluate_model(model, test_sequence, mean_target)
    final_training_loss = model.evaluate(train_sequence, verbose=0)
    final_validation_loss = model.evaluate(validation_sequence, verbose=0)
    report = {
        "status": "prototype reconstruction benchmark; not orthosis or clinical validation",
        "task": {
            "input": "single vertex-occupancy channel only",
            "target": "778 ordered normalized vertex coordinates",
            "normalization": "per-sample vertex bounds only",
            "vertex_correspondence": (
                "assumed consistent across samples; verify before anatomical interpretation"
            ),
        },
        "seed": seed,
        "requested_epochs": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "restored_model_training_loss_mse": float(final_training_loss),
        "restored_model_validation_loss_mse": float(final_validation_loss),
        "test": test_metrics,
    }
    model.save(str(dataset_dir / "reconstruction_baseline.h5"), include_optimizer=True)
    (dataset_dir / "reconstruction_baseline_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.home() / "Datasets" / "FreiHAND" / "reconstruction_prototype",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = train(
            args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError, IndexError) as error:
        parser.error(str(error))
    print("Completed {} training epoch(s).".format(report["epochs_completed"]))
    print("Test samples: {}".format(report["test"]["sample_count"]))
    print(
        "Mean vertex error (mm): CNN {:.3f}; train-mean baseline {:.3f}".format(
            report["test"]["model_mean_vertex_error_mm"]["mean"],
            report["test"]["training_mean_shape_baseline_error_mm"]["mean"],
        )
    )
    print("Saved model and metrics under {}".format(args.dataset_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
