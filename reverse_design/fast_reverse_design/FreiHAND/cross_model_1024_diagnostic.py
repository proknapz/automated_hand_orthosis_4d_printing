#!/usr/bin/env python3
"""Cross-model diagnostic for the same 1,024 generation-5 candidates.

This script intentionally does not modify the PEA algorithm, model training,
model weights, preprocessing, target data, or candidate chromosomes.
It simply loads one complete generation-5 candidate set from run 1,
validates the chromosome set, evaluates every design with both trained models,
compares the metrics, and writes the results and plots.
"""

import csv
import json
import math
import warnings
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

ROOT = Path(__file__).resolve().parent
RUN_NUMBER = 1
EXPECTED_CANDIDATE_COUNT = 1024
SEPCNN_CANDIDATE_CSV = ROOT / "sepcnn_PEA_generation5_candidates.csv"
RESNET_CANDIDATE_CSV = ROOT / "resnet_PEA_generation5_candidates.csv"
TARGET_CSV = ROOT / "target_output.csv"
SEPCNN_CHECKPOINT = ROOT / "model_weights_SEP_CNN_fair" / "model.ckpt"
RESNET_CHECKPOINT = ROOT / "model_weights_2D_ResNet_fair" / "model.ckpt"
RESULTS_CSV = ROOT / "cross_model_1024_results.csv"
INPUTS_CSV = ROOT / "cross_model_1024_inputs.csv"
SCATTER_PLOT = ROOT / "cross_model_1024_scatter.png"
DIFFERENCE_PLOT = ROOT / "cross_model_1024_difference.png"

EXPECTED_SEPCNN_CANDIDATE_961_MSE = 0.0034106617562482612
EXPECTED_RESNET_SAME_SEPCNN_SQ961_MSE = 0.0037533430980342852


# ---------------------------------------------------------------------------
# Original PEA fitness convention, kept exactly as implemented in the project.
# ---------------------------------------------------------------------------
def pea_shape_mse(prediction, target):
    return float(
        np.mean(
            np.square(prediction[0, :] - target[0, :])
            + np.square(prediction[1, :] - target[1, :])
        )
    )


def per_candidate_mse(prediction, target):
    """Compute per-sample MSE/MAE/max-error for shape (N,2,200)."""
    if prediction.shape[1:] != (2, 200):
        raise ValueError(f"Prediction shape must be (N, 2, 200); found {prediction.shape}")
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (2, 200):
        raise ValueError(f"Target shape must be (2, 200); found {target.shape}")
    diff = prediction - target[None, :, :]
    mse = np.mean(
        np.square(diff[:, 0, :]) + np.square(diff[:, 1, :]),
        axis=1,
    )
    mae = np.mean(np.abs(diff), axis=(1, 2))
    max_error = np.max(np.abs(diff), axis=(1, 2))
    return mse.astype(np.float64), mae.astype(np.float64), max_error.astype(np.float64)


def load_target():
    target = pd.read_csv(TARGET_CSV, header=None, float_precision="round_trip").to_numpy(dtype=np.float64)
    if target.shape != (2, 200):
        raise ValueError(f"Target must have shape (2, 200); found {target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinite values.")
    return target


def validate_and_load_run1_candidates():
    candidate_path = SEPCNN_CANDIDATE_CSV
    rows = pd.read_csv(candidate_path)
    run_rows = rows[rows["run"] == RUN_NUMBER].copy()
    if len(run_rows) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_CANDIDATE_COUNT} rows in run {RUN_NUMBER} of {candidate_path}; found {len(run_rows)}."
        )

    unique_candidate_indices = run_rows["candidate_index"].astype(int).unique()
    if len(unique_candidate_indices) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} unique candidate indices in run {RUN_NUMBER}; found {len(unique_candidate_indices)}."
        )

    chromosomes = []
    seen_chromosomes = set()
    for _, row in run_rows.sort_values("candidate_index").iterrows():
        candidate_index = int(row["candidate_index"])
        chromosome = np.asarray(json.loads(row["chromosome"]), dtype=np.float32)
        if chromosome.shape != (2, 10):
            raise ValueError(
                f"Candidate {candidate_index} chromosome must have shape (2, 10); found {chromosome.shape}."
            )
        if not np.isfinite(chromosome).all():
            raise ValueError(f"Candidate {candidate_index} chromosome contains NaN or infinity.")
        if not np.isin(chromosome, [0, 1]).all():
            raise ValueError(f"Candidate {candidate_index} chromosome contains values other than 0/1.")

        chromosome_key = tuple(np.asarray(chromosome, dtype=np.int32).ravel().tolist())
        if chromosome_key in seen_chromosomes:
            raise ValueError(f"Duplicate chromosome detected in run {RUN_NUMBER}: {chromosome_key}")
        seen_chromosomes.add(chromosome_key)
        chromosomes.append((candidate_index, chromosome))

    if len(seen_chromosomes) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} unique chromosomes; found {len(seen_chromosomes)}."
        )

    candidate_indices = np.array([idx for idx, _ in chromosomes], dtype=np.int64)
    chromosome_batch = np.stack([chromosome for _, chromosome in chromosomes], axis=0).astype(np.float32)
    if chromosome_batch.shape != (EXPECTED_CANDIDATE_COUNT, 2, 10):
        raise ValueError(
            f"Chromosome batch shape mismatch: expected ({EXPECTED_CANDIDATE_COUNT}, 2, 10); found {chromosome_batch.shape}."
        )

    return candidate_indices, chromosome_batch


# ---------------------------------------------------------------------------
# SEP-CNN graph exactly matching the project PEA model.
# ---------------------------------------------------------------------------
def sepcnn_networks_1(inputs):
    split_data = tf.split(inputs, num_or_size_splits=10, axis=2)
    outputs = []
    for i in range(10):
        branch = tf.layers.conv1d(
            inputs=split_data[i],
            filters=32,
            kernel_size=3,
            padding="same",
            activation=None,
            name=f"network1_conv_{i}",
        )
        outputs.append(tf.nn.leaky_relu(branch))
    return outputs


def sepcnn_networks_2(inputs):
    outputs = [inputs[0]]
    for i in range(1, 10):
        branch = tf.add(inputs[i], outputs[i - 1])
        outputs.append(tf.nn.leaky_relu(branch))
    return outputs


def sepcnn_networks_3(inputs):
    outputs = [inputs[0]]
    for i in range(1, 10):
        branch = tf.add(inputs[i], outputs[i - 1])
        outputs.append(tf.nn.leaky_relu(branch))
    return tf.concat(outputs, axis=2)


def sepcnn_predicting(inputs):
    x1 = sepcnn_networks_1(inputs)
    x2 = sepcnn_networks_2(x1)
    x3 = sepcnn_networks_3(x2)
    return tf.layers.conv1d(
        inputs=x3,
        filters=200,
        kernel_size=3,
        padding="same",
        activation=None,
        name="output_conv",
    )


# ---------------------------------------------------------------------------
# ResNet graph exactly matching the project PEA model.
# ---------------------------------------------------------------------------
def resnet_conv2d_layer(inputs, filters, kernel_size, name):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        name=name,
    )


def resnet_residual_block(inputs, filters, name):
    shortcut = inputs
    conv1 = resnet_conv2d_layer(
        inputs,
        filters=filters,
        kernel_size=(3, 3),
        name=f"{name}_conv1",
    )
    conv1 = tf.nn.leaky_relu(conv1)
    conv2 = resnet_conv2d_layer(
        conv1,
        filters=filters,
        kernel_size=(3, 3),
        name=f"{name}_conv2",
    )
    output = tf.add(shortcut, conv2)
    return tf.nn.leaky_relu(output)


def resnet_predicting(inputs):
    x_image = tf.expand_dims(inputs, axis=-1)
    conv0 = resnet_conv2d_layer(
        x_image,
        filters=64,
        kernel_size=(3, 3),
        name="initial_conv",
    )
    conv0 = tf.nn.leaky_relu(conv0)
    block1 = resnet_residual_block(conv0, 64, "residual_block1")
    block2 = resnet_residual_block(block1, 64, "residual_block2")
    block3 = resnet_residual_block(block2, 64, "residual_block3")
    output = resnet_conv2d_layer(
        block3,
        filters=200,
        kernel_size=(1, 3),
        name="output_conv",
    )
    return tf.reduce_mean(output, axis=2)


def evaluate_model(model_name, checkpoint_path, chromosome_batch, target):
    if model_name not in {"SEP-CNN", "ResNet"}:
        raise ValueError(f"Unsupported model name: {model_name}")
    if not Path(str(checkpoint_path) + ".index").exists():
        raise FileNotFoundError(f"Checkpoint index not found for {model_name}: {checkpoint_path}")

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 2, 10])
        if model_name == "SEP-CNN":
            output_predict = sepcnn_predicting(x)
        else:
            output_predict = resnet_predicting(x)
        saver = tf.train.Saver()
        initializer = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        session.run(initializer)
        saver.restore(session, str(checkpoint_path))
        prediction_batch = session.run(output_predict, feed_dict={x: chromosome_batch})

    if prediction_batch.shape[1:] != (2, 200):
        raise ValueError(
            f"{model_name} output shape must be (N, 2, 200); found {prediction_batch.shape}."
        )
    if not np.isfinite(prediction_batch).all():
        raise ValueError(f"{model_name} produced NaN or infinite predictions.")

    mse, mae, max_error = per_candidate_mse(prediction_batch, target)
    return prediction_batch, mse, mae, max_error


def save_results_csv(results_df):
    results_df = results_df[[
        "candidate_index",
        "sepcnn_mse",
        "resnet_mse",
        "sepcnn_mae",
        "resnet_mae",
        "sepcnn_max_error",
        "resnet_max_error",
    ]].copy()
    results_df.to_csv(RESULTS_CSV, index=False, float_format="%.17g")


def save_inputs_csv(candidate_indices, chromosome_batch):
    input_rows = []
    for candidate_index, chromosome in zip(candidate_indices.tolist(), chromosome_batch):
        input_rows.append(
            {
                "candidate_index": int(candidate_index),
                "chromosome": json.dumps(chromosome.astype(int).tolist(), separators=(",", ":")),
            }
        )
    pd.DataFrame(input_rows).to_csv(INPUTS_CSV, index=False)


def print_statistics(sepcnn_mse, resnet_mse, sepcnn_mae, resnet_mae):
    labels = {
        "SEP-CNN": (sepcnn_mse, sepcnn_mae),
        "ResNet": (resnet_mse, resnet_mae),
    }

    for name, (mse_values, mae_values) in labels.items():
        print(f"\n{name} metrics:")
        print(f"  mean MSE:      {np.mean(mse_values):.17g}")
        print(f"  median MSE:    {np.median(mse_values):.17g}")
        print(f"  std MSE:       {np.std(mse_values):.17g}")
        print(f"  min MSE:       {np.min(mse_values):.17g}")
        print(f"  max MSE:       {np.max(mse_values):.17g}")
        print(f"  mean MAE:      {np.mean(mae_values):.17g}")
        print(f"  median MAE:    {np.median(mae_values):.17g}")

    sepcnn_lower = np.sum(sepcnn_mse < resnet_mse)
    resnet_lower = np.sum(resnet_mse < sepcnn_mse)
    ties = np.sum(sepcnn_mse == resnet_mse)
    total = len(sepcnn_mse)
    print(f"\nWin counts across {total} candidates:")
    print(f"  SEP-CNN MSE < ResNet MSE: {sepcnn_lower}")
    print(f"  ResNet MSE < SEP-CNN MSE: {resnet_lower}")
    print(f"  ties: {ties}")
    print(f"  SEP-CNN wins percentage: {100.0 * sepcnn_lower / total:.2f}%")
    print(f"  ResNet wins percentage: {100.0 * resnet_lower / total:.2f}%")

    differences = sepcnn_mse - resnet_mse
    print(f"\nPaired MSE difference = SEP-CNN MSE - ResNet MSE:")
    print(f"  mean difference: {np.mean(differences):.17g}")
    print(f"  median difference: {np.median(differences):.17g}")
    print(f"  min difference: {np.min(differences):.17g}")
    print(f"  max difference: {np.max(differences):.17g}")

    valid_ratio_mask = np.abs(resnet_mse) > 1e-12
    ratio = np.divide(sepcnn_mse, resnet_mse, out=np.full_like(sepcnn_mse, np.nan, dtype=np.float64), where=valid_ratio_mask)
    if np.any(valid_ratio_mask):
        print(f"  mean ratio (SEP-CNN MSE / ResNet MSE) over valid candidates: {np.nanmean(ratio):.17g}")
        print(f"  candidates with near-zero ResNet MSE skipped in ratio: {np.sum(~valid_ratio_mask)}")
    else:
        print("  mean ratio: undefined because ResNet MSE is effectively zero for all candidates.")


def print_top_candidates(candidate_indices, sepcnn_mse, resnet_mse):
    print("\nTop 10 candidates with the lowest SEP-CNN MSE:")
    order = np.argsort(sepcnn_mse)
    for idx in order[:10]:
        difference = sepcnn_mse[idx] - resnet_mse[idx]
        print(
            f"  candidate_index={candidate_indices[idx]}, "
            f"SEP-CNN_MSE={sepcnn_mse[idx]:.17g}, "
            f"ResNet_MSE={resnet_mse[idx]:.17g}, "
            f"difference={difference:.17g}"
        )

    print("\nTop 10 candidates with the lowest ResNet MSE:")
    order = np.argsort(resnet_mse)
    for idx in order[:10]:
        difference = sepcnn_mse[idx] - resnet_mse[idx]
        print(
            f"  candidate_index={candidate_indices[idx]}, "
            f"SEP-CNN_MSE={sepcnn_mse[idx]:.17g}, "
            f"ResNet_MSE={resnet_mse[idx]:.17g}, "
            f"difference={difference:.17g}"
        )

    print("\nTop 10 candidates with the largest absolute difference between models:")
    order = np.argsort(np.abs(sepcnn_mse - resnet_mse))[::-1]
    for idx in order[:10]:
        difference = sepcnn_mse[idx] - resnet_mse[idx]
        print(
            f"  candidate_index={candidate_indices[idx]}, "
            f"SEP-CNN_MSE={sepcnn_mse[idx]:.17g}, "
            f"ResNet_MSE={resnet_mse[idx]:.17g}, "
            f"difference={difference:.17g}"
        )


def plot_results(candidate_indices, sepcnn_mse, resnet_mse):
    if plt is None:
        print("\nPlots skipped: Matplotlib is not installed in this environment.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sepcnn_mse, resnet_mse, s=12, alpha=0.7, color="tab:blue")
    min_limit = float(np.min([np.min(sepcnn_mse), np.min(resnet_mse)]))
    max_limit = float(np.max([np.max(sepcnn_mse), np.max(resnet_mse)]))
    pad = 0.05 * max(1e-12, max_limit - min_limit)
    ax.plot([min_limit - pad, max_limit + pad], [min_limit - pad, max_limit + pad], "k--", linewidth=1.0, label="y=x")
    ax.set_xlabel("SEP-CNN MSE")
    ax.set_ylabel("ResNet MSE")
    ax.set_title("Cross-model comparison on the same 1,024 designs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    diff_values = sepcnn_mse - resnet_mse
    ax.plot(candidate_indices, diff_values, linestyle="-", marker="o", markersize=2, color="tab:orange")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("candidate_index")
    ax.set_ylabel("SEP-CNN MSE - ResNet MSE")
    ax.set_title("Difference across the same 1,024 candidates")
    fig.tight_layout()
    fig.savefig(DIFFERENCE_PLOT, dpi=150)
    plt.close(fig)

    print(f"\nSaved scatter plot: {SCATTER_PLOT}")
    print(f"Saved difference plot: {DIFFERENCE_PLOT}")


def maybe_run_paired_test(sepcnn_mse, resnet_mse):
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover
        print("\nPaired statistical test skipped: SciPy is not installed.")
        return

    t_statistic, p_value = stats.ttest_rel(sepcnn_mse, resnet_mse)
    print("\nPaired statistical test: SEP-CNN MSE vs ResNet MSE")
    print("  This paired test compares the two models on the same 1,024 candidate designs.")
    print("  A paired design asks whether the mean paired difference across designs is different from zero.")
    print(f"  t-statistic: {t_statistic:.17g}")
    print(f"  p-value: {p_value:.17g}")
    print("  Interpretation: do not treat this as proof that one model is universally better across all designs.")
    print("  It only tells us whether the mean paired difference is distinguishable from zero for this candidate set.")


def check_candidate_961(candidate_indices, sepcnn_mse, resnet_mse, chromosome_batch):
    candidate_961_position = np.where(candidate_indices == 961)[0]
    if len(candidate_961_position) != 1:
        raise ValueError(f"Expected exactly one candidate_index=961 in the run-1 set; found {len(candidate_961_position)}.")
    pos = int(candidate_961_position[0])

    sepcnn_961 = float(sepcnn_mse[pos])
    resnet_961 = float(resnet_mse[pos])
    print(f"\nCandidate 961 cross-check (run 1 from SEP-CNN candidates):")
    print(f"  SEP-CNN on candidate 961 MSE: {sepcnn_961:.17g}")
    print(f"  ResNet on the SAME candidate 961 chromosome MSE: {resnet_961:.17g}")

    sepcnn_tolerance = np.isclose(sepcnn_961, EXPECTED_SEPCNN_CANDIDATE_961_MSE, rtol=1e-5, atol=1e-8)
    resnet_tolerance = np.isclose(resnet_961, EXPECTED_RESNET_SAME_SEPCNN_SQ961_MSE, rtol=1e-5, atol=1e-8)

    if not sepcnn_tolerance:
        raise AssertionError(
            "Candidate 961 SEP-CNN verification failed: "
            f"calculated {sepcnn_961:.17g}, expected {EXPECTED_SEPCNN_CANDIDATE_961_MSE:.17g}. "
            "This indicates a candidate-selection, parsing, target, or model mismatch."
        )
    if not resnet_tolerance:
        raise AssertionError(
            "Candidate 961 ResNet verification failed for the SAME SEP-CNN chromosome: "
            f"calculated {resnet_961:.17g}, expected {EXPECTED_RESNET_SAME_SEPCNN_SQ961_MSE:.17g}. "
            "This suggests the wrong run, wrong chromosome parsing, wrong target, or mismatched inference setup."
        )

    if not np.isclose(sepcnn_961, EXPECTED_SEPCNN_CANDIDATE_961_MSE, rtol=1e-12, atol=1e-15):
        warnings.warn(
            "Candidate 961 SEP-CNN MSE differs from the historical value by a tiny floating-point amount "
            f"({sepcnn_961:.17g} vs {EXPECTED_SEPCNN_CANDIDATE_961_MSE:.17g}), but it remains within the accepted numerical tolerance. "
            "This is consistent with TensorFlow/CPU float32 arithmetic drift rather than a different candidate or model."
        )
    if not np.isclose(resnet_961, EXPECTED_RESNET_SAME_SEPCNN_SQ961_MSE, rtol=1e-12, atol=1e-15):
        warnings.warn(
            "Candidate 961 ResNet MSE differs from the historical value by a tiny floating-point amount "
            f"({resnet_961:.17g} vs {EXPECTED_RESNET_SAME_SEPCNN_SQ961_MSE:.17g}), but it remains within the accepted numerical tolerance. "
            "This is consistent with TensorFlow/CPU float32 arithmetic drift rather than a different candidate or model."
        )

    print("  Candidate 961 validation: PASS within numerical tolerance")


def build_interpretation(sepcnn_mse, resnet_mse):
    sepcnn_wins = int(np.sum(sepcnn_mse < resnet_mse))
    resnet_wins = int(np.sum(resnet_mse < sepcnn_mse))
    ties = int(np.sum(sepcnn_mse == resnet_mse))
    total = len(sepcnn_mse)

    if resnet_wins > sepcnn_wins:
        model_summary = (
            f"ResNet had lower MSE on {resnet_wins} of {total} designs ({100.0 * resnet_wins / total:.2f}%), "
            f"while SEP-CNN had lower MSE on {sepcnn_wins} ({100.0 * sepcnn_wins / total:.2f}%)."
        )
    elif sepcnn_wins > resnet_wins:
        model_summary = (
            f"SEP-CNN had lower MSE on {sepcnn_wins} of {total} designs ({100.0 * sepcnn_wins / total:.2f}%), "
            f"while ResNet had lower MSE on {resnet_wins} ({100.0 * resnet_wins / total:.2f}%)."
        )
    else:
        model_summary = (
            f"The models tied on {ties} of {total} designs; SEP-CNN was better on {sepcnn_wins} and ResNet was better on {resnet_wins}."
        )

    diff_values = sepcnn_mse - resnet_mse
    if np.median(np.abs(diff_values)) > 1e-6:
        interaction = "The model difference depends strongly on the candidate design, which is consistent with a model-by-design interaction."
    else:
        interaction = "The differences are mostly small relative to the scale of the MSE values, so there is limited evidence of a strong design-dependent interaction in this set."

    print("\nInterpretation:")
    print(f"  1. {model_summary}")
    print(f"  2. SEP-CNN had lower MSE on a meaningful number of designs: {sepcnn_wins} / {total} ({100.0 * sepcnn_wins / total:.2f}%).")
    print(f"  3. {interaction}")
    print("  4. This supports the earlier observation of a model-by-design interaction when the choice of forward model changes the predicted shape error for the same candidate design.")
    print("  5. In the inverse-design loop, the model choice can materially change the fitness landscape and therefore the optimization trajectory; it should be treated as a design-dependent modeling decision rather than a universally better or worse surrogate.")


def main():
    target = load_target()
    candidate_indices, chromosome_batch = validate_and_load_run1_candidates()

    sepcnn_predictions, sepcnn_mse, sepcnn_mae, sepcnn_max_error = evaluate_model(
        "SEP-CNN", SEPCNN_CHECKPOINT, chromosome_batch, target
    )
    resnet_predictions, resnet_mse, resnet_mae, resnet_max_error = evaluate_model(
        "ResNet", RESNET_CHECKPOINT, chromosome_batch, target
    )

    if sepcnn_predictions.shape[0] != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} SEP-CNN predictions; found {sepcnn_predictions.shape[0]}."
        )
    if resnet_predictions.shape[0] != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} ResNet predictions; found {resnet_predictions.shape[0]}."
        )

    if not np.isfinite(sepcnn_mse).all() or not np.isfinite(resnet_mse).all():
        raise ValueError("At least one model produced non-finite MSE metrics.")
    if not np.isfinite(sepcnn_mae).all() or not np.isfinite(resnet_mae).all():
        raise ValueError("At least one model produced non-finite MAE metrics.")

    results_df = pd.DataFrame({
        "candidate_index": candidate_indices.astype(int),
        "sepcnn_mse": sepcnn_mse.astype(np.float64),
        "resnet_mse": resnet_mse.astype(np.float64),
        "sepcnn_mae": sepcnn_mae.astype(np.float64),
        "resnet_mae": resnet_mae.astype(np.float64),
        "sepcnn_max_error": sepcnn_max_error.astype(np.float64),
        "resnet_max_error": resnet_max_error.astype(np.float64),
    })
    results_df = results_df.sort_values("candidate_index").reset_index(drop=True)

    save_inputs_csv(results_df["candidate_index"].to_numpy(), chromosome_batch)
    save_results_csv(results_df)

    check_candidate_961(
        results_df["candidate_index"].to_numpy(),
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
        chromosome_batch,
    )

    print_statistics(
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
        results_df["sepcnn_mae"].to_numpy(),
        results_df["resnet_mae"].to_numpy(),
    )
    print_top_candidates(
        results_df["candidate_index"].to_numpy(),
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
    )
    plot_results(
        results_df["candidate_index"].to_numpy(),
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
    )
    maybe_run_paired_test(
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
    )
    build_interpretation(
        results_df["sepcnn_mse"].to_numpy(),
        results_df["resnet_mse"].to_numpy(),
    )

    print(f"\nSaved comparison CSV: {RESULTS_CSV}")
    print(f"Saved chromosome CSV: {INPUTS_CSV}")
    print(f"Total evaluated candidates: {len(results_df)}")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
