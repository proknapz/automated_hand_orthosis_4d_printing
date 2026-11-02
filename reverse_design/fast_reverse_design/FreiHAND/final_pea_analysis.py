#!/usr/bin/env python3
"""Final PEA comparison script for SEP-CNN + PEA vs 2D ResNet + PEA.

This script reads the available PEA diagnostics and timing files, reconstructs the
actual final selected design from the existing algorithm, evaluates the final
predicted shape for each model, and compares the outcomes without changing the
PEA logic, mutation rules, fitness function, or search settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

ROOT = Path(__file__).resolve().parent
TARGET_CSV = ROOT / "target_output.csv"
SEPCNN_DIAG = ROOT / "sepcnn_PEA_generation_diagnostics.csv"
RESNET_DIAG = ROOT / "resnet_PEA_generation_diagnostics.csv"
SEPCNN_TIME = ROOT / "PEA_design_time" / "sepcnn_PEA_time_mutate0.55.csv"
RESNET_TIME = ROOT / "PEA_design_time" / "resnet_PEA_time_mutate0.55.csv"
SEPCNN_CHKPT = ROOT / "model_weights_SEP_CNN_fair" / "model.ckpt"
RESNET_CHKPT = ROOT / "model_weights_2D_ResNet_fair" / "model.ckpt"


def pea_shape_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.mean(
            np.square(prediction[0, :] - target[0, :])
            + np.square(prediction[1, :] - target[1, :])
        )
    )


def shape_metrics(prediction: np.ndarray, target: np.ndarray):
    diff = prediction - target
    mse = float(np.mean(np.square(diff[0, :]) + np.square(diff[1, :])))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    return mse, mae, max_abs


def load_target() -> np.ndarray:
    target = pd.read_csv(TARGET_CSV, header=None).to_numpy(dtype=np.float64)
    if target.shape != (2, 200):
        raise ValueError(f"Target must be shape (2, 200), got {target.shape}")
    return target


def load_generation_diagnostics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["generation"] == 5].copy()
    df["best_chromosome"] = df["best_chromosome"].apply(json.loads)
    return df.reset_index(drop=True)


def load_runtime_stats(csv_path: Path) -> pd.Series:
    vals = pd.read_csv(csv_path, header=None)[0].to_numpy(dtype=float)
    if vals.size > 10 and np.isclose(vals[-1], vals[:-1].mean(), rtol=0.2, atol=0.2):
        vals = vals[:-1]
    return pd.Series(vals)


def sepcnn_predicting(inputs):
    def networks_1(x):
        split_data = tf.split(x, num_or_size_splits=10, axis=2)
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

    def networks_2(x):
        outputs = [x[0]]
        for i in range(1, 10):
            branch = tf.add(x[i], outputs[i - 1])
            outputs.append(tf.nn.leaky_relu(branch))
        return outputs

    def networks_3(x):
        outputs = [x[0]]
        for i in range(1, 10):
            branch = tf.add(x[i], outputs[i - 1])
            outputs.append(tf.nn.leaky_relu(branch))
        return tf.concat(outputs, axis=2)

    x1 = networks_1(inputs)
    x2 = networks_2(x1)
    x3 = networks_3(x2)
    return tf.layers.conv1d(
        inputs=x3,
        filters=200,
        kernel_size=3,
        padding="same",
        activation=None,
        name="output_conv",
    )


def resnet_predicting(inputs):
    def conv2d_layer(x, filters, kernel_size, name):
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            name=name,
        )

    def residual_block(x, filters, name):
        shortcut = x
        conv1 = conv2d_layer(
            x,
            filters=filters,
            kernel_size=(3, 3),
            name=f"{name}_conv1",
        )
        conv1 = tf.nn.leaky_relu(conv1)
        conv2 = conv2d_layer(
            conv1,
            filters=filters,
            kernel_size=(3, 3),
            name=f"{name}_conv2",
        )
        output = tf.add(shortcut, conv2)
        output = tf.nn.leaky_relu(output)
        return output

    x_image = tf.expand_dims(inputs, axis=-1)
    conv0 = conv2d_layer(
        x_image,
        filters=64,
        kernel_size=(3, 3),
        name="initial_conv",
    )
    conv0 = tf.nn.leaky_relu(conv0)
    block1 = residual_block(conv0, 64, "residual_block1")
    block2 = residual_block(block1, 64, "residual_block2")
    block3 = residual_block(block2, 64, "residual_block3")
    output = conv2d_layer(
        block3,
        filters=200,
        kernel_size=(1, 3),
        name="output_conv",
    )
    output = tf.reduce_mean(output, axis=2)
    return output


def predict_with_model(model_name: str, chromosome: np.ndarray) -> np.ndarray:
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 2, 10])
        if model_name == "SEP-CNN":
            predictor = sepcnn_predicting(x)
            checkpoint = SEPCNN_CHKPT
        elif model_name == "ResNet":
            predictor = resnet_predicting(x)
            checkpoint = RESNET_CHKPT
        else:
            raise ValueError(f"Unknown model: {model_name}")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, str(checkpoint))
            prediction = sess.run(predictor, feed_dict={x: chromosome[None, :, :]})
    return prediction.reshape(2, 200)


def save_csv(path: Path, array: np.ndarray):
    pd.DataFrame(array).to_csv(path, index=False, header=False)


def plot_convergence(summary_df: pd.DataFrame, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in summary_df.groupby("model"):
        grouped = group.sort_values("generation")
        ax.plot(grouped["generation"], grouped["best_shape_error"], marker="o", label=model_name)
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best shape error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_runtime_comparison(runtime_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    runtime_df.boxplot(column="runtime", by="model", ax=ax)
    ax.set_title("PEA runtime comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Runtime (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_final_shape(target: np.ndarray, sep_pred: np.ndarray, res_pred: np.ndarray, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    x = np.arange(target.shape[1])
    axes[0].plot(x, target[0], label="Target x")
    axes[0].plot(x, sep_pred[0], label="SEP-CNN x")
    axes[0].plot(x, res_pred[0], label="ResNet x")
    axes[0].set_ylabel("x")
    axes[0].legend(loc="best")
    axes[0].set_title("Final predicted shapes vs. target")

    axes[1].plot(x, target[1], label="Target y")
    axes[1].plot(x, sep_pred[1], label="SEP-CNN y")
    axes[1].plot(x, res_pred[1], label="ResNet y")
    axes[1].set_ylabel("y")
    axes[1].legend(loc="best")

    axes[2].plot(x, np.abs(sep_pred - target).mean(axis=0), label="SEP-CNN abs. error")
    axes[2].plot(x, np.abs(res_pred - target).mean(axis=0), label="ResNet abs. error")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Mean abs. error")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    target = load_target()
    sep_diag = load_generation_diagnostics(SEPCNN_DIAG)
    res_diag = load_generation_diagnostics(RESNET_DIAG)

    def unique_design_count(df: pd.DataFrame) -> int:
        return len({tuple(tuple(row) for row in chromosome) for chromosome in df["best_chromosome"]})

    sep_final = sep_diag.iloc[0].to_dict()
    res_final = res_diag.iloc[0].to_dict()
    sep_chromosome = np.asarray(sep_final["best_chromosome"], dtype=np.int8)
    res_chromosome = np.asarray(res_final["best_chromosome"], dtype=np.int8)

    sep_pred = predict_with_model("SEP-CNN", sep_chromosome)
    res_pred = predict_with_model("ResNet", res_chromosome)

    save_csv(ROOT / "final_sepcnn_prediction.csv", sep_pred)
    save_csv(ROOT / "final_resnet_prediction.csv", res_pred)
    save_csv(ROOT / "final_target.csv", target)

    sep_mse, sep_mae, sep_max = shape_metrics(sep_pred, target)
    res_mse, res_mae, res_max = shape_metrics(res_pred, target)

    sep_run_summary = sep_diag[["run", "generation", "population_size", "best_shape_error", "best_candidate_index"]].copy()
    sep_run_summary["model"] = "SEP-CNN"
    res_run_summary = res_diag[["run", "generation", "population_size", "best_shape_error", "best_candidate_index"]].copy()
    res_run_summary["model"] = "ResNet"
    combined_generation = pd.concat([sep_run_summary, res_run_summary], ignore_index=True)
    combined_generation.to_csv(ROOT / "final_pea_comparison.csv", index=False)

    sep_times = load_runtime_stats(SEPCNN_TIME)
    res_times = load_runtime_stats(RESNET_TIME)
    runtime_df = pd.DataFrame({
        "model": ["SEP-CNN"] * len(sep_times),
        "runtime": sep_times.to_numpy(),
    })
    runtime_df = pd.concat([
        runtime_df,
        pd.DataFrame({
            "model": ["ResNet"] * len(res_times),
            "runtime": res_times.to_numpy(),
        })
    ], ignore_index=True)
    runtime_df.to_csv(ROOT / "final_model_runtime.csv", index=False)

    model_summary = pd.DataFrame({
        "model": ["SEP-CNN", "ResNet"],
        "mean_final_shape_error": [sep_diag["best_shape_error"].mean(), res_diag["best_shape_error"].mean()],
        "median_final_shape_error": [sep_diag["best_shape_error"].median(), res_diag["best_shape_error"].median()],
        "best_final_shape_error": [sep_diag["best_shape_error"].min(), res_diag["best_shape_error"].min()],
        "worst_final_shape_error": [sep_diag["best_shape_error"].max(), res_diag["best_shape_error"].max()],
        "mean_runtime": [sep_times.mean(), res_times.mean()],
        "runtime_std_dev": [sep_times.std(ddof=1) if len(sep_times) > 1 else 0.0, res_times.std(ddof=1) if len(res_times) > 1 else 0.0],
        "final_design": [sep_chromosome.tolist(), res_chromosome.tolist()],
        "num_unique_final_designs": [unique_design_count(sep_diag), unique_design_count(res_diag)],
    })
    model_summary.to_csv(ROOT / "final_model_pea_comparison.csv", index=False)

    all_gen_mean = (
        combined_generation.groupby(["model", "generation"], as_index=False)["best_shape_error"].mean()
    )
    plot_convergence(all_gen_mean.assign(model=all_gen_mean["model"]), "PEA Convergence: SEP-CNN vs ResNet", ROOT / "pea_convergence.png")
    plot_runtime_comparison(runtime_df, ROOT / "pea_runtime_comparison.png")
    plot_final_shape(target, sep_pred, res_pred, ROOT / "final_shape_comparison.png")

    # Compare final designs and predictions.
    final_design_diff = np.sum(sep_chromosome != res_chromosome)
    final_design_pct = 100.0 * final_design_diff / sep_chromosome.size
    pred_diff = sep_pred - res_pred
    pred_mse = float(np.mean(np.square(pred_diff[0, :]) + np.square(pred_diff[1, :])))
    pred_mae = float(np.mean(np.abs(pred_diff)))
    pred_max = float(np.max(np.abs(pred_diff)))

    result = {
        "sep_final_chromosome": sep_chromosome.tolist(),
        "res_final_chromosome": res_chromosome.tolist(),
        "hamming_distance": int(final_design_diff),
        "percent_genes_different": float(final_design_pct),
        "sep_final_mse": sep_mse,
        "res_final_mse": res_mse,
        "sep_final_mae": sep_mae,
        "res_final_mae": res_mae,
        "sep_final_max_abs_error": sep_max,
        "res_final_max_abs_error": res_max,
        "pred_shape_mse": pred_mse,
        "pred_shape_mae": pred_mae,
        "pred_shape_max_abs_diff": pred_max,
        "sep_runtime_mean": float(sep_times.mean()),
        "res_runtime_mean": float(res_times.mean()),
        "sep_runtime_std": float(sep_times.std(ddof=1) if len(sep_times) > 1 else 0.0),
        "res_runtime_std": float(res_times.std(ddof=1) if len(res_times) > 1 else 0.0),
    }
    with open(ROOT / "final_pea_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Final SEP-CNN chromosome:", sep_chromosome.tolist())
    print("Final ResNet chromosome:", res_chromosome.tolist())
    print("Hamming distance:", int(final_design_diff), "of 20 genes ({:.2f}%)".format(final_design_pct))
    print("SEP-CNN final MSE:", sep_mse)
    print("ResNet final MSE:", res_mse)
    print("SEP-CNN runtime mean:", float(sep_times.mean()))
    print("ResNet runtime mean:", float(res_times.mean()))
    print("Saved outputs to:", ROOT)


if __name__ == "__main__":
    main()
