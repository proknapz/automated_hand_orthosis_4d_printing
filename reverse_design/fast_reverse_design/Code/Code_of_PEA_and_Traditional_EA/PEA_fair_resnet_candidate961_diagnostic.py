"""Diagnostic-only replay of ResNet generation-5 candidate 961.

This script does not run or modify PEA selection, crossover, mutation, fitness,
population, generation, model architecture, checkpoints, or target data. It
loads the recorded candidate chromosome and evaluates it once with the same
ResNet graph/checkpoint used by PEA_fair_resnet_instrumented.py.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CSV = ROOT / "resnet_PEA_generation5_candidates.csv"
TARGET_CSV = ROOT / "target_shape" / "target_test.csv"
CHECKPOINT = ROOT / "model_weights_2D_ResNet_fair" / "model.ckpt"
PREDICTION_CSV = ROOT / "resnet_candidate_961_prediction.csv"
SAVED_TARGET_CSV = ROOT / "target_output.csv"
COMPARISON_CSV = ROOT / "candidate_961_model_comparison.csv"
CANDIDATE_INDEX = 961
RUN_NUMBER = 1
EXPECTED_PEA_MSE = 0.00011636296067893222


def conv2d_layer(inputs, filters, kernel_size, name):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        name=name,
    )


def residual_block(inputs, filters, name):
    shortcut = inputs
    conv1 = conv2d_layer(
        inputs,
        filters=filters,
        kernel_size=(3, 3),
        name=name + "_conv1",
    )
    conv1 = tf.nn.leaky_relu(conv1)
    conv2 = conv2d_layer(
        conv1,
        filters=filters,
        kernel_size=(3, 3),
        name=name + "_conv2",
    )
    output = tf.add(shortcut, conv2)
    return tf.nn.leaky_relu(output)


def predicting(inputs):
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
    # This is the original ResNet model's width reduction; its output remains
    # the complete (batch, 2, 200) prediction used by PEA.
    return tf.reduce_mean(output, axis=2)


def load_candidate():
    rows = pd.read_csv(CANDIDATE_CSV)
    selected = rows[
        (rows["run"] == RUN_NUMBER)
        & (rows["candidate_index"] == CANDIDATE_INDEX)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one run {} candidate {} row in {}; found {}".format(
                RUN_NUMBER, CANDIDATE_INDEX, CANDIDATE_CSV, len(selected)
            )
        )
    row = selected.iloc[0]
    chromosome = np.asarray(json.loads(row["chromosome"]), dtype=np.float32)
    if chromosome.shape != (2, 10):
        raise ValueError("Recorded chromosome must have shape (2, 10)")
    return chromosome[np.newaxis, :, :], float(row["shape_error"])


def pea_shape_mse(prediction, target):
    # Keep the original PEA fitness_function2 metric convention exactly:
    # mean of the sum of squared errors across the two output channels.
    return float(
        np.mean(
            np.square(prediction[0, :] - target[0, :])
            + np.square(prediction[1, :] - target[1, :])
        )
    )


def refresh_comparison():
    model_files = {
        "SEP-CNN": (
            ROOT / "sepcnn_candidate_961_prediction.csv",
            ROOT / "sepcnn_PEA_generation5_candidates.csv",
        ),
        "ResNet": (PREDICTION_CSV, CANDIDATE_CSV),
    }
    target = pd.read_csv(
        SAVED_TARGET_CSV, header=None, float_precision="round_trip"
    ).to_numpy(dtype=np.float64)
    comparison_rows = []
    for model_name, (prediction_path, candidate_path) in model_files.items():
        if not prediction_path.exists():
            continue
        prediction = pd.read_csv(
            prediction_path, header=None, float_precision="round_trip"
        ).to_numpy(dtype=np.float64)
        candidate_rows = pd.read_csv(candidate_path)
        candidate = candidate_rows[
            (candidate_rows["run"] == RUN_NUMBER)
            & (candidate_rows["candidate_index"] == CANDIDATE_INDEX)
        ]
        if len(candidate) != 1:
            raise ValueError("Expected one recorded candidate row for {}".format(model_name))
        candidate_shape_error = float(candidate.iloc[0]["shape_error"])
        difference = prediction - target
        mse = pea_shape_mse(prediction, target)
        if not np.isclose(mse, candidate_shape_error, rtol=1e-12, atol=1e-15):
            raise AssertionError(
                "{} prediction MSE {} does not match its recorded PEA shape error {}".format(
                    model_name, mse, candidate_shape_error
                )
            )
        comparison_rows.append(
            [
                model_name,
                CANDIDATE_INDEX,
                mse,
                float(np.mean(np.abs(difference))),
                float(np.max(np.abs(difference))),
            ]
        )
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["model", "candidate_index", "mse", "mae", "max_absolute_error"]
        )
        writer.writerows(comparison_rows)


def main():
    if not Path(str(CHECKPOINT) + ".index").exists():
        raise FileNotFoundError("ResNet checkpoint index not found: {}".format(CHECKPOINT))

    candidate, recorded_shape_error = load_candidate()
    target = pd.read_csv(TARGET_CSV, header=None).to_numpy(dtype=np.float64)
    if target.shape != (2, 200):
        raise ValueError("Target output must have shape (2, 200), found {}".format(target.shape))

    x = tf.placeholder(tf.float32, shape=[None, 2, 10])
    output_predict = predicting(x)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, str(CHECKPOINT))
        # Fetch the complete model output (batch, 2, 200), before any reshape.
        raw_prediction = session.run(output_predict, feed_dict={x: candidate})

    if raw_prediction.shape != (1, 2, 200):
        raise ValueError("Expected full ResNet output (1, 2, 200), found {}".format(raw_prediction.shape))
    prediction = np.asarray(raw_prediction[0], dtype=np.float64)
    mse = pea_shape_mse(prediction, target)
    absolute_error = np.abs(prediction - target)
    mae = float(np.mean(absolute_error))
    max_absolute_error = float(np.max(absolute_error))

    if not np.isclose(mse, recorded_shape_error, rtol=1e-12, atol=1e-15):
        raise AssertionError(
            "Recomputed ResNet MSE {} does not match recorded PEA shape error {}".format(
                mse, recorded_shape_error
            )
        )
    if not np.isclose(mse, EXPECTED_PEA_MSE, rtol=1e-12, atol=1e-15):
        raise AssertionError(
            "Recomputed ResNet MSE {} does not match expected value {}".format(
                mse, EXPECTED_PEA_MSE
            )
        )

    pd.DataFrame(prediction).to_csv(PREDICTION_CSV, index=False, header=False, float_format="%.17g")
    pd.DataFrame(target).to_csv(SAVED_TARGET_CSV, index=False, header=False, float_format="%.17g")
    refresh_comparison()

    print("Model: ResNet")
    print("Candidate index: {} (generation 5, run {})".format(CANDIDATE_INDEX, RUN_NUMBER))
    print("Full prediction shape: {}".format(raw_prediction.shape))
    print("MSE / PEA shape error: {:.17g}".format(mse))
    print("Recorded PEA shape error: {:.17g}".format(recorded_shape_error))
    print("Verification: PASS (matches recorded PEA shape error)")
    print("MAE: {:.17g}".format(mae))
    print("Maximum absolute error: {:.17g}".format(max_absolute_error))
    print("Saved prediction: {}".format(PREDICTION_CSV))
    print("Saved target: {}".format(SAVED_TARGET_CSV))
    print("Updated comparison: {}".format(COMPARISON_CSV))


if __name__ == "__main__":
    main()
