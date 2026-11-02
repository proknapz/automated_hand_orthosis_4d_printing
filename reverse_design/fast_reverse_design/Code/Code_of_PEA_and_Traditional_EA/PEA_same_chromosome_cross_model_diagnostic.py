"""Controlled 2x2 cross-model evaluation of the recorded generation-5 candidates.

This diagnostic reads both complete candidate chromosomes from the existing
PEA generation-5 CSVs and evaluates each chromosome with both unchanged,
trained forward models. It does not run PEA, alter preprocessing, or train or
modify either model/checkpoint.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

# Use the repository's existing TensorFlow 1.x graph API under TensorFlow 2.10.
tf.disable_eager_execution()

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_INDEX = 961
RUN_NUMBER = 1
TARGET_CSV = ROOT / "target_shape" / "target_test.csv"
SAVED_TARGET_CSV = ROOT / "target_output.csv"
INPUTS_CSV = ROOT / "same_chromosome_inputs.csv"
COMPARISON_CSV = ROOT / "same_chromosome_cross_model_comparison.csv"

CANDIDATE_FILES = {
    "SEP-CNN": ROOT / "sepcnn_PEA_generation5_candidates.csv",
    "ResNet": ROOT / "resnet_PEA_generation5_candidates.csv",
}
CHECKPOINTS = {
    "SEP-CNN": ROOT / "model_weights_SEP_CNN_fair" / "model.ckpt",
    "ResNet": ROOT / "model_weights_2D_ResNet_fair" / "model.ckpt",
}
PREDICTION_FILES = {
    ("SEP-CNN", "SEP-CNN"): ROOT / "sepcnn_chromosome_sepcnn_prediction.csv",
    ("SEP-CNN", "ResNet"): ROOT / "sepcnn_chromosome_resnet_prediction.csv",
    ("ResNet", "SEP-CNN"): ROOT / "resnet_chromosome_sepcnn_prediction.csv",
    ("ResNet", "ResNet"): ROOT / "resnet_chromosome_resnet_prediction.csv",
}
EXPECTED_CHROMOSOMES = {
    # These are verification values only. Inputs used below are loaded from the
    # candidate CSVs, not reconstructed from these literals.
    "SEP-CNN": np.asarray(
        [[1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
         [1, 1, 1, 0, 1, 0, 0, 0, 0, 1]],
        dtype=np.int32,
    ),
    "ResNet": np.asarray(
        [[1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]],
        dtype=np.int32,
    ),
}
EXPECTED_DIAGONAL_MSE = {
    "SEP-CNN": 0.0034106617562482612,
    "ResNet": 0.00011636296067893222,
}


def load_recorded_chromosome(source_name):
    """Read the exact 2x10 chromosome and PEA shape error from candidate CSV."""
    candidate_path = CANDIDATE_FILES[source_name]
    rows = pd.read_csv(candidate_path)
    selected = rows[
        (rows["run"] == RUN_NUMBER)
        & (rows["candidate_index"] == CANDIDATE_INDEX)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one run {} candidate {} in {}; found {}".format(
                RUN_NUMBER, CANDIDATE_INDEX, candidate_path, len(selected)
            )
        )

    row = selected.iloc[0]
    chromosome = np.asarray(json.loads(row["chromosome"]), dtype=np.int32)
    if chromosome.shape != (2, 10):
        raise ValueError(
            "{} candidate chromosome must have shape (2, 10), found {}".format(
                source_name, chromosome.shape
            )
        )
    if not np.isin(chromosome, [0, 1]).all():
        raise ValueError("{} chromosome contains values other than 0 and 1".format(source_name))
    if not np.array_equal(chromosome, EXPECTED_CHROMOSOMES[source_name]):
        raise ValueError(
            "{} CSV chromosome does not match the expected experiment artifact. "
            "Found {}".format(source_name, chromosome.tolist())
        )
    return chromosome, float(row["shape_error"])


def pea_shape_mse(prediction, target):
    """Match the original PEA fitness_function2 shape-error convention."""
    return float(
        np.mean(
            np.square(prediction[0, :] - target[0, :])
            + np.square(prediction[1, :] - target[1, :])
        )
    )


# The following SEP-CNN graph functions match the instrumented PEA model.
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
            name="network1_conv_{}".format(i),
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


# The following ResNet graph functions match the instrumented PEA model.
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
        name=name + "_conv1",
    )
    conv1 = tf.nn.leaky_relu(conv1)
    conv2 = resnet_conv2d_layer(
        conv1,
        filters=filters,
        kernel_size=(3, 3),
        name=name + "_conv2",
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
    # Original ResNet reduction over the width axis; resulting output remains
    # the model's full (batch, 2, 200) prediction as used by PEA.
    return tf.reduce_mean(output, axis=2)


def evaluate_model(model_name, chromosomes):
    """Restore one exact checkpoint and run each chromosome as (1,2,10)."""
    checkpoint = CHECKPOINTS[model_name]
    if not Path(str(checkpoint) + ".index").exists():
        raise FileNotFoundError("Checkpoint index not found: {}".format(checkpoint))

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
    predictions = {}
    with tf.Session(graph=graph, config=config) as session:
        session.run(initializer)
        saver.restore(session, str(checkpoint))
        for chromosome_source, chromosome in chromosomes.items():
            model_input = chromosome.astype(np.float32)[np.newaxis, :, :]
            if model_input.shape != (1, 2, 10):
                raise ValueError("Input must be (1, 2, 10), found {}".format(model_input.shape))
            # Fetch output_predict directly. No script-side reshape/reordering.
            prediction = session.run(output_predict, feed_dict={x: model_input})
            if prediction.shape != (1, 2, 200):
                raise ValueError(
                    "{} returned {}, expected (1, 2, 200)".format(
                        model_name, prediction.shape
                    )
                )
            predictions[chromosome_source] = prediction
    return predictions


def save_chromosome_inputs(chromosomes):
    columns = ["chromosome_source", "candidate_index", "channel"] + [
        "bit_{}".format(index) for index in range(10)
    ]
    with INPUTS_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for source_name, chromosome in chromosomes.items():
            for channel_index, channel in enumerate(chromosome):
                writer.writerow(
                    [source_name, CANDIDATE_INDEX, channel_index] + channel.astype(int).tolist()
                )


def verify_diagonal(results):
    """Abort before writing outputs if either baseline diagonal fails."""
    checks = [
        ("SEP-CNN", "SEP-CNN", "SEP-CNN"),
        ("ResNet", "ResNet", "ResNet"),
    ]
    for chromosome_source, model_name, expected_key in checks:
        prediction = results[(chromosome_source, model_name)][0]
        actual = pea_shape_mse(prediction, TARGET)
        expected = EXPECTED_DIAGONAL_MSE[expected_key]
        print(
            "Diagonal check {} chromosome -> {}: calculated {:.17g}; expected {:.17g}".format(
                chromosome_source, model_name, actual, expected
            )
        )
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-15):
            raise AssertionError(
                "Diagonal verification failed for {} chromosome -> {}. "
                "Calculated MSE {:.17g}, expected {:.17g}. No comparison or "
                "prediction files have been written.".format(
                    chromosome_source, model_name, actual, expected
                )
            )


def save_results(chromosomes, results):
    comparison_rows = []
    for chromosome_source in ("SEP-CNN", "ResNet"):
        for model_name in ("SEP-CNN", "ResNet"):
            raw_prediction = results[(chromosome_source, model_name)]
            prediction = raw_prediction[0]
            difference = prediction - TARGET
            mse = pea_shape_mse(prediction, TARGET)
            mae = float(np.mean(np.abs(difference)))
            max_absolute_error = float(np.max(np.abs(difference)))
            prediction_path = PREDICTION_FILES[(chromosome_source, model_name)]
            # Save the full 2x200 values; CSV is headerless, like target_output.csv.
            pd.DataFrame(prediction).to_csv(
                prediction_path,
                index=False,
                header=False,
                float_format="%.17g",
            )
            comparison_rows.append(
                [
                    chromosome_source,
                    CANDIDATE_INDEX,
                    model_name,
                    mse,
                    mae,
                    max_absolute_error,
                ]
            )

    pd.DataFrame(TARGET).to_csv(
        SAVED_TARGET_CSV,
        index=False,
        header=False,
        float_format="%.17g",
    )
    save_chromosome_inputs(chromosomes)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "chromosome_source",
                "candidate_index",
                "model",
                "mse",
                "mae",
                "max_absolute_error",
            ]
        )
        writer.writerows(comparison_rows)


def print_interpretation(results):
    mse_values = {
        (source, model): pea_shape_mse(results[(source, model)][0], TARGET)
        for source in ("SEP-CNN", "ResNet")
        for model in ("SEP-CNN", "ResNet")
    }

    print("\nSame-chromosome model effect (SEP-CNN MSE / ResNet MSE):")
    for source in ("SEP-CNN", "ResNet"):
        sep_error = mse_values[(source, "SEP-CNN")]
        resnet_error = mse_values[(source, "ResNet")]
        print(
            "  {} chromosome: SEP-CNN={:.17g}; ResNet={:.17g}; ratio={:.6g}x".format(
                source, sep_error, resnet_error, sep_error / resnet_error
            )
        )

    print("\nSame-model chromosome effect (SEP-CNN chromosome MSE / ResNet chromosome MSE):")
    for model_name in ("SEP-CNN", "ResNet"):
        sep_chromosome_error = mse_values[("SEP-CNN", model_name)]
        resnet_chromosome_error = mse_values[("ResNet", model_name)]
        print(
            "  {} model: SEP-CNN chromosome={:.17g}; ResNet chromosome={:.17g}; ratio={:.6g}x".format(
                model_name,
                sep_chromosome_error,
                resnet_chromosome_error,
                sep_chromosome_error / resnet_chromosome_error,
            )
        )


def main():
    global TARGET

    # Inputs are extracted directly from the recorded candidate CSV artifacts.
    chromosomes = {}
    recorded_shape_errors = {}
    for source_name in ("SEP-CNN", "ResNet"):
        chromosome, recorded_error = load_recorded_chromosome(source_name)
        chromosomes[source_name] = chromosome
        recorded_shape_errors[source_name] = recorded_error
        print(
            "Verified {} candidate {} chromosome from CSV: {} (recorded shape error {:.17g})".format(
                source_name,
                CANDIDATE_INDEX,
                chromosome.tolist(),
                recorded_error,
            )
        )

    target = pd.read_csv(TARGET_CSV, header=None).to_numpy()
    if target.shape != (2, 200):
        raise ValueError("Target must have shape (2, 200), found {}".format(target.shape))
    TARGET = target

    # Load each unchanged checkpoint once; evaluate both exact chromosomes.
    results = {}
    for model_name in ("SEP-CNN", "ResNet"):
        model_predictions = evaluate_model(model_name, chromosomes)
        for chromosome_source, prediction in model_predictions.items():
            results[(chromosome_source, model_name)] = prediction

    # Validate the two original diagonal results before writing any new output.
    verify_diagonal(results)

    # Verify recomputed diagonals against the exact PEA candidate CSV rows too.
    for source_name in ("SEP-CNN", "ResNet"):
        diagonal_mse = pea_shape_mse(results[(source_name, source_name)][0], TARGET)
        if not np.isclose(
            diagonal_mse,
            recorded_shape_errors[source_name],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise AssertionError(
                "{} diagonal MSE {:.17g} does not match candidate CSV shape error {:.17g}".format(
                    source_name, diagonal_mse, recorded_shape_errors[source_name]
                )
            )

    save_results(chromosomes, results)
    print_interpretation(results)
    print("\nSaved comparison: {}".format(COMPARISON_CSV))
    print("Saved chromosome inputs: {}".format(INPUTS_CSV))
    print("Saved target: {}".format(SAVED_TARGET_CSV))
    for path in PREDICTION_FILES.values():
        print("Saved prediction: {}".format(path))


if __name__ == "__main__":
    main()
