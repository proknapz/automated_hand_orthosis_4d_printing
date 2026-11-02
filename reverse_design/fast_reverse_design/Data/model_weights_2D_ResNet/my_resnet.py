import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import time
import os

# ============================================================
# DATA LOADING
# ============================================================

def load_and_process_data(file_path):

    data = pd.read_csv(
        file_path,
        header=None,
        skiprows=1
    )

    inputs = []
    outputs = []

    # Wang pairs consecutive rows together
    for i in range(1, len(data), 2):

        input_up = data.iloc[i - 1, :10].to_numpy().astype(float)
        input_down = data.iloc[i, :10].to_numpy().astype(float)

        output_up = (
            data.iloc[i - 1, 10:].to_numpy().astype(float)
            / 1000.0
        )

        output_down = (
            data.iloc[i, 10:].to_numpy().astype(float)
            / 1000.0
        )

        input_data = np.array([
            input_up,
            input_down
        ])

        output_data = np.array([
            output_up,
            output_down
        ])

        inputs.append(input_data.reshape(2, 10))
        outputs.append(output_data.reshape(2, 200))

    return np.array(inputs), np.array(outputs)


# ============================================================
# CURVE UNIFICATION
# ============================================================

def Curve_unification(x):

    len_x = len(x)

    x1 = np.reshape(
        x,
        (len_x, 2, 200)
    )

    x1_start = np.concatenate(
        [
            np.full(
                (len_x, 1, 200),
                x1[0, 0, 0]
            ),

            np.full(
                (len_x, 1, 200),
                x1[0, 1, 0]
            )
        ],
        axis=1
    )

    x1 = x1 - x1_start

    new_x1 = np.reshape(
        x1,
        (len_x, 2, 200)
    )

    new_x1[:, [0, 1], :] = \
        new_x1[:, [1, 0], :]

    new_x1[:, 0, :] = \
        -new_x1[:, 0, :]

    return new_x1


# ============================================================
# LOAD DATA
# ============================================================

train_inputs, train_outputs = load_and_process_data(
    './train_data.csv'
)

test_inputs, test_outputs = load_and_process_data(
    './test_data.csv'
)

train_outputs = Curve_unification(
    train_outputs
)

test_outputs = Curve_unification(
    test_outputs
)

print("Train inputs:", train_inputs.shape)
print("Train outputs:", train_outputs.shape)

print("Test inputs:", test_inputs.shape)
print("Test outputs:", test_outputs.shape)


# ============================================================
# 2D RESNET
# ============================================================

def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        name):

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation=None,
        name=name
    )


def residual_block(
        inputs,
        filters,
        name):

    with tf.variable_scope(name):

        shortcut = inputs

        x = conv2d_layer(
            inputs,
            filters,
            3,
            'conv1'
        )

        x = tf.nn.leaky_relu(x)

        x = conv2d_layer(
            x,
            filters,
            3,
            'conv2'
        )

        x = x + shortcut

        x = tf.nn.leaky_relu(x)

        return x


def resnet_model(x):

    with tf.variable_scope(
            "resnet",
            reuse=tf.AUTO_REUSE):

        # ----------------------------------------------------
        # Input:
        #
        # (batch, 2, 10)
        #
        # Add channel dimension:
        #
        # (batch, 2, 10, 1)
        # ----------------------------------------------------

        x = tf.expand_dims(
            x,
            axis=-1
        )

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

        x = conv2d_layer(
            x,
            filters=64,
            kernel_size=3,
            name='input_conv'
        )

        x = tf.nn.leaky_relu(x)

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        x = residual_block(
            x,
            64,
            'block1'
        )

        x = residual_block(
            x,
            64,
            'block2'
        )

        x = residual_block(
            x,
            64,
            'block3'
        )

        # ----------------------------------------------------
        # Produce 200 output channels
        #
        # Current shape:
        #
        # (batch, 2, 10, 64)
        #
        # Convert to:
        #
        # (batch, 2, 10, 200)
        # ----------------------------------------------------

        x = conv2d_layer(
            x,
            filters=200,
            kernel_size=(1, 3),
            name='output_conv'
        )

        # ----------------------------------------------------
        # We need 200 outputs for each of the 2 rows.
        #
        # Collapse the width dimension.
        # ----------------------------------------------------

        x = tf.reduce_mean(
            x,
            axis=2
        )

        # Shape:
        #
        # (batch, 2, 200)

        return x


# ============================================================
# MODEL
# ============================================================

x = tf.placeholder(
    tf.float32,
    shape=[None, 2, 10],
    name='input'
)

y = tf.placeholder(
    tf.float32,
    shape=[None, 2, 200],
    name='target'
)

output_predict = resnet_model(x)

print("Model output tensor:", output_predict)


# ============================================================
# LOSS
# ============================================================

MSE_loss = tf.reduce_mean(
    tf.square(
        output_predict - y
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

learning_rate = 1e-4

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate
)

train_op = optimizer.minimize(
    MSE_loss
)


# ============================================================
# TRAINING
# ============================================================

batch_size = 60

number_iterations = 100

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

saver = tf.train.Saver()


with tf.Session(config=config) as sess:

    sess.run(
        tf.global_variables_initializer()
    )

    for iteration in range(number_iterations):

        indices = np.random.choice(
            train_inputs.shape[0],
            batch_size,
            replace=False
        )

        batch_inputs = \
            train_inputs[indices]

        batch_outputs = \
            train_outputs[indices]

        _, loss = sess.run(
            [
                train_op,
                MSE_loss
            ],
            feed_dict={
                x: batch_inputs,
                y: batch_outputs
            }
        )

        if iteration % 10 == 0:

            print(
                "Iteration:",
                iteration,
                "Training MSE:",
                loss
            )

    # ========================================================
    # TEST
    # ========================================================

    test_loss = sess.run(
        MSE_loss,
        feed_dict={
            x: test_inputs,
            y: test_outputs
        }
    )

    print()
    print("==============================")
    print("Test MSE:", test_loss)
    print("==============================")

    # ========================================================
    # SAVE MODEL
    # ========================================================

    saver.save(
        sess,
        './model_weights_2D_ResNet/model.ckpt'
    )

    print(
        "2D ResNet model saved."
    )