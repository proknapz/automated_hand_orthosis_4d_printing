import os
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)

BATCH_SIZE = 60
LEARNING_RATE = 1e-4
TRAINING_STEPS = 10000


def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1)
    inputs, outputs = [], []

    for i in range(1, len(data), 2):
        input_up = data.iloc[i - 1, :10].to_numpy().astype(float)
        input_down = data.iloc[i, :10].to_numpy().astype(float)
        output_up = data.iloc[i - 1, 10:].to_numpy().astype(float) / 1000
        output_down = data.iloc[i, 10:].to_numpy().astype(float) / 1000

        inputs.append(np.array([input_up, input_down]).reshape(2, 10))
        outputs.append(np.array([output_up, output_down]).reshape(2, 200))

    return np.array(inputs), np.array(outputs)


def Curve_unification(x):
    len_x = len(x)
    x1 = np.reshape(x, (len_x, 2, 200))

    x1_start = np.concatenate([
        np.full((len_x, 1, 200), x1[0, 0, 0]),
        np.full((len_x, 1, 200), x1[0, 1, 0])
    ], axis=1)

    x1 = x1 - x1_start
    new_x1 = np.reshape(x1, (len_x, 2, 200))
    new_x1[:, [0, 1], :] = new_x1[:, [1, 0], :]
    new_x1[:, 0, :] = -new_x1[:, 0, :]
    return new_x1


def networks_1(x):
    split_data = tf.split(x, num_or_size_splits=10, axis=2)
    outputs = []

    for i in range(10):
        branch = tf.layers.conv1d(
            inputs=split_data[i],
            filters=32,
            kernel_size=3,
            padding='same',
            activation=None,
            name='network1_conv_{}'.format(i)
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


def predicting(x):
    x1 = networks_1(x)
    x2 = networks_2(x1)
    x3 = networks_3(x2)

    return tf.layers.conv1d(
        inputs=x3,
        filters=200,
        kernel_size=3,
        padding='same',
        activation=None,
        name='output_conv'
    )


train_file = './train_data/train_data.csv'
test_file = './test_data/test_data.csv'

train_inputs, train_outputs = load_and_process_data(train_file)
test_inputs, test_outputs = load_and_process_data(test_file)

train_outputs = Curve_unification(train_outputs)
test_outputs = Curve_unification(test_outputs)

print('Train inputs:', train_inputs.shape)
print('Train outputs:', train_outputs.shape)
print('Test inputs:', test_inputs.shape)
print('Test outputs:', test_outputs.shape)

x = tf.placeholder(tf.float32, shape=[None, 2, 10])
y = tf.placeholder(tf.float32, shape=[None, 2, 200])

output_predict = predicting(x)

# MSE only for a controlled architecture comparison.
loss = tf.reduce_mean(tf.square(output_predict - y))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)

save_dir = './model_weights_SEP_CNN_fair'
os.makedirs(save_dir, exist_ok=True)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, TRAINING_STEPS + 1):
        indices = np.random.choice(len(train_inputs), BATCH_SIZE, replace=False)
        _, train_loss = sess.run(
            [train_op, loss],
            feed_dict={x: train_inputs[indices], y: train_outputs[indices]}
        )

        if step % 500 == 0:
            print('Step {}/{} - Training MSE: {}'.format(
                step, TRAINING_STEPS, train_loss
            ))

    test_mse = sess.run(
        loss, feed_dict={x: test_inputs, y: test_outputs}
    )

    print('\n========================================')
    print('SEP-CNN FAIR TRAINING RESULTS')
    print('========================================')
    print('Training steps:', TRAINING_STEPS)
    print('Test samples:', len(test_inputs))
    print('Test MSE:', test_mse)
    print('========================================')

    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
    print('\nModel saved to:', save_dir)
