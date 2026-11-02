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


def conv2d_layer(x, filters, kernel_size, name):
    return tf.layers.conv2d(
        inputs=x, filters=filters, kernel_size=kernel_size,
        padding='same', activation=None, name=name
    )


def residual_block(x, filters, name):
    shortcut = x
    conv1 = tf.nn.leaky_relu(conv2d_layer(
        x, filters, (3, 3), name + '_conv1'
    ))
    conv2 = conv2d_layer(conv1, filters, (3, 3), name + '_conv2')
    return tf.nn.leaky_relu(tf.add(shortcut, conv2))


def resnet_model(x):
    x_image = tf.expand_dims(x, axis=-1)

    conv0 = tf.nn.leaky_relu(
        conv2d_layer(x_image, 64, (3, 3), 'initial_conv')
    )

    block1 = residual_block(conv0, 64, 'residual_block1')
    block2 = residual_block(block1, 64, 'residual_block2')
    block3 = residual_block(block2, 64, 'residual_block3')

    output = conv2d_layer(block3, 200, (1, 3), 'output_conv')
    return tf.reduce_mean(output, axis=2)


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

output_predict = resnet_model(x)
loss = tf.reduce_mean(tf.square(output_predict - y))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)

save_dir = './model_weights_2D_ResNet_fair'
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
    print('2D RESNET FAIR TRAINING RESULTS')
    print('========================================')
    print('Training steps:', TRAINING_STEPS)
    print('Test samples:', len(test_inputs))
    print('Test MSE:', test_mse)
    print('========================================')

    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
    print('\nModel saved to:', save_dir)
