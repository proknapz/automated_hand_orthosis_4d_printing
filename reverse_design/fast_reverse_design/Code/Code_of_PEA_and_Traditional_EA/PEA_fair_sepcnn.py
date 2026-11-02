import time

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import json
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import os
import csv
import itertools

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# np.set_printoptions(threshold=np.inf)


def initialization_population(population_size):
    population = np.random.randint(2, size=(population_size, 1, 2, 10))
    return population


def Curve_unification(x):
    x1 = x
    x1_endpoint_dis = np.sqrt(
        np.square(x1[0, -1]) + np.square(x1[1, -1])
    )

    vector_one = x1[:, -1]
    vector_two = np.array([x1_endpoint_dis, 0.])

    cos_angle = (
        (vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1]) /
        (
            np.sqrt(np.square(vector_one[0]) + np.square(vector_one[1])) *
            np.sqrt(np.square(vector_two[0]) + np.square(vector_two[1]))
        )
    )

    angle = np.arccos(cos_angle)

    angle_matrix_ni = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    angle_matrix_shun = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])

    if x1[1, -1] < 0:
        new_x1 = np.dot(angle_matrix_ni, x1)
    else:
        new_x1 = np.dot(angle_matrix_shun, x1)

    return new_x1


def fitness_function1(x, y, gen):
    a = gen * 20
    b = (gen + 1) * 20 - 1

    fit1 = -np.mean(
        np.square(x[0, a:b] - y[0, a:b]) +
        np.square(x[1, a:b] - y[1, a:b])
    )

    return fit1


def fitness_function2(x, y):
    fit2 = -np.mean(
        np.square(x[0, :] - y[0, :]) +
        np.square(x[1, :] - y[1, :])
    )

    return fit2


GENERATION_RESULTS_CSV = './sepcnn_PEA_generation_results.csv'


def record_generation_result(generation, best_fitness):

    with open(GENERATION_RESULTS_CSV, 'a', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            generation,
            best_fitness,
            -best_fitness
        ])


def cross_function(x, number, gen):

    length_x = len(x)

    x1 = x[0:int(number / 20)]
    x2 = x[int(number / 20):int(2 * number / 20)]
    x3 = x[int(2 * number / 20):int(3 * number / 20)]
    x4 = x[int(3 * number / 20):int(4 * number / 20)]
    x5 = x[int(4 * number / 20):]

    if gen > 3:
        gen = 3

    x1_cross = []

    for _ in range(int(3 * number / 40)):

        x1_id1, x1_id2 = np.random.choice(
            int(number / 20), 2, replace=False
        )

        x1_aa = x1[x1_id1].copy()
        x1_bb = x1[x1_id2].copy()

        x1_aa[:, 1, gen + 1:], x1_bb[:, 1, gen + 1:] = (
            x1_bb[:, 1, gen + 1:].copy(),
            x1_aa[:, 1, gen + 1:].copy()
        )

        x1_cross.append(x1_aa)
        x1_cross.append(x1_bb)

    x1_cross = np.array(x1_cross)
    new_x1 = np.concatenate((x1, x1_cross), axis=0)

    x2_cross = []

    for _ in range(int(number / 10)):

        x2_id1, x2_id2 = np.random.choice(
            int(number / 20), 2, replace=False
        )

        x2_aa = x2[x2_id1].copy()
        x2_bb = x2[x2_id2].copy()

        x2_aa[:, :, gen + 2:gen + 4], x2_bb[:, :, gen + 2:gen + 4] = (
            x2_bb[:, :, -3:-1].copy(),
            x2_aa[:, :, -3:-1].copy()
        )

        x2_cross.append(x2_aa)
        x2_cross.append(x2_bb)

    x2_cross = np.array(x2_cross)
    new_x2 = x2_cross

    x3_cross = []

    for _ in range(int(number / 10)):

        x3_id1, x3_id2 = np.random.choice(
            int(number / 20), 2, replace=False
        )

        x3_aa = x3[x3_id1].copy()
        x3_bb = x3[x3_id2].copy()

        x3_aa[:, :, gen + 2:gen + 4], x3_bb[:, :, gen + 2:gen + 4] = (
            x3_bb[:, :, -3:-1].copy(),
            x3_aa[:, :, -3:-1].copy()
        )

        x3_cross.append(x3_aa)
        x3_cross.append(x3_bb)

    x3_cross = np.array(x3_cross)
    new_x3 = x3_cross

    x4_cross = []

    for _ in range(int(number / 10)):

        x4_id1, x4_id2 = np.random.choice(
            int(number / 20), 2, replace=False
        )

        x4_aa = x4[x4_id1].copy()
        x4_bb = x4[x4_id2].copy()

        x4_aa[:, :, gen + 2:gen + 4], x4_bb[:, :, gen + 2:gen + 4] = (
            x4_bb[:, :, -3:-1].copy(),
            x4_aa[:, :, -3:-1].copy()
        )

        x4_cross.append(x4_aa)
        x4_cross.append(x4_bb)

    x4_cross = np.array(x4_cross)
    new_x4 = x4_cross

    x5_cross = []

    for _ in range(int(number / 10)):

        x5_id1, x5_id2 = np.random.choice(
            int(length_x - number / 5), 2, replace=False
        )

        x5_aa = x5[x5_id1].copy()
        x5_bb = x5[x5_id2].copy()

        x5_aa[:, :, gen + 2:gen + 4], x5_bb[:, :, gen + 2:gen + 4] = (
            x5_bb[:, :, -3:-1].copy(),
            x5_aa[:, :, -3:-1].copy()
        )

        x5_cross.append(x5_aa)
        x5_cross.append(x5_bb)

    x5_cross = np.array(x5_cross)
    new_x5 = x5_cross

    new_populations = np.concatenate(
        (new_x1, new_x2, new_x3, new_x4, new_x5),
        axis=0
    )

    return new_populations


def mutate_function(x, gen):

    if gen > 3:
        gen = 3

    sub_arrays = x[:, :, :, gen + 1]

    sub_array_types = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    counts = {
        str(arr_type): np.sum(
            np.all(sub_arrays == arr_type, axis=2)
        )
        for arr_type in sub_array_types
    }

    while any(count < 430 for count in counts.values()):

        min_count_type = min(counts, key=counts.get)
        max_count_type = max(counts, key=counts.get)

        num_needed = 430 - counts[min_count_type]

        indices = np.where(
            np.all(
                sub_arrays == np.array(eval(max_count_type)),
                axis=2
            )
        )[0]

        selected_indices = np.sort(indices)[-num_needed:]

        for idx in selected_indices:
            x[idx, :, :, gen + 1] = np.array(
                eval(min_count_type)
            )

        counts = {
            str(arr_type): np.sum(
                np.all(sub_arrays == arr_type, axis=2)
            )
            for arr_type in sub_array_types
        }

        mutation_flag = np.random.rand(2000) < 0.55

        if gen <= 3:

            for idx in np.where(mutation_flag)[0]:

                row, col = np.random.randint(
                    0,
                    2
                ), np.random.randint(
                    gen + 2,
                    10
                )

                x[idx, 0, row, col] = (
                    1 - x[idx, 0, row, col]
                )

    return x


def all_possibilities(x, gen):

    combinations = list(
        itertools.product([0, 1], repeat=10)
    )

    array_2_10 = np.array(
        combinations
    ).reshape(-1, 2, 5)

    array_2_10 = np.reshape(
        array_2_10,
        (1024, 1, 2, 5)
    )

    x_5 = x[0, :, :, :gen + 1]

    x_5 = np.tile(
        x_5,
        (1024, 1, 1, 1)
    )

    x_all = np.concatenate(
        (x_5, array_2_10),
        axis=3
    )

    return x_all


# ============================================================
# SEP-CNN NETWORK
# ============================================================
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
# ============================================================
# MODEL SETUP
# ============================================================

x = tf.placeholder(
    tf.float32,
    shape=[None, 2, 10]
)

y = tf.placeholder(
    tf.float32,
    shape=[None, 2, 200]
)

output_predict = predicting(x)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

load_pretrain_model = True

populations_size = 2000


with open(GENERATION_RESULTS_CSV, 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([
        'generation',
        'best_fitness',
        'shape_error'
    ])


# ============================================================
# PEA TESTING
# ============================================================

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    saver.restore(
        sess,
        './model_weights_SEP_CNN_fair/model.ckpt'
    )

    print("FAIR SEP-CNN CHECKPOINT LOADED")
    print("Running SEP-CNN PEA...")

    target_output_data = pd.read_csv(
        './target_shape/target_test.csv',
        header=None
    )

    target_output_data = target_output_data.to_numpy()

    times_per_iteration = []

    # PATCH:
    # This makes sure we only print the prediction confirmation once.
    prediction_tested = False

    for ii in range(10):

        start_time = time.time()

        print(
            '$$$$$$$$$$$$$$$$-------frequency------$$$$$$$$$$$$$---------------------------------',
            str(ii)
        )

        skip = False

        total_input = initialization_population(
            population_size=populations_size
        )

        chromosomes = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        for gen in range(6):

            print(
                '====================================generation===================================:'
                + str(gen)
            )

            scores_dict = {
                str(array): []
                for array in chromosomes
            }

            fit2_scores_list = []

            numbers = len(total_input)

            predicting_total_data = []

            for i in range(numbers):

                input_data = total_input[i, :, :, :]

                input_data_gen = np.reshape(
                    input_data[:, :, gen],
                    (2)
                )

                # ====================================================
                # PATCH:
                # Confirm SEP-CNN predicting() is actually running.
                # This prints ONLY ONCE.
                # ====================================================

                if not prediction_tested:

                    print(
                        "SEP-CNN predicting() is working"
                    )

                    prediction_tested = True

                # ====================================================
                # SEP-CNN FORWARD PREDICTION
                # ====================================================

                predicting_data = sess.run(
                    output_predict,
                    feed_dict={
                        x: input_data
                    }
                )

                predicting_data = np.reshape(
                    predicting_data,
                    (2, 200)
                )

                predicting_total_data.append(
                    predicting_data
                )

                # ====================================================
                # FITNESS
                # ====================================================

                fit_score = fitness_function1(
                    predicting_data,
                    target_output_data,
                    gen
                )

                fit2_scores = fitness_function2(
                    predicting_data,
                    target_output_data
                )

                fit2_scores_list.append(
                    fit2_scores
                )

                for chromosome in chromosomes:

                    if np.array_equal(
                        input_data_gen,
                        chromosome
                    ):

                        scores_dict[
                            str(chromosome)
                        ].append(
                            fit_score
                        )

                # ====================================================
                # SUCCESS CONDITION
                # ====================================================

                if fit2_scores >= -(1e-6):

                    highest_fit2_score = np.max(
                        fit2_scores_list
                    )

                    record_generation_result(
                        gen,
                        highest_fit2_score
                    )

                    print(
                        '--------generation:' +
                        str(gen) +
                        '-----' +
                        'number:' +
                        str(i)
                    )

                    csv_file_path = (
                        './PEA_total_gen/'
                        'sepcnn_PEA_total_gen_mutate0.55.csv'
                    )

                    data_to_write = [
                        gen,
                        i
                    ]

                    with open(
                        csv_file_path,
                        'a',
                        newline=''
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            data_to_write
                        )

                    print(
                        '=====input_data=====',
                        input_data
                    )

                    skip = True

                    break

            # ========================================================
            # STOP CURRENT RUN IF TARGET WAS FOUND
            # ========================================================

            if skip:
                break

            # ========================================================
            # AVERAGE SCORES
            # ========================================================

            average_scores = {
                array:
                    np.mean(scores)
                    if scores
                    else 0
                for array, scores in scores_dict.items()
            }

            fit2_total_scores = np.array([
                fit2_scores_list
            ])

            fit2_average_scores = np.mean(
                fit2_total_scores
            )

            fit2_total_scores = np.reshape(
                fit2_total_scores,
                (numbers, 1)
            )

            highest_average_score = max(
                average_scores.values()
            )

            highest_fit2_score = np.max(
                fit2_total_scores
            )

            record_generation_result(
                gen,
                highest_fit2_score
            )

            highest_values_index = np.argmax(
                fit2_total_scores
            )

            optimal_individual = total_input[
                highest_values_index
            ]

            predicting_total_data = np.array(
                predicting_total_data
            )

            optimal_prediction = predicting_total_data[
                highest_values_index
            ]

            # ========================================================
            # FIND BEST CHROMOSOME
            # ========================================================

            highest_scoring_arrays = [
                array
                for array, score
                in average_scores.items()
                if score == highest_average_score
            ]

            array_elements = (
                highest_scoring_arrays[0]
                .strip('[]')
                .split()
            )

            highest_array = np.array([
                int(element)
                for element in array_elements
            ])

            # ========================================================
            # FILTER POPULATION
            # ========================================================

            mask = (
                total_input[:, :, :, gen]
                == highest_array
            ).all(axis=2)

            filtered_array = total_input[
                mask
            ]

            fit2_filtered_array = fit2_total_scores[
                mask
            ]

            sorted_indices = np.argsort(
                fit2_filtered_array
            )[::-1]

            excellent_populations = (
                filtered_array[
                    sorted_indices
                ]
            )

            length = len(
                excellent_populations
            )

            excellent_populations = np.reshape(
                excellent_populations,
                (
                    length,
                    1,
                    2,
                    10
                )
            )

            # ========================================================
            # CROSSOVER + MUTATION
            # ========================================================

            if gen < 4:

                Crossed_populations = cross_function(
                    excellent_populations,
                    populations_size,
                    gen
                )

                Mutated_populations = mutate_function(
                    Crossed_populations,
                    gen
                )

                total_input = Mutated_populations

            # ========================================================
            # ALL POSSIBILITIES
            # ========================================================

            if gen == 4:

                all_populations = all_possibilities(
                    excellent_populations,
                    gen
                )

                total_input = all_populations

        # ============================================================
        # END OF ONE PEA RUN
        # ============================================================

        end_time = time.time()

        interation_time = (
            end_time - start_time
        )

        times_per_iteration.append(
            interation_time
        )

        if skip:
            continue

    # ================================================================
    # SAVE TIMING RESULTS
    # ================================================================

    times_per_iteration = np.array(
        times_per_iteration
    )

    average_time = np.mean(
        times_per_iteration
    )

    total_time = np.append(
        times_per_iteration,
        [average_time]
    )

    df = pd.DataFrame(
        data=total_time
    )

    df.to_csv(
        './PEA_design_time/sepcnn_PEA_time_mutate0.55.csv',
        index=False,
        header=False
    )

    print()
    print("==========================================")
    print("FAIR SEP-CNN PEA COMPLETE")
    print("==========================================")
    print(
        "Average PEA time:",
        average_time,
        "seconds"
    )