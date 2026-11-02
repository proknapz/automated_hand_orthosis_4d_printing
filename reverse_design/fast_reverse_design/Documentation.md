
Commands to run:
py -3.8 -m venv .venv

run this if ps1 does not work.
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

.\.venv\Scripts\Activate.ps1  

python -m pip install --upgrade pip
python -m pip install tensorflow==2.10.1
python -m pip install numpy matplotlib pandas

python -c "import numpy, pandas, matplotlib, tensorflow; print('All imports successful'); print('TensorFlow:', tensorflow.__version__)"


Test the actual script:
python ".\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py"

python ".\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py"
2026-09-28 12:10:50.001295: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-09-28 12:10:50.001433: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:171: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_0 = tf.layers.conv1d(inputs=x_cut[0], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:173: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_1 = tf.layers.conv1d(inputs=x_cut[1], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:175: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_2 = tf.layers.conv1d(inputs=x_cut[2], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:177: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_3 = tf.layers.conv1d(inputs=x_cut[3], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:179: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_4 = tf.layers.conv1d(inputs=x_cut[4], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:181: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_5 = tf.layers.conv1d(inputs=x_cut[5], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:183: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_6 = tf.layers.conv1d(inputs=x_cut[6], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:185: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_7 = tf.layers.conv1d(inputs=x_cut[7], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:187: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_8 = tf.layers.conv1d(inputs=x_cut[8], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:189: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv1_9 = tf.layers.conv1d(inputs=x_cut[9], filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:196: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_0 = tf.layers.conv1d(inputs=x0, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:198: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_1 = tf.layers.conv1d(inputs=conv2_0 + x1, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:200: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_2 = tf.layers.conv1d(inputs=conv2_1 + x2, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:202: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_3 = tf.layers.conv1d(inputs=conv2_2 + x3, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:204: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_4 = tf.layers.conv1d(inputs=conv2_3 + x4, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:206: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_5 = tf.layers.conv1d(inputs=conv2_4 + x5, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:208: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_6 = tf.layers.conv1d(inputs=conv2_5 + x6, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:210: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_7 = tf.layers.conv1d(inputs=conv2_6 + x7, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:212: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_8 = tf.layers.conv1d(inputs=conv2_7 + x8, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:214: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv2_9 = tf.layers.conv1d(inputs=conv2_8 + x9, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:220: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_0 = tf.layers.conv1d(inputs=x0, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:222: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_1 = tf.layers.conv1d(inputs=conv3_0 + x1, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:224: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_2 = tf.layers.conv1d(inputs=conv3_1 + x2, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:226: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_3 = tf.layers.conv1d(inputs=conv3_2 + x3, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:228: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_4 = tf.layers.conv1d(inputs=conv3_3 + x4, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:230: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_5 = tf.layers.conv1d(inputs=conv3_4 + x5, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:232: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_6 = tf.layers.conv1d(inputs=conv3_5 + x6, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:234: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_7 = tf.layers.conv1d(inputs=conv3_6 + x7, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:236: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_8 = tf.layers.conv1d(inputs=conv3_7 + x8, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:238: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  conv3_9 = tf.layers.conv1d(inputs=conv3_8 + x9, filters=filter, kernel_size=3, padding='same')
.\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py:255: UserWarning: `tf.layers.conv1d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv1D` instead.
  Net4_output = tf.layers.conv1d(inputs=Net3_output, filters=200, kernel_size=3, padding='same')
2026-09-28 12:10:54.046419: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-09-28 12:10:54.050000: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2026-09-28 12:10:54.050110: W tensorflow/stream_executor/cuda/cuda_driver.cc:263] failed call to cuInit: UNKNOWN ERROR (303)
2026-09-28 12:10:54.058158: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: DESKTOP-3J2FG8E
2026-09-28 12:10:54.058437: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: DESKTOP-3J2FG8E
2026-09-28 12:10:54.073729: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
$$$$$$$$$$$$$$$$-------frequency------$$$$$$$$$$$$$--------------------------------- 0
====================================generation===================================:0
====================================generation===================================:1
====================================generation===================================:2
====================================generation===================================:3
====================================generation===================================:4
--------generation:4-----number:41
Traceback (most recent call last):
  File ".\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py", line 334, in <module>
    with open(csv_file_path, 'a', newline='') as file:
FileNotFoundError: [Errno 2] No such file or directory: './PEA_total_gen/PEA_total_gen_mutate0.55.csv'


since you got this error: 
mkdir .\PEA_total_gen
mkdir .\PEA_design_time

python ".\Code_of_PEA_and_Traditional_EA\PEA_for_reverse_design.py" change iterations to 10

after check 
PEA_total_gen_mutate0.55.csv
PEA_time_mutate0.55.csv  

Successfully reproduced the Wang et al. fast reverse-design implementation using Python 3.8.10 and TensorFlow 2.10.1. The pretrained SEP-CNN model loaded successfully, and the PEA reverse-design algorithm completed a 10-generation test run without runtime errors. The generated PEA_total_gen_mutate0.55.csv and PEA_time_mutate0.55.csv files were used to verify the optimization and timing outputs.

We should inspect Wang's original network first, then design the ResNet around the same input/output contract.

The original Wang model expects:

2 × 10

which means 20 values at its model input, while your CSV has 10 design values.

The reason this works in Wang's code is that the PEA's 2 × 10 representation is part of the evolutionary representation, and the model's internal processing needs to be examined to determine exactly how those values correspond to the 10 design variables.


go to CNN_Train.py and get the shapes of each layer

        2 × 10
          │
          ▼
       CNN
          │
          ▼
        2 × 200


( batch, 2, 10 )
       ↓
Conv1D 320
       ↓
( batch, 2, 320 )
       ↓
Conv1D 320
       ↓
( batch, 2, 320 )
       ↓
Conv1D 320
       ↓
( batch, 2, 320 )
       ↓
Conv1D 200
       ↓
( batch, 2, 200 )



input is only 2×10.

write the 2D ResNet training version of cnn_train.py while preserving Wang's:

data loading
(2,10) input
(2,200) output
/1000 scaling
Curve_unification
train/test split
TensorFlow 2.10 / TF1-compatible style


project becomes
                 Wang Dataset
                      │
                      ▼
                10 variables
                      │
              pair 2 rows together
                      │
                      ▼
                  2 × 10
                      │
                      ▼
              ┌──────────────┐
              │  2D ResNet   │
              └──────┬───────┘
                     │
                     ▼
                  2 × 200
                     │
                     ▼
              predicted curve



after running PEA for reverse design for both models 
compare the PEA_design_time and 


changed model in PEA_for_reverse_design to load the resnet model and go through a new cnn



after running  python .\Code_DL_model_prediction\train_resnet_fair.py
output
Train inputs: (6000, 2, 10)
Train outputs: (6000, 2, 200)
Test inputs: (2000, 2, 10)
Test outputs: (2000, 2, 200)

Step 500/10000 - Training MSE: 0.0017630126094445586
Step 1000/10000 - Training MSE: 0.0005269658286124468
Step 1500/10000 - Training MSE: 0.000334757671225816
Step 2000/10000 - Training MSE: 0.00025376741541549563
Step 2500/10000 - Training MSE: 0.00020919223607052118
Step 3000/10000 - Training MSE: 0.00020660074369516224
Step 3500/10000 - Training MSE: 0.0001975956402020529
Step 4000/10000 - Training MSE: 0.00011772762081818655
Step 4500/10000 - Training MSE: 0.00013570222654379904
Step 5000/10000 - Training MSE: 7.684298179810867e-05
Step 5500/10000 - Training MSE: 0.00012205734674353153
Step 6000/10000 - Training MSE: 7.272041693795472e-05
Step 6500/10000 - Training MSE: 6.952779222046956e-05
Step 7000/10000 - Training MSE: 0.00011085884761996567
Step 7500/10000 - Training MSE: 9.395582310389727e-05
Step 8000/10000 - Training MSE: 8.376217010663822e-05
Step 8500/10000 - Training MSE: 5.1803777751047164e-05
Step 9000/10000 - Training MSE: 5.56357299501542e-05
Step 9500/10000 - Training MSE: 5.572723239311017e-05
Step 10000/10000 - Training MSE: 6.648951239185408e-05

========================================
2D RESNET FAIR TRAINING RESULTS
========================================
Training steps: 10000
Test samples: 2000
Test MSE: 5.5905177e-05
========================================

Model saved to: ./model_weights_2D_ResNet_fair

python .\Code_DL_model_prediction\train_sepcnn_fair.py

========================================
SEP-CNN FAIR TRAINING RESULTS
========================================
Training steps: 10000
Test samples: 2000
Test MSE: 0.0022869948
========================================

Step 500/10000 - Training MSE: 0.03569081053137779
Step 1000/10000 - Training MSE: 0.015467173419892788
Step 1500/10000 - Training MSE: 0.009645983576774597
Step 2000/10000 - Training MSE: 0.007364819757640362
Step 2500/10000 - Training MSE: 0.006622928660362959
Step 3000/10000 - Training MSE: 0.008834278210997581
Step 3500/10000 - Training MSE: 0.008116338402032852
Step 4000/10000 - Training MSE: 0.005350393708795309
Step 4500/10000 - Training MSE: 0.004071924369782209
Step 5000/10000 - Training MSE: 0.0037082440685480833
Step 5500/10000 - Training MSE: 0.004688166081905365
Step 6000/10000 - Training MSE: 0.002593549434095621
Step 6500/10000 - Training MSE: 0.003865926992148161
Step 7000/10000 - Training MSE: 0.0038141750264912844
Step 7500/10000 - Training MSE: 0.002726334147155285
Step 8000/10000 - Training MSE: 0.003789616050198674
Step 8500/10000 - Training MSE: 0.0030758948996663094
Step 9000/10000 - Training MSE: 0.002720039337873459
Step 9500/10000 - Training MSE: 0.002371744252741337
Step 10000/10000 - Training MSE: 0.0018985497299581766


Under a controlled training experiment using the same 10,000 optimizer steps, batch size, learning rate, preprocessing, and MSE objective, the 2D ResNet achieved a test MSE of 5.59 × 10⁻⁵, compared with 2.29 × 10⁻³ for the SEP-CNN. Thus, the ResNet achieved a substantially lower test MSE under the controlled training budget.


python .\Code_of_PEA_and_Traditional_EA\PEA_fair_resnet.py


python ./Code_of_PEA_and_Traditional_EA/PEA_fair_sepcnn_instrumented.py
python ./Code_of_PEA_and_Traditional_EA/PEA_fair_resnet_instrumented.py

The repeated generation-5 result is caused by the deterministic structure of the PEA's final search, not by duplicate candidates or a collapsed population.

You have experimentally verified:

1,024 candidates
1,024 unique candidates
exactly 1 unique prefix
same winning candidate index
same result across 10 runs

for both archectectures



from inspecting PEA_gen5

ResNet produces a much better minimum within the searched design space, while the average prediction error across the 1,024 candidates is similar to SEP-CNN.

The PEA itself is deterministic at generation 5


We performed a controlled 2×2 cross-model experiment to separate the effects of the candidate design and the learned forward model. The results showed a strong interaction between the two: neither model consistently produced lower prediction error across both designs. This indicates that both the selected 4D-printing configuration and the forward model influence the predicted shape error."

herefore, we do not interpret the lower original ResNet PEA error as evidence that ResNet is universally superior. Instead, the results motivate evaluating forward models across a common set of candidate designs

model-by-design interaction rather than a universally superior forward model.

hromosome represents:

a candidate material/structural configuration for the 4D-printed structur