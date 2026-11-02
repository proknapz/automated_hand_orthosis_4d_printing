# FreiHAND spatial representation (Person 2)

This folder contains separate Person 1 ingestion/normalization and Person 2 spatial-encoder work; neither modifies the Wang SEP-CNN/ResNet/PEA baseline. The local FreiHAND training data has now been inspected on 50 evenly spaced samples: each inspected sample had 778 vertices and 21 keypoints, and all 50 passed normalization round-trip and prototype voxel-encoding checks. 

**NEW: MANO topology integration** now enables complete surface representation with 1538 triangular faces for orthosis design workflows.

The measured report is `results/real_sample_inspection.md`; this is geometry-pipeline validation, not clinical validation.

## Person 1 geometry ingestion

The FreiHAND-specific annotation reader follows the annotation layout and sample/version mapping in the [official reference repository](https://github.com/lmb-freiburg/freihand). Training records include aligned keypoints, vertices, MANO parameters, camera intrinsics, and a reference-bone scale. The actual files inspected contain 778 vertices and 21 keypoints per sample. Triangle faces can now be loaded from MANO models for complete surface topology.

### Download and unpack the real dataset

The [official FreiHAND dataset page](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) lists **FreiHAND Dataset v2 (3.7 GB)**. Download it from that page, or use the official archive URL below. Save data outside this Git repository; it is large and should not be committed.

The dataset is provided for research use only, without warranty; **commercial use is prohibited**, and research use must cite the FreiHAND paper. Review the terms on the official page before downloading or using the data. Ensure the destination drive has enough free space for the archive and extracted files.

Run the Python downloader from the repository root. It uses only the standard library (including ZIP64 support), downloads the training archive, safely extracts it, and prints the detected dataset root:

```powershell
.\.venv\Scripts\python.exe FreiHAND\download_dataset.py
```

By default, the archive and extracted data go to `%USERPROFILE%\Datasets\FreiHAND`, outside the repository. To use a different folder, pass `--data-dir`. The downloader reuses an existing archive and skips download/extraction when the training annotation files are already found.

```powershell
.\.venv\Scripts\python.exe FreiHAND\download_dataset.py --data-dir "D:\Datasets\FreiHAND"
```

The official page also offers a separate **FreiHAND v2 evaluation set with annotations (724 MB)**. It is optional and does not replace the training annotations. Download and extract it alongside the training set with:

```powershell
.\.venv\Scripts\python.exe FreiHAND\download_dataset.py --include-eval
```

The script prints the training dataset root. Use that directory for real-data visualization and validation. The training archive includes sampled vertex coordinates, but does not include triangle faces. External, version-compatible MANO assets are needed if face connectivity is required.

```python
from FreiHAND.data_loader import FreiHANDDataset
from pathlib import Path

# Without MANO topology (original behavior)
dataset = FreiHANDDataset(
    dataset_root=r"path\to\the\folder\containing\training_K.json",
    split="training",
    image_version="gs",
    units="m",
)
raw_geometry = dataset.load_sample(0)
# raw_geometry.faces is None

# With MANO topology (NEW - requires MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/)
dataset_with_mano = FreiHANDDataset(
    dataset_root=r"path\to\the\folder\containing\training_K.json",
    split="training",
    image_version="gs",
    units="m",
    mano_pkl_path=Path(r"path\to\MANO_RIGHT.pkl"),  # NEW parameter
)
complete_geometry = dataset_with_mano.load_sample(0)
# complete_geometry.vertices: (778, 3)
# complete_geometry.keypoints: (21, 3)
# complete_geometry.faces: (1538, 3) ← NEW!
```

For the inspected training version, `raw_geometry.vertices` is `(778, 3)`, `raw_geometry.keypoints` is `(21, 3)`, and `raw_geometry.faces` is `None` (or loaded from MANO if available). Metadata includes the `(3, 3)` camera intrinsic matrix, MANO parameter row, and provided reference-bone scale in meters. The official evaluator computes error in stored XYZ coordinates and multiplies by 100 to report centimeters, establishing meters as the source unit. Locally, `training_scale` exactly matched `norm(keypoints[9] - keypoints[10])` across all 32,560 training records.

For generic geometry or MANO topology, `load_geometry_npz` reads a non-pickled archive containing optional `vertices`, `keypoints`, and `faces` arrays:

```python
from FreiHAND.data_loader import load_geometry_npz
from FreiHAND.normalize_hand import normalize_hand, denormalize_hand
from pathlib import Path

# Load NPZ with optional MANO topology
raw_geometry = load_geometry_npz(
    r"path\to\sample.npz", 
    sample_id="sample-001", 
    units="mm",
    mano_pkl_path=Path(r"path\to\MANO_RIGHT.pkl"),  # NEW parameter (optional)
)
normalized_geometry, transform = normalize_hand(raw_geometry)
restored_geometry = denormalize_hand(normalized_geometry, transform)
```

### MANO Topology Integration

Triangle face topology can now be loaded from MANO hand models to provide complete surface representation for orthosis design. See [MANO_INTEGRATION.md](./MANO_INTEGRATION.md) for setup and usage instructions.

The current normalizer centers the bounds of all provided points, applies one uniform scale, and preserves axes without rotation or mirroring. FreiHAND points are best interpreted as camera-relative from the official intrinsic-matrix projection code; formal origin, axis signs, and handedness remain unspecified. It stores the center, inverse scale, identity rotation, and source-unit label; the original reference-bone `hand_scale` remains in `metadata`. The agreed Person 1 handoff policy is to treat raw meter-valued geometry as authoritative and use bounding-box-normalized geometry only as a separate, reversible model-input copy. Its `[-1, 1]` bounds align with Person 2's draft encoder interface. Do not infer physical orthosis dimensions from normalized coordinates alone.

Raw and normalized 3D point plots (axes can be rotated interactively in Matplotlib):

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_hand.py
.\.venv\Scripts\python.exe FreiHAND\visualize_hand.py --dataset-root path\to\FreiHAND_pub_v2 --index 0 --output FreiHAND\visualizations\01_training_00000000.png
```

The no-argument example is synthetic. Dataset-backed visualization plots the vertex point cloud and keypoints; the loader does not provide mesh faces.

After running `download_dataset.py`, run the end-to-end real-data validation script. It locates the training annotation folder, visualizes sample 0, then checks the default indices 0, 100, 1000, and 32000:

```powershell
.\.venv\Scripts\python.exe FreiHAND\run_real_validation.py
```

To choose a different sample set or extraction location:

```powershell
.\.venv\Scripts\python.exe FreiHAND\run_real_validation.py --data-dir "D:\Datasets\FreiHAND" --sample-index 5 --indices 5 200 5000
```

The script writes `FreiHAND\visualizations\01_training_00000000.png` by default. Its 3D plot contains the sample's vertices and keypoints; mesh faces are unavailable.

Run the 50-sample real-data integration check (normalization, inverse transform, deterministic 96³ encoding with named landmarks) and create the report:

```powershell
.\.venv\Scripts\python.exe FreiHAND\inspect_real_dataset.py --count 50
```

It samples indices evenly from the training range and writes `FreiHAND\results\real_sample_inspection.md`. The encoder surface channel is only vertex occupancy without faces; quantization errors are reported in normalized coordinates or meters, not continuous-surface error.

Compare resolutions on the same real samples. The current provisional engineering caps are 32 MiB for one dense output tensor and 5 seconds to encode one sample:

```powershell
.\.venv\Scripts\python.exe FreiHAND\compare_resolutions.py --count 50
```

This writes `FreiHAND\results\resolution_comparison.md`, including point and depth quantization, named-landmark cell collisions, adjacent fingertip landmark-separation error, tensor size, and encode time. The fingertip metric is not true surface clearance; the caps exclude model activations and batch storage.

Export selected real training samples as compressed, framework-neutral NPZ tensors. Files are written outside the repository by default, under `<dataset-root>\encoded_prototype\`; each sample stores its binary tensor, ordered channel names, sample identity, source units, and inverse-normalization transform metadata:

```powershell
.\.venv\Scripts\python.exe FreiHAND\export_encoded_samples.py --dataset-root "$datasetRoot" --indices 0 100 1000 32000
```

Each file is named `training_########.npz` and contains `representation` shaped `(96, 96, 96, 22)` (`uint8`, values 0/1), `keypoint_names`, and JSON text in `metadata_json`. Load with `numpy.load(path, allow_pickle=False)`. Existing exports are not overwritten unless `--overwrite` is supplied. Specify `--output-dir` to choose another destination or `--resolution` for an explicit non-default prototype experiment. These exports are model inputs, not clinical or fit-validation artifacts.

### Keras input loader

`FreiHAND.keras_voxel_loader.FreiHANDVoxelSequence` reads the exported NPZ files and yields channels-last `float32` feature batches shaped `(B, 96, 96, 96, 22)`. Stored archives remain compact `uint8`; conversion occurs as batches are read. Start with batch size 1: one converted sample is about 74.25 MiB, before temporary arrays, model activations, or other runtime overhead. The sequence yields features only because the archives do not contain prediction targets; metadata such as sample IDs and inverse-normalization values is available separately via `metadata_for_batch`.

```python
from pathlib import Path

from FreiHAND.keras_voxel_loader import FreiHANDVoxelSequence

archives = sorted(Path(r"path\to\encoded_prototype").glob("training_*.npz"))
inputs = FreiHANDVoxelSequence(archives, batch_size=1)
# Predictions only; supervised training still needs a separately defined target dataset.
# predictions = model.predict(inputs)
sample_metadata = inputs.metadata_for_batch(0)
```

TensorFlow is required only for this optional loader and its tests; TensorFlow 2.10.1 was runtime-checked in the project environment. The base encoder, archive exporter, and their tests do not import TensorFlow. Loader tests run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s FreiHAND\tests -p "test_keras_voxel_loader.py" -v
```

### Geometry-reconstruction benchmark

The first supervised target is **ordered normalized vertex coordinates**, shape `(778, 3)`. To avoid target leakage, this benchmark supplies only the vertex-occupancy channel as input: landmark channels are absent, and its normalization is computed from vertices alone. The split is a fixed-seed 70/15/15 sample-level partition of 200 evenly spaced FreiHAND training indices. This is a small representation benchmark, not an orthosis-design or clinical evaluation. Per-index vertex error assumes consistent vertex correspondence across samples; confirm that assumption before interpreting errors anatomically.

Prepare the benchmark archives and split manifest outside the repository:

```powershell
.\.venv\Scripts\python.exe FreiHAND\export_reconstruction_samples.py --dataset-root "$datasetRoot" --sample-count 200
```

The command writes `reconstruction_prototype\training_########.npz` files containing one-channel `occupancy` and `target_vertices`, plus `split_manifest.json`. Use `FreiHANDReconstructionSequence` from `FreiHAND.keras_voxel_loader` to read `(occupancy_batch, vertex_target_batch)` pairs. The existing all-channel sequence above remains for inference/input-pipeline checks; do not use it for this reconstruction task because it contains the keypoint labels as input channels.

Run the compact 3D-CNN baseline after preparing that subset:

```powershell
.\.venv\Scripts\python.exe FreiHAND\train_reconstruction_baseline.py --dataset-dir "$datasetRoot\reconstruction_prototype" --epochs 10 --batch-size 1
```

The run saves `reconstruction_baseline.h5` and `reconstruction_baseline_metrics.json` beside the benchmark archives. The report compares held-out mean per-vertex Euclidean error against a training-mean-shape baseline, in normalized units and millimeters. This small experiment only tests whether a CNN can recover the chosen vertex target from occupancy; it does not test orthosis design or clinical performance. Check the vertex-correspondence assumption before interpreting per-index error as anatomical error.

To visualize the real normalized geometry beside its encoded voxel-cell centers:

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_representation.py --dataset-root "$datasetRoot" --index 0
```

This writes `FreiHAND\visualizations\03_spatial_representation_training_00000000.png` by default. Since face topology is absent, the surface channel shows vertex occupancy rather than a densely sampled continuous surface.

Test synthetic ingestion/normalization and the spatial encoder independently:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s FreiHAND\tests -p "test_geometry_pipeline.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s FreiHAND\tests -p "test_encode_hand.py" -v
```

See [geometry_pipeline.md](./geometry_pipeline.md) for verified reference-code findings, normalization formulas, limitations, and handoff questions.
Use [domain_requirements_review.md](./domain_requirements_review.md) as a fill-in worksheet for a qualified orthosis/hand-domain reviewer. It asks for intended posture/load, contact/keep-clear regions, measurable fit/clearance limits, representation-error budget, validation data, and written sign-off; it intentionally proposes no clinical thresholds.
The worksheet is not required to continue prototype encoder development. Complete it before making fit, clearance, safety, or clinical claims or evaluating an actual orthosis against this representation.

## Input and normalization boundary

The encoder accepts normalized XYZ arrays; it is independent of the FreiHAND file layout:

- `vertices`: required finite array shaped `(N, 3)`, `N > 0`.
- `faces`: optional integer triangle indices shaped `(F, 3)`, zero-based into `vertices`.
- `keypoints`: optional finite array shaped `(K, 3)`, `K > 0` when supplied.

The proposed default coordinate cube is inclusive `[-1, 1]` on each axis. This is an explicit provisional assumption only; the encoder does **not** translate, rotate, rescale, or infer axes. Confirm the normalized coordinate frame, handedness, physical-scale metadata, and keypoint conventions with Person 1 before integration. Out-of-bounds and invalid arrays raise `ValueError`; geometry is never silently cropped.

## Output

`encode_hand_geometry(vertices, faces=None, keypoints=None, keypoint_names=None, resolution=96, bounds=(-1.0, 1.0))` returns a NumPy `uint8` array. When `keypoint_names` are supplied, the output shape is `(96, 96, 96, 1 + K)`; for FreiHAND's 21 labeled landmarks, that is `(96, 96, 96, 22)`. Without names, the backward-compatible shape is `(96, 96, 96, 2)`.

1. Channel 0 is vertex/surface occupancy: cell receives a vertex or, when faces are provided, deterministic samples from its triangle surface.
2. Named channels that follow channel 0 each preserve one keypoint's identity. Pass `FREIHAND_KEYPOINT_NAMES` for FreiHAND's official ordered keypoints. The anatomical chain labels are in `recommended_representation.md`.

Values are binary `{0, 1}`. Face triangles are sampled deterministically at no more than half-cell spacing along their longest edge. With no faces, only vertices are marked; no unobserved surface is synthesized. `decode_named_keypoints` returns cell centers separately per landmark channel but cannot recover sub-cell detail. The normalization transform and reference-bone scale must be retained separately.

The 96 resolution is the selected prototype setting under provisional limits of 32 MiB per dense sample tensor and 5 seconds encoding time per sample. The derived point-to-cell-center bound is `sqrt(3)/96` normalized units, and the per-axis depth bound is `1/96`; these are quantization guarantees, not fit or clinical tolerances. Resource limits exclude model activations/batch storage.

## Install / environment

The repository `.venv` inspected for this work already contains Python 3.8.10, NumPy, and Matplotlib. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s FreiHAND\tests -v
```

The geometry encoder and visualization require NumPy and Matplotlib. TensorFlow is an optional dependency used only by the Keras archive loader and its tests; TensorFlow 2.10.1 was verified in this workspace.

## Run

Generate the built-in synthetic example visualization:

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_representation.py
```

The script writes `FreiHAND\visualizations\03_spatial_representation_synthetic.png` and prints that its input is synthetic, not FreiHAND data.

For normalized sample arrays, save an `.npz` with a required `vertices` array and optional `faces` and `keypoints` arrays, then run:

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_representation.py --input path\to\normalized_sample.npz --output FreiHAND\visualizations\03_spatial_representation.png
```

Programmatic usage:

```python
from FreiHAND.encode_hand import encode_hand_geometry, decode_hand_geometry

voxel_grid = encode_hand_geometry(vertices, faces=faces, keypoints=keypoints)
surface_cell_centers, keypoint_cell_centers = decode_hand_geometry(voxel_grid)
```

## Documents and tests

- `representation_comparison.md`: candidate tradeoffs, measured real-sample resolution study, recommendation, and limitations.
- `recommended_representation.md`: coordinate/tensor/input/output contract, anatomical landmark ordering, Person 2/3 measurement plan, and remaining design decisions.
- `results/resolution_comparison.md`: measurements from five voxel resolutions on 50 real training samples.
- `representation_contract.json`: machine-readable prototype engineering contract.
- `export_encoded_samples.py`: exports selected real FreiHAND annotations to compressed model-input tensors and preserves the inverse transform metadata.
- `tests/test_encode_hand.py`: synthetic checks for shape, type/range, determinism, invalid data, bounds, axis direction, multiple shapes, triangle sampling, and approximate decoding.

Run tests from the repository root with the command above. The implementation has also been exercised on 50 real FreiHAND training samples; the current run is recorded in `results/real_sample_inspection.md`. That check is geometry/encoder validation, not clinical validation.

## Known limitations / review needed

- Bounds, formal source-frame origin/axis signs/handedness, and orthosis-specific tolerances remain to be reviewed. Official FreiHAND evaluation code supports meters as the source unit. Normalized geometry alone cannot recover physical scale.
- The generic unlabeled-keypoint mode still unions keypoints; use separate named channels when landmark identity matters.
- Occupancy loses sub-voxel geometry, topology, normals, signed distance, and interior; gaps/thin features depend on resolution.
- Vertices without faces form sparse occupancy and do not describe a continuous surface.
- The 96-cube setting is selected for prototype engineering evaluation under provisional payload/runtime caps. Mathematical quantization bounds apply to encoded points only; actual surface clearance and clinical/fit acceptance are out of scope.
- Person 3 should define and review orthosis-related acceptance thresholds before the representation is considered validated.
- The voxel tensor could feed a future 3D convolutional model, but no future ResNet or PEA encoding is implemented or validated here.
