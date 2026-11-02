# FreiHAND geometry pipeline: verified findings and draft preprocessing

## Status at implementation

**Real FreiHAND training files are available locally outside the repository** under the user's dataset directory. The loader was run on 50 training annotation indices evenly spaced from 0 through 32,559; all 50 loaded and passed deterministic normalization, inverse-transform, bounds, and Person 2 voxel-encoding checks. The generated report is [real_sample_inspection.md](./results/real_sample_inspection.md). The dataset's large `training_verts.json` was inspected with memory-mapped per-sample indexing; it was not loaded wholesale.

Dataset source-level facts below come from the official [FreiHAND dataset page](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html), [reference repository](https://github.com/lmb-freiburg/freihand), [`utils/fh_utils.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/fh_utils.py), [`pred.py`](https://github.com/lmb-freiburg/freihand/blob/master/pred.py), [`eval.py`](https://github.com/lmb-freiburg/freihand/blob/master/eval.py), and [`utils/eval_util.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py). Local measurements were made against the downloaded v2 training annotations.

- Locally inspected annotation files are aligned across 32,560 unique training indices. Each selected `training_verts.json` record has **778 XYZ vertices**; each selected `training_xyz.json` record has **21 XYZ keypoints**. The official page and prediction interface describe coordinates/reference scale as metric. The official evaluation code resolves the SI prefix: `utils/eval_util.py` computes Euclidean error directly in the supplied XYZ units, while `eval.py` reports mean error in centimeters as `xyz_mean3d * 100.0` and uses thresholds through `0.05` before that conversion. Therefore FreiHAND XYZ and reference-bone values are in **meters**. This repository labels FreiHAND samples `"m"`; custom NPZ inputs must provide their own explicit unit label.
- **Full-training verification:** for all 32,560 local `training_xyz.json` records, the value in `training_scale.json` exactly equals `norm(xyz[9] - xyz[10])` (maximum absolute difference `0.0`). The official reference `pred.py` identifies this as the middle-finger proximal-phalanx reference bone. Thus the scale and XYZ arrays use the same numeric coordinate unit.
- The official `projectPoints` helper in `fh_utils.py` computes homogeneous image coordinates using `K @ xyz.T`, then divides by the third component. Across all training samples, the two focal-length diagonal entries of `K` are positive and every annotated keypoint has positive Z (`0.29707` to `1.04283` in the local files). This supports an **inferred camera-relative pinhole projection frame**, with positive Z as positive projection depth. Under the standard intrinsic-matrix convention, positive X projects toward increasing image `u` and positive Y toward increasing image `v` (downward in image coordinates). The official documents/code inspected here do not explicitly specify the 3D origin, handedness, or axis names/signs as a formal coordinate-frame contract; treat those axis statements as inferred, not authoritative.
- The sampled raw vertex bounds across the earlier 50-sample report were X `[-0.09633, 0.10658]`, Y `[-0.09015, 0.10109]`, Z `[0.31354, 0.98067]` meters.
- The official MANO helper `split_theta` slices each annotation as `theta[:, :48]`, `theta[:, 48:58]`, `theta[:, 58:60]`, and `theta[:, 60:]`; actual training rows loaded here conform to the loader's 2D `(1, P), P >= 61` check.
- FreiHAND training annotations index unique samples; image variants `gs`, `hom`, `sample`, and `auto` are mapped to distinct image IDs by adding multiples of 32,560. The annotation sample index is not the image filename index for every variant.
- Vertex positions are present in `training_verts.json`. Triangle face topology is not included in the FreiHAND archive, so `faces` remains `None`; no faces or anatomical vertex labels are invented.
- **Verified route to candidate topology:** the official FreiHAND [README](https://github.com/lmb-freiburg/freihand#advanced-setup-allows-for-visualization-of-mano-shape-annotation) describes obtaining MANO models separately for shape rendering; [`setup_mano.py`](https://github.com/lmb-freiburg/freihand/blob/master/setup_mano.py) expects `MANO_RIGHT.pkl`, and official [`utils/model.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/model.py) obtains faces from `self.model.f`. The licensed MANO asset is not present in this local dataset/repository. Before using its faces on `training_verts`, obtain the authorized asset and verify the expected model version/hash, 778-vertex indexing, face index range, and correspondence by regenerating a sample with the official model code and comparing against the stored vertices. Topology from MANO does not by itself supply validated contact regions or clinical surface labels.

The official project page and version/license guidance are linked from the [reference README](https://github.com/lmb-freiburg/freihand#readme). Confirm the dataset version/license and MANO model compatibility before processing it.

## Implemented interfaces

### Official-format annotation reader

```python
from FreiHAND.data_loader import FreiHANDDataset

dataset = FreiHANDDataset(
    dataset_root=r"path\to\FreiHAND_pub_v2",
    split="training",
    image_version="gs",
    units="m",
)
sample = dataset.load_sample(0)
```

`FreiHANDDataset` reads the aligned camera, MANO, and keypoint annotations. If available, it also indexes `training_verts.json` via memory mapping and loads the selected vertex sample without loading the 1.67 GB JSON file into memory. It loads the small `training_scale.json` alongside annotations. The returned generic `HandGeometry` contains:

- `sample_id`: e.g. `training:00000000`, based on split and annotation index.
- `keypoints`: `(21, 3)` float64, because the official reference plot accesses 21 3D landmarks.
- `vertices`: `(778, 3)` float64 for the inspected training version; `None` if a vertex file is absent.
- `faces`: `None`; the dataset vertex JSON does not supply triangle connectivity.
- `units`: caller-overridable; defaults to `"m"` based on the official evaluator's centimeter conversion from stored XYZ distances.
- `metadata`: annotation index, split/version, mapped relative image index/path, camera intrinsics `(3, 3)`, MANO parameter row, and optional reference-bone `hand_scale`.
- `metadata["source_coordinate_frame_note"]` and `metadata["source_coordinate_units_note"]` record the source-frame/unit caveats and clarify that normalized coordinates are dimensionless.

`load_freihand_sample(...)` is the single-sample convenience function. The reference image mapping is exposed for training variants; image files are not opened by the geometry loader. Evaluation annotations must exist locally and use `gs`.

### Generic geometry interchange

`load_geometry_npz(path, sample_id=None, units="unknown")` accepts a non-pickled `.npz` archive with optional `vertices`, `keypoints`, and `faces` arrays. At least vertices or keypoints must be present. Faces are zero-based triangles with shape `(F, 3)` and are checked against vertices. Caller metadata is preferred; otherwise the archive stem is used as a file identifier, not represented as an official FreiHAND ID.

This makes normalized vertices/keypoints available to Person 2 without coupling the encoder to dataset JSON or image naming. A verified face topology is still needed for surface-sampled voxelization; the current encoder falls back to vertex occupancy.

## Draft normalization rule

`normalize_hand(geometry, output_bounds=(-1, 1))` currently:

1. Validates and uses the union of vertices and keypoints.
2. Sets translation center `c = (minimum + maximum) / 2` per input axis.
3. Finds the largest axis extent `E = max(maximum - minimum)`.
4. Uses one uniform scale `s = E / (upper_bound - lower_bound)`.
5. Preserves axis orientation with the identity rotation `R`.
6. Computes each point `p' = ((p - c) @ R.T) / s + (lower_bound + upper_bound)/2`.

`denormalize_hand` inverts this exactly (within floating-point precision):

`p = ((p' - bounds_midpoint) * s) @ R + c`.

The returned `GeometryTransform` stores `center`, `scale`, `rotation`, output bounds, and original unit label. Uniform scale preserves proportions, and `scale` plus the source-unit label retains source dimensions. For FreiHAND, source coordinates are meters; normalized coordinates are dimensionless. The same transform applies to keypoints and vertices. Faces and sample ID are copied unchanged. `normalize_hand` creates a separate object and does not overwrite the source geometry.

This is **not** an anatomical origin or orientation normalization. No axis rotation, mirroring, camera-to-hand conversion, or projection correction is performed. Bounding-box centering loses the raw translation from the point arrays but records it for inversion. Uniform scale removes size from the normalized coordinates but records its factor. Different poses can therefore have different local orientation, and the scale depends on the extent of the supplied points. It is a reversible draft chosen to align with Person 2's provisional `[-1, 1]` bounds, not an approved patient-fit convention.

The training annotations also include a dataset-defined reference-bone metric length in `training_scale.json`; it exactly matches the XYZ distance between keypoints 9 and 10 across all 32,560 local training records. The normalizer preserves it in geometry metadata but does not use it as the scale denominator.

**Current Person 1 handoff policy:** raw meter-valued geometry is authoritative. Bounding-box-normalized geometry is an optional, separate, reversible model-input copy; its inverse transform and reference-bone scale must travel with it. Do not use normalized coordinates alone to specify physical orthosis dimensions.

## Visualization and validation

Raw and normalized geometry are plotted side by side in rotatable Matplotlib 3D axes with sample label, coordinate axes, and keypoints:

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_hand.py
```

Default writes `FreiHAND\visualizations\01_raw_and_normalized_synthetic.png` and labels the plot as synthetic. For actual downloaded annotations:

```powershell
.\.venv\Scripts\python.exe FreiHAND\visualize_hand.py --dataset-root path\to\FreiHAND_pub_v2 --index 0 --output FreiHAND\visualizations\01_training_00000000.png
```

This displays the 778 vertex point cloud and keypoints when the training vertex annotations exist. Faces are not available through this loader.

Run the real-data normalization and spatial-encoder check on 50 evenly spaced training samples and write the measured summary:

```powershell
.\.venv\Scripts\python.exe FreiHAND\inspect_real_dataset.py --count 50
```

The default report is `FreiHAND\results\real_sample_inspection.md`. It records per-sample vertex/keypoint counts, normalization scale, provided meter-valued reference-bone scale, occupancy counts, and quantization error to cell centers. The quantization metric is in normalized coordinates and is not a continuous-surface reconstruction error.

Selected-index checks (errors are reported per index; any failure yields a nonzero process exit):

```powershell
.\.venv\Scripts\python.exe FreiHAND\validate_geometry.py --dataset-root path\to\FreiHAND_pub_v2 --indices 0 100 1000 32000
```

Synthetic tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s FreiHAND\tests -p "test_geometry_pipeline.py" -v
```

## Limits and handoff questions

- Real array dimensions and coordinate ranges were measured for 50 training records, but not for every train/evaluation record.
- Camera-relative projection is supported by the official K projection helper and is the best-supported working interpretation. Positive projection depth and image-plane directions are inferred from the projection formula and local data. Exact formal origin, handedness, and axis naming are not specified by the documentation inspected.
- XYZ and reference-bone values are in meters, and the reference-bone value matches `norm(xyz[9] - xyz[10])` exactly for all 32,560 local training records.
- Face topology requires the authorized, version-compatible MANO model asset and a vertex-correspondence integration test. Anatomical vertex-to-keypoint/surface-region labels are not provided by the dataset; keypoint chains do not define surface regions.
- The 50-sample report found normalization factors ranging from `0.05653` to `0.09547` meters per normalized unit, while reference-bone values ranged from `0.01651` to `0.03418` meters. This variation is one reason bbox scaling should not be treated as preservation of patient size; retain the supplied reference scale and discuss the normalization rule with the team.
- The 21 keypoint ordering and finger-chain labels are supported by the version-matched official helper and plotting code. Intermediate ordinal labels are not clinical joint names; this ordering does not provide anatomical labels for the 778 surface vertices.
- The camera intrinsic matrix is preserved as metadata; this pipeline does not use it to deproject or transform coordinates.
- Keypoint-only annotations can be centered/scaled but cannot be treated as a dense hand surface by Person 2. A mesh/point-surface source is required for the voxel surface channel.
- Person 2 and Person 3 should review the optional bounding-center origin, per-sample scale, retained axes, and implications of pose-dependent bounds before using normalized coordinates for design. Do not call the transform clinically validated.
- Use [domain_requirements_review.md](./domain_requirements_review.md) to collect a qualified review of intended use, posture/load, fit/clearance limits, validation data, and engineering error budgets. Do not supply thresholds on the reviewer's behalf.
