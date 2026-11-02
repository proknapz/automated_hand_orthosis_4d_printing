# Recommended representation contract (prototype)

## Status and evidence labels

- **Verified:** the repository's assignment requires a generic normalized geometry interface and orthosis-relevant preservation of shape, depth, joints, finger gaps, wrist alignment, and recoverable physical size.
- **Verified:** the local FreiHAND training set was processed at 50 evenly spaced indices. Each selected sample contained `(778, 3)` vertices, `(21, 3)` keypoints, and one positive reference-bone scale value. All 50 passed normalization and named-landmark encoding checks. Details are in `results/real_sample_inspection.md`.
- **Verified:** the official [`eval_util.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py) calculates Euclidean error directly from stored XYZ values; [`eval.py`](https://github.com/lmb-freiburg/freihand/blob/master/eval.py) reports the result in centimeters by multiplying by 100 and evaluates thresholds up to `0.05` in the unscaled coordinates. This establishes the dataset XYZ unit as meters. For all 32,560 local training records, `training_scale` exactly equals `norm(xyz[9] - xyz[10])`.
- **Verified anatomical ordering:** the official FreiHAND helper maps its 21 keypoints to a wrist and five wrist-to-tip chains; tips are indices 4, 8, 12, 16, and 20. This supports the finger/chain labels below. Intermediate joints use ordinal names rather than asserting a specific clinical joint name. See the official [`mano_utils.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/mano_utils.py) and [`fh_utils.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/fh_utils.py).
- **Operational convention / unresolved source fact:** the encoder uses the supplied XYZ components without rotation or reflection; normalization translates and uniformly scales them and retains the inverse transform. The official projection code supports a camera-relative interpretation, but the inspected source does not formally declare origin, axis names/signs, or handedness. The FreiHAND archive does not include face connectivity or per-face/per-vertex anatomical surface labels.
- **Verified topology pathway, not yet integrated:** the official FreiHAND [README](https://github.com/lmb-freiburg/freihand#advanced-setup-allows-for-visualization-of-mano-shape-annotation) requires separately obtained MANO model files for mesh rendering; [`setup_mano.py`](https://github.com/lmb-freiburg/freihand/blob/master/setup_mano.py) expects `MANO_RIGHT.pkl`, and [`utils/model.py`](https://github.com/lmb-freiburg/freihand/blob/master/utils/model.py) uses that model's `f` faces with its generated vertices. That licensed model asset is absent locally. Treat its topology as a candidate only after obtaining the authorized asset and verifying its vertex indexing against FreiHAND annotations.
- **Prototype decision:** proceed with 96-cubed encoding for research-prototype engineering evaluation. The project accepts the prototype checks below as implementation criteria only; they are not domain-reviewed fit tolerances or clinical acceptance criteria.
- **Design limitation:** the 96-cubed setting is selected under the explicit per-sample compute limits below. No actual orthosis fit, pressure, comfort, clearance, or clinical performance is evaluated.

## Tensor specification

| Field | Contract |
|---|---|
| Representation type | Dense 3D binary occupancy grid |
| Spatial dimensions | `96 x 96 x 96`, axis order `(X, Y, Z)` |
| Full array shape | `(96, 96, 96, 22)`; channels-last |
| Data type | `numpy.uint8` |
| Value range | `{0, 1}` |
| Bounds | Inclusive normalized cube `[-1.0, +1.0]` on X, Y, and Z |
| Origin | Grid index `(0, 0, 0)` is the cell at `(-1, -1, -1)`; the coordinate origin `(0,0,0)` is at the cube center |
| Orientation | The encoder preserves incoming X/Y/Z without rotating or reflecting. Axis/anatomical meaning and sign must be supplied by the upstream normalizer and agreed by the team. |
| Cell width | `2 / 96 = 0.0208333` normalized coordinate units per axis |
| Physical scale | Not embedded. Retain the inverse transform, `"m"` source-unit label, and dataset reference-bone scale. Convert errors to millimeters with `1 m = 1000 mm`; do not use normalized coordinates alone as physical dimensions. |
| Channels 0 | Vertex/surface occupancy: `1` for a cell receiving an input vertex or deterministic sample from an input triangle; otherwise `0`. Current FreiHAND file inputs have no faces, so this means vertex occupancy only. |
| Channels 1-21 | One named occupancy channel per keypoint, in the exact order in the table below. A channel's occupied cell preserves that landmark identity even if two landmarks quantize to the same XYZ cell. |
| Single-sample tensor payload | `96^3 * 22 * 1 byte = 19,464,192 bytes = 18.5625 MiB`. This excludes temporary arrays, model activations, and batch storage. |

FreiHAND keypoint channel order:

| Channel | Keypoint index | Name |
|---:|---:|---|
| 1 | 0 | wrist |
| 2-5 | 1-4 | thumb_joint_1, thumb_joint_2, thumb_joint_3, thumb_tip |
| 6-9 | 5-8 | index_joint_1, index_joint_2, index_joint_3, index_tip |
| 10-13 | 9-12 | middle_joint_1, middle_joint_2, middle_joint_3, middle_tip |
| 14-17 | 13-16 | ring_joint_1, ring_joint_2, ring_joint_3, ring_tip |
| 18-21 | 17-20 | pinky_joint_1, pinky_joint_2, pinky_joint_3, pinky_tip |

Multiple channels may be `1` at the same spatial cell. Face indices are zero-based triples into `vertices`. Triangle faces are sampled on a deterministic barycentric lattice with longest-edge sample spacing no greater than half a voxel width; the encoded object is still a sampled surface, not a filled solid. Without faces, channel 0 only marks vertices. Inputs must be finite XYZ arrays and fit entirely within configured bounds; the encoder raises `ValueError` rather than clipping.

## Input interface

```python
encode_hand_geometry(
    vertices,
    faces=None,
    keypoints=None,
    keypoint_names=None,
    resolution=96,
    bounds=(-1.0, 1.0),
) -> numpy.ndarray
```

- `vertices`: required non-empty `(N, 3)` array of finite normalized coordinates.
- `faces`: optional `(F, 3)` integer triangle-index array; indices must be valid for `vertices`.
- `keypoints`: optional non-empty `(K, 3)` array of finite normalized coordinates.
- `keypoint_names`: optional unique names, one per keypoint. When supplied, output has one keypoint channel per name. Pass `FREIHAND_KEYPOINT_NAMES` for official FreiHAND order. Without names, the backward-compatible two-channel surface-plus-union-keypoint format is returned.
- `resolution`: integer at least 2; default 96. Non-default resolution changes every spatial dimension consistently.
- `bounds`: finite `(minimum, maximum)` pair with `minimum < maximum`; same bounds apply to all axes.

The encoder expects Person 1's optional normalized geometry and does not normalize, reorder, transform, impute, rescale, or infer camera coordinates. FreiHAND source coordinates are in meters, and their camera-relative interpretation is supported by official projection code; the formal source origin, axis signs, and handedness remain unspecified. Person 1's raw geometry is authoritative; retain it alongside the optional normalized copy, source reference-bone scale, and normalization transform. The voxel bounds and per-sample bounding-box normalization remain **provisional engineering conventions**, not a patient anatomical frame or an agreed clinical/design frame.

## Quantization

For each coordinate component `p` and resolution `R`, its cell index is:

`floor((p - lower_bound) * R / (upper_bound - lower_bound))`

The inclusive upper-bound coordinate maps to index `R - 1`. Every other accepted coordinate maps to its unique half-open cell. No accepted coordinate is clipped from outside the bounds; out-of-range points fail before encoding. Grid axis 0 increases with input X, axis 1 with input Y, and axis 2 with input Z.

## Approximate decoder

`decode_hand_geometry(representation, bounds=(-1, 1))` returns two XYZ point arrays: centers of occupied channel-0 cells and centers of any keypoint channel. `decode_named_keypoints(representation, keypoint_names, bounds=(-1, 1))` returns cell centers per name. Neither decoder recovers original vertices, triangle connectivity, sub-cell locations, or physical units. These decoders are for coarse geometric checks and visualization only.

## Preserved and lost information

**Preserved approximately:** coarse 3D location and vertex/surface distribution; depth/out-of-plane placement; named keypoint locations to voxel precision; shared frame and extent when upstream normalization is consistent. Cell-center decoding has per-axis quantization error up to half a cell, and Euclidean error up to half a voxel diagonal (`sqrt(3) * cell_width / 2`) for a point assigned to its cell.

**Lost or not represented:** within-cell shape, exact vertex coordinates, face topology, surface normals, signed distance/interior, sub-voxel fingertip/joint gaps, scale unless carried separately, and geometry omitted upstream. Named identities survive in channels, but location is quantized. Small clearances or thin parts can be under-resolved or merge. Vertices-only input is not densified into a surface.

## Prototype engineering checks (not clinical tolerances)

For this research prototype, use the following deterministic implementation checks. They are derived from the selected grid and resource caps, not selected as safe fit tolerances. The voxel bounds are mathematical guarantees for point-to-cell-center quantization only; they say nothing about source geometry accuracy or orthosis fit.

For resolution `R=96` over `[-1,1]`, cell width is `w=2/R=1/48` normalized units:

- Per-point 3D distance to its assigned cell center is bounded by `sqrt(3) * w / 2 = sqrt(3)/96 = 0.018043` normalized units.
- Per-axis depth error to a cell center is bounded by `w/2 = 1/96 = 0.010417` normalized units.
- Absolute error in a distance between two quantized landmark centers is bounded by `sqrt(3) * w = sqrt(3)/48 = 0.036085` normalized units, by the triangle inequality. Keep the measured sample statistics too.
- Convert normalized bounds to source meters per sample by multiplying by that sample's retained `GeometryTransform.scale`. They are not one fixed millimeter tolerance across hands because normalization is per sample.
- Encoding must reject out-of-bounds input (zero silent clipping), preserve each named landmark channel, be deterministic, return binary `uint8`, and remain within the provisional 32 MiB tensor-payload and 5-second-per-sample engineering caps.

Surface deviation, adjacent-digit skin clearance, hand-to-device clearance, intrusion, fit, and clinical outcomes remain **not evaluated**. Use the reviewer worksheet if or when those questions enter scope; approved limits also require suitable hand and device geometry, posture, and registration.

| Requirement | Proposed measurable quantity | Current status / limitation |
|---|---|---|
| Preserve source geometry locations | Per-point Euclidean distance from each normalized source vertex/keypoint to its assigned voxel-cell center; report mean, 95th percentile, and maximum in normalized coordinates and convert through the retained transform to meters. | Prototype bound: `sqrt(3)/96 = 0.018043` normalized units at 96³. Point quantization only; not surface reconstruction or fit error. |
| Preserve named landmarks | One occupancy channel per verified ordered landmark; test its channel remains populated and report location error and landmark positions sharing spatial indices. | Prototype gate: all 21 ordered channels populated. Labels name wrist/finger chain/ordinal joint; not clinical joint annotations. All 21 were present and occupied distinct positions in the 50-sample comparison. |
| Preserve finger separation | Compare distances between adjacent named fingertip landmarks before/after encoding; report absolute errors and any spatial-cell coincidences. | Prototype reporting only; two-point distance error bound `sqrt(3)/48 = 0.036085` normalized units. This is not surface clearance; true inter-finger gap width requires verified surfaces and anatomical regions. |
| Preserve surface shape/depth | Symmetric point-to-surface distance (for example, sampled bidirectional distance) and maximum depth-coordinate error after decoding in the same frame. | Depth-coordinate error for input points is measurable now. Surface distance against the continuous source surface is blocked until verified faces or another surface reference is available. |
| Preserve physical size | Compare source and restored reference-bone length and report whether raw metric geometry plus normalization transform reproduces it. | Raw coordinates and reference scale are in meters. Normalized tensor alone does not preserve physical size; retain the raw geometry, reference scale, and transform. |
| Avoid clipping | Number of source points outside configured bounds and number omitted by encoding. Out-of-bounds input must raise rather than be clipped. | Prototype gate: zero silent clipping; out-of-bounds input raises an error. Not a fit-quality threshold. |
| Reproducibility and cost | Exact tensor equality across repeated runs; output bytes; median/p95/max encode time under a documented runtime. | Prototype gates: deterministic equality, binary `uint8`, at most 32 MiB dense tensor payload and 5 seconds per sample. These provisional caps exclude model activations and batch storage. |

### Decision gates

1. The 50-sample comparison is in `results/resolution_comparison.md`. Use 96³ as the selected prototype setting under provisional 32 MiB/5-second caps; 128³ exceeds the tensor-payload cap. Apply the prototype quantization checks above, not clinical acceptance criteria.
2. Actual surface comparisons are outside prototype scope. If needed later, obtain the MANO asset through its authorized distribution, verify its expected hash/version and 778-vertex ordering against FreiHAND samples, and only then attach its faces. MANO faces alone do not provide validated orthosis contact zones or anatomical surface labels.
3. The fingertip measurement remains a landmark-separation proxy. Actual skin or hand-device clearance requires suitable posture-specific hand and device surfaces, registration, and reviewed regions/method.
4. Keep engineering representation checks, orthosis fit acceptance, and clinical validation as separate claims.

## Prototype scope and later review

Proceed with 96³ for prototype encoding and algorithm testing under the engineering gates above. The `domain_requirements_review.md` worksheet is **not a blocker** for that limited activity. Complete it before making fit, clearance, safety, or clinical claims or evaluating an actual orthosis against the representation.

The implementation has been checked on 50 real FreiHAND training samples. Official evaluation-code conventions support meters as the raw XYZ unit. No clinical suitability or orthosis-fit acceptance is claimed.
