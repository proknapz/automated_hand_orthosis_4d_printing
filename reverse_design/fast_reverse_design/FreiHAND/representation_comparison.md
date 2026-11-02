# Spatial representation comparison

## Evidence and scope

**Verified repository facts:** the supplied FreiHAND task assignment describes normalized geometry as a shared, generic hand-geometry input and lists orthosis-relevant shape/depth/pose and physical-size requirements. The local official-format training data was inspected at 50 evenly spaced indices. Every selected sample had 778 vertices, 21 keypoints, and a positive reference-bone scale. The official reference code defines a wrist-to-fingertip chain order; the implementation preserves all 21 keypoint identities in separate channels. The 96-cubed named-landmark encoder was run on these normalized samples. Detailed results are in `results/real_sample_inspection.md` and `results/resolution_comparison.md`. Faces and orthosis acceptance tolerances remain unavailable.

The comparison among all five representation types remains an **engineering assessment**, not a head-to-head benchmark: only the 3D voxel encoder has been implemented and measured on real samples. Example sizes for alternatives use declared candidate settings (128-square 2D grids, 2,048-point clouds, float32 where indicated) to compare payload sizes; they are not measured per-sample performance or optimal settings. No clinical suitability is inferred.

## Criteria fixed before implementation

1. Preserve 3D location and out-of-plane pose; do not project away depth.
2. Make representation bounds and cell size explicit and reject out-of-bounds geometry rather than silently cropping it.
3. Preserve the 21 FreiHAND keypoint identities in separate named channels; finger-chain labels follow the official ordering, while intermediate joints use ordinal rather than clinical joint names.
4. Keep physical size recoverable through upstream transform metadata; normalized coordinates alone cannot encode absolute dimensions.
5. Provide fixed tensor dimensions for the first CNN-oriented interface, deterministic encoding, and an approximate decoder for validation.
6. Assess orthosis-specific measurements with Person 3. Relevant measures include keypoint distances/angles, palm/finger gaps, surface deviation, and depth error. No clinical or orthosis-specific tolerances have been agreed.

## Candidate comparison

| Candidate | Example tensor and storage | Shape/depth and information loss | CNN / future spatial ResNet | PEA consideration | Reproducibility and cost |
|---|---|---|---|---|---|
| 2D occupancy | `128 x 128 x 1`, uint8: 16 KiB | Preserves one silhouette/projection only; loses depth ordering, palm thickness/curvature, and out-of-plane finger/thumb pose. Occlusion merges separated parts in projection. | Direct 2D CNN input; spatial relationships only in the chosen projection. | Easy to flatten, but lacks geometry needed to reason about depth and fit. | Small and cheap; view direction and projection convention strongly affect result. |
| 2D occupancy + depth | `128 x 128 x 2`, float32: 128 KiB | Adds a depth value per pixel, but still represents only visible/frontmost surface; hidden side, back surface, and multiple depths per pixel are lost. | Direct 2D CNN input with depth channel. | Can support visible-surface features; cannot by itself represent full hand volume/surface. | Moderate cost; depends on projection, visibility, and a stable depth convention. |
| 3D voxel occupancy | Selected candidate: `96 x 96 x 96 x 22`, uint8: 18.56 MiB (one vertex channel plus 21 named landmark channels) | Preserves coarse 3D location, vertex occupancy, and individual landmark identity; quantizes within-cell geometry, thin structures/gaps may disappear. No continuous surface or signed distance field without faces/other surface encoding. | Fixed regular tensor for 3D convolutions/spatial ResNet; memory and compute grow cubically in spatial resolution and linearly with channels. | Future optimizer mapping remains unspecified; binary occupancy changes are discontinuous. | Deterministic given frame/bounds/resolution. Output payload scales as `R^3 * channels`; face sampling cost remains dependent on faces and sample density. |
| Point-based | Example `2048 x 3`, float32: 24 KiB | Retains 3D samples and often finer shape at low storage, but loses between-sample surface detail; results depend on sampling/ordering unless standardized. | Natural for point networks; not a regular grid for a conventional spatial ResNet without a point-to-grid or graph operation. | Could retain a geometric point set, but chromosome-to-shape mapping needs a separate design. | Sampling and distance calculations can be costly; fixed-count resampling rules and frame alignment are required. |
| Mesh-derived | Variable `V x 3` float32 vertices plus `F x 3` integer faces; size depends on input, topology, and index dtype | Can retain connected surface topology and surface curvature better than occupancy, but quality depends on mesh validity and consistent topology. Faces may not be supplied by the upstream pipeline. | Not directly a dense CNN tensor; mesh/graph network or rasterization needed. | Rich surface data, but topology-aware design encoding is a distinct future decision. | Preserves source detail without resampling when topology is valid; variable meshes and topology complicate batching and cross-sample comparisons. |

Storage numbers above are raw array payloads only (not framework buffers, metadata, temporary arrays, or model activations). For the selected named-landmark candidate, `96^3 * 22 * 1 byte = 19,464,192 bytes = 18.5625 MiB`. A higher spatial resolution substantially increases memory and convolution cost.

## Real-data resolution comparison

On 50 training indices evenly spaced from 0 through 32,559, five resolutions (32, 48, 64, 96, 128) were measured. The provisional operational caps were **32 MiB per dense uint8 sample tensor** and **5 seconds per sample**. The 128-cubed, 22-channel tensor requires 44 MiB and exceeds the payload cap. The 96-cubed tensor requires 18.56 MiB and is the highest tested resolution within those caps. The comparison report records environment-specific timing for reproducibility.

At 96 cubed, every one of the 21 named landmark channels remained occupied, and all 21 landmark locations occupied distinct XYZ cells across these 50 samples. The worst measured point-to-cell-center error was `0.0016893 m`; worst Z-only error was `0.00098509 m`; the worst adjacent fingertip landmark-separation error was `0.00208575 m`. The official evaluator's raw-error-to-centimeter conversion establishes meters as the source unit. These are point/landmark quantization metrics, not clinical tolerances or inter-surface clearance. Full table and method: `results/resolution_comparison.md`.

The raw vertex coordinates in the selected records ranged across bounds X `[-0.09633, 0.10658]`, Y `[-0.09015, 0.10109]`, Z `[0.31354, 0.98067]` meters. The reference-bone value exactly matched `norm(xyz[9] - xyz[10])` across all 32,560 local training records. Official `eval.py` multiplies raw XYZ mean error by 100 for a centimeter result, establishing the meter scale. The bbox normalizer scale was 0.05653–0.09547 meters per normalized unit. This demonstrates why normalized coordinates alone must not replace retained size metadata.

## Recommendation

Use **3D vertex occupancy plus 21 named keypoint occupancy channels** as the selected prototype representation: fixed `96 x 96 x 96 x 22`, uint8. It preserves three-axis spatial placement, makes depth explicit rather than collapsing it into a view, has fixed dimensions for batching and 3D convolution, and does not merge landmark identity. The FreiHAND names preserve wrist/finger-chain labels from official order; three intermediate joints per digit are given ordinal names, not clinical MCP/PIP/DIP assertions.

Proceed with 96³ for prototype engineering checks only. For `[-1,1]` bounds, cell width is `1/48` normalized coordinate units, point-to-cell-center error is bounded by `sqrt(3)/96`, and per-axis error by `1/96`; physical error varies per sample and is recovered through its retained transform. Adjacent fingertip distance is only a landmark gap proxy. True surface clearance, narrow finger gaps, and surface deviation remain unmeasured because verified face topology and reviewed surface regions are absent. The dense tensor cost in a future model depends on batching and activations and must be profiled on target hardware. This does **not** solve a future PEA parameterization or establish clinical adequacy. A point/mesh representation may be preferable if fine surface fidelity dominates.

The encoder marks mesh triangle samples at deterministic sub-cell spacing when faces are supplied; with vertices only, it marks the input vertices and makes no claim to reconstruct unobserved surface patches. The decoder returns occupied cell centers, not a mesh. Point/keypoint/depth and adjacent fingertip-distance measurements are available in the reports. Continuous surface-distance and anatomical clearance validation remain outstanding until topology/regions and acceptance thresholds are available.

## Limitations / next evidence needed

- Official evaluation code supports meters as the unit. Projection code supports a camera-relative interpretation, but formal origin, axis signs, and handedness remain unspecified; preserve input XYZ without rotation/reflection unless a defined transformation is approved.
- The 32 MiB tensor-payload and 5-second encoding caps are prototype engineering assumptions; profile model activations and batch storage on target hardware before model integration.
- Quantization bounds are for encoding error only. Use `domain_requirements_review.md` if project scope expands to orthosis fit or clinical claims. Anatomical channel labels preserve landmark identity; fingertip separation does not equal actual surface gap.
- The measurable Person 2/3 criteria and blockers are in `recommended_representation.md`. Current non-voxel candidate scores remain qualitative.
- Validate against camera-derived geometry when that pipeline exists; the current result only covers FreiHAND training annotations.
