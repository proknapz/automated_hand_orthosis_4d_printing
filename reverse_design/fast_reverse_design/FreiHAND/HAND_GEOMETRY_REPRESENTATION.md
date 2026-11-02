# Hand Geometrical Representation & Foundation

## Overview
The foundation of the 4D-printed orthosis system is a high-fidelity, anatomically aware geometrical representation of the human hand. This module transforms raw 3D hand data into a structured **Voxelized Design Space**, allowing the system to optimize material placement based on clinical needs.

---

## The Geometry Pipeline
The transition from a physical hand to a printable orthosis follows a four-stage transformation process:

### 1. Parametric Surface Modeling (MANO Integration)
To ensure consistency across different patients and poses, the system utilizes a parametric hand model (based on the MANO framework).
- **Input:** Patient-specific hand measurements and pose parameters.
- **Output:** A high-resolution triangular mesh ($\text{Vertices}, \text{Faces}$).
- **Purpose:** Provides a smooth, mathematically defined surface that serves as the ground truth for the orthosis boundaries.

### 2. Anatomical Surface Segmentation
A raw mesh is not enough; the system must "understand" the hand's anatomy to apply different support levels.
- **Process:** The surface is partitioned into functional regions:
    - **Phalangeal Zones:** High-support areas for joint stabilization.
    - **Interphalangeal Gaps:** Movement-preservation zones to allow flexibility.
    - **Palmar Base:** Structural anchor points for the device.
- **Output:** A color-coded segmentation map where each vertex is assigned an anatomical ID.

### 3. Discrete Voxelization
To enable the **Inverse Design Optimization (Phase 4B)**, the continuous surface is converted into a discrete 3D grid (Voxels).
- **Logic:** The system casts a 3D bounding box around the hand and identifies which voxels are "Inside" the hand volume and which are "Outside."
- **Resolution:** Optimized for a balance between fabrication precision and computational efficiency.
- **Output:** A binary 3D tensor $\mathbb{R}^{X \times Y \times Z}$ representing the hand's occupancy.

### 4. Design Space Mapping
The final step creates the "canvas" for the 4D-printing algorithm.
- **Offsetting:** The system generates a thin "shell" of voxels around the hand surface.
- **Constraints:** Voxel indices are mapped to the anatomical segments defined in Step 2.
- **Result:** A set of available coordinates where the optimizer can "turn on" or "turn off" material to create the orthosis.

---

## Technical Specifications

| Component | Method | Key Metric | Purpose |
| :--- | :--- | :--- | :--- |
| **Base Mesh** | MANO / Triangular Mesh | Vertex Density | Anatomical Accuracy |
| **Segmentation** | Region-based Masking | $\text{Zone ID} \in \{1...N\}$ | Clinical Targeting |
| **Voxelization** | Surface-to-Grid Mapping | $\text{Voxel Size (mm)}^3$ | Optimization Grid |
| **Representation** | Binary Occupancy Grid | $\text{Tensors (NumPy)}$ | Inverse Design Input |

---

## Why This Matters (Clinical Impact)

This representation is not just a 3D model; it is a **Clinical Map**. Because the geometry is segmented and voxelized:

1.  **Patient-Specific Fit:** The orthosis is grown directly from the patient's unique hand geometry, eliminating the need for manual casting.
2.  **Targeted Support:** We can tell the optimizer: *"Provide 90% support in the Phalangeal Zone, but only 20% in the Joint Gap,"* ensuring the patient is supported without losing mobility.
3.  **Fabrication Ready:** The voxel grid translates directly into STL triangles, meaning the "Digital Twin" can be sent to a 3D printer without manual cleanup.

---

## Related Documentation
- `FreiHAND/geometry_pipeline.md` $\rightarrow$ Detailed implementation logic.
- `FreiHAND/SURFACE_VOXELIZATION_GUIDE.md` $\rightarrow$ Voxel conversion mathematics.
- `FreiHAND/SURFACE_SEGMENTATION_GUIDE.md` $\rightarrow$ Anatomical zoning rules.
- `PHASE_4C_SUMMARY.md` $\rightarrow$ How this geometry became a final STL product.
