# MANO Topology Integration Guide

This document describes how to integrate MANO hand model face topology with FreiHAND geometry for complete surface representation.

## Overview

FreiHAND provides vertex positions but not triangle connectivity (faces). The MANO hand model provides a standard 778-vertex hand mesh with 1,538 triangular faces. This integration loads that face topology to enable:

- Surface voxelization for orthosis design
- Contact surface analysis
- 3D mesh reconstruction and visualization
- Complete geometry export for CAD/manufacturing

## Prerequisites

1. **MANO Model License**: The MANO model requires registration and agreement with the license terms.
   - Visit: https://mano.is.tue.mpg.de/
   - Download `MANO_RIGHT.pkl` (right hand model)
   - Store in a known location on your system

2. **Python Environment**: Same Python environment as FreiHAND (already set up)

## Setup

### Step 1: Download MANO Model

1. Go to https://mano.is.tue.mpg.de/
2. Create an account and agree to the license
3. Download `MANO_RIGHT.pkl` (or `MANO_LEFT.pkl` for left hand)
4. Save to a location like: `C:\path\to\MANO_RIGHT.pkl`

### Step 2: Use in Code

#### Option A: Load FreiHAND Dataset with MANO Topology

```python
from pathlib import Path
from FreiHAND.data_loader import FreiHANDDataset

# Point to your FreiHAND dataset and MANO model
dataset = FreiHANDDataset(
    root=r"C:\path\to\FreiHAND_pub_v2",
    split="training",
    image_version="gs",
    units="m",
    mano_pkl_path=Path(r"C:\path\to\MANO_RIGHT.pkl"),  # Enable MANO topology
)

# Load a sample with face topology
sample = dataset.load_sample(0)

print(f"Sample ID: {sample.sample_id}")
print(f"Vertices shape: {sample.vertices.shape}")  # (778, 3)
print(f"Faces shape: {sample.faces.shape}")        # (1538, 3) - NOW INCLUDED!
print(f"Keypoints shape: {sample.keypoints.shape}")  # (21, 3)
```

#### Option B: Load Geometry NPZ with MANO Topology

```python
from pathlib import Path
from FreiHAND.data_loader import load_geometry_npz

geometry = load_geometry_npz(
    path=Path(r"C:\path\to\hand_geometry.npz"),
    units="m",
    mano_pkl_path=Path(r"C:\path\to\MANO_RIGHT.pkl"),  # Enable MANO topology
)

print(f"Vertices: {geometry.vertices.shape}")
print(f"Faces: {geometry.faces.shape}")
```

#### Option C: Direct MANO Loader (for custom workflows)

```python
from pathlib import Path
from FreiHAND.mano_loader import load_mano_faces, validate_faces_with_vertices

# Load faces from MANO
mano_path = Path(r"C:\path\to\MANO_RIGHT.pkl")
faces = load_mano_faces(mano_path)

print(f"MANO topology: {faces.shape}")  # (1538, 3)

# Validate against your vertex data
is_valid, msg = validate_faces_with_vertices(faces, vertices, sample_id="my_sample")
if is_valid:
    print("Face topology is compatible!")
else:
    print(f"Validation error: {msg}")
```

## Data Format

The MANO face topology provides:

- **Vertices**: 778 per hand (inherited from FreiHAND dataset)
- **Faces**: 1,538 triangles (loaded from MANO)
- **Face Format**: Zero-based vertex indices, shape `(1538, 3)`
- **Vertex Correspondence**: Exact match with FreiHAND's 778-vertex layout

## Validation

The integration includes automatic validation:

```python
# Automatic during dataset loading
try:
    sample = dataset.load_sample(0)
    print("✓ Geometry loaded and validated successfully")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
```

Manual validation is also available:

```python
from FreiHAND.mano_loader import validate_faces_with_vertices

is_valid, msg = validate_faces_with_vertices(
    faces=sample.faces,
    vertices=sample.vertices,
    sample_id=sample.sample_id
)

if is_valid:
    print("✓ Geometry is valid")
else:
    print(f"✗ {msg}")
```

## Troubleshooting

### Error: `FileNotFoundError: MANO model not found`

**Problem**: The path to MANO_RIGHT.pkl is incorrect or the file doesn't exist.

**Solution**:
```python
# Verify the path exists
from pathlib import Path
mano_path = Path(r"C:\path\to\MANO_RIGHT.pkl")
print(f"File exists: {mano_path.is_file()}")

# Use absolute path, not relative
```

### Error: `ValueError: MANO model does not have face topology`

**Problem**: The .pkl file is not a valid MANO model.

**Solution**:
- Download from the official MANO website: https://mano.is.tue.mpg.de/
- Verify the file is `MANO_RIGHT.pkl` or `MANO_LEFT.pkl`
- Check file size (~500 KB for MANO models)

### Error: `Face topology references vertex ... expected maximum ...`

**Problem**: MANO model version mismatch with FreiHAND's 778-vertex layout.

**Solution**:
- Ensure you're using MANO v1.1 or later (standard 778 vertices)
- If using a custom MANO version, verify vertex count matches

## Performance Notes

- **MANO Loading**: One-time operation on dataset initialization (~10-50 ms)
- **Memory**: ~15 KB per loaded sample (faces array)
- **Downstream**: Face topology enables voxelization and surface analysis

## Next Steps

With MANO topology integrated, you can now:

1. **Voxelize** hand geometry including surface contact regions
2. **Extract** keep-clear zones for orthosis design
3. **Analyze** surface curvature and contact pressure
4. **Export** complete mesh for CAD and manufacturing workflows

See `geometry_pipeline.md` for voxelization and normalization workflows.

## References

- **MANO Model**: https://mano.is.tue.mpg.de/
- **FreiHAND Dataset**: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
- **FreiHAND Code**: https://github.com/lmb-freiburg/freihand
