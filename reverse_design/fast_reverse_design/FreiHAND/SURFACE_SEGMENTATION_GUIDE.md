# Hand Surface Segmentation for Orthosis Design

## Overview

The `hand_segmentation` module provides tools to segment FreiHAND mesh geometry into anatomical regions, enabling orthosis design by:

- **Identifying anatomical regions** (palm, fingers, wrist)
- **Defining contact surfaces** (which parts should touch the orthosis)
- **Defining keep-clear zones** (joints, moving parts that must stay clear)
- **Generating voxel masks** for downstream voxelization

## Quick Start

```python
from FreiHAND.data_loader import FreiHANDDataset
from FreiHAND.hand_segmentation import segment_freihand_sample
from pathlib import Path

# Load a hand sample with MANO topology
dataset = FreiHANDDataset(
    root=r"C:\path\to\FreiHAND_pub_v2",
    mano_pkl_path=Path(r"C:\path\to\MANO_RIGHT.pkl"),
)
sample = dataset.load_sample(0)

# Segment into anatomical regions
segmentation = segment_freihand_sample(sample)

# Get contact mask (vertices that should touch orthosis)
contact_mask = segmentation.get_contact_mask()

# Print summary
print(segmentation.summary())
```

## Anatomical Regions

The segmentation identifies seven anatomical regions:

| Region | Purpose | Typical Contact | Notes |
|--------|---------|-----------------|-------|
| **PALM** | Central hand surface | High contact | Primary support surface |
| **THUMB** | Thumb digit | Grip support | Can be abducted independently |
| **INDEX** | Index finger | Grip support | Includes palm side |
| **MIDDLE** | Middle finger | Support | Longest finger |
| **RING** | Ring finger | Support | Paired with pinky |
| **PINKY** | Pinky finger | Support | Smallest digit |
| **WRIST** | Wrist/forearm junction | Contact/support | Depends on orthosis type |

## Surface Types

Each region can be assigned a surface type:

- **CONTACT**: Should actively touch orthosis (palmar surface, pressure areas)
- **SUPPORT**: Should be held but with minimal pressure (finger sides)
- **KEEP_CLEAR**: Must NOT touch (joints, interior surfaces, sensitive areas)
- **VARIABLE**: Depends on specific orthosis design (flexible assignment)
- **UNKNOWN**: Unclassified

## Basic Usage

### 1. Create Segmentation

```python
from FreiHAND.hand_segmentation import HandSegmentation

segmentation = HandSegmentation(
    vertices=sample.vertices,     # (778, 3)
    keypoints=sample.keypoints,   # (21, 3)
    faces=sample.faces,           # (1538, 3)
)
```

### 2. Segment by Spatial Clustering

```python
# Automatically cluster vertices into anatomical regions
regions = segmentation.segment_by_spatial_clustering()

# Returns: {
#   AnatomicalRegion.PALM: {0, 1, 5, 10, ...},
#   AnatomicalRegion.THUMB: {45, 46, 47, ...},
#   ...
# }
```

### 3. Assign Surface Types

```python
# Use default orthosis assumptions
region_properties = segmentation.assign_surface_types()

# Or customize for specific orthosis type
custom_types = {
    AnatomicalRegion.PALM: SurfaceType.CONTACT,
    AnatomicalRegion.THUMB: SurfaceType.SUPPORT,
    AnatomicalRegion.WRIST: SurfaceType.CONTACT,
    # ... specify for each region
}
region_properties = segmentation.assign_surface_types(custom_types)
```

### 4. Get Masks for Voxelization

```python
# Boolean mask for contact vertices
contact_mask = segmentation.get_contact_mask()  # (778,)

# Boolean mask for keep-clear vertices
keep_clear_mask = segmentation.get_keep_clear_mask()  # (778,)

# Use in voxelization:
# - Mark contact vertices with high priority
# - Exclude keep-clear vertices
# - Variable vertices depend on specific orthosis design
```

### 5. Analyze Regions

```python
# Get region properties
for region, props in region_properties.items():
    print(f"{region.value}:")
    print(f"  Vertices: {len(props.vertex_indices)}")
    print(f"  Surface area: {props.surface_area:.4f} m²")
    print(f"  Surface type: {props.surface_type.value}")
    print(f"  Centroid: {props.centroid}")
```

## Advanced Usage

### Custom Region Mapping

```python
# Define custom surface types based on your orthosis needs
my_orthosis_types = {
    AnatomicalRegion.PALM: SurfaceType.CONTACT,      # Pressure support
    AnatomicalRegion.WRIST: SurfaceType.CONTACT,     # Wrist stabilization
    AnatomicalRegion.THUMB: SurfaceType.KEEP_CLEAR,  # Free thumb
    AnatomicalRegion.INDEX: SurfaceType.SUPPORT,     # Gentle support
    AnatomicalRegion.MIDDLE: SurfaceType.SUPPORT,
    AnatomicalRegion.RING: SurfaceType.SUPPORT,
    AnatomicalRegion.PINKY: SurfaceType.SUPPORT,
}

segmentation.assign_surface_types(my_orthosis_types)
```

### Filter Vertices by Region

```python
# Get all vertices in a specific region
palm_vertices = region_properties[AnatomicalRegion.PALM].vertex_indices
palm_coords = sample.vertices[list(palm_vertices)]

# Get centroids for geometric analysis
for region, props in region_properties.items():
    print(f"{region.value} centroid: {props.centroid}")
```

### Surface Area Analysis

```python
# Get surface areas per region (requires faces)
total_surface_area = sum(props.surface_area 
                        for props in region_properties.values())

for region, props in region_properties.items():
    percentage = 100 * props.surface_area / total_surface_area
    print(f"{region.value}: {percentage:.1f}% of total surface")
```

## Segmentation Algorithm

The spatial clustering approach:

1. **Define finger regions** using FreiHAND keypoint landmarks
   - Use midpoint between metacarpal and fingertip
   - Find all vertices within threshold distance of finger axis

2. **Assign remaining vertices** based on wrist proximity
   - Vertices close to wrist → PALM
   - Vertices far from wrist → WRIST

3. **Calculate surface properties**
   - Centroid: Mean position of all vertices in region
   - Surface area: Sum of triangle areas in region (if faces available)
   - Surface type: Assigned based on anatomical location

## Customization Points

### 1. Adjust Clustering Threshold

The `threshold` parameter (currently 0.03m) controls region separation:

```python
# Edit in hand_segmentation.py, segment_by_spatial_clustering():
threshold = 0.02  # Tighter clustering (smaller regions)
threshold = 0.05  # Looser clustering (larger regions)
```

### 2. Define Custom Keypoint Mapping

```python
# Override KEYPOINT_MAPPING for different hand models
segmentation.KEYPOINT_MAPPING = {
    "wrist": 0,
    "thumb_mcp": 2,
    # ... customize for your hand model
}
```

### 3. Implement Different Clustering Algorithm

The current approach uses geometric proximity. Future enhancements could use:
- Curvature-based segmentation
- Connectivity-based clustering
- Machine learning methods
- Manual annotation framework

## Validation & Testing

All segmentation functionality is covered by 16 unit tests:

```bash
python -m unittest FreiHAND.tests.test_hand_segmentation -v
```

Tests verify:
- Input validation (shapes, dtypes)
- Segmentation completeness (all vertices assigned)
- Region coverage (correct anatomical grouping)
- Surface type assignment
- Mask generation (contact, keep-clear)
- Region property calculation

## Integration with Other Modules

### With MANO Topology
```python
# Segmentation works best with MANO topology
segmentation = HandSegmentation(
    vertices=sample.vertices,
    keypoints=sample.keypoints,
    faces=sample.faces,  # MANO provides complete surface
)
```

### With Voxelization (Phase 3)
```python
# Use contact masks in voxelization
contact_mask = segmentation.get_contact_mask()

# Mark contact regions in voxel grid
for voxel_idx in contact_region_voxels:
    if contact_mask[vertex_to_voxel[voxel_idx]]:
        voxel_type[voxel_idx] = CONTACT_TYPE
```

### With Domain Requirements
```python
# Each region can have specific requirements
requirements = {
    AnatomicalRegion.PALM: {
        "contact_pressure": "high",
        "max_surface_deviation": 5,  # mm
        "required_contact_area": 50,  # cm²
    },
    AnatomicalRegion.WRIST: {
        "contact_pressure": "medium",
        "max_surface_deviation": 3,
    },
}
```

## Performance

- **Segmentation time**: ~50-100 ms per sample
- **Memory**: ~1 MB per sample (vertex labels, region properties)
- **Scalability**: Linear with number of vertices

## Troubleshooting

### Regions are too large or too small

Adjust the clustering threshold:
```python
# In segment_by_spatial_clustering()
threshold = 0.02  # Smaller = tighter regions
```

### Some vertices aren't being assigned

Verify keypoints are correct:
```python
assert sample.keypoints.shape == (21, 3)
assert sample.vertices.shape == (778, 3)
```

### Contact mask is empty or all True

Check the default surface type mapping:
```python
segmentation.assign_surface_types()  # Use defaults
# Or provide explicit mapping
```

## Next Steps

Once segmentation is complete:

1. **Phase 3**: [Voxelization](surface_voxelization.md)
   - Use contact masks for surface-aware voxelization
   - Identify pressure areas in voxel space

2. **Phase 4**: Domain Requirements
   - Validate geometry against clinical requirements
   - Measure surface deviation per region

3. **Phase 5**: Device Registration
   - Align orthosis CAD to segmented hand geometry
   - Calculate fit metrics per region

## References

- **FreiHAND Dataset**: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
- **MANO Hand Model**: https://mano.is.tue.mpg.de/
- **Test Suite**: `FreiHAND/tests/test_hand_segmentation.py`
- **Implementation**: `FreiHAND/hand_segmentation.py`

## API Reference

### HandSegmentation

```python
class HandSegmentation:
    """Segment hand geometry into anatomical regions."""
    
    def __init__(vertices, keypoints, faces=None)
    def segment_by_spatial_clustering() -> Dict[AnatomicalRegion, Set[int]]
    def assign_surface_types(contact_regions=None) -> Dict[AnatomicalRegion, RegionProperties]
    def get_contact_mask() -> NDArray[bool]
    def get_keep_clear_mask() -> NDArray[bool]
    def summary() -> str
```

### Enums

```python
class AnatomicalRegion(Enum):
    PALM, THUMB, INDEX, MIDDLE, RING, PINKY, WRIST, UNKNOWN

class SurfaceType(Enum):
    CONTACT, SUPPORT, KEEP_CLEAR, VARIABLE, UNKNOWN
```

### Data Classes

```python
@dataclass
class RegionProperties:
    region: AnatomicalRegion
    vertex_indices: Set[int]
    face_indices: Set[int]
    centroid: NDArray[float64]
    surface_area: float
    surface_type: SurfaceType
    contact_pressure: Optional[str]
    notes: str
```

## License & Attribution

This module builds on:
- FreiHAND Dataset (Zimmermann et al., ICCV 2019)
- MANO Hand Model (Pavlakos et al., CVPR 2018)

See respective licenses and cite appropriately in research.
