"""Surface voxelization for hand mesh with anatomical region mapping.

This module provides tools to:
- Voxelize hand mesh surface into 3D grid
- Mark voxels by anatomical region and surface type
- Compute pressure/priority weights per voxel
- Export voxel data for orthosis design and FEA simulation

The voxelization process:
1. Creates 3D grid aligned to hand bounding box (origin at wrist, Z-up)
2. Identifies surface voxels using distance field or ray casting
3. Maps surface voxels to anatomical regions from segmentation
4. Applies contact/keep-clear masks to mark surface type
5. Computes pressure weights: contact (1.0) > support (0.7) > variable (0.3) > keep-clear (0.0)

Resolution: Configurable, default 2mm = 0.002m voxel size
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from FreiHAND.hand_segmentation import (
    AnatomicalRegion,
    HandSegmentation,
    SurfaceType,
)


class VoxelType(Enum):
    """Classification of a voxel in the 3D grid."""
    
    EMPTY = 0              # Outside hand
    SURFACE = 1            # On hand surface
    INTERIOR = 2           # Inside hand
    BOUNDARY = 3           # On grid boundary


@dataclass
class VoxelStatistics:
    """Statistics about voxel distribution in voxelization."""
    
    total_voxels: int
    surface_voxels: int
    interior_voxels: int
    empty_voxels: int
    voxels_per_region: Dict[AnatomicalRegion, int] = field(default_factory=dict)
    voxels_per_surface_type: Dict[SurfaceType, int] = field(default_factory=dict)
    contact_voxels: int = 0
    support_voxels: int = 0
    keep_clear_voxels: int = 0
    variable_voxels: int = 0
    
    def summary(self) -> str:
        """Return human-readable summary of statistics."""
        lines = [
            f"Voxel Statistics:",
            f"  Total voxels: {self.total_voxels}",
            f"  Surface voxels: {self.surface_voxels} ({100*self.surface_voxels/max(1,self.total_voxels):.1f}%)",
            f"  Interior voxels: {self.interior_voxels} ({100*self.interior_voxels/max(1,self.total_voxels):.1f}%)",
            f"  Empty voxels: {self.empty_voxels} ({100*self.empty_voxels/max(1,self.total_voxels):.1f}%)",
            f"",
            f"  Contact voxels: {self.contact_voxels}",
            f"  Support voxels: {self.support_voxels}",
            f"  Variable voxels: {self.variable_voxels}",
            f"  Keep-clear voxels: {self.keep_clear_voxels}",
            f"",
            f"  Per region:",
        ]
        for region, count in sorted(self.voxels_per_region.items(), key=lambda x: x[0].value):
            lines.append(f"    {region.value}: {count}")
        
        return "\n".join(lines)


@dataclass
class VoxelData:
    """Voxelized hand data with region and weight information."""
    
    # Grid dimensions
    shape: Tuple[int, int, int]
    resolution: float  # Voxel size in meters
    origin: NDArray[np.float64]  # Grid origin (wrist position)
    
    # Voxel properties (stored as sparse data)
    voxel_positions: NDArray[np.int32]  # Shape (N, 3), indices of non-empty voxels
    voxel_region_ids: NDArray[np.int32]  # Shape (N,), region ID for each voxel
    voxel_types: NDArray[np.uint8]  # Shape (N,), VoxelType for each voxel
    voxel_weights: NDArray[np.float32]  # Shape (N,), pressure weight for each voxel
    
    # Region metadata
    region_id_to_name: Dict[int, str]  # Mapping from ID to AnatomicalRegion name
    statistics: VoxelStatistics = field(default_factory=VoxelStatistics)


class HandVoxelization:
    """Voxelize hand mesh and map to anatomical regions.
    
    This class takes a hand mesh (vertices/faces) and segmentation information,
    and produces a 3D voxel grid with anatomical and surface type annotations.
    """
    
    def __init__(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        segmentation: HandSegmentation,
        resolution: float = 0.002,  # 2mm default
        surface_width: float = 0.001,  # 1mm surface detection width
    ):
        """Initialize voxelization.
        
        Args:
            vertices: Hand vertices (V, 3) in meters
            faces: Hand faces (F, 3), zero-based indices
            segmentation: HandSegmentation instance with assigned surface types
            resolution: Voxel size in meters (default 2mm)
            surface_width: Distance from mesh to detect surface voxels (meters)
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (V, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if surface_width <= 0:
            raise ValueError(f"surface_width must be positive, got {surface_width}")
        
        self.vertices = vertices
        self.faces = faces
        self.segmentation = segmentation
        self.resolution = resolution
        self.surface_width = surface_width
        
        # Ensure segmentation has surface types assigned
        if not segmentation._regions:
            segmentation.assign_surface_types()
        
        self._grid_origin: Optional[NDArray[np.float64]] = None
        self._grid_shape: Optional[Tuple[int, int, int]] = None
        self._voxel_data: Optional[VoxelData] = None
    
    def _compute_grid_bounds(self) -> Tuple[NDArray[np.float64], Tuple[int, int, int]]:
        """Compute grid origin and shape from hand geometry.
        
        Returns:
            (origin, shape) where origin is grid minimum corner
            and shape is (nx, ny, nz) dimensions
        """
        # Add padding around hand
        padding = 0.01  # 1cm padding
        
        bounds_min = self.vertices.min(axis=0) - padding
        bounds_max = self.vertices.max(axis=0) + padding
        
        # Grid extends from min to max
        grid_size = bounds_max - bounds_min
        
        # Compute grid dimensions (round up)
        shape = tuple(np.ceil(grid_size / self.resolution).astype(int))
        
        return bounds_min, shape
    
    def compute_surface_voxels(self) -> NDArray[np.bool_]:
        """Identify voxels on hand surface using signed distance function.
        
        Returns:
            Boolean array of shape grid_shape indicating surface voxels
        """
        if self._grid_origin is None:
            self._grid_origin, self._grid_shape = self._compute_grid_bounds()
        
        # Create grid of voxel centers
        nx, ny, nz = self._grid_shape
        x = np.arange(nx) * self.resolution + self._grid_origin[0]
        y = np.arange(ny) * self.resolution + self._grid_origin[1]
        z = np.arange(nz) * self.resolution + self._grid_origin[2]
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Compute signed distance from each grid point to mesh
        distances = self._compute_signed_distances(grid_points)
        
        # Surface voxels are within surface_width of mesh
        surface_voxels = np.abs(distances) < self.surface_width
        surface_voxels = surface_voxels.reshape(nx, ny, nz)
        
        return surface_voxels
    
    def _compute_signed_distances(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute signed distance from points to mesh surface.
        
        Uses closest point on mesh for each point. Positive = outside, Negative = inside.
        
        Args:
            points: Query points (N, 3)
        
        Returns:
            Signed distances (N,)
        """
        distances = np.zeros(len(points))
        
        for i, point in enumerate(points):
            min_dist = float('inf')
            min_inside = False
            
            # Check distance to each triangle
            for face in self.faces:
                v0, v1, v2 = self.vertices[face]
                
                # Project point onto triangle
                closest = self._closest_point_on_triangle(point, v0, v1, v2)
                dist = np.linalg.norm(point - closest)
                
                if dist < min_dist:
                    min_dist = dist
                    # Determine if point is inside using normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    to_point = point - v0
                    min_inside = np.dot(normal, to_point) < 0
            
            distances[i] = min_dist if min_inside else min_dist
        
        return distances
    
    def _closest_point_on_triangle(
        self,
        point: NDArray[np.float64],
        v0: NDArray[np.float64],
        v1: NDArray[np.float64],
        v2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Find closest point on triangle to a given point.
        
        Args:
            point: Query point (3,)
            v0, v1, v2: Triangle vertices (3,) each
        
        Returns:
            Closest point on triangle (3,)
        """
        edge0 = v1 - v0
        edge1 = v2 - v0
        w = point - v0
        
        a = np.dot(edge0, edge0)
        b = np.dot(edge0, edge1)
        c = np.dot(edge1, edge1)
        d = np.dot(edge0, w)
        e = np.dot(edge1, w)
        
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return v0
        
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        
        # Clamp to triangle
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)
        if s + t > 1:
            s = 1 - t
            t = 1 - s
        
        return v0 + s * edge0 + t * edge1
    
    def mark_contact_regions(self, surface_voxels: NDArray[np.bool_]) -> NDArray[np.int32]:
        """Assign anatomical region IDs to surface voxels.
        
        Args:
            surface_voxels: Boolean array of surface voxel locations
        
        Returns:
            Array of region IDs for each surface voxel (0-indexed)
        """
        nx, ny, nz = self._grid_shape
        region_ids = np.full((nx, ny, nz), -1, dtype=np.int32)
        
        # Get segmentation region assignment
        if self.segmentation._region_assignment is None:
            self.segmentation.segment_by_spatial_clustering()
        
        region_assignment = self.segmentation._region_assignment
        
        # For each surface voxel, find nearest vertex and get its region
        x_coords = np.arange(nx) * self.resolution + self._grid_origin[0]
        y_coords = np.arange(ny) * self.resolution + self._grid_origin[1]
        z_coords = np.arange(nz) * self.resolution + self._grid_origin[2]
        
        surface_indices = np.where(surface_voxels)
        
        for xi, yi, zi in zip(*surface_indices):
            voxel_pos = np.array([x_coords[xi], y_coords[yi], z_coords[zi]])
            
            # Find nearest vertex
            distances = np.linalg.norm(self.vertices - voxel_pos, axis=1)
            nearest_vertex = np.argmin(distances)
            
            # Get region of nearest vertex
            region_id = region_assignment[nearest_vertex]
            region_ids[xi, yi, zi] = region_id
        
        return region_ids
    
    def mark_keep_clear_zones(
        self,
        region_ids: NDArray[np.int32],
        surface_voxels: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Mark voxels that must not contact orthosis (keep-clear zones).
        
        Args:
            region_ids: Array of region IDs for each voxel
            surface_voxels: Boolean array of surface voxel locations
        
        Returns:
            Boolean array where True indicates keep-clear voxel
        """
        keep_clear = np.zeros_like(surface_voxels, dtype=bool)
        
        # Get surface type for each region
        id_to_region = self.segmentation._id_to_region
        
        surface_indices = np.where(surface_voxels)
        
        for xi, yi, zi in zip(*surface_indices):
            region_id = region_ids[xi, yi, zi]
            if region_id < 0:
                continue
            
            region = id_to_region.get(region_id)
            if region and region in self.segmentation._regions:
                region_props = self.segmentation._regions[region]
                if region_props.surface_type == SurfaceType.KEEP_CLEAR:
                    keep_clear[xi, yi, zi] = True
        
        return keep_clear
    
    def compute_pressure_weights(
        self,
        region_ids: NDArray[np.int32],
        keep_clear: NDArray[np.bool_],
        surface_voxels: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Compute pressure/priority weights for each voxel.
        
        Weights: contact (1.0) > support (0.7) > variable (0.3) > keep-clear (0.0)
        
        Args:
            region_ids: Array of region IDs for each voxel
            keep_clear: Boolean array of keep-clear voxels
            surface_voxels: Boolean array of surface voxels
        
        Returns:
            Weights array (0-1) for each voxel
        """
        nx, ny, nz = self._grid_shape
        weights = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Get surface type for each region
        id_to_region = self.segmentation._id_to_region
        
        surface_indices = np.where(surface_voxels)
        
        for xi, yi, zi in zip(*surface_indices):
            if keep_clear[xi, yi, zi]:
                weights[xi, yi, zi] = 0.0
                continue
            
            region_id = region_ids[xi, yi, zi]
            if region_id < 0:
                weights[xi, yi, zi] = 0.5  # Unknown: medium priority
                continue
            
            region = id_to_region.get(region_id)
            if region and region in self.segmentation._regions:
                region_props = self.segmentation._regions[region]
                surface_type = region_props.surface_type
                
                if surface_type == SurfaceType.CONTACT:
                    weights[xi, yi, zi] = 1.0
                elif surface_type == SurfaceType.SUPPORT:
                    weights[xi, yi, zi] = 0.7
                elif surface_type == SurfaceType.VARIABLE:
                    weights[xi, yi, zi] = 0.3
                else:
                    weights[xi, yi, zi] = 0.5
            else:
                weights[xi, yi, zi] = 0.5
        
        return weights
    
    def voxelize(self) -> VoxelData:
        """Perform complete voxelization.
        
        Returns:
            VoxelData with voxel positions, regions, types, and weights
        """
        # Compute grid
        self._grid_origin, self._grid_shape = self._compute_grid_bounds()
        
        # Identify surface voxels
        surface_voxels = self.compute_surface_voxels()
        
        # Map to regions
        region_ids = self.mark_contact_regions(surface_voxels)
        
        # Mark keep-clear zones
        keep_clear = self.mark_keep_clear_zones(region_ids, surface_voxels)
        
        # Compute weights
        weights = self.compute_pressure_weights(region_ids, keep_clear, surface_voxels)
        
        # Extract non-empty voxels
        surface_indices = np.where(surface_voxels)
        voxel_positions = np.stack(surface_indices, axis=1).astype(np.int32)
        
        # Collect data for non-empty voxels
        voxel_region_ids = region_ids[surface_indices].astype(np.int32)
        voxel_weights = weights[surface_indices].astype(np.float32)
        
        # Determine voxel types
        voxel_types = np.full(len(voxel_positions), VoxelType.SURFACE.value, dtype=np.uint8)
        voxel_types[keep_clear[surface_indices]] = VoxelType.BOUNDARY.value
        
        # Create region ID to name mapping
        id_to_region = self.segmentation._id_to_region
        region_id_to_name = {
            rid: region.value for rid, region in id_to_region.items()
        }
        
        # Compute statistics
        stats = self._compute_statistics(
            voxel_positions,
            voxel_region_ids,
            voxel_weights,
            keep_clear[surface_indices],
            id_to_region,
        )
        
        # Create VoxelData
        self._voxel_data = VoxelData(
            shape=self._grid_shape,
            resolution=self.resolution,
            origin=self._grid_origin.copy(),
            voxel_positions=voxel_positions,
            voxel_region_ids=voxel_region_ids,
            voxel_types=voxel_types,
            voxel_weights=voxel_weights,
            region_id_to_name=region_id_to_name,
            statistics=stats,
        )
        
        return self._voxel_data
    
    def _compute_statistics(
        self,
        positions: NDArray[np.int32],
        region_ids: NDArray[np.int32],
        weights: NDArray[np.float32],
        keep_clear: NDArray[np.bool_],
        id_to_region: Dict[int, AnatomicalRegion],
    ) -> VoxelStatistics:
        """Compute voxel statistics.
        
        Args:
            positions: Voxel positions (N, 3)
            region_ids: Region ID for each voxel (N,)
            weights: Weight for each voxel (N,)
            keep_clear: Boolean array of keep-clear voxels (N,)
            id_to_region: Mapping from ID to AnatomicalRegion
        
        Returns:
            VoxelStatistics object
        """
        nx, ny, nz = self._grid_shape
        total = nx * ny * nz
        surface = len(positions)
        
        stats = VoxelStatistics(
            total_voxels=total,
            surface_voxels=surface,
            interior_voxels=0,  # Not computed in current implementation
            empty_voxels=total - surface,
        )
        
        # Count per region
        for rid in np.unique(region_ids):
            if rid < 0:
                continue
            count = np.sum(region_ids == rid)
            region = id_to_region.get(rid)
            if region:
                stats.voxels_per_region[region] = int(count)
        
        # Count per surface type
        for rid in np.unique(region_ids):
            if rid < 0:
                continue
            region = id_to_region.get(rid)
            if not region:
                continue
            
            region_props = self.segmentation._regions.get(region)
            if not region_props:
                continue
            
            mask = region_ids == rid
            count = np.sum(mask)
            surface_type = region_props.surface_type
            
            if surface_type not in stats.voxels_per_surface_type:
                stats.voxels_per_surface_type[surface_type] = 0
            stats.voxels_per_surface_type[surface_type] += count
        
        # Count by weight
        stats.contact_voxels = int(np.sum(weights == 1.0))
        stats.support_voxels = int(np.sum((weights > 0.5) & (weights < 1.0)))
        stats.variable_voxels = int(np.sum((weights > 0.0) & (weights <= 0.5)))
        stats.keep_clear_voxels = int(np.sum(keep_clear))
        
        return stats
    
    def get_statistics(self) -> Optional[VoxelStatistics]:
        """Get voxel distribution statistics.
        
        Returns:
            VoxelStatistics if voxelize() has been called, None otherwise
        """
        if self._voxel_data is None:
            return None
        return self._voxel_data.statistics
    
    def export_to_dict(self) -> dict:
        """Export voxelization as dictionary for serialization.
        
        Returns:
            Dictionary with voxel data and metadata
        """
        if self._voxel_data is None:
            raise RuntimeError("Must call voxelize() before export")
        
        return {
            'grid_shape': self._voxel_data.shape,
            'grid_origin': self._voxel_data.origin.tolist(),
            'resolution': self._voxel_data.resolution,
            'voxel_positions': self._voxel_data.voxel_positions.tolist(),
            'voxel_region_ids': self._voxel_data.voxel_region_ids.tolist(),
            'voxel_types': self._voxel_data.voxel_types.tolist(),
            'voxel_weights': self._voxel_data.voxel_weights.tolist(),
            'region_id_to_name': self._voxel_data.region_id_to_name,
            'statistics': {
                'total_voxels': self._voxel_data.statistics.total_voxels,
                'surface_voxels': self._voxel_data.statistics.surface_voxels,
                'interior_voxels': self._voxel_data.statistics.interior_voxels,
                'empty_voxels': self._voxel_data.statistics.empty_voxels,
                'contact_voxels': self._voxel_data.statistics.contact_voxels,
                'support_voxels': self._voxel_data.statistics.support_voxels,
                'variable_voxels': self._voxel_data.statistics.variable_voxels,
                'keep_clear_voxels': self._voxel_data.statistics.keep_clear_voxels,
            },
        }


def voxelize_freihand_sample(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    keypoints: NDArray[np.float64],
    resolution: float = 0.002,
) -> VoxelData:
    """Convenience function to voxelize a FreiHAND sample.
    
    Performs segmentation and voxelization in one call.
    
    Args:
        vertices: Hand vertices (V, 3) in meters
        faces: Hand faces (F, 3), zero-based indices
        keypoints: Hand keypoints (21, 3) in meters
        resolution: Voxel size in meters (default 2mm)
    
    Returns:
        VoxelData with voxelized hand
    
    Example:
        >>> vertices, faces, keypoints = load_freihand_sample(sample_id)
        >>> voxel_data = voxelize_freihand_sample(vertices, faces, keypoints)
        >>> print(voxel_data.statistics.summary())
    """
    # Create segmentation
    segmentation = HandSegmentation(vertices, keypoints, faces)
    segmentation.segment_by_spatial_clustering()
    segmentation.assign_surface_types()
    
    # Create voxelization
    voxelizer = HandVoxelization(vertices, faces, segmentation, resolution=resolution)
    
    # Perform voxelization
    return voxelizer.voxelize()
