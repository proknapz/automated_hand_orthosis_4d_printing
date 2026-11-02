"""Hand surface segmentation and anatomical region labeling.

This module provides tools to:
- Segment FreiHAND mesh into anatomical regions (palm, fingers, etc.)
- Define contact and keep-clear zones for orthosis design
- Analyze surface properties per region
- Generate contact region masks for voxelization

Anatomical Regions:
- Palmar surface: Contact surface for grip/support
- Dorsal surface: Top of hand
- Thenar region: Thumb and base
- Hypothenar region: Pinky and base
- Metacarpal region: Back of hand
- Finger regions: Individual finger segments
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class AnatomicalRegion(Enum):
    """Anatomical regions of the hand."""
    
    PALM = "palm"
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
    WRIST = "wrist"
    UNKNOWN = "unknown"


class SurfaceType(Enum):
    """Type of hand surface."""
    
    CONTACT = "contact"       # Should touch orthosis (palmar, pressure areas)
    SUPPORT = "support"       # Should be held but minimal pressure
    KEEP_CLEAR = "keep_clear"  # Must not touch (joints, moving parts)
    VARIABLE = "variable"     # Depends on orthosis design
    UNKNOWN = "unknown"


@dataclass
class RegionProperties:
    """Properties of an anatomical region."""
    
    region: AnatomicalRegion
    vertex_indices: Set[int]
    face_indices: Set[int]
    centroid: NDArray[np.float64]
    surface_area: float
    surface_type: SurfaceType
    contact_pressure: Optional[str] = None  # e.g., "high", "medium", "low"
    notes: str = ""


class HandSegmentation:
    """Segment FreiHAND mesh into anatomical regions.
    
    Uses a combination of:
    - Spatial clustering based on hand keypoints
    - Geometric features (curvature, connectivity)
    - User-defined anatomical landmarks
    """
    
    # FreiHAND keypoint indices for anatomical landmarks
    # From https://github.com/lmb-freiburg/freihand
    KEYPOINT_MAPPING = {
        "wrist": 0,
        "palm": 9,  # Middle finger proximal phalanx
        "thumb_mcp": 2,      # Thumb metacarpal
        "thumb_tip": 4,      # Thumb tip
        "index_mcp": 5,      # Index metacarpal
        "index_tip": 8,      # Index tip
        "middle_mcp": 9,     # Middle metacarpal
        "middle_tip": 12,    # Middle tip
        "ring_mcp": 13,      # Ring metacarpal
        "ring_tip": 16,      # Ring tip
        "pinky_mcp": 17,     # Pinky metacarpal
        "pinky_tip": 20,     # Pinky tip
    }
    
    def __init__(
        self,
        vertices: NDArray[np.float64],
        keypoints: NDArray[np.float64],
        faces: Optional[NDArray[np.int64]] = None,
    ):
        """Initialize segmentation.
        
        Args:
            vertices: Hand vertices (V, 3)
            keypoints: Hand keypoints (21, 3)
            faces: Hand faces (F, 3), optional
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (V, 3), got {vertices.shape}")
        if keypoints.ndim != 2 or keypoints.shape != (21, 3):
            raise ValueError(f"keypoints must have shape (21, 3), got {keypoints.shape}")
        if faces is not None:
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
        
        self.vertices = vertices
        self.keypoints = keypoints
        self.faces = faces
        self._region_assignment: Optional[NDArray[np.int32]] = None
        self._region_to_id: Dict[AnatomicalRegion, int] = {}
        self._id_to_region: Dict[int, AnatomicalRegion] = {}
        self._regions: Dict[AnatomicalRegion, RegionProperties] = {}
    
    def segment_by_spatial_clustering(self) -> Dict[AnatomicalRegion, Set[int]]:
        """Segment hand using spatial clustering around keypoints.
        
        Returns:
            Dictionary mapping anatomical regions to vertex indices
        """
        regions = {}
        
        # Create mapping from region to numeric ID for storage
        region_to_id = {
            AnatomicalRegion.PALM: 0,
            AnatomicalRegion.THUMB: 1,
            AnatomicalRegion.INDEX: 2,
            AnatomicalRegion.MIDDLE: 3,
            AnatomicalRegion.RING: 4,
            AnatomicalRegion.PINKY: 5,
            AnatomicalRegion.WRIST: 6,
        }
        id_to_region = {v: k for k, v in region_to_id.items()}
        
        vertex_labels = np.full(self.vertices.shape[0], -1, dtype=np.int32)
        
        # Distance threshold for assigning vertices to regions (meters)
        # Adjust based on hand size (typical hand: 0.15-0.20m palm width)
        threshold = 0.03  # 3 cm
        
        # Define finger regions using keypoint ranges
        finger_regions = [
            (AnatomicalRegion.THUMB, [self.KEYPOINT_MAPPING["thumb_mcp"], 
                                     self.KEYPOINT_MAPPING["thumb_tip"]]),
            (AnatomicalRegion.INDEX, [self.KEYPOINT_MAPPING["index_mcp"],
                                     self.KEYPOINT_MAPPING["index_tip"]]),
            (AnatomicalRegion.MIDDLE, [self.KEYPOINT_MAPPING["middle_mcp"],
                                      self.KEYPOINT_MAPPING["middle_tip"]]),
            (AnatomicalRegion.RING, [self.KEYPOINT_MAPPING["ring_mcp"],
                                    self.KEYPOINT_MAPPING["ring_tip"]]),
            (AnatomicalRegion.PINKY, [self.KEYPOINT_MAPPING["pinky_mcp"],
                                     self.KEYPOINT_MAPPING["pinky_tip"]]),
        ]
        
        # Assign finger vertices
        for region_name, (start_kp, end_kp) in finger_regions:
            start_point = self.keypoints[start_kp]
            end_point = self.keypoints[end_kp]
            
            # Find vertices close to the finger axis
            distances = self._point_to_segment_distances(
                self.vertices, start_point, end_point
            )
            finger_vertices = np.where(distances < threshold)[0]
            
            vertex_labels[finger_vertices] = region_to_id[region_name]
            regions[region_name] = set(finger_vertices.tolist())
        
        # Assign palm/remaining vertices
        unassigned = np.where(vertex_labels == -1)[0]
        wrist_point = self.keypoints[self.KEYPOINT_MAPPING["wrist"]]
        
        # Vertices closest to wrist are palm
        distances = np.linalg.norm(self.vertices[unassigned] - wrist_point, axis=1)
        palm_threshold = np.percentile(distances, 70)  # Top 70% closest to wrist
        
        palm_mask = distances < palm_threshold
        palm_vertices = unassigned[palm_mask]
        
        vertex_labels[palm_vertices] = region_to_id[AnatomicalRegion.PALM]
        regions[AnatomicalRegion.PALM] = set(palm_vertices.tolist())
        
        # Remaining unassigned are labeled as wrist
        remaining = unassigned[~palm_mask]
        vertex_labels[remaining] = region_to_id[AnatomicalRegion.WRIST]
        regions[AnatomicalRegion.WRIST] = set(remaining.tolist())
        
        # Store both the labels and the mapping
        self._region_assignment = vertex_labels
        self._region_to_id = region_to_id
        self._id_to_region = id_to_region
        
        return regions
    
    def assign_surface_types(
        self,
        contact_regions: Optional[Dict[AnatomicalRegion, SurfaceType]] = None,
    ) -> Dict[AnatomicalRegion, RegionProperties]:
        """Assign surface types (contact, keep-clear, etc.) to regions.
        
        Args:
            contact_regions: Custom mapping of regions to surface types.
                           If None, uses default orthosis design assumptions.
        
        Returns:
            Dictionary with region properties including surface type
        """
        if contact_regions is None:
            # Default orthosis assumptions
            contact_regions = {
                AnatomicalRegion.PALM: SurfaceType.CONTACT,
                AnatomicalRegion.THUMB: SurfaceType.SUPPORT,
                AnatomicalRegion.INDEX: SurfaceType.SUPPORT,
                AnatomicalRegion.MIDDLE: SurfaceType.SUPPORT,
                AnatomicalRegion.RING: SurfaceType.SUPPORT,
                AnatomicalRegion.PINKY: SurfaceType.SUPPORT,
                AnatomicalRegion.WRIST: SurfaceType.CONTACT,
            }
        
        if self._region_assignment is None:
            self.segment_by_spatial_clustering()
        
        self._regions = {}
        
        for region, vertices in self._get_region_vertices().items():
            if not vertices:
                continue
            
            vertices_array = np.array(sorted(list(vertices)))
            centroid = self.vertices[vertices_array].mean(axis=0)
            
            # Calculate surface area if faces available
            surface_area = 0.0
            face_indices: Set[int] = set()
            
            if self.faces is not None:
                for face_idx, face in enumerate(self.faces):
                    if all(v in vertices for v in face):
                        face_indices.add(face_idx)
                        # Calculate triangle area (Heron's formula)
                        v0, v1, v2 = self.vertices[face]
                        a = np.linalg.norm(v1 - v0)
                        b = np.linalg.norm(v2 - v1)
                        c = np.linalg.norm(v0 - v2)
                        s = (a + b + c) / 2
                        if s > 0:
                            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                            surface_area += area
            
            surface_type = contact_regions.get(region, SurfaceType.UNKNOWN)
            
            self._regions[region] = RegionProperties(
                region=region,
                vertex_indices=vertices,
                face_indices=face_indices,
                centroid=centroid,
                surface_area=surface_area,
                surface_type=surface_type,
            )
        
        return self._regions
    
    def get_contact_mask(self) -> NDArray[np.bool_]:
        """Get boolean mask of vertices that should contact orthosis.
        
        Returns:
            Boolean array (V,) where True indicates contact vertex
        """
        if not self._regions:
            self.assign_surface_types()
        
        mask = np.zeros(self.vertices.shape[0], dtype=bool)
        
        for region_props in self._regions.values():
            if region_props.surface_type == SurfaceType.CONTACT:
                for vertex_idx in region_props.vertex_indices:
                    mask[vertex_idx] = True
        
        return mask
    
    def get_keep_clear_mask(self) -> NDArray[np.bool_]:
        """Get boolean mask of vertices that must NOT touch orthosis.
        
        Returns:
            Boolean array (V,) where True indicates keep-clear vertex
        """
        if not self._regions:
            self.assign_surface_types()
        
        mask = np.zeros(self.vertices.shape[0], dtype=bool)
        
        for region_props in self._regions.values():
            if region_props.surface_type == SurfaceType.KEEP_CLEAR:
                for vertex_idx in region_props.vertex_indices:
                    mask[vertex_idx] = True
        
        return mask
    
    def summary(self) -> str:
        """Get human-readable summary of segmentation."""
        if not self._regions:
            return "No segmentation performed yet"
        
        lines = ["Hand Surface Segmentation Summary", "=" * 50]
        
        for region, props in sorted(self._regions.items(), 
                                   key=lambda x: x[0].value):
            lines.append(
                f"\n{region.value.upper()}:"
                f"\n  Vertices: {len(props.vertex_indices)}"
                f"\n  Faces: {len(props.face_indices)}"
                f"\n  Surface area: {props.surface_area:.4f} m²"
                f"\n  Surface type: {props.surface_type.value}"
                f"\n  Centroid: {props.centroid}"
            )
        
        return "\n".join(lines)
    
    # Helper methods
    
    def _get_region_vertices(self) -> Dict[AnatomicalRegion, Set[int]]:
        """Get vertex indices for each region."""
        if self._region_assignment is None:
            raise ValueError("No segmentation performed. Call segment_by_spatial_clustering first.")
        
        regions = {}
        for region_enum, region_id in getattr(self, '_region_to_id', {}).items():
            vertices = np.where(self._region_assignment == region_id)[0]
            regions[region_enum] = set(vertices.tolist())
        
        return regions
    
    @staticmethod
    def _point_to_segment_distances(
        points: NDArray[np.float64],
        segment_start: NDArray[np.float64],
        segment_end: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate distance from points to line segment.
        
        Args:
            points: Points to measure from (N, 3)
            segment_start: Start of line segment (3,)
            segment_end: End of line segment (3,)
        
        Returns:
            Distances (N,)
        """
        # Vector from start to end
        segment_vec = segment_end - segment_start
        segment_length_sq = np.sum(segment_vec ** 2)
        
        # Vector from start to each point
        point_vecs = points - segment_start
        
        # Project each point onto the segment
        t = np.dot(point_vecs, segment_vec) / segment_length_sq
        t = np.clip(t, 0, 1)
        
        # Closest point on segment
        closest_points = segment_start + t[:, np.newaxis] * segment_vec
        
        # Distance
        distances = np.linalg.norm(points - closest_points, axis=1)
        
        return distances


def segment_freihand_sample(geometry) -> HandSegmentation:
    """Convenience function to segment a FreiHAND sample.
    
    Args:
        geometry: FreiHAND HandGeometry object (from data_loader)
    
    Returns:
        HandSegmentation object with regions assigned
    """
    segmentation = HandSegmentation(
        vertices=geometry.vertices,
        keypoints=geometry.keypoints,
        faces=geometry.faces,
    )
    segmentation.assign_surface_types()
    return segmentation
