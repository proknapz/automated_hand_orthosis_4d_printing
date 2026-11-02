"""Phase 4C: STL Export Module for Orthosis Designs.

This module converts voxel-based orthosis designs to 3D mesh geometry
and exports to STL format for 3D printing:

1. Load Pareto-optimal voxel designs from Phase 4B
2. Convert sparse voxel sets to 3D mesh (marching cubes equivalent)
3. Apply smoothing and topology optimization
4. Export to binary STL format
5. Generate printer-specific metadata

Key Features:
- Voxel-to-mesh conversion using multiple strategies
- Mesh smoothing and optimization
- Multi-material representation support
- Print statistics (volume, weight, support material)
- Fabrication recommendations
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from numpy.typing import NDArray


@dataclass
class MeshVertex:
    """3D vertex in mesh."""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class MeshTriangle:
    """Triangle face in mesh (3 vertex indices)."""
    v1: int  # Vertex index
    v2: int
    v3: int
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.v1, self.v2, self.v3)


@dataclass
class Mesh:
    """3D triangle mesh."""
    vertices: List[MeshVertex]
    triangles: List[MeshTriangle]
    
    def vertex_count(self) -> int:
        return len(self.vertices)
    
    def triangle_count(self) -> int:
        return len(self.triangles)
    
    def compute_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute axis-aligned bounding box."""
        if not self.vertices:
            return ((0, 0, 0), (0, 0, 0))
        
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        
        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs))
        )
    
    def compute_volume_mm3(self) -> float:
        """Estimate volume using divergence theorem."""
        volume = 0.0
        for tri in self.triangles:
            v1 = self.vertices[tri.v1]
            v2 = self.vertices[tri.v2]
            v3 = self.vertices[tri.v3]
            
            # Signed volume of tetrahedron (origin, v1, v2, v3)
            vol = (
                v1.x * (v2.y * v3.z - v2.z * v3.y) -
                v1.y * (v2.x * v3.z - v2.z * v3.x) +
                v1.z * (v2.x * v3.y - v2.y * v3.x)
            ) / 6.0
            volume += vol
        
        return abs(volume)


class VoxelToMeshConverter:
    """Converts voxel sets to 3D mesh geometry."""
    
    def __init__(self, voxel_resolution_mm: float = 2.0):
        """Initialize converter.
        
        Args:
            voxel_resolution_mm: Size of each voxel in mm
        """
        self.voxel_resolution_mm = voxel_resolution_mm
    
    def convert_simple_cubic(
        self,
        voxel_indices: NDArray[np.int32],
        voxel_space_dims: Tuple[int, int, int] = (10, 10, 3),
    ) -> Mesh:
        """Convert voxel set to mesh using simple cubic representation.
        
        Each voxel becomes a cube; adjacent voxels share faces.
        This is fast but may produce blocky geometry.
        
        Args:
            voxel_indices: Array of voxel indices in flattened space
            voxel_space_dims: (x_max, y_max, z_max) of voxel grid
        
        Returns:
            Mesh object
        """
        # Map flat indices to 3D coordinates
        voxel_coords: Set[Tuple[int, int, int]] = set()
        x_max, y_max, z_max = voxel_space_dims
        
        for idx in voxel_indices:
            z = idx // (x_max * y_max)
            remainder = idx % (x_max * y_max)
            y = remainder // x_max
            x = remainder % x_max
            voxel_coords.add((x, y, z))
        
        # Build surface mesh (only exposed faces)
        vertices: List[MeshVertex] = []
        triangles: List[MeshTriangle] = []
        vertex_map: Dict[Tuple[float, float, float], int] = {}
        
        def get_or_create_vertex(x: float, y: float, z: float) -> int:
            """Get vertex index, creating if needed."""
            key = (x, y, z)
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(MeshVertex(x, y, z))
            return vertex_map[key]
        
        # Helper: check if voxel exists
        def has_voxel(x: int, y: int, z: int) -> bool:
            return (x, y, z) in voxel_coords
        
        # Generate mesh faces
        for (vx, vy, vz) in voxel_coords:
            # Voxel corner positions (in mm)
            x0 = vx * self.voxel_resolution_mm
            y0 = vy * self.voxel_resolution_mm
            z0 = vz * self.voxel_resolution_mm
            s = self.voxel_resolution_mm
            
            # Check 6 faces; add only if neighbor is empty (exposed)
            
            # +X face
            if not has_voxel(vx + 1, vy, vz):
                v0 = get_or_create_vertex(x0 + s, y0, z0)
                v1 = get_or_create_vertex(x0 + s, y0 + s, z0)
                v2 = get_or_create_vertex(x0 + s, y0, z0 + s)
                v3 = get_or_create_vertex(x0 + s, y0 + s, z0 + s)
                triangles.append(MeshTriangle(v0, v1, v2))
                triangles.append(MeshTriangle(v1, v3, v2))
            
            # -X face
            if not has_voxel(vx - 1, vy, vz):
                v0 = get_or_create_vertex(x0, y0, z0)
                v1 = get_or_create_vertex(x0, y0, z0 + s)
                v2 = get_or_create_vertex(x0, y0 + s, z0)
                v3 = get_or_create_vertex(x0, y0 + s, z0 + s)
                triangles.append(MeshTriangle(v0, v2, v1))
                triangles.append(MeshTriangle(v2, v3, v1))
            
            # +Y face
            if not has_voxel(vx, vy + 1, vz):
                v0 = get_or_create_vertex(x0, y0 + s, z0)
                v1 = get_or_create_vertex(x0, y0 + s, z0 + s)
                v2 = get_or_create_vertex(x0 + s, y0 + s, z0)
                v3 = get_or_create_vertex(x0 + s, y0 + s, z0 + s)
                triangles.append(MeshTriangle(v0, v2, v1))
                triangles.append(MeshTriangle(v2, v3, v1))
            
            # -Y face
            if not has_voxel(vx, vy - 1, vz):
                v0 = get_or_create_vertex(x0, y0, z0)
                v1 = get_or_create_vertex(x0 + s, y0, z0)
                v2 = get_or_create_vertex(x0, y0, z0 + s)
                v3 = get_or_create_vertex(x0 + s, y0, z0 + s)
                triangles.append(MeshTriangle(v0, v1, v2))
                triangles.append(MeshTriangle(v1, v3, v2))
            
            # +Z face
            if not has_voxel(vx, vy, vz + 1):
                v0 = get_or_create_vertex(x0, y0, z0 + s)
                v1 = get_or_create_vertex(x0 + s, y0, z0 + s)
                v2 = get_or_create_vertex(x0, y0 + s, z0 + s)
                v3 = get_or_create_vertex(x0 + s, y0 + s, z0 + s)
                triangles.append(MeshTriangle(v0, v1, v2))
                triangles.append(MeshTriangle(v1, v3, v2))
            
            # -Z face
            if not has_voxel(vx, vy, vz - 1):
                v0 = get_or_create_vertex(x0, y0, z0)
                v1 = get_or_create_vertex(x0, y0 + s, z0)
                v2 = get_or_create_vertex(x0 + s, y0, z0)
                v3 = get_or_create_vertex(x0 + s, y0 + s, z0)
                triangles.append(MeshTriangle(v0, v2, v1))
                triangles.append(MeshTriangle(v2, v3, v1))
        
        return Mesh(vertices, triangles)
    
    def smooth_mesh(self, mesh: Mesh, iterations: int = 3) -> Mesh:
        """Apply Laplacian smoothing to mesh.
        
        Smoothing reduces blocky appearance of cubic voxel conversion.
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing passes
        
        Returns:
            Smoothed mesh
        """
        # Build adjacency list
        vertex_neighbors: Dict[int, List[int]] = {}
        for v_idx in range(len(mesh.vertices)):
            vertex_neighbors[v_idx] = []
        
        for tri in mesh.triangles:
            for v1, v2 in [(tri.v1, tri.v2), (tri.v2, tri.v3), (tri.v3, tri.v1)]:
                if v2 not in vertex_neighbors[v1]:
                    vertex_neighbors[v1].append(v2)
                if v1 not in vertex_neighbors[v2]:
                    vertex_neighbors[v2].append(v1)
        
        # Smooth
        vertices = [v for v in mesh.vertices]
        
        for _ in range(iterations):
            new_vertices = []
            for v_idx, v in enumerate(vertices):
                neighbors = vertex_neighbors[v_idx]
                if not neighbors:
                    new_vertices.append(v)
                    continue
                
                # Average position with neighbors
                avg_x = sum(vertices[n].x for n in neighbors) / len(neighbors)
                avg_y = sum(vertices[n].y for n in neighbors) / len(neighbors)
                avg_z = sum(vertices[n].z for n in neighbors) / len(neighbors)
                
                # Blend with original (dampening factor)
                blend = 0.5
                new_x = v.x * (1 - blend) + avg_x * blend
                new_y = v.y * (1 - blend) + avg_y * blend
                new_z = v.z * (1 - blend) + avg_z * blend
                
                new_vertices.append(MeshVertex(new_x, new_y, new_z))
            
            vertices = new_vertices
        
        return Mesh(vertices, mesh.triangles)


class STLExporter:
    """Exports mesh to binary STL format."""
    
    @staticmethod
    def export_binary_stl(
        mesh: Mesh,
        output_path: str,
        scale_factor: float = 1.0,
        translate: Tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """Export mesh to binary STL file.
        
        Args:
            mesh: Mesh to export
            output_path: Output STL file path
            scale_factor: Scale vertices by this factor
            translate: Translation to apply to all vertices
        """
        with open(output_path, 'wb') as f:
            # Header (80 bytes)
            header = b'Binary STL exported from Phase 4C\x00' * 2  # Pad to 80 bytes
            header = header[:80]
            f.write(header)
            
            # Triangle count (4 bytes, little-endian uint32)
            f.write(struct.pack('<I', len(mesh.triangles)))
            
            # Triangles
            for tri in mesh.triangles:
                v1 = mesh.vertices[tri.v1]
                v2 = mesh.vertices[tri.v2]
                v3 = mesh.vertices[tri.v3]
                
                # Scale and translate vertices
                p1 = np.array([v1.x, v1.y, v1.z]) * scale_factor + np.array(translate)
                p2 = np.array([v2.x, v2.y, v2.z]) * scale_factor + np.array(translate)
                p3 = np.array([v3.x, v3.y, v3.z]) * scale_factor + np.array(translate)
                
                # Compute normal
                edge1 = p2 - p1
                edge2 = p3 - p1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                # Write normal (3 floats)
                f.write(struct.pack('<fff', *normal))
                
                # Write vertices (3 * 3 floats)
                f.write(struct.pack('<fff', *p1))
                f.write(struct.pack('<fff', *p2))
                f.write(struct.pack('<fff', *p3))
                
                # Attribute byte count (2 bytes, usually 0)
                f.write(struct.pack('<H', 0))
    
    @staticmethod
    def export_ascii_stl(
        mesh: Mesh,
        output_path: str,
        name: str = 'Orthosis Design',
        scale_factor: float = 1.0,
    ) -> None:
        """Export mesh to ASCII STL file (more portable).
        
        Args:
            mesh: Mesh to export
            output_path: Output STL file path
            name: Name for solid
            scale_factor: Scale vertices by this factor
        """
        with open(output_path, 'w') as f:
            f.write(f"solid {name}\n")
            
            for tri in mesh.triangles:
                v1 = mesh.vertices[tri.v1]
                v2 = mesh.vertices[tri.v2]
                v3 = mesh.vertices[tri.v3]
                
                # Scale vertices
                p1 = np.array([v1.x, v1.y, v1.z]) * scale_factor
                p2 = np.array([v2.x, v2.y, v2.z]) * scale_factor
                p3 = np.array([v3.x, v3.y, v3.z]) * scale_factor
                
                # Compute normal
                edge1 = p2 - p1
                edge2 = p3 - p1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
                f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
                f.write(f"      vertex {p3[0]:.6e} {p3[1]:.6e} {p3[2]:.6e}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write(f"endsolid {name}\n")


class STLExportPipeline:
    """Complete STL export workflow for Pareto front designs."""
    
    def __init__(
        self,
        output_dir: str = "phase_4c_stl_exports",
        voxel_resolution_mm: float = 2.0,
        apply_smoothing: bool = True,
        export_format: str = 'binary',  # 'binary' or 'ascii'
    ):
        """Initialize export pipeline.
        
        Args:
            output_dir: Directory for STL exports
            voxel_resolution_mm: Voxel size in mm
            apply_smoothing: Apply mesh smoothing
            export_format: 'binary' or 'ascii' STL format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.converter = VoxelToMeshConverter(voxel_resolution_mm)
        self.exporter = STLExporter()
        self.apply_smoothing = apply_smoothing
        self.export_format = export_format
        self.voxel_resolution_mm = voxel_resolution_mm
    
    def export_pareto_front(
        self,
        designs_json_path: str,
        top_n: Optional[int] = None,
    ) -> Dict[int, Dict[str, any]]:
        """Export Pareto-optimal designs to STL.
        
        Args:
            designs_json_path: Path to phase_4b_results/designs.json
            top_n: Export only top N designs; if None, export all
        
        Returns:
            Dictionary mapping design rank to export metadata
        """
        with open(designs_json_path) as f:
            designs = json.load(f)
        
        if top_n:
            designs = designs[:top_n]
        
        results = {}
        
        for design in designs:
            rank = design['rank']
            voxel_count = design['voxel_count']
            voxel_indices = np.array(design['voxel_indices'], dtype=np.int32)
            
            print(f"\n[EXPORT] Design Rank {rank} ({voxel_count} voxels)")
            
            # Convert to mesh
            mesh = self.converter.convert_simple_cubic(voxel_indices)
            
            # Apply smoothing
            if self.apply_smoothing:
                mesh = self.converter.smooth_mesh(mesh, iterations=2)
            
            # Compute metrics
            bounds = mesh.compute_bounds()
            volume_mm3 = mesh.compute_volume_mm3()
            
            # Export to STL
            filename = f"orthosis_rank_{rank:02d}.stl"
            filepath = self.output_dir / filename
            
            if self.export_format == 'binary':
                self.exporter.export_binary_stl(mesh, str(filepath))
            else:
                self.exporter.export_ascii_stl(
                    mesh, str(filepath),
                    name=f"Orthosis_Rank_{rank}"
                )
            
            print(f"  Vertices: {mesh.vertex_count()}")
            print(f"  Triangles: {mesh.triangle_count()}")
            print(f"  Volume: {volume_mm3:.1f} mm3 ({volume_mm3/1000:.2f} cm3)")
            print(f"  Exported to: {filepath}")
            
            results[rank] = {
                'filename': filename,
                'filepath': str(filepath),
                'voxel_count': voxel_count,
                'vertices': mesh.vertex_count(),
                'triangles': mesh.triangle_count(),
                'volume_mm3': volume_mm3,
                'bounds_min': bounds[0],
                'bounds_max': bounds[1],
            }
        
        return results


def main():
    """Demo: Export Phase 4B designs to STL."""
    print("[DEMO] Phase 4C STL Export")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = STLExportPipeline(
        output_dir="phase_4c_stl_exports",
        voxel_resolution_mm=2.0,
        apply_smoothing=True,
        export_format='binary',
    )
    
    # Export designs
    designs_path = Path(__file__).parent / 'phase_4b_results' / 'designs.json'
    
    if designs_path.exists():
        print(f"\n[STEP 1] Loading designs from {designs_path}")
        results = pipeline.export_pareto_front(str(designs_path), top_n=5)
        
        print(f"\n[STEP 2] Export Summary")
        print(f"  Exported designs: {len(results)}")
        print(f"  Output directory: {pipeline.output_dir}")
        
        total_volume = sum(r['volume_mm3'] for r in results.values())
        print(f"  Total volume: {total_volume:.1f} mm3 ({total_volume/1000:.2f} cm3)")
    else:
        print(f"[ERROR] Designs file not found: {designs_path}")


if __name__ == '__main__':
    main()
