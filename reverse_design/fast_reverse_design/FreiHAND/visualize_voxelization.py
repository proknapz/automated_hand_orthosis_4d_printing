"""Simple visualization of hand surface voxelization.

Generates PNG files showing:
1. Hand mesh with anatomical segmentation
2. Voxelized surface with pressure heatmap
3. Voxelization statistics
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

try:
    from .data_loader import HandGeometry
    from .hand_segmentation import segment_freihand_sample, AnatomicalRegion, SurfaceType
    from .surface_voxelization import voxelize_freihand_sample
except ImportError:
    from data_loader import HandGeometry
    from hand_segmentation import segment_freihand_sample, AnatomicalRegion, SurfaceType
    from surface_voxelization import voxelize_freihand_sample


def create_synthetic_hand_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic hand geometry for visualization."""
    # Palm cluster at origin
    palm = np.random.randn(20, 3) * 0.01
    
    # Fingers extending upward
    thumb = np.random.randn(15, 3) * 0.008 + np.array([0.025, 0.01, 0.00])
    index = np.random.randn(15, 3) * 0.008 + np.array([0.005, 0.04, 0.00])
    middle = np.random.randn(15, 3) * 0.008 + np.array([0.000, 0.05, 0.00])
    ring = np.random.randn(15, 3) * 0.008 + np.array([-0.005, 0.04, 0.00])
    pinky = np.random.randn(15, 3) * 0.008 + np.array([-0.025, 0.03, 0.00])
    
    # Wrist extending downward
    wrist = np.random.randn(20, 3) * 0.01 + np.array([0.00, -0.05, 0.00])
    
    vertices = np.vstack([palm, thumb, index, middle, ring, pinky, wrist])
    
    # Create simple triangulated faces for the mesh
    num_vertices = len(vertices)
    faces = []
    for i in range(num_vertices - 2):
        faces.append([i, i+1, i+2])
    faces = np.array(faces, dtype=np.int64)
    
    # Define 21 keypoints (FreiHAND standard)
    keypoints = np.array([
        [0.0, 0.0, 0.0],        # Wrist
        [0.020, 0.005, 0.0],    # Thumb CMC
        [0.025, 0.010, 0.0],    # Thumb MCP
        [0.028, 0.015, 0.0],    # Thumb PIP
        [0.030, 0.020, 0.0],    # Thumb DIP
        [-0.010, 0.030, 0.0],   # Index CMC
        [-0.005, 0.035, 0.0],   # Index MCP
        [0.000, 0.040, 0.0],    # Index PIP
        [0.005, 0.045, 0.0],    # Index DIP
        [-0.015, 0.040, 0.0],   # Middle CMC
        [-0.010, 0.045, 0.0],   # Middle MCP
        [-0.005, 0.050, 0.0],   # Middle PIP
        [0.000, 0.055, 0.0],    # Middle DIP
        [-0.020, 0.035, 0.0],   # Ring CMC
        [-0.015, 0.040, 0.0],   # Ring MCP
        [-0.010, 0.045, 0.0],   # Ring PIP
        [-0.005, 0.050, 0.0],   # Ring DIP
        [-0.030, 0.025, 0.0],   # Pinky CMC
        [-0.025, 0.030, 0.0],   # Pinky MCP
        [-0.020, 0.035, 0.0],   # Pinky PIP
        [-0.015, 0.040, 0.0],   # Pinky DIP
    ], dtype=np.float64)
    
    return vertices, keypoints, faces


def visualize_hand_geometry(
    vertices: np.ndarray,
    keypoints: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """Create simple 3D visualization of hand geometry."""
    fig = plt.figure(figsize=(14, 6))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='lightblue', s=20, alpha=0.6, label='Vertices')
    ax1.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
               c='red', marker='o', s=100, alpha=0.8, label='Keypoints')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Hand Geometry - 3D View')
    ax1.legend()
    ax1.view_init(elev=20, azim=45)
    
    # Top-down 2D view
    ax2 = fig.add_subplot(122)
    ax2.scatter(vertices[:, 0], vertices[:, 1], c='lightblue', s=20, alpha=0.6, label='Vertices')
    ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='red', marker='o', s=100, alpha=0.8, label='Keypoints')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Hand - Top View')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    plt.close()


def visualize_voxelization(
    voxel_data,
    output_path: Optional[Path] = None,
) -> None:
    """Visualize voxelized hand with pressure weights."""
    fig = plt.figure(figsize=(16, 6))
    
    voxel_positions = voxel_data.voxel_positions
    voxel_weights = voxel_data.voxel_weights
    
    if len(voxel_positions) == 0:
        print("[!] No voxels to visualize")
        return
    
    # Convert grid indices to actual positions
    voxel_coords = voxel_positions.astype(np.float64) * voxel_data.resolution + voxel_data.origin
    
    # 3D scatter with pressure weights
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2],
                         c=voxel_weights, cmap='RdYlGn', s=10, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Voxelized Hand - 3D')
    cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.1)
    cbar1.set_label('Pressure Weight')
    ax1.view_init(elev=20, azim=45)
    
    # Top-down view
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(voxel_coords[:, 0], voxel_coords[:, 1],
                          c=voxel_weights, cmap='RdYlGn', s=20, alpha=0.7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Voxelized Hand - Top View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Pressure Weight')
    
    # Pressure histogram
    ax3 = fig.add_subplot(133)
    ax3.hist(voxel_weights, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Pressure Weight')
    ax3.set_ylabel('Number of Voxels')
    ax3.set_title('Pressure Weight Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    plt.close()


def visualize_voxelization_stats(
    voxel_data,
    output_path: Optional[Path] = None,
) -> None:
    """Visualize voxelization statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats = voxel_data.statistics
    voxel_weights = voxel_data.voxel_weights
    
    # Total voxels pie chart
    ax = axes[0, 0]
    sizes = [stats.surface_voxels, stats.empty_voxels]
    labels = [f'Surface\n({stats.surface_voxels})', f'Empty\n({stats.empty_voxels})']
    colors = ['#66b3ff', '#ffffff']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Voxel Type Distribution')
    
    # Surface type distribution
    ax = axes[0, 1]
    surface_types = {
        'Contact': stats.contact_voxels,
        'Support': stats.support_voxels,
        'Variable': stats.variable_voxels,
        'Keep-Clear': stats.keep_clear_voxels,
    }
    types_list = list(surface_types.keys())
    counts_list = list(surface_types.values())
    colors_bar = ['red', 'orange', 'yellow', 'purple']
    ax.bar(types_list, counts_list, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Voxels')
    ax.set_title('Surface Type Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Pressure weight statistics
    ax = axes[1, 0]
    stats_data = {
        'Min': np.min(voxel_weights),
        'Max': np.max(voxel_weights),
        'Mean': np.mean(voxel_weights),
        'Median': np.median(voxel_weights),
    }
    ax.bar(stats_data.keys(), stats_data.values(), color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Weight Value')
    ax.set_title('Pressure Weight Statistics')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
Voxelization Summary
{'='*45}

Grid Configuration:
  Resolution: {voxel_data.resolution*1000:.1f} mm
  Grid Shape: {voxel_data.shape}
  Grid Origin: ({voxel_data.origin[0]:.4f}, {voxel_data.origin[1]:.4f}, {voxel_data.origin[2]:.4f})

Voxel Count:
  Total Voxels: {stats.total_voxels:,}
  Surface Voxels: {stats.surface_voxels:,}
  Empty Voxels: {stats.empty_voxels:,}
  Sparsity: {100*(1-stats.surface_voxels/stats.total_voxels):.1f}%

Surface Types:
  Contact: {stats.contact_voxels}
  Support: {stats.support_voxels}
  Variable: {stats.variable_voxels}
  Keep-Clear: {stats.keep_clear_voxels}

Pressure Weights:
  Min: {np.min(voxel_weights):.3f}
  Max: {np.max(voxel_weights):.3f}
  Mean: {np.mean(voxel_weights):.3f}
  Std: {np.std(voxel_weights):.3f}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontfamily='monospace',
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
    plt.close()


def main():
    """Generate voxelization visualizations."""
    parser = argparse.ArgumentParser(description='Visualize hand surface voxelization')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--resolution', type=float, default=0.002,
                       help='Voxel resolution in meters (default 2mm)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("[*] Generating synthetic hand data...")
    vertices, keypoints, faces = create_synthetic_hand_data()
    
    print("[*] Creating hand geometry visualization...")
    visualize_hand_geometry(vertices, keypoints,
                           output_path=output_dir / 'voxelization_01_hand_geometry.png')
    
    print("[*] Segmenting hand into anatomical regions...")
    geometry = HandGeometry(
        sample_id='visualization',
        vertices=vertices,
        keypoints=keypoints,
        faces=faces,
        units='m',
    )
    segmentation = segment_freihand_sample(geometry)
    
    print(f"[*] Voxelizing with {args.resolution*1000:.1f}mm resolution...")
    voxel_data = voxelize_freihand_sample(
        vertices=vertices,
        faces=faces,
        keypoints=keypoints,
        resolution=args.resolution,
    )
    
    print("[*] Creating voxelization visualization...")
    visualize_voxelization(voxel_data,
                          output_path=output_dir / 'voxelization_02_pressure_map.png')
    
    print("[*] Creating statistics visualization...")
    visualize_voxelization_stats(voxel_data,
                                output_path=output_dir / 'voxelization_03_statistics.png')
    
    print(f"\n[OK] Visualizations complete!")
    print(f"[*] Output directory: {output_dir}")
    print("Generated files:")
    print("  - voxelization_01_hand_geometry.png")
    print("  - voxelization_02_pressure_map.png")
    print("  - voxelization_03_statistics.png")


if __name__ == '__main__':
    main()
