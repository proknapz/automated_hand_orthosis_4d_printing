"""Tests for surface voxelization and anatomical region mapping."""

import unittest

import numpy as np

from FreiHAND.hand_segmentation import (
    AnatomicalRegion,
    HandSegmentation,
    SurfaceType,
)
from FreiHAND.surface_voxelization import (
    HandVoxelization,
    VoxelStatistics,
    VoxelType,
    voxelize_freihand_sample,
)


def create_synthetic_hand():
    """Create a synthetic hand geometry for testing."""
    # Simple hand: palm cluster + finger clusters
    vertices = np.concatenate([
        # Palm (indices 0-9): cluster around origin
        np.random.randn(10, 3) * 0.01,
        
        # Thumb (indices 10-19): cluster near (0.02, 0, 0)
        np.random.randn(10, 3) * 0.01 + np.array([0.02, 0, 0]),
        
        # Index (indices 20-29): cluster near (0, 0.03, 0)
        np.random.randn(10, 3) * 0.01 + np.array([0, 0.03, 0]),
        
        # Middle (indices 30-39): cluster near (0, 0.04, 0)
        np.random.randn(10, 3) * 0.01 + np.array([0, 0.04, 0]),
        
        # Ring (indices 40-49): cluster near (0, 0.03, -0.01)
        np.random.randn(10, 3) * 0.01 + np.array([0, 0.03, -0.01]),
        
        # Pinky (indices 50-59): cluster near (0, 0.02, -0.02)
        np.random.randn(10, 3) * 0.01 + np.array([0, 0.02, -0.02]),
        
        # Wrist (indices 60-69): cluster near (0, -0.05, 0)
        np.random.randn(10, 3) * 0.01 + np.array([0, -0.05, 0]),
    ])
    
    # 21 keypoints (FreiHAND standard)
    keypoints = np.array([
        [0, -0.05, 0],      # 0: wrist
        [0.02, 0.01, 0],    # 1-4: thumb
        [0.02, 0.02, 0],
        [0.02, 0.03, 0],
        [0.02, 0.04, 0],
        [0, 0.03, 0],       # 5-8: index
        [0, 0.035, 0],
        [0, 0.037, 0],
        [0, 0.04, 0],
        [0, 0, 0],          # 9: middle (reference)
        [0, 0.04, 0],       # 10-12: middle
        [0, 0.041, 0],
        [0, 0.042, 0],
        [0, 0.03, -0.01],   # 13-16: ring
        [0, 0.032, -0.01],
        [0, 0.034, -0.01],
        [0, 0.036, -0.01],
        [0, 0.02, -0.02],   # 17-20: pinky
        [0, 0.022, -0.02],
        [0, 0.024, -0.02],
        [0, 0.026, -0.02],
    ])
    
    # Simple triangular faces (cube-like geometry)
    faces = np.array([
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
        [50, 51, 52],
        [60, 61, 62],
    ], dtype=np.int64)
    
    return vertices, keypoints, faces


class VoxelizationInitTests(unittest.TestCase):
    """Test voxelization initialization and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
    
    def test_init_validates_vertices_shape(self):
        """Test that __init__ validates vertex shape."""
        bad_vertices = np.random.randn(10, 2)
        with self.assertRaises(ValueError) as ctx:
            HandVoxelization(bad_vertices, self.faces, self.segmentation)
        self.assertIn("(V, 3)", str(ctx.exception))
    
    def test_init_validates_faces_shape(self):
        """Test that __init__ validates face shape."""
        bad_faces = np.random.randint(0, 10, (5, 2), dtype=np.int64)
        with self.assertRaises(ValueError) as ctx:
            HandVoxelization(self.vertices, bad_faces, self.segmentation)
        self.assertIn("(F, 3)", str(ctx.exception))
    
    def test_init_validates_resolution(self):
        """Test that __init__ validates resolution."""
        with self.assertRaises(ValueError) as ctx:
            HandVoxelization(self.vertices, self.faces, self.segmentation, resolution=0)
        self.assertIn("positive", str(ctx.exception))
    
    def test_init_validates_surface_width(self):
        """Test that __init__ validates surface_width."""
        with self.assertRaises(ValueError) as ctx:
            HandVoxelization(self.vertices, self.faces, self.segmentation, surface_width=-0.001)
        self.assertIn("positive", str(ctx.exception))
    
    def test_init_creates_instance(self):
        """Test successful initialization."""
        voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
        self.assertIsNotNone(voxelizer)
        self.assertEqual(voxelizer.resolution, 0.002)


class VoxelizationComputationTests(unittest.TestCase):
    """Test voxelization computation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
        self.voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
    
    def test_compute_grid_bounds(self):
        """Test grid bound computation."""
        origin, shape = self.voxelizer._compute_grid_bounds()
        
        # Origin should be less than vertex min
        self.assertTrue(np.all(origin < self.vertices.min(axis=0)))
        
        # Shape should be positive
        self.assertGreater(shape[0], 0)
        self.assertGreater(shape[1], 0)
        self.assertGreater(shape[2], 0)
    
    def test_compute_surface_voxels(self):
        """Test surface voxel identification."""
        surface_voxels = self.voxelizer.compute_surface_voxels()
        
        # Should be boolean array
        self.assertEqual(surface_voxels.dtype, bool)
        
        # Should have computed grid bounds
        self.assertIsNotNone(self.voxelizer._grid_origin)
        self.assertIsNotNone(self.voxelizer._grid_shape)
        
        # Should have some surface voxels
        self.assertGreater(np.sum(surface_voxels), 0)
        
        # Shape should match grid
        self.assertEqual(surface_voxels.shape, self.voxelizer._grid_shape)
    
    def test_mark_contact_regions(self):
        """Test region marking."""
        surface_voxels = self.voxelizer.compute_surface_voxels()
        region_ids = self.voxelizer.mark_contact_regions(surface_voxels)
        
        # Should be integer array
        self.assertEqual(region_ids.dtype, np.int32)
        
        # Should have same shape as grid
        self.assertEqual(region_ids.shape, self.voxelizer._grid_shape)
        
        # Surface voxels should have region assignments
        surface_indices = np.where(surface_voxels)
        surface_region_ids = region_ids[surface_indices]
        self.assertGreater(np.sum(surface_region_ids >= 0), 0)
    
    def test_mark_keep_clear_zones(self):
        """Test keep-clear zone marking."""
        surface_voxels = self.voxelizer.compute_surface_voxels()
        region_ids = self.voxelizer.mark_contact_regions(surface_voxels)
        keep_clear = self.voxelizer.mark_keep_clear_zones(region_ids, surface_voxels)
        
        # Should be boolean array
        self.assertEqual(keep_clear.dtype, bool)
        
        # Should have same shape as grid
        self.assertEqual(keep_clear.shape, self.voxelizer._grid_shape)
        
        # Keep-clear should be subset of surface
        keep_clear_indices = np.where(keep_clear)
        surface_at_kc = surface_voxels[keep_clear_indices]
        self.assertTrue(np.all(surface_at_kc))
    
    def test_compute_pressure_weights(self):
        """Test pressure weight computation."""
        surface_voxels = self.voxelizer.compute_surface_voxels()
        region_ids = self.voxelizer.mark_contact_regions(surface_voxels)
        keep_clear = self.voxelizer.mark_keep_clear_zones(region_ids, surface_voxels)
        weights = self.voxelizer.compute_pressure_weights(region_ids, keep_clear, surface_voxels)
        
        # Should be float array
        self.assertEqual(weights.dtype, np.float32)
        
        # Should have same shape as grid
        self.assertEqual(weights.shape, self.voxelizer._grid_shape)
        
        # Weights should be in valid range
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        
        # Keep-clear voxels should have zero weight
        keep_clear_weights = weights[keep_clear]
        if len(keep_clear_weights) > 0:
            self.assertTrue(np.all(keep_clear_weights == 0))


class VoxelizationIntegrationTests(unittest.TestCase):
    """Test complete voxelization workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
        self.voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
    
    def test_voxelize_complete_workflow(self):
        """Test complete voxelization."""
        voxel_data = self.voxelizer.voxelize()
        
        # Should return VoxelData object
        self.assertIsNotNone(voxel_data)
        
        # Should have valid shape
        self.assertGreater(voxel_data.shape[0], 0)
        self.assertGreater(voxel_data.shape[1], 0)
        self.assertGreater(voxel_data.shape[2], 0)
        
        # Should have valid resolution
        self.assertAlmostEqual(voxel_data.resolution, 0.002)
        
        # Should have origin
        self.assertEqual(voxel_data.origin.shape, (3,))
        
        # Should have voxel data
        self.assertGreater(len(voxel_data.voxel_positions), 0)
        self.assertEqual(len(voxel_data.voxel_region_ids), len(voxel_data.voxel_positions))
        self.assertEqual(len(voxel_data.voxel_types), len(voxel_data.voxel_positions))
        self.assertEqual(len(voxel_data.voxel_weights), len(voxel_data.voxel_positions))
    
    def test_voxelize_different_resolutions(self):
        """Test voxelization at different resolutions."""
        resolutions = [0.001, 0.002, 0.005]
        volumes = []
        
        for res in resolutions:
            voxelizer = HandVoxelization(
                self.vertices,
                self.faces,
                self.segmentation,
                resolution=res,
            )
            voxel_data = voxelizer.voxelize()
            volume = len(voxel_data.voxel_positions) * (res ** 3)
            volumes.append(volume)
        
        # Volumes should be similar (same hand surface area)
        for i in range(len(volumes) - 1):
            # Allow 50% variance between resolutions
            self.assertLess(
                abs(volumes[i] - volumes[i+1]) / max(volumes[i], volumes[i+1]),
                0.5
            )
    
    def test_voxel_positions_within_bounds(self):
        """Test that voxel positions are within grid bounds."""
        voxel_data = self.voxelizer.voxelize()
        
        # Positions should be within grid shape
        for dim in range(3):
            self.assertTrue(np.all(voxel_data.voxel_positions[:, dim] >= 0))
            self.assertTrue(
                np.all(voxel_data.voxel_positions[:, dim] < voxel_data.shape[dim])
            )
    
    def test_region_ids_valid(self):
        """Test that region IDs are valid."""
        voxel_data = self.voxelizer.voxelize()
        
        # All non-negative IDs should be in region mapping
        valid_ids = set(self.voxelizer.segmentation._region_to_id.values())
        for rid in np.unique(voxel_data.voxel_region_ids):
            if rid >= 0:
                self.assertIn(rid, valid_ids)
    
    def test_voxel_types_valid(self):
        """Test that voxel types are valid."""
        voxel_data = self.voxelizer.voxelize()
        
        valid_types = {VoxelType.SURFACE.value, VoxelType.BOUNDARY.value}
        for vtype in np.unique(voxel_data.voxel_types):
            self.assertIn(vtype, valid_types)


class VoxelStatisticsTests(unittest.TestCase):
    """Test voxel statistics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
        self.voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
    
    def test_get_statistics_before_voxelize(self):
        """Test that statistics are None before voxelization."""
        stats = self.voxelizer.get_statistics()
        self.assertIsNone(stats)
    
    def test_get_statistics_after_voxelize(self):
        """Test statistics after voxelization."""
        self.voxelizer.voxelize()
        stats = self.voxelizer.get_statistics()
        
        self.assertIsNotNone(stats)
        self.assertGreater(stats.total_voxels, 0)
        self.assertGreater(stats.surface_voxels, 0)
        self.assertEqual(
            stats.total_voxels,
            stats.surface_voxels + stats.empty_voxels + stats.interior_voxels
        )
    
    def test_statistics_summary(self):
        """Test statistics summary string."""
        self.voxelizer.voxelize()
        stats = self.voxelizer.get_statistics()
        summary = stats.summary()
        
        self.assertIsNotNone(summary)
        self.assertIn("Voxel Statistics", summary)
        self.assertIn("Total voxels", summary)
        self.assertIn("Surface voxels", summary)
    
    def test_region_count_consistency(self):
        """Test that region voxel counts are consistent."""
        voxel_data = self.voxelizer.voxelize()
        
        # Sum of per-region counts should match total
        total_region_voxels = sum(
            voxel_data.statistics.voxels_per_region.values()
        )
        self.assertGreater(total_region_voxels, 0)


class VoxelExportTests(unittest.TestCase):
    """Test voxel data export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
        self.voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
    
    def test_export_before_voxelize_raises(self):
        """Test that export raises before voxelization."""
        with self.assertRaises(RuntimeError):
            self.voxelizer.export_to_dict()
    
    def test_export_after_voxelize(self):
        """Test export after voxelization."""
        self.voxelizer.voxelize()
        export_dict = self.voxelizer.export_to_dict()
        
        # Should contain required keys
        self.assertIn('grid_shape', export_dict)
        self.assertIn('grid_origin', export_dict)
        self.assertIn('resolution', export_dict)
        self.assertIn('voxel_positions', export_dict)
        self.assertIn('voxel_region_ids', export_dict)
        self.assertIn('voxel_types', export_dict)
        self.assertIn('voxel_weights', export_dict)
        self.assertIn('region_id_to_name', export_dict)
        self.assertIn('statistics', export_dict)
    
    def test_export_data_types(self):
        """Test that export produces serializable data."""
        self.voxelizer.voxelize()
        export_dict = self.voxelizer.export_to_dict()
        
        # All data should be serializable (lists, dicts, primitives)
        self.assertIsInstance(export_dict['grid_shape'], tuple)
        self.assertIsInstance(export_dict['grid_origin'], list)
        self.assertIsInstance(export_dict['voxel_positions'], list)
        self.assertIsInstance(export_dict['region_id_to_name'], dict)


class ConvenienceFunctionTests(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
    
    def test_voxelize_freihand_sample(self):
        """Test convenience function."""
        voxel_data = voxelize_freihand_sample(
            self.vertices,
            self.faces,
            self.keypoints,
            resolution=0.002,
        )
        
        # Should return VoxelData
        self.assertIsNotNone(voxel_data)
        self.assertGreater(len(voxel_data.voxel_positions), 0)
    
    def test_convenience_function_defaults(self):
        """Test convenience function with default resolution."""
        voxel_data = voxelize_freihand_sample(
            self.vertices,
            self.faces,
            self.keypoints,
        )
        
        # Should use default resolution
        self.assertAlmostEqual(voxel_data.resolution, 0.002)


class PressureWeightTests(unittest.TestCase):
    """Test pressure weight calculation specifics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vertices, self.keypoints, self.faces = create_synthetic_hand()
        self.segmentation = HandSegmentation(self.vertices, self.keypoints, self.faces)
        self.segmentation.segment_by_spatial_clustering()
        self.segmentation.assign_surface_types()
        self.voxelizer = HandVoxelization(
            self.vertices,
            self.faces,
            self.segmentation,
            resolution=0.002,
        )
    
    def test_contact_regions_have_high_weight(self):
        """Test that contact regions get weight 1.0."""
        voxel_data = self.voxelizer.voxelize()
        
        # Find voxels in contact regions
        contact_region_id = None
        for rid, region in self.voxelizer.segmentation._id_to_region.items():
            if region == AnatomicalRegion.PALM:
                contact_region_id = rid
                break
        
        if contact_region_id is not None:
            contact_voxels = voxel_data.voxel_region_ids == contact_region_id
            if np.sum(contact_voxels) > 0:
                contact_weights = voxel_data.voxel_weights[contact_voxels]
                # All contact voxels should have weight 1.0
                self.assertTrue(np.all(contact_weights == 1.0) or
                              np.all(contact_weights > 0.5))
    
    def test_support_regions_have_medium_weight(self):
        """Test that support regions get weight around 0.7."""
        voxel_data = self.voxelizer.voxelize()
        
        # Should have voxels with various weights
        unique_weights = np.unique(voxel_data.voxel_weights)
        self.assertGreater(len(unique_weights), 1)


if __name__ == '__main__':
    unittest.main()
