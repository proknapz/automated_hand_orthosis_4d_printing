"""Tests for hand segmentation and anatomical region labeling."""

import unittest

import numpy as np

from FreiHAND.hand_segmentation import (
    AnatomicalRegion,
    HandSegmentation,
    SurfaceType,
    segment_freihand_sample,
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
    
    # Simple triangular faces (optional)
    faces = np.array([
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ])
    
    return vertices, keypoints, faces


class HandSegmentationInitTests(unittest.TestCase):
    """Test segmentation initialization and validation."""
    
    def test_init_validates_vertices_shape(self):
        """Test that __init__ validates vertex shape."""
        vertices = np.random.randn(10, 2)  # Wrong shape
        keypoints = np.random.randn(21, 3)
        
        with self.assertRaises(ValueError) as ctx:
            HandSegmentation(vertices, keypoints)
        self.assertIn("(V, 3)", str(ctx.exception))
    
    def test_init_validates_keypoints_shape(self):
        """Test that __init__ validates keypoint shape."""
        vertices = np.random.randn(778, 3)
        keypoints = np.random.randn(20, 3)  # Should be 21
        
        with self.assertRaises(ValueError) as ctx:
            HandSegmentation(vertices, keypoints)
        self.assertIn("(21, 3)", str(ctx.exception))
    
    def test_init_accepts_optional_faces(self):
        """Test that faces parameter is optional."""
        vertices, keypoints, _ = create_synthetic_hand()
        
        # Should work without faces
        seg1 = HandSegmentation(vertices, keypoints)
        self.assertIsNone(seg1.faces)
        
        # Should work with faces
        faces = np.array([[0, 1, 2]])
        seg2 = HandSegmentation(vertices, keypoints, faces)
        self.assertIsNotNone(seg2.faces)
    
    def test_init_validates_faces_shape(self):
        """Test that __init__ validates face shape."""
        vertices = np.random.randn(10, 3)
        keypoints = np.random.randn(21, 3)
        faces = np.array([[0, 1]])  # Wrong shape
        
        with self.assertRaises(ValueError) as ctx:
            HandSegmentation(vertices, keypoints, faces)
        self.assertIn("(F, 3)", str(ctx.exception))


class SpatialClusteringTests(unittest.TestCase):
    """Test spatial clustering segmentation."""
    
    def test_segment_by_spatial_clustering_returns_regions(self):
        """Test that clustering returns all expected regions."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        regions = seg.segment_by_spatial_clustering()
        
        self.assertIsInstance(regions, dict)
        self.assertIn(AnatomicalRegion.PALM, regions)
        self.assertIn(AnatomicalRegion.WRIST, regions)
    
    def test_segment_assigns_all_vertices(self):
        """Test that all vertices are assigned to a region."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        seg.segment_by_spatial_clustering()
        
        # Check that region_assignment has correct shape
        self.assertEqual(seg._region_assignment.shape, (vertices.shape[0],))
        
        # Check that no vertices are unassigned
        unassigned = np.where(seg._region_assignment == -1)[0]
        self.assertEqual(len(unassigned), 0)
    
    def test_segment_regions_cover_all_vertices(self):
        """Test that region assignments cover all vertices."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        regions = seg.segment_by_spatial_clustering()
        
        # Collect all assigned vertices (may have overlaps at boundaries)
        all_vertices = set()
        for vertex_set in regions.values():
            all_vertices.update(vertex_set)
        
        # All vertices should be assigned to at least one region
        self.assertEqual(len(all_vertices), vertices.shape[0],
                        "All vertices must be assigned to a region")


class SurfaceTypeAssignmentTests(unittest.TestCase):
    """Test surface type assignment."""
    
    def test_assign_surface_types_returns_region_properties(self):
        """Test that assign_surface_types returns proper structure."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        regions = seg.assign_surface_types()
        
        self.assertIsInstance(regions, dict)
        for region, props in regions.items():
            self.assertIsInstance(region, AnatomicalRegion)
            self.assertIsNotNone(props.region)
            self.assertIsNotNone(props.vertex_indices)
            self.assertIsNotNone(props.surface_area)
            self.assertIsNotNone(props.surface_type)
    
    def test_assign_surface_types_with_custom_mapping(self):
        """Test custom surface type mapping."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        custom_mapping = {
            AnatomicalRegion.PALM: SurfaceType.KEEP_CLEAR,
            AnatomicalRegion.WRIST: SurfaceType.SUPPORT,
        }
        
        regions = seg.assign_surface_types(custom_mapping)
        
        # Check that custom types are applied
        self.assertEqual(
            regions[AnatomicalRegion.PALM].surface_type,
            SurfaceType.KEEP_CLEAR
        )
        self.assertEqual(
            regions[AnatomicalRegion.WRIST].surface_type,
            SurfaceType.SUPPORT
        )
    
    def test_assign_surface_types_calculates_centroid(self):
        """Test that centroids are calculated."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        regions = seg.assign_surface_types()
        
        for region, props in regions.items():
            # Centroid should be within vertex bounds
            self.assertEqual(props.centroid.shape, (3,))
            self.assertTrue(np.all(np.isfinite(props.centroid)))


class ContactMaskTests(unittest.TestCase):
    """Test contact mask generation."""
    
    def test_get_contact_mask_returns_boolean_array(self):
        """Test that get_contact_mask returns proper boolean mask."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        mask = seg.get_contact_mask()
        
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (vertices.shape[0],))
        self.assertTrue(np.any(mask))  # At least some contact regions
    
    def test_get_keep_clear_mask_returns_boolean_array(self):
        """Test that get_keep_clear_mask returns proper boolean mask."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        seg.assign_surface_types()
        
        mask = seg.get_keep_clear_mask()
        
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (vertices.shape[0],))
    
    def test_contact_and_keep_clear_masks_dont_overlap(self):
        """Test that contact and keep-clear masks are mutually exclusive."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        
        contact = seg.get_contact_mask()
        keep_clear = seg.get_keep_clear_mask()
        
        # They should not overlap
        overlap = contact & keep_clear
        self.assertEqual(np.sum(overlap), 0, "Contact and keep-clear should not overlap")


class SummaryTests(unittest.TestCase):
    """Test summary generation."""
    
    def test_summary_before_segmentation(self):
        """Test summary before segmentation."""
        vertices, keypoints, _ = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints)
        
        summary = seg.summary()
        self.assertIn("No segmentation", summary)
    
    def test_summary_after_segmentation(self):
        """Test summary after segmentation."""
        vertices, keypoints, faces = create_synthetic_hand()
        seg = HandSegmentation(vertices, keypoints, faces)
        seg.assign_surface_types()
        
        summary = seg.summary()
        
        self.assertIn("Segmentation Summary", summary)
        self.assertIn("PALM", summary)
        self.assertIn("Vertices:", summary)
        self.assertIn("Surface type:", summary)


class ConvenienceFunctionsTests(unittest.TestCase):
    """Test convenience functions."""
    
    def test_segment_freihand_sample(self):
        """Test convenience function for FreiHAND samples."""
        from FreiHAND.data_loader import HandGeometry
        
        vertices, keypoints, faces = create_synthetic_hand()
        
        geometry = HandGeometry(
            sample_id="test",
            vertices=vertices,
            keypoints=keypoints,
            faces=faces,
        )
        
        seg = segment_freihand_sample(geometry)
        
        self.assertIsInstance(seg, HandSegmentation)
        self.assertIsNotNone(seg._regions)
        self.assertTrue(len(seg._regions) > 0)


if __name__ == "__main__":
    unittest.main()
