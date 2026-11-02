"""Unit tests for orthosis_design_space.py (Phase 4A design space module).

Tests:
- DesignZoneType and DesignConstraint enums/dataclasses
- DesignSpace initialization and state
- OrthosisDesignSpace.build() with mock voxel data
- Support region identification
- Movement zone definition
- Thickness map generation
- Seed candidate generation
- Design space summary and statistics
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from FreiHAND.hand_segmentation import AnatomicalRegion, SurfaceType
from FreiHAND.orthosis_design_space import (
    DesignConstraint,
    DesignSpace,
    DesignZoneType,
    OrthosisDesignSpace,
)
from FreiHAND.surface_voxelization import VoxelData, VoxelType


class MockHandSegmentation:
    """Mock HandSegmentation for testing."""
    pass


class MockVoxelData:
    """Create mock VoxelData for testing."""
    
    @staticmethod
    def create_minimal(num_voxels: int = 100) -> VoxelData:
        """Create minimal valid VoxelData for testing."""
        # Create random voxel positions
        np.random.seed(42)
        positions = np.random.randint(0, 50, size=(num_voxels, 3), dtype=np.int32)
        
        # Create region and type data
        region_ids = np.random.randint(0, 6, size=num_voxels, dtype=np.int32)
        types = np.full(num_voxels, VoxelType.SURFACE.value, dtype=np.uint8)
        
        # Create weights (higher for critical regions)
        weights = np.random.rand(num_voxels).astype(np.float32)
        
        # Region mapping
        region_names = ["palm", "thumb", "index", "middle", "ring", "pinky"]
        region_id_to_name = {i: region_names[i] for i in range(len(region_names))}
        
        # Import VoxelStatistics to create proper instance
        from FreiHAND.surface_voxelization import VoxelStatistics
        stats = VoxelStatistics(
            total_voxels=num_voxels * 10,
            surface_voxels=num_voxels,
            interior_voxels=0,
            empty_voxels=num_voxels * 9,
        )
        
        return VoxelData(
            shape=(100, 100, 150),
            resolution=0.002,  # 2mm
            origin=np.array([0.0, 0.0, 0.0]),
            voxel_positions=positions,
            voxel_region_ids=region_ids,
            voxel_types=types,
            voxel_weights=weights,
            region_id_to_name=region_id_to_name,
            statistics=stats,
        )


class TestDesignZoneType:
    """Test DesignZoneType enum."""
    
    def test_enum_values(self):
        """Test that enum values are defined correctly."""
        assert DesignZoneType.FORBIDDEN.value == 0
        assert DesignZoneType.FLEXIBLE.value == 1
        assert DesignZoneType.SUPPORT.value == 2
        assert DesignZoneType.REINFORCE.value == 3
    
    def test_enum_names(self):
        """Test enum names are accessible."""
        assert DesignZoneType.FORBIDDEN.name == "FORBIDDEN"
        assert DesignZoneType.FLEXIBLE.name == "FLEXIBLE"


class TestDesignConstraint:
    """Test DesignConstraint dataclass."""
    
    def test_constraint_creation(self):
        """Test creating a design constraint."""
        voxels = np.array([(0, 0, 0), (1, 1, 1)], dtype=np.int32)
        constraint = DesignConstraint(
            zone_type=DesignZoneType.SUPPORT,
            region=AnatomicalRegion.THUMB,
            support_priority=0.85,
            movement_restriction=0.5,
            voxel_indices=voxels,
            thickness_mm=(2.0, 5.0),
            fill_fraction=(0.4, 0.8),
        )
        
        assert constraint.support_priority == 0.85
        assert constraint.zone_type == DesignZoneType.SUPPORT
        assert len(constraint.voxel_indices) == 2
        assert constraint.thickness_mm == (2.0, 5.0)
    
    def test_constraint_defaults(self):
        """Test constraint with default values."""
        constraint = DesignConstraint(
            zone_type=DesignZoneType.FLEXIBLE,
            region=AnatomicalRegion.INDEX,
            support_priority=0.8,
            movement_restriction=0.6,
        )
        
        assert constraint.thickness_mm == (0.0, 8.0)
        assert constraint.fill_fraction == (0.0, 1.0)


class TestDesignSpace:
    """Test DesignSpace dataclass."""
    
    def test_design_space_creation(self):
        """Test creating a design space."""
        zone_map = {(0, 0, 0): DesignZoneType.SUPPORT, (1, 1, 1): DesignZoneType.FLEXIBLE}
        support_map = {(0, 0, 0): 0.8, (1, 1, 1): 0.5}
        movement_map = {(0, 0, 0): 0.3, (1, 1, 1): 0.7}
        thickness_map = {(0, 0, 0): (2.0, 5.0), (1, 1, 1): (0.0, 2.0)}
        
        ds = DesignSpace(
            grid_shape=(100, 100, 150),
            grid_origin=np.array([0.0, 0.0, 0.0]),
            voxel_resolution=0.002,
            zone_map=zone_map,
            support_map=support_map,
            movement_map=movement_map,
            thickness_map=thickness_map,
        )
        
        assert ds.total_voxels() == 1_500_000
        assert ds.available_voxels() == 2
    
    def test_design_space_summary(self):
        """Test design space summary string generation."""
        ds = DesignSpace(
            grid_shape=(100, 100, 150),
            grid_origin=np.array([0.0, 0.0, 0.0]),
            voxel_resolution=0.002,
            zone_map={},
            support_map={},
            movement_map={},
            thickness_map={},
            max_material_voxels=10000,
        )
        
        summary = ds.summary()
        assert "Design Space Summary" in summary
        assert "10" in summary  # May be formatted as 10,000 or 10000
        assert "1.2" in summary


class TestOrthosisDesignSpace:
    """Test OrthosisDesignSpace builder."""
    
    def test_initialization(self):
        """Test OrthosisDesignSpace initialization."""
        voxel_data = MockVoxelData.create_minimal()
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        
        assert builder.voxel_data is not None
        assert builder.design_params is not None
        assert "max_material_voxels" in builder.design_params
        assert "support_priorities" in builder.design_params
    
    def test_load_design_params_default(self):
        """Test loading design parameters with defaults."""
        voxel_data = MockVoxelData.create_minimal()
        segmentation = MockHandSegmentation()
        
        # With non-existent path, should use defaults
        builder = OrthosisDesignSpace(voxel_data, segmentation, Path("/nonexistent/path.json"))
        
        assert builder.design_params["weight_budget_grams"] == 80
        assert builder.design_params["max_material_voxels"] == 10000
    
    def test_load_design_params_from_file(self):
        """Test loading design parameters from JSON file."""
        voxel_data = MockVoxelData.create_minimal()
        segmentation = MockHandSegmentation()
        
        # Create temporary params file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            params = {
                "weight_budget_grams": 100,
                "max_material_voxels": 15000,
                "material_density_g_cm3": 1.5,
                "support_priorities": {"wrist": 0.95, "thumb": 0.80},
                "movement_restrictions": {"wrist": 0.3, "thumb": 0.5},
            }
            json.dump(params, f)
            temp_path = Path(f.name)
        
        try:
            builder = OrthosisDesignSpace(voxel_data, segmentation, temp_path)
            assert builder.design_params["weight_budget_grams"] == 100
            assert builder.design_params["max_material_voxels"] == 15000
        finally:
            temp_path.unlink()
    
    def test_build_creates_design_space(self):
        """Test that build() creates a valid DesignSpace."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=50)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        design_space = builder.build()
        
        assert design_space is not None
        assert design_space.grid_shape == (100, 100, 150)
        assert len(design_space.zone_map) > 0
        assert len(design_space.support_map) > 0
        assert len(design_space.movement_map) > 0
        assert len(design_space.thickness_map) > 0
        assert len(design_space.constraints) > 0
        assert len(design_space.seed_candidates) > 0
    
    def test_identify_support_regions(self):
        """Test support region identification."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=100)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        zone_map, support_map = builder._identify_support_regions()
        
        assert len(zone_map) == len(support_map)
        assert len(zone_map) == 100
        
        # Check that zone types are valid
        for zone_type in zone_map.values():
            assert zone_type in DesignZoneType
        
        # Check that support priorities are in valid range
        for priority in support_map.values():
            assert 0.0 <= priority <= 1.0
    
    def test_define_movement_zones(self):
        """Test movement zone definition."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=100)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        zone_map, _ = builder._identify_support_regions()
        movement_map = builder._define_movement_zones(zone_map)
        
        assert len(movement_map) == len(zone_map)
        
        # Check movement factors are in valid range
        for factor in movement_map.values():
            assert 0.0 <= factor <= 1.0
    
    def test_create_thickness_maps(self):
        """Test thickness map generation."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=100)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        zone_map, _ = builder._identify_support_regions()
        movement_map = builder._define_movement_zones(zone_map)
        thickness_map = builder._create_thickness_maps(zone_map, movement_map)
        
        assert len(thickness_map) == len(zone_map)
        
        # Check that thickness ranges are valid
        for min_t, max_t in thickness_map.values():
            assert 0.0 <= min_t <= max_t <= 8.0
    
    def test_generate_seed_candidates(self):
        """Test seed candidate generation."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=200)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        design_space = builder.build()
        
        seeds = design_space.seed_candidates
        assert len(seeds) == 3  # Default: max, balanced, minimal
        
        # Check seed sizes
        assert len(seeds[0]) > len(seeds[2])  # Max > minimal
        
        # Check all seeds respect budget
        max_voxels = design_space.max_material_voxels
        for seed in seeds:
            assert len(seed) <= max_voxels


class TestDesignSpaceIntegration:
    """Integration tests with full build pipeline."""
    
    def test_full_pipeline(self):
        """Test complete design space build pipeline."""
        voxel_data = MockVoxelData.create_minimal(num_voxels=150)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data, segmentation)
        design_space = builder.build()
        
        # Verify complete design space
        assert design_space.total_voxels() == 1_500_000
        assert design_space.available_voxels() == 150
        
        # Verify summary works
        summary = design_space.summary()
        assert "Design Space Summary" in summary
        assert "1,500,000" in summary or "1500000" in summary or "1.5" in summary
        assert "3" in summary  # Seed candidates
    
    def test_multiple_builds(self):
        """Test that builder can be used multiple times."""
        voxel_data1 = MockVoxelData.create_minimal(num_voxels=50)
        voxel_data2 = MockVoxelData.create_minimal(num_voxels=100)
        segmentation = MockHandSegmentation()
        
        builder = OrthosisDesignSpace(voxel_data1, segmentation)
        ds1 = builder.build()
        
        builder2 = OrthosisDesignSpace(voxel_data2, segmentation)
        ds2 = builder2.build()
        
        assert ds1.available_voxels() == 50
        assert ds2.available_voxels() == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
