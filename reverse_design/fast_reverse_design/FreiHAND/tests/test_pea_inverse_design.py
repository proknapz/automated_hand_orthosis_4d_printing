"""Unit tests for pea_inverse_design.py (Phase 4B optimizer module).

Tests:
- DesignIndividual creation and metrics
- DesignIndividual dominance comparison
- FitnessEvaluator with mock design space
- PEAOrthosisOptimizer initialization
- Single generation optimization
- Population statistics
- Export functionality
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from FreiHAND.orthosis_design_space import DesignSpace, DesignZoneType
from FreiHAND.pea_inverse_design import (
    DesignEncoding,
    DesignIndividual,
    FitnessEvaluator,
    PEAOrthosisOptimizer,
    PopulationStatistics,
)


class MockDesignSpace:
    """Create mock DesignSpace for testing."""
    
    @staticmethod
    def create() -> DesignSpace:
        """Create a minimal valid DesignSpace for testing."""
        # Create simple maps
        zone_map = {}
        support_map = {}
        movement_map = {}
        thickness_map = {}
        
        # Add some voxels to the maps
        for i in range(100):
            pos = (i % 10, (i // 10) % 5, i % 20)
            zone_type = [
                DesignZoneType.SUPPORT,
                DesignZoneType.FLEXIBLE,
                DesignZoneType.REINFORCE,
            ][i % 3]
            
            zone_map[pos] = zone_type
            support_map[pos] = 0.5 + (i % 50) / 100.0  # 0.5-0.99
            movement_map[pos] = (i % 100) / 100.0  # 0.0-0.99
            thickness_map[pos] = (1.0, 5.0)
        
        # Create seed candidates
        seed1 = np.array(list(range(0, 20)), dtype=np.int32)
        seed2 = np.array(list(range(10, 40)), dtype=np.int32)
        seed3 = np.array(list(range(0, 10)), dtype=np.int32)
        
        return DesignSpace(
            grid_shape=(10, 5, 20),
            grid_origin=np.array([0.0, 0.0, 0.0]),
            voxel_resolution=0.002,
            zone_map=zone_map,
            support_map=support_map,
            movement_map=movement_map,
            thickness_map=thickness_map,
            constraints=[],
            max_material_voxels=1000,
            material_density=1.2,
            design_params={},
            seed_candidates=[seed1, seed2, seed3],
        )


class TestDesignIndividual:
    """Test DesignIndividual dataclass."""
    
    def test_creation(self):
        """Test creating a design individual."""
        voxels = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ind = DesignIndividual(voxel_indices=voxels)
        
        assert len(ind.voxel_indices) == 5
        assert ind.voxel_count() == 5
        assert ind.generation == 0
    
    def test_voxel_metrics(self):
        """Test voxel count and weight calculations."""
        voxels = np.array(list(range(100)), dtype=np.int32)
        ind = DesignIndividual(voxel_indices=voxels)
        
        assert ind.voxel_count() == 100
        
        volume = ind.voxel_volume_cm3(voxel_resolution_m=0.002)
        assert volume > 0
        
        weight = ind.weight_grams(material_density_g_cm3=1.2)
        assert weight > 0
        assert abs(weight - volume * 1.2) < 0.01
    
    def test_dominance_simple(self):
        """Test simple dominance comparison."""
        ind1 = DesignIndividual(
            voxel_indices=np.array([1, 2, 3]),
            support_quality=0.8,
            movement_preservation=0.7,
            material_efficiency=0.6,
            fabrication_feasibility=0.7,
        )
        
        ind2 = DesignIndividual(
            voxel_indices=np.array([1, 2]),
            support_quality=0.6,
            movement_preservation=0.5,
            material_efficiency=0.5,
            fabrication_feasibility=0.5,
        )
        
        # ind1 should dominate ind2 (better in all objectives)
        assert ind1.dominates(ind2)
        assert not ind2.dominates(ind1)
    
    def test_dominance_non_dominated(self):
        """Test that non-dominated individuals don't dominate each other."""
        ind1 = DesignIndividual(
            voxel_indices=np.array([1, 2, 3]),
            support_quality=0.9,
            movement_preservation=0.5,
            material_efficiency=0.6,
            fabrication_feasibility=0.6,
        )
        
        ind2 = DesignIndividual(
            voxel_indices=np.array([1, 2]),
            support_quality=0.7,
            movement_preservation=0.9,
            material_efficiency=0.8,
            fabrication_feasibility=0.7,
        )
        
        # Neither dominates (trade-off in objectives)
        assert not ind1.dominates(ind2)
        assert not ind2.dominates(ind1)


class TestPopulationStatistics:
    """Test PopulationStatistics dataclass."""
    
    def test_creation(self):
        """Test creating statistics."""
        stats = PopulationStatistics(
            generation=0,
            population_size=100,
            avg_voxel_count=50.5,
            pareto_front_size=10,
        )
        
        assert stats.generation == 0
        assert stats.population_size == 100
        assert stats.pareto_front_size == 10
    
    def test_summary(self):
        """Test statistics summary string."""
        stats = PopulationStatistics(
            generation=5,
            population_size=100,
            avg_voxel_count=50.5,
            min_voxel_count=10,
            max_voxel_count=100,
            avg_support=0.75,
            avg_movement=0.65,
            pareto_front_size=15,
        )
        
        summary = stats.summary()
        assert "Generation 5" in summary
        assert "100" in summary or "100.0" in summary


class TestFitnessEvaluator:
    """Test FitnessEvaluator."""
    
    def test_initialization(self):
        """Test creating a fitness evaluator."""
        design_space = MockDesignSpace.create()
        evaluator = FitnessEvaluator(design_space)
        
        assert evaluator.design_space is not None
    
    def test_evaluate_empty(self):
        """Test evaluating empty design."""
        design_space = MockDesignSpace.create()
        evaluator = FitnessEvaluator(design_space)
        
        ind = DesignIndividual(voxel_indices=np.array([], dtype=np.int32))
        result = evaluator.evaluate(ind)
        
        assert result.support_quality == 0.0
        assert result.movement_preservation == 1.0
        assert result.material_efficiency == 1.0
    
    def test_evaluate_design(self):
        """Test evaluating a non-empty design."""
        design_space = MockDesignSpace.create()
        evaluator = FitnessEvaluator(design_space)
        
        ind = DesignIndividual(voxel_indices=np.array([1, 2, 3, 4, 5], dtype=np.int32))
        result = evaluator.evaluate(ind)
        
        assert result.support_quality is not None
        assert result.movement_preservation is not None
        assert result.material_efficiency is not None
        assert result.fabrication_feasibility is not None
        
        # All should be in valid range
        assert 0.0 <= result.support_quality <= 1.0
        assert 0.0 <= result.movement_preservation <= 1.0
        assert 0.0 <= result.material_efficiency <= 1.0
        assert 0.0 <= result.fabrication_feasibility <= 1.0


class TestPEAOrthosisOptimizer:
    """Test PEAOrthosisOptimizer."""
    
    def test_initialization(self):
        """Test creating an optimizer."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(
            design_space,
            population_size=20,
            generations=2,
        )
        
        assert optimizer.population_size == 20
        assert optimizer.max_generations == 2
        assert len(optimizer.population) == 0  # Not initialized yet
    
    def test_initialization_population(self):
        """Test population initialization."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(
            design_space,
            population_size=20,
            generations=2,
        )
        
        optimizer._initialize_population()
        
        assert len(optimizer.population) == 20
        # Some seeds should be used
        assert any(ind.generation == 0 for ind in optimizer.population)
    
    def test_generate_random_design(self):
        """Test random design generation."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(design_space)
        
        design = optimizer._generate_random_design()
        
        assert len(design) > 0
        assert len(design) <= design_space.max_material_voxels
        assert all(0 <= v < np.prod(design_space.grid_shape) for v in design)
    
    def test_single_generation(self):
        """Test one generation of optimization."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(
            design_space,
            population_size=20,
            generations=1,
        )
        
        optimizer._initialize_population()
        optimizer._evaluate_population()
        optimizer._update_pareto_front()
        
        assert len(optimizer.pareto_front) > 0
        assert len(optimizer.pareto_front) <= optimizer.population_size
    
    def test_crossover(self):
        """Test crossover operator."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(design_space)
        
        parent1 = DesignIndividual(voxel_indices=np.array([1, 2, 3, 4], dtype=np.int32))
        parent2 = DesignIndividual(voxel_indices=np.array([3, 4, 5, 6], dtype=np.int32))
        
        child = optimizer._crossover(parent1, parent2)
        
        assert isinstance(child, DesignIndividual)
        assert len(child.voxel_indices) > 0
        assert len(child.voxel_indices) <= design_space.max_material_voxels
    
    def test_mutate(self):
        """Test mutation operator."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(design_space)
        
        parent = DesignIndividual(voxel_indices=np.array(list(range(20)), dtype=np.int32))
        
        child = optimizer._mutate(parent)
        
        assert isinstance(child, DesignIndividual)
        assert len(child.voxel_indices) > 0
        assert len(child.voxel_indices) <= design_space.max_material_voxels
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(design_space)
        
        # Create population with fitness
        optimizer.population = [
            DesignIndividual(
                voxel_indices=np.array(list(range(i+5)), dtype=np.int32),
                support_quality=0.5 + i*0.05,
                movement_preservation=0.6 + i*0.04,
                material_efficiency=0.7 + i*0.03,
                fabrication_feasibility=0.8 + i*0.02,
            )
            for i in range(10)
        ]
        optimizer.pareto_front = optimizer.population[:3]
        
        stats = optimizer._compute_statistics(0)
        
        assert stats.generation == 0
        assert stats.population_size == 10
        assert stats.pareto_front_size == 3
        assert stats.avg_voxel_count > 0
    
    def test_export_results(self):
        """Test result export."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(design_space)
        
        # Create mock results
        optimizer.pareto_front = [
            DesignIndividual(
                voxel_indices=np.array(list(range(i+5)), dtype=np.int32),
                support_quality=0.7,
                movement_preservation=0.6,
                material_efficiency=0.8,
                fabrication_feasibility=0.75,
            )
            for i in range(3)
        ]
        optimizer.generation_history = [
            PopulationStatistics(generation=0, population_size=20, pareto_front_size=3),
            PopulationStatistics(generation=1, population_size=20, pareto_front_size=5),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            optimizer.export_results(output_dir)
            
            # Check files exist
            assert (output_dir / "designs.json").exists()
            assert (output_dir / "statistics.json").exists()
            assert (output_dir / "top_design.json").exists()
            
            # Check content
            with open(output_dir / "designs.json") as f:
                designs = json.load(f)
                assert len(designs) == 3
            
            with open(output_dir / "statistics.json") as f:
                stats = json.load(f)
                assert len(stats) == 2


class TestOptimizationIntegration:
    """Integration tests for full optimization pipeline."""
    
    def test_short_optimization(self):
        """Test short optimization run."""
        design_space = MockDesignSpace.create()
        optimizer = PEAOrthosisOptimizer(
            design_space,
            population_size=15,
            generations=2,
        )
        
        front, history = optimizer.optimize()
        
        assert len(front) > 0
        assert len(history) == 2
        assert all(stats.population_size == 15 for stats in history)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
