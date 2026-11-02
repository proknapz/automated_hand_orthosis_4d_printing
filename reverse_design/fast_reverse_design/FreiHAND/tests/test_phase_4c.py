"""Unit Tests for Phase 4C: Validation, Export & Reporting.

Tests all three Phase 4C modules:
- FEA validation (heuristic and structural metrics)
- STL export (mesh generation and file writing)
- Comprehensive reporting (ranking and analysis)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from phase_4c_fea_validation import (
    FEAValidator,
    PatientLoadCase,
    LoadCaseType,
    FEAResult,
)
from phase_4c_stl_export import (
    Mesh,
    MeshVertex,
    MeshTriangle,
    VoxelToMeshConverter,
    STLExporter,
    STLExportPipeline,
)
from phase_4c_comprehensive_report import (
    ComprehensiveAnalyzer,
    DesignRanking,
    ReportGenerator,
)


class TestPatientLoadCase:
    """Test load case definitions."""
    
    def test_clinical_standard(self):
        """Test clinical standard load case."""
        load = PatientLoadCase.clinical_standard()
        assert load.force_magnitude_n == 50.0
        assert load.load_type == LoadCaseType.STATIC_GRIP
        assert load.max_allowable_stress_mpa == 25.0
    
    def test_high_performance(self):
        """Test high-performance load case."""
        load = PatientLoadCase.high_performance()
        assert load.force_magnitude_n == 100.0
        assert load.load_type == LoadCaseType.DYNAMIC_GRASP
        assert load.min_support_quality == 0.85


class TestFEAValidator:
    """Test FEA validation module."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = FEAValidator(design_space_voxels=300)
        assert validator.design_space_voxels == 300
        assert validator.voxel_resolution_mm == 2.0
        assert validator.material_density_g_cm3 == 1.2
    
    def test_heuristic_validation(self):
        """Test heuristic FEA validation."""
        validator = FEAValidator(design_space_voxels=300, use_fea=False)
        
        # Create mock design
        voxel_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        load_case = PatientLoadCase.clinical_standard()
        
        result = validator.validate_design(
            design_rank=1,
            voxel_count=5,
            voxel_indices=voxel_indices,
            load_case=load_case,
        )
        
        assert isinstance(result, FEAResult)
        assert result.design_rank == 1
        assert result.design_voxel_count == 5
        assert result.max_von_mises_stress_mpa > 0
        assert 0 <= result.support_effectiveness <= 1
    
    def test_fea_result_summary(self):
        """Test FEA result summary dict."""
        result = FEAResult(
            design_rank=1,
            design_voxel_count=50,
            load_case=LoadCaseType.STATIC_GRIP,
            max_von_mises_stress_mpa=15.5,
            avg_von_mises_stress_mpa=10.2,
            peak_strain=0.005,
            avg_strain=0.003,
            max_deformation_mm=1.5,
            avg_deformation_mm=0.8,
            material_utilization=0.7,
            support_effectiveness=0.82,
            contact_pressure_peak_mpa=8.3,
            passes_stress_limit=True,
            passes_strain_limit=True,
            passes_deformation_limit=True,
            passes_support_requirement=True,
            overall_pass=True,
            simulation_time_s=0.25,
            mesh_elements=250,
        )
        
        summary = result.summary_dict()
        assert summary['design_rank'] == 1
        assert summary['passes_all_checks'] is True


class TestMesh:
    """Test mesh data structure."""
    
    def test_mesh_creation(self):
        """Test mesh creation."""
        vertices = [
            MeshVertex(0, 0, 0),
            MeshVertex(1, 0, 0),
            MeshVertex(0, 1, 0),
        ]
        triangles = [MeshTriangle(0, 1, 2)]
        mesh = Mesh(vertices, triangles)
        
        assert mesh.vertex_count() == 3
        assert mesh.triangle_count() == 1
    
    def test_mesh_bounds(self):
        """Test mesh bounding box computation."""
        vertices = [
            MeshVertex(0, 0, 0),
            MeshVertex(10, 0, 0),
            MeshVertex(0, 20, 0),
            MeshVertex(0, 0, 30),
        ]
        triangles = [MeshTriangle(0, 1, 2)]
        mesh = Mesh(vertices, triangles)
        
        bounds = mesh.compute_bounds()
        assert bounds[0] == (0, 0, 0)
        assert bounds[1] == (10, 20, 30)


class TestVoxelToMeshConverter:
    """Test voxel-to-mesh conversion."""
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = VoxelToMeshConverter(voxel_resolution_mm=2.0)
        assert converter.voxel_resolution_mm == 2.0
    
    def test_simple_cubic_conversion(self):
        """Test simple cubic voxel-to-mesh conversion."""
        converter = VoxelToMeshConverter(voxel_resolution_mm=1.0)
        
        # Create a 2x2x2 voxel grid; place voxels at positions 0,1,8,9
        # (creates a 2x1 block of material)
        voxel_indices = np.array([0, 1, 8, 9], dtype=np.int32)
        
        mesh = converter.convert_simple_cubic(
            voxel_indices,
            voxel_space_dims=(10, 10, 3)
        )
        
        assert mesh.vertex_count() > 0
        assert mesh.triangle_count() > 0
    
    def test_mesh_smoothing(self):
        """Test mesh smoothing."""
        converter = VoxelToMeshConverter(voxel_resolution_mm=1.0)
        
        # Create simple mesh
        vertices = [
            MeshVertex(0, 0, 0),
            MeshVertex(2, 0, 0),
            MeshVertex(1, 2, 0),
            MeshVertex(1, 1, 2),
        ]
        triangles = [
            MeshTriangle(0, 1, 2),
            MeshTriangle(0, 1, 3),
            MeshTriangle(1, 2, 3),
        ]
        mesh = Mesh(vertices, triangles)
        
        # Smooth
        smoothed = converter.smooth_mesh(mesh, iterations=1)
        
        assert smoothed.vertex_count() == mesh.vertex_count()
        assert smoothed.triangle_count() == mesh.triangle_count()


class TestSTLExporter:
    """Test STL export."""
    
    def test_binary_stl_export(self):
        """Test binary STL export."""
        vertices = [
            MeshVertex(0, 0, 0),
            MeshVertex(1, 0, 0),
            MeshVertex(0, 1, 0),
        ]
        triangles = [MeshTriangle(0, 1, 2)]
        mesh = Mesh(vertices, triangles)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            output_path = f.name
        
        try:
            STLExporter.export_binary_stl(mesh, output_path)
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_ascii_stl_export(self):
        """Test ASCII STL export."""
        vertices = [
            MeshVertex(0, 0, 0),
            MeshVertex(1, 0, 0),
            MeshVertex(0, 1, 0),
        ]
        triangles = [MeshTriangle(0, 1, 2)]
        mesh = Mesh(vertices, triangles)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            output_path = f.name
        
        try:
            STLExporter.export_ascii_stl(mesh, output_path, name="test_solid")
            assert Path(output_path).exists()
            
            # Check content
            with open(output_path) as f:
                content = f.read()
            assert "solid test_solid" in content
            assert "endsolid" in content
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestDesignRanking:
    """Test design ranking."""
    
    def test_ranking_creation(self):
        """Test design ranking creation."""
        ranking = DesignRanking(
            rank_in_pareto=1,
            voxel_count=50,
            weight_grams=0.6,
            support_quality=0.85,
            movement_preservation=0.50,
            material_efficiency=0.90,
            fabrication_feasibility=0.95,
        )
        
        assert ranking.rank_in_pareto == 1
        assert ranking.weight_grams == 0.6
    
    def test_composite_score(self):
        """Test composite score computation."""
        ranking = DesignRanking(
            rank_in_pareto=1,
            voxel_count=50,
            weight_grams=0.6,
            support_quality=0.8,
            movement_preservation=0.5,
            material_efficiency=0.9,
            fabrication_feasibility=0.9,
        )
        
        score = ranking.compute_composite_score()
        assert 0 <= score <= 1.0


class TestReportGenerator:
    """Test report generation."""
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        rankings = [
            DesignRanking(
                rank_in_pareto=1,
                voxel_count=50,
                weight_grams=0.6,
                support_quality=0.85,
                movement_preservation=0.50,
                material_efficiency=0.90,
                fabrication_feasibility=0.95,
                composite_score=0.85,
                clinical_recommendation="High support therapy",
            )
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            result = ReportGenerator.generate_html_report(rankings, output_path)
            assert Path(result).exists()
            
            with open(result) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "Design Rank 1" in content
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        rankings = [
            DesignRanking(
                rank_in_pareto=1,
                voxel_count=50,
                weight_grams=0.6,
                support_quality=0.85,
                movement_preservation=0.50,
                material_efficiency=0.90,
                fabrication_feasibility=0.95,
            )
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            result = ReportGenerator.generate_json_report(rankings, output_path)
            assert Path(result).exists()
            
            with open(result) as f:
                data = json.load(f)
            assert 'designs' in data
            assert len(data['designs']) == 1
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        rankings = [
            DesignRanking(
                rank_in_pareto=1,
                voxel_count=50,
                weight_grams=0.6,
                support_quality=0.85,
                movement_preservation=0.50,
                material_efficiency=0.90,
                fabrication_feasibility=0.95,
            )
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
            output_path = f.name
        
        try:
            result = ReportGenerator.generate_markdown_report(rankings, output_path)
            assert Path(result).exists()
            
            with open(result) as f:
                content = f.read()
            assert "# Phase 4C" in content
            assert "Design Rank 1" in content
        finally:
            Path(output_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
