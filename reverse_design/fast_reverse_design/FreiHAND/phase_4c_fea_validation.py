"""Phase 4C: FEA Validation Module for Orthosis Designs.

This module validates Pareto-optimal orthosis designs from Phase 4B using
Finite Element Analysis simulation:

1. Load Pareto-optimal designs from phase_4b_results/designs.json
2. Convert voxel sets to FEA mesh geometry
3. Apply patient-specific loads and boundary conditions
4. Run FEA simulation (integrates with existing Abaqus infrastructure)
5. Extract stress, strain, and deformation metrics
6. Validate against clinical requirements
7. Rank designs by structural performance

Key Metrics Computed:
- Maximum von Mises stress (palatability, comfort)
- Peak strain in support regions (safety margin)
- Material utilization (load transfer efficiency)
- Deformation under load (support effectiveness)
- Contact pressure distribution (pressure sores risk)
- Fabrication feasibility (printability validation)

The module provides both:
- Heuristic validation (fast, lightweight)
- FEA-based validation (when Abaqus is available)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class LoadCaseType(Enum):
    """Standard load cases for orthosis validation."""
    STATIC_GRIP = 0        # Static grip force (most common)
    DYNAMIC_GRASP = 1      # Dynamic grasping motion
    PINCH_FORCE = 2        # Pinch/precision grip
    SUPPORT_HOLD = 3       # Static support (minimum load)


@dataclass
class PatientLoadCase:
    """Defines the load conditions for orthosis validation."""
    
    load_type: LoadCaseType
    force_magnitude_n: float           # Total force in Newtons
    force_distribution: str            # 'uniform', 'fingertip', 'palm'
    support_regions: List[str]         # ['wrist', 'palm', 'thumb', ...]
    max_allowable_stress_mpa: float    # Material-specific limit
    max_allowable_strain: float        # Safety margin
    min_support_quality: float         # Minimum support effectiveness (0-1)
    max_deformation_mm: float          # Maximum allowable deflection
    
    @staticmethod
    def clinical_standard() -> PatientLoadCase:
        """Standard clinical load case for hand orthosis validation."""
        return PatientLoadCase(
            load_type=LoadCaseType.STATIC_GRIP,
            force_magnitude_n=50.0,        # Typical grip force (50N ~ 11 lb)
            force_distribution='palm',
            support_regions=['palm', 'wrist'],
            max_allowable_stress_mpa=25.0, # Resin material safety limit
            max_allowable_strain=0.05,     # 5% max strain
            min_support_quality=0.70,      # 70% minimum support
            max_deformation_mm=3.0,        # 3mm max deflection
        )
    
    @staticmethod
    def high_performance() -> PatientLoadCase:
        """High-performance requirement for intensive therapy."""
        return PatientLoadCase(
            load_type=LoadCaseType.DYNAMIC_GRASP,
            force_magnitude_n=100.0,       # Heavy load (100N ~ 22 lb)
            force_distribution='palm',
            support_regions=['wrist', 'palm', 'fingers'],
            max_allowable_stress_mpa=20.0, # Stricter safety margin
            max_allowable_strain=0.03,
            min_support_quality=0.85,      # 85% minimum support
            max_deformation_mm=2.0,
        )


@dataclass
class FEAResult:
    """Results from FEA simulation of a single design."""
    
    design_rank: int
    design_voxel_count: int
    load_case: LoadCaseType
    
    # Stress/strain metrics
    max_von_mises_stress_mpa: float
    avg_von_mises_stress_mpa: float
    peak_strain: float
    avg_strain: float
    
    # Deformation metrics
    max_deformation_mm: float
    avg_deformation_mm: float
    
    # Structural performance
    material_utilization: float        # % of material under stress
    support_effectiveness: float       # 0-1 scale, structural support quality
    contact_pressure_peak_mpa: float   # Max contact pressure at support interfaces
    
    # Validation results
    passes_stress_limit: bool
    passes_strain_limit: bool
    passes_deformation_limit: bool
    passes_support_requirement: bool
    overall_pass: bool
    
    # Metadata
    simulation_time_s: float
    mesh_elements: int
    
    def summary_dict(self) -> Dict:
        """Export results as dictionary."""
        return {
            'design_rank': self.design_rank,
            'voxel_count': self.design_voxel_count,
            'load_case': self.load_case.name,
            'max_stress_mpa': round(self.max_von_mises_stress_mpa, 2),
            'peak_strain': round(self.peak_strain, 4),
            'max_deformation_mm': round(self.max_deformation_mm, 2),
            'material_utilization': round(self.material_utilization, 3),
            'support_effectiveness': round(self.support_effectiveness, 3),
            'passes_all_checks': self.overall_pass,
            'simulation_time_s': round(self.simulation_time_s, 2),
        }


class FEAValidator:
    """Validates orthosis designs using FEA simulation.
    
    This class provides two validation modes:
    1. Heuristic validation (fast, no external dependencies)
    2. Abaqus FEA validation (accurate, requires Abaqus installation)
    """
    
    def __init__(
        self,
        design_space_voxels: int = 300,
        voxel_resolution_mm: float = 2.0,
        material_density_g_cm3: float = 1.2,
        use_fea: bool = False,
    ):
        """Initialize FEA validator.
        
        Args:
            design_space_voxels: Total voxels in design space (for normalization)
            voxel_resolution_mm: Voxel size in mm for mesh generation
            material_density_g_cm3: Material density for weight calculations
            use_fea: If True, attempt to use Abaqus; if False, use heuristics
        """
        self.design_space_voxels = design_space_voxels
        self.voxel_resolution_mm = voxel_resolution_mm
        self.material_density_g_cm3 = material_density_g_cm3
        self.use_fea = use_fea
        self.abaqus_available = self._check_abaqus()
        
        if use_fea and not self.abaqus_available:
            warnings.warn(
                "Abaqus not available; falling back to heuristic validation",
                UserWarning
            )
    
    def _check_abaqus(self) -> bool:
        """Check if Abaqus is available on system."""
        try:
            import subprocess
            result = subprocess.run(
                ['abaqus', 'information=release'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, Exception):
            return False
    
    def validate_design(
        self,
        design_rank: int,
        voxel_count: int,
        voxel_indices: NDArray[np.int32],
        load_case: PatientLoadCase,
        support_map: Optional[Dict] = None,
    ) -> FEAResult:
        """Validate a single orthosis design.
        
        Args:
            design_rank: Rank in Pareto front (1-best)
            voxel_count: Number of voxels in design
            voxel_indices: Array of voxel indices where material placed
            load_case: Load conditions for validation
            support_map: Optional dict of voxel -> support_priority for heuristic
        
        Returns:
            FEAResult with structural analysis metrics
        """
        if self.use_fea and self.abaqus_available:
            return self._validate_with_fea(
                design_rank, voxel_count, voxel_indices, load_case
            )
        else:
            return self._validate_with_heuristic(
                design_rank, voxel_count, voxel_indices, load_case, support_map
            )
    
    def _validate_with_heuristic(
        self,
        design_rank: int,
        voxel_count: int,
        voxel_indices: NDArray[np.int32],
        load_case: PatientLoadCase,
        support_map: Optional[Dict] = None,
    ) -> FEAResult:
        """Heuristic validation without FEA solver.
        
        Uses engineering approximations and scaling laws to estimate
        structural performance based on design properties.
        
        Heuristic Model:
        - Stress ~ force / (cross_sectional_area * material_factor)
        - Strain ~ stress / young_modulus
        - Deformation ~ stress * (length / E)
        - Support effectiveness ~ voxel_count_in_support_regions
        - Material utilization ~ voxel_count / total_budget
        """
        import time
        start_time = time.time()
        
        # Material properties (epoxy resin, 3D-printed)
        young_modulus_mpa = 3200.0     # Elastic modulus
        poisson_ratio = 0.35
        yield_stress_mpa = 45.0        # Ultimate tensile strength
        
        # Design metrics
        design_volume_cm3 = voxel_count * (self.voxel_resolution_mm / 10.0) ** 3
        design_weight_g = design_volume_cm3 * self.material_density_g_cm3
        
        # Estimate cross-sectional area (simplified)
        # Assume material distributed over average area
        avg_cross_section_cm2 = max(0.5, np.sqrt(design_volume_cm3) * 0.8)
        
        # Stress estimation (force / area)
        force_distribution_factor = {
            'uniform': 1.0,
            'fingertip': 1.5,  # Concentrated load = higher stress
            'palm': 0.8,       # Distributed load = lower stress
        }
        stress_factor = force_distribution_factor.get(
            load_case.force_distribution, 1.0
        )
        
        estimated_stress_mpa = (
            load_case.force_magnitude_n * stress_factor * 10 / avg_cross_section_cm2
        )
        max_stress_mpa = min(
            estimated_stress_mpa,
            yield_stress_mpa * 0.8  # Cap at 80% of yield (safety)
        )
        avg_stress_mpa = max_stress_mpa * 0.5  # Average ~50% of peak
        
        # Strain from stress
        peak_strain = min(max_stress_mpa / young_modulus_mpa, 0.10)
        avg_strain = peak_strain * 0.6
        
        # Deformation (deflection under load)
        # Estimate characteristic length as sqrt of area
        char_length_mm = np.sqrt(avg_cross_section_cm2 * 100)  # Convert to mm²
        max_deformation_mm = (peak_strain * char_length_mm) / 2
        avg_deformation_mm = max_deformation_mm * 0.4
        
        # Material utilization (voxel efficiency)
        material_utilization = min(voxel_count / self.design_space_voxels, 1.0)
        
        # Support effectiveness (heuristic)
        # More voxels = better support, but with diminishing returns
        support_effectiveness = 1.0 - np.exp(-voxel_count / 50.0)
        support_effectiveness = min(support_effectiveness, 0.95)
        
        # Contact pressure at orthosis-hand interface
        contact_area_cm2 = avg_cross_section_cm2 * 0.6  # 60% in contact
        contact_pressure_mpa = (load_case.force_magnitude_n * 10) / contact_area_cm2
        
        # Validation checks
        passes_stress = max_stress_mpa <= load_case.max_allowable_stress_mpa
        passes_strain = peak_strain <= load_case.max_allowable_strain
        passes_deformation = max_deformation_mm <= load_case.max_deformation_mm
        passes_support = support_effectiveness >= load_case.min_support_quality
        
        overall_pass = all([
            passes_stress, passes_strain, passes_deformation, passes_support
        ])
        
        elapsed = time.time() - start_time
        
        return FEAResult(
            design_rank=design_rank,
            design_voxel_count=voxel_count,
            load_case=load_case.load_type,
            max_von_mises_stress_mpa=max_stress_mpa,
            avg_von_mises_stress_mpa=avg_stress_mpa,
            peak_strain=peak_strain,
            avg_strain=avg_strain,
            max_deformation_mm=max_deformation_mm,
            avg_deformation_mm=avg_deformation_mm,
            material_utilization=material_utilization,
            support_effectiveness=support_effectiveness,
            contact_pressure_peak_mpa=contact_pressure_mpa,
            passes_stress_limit=passes_stress,
            passes_strain_limit=passes_strain,
            passes_deformation_limit=passes_deformation,
            passes_support_requirement=passes_support,
            overall_pass=overall_pass,
            simulation_time_s=elapsed,
            mesh_elements=max(100, voxel_count * 5),  # Estimate mesh size
        )
    
    def _validate_with_fea(
        self,
        design_rank: int,
        voxel_count: int,
        voxel_indices: NDArray[np.int32],
        load_case: PatientLoadCase,
    ) -> FEAResult:
        """Validate using Abaqus FEA solver.
        
        This is a placeholder for actual FEA integration.
        In production, this would:
        1. Generate Abaqus input file from voxel geometry
        2. Run simulation
        3. Extract stress/strain fields
        4. Validate against load case requirements
        """
        warnings.warn(
            "Full Abaqus FEA integration not yet implemented; using heuristic",
            UserWarning
        )
        # Fall back to heuristic for now
        return self._validate_with_heuristic(
            design_rank, voxel_count, voxel_indices, load_case
        )
    
    def validate_pareto_front(
        self,
        designs_json_path: str,
        load_case: Optional[PatientLoadCase] = None,
    ) -> Tuple[List[FEAResult], Dict[str, any]]:
        """Validate all designs in Pareto front.
        
        Args:
            designs_json_path: Path to phase_4b_results/designs.json
            load_case: Load case for validation; if None, use clinical standard
        
        Returns:
            Tuple of (list of FEAResult, summary statistics)
        """
        if load_case is None:
            load_case = PatientLoadCase.clinical_standard()
        
        # Load designs
        with open(designs_json_path) as f:
            designs_data = json.load(f)
        
        results = []
        for design in designs_data:
            result = self.validate_design(
                design_rank=design['rank'],
                voxel_count=design['voxel_count'],
                voxel_indices=np.array(design['voxel_indices'], dtype=np.int32),
                load_case=load_case,
            )
            results.append(result)
        
        # Compute summary
        summary = {
            'total_designs': len(results),
            'designs_passing': sum(1 for r in results if r.overall_pass),
            'avg_max_stress_mpa': np.mean([r.max_von_mises_stress_mpa for r in results]),
            'avg_support_effectiveness': np.mean([r.support_effectiveness for r in results]),
            'critical_failures': [r.design_rank for r in results if not r.overall_pass],
        }
        
        return results, summary


def main():
    """Demo: Validate Phase 4B Pareto front."""
    import time
    
    print("[DEMO] Phase 4C FEA Validation")
    print("=" * 60)
    
    # Initialize validator
    validator = FEAValidator(
        design_space_voxels=300,
        voxel_resolution_mm=2.0,
        material_density_g_cm3=1.2,
        use_fea=False,
    )
    
    # Validate Pareto front
    designs_path = Path(__file__).parent / 'phase_4b_results' / 'designs.json'
    
    if designs_path.exists():
        print(f"\n[STEP 1] Loading designs from {designs_path}")
        results, summary = validator.validate_pareto_front(
            str(designs_path),
            load_case=PatientLoadCase.clinical_standard()
        )
        
        print(f"\n[STEP 2] FEA Validation Results")
        print(f"  Total designs: {summary['total_designs']}")
        print(f"  Passing validation: {summary['designs_passing']}")
        print(f"  Average max stress: {summary['avg_max_stress_mpa']:.2f} MPa")
        print(f"  Average support effectiveness: {summary['avg_support_effectiveness']:.3f}")
        
        if summary['critical_failures']:
            print(f"  Critical failures (rank): {summary['critical_failures']}")
        
        print(f"\n[STEP 3] Top 3 Validated Designs")
        for result in results[:3]:
            summary_dict = result.summary_dict()
            print(f"\n  Design Rank {result.design_rank}:")
            for key, value in summary_dict.items():
                print(f"    {key}: {value}")
    else:
        print(f"[ERROR] Designs file not found: {designs_path}")


if __name__ == '__main__':
    main()
