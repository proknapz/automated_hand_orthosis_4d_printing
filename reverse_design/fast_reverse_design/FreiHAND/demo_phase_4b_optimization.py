#!/usr/bin/env python3
"""Demo: Phase 4B Inverse Design Optimization with Mock Data.

This script demonstrates the complete Phase 4B workflow:
1. Load (or create mock) design space from Phase 4A
2. Initialize PEA optimizer with population and evolution settings
3. Run multi-generation evolutionary optimization
4. Extract and rank Pareto-optimal designs
5. Export results for Phase 4C fabrication

Usage:
    python demo_phase_4b_optimization.py
"""

import json
from pathlib import Path

import numpy as np

from FreiHAND.orthosis_design_space import DesignSpace, DesignZoneType
from FreiHAND.pea_inverse_design import (
    DesignEncoding,
    PEAOrthosisOptimizer,
    PopulationStatistics,
)


def create_demo_design_space() -> DesignSpace:
    """Create a mock design space for demonstration."""
    print("[STEP 1] Creating mock design space...")
    
    # Create constraint maps
    zone_map = {}
    support_map = {}
    movement_map = {}
    thickness_map = {}
    
    # Populate with 300 voxels simulating hand anatomy
    # Higher support priority near "wrist" region
    for i in range(300):
        x, y, z = (i % 10, (i // 10) % 10, (i // 100) % 3)
        pos = (x, y, z)
        
        # Zones based on position (simulate wrist at top)
        if z == 2:
            zone_type = DesignZoneType.REINFORCE
            base_priority = 0.95
        elif z == 1:
            zone_type = DesignZoneType.SUPPORT
            base_priority = 0.80
        else:
            zone_type = DesignZoneType.FLEXIBLE
            base_priority = 0.70
        
        zone_map[pos] = zone_type
        support_map[pos] = base_priority + np.random.uniform(-0.05, 0.05)
        movement_map[pos] = 0.3 + np.random.uniform(0.0, 0.4)
        thickness_map[pos] = (1.0, 6.0)
    
    # Create seed candidates
    seed1 = np.array(list(range(0, 50)), dtype=np.int32)    # Top region (wrist)
    seed2 = np.array(list(range(50, 150)), dtype=np.int32)  # Mid region
    seed3 = np.array(list(range(0, 30)), dtype=np.int32)    # Small/minimal
    
    ds = DesignSpace(
        grid_shape=(10, 10, 3),
        grid_origin=np.array([0.0, 0.0, 0.0]),
        voxel_resolution=0.002,  # 2mm
        zone_map=zone_map,
        support_map=support_map,
        movement_map=movement_map,
        thickness_map=thickness_map,
        constraints=[],
        max_material_voxels=200,
        material_density=1.2,
        design_params={
            "weight_budget_grams": 80,
            "support_priorities": {
                "wrist": 0.95,
                "thumb": 0.85,
                "index": 0.80,
                "middle": 0.80,
                "ring": 0.75,
                "pinky": 0.75,
                "palm": 0.70,
            },
        },
        seed_candidates=[seed1, seed2, seed3],
    )
    
    print(f"  [OK] Created design space: {len(zone_map)} voxels")
    print(f"       Grid: {ds.grid_shape}, Max material: {ds.max_material_voxels}")
    print()
    
    return ds


def run_optimization_demo():
    """Run complete Phase 4B optimization demo."""
    print("="*80)
    print("PHASE 4B DEMO: INVERSE DESIGN OPTIMIZATION WITH PEA")
    print("="*80)
    print()
    
    # Step 1: Load/create design space
    design_space = create_demo_design_space()
    
    # Step 2: Initialize optimizer
    print("[STEP 2] Initializing PEA optimizer...")
    optimizer = PEAOrthosisOptimizer(
        design_space=design_space,
        population_size=30,       # Small for demo
        generations=5,            # Quick run
        encoding=DesignEncoding.VOXEL_SET,
    )
    print(f"  [OK] Optimizer configured")
    print(f"       Population size: {optimizer.population_size}")
    print(f"       Generations: {optimizer.max_generations}")
    print()
    
    # Step 3: Run optimization
    print("[STEP 3] Running evolutionary optimization...")
    pareto_front, generation_history = optimizer.optimize()
    print()
    
    # Step 4: Analyze results
    print("[STEP 4] Analyzing results...")
    print(f"  [OK] Final Pareto front size: {len(pareto_front)}")
    print()
    
    # Display top designs
    print("  Top Pareto-Optimal Designs:")
    print("  " + "-" * 76)
    print(f"  {'Rank':<5} {'Voxels':<8} {'Weight':<10} {'Support':<10} {'Movement':<10} {'Efficiency':<10}")
    print("  " + "-" * 76)
    
    for rank, design in enumerate(sorted(pareto_front, 
                                         key=lambda d: d.support_quality or 0.0,
                                         reverse=True)[:5]):
        weight = design.weight_grams()
        support = design.support_quality or 0.0
        movement = design.movement_preservation or 0.0
        efficiency = design.material_efficiency or 0.0
        
        print(f"  {rank+1:<5} {design.voxel_count():<8} {weight:>8.2f}g {support:>9.3f} {movement:>9.3f} {efficiency:>9.3f}")
    print("  " + "-" * 76)
    print()
    
    # Generation-wise statistics
    print("  Generation Statistics:")
    print("  " + "-" * 76)
    print(f"  {'Gen':<4} {'Pop':<4} {'Avg Vox':<10} {'Support':<10} {'Movement':<10} {'Pareto':<8}")
    print("  " + "-" * 76)
    
    for stats in generation_history:
        print(f"  {stats.generation:<4} {stats.population_size:<4} {stats.avg_voxel_count:>8.1f} "
              f"{stats.avg_support:>9.3f} {stats.avg_movement:>9.3f} {stats.pareto_front_size:>7}")
    print("  " + "-" * 76)
    print()
    
    # Step 5: Export results
    print("[STEP 5] Exporting results...")
    output_dir = Path("FreiHAND") / "phase_4b_results"
    optimizer.export_results(output_dir)
    
    # Display exported files
    if output_dir.exists():
        rel_path = str(output_dir)
        print(f"  [OK] Results exported to: {rel_path}")
        for file in sorted(output_dir.glob("*.json")):
            size = file.stat().st_size
            print(f"       - {file.name} ({size:,} bytes)")
    print()
    
    print("="*80)
    print("PHASE 4B DEMO COMPLETE")
    print("="*80)
    print()
    print("Next Steps:")
    print("  1. Review Pareto-optimal designs in phase_4b_results/designs.json")
    print("  2. Select top candidates for Phase 4C FEA validation")
    print("  3. Export STL for 3D printing")
    print("  4. Validate in FEA simulation")
    print("  5. Fabricate and clinically test")
    print()


if __name__ == "__main__":
    run_optimization_demo()
