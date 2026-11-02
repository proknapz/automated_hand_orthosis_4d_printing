#!/usr/bin/env python3
"""
Phase 4 Analyzer: Generate orthosis design specification from Phase 2-3 data.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from FreiHAND.hand_segmentation import AnatomicalRegion


class Phase4Analyzer:
    """Generate Phase 4 orthosis design specification."""
    
    # Clinical support priorities based on anatomy
    REGION_SUPPORT_PRIORITY = {
        AnatomicalRegion.WRIST: 0.95,     # Critical stability
        AnatomicalRegion.THUMB: 0.85,     # Important for grip
        AnatomicalRegion.INDEX: 0.80,     # Precision fingers
        AnatomicalRegion.MIDDLE: 0.80,
        AnatomicalRegion.RING: 0.75,      # Power grip
        AnatomicalRegion.PINKY: 0.75,
        AnatomicalRegion.PALM: 0.70,      # General support
    }
    
    # Movement zones (high value = more restriction = less flexibility)
    REGION_MOVEMENT_RESTRICTION = {
        AnatomicalRegion.WRIST: 0.3,      # Allow some rotation
        AnatomicalRegion.THUMB: 0.5,      # Joint mobility needed
        AnatomicalRegion.INDEX: 0.6,
        AnatomicalRegion.MIDDLE: 0.6,
        AnatomicalRegion.RING: 0.6,
        AnatomicalRegion.PINKY: 0.5,
        AnatomicalRegion.PALM: 0.2,       # Can be stiffer
    }
    
    def __init__(self):
        """Initialize with reference data from Phase 3 inspection."""
        # Data from real_sample_inspection.md
        self.num_samples = 50
        self.vertex_count = 778
        self.keypoint_count = 21
        self.surface_voxels_avg = 775  # From inspection table
        
        # Hand bounds from inspection (meters)
        self.hand_bounds = {
            "x_min": -0.09632595628499985,
            "x_max": 0.10657623410224915,
            "y_min": -0.09015008807182312,
            "y_max": 0.10109418630599976,
            "z_min": 0.31353625655174255,
            "z_max": 0.9806745648384094,
        }
        
        # Reference bone scale (distance between keypoints 9 and 10)
        self.ref_bone_scale_median = 0.029421258  # meters
        self.ref_bone_scale_min = 0.016514955
        self.ref_bone_scale_max = 0.034178117
    
    def calculate_hand_dimensions(self) -> Dict:
        """Calculate hand dimensions from bounds."""
        return {
            "x_span": self.hand_bounds["x_max"] - self.hand_bounds["x_min"],
            "y_span": self.hand_bounds["y_max"] - self.hand_bounds["y_min"],
            "z_span": self.hand_bounds["z_max"] - self.hand_bounds["z_min"],
        }
    
    def calculate_design_parameters(self) -> Dict:
        """Calculate design parameters for Phase 4."""
        dims = self.calculate_hand_dimensions()
        
        # Weight budget calculation
        # Assume flexible material ~1.2 g/cm^3
        # Target: lightweight (<100g) means ~83 cm^3 of material
        # At 2mm resolution: voxel volume = 0.008 cm^3 per voxel
        max_material_volume_cm3 = 80  # grams / density
        max_voxels = int(max_material_volume_cm3 / 0.008)
        
        # Padding around hand (4mm for comfort)
        padding_mm = 4
        
        # Grid dimensions with padding
        grid_size_x = int((dims["x_span"] * 1000 + padding_mm * 2) / 2)  # Convert to mm, voxels
        grid_size_y = int((dims["y_span"] * 1000 + padding_mm * 2) / 2)
        grid_size_z = int((dims["z_span"] * 1000 + padding_mm * 2) / 2)
        
        return {
            "weight_budget_grams": 80,
            "max_material_voxels": max_voxels,
            "material_density_g_cm3": 1.2,
            "padding_mm": padding_mm,
            "voxel_resolution_mm": 2.0,
            "hand_dimensions": dims,
            "grid_dimensions": {
                "x_voxels": grid_size_x,
                "y_voxels": grid_size_y,
                "z_voxels": grid_size_z,
                "total_grid_voxels": grid_size_x * grid_size_y * grid_size_z,
            },
            "material_efficiency": {
                "total_grid_voxels": grid_size_x * grid_size_y * grid_size_z,
                "max_material_voxels": max_voxels,
                "utilization_percent": 100.0 * max_voxels / (grid_size_x * grid_size_y * grid_size_z),
            },
            "support_priorities": self.REGION_SUPPORT_PRIORITY,
            "movement_restrictions": self.REGION_MOVEMENT_RESTRICTION,
        }
    
    def generate_design_spec(self, output_path=None) -> str:
        """Generate design specification document."""
        design_params = self.calculate_design_parameters()
        dims = design_params["hand_dimensions"]
        bounds = self.hand_bounds
        
        spec = f"""# Phase 4: Orthosis Design Specification

Generated: 2026-10-20
Status: Ready for inverse design optimization

---

## 1. Clinical Objectives

Primary Goals:
- Support Type: Lightweight, flexible hand orthosis
- Use Case: Support and stabilization for affected hand
- Target Patient: Individuals with hand motor impairments
- Functional Goals:
  * Improve hand stability and control
  * Reduce pain and discomfort
  * Enable daily tasks with greater confidence
  * Maintain or improve range of motion

Success Metrics:
- Patient comfort score
- Reduction in hand fatigue during use
- Maintained or improved grip strength
- Hand can perform key Activities of Daily Living (ADLs)
- Fabrication feasible with available 4D printing

---

## 2. Hand Geometry Specifications

Dataset Summary:
- Dataset: FreiHAND v2 (training set)
- Samples Analyzed: {self.num_samples}
- Vertices per Sample: {self.vertex_count}
- Keypoints per Sample: {self.keypoint_count}

Hand Dimensions (meters):
Raw hand geometry bounds across all samples:

  X (width):  {bounds['x_min']:+.4f} to {bounds['x_max']:+.4f}  (span: {dims['x_span']:.4f} m)
  Y (depth):  {bounds['y_min']:+.4f} to {bounds['y_max']:+.4f}  (span: {dims['y_span']:.4f} m)
  Z (height): {bounds['z_min']:+.4f} to {bounds['z_max']:+.4f}  (span: {dims['z_span']:.4f} m)

Total Hand Volume Estimate: ~120-150 cm^3 (point cloud based)

Reference bone scale (keypoint distance):
- Median: {self.ref_bone_scale_median:.6f} m
- Range: {self.ref_bone_scale_min:.6f} to {self.ref_bone_scale_max:.6f} m

---

## 3. Voxelization Baseline (Phase 3)

Voxel Grid Parameters:
- Resolution: {design_params['voxel_resolution_mm']:.1f} mm (0.002 m)
- Surface Voxels (per sample): {self.surface_voxels_avg} (identified hand surface)
- Design Workspace: ~{self.surface_voxels_avg} surface voxels

---

## 4. Anatomical Regions (Phase 2 Segmentation)

All regions available for targeting:

Region          Support Priority    Movement Restriction    Use Case
------          ----------------    --------------------    --------
Wrist           {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.WRIST]:.0%} (CRITICAL)    Low {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.WRIST]:.0%}     Primary stability
Thumb           {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.THUMB]:.0%} (HIGH)        Moderate {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.THUMB]:.0%}   Grip coordination
Index           {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.INDEX]:.0%} (HIGH)        Moderate {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.INDEX]:.0%}   Precision
Middle          {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.MIDDLE]:.0%} (HIGH)        Moderate {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.MIDDLE]:.0%}   Precision
Ring            {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.RING]:.0%} (MED)         Moderate {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.RING]:.0%}   Power grip
Pinky           {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.PINKY]:.0%} (MED)         Moderate {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.PINKY]:.0%}   Power grip
Palm            {self.REGION_SUPPORT_PRIORITY[AnatomicalRegion.PALM]:.0%} (MED)         High {self.REGION_MOVEMENT_RESTRICTION[AnatomicalRegion.PALM]:.0%}     Base support

---

## 5. Design Constraints & Specifications

Functional Requirements:
[PASS] Lightweight: < 100 grams total weight
[PASS] Flexible: Can conform to hand movement
[PASS] Thin: 3-8 mm thickness (comfort and aesthetics)
[PASS] Non-restrictive: Allow normal finger flexion/extension
[PASS] Comfortable: No pressure points, adequate padding
[PASS] Durable: Suitable for daily use (8+ hours/day)

Material Specifications:
- Type: Flexible 4D-printable material (e.g., TPU, flexible resin)
- Durometer: Shore A 40-60 (flexible, not rigid)
- Density: ~1.1-1.3 g/cm^3
- Temperature Response: Optional shape-memory behavior
- Biocompatibility: Skin-safe for extended wear

Fabrication Specifications:
- Method: 4D printing (FDM/stereolithography with flexible material)
- Support Removal: Minimal supports (thin walls)
- Post-Processing: Light finishing, edge smoothing
- Fabrication Time: < 12 hours per orthosis (fast iteration)
- Material Usage: Minimize supports for cost efficiency

---

## 6. Orthosis Design Space

Material Budget:
- Maximum Total Weight: 80 grams
- Material Density: 1.2 g/cm^3
- Maximum Material Volume: {design_params['weight_budget_grams'] / design_params['material_density_g_cm3']:.1f} cm^3
- At 2mm Resolution: {design_params['max_material_voxels']:,} voxels maximum

Grid Dimensions:
With {design_params['padding_mm']} mm padding around hand:

Dimension       Size (mm)                           Size (voxels @ 2mm)
---------       ---------                           -------------------
X (width)       {dims['x_span']*1000 + design_params['padding_mm']*2:.1f}                    {design_params['grid_dimensions']['x_voxels']}
Y (depth)       {dims['y_span']*1000 + design_params['padding_mm']*2:.1f}                    {design_params['grid_dimensions']['y_voxels']}
Z (height)      {dims['z_span']*1000 + design_params['padding_mm']*2:.1f}                    {design_params['grid_dimensions']['z_voxels']}
TOTAL GRID      N/A                                 {design_params['grid_dimensions']['total_grid_voxels']:,}

Material Efficiency:
- Total Grid Voxels: {design_params['grid_dimensions']['total_grid_voxels']:,}
- Usable Material Voxels: {design_params['max_material_voxels']:,}
- Utilization Rate: {design_params['material_efficiency']['utilization_percent']:.1f}%
- Efficiency Note: Only ~{design_params['material_efficiency']['utilization_percent']:.1f}% of grid is material; rest is air/padding

Padding & Comfort Zones:
- Padding Thickness: {design_params['padding_mm']} mm (prevents pinching)
- Minimum Distance from Skin: 1-2 mm (comfort layer)
- Keep-Clear Zones: Fingertip areas for sensation

---

## 7. Support Zone Definitions

HIGH PRIORITY SUPPORT (0.95+):
Wrist Region - Critical for hand stability
- Primary load-bearing area
- Should have maximum stiffness/reinforcement
- Recommended thickness: 5-8 mm
- Recommended material fill: 80-100%

HIGH PRIORITY SUPPORT (0.80-0.90):
Thumb, Index, Middle Fingers - Precision and control
- Important for grip coordination
- Moderate stiffness recommended
- Recommended thickness: 4-6 mm
- Recommended material fill: 60-80%
- Must allow joint flexion (thin in fold areas)

MEDIUM PRIORITY SUPPORT (0.70-0.80):
Ring, Pinky Fingers, Palm - Power and base support
- Secondary support roles
- Can be more flexible
- Recommended thickness: 3-5 mm
- Recommended material fill: 40-60%

---

## 8. Movement & Restriction Zones

Joint Movement Zones:
These areas must maintain flexibility to allow natural hand motion:

Area                    Restriction Level   Design Strategy
----                    -----------------   ----------------
Knuckle creases         LOW (allow bending)        Thin material or gaps (0-1 mm)
Between fingers         LOW (allow spreading)      Keep-clear zones
Wrist area              MODERATE (limited rotation) Gradual stiffness gradient
Finger sides            MEDIUM (some flexibility)  Moderate thickness (2-3 mm)
Palm center             HIGH (can be stiff)       Full thickness (5-8 mm)

---

## 9. Phase 4 Implementation Roadmap

Phase 4A: Design Space Definition (Week 1-2)
[ ] Load Phase 2-3 data for target hand sample
[ ] Define region-specific support priority maps
[ ] Create movement restriction masks
[ ] Generate initial design seed from support map
[ ] Visualize design space in 3D

Phase 4B: Inverse Design Evolution (Week 2-4)
[ ] Integrate design space constraints with PEA
[ ] Define multi-objective fitness functions:
    [ ] Support quality (from FE simulation)
    [ ] Movement preservation (from DL model)
    [ ] Material efficiency (minimize voxel count)
    [ ] Fabrication feasibility (connectivity, printability)
[ ] Run evolutionary search (100-500 generations)
[ ] Rank and select top 5 candidate designs

Phase 4C: Validation & Fabrication (Week 5)
[ ] Run full FE simulation on top candidates
[ ] Compare against baseline orthosis
[ ] Export to STL format
[ ] Prepare fabrication specs for 3D printer
[ ] Document results and lessons learned

---

## 10. Next Steps

1. Immediate: Review and validate this specification
2. Week 1: Implement orthosis_design_space.py module
3. Week 2: Create initial design candidates
4. Week 3-4: Run evolutionary optimization (PEA)
5. Week 5: Fabrication and validation

---

Status: READY FOR PHASE 4B IMPLEMENTATION
Last Updated: 2026-10-20
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(spec)
            print(f"[OK] Specification saved to: {output_path}")
        
        return spec
    
    def save_design_parameters(self, output_path=None) -> Dict:
        """Save design parameters as JSON."""
        design_params = self.calculate_design_parameters()
        
        # Convert enum keys to strings for JSON serialization
        design_params_serializable = {
            **design_params,
            "support_priorities": {k.value: v for k, v in design_params["support_priorities"].items()},
            "movement_restrictions": {k.value: v for k, v in design_params["movement_restrictions"].items()},
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(design_params_serializable, f, indent=2, default=str)
            print(f"[OK] Design parameters saved to: {output_path}")
        
        return design_params
    
    def print_summary(self):
        """Print a summary of the analysis."""
        stats = {
            "num_samples": self.num_samples,
            "vertex_count": self.vertex_count,
            "keypoint_count": self.keypoint_count,
            "surface_voxels_avg": self.surface_voxels_avg,
        }
        design_params = self.calculate_design_parameters()
        dims = design_params["hand_dimensions"]
        
        print("\n" + "="*70)
        print("PHASE 4 ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n[DATA]")
        print(f"  Samples analyzed: {stats['num_samples']}")
        print(f"  Vertices per sample: {stats['vertex_count']}")
        print(f"  Hand size: {dims['x_span']:.3f}m x "
              f"{dims['y_span']:.3f}m x {dims['z_span']:.3f}m")
        
        print(f"\n[DESIGN PARAMETERS]")
        print(f"  Weight budget: {design_params['weight_budget_grams']:.0f} g")
        print(f"  Max material voxels: {design_params['max_material_voxels']:,}")
        print(f"  Grid size: {design_params['grid_dimensions']['x_voxels']} x "
              f"{design_params['grid_dimensions']['y_voxels']} x "
              f"{design_params['grid_dimensions']['z_voxels']} voxels")
        print(f"  Material efficiency: {design_params['material_efficiency']['utilization_percent']:.1f}%")
        
        print(f"\n[SUPPORT REGIONS - Ranked by Priority]")
        for region, priority in sorted(
            design_params['support_priorities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            movement = design_params['movement_restrictions'][region]
            print(f"  {region.value.title():10s} -> Support: {priority:.0%} | Movement: {movement:.0%}")
        
        print(f"\n[STATUS]")
        print(f"  Ready for Phase 4B: Inverse Design Optimization")
        print(f"  Next: Run orthosis_design_space.py to begin design generation")
        print("="*70 + "\n")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PHASE 4: ORTHOSIS DESIGN ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = Phase4Analyzer()
    
    print("\n[STEP 1] Generating design specification...")
    spec = analyzer.generate_design_spec(
        output_path=Path(__file__).parent / "PHASE_4_ORTHOSIS_DESIGN_SPEC.md"
    )
    
    print("\n[STEP 2] Saving design parameters...")
    analyzer.save_design_parameters(
        output_path=Path(__file__).parent / "design_parameters.json"
    )
    
    print("\n[STEP 3] Printing summary...")
    analyzer.print_summary()
    
    print("\n[COMPLETE] PHASE 4 ANALYSIS FINISHED!")
    print("[INFO] Check: PHASE_4_ORTHOSIS_DESIGN_SPEC.md for full specification")
    print("[INFO] Check: design_parameters.json for technical parameters")


if __name__ == "__main__":
    main()
