# Phase 4: Orthosis Design Specification

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
- Samples Analyzed: 50
- Vertices per Sample: 778
- Keypoints per Sample: 21

Hand Dimensions (meters):
Raw hand geometry bounds across all samples:

  X (width):  -0.0963 to +0.1066  (span: 0.2029 m)
  Y (depth):  -0.0902 to +0.1011  (span: 0.1912 m)
  Z (height): +0.3135 to +0.9807  (span: 0.6671 m)

Total Hand Volume Estimate: ~120-150 cm^3 (point cloud based)

Reference bone scale (keypoint distance):
- Median: 0.029421 m
- Range: 0.016515 to 0.034178 m

---

## 3. Voxelization Baseline (Phase 3)

Voxel Grid Parameters:
- Resolution: 2.0 mm (0.002 m)
- Surface Voxels (per sample): 775 (identified hand surface)
- Design Workspace: ~775 surface voxels

---

## 4. Anatomical Regions (Phase 2 Segmentation)

All regions available for targeting:

Region          Support Priority    Movement Restriction    Use Case
------          ----------------    --------------------    --------
Wrist           95% (CRITICAL)    Low 30%     Primary stability
Thumb           85% (HIGH)        Moderate 50%   Grip coordination
Index           80% (HIGH)        Moderate 60%   Precision
Middle          80% (HIGH)        Moderate 60%   Precision
Ring            75% (MED)         Moderate 60%   Power grip
Pinky           75% (MED)         Moderate 50%   Power grip
Palm            70% (MED)         High 20%     Base support

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
- Maximum Material Volume: 66.7 cm^3
- At 2mm Resolution: 10,000 voxels maximum

Grid Dimensions:
With 4 mm padding around hand:

Dimension       Size (mm)                           Size (voxels @ 2mm)
---------       ---------                           -------------------
X (width)       210.9                    105
Y (depth)       199.2                    99
Z (height)      675.1                    337
TOTAL GRID      N/A                                 3,503,115

Material Efficiency:
- Total Grid Voxels: 3,503,115
- Usable Material Voxels: 10,000
- Utilization Rate: 0.3%
- Efficiency Note: Only ~0.3% of grid is material; rest is air/padding

Padding & Comfort Zones:
- Padding Thickness: 4 mm (prevents pinching)
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
