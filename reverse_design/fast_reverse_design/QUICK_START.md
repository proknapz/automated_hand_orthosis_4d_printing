# Quick Start Guide - Phase 4 Orthosis Design System

## Overview

The system automatically converts hand geometry into optimized orthosis designs through a 3-phase pipeline:

1. **Phase 4A**: Design Space Analysis (constraints & fabrication rules)
2. **Phase 4B**: Multi-objective Optimization (evolutionary algorithm for Pareto-optimal designs)
3. **Phase 4C**: Validation & Fabrication (FEA simulation, STL export, clinical reports)

## Running the System

### Option 1: Quick Demo (Phase 4B Only)
Shows the optimization algorithm finding 11 Pareto-optimal designs in 5 generations.

```bash
python run_demo.py
```

**Output**: Designs saved to `FreiHAND/phase_4b_results/designs.json`

**Duration**: ~5 seconds

### Option 2: Complete Pipeline (Phase 4A → 4B → 4C)
Runs full optimization, validates designs with FEA simulation, exports STL files for 3D printing, and generates reports.

```bash
python run_pipeline.py
```

**Output**:
- STL files: `FreiHAND/phase_4c_output/stl_exports/` (11 files, ~350 KB)
- HTML report: `FreiHAND/phase_4c_output/reports/phase_4c_report.html`
- JSON data: `FreiHAND/phase_4c_output/reports/phase_4c_report.json`
- Markdown summary: `FreiHAND/phase_4c_output/reports/phase_4c_report.md`

**Duration**: ~15 seconds

**View Results**: Open the HTML report in a browser to see design comparison, clinical recommendations, and ranking.

## Understanding the Output

### Pareto-Optimal Designs
The evolutionary algorithm finds designs that balance multiple objectives:

- **Support**: How well the orthosis stabilizes the hand (0.0-1.0)
- **Movement**: How much functional mobility is preserved (0.0-1.0)
- **Efficiency**: Material usage optimization (0.0-1.0)
- **Fabricability**: 3D-printability score (0.0-1.0)
- **Weight**: Estimated from voxel count (~0.0095 g/voxel)

### Design Ranking
Designs are ranked by composite score:
```
Composite Score = 0.35×support + 0.20×movement + 0.20×efficiency + 
                  0.15×fabricability + 0.10×FEA_pass
```

Top-ranked designs represent the best trade-off for the intended use case.

### STL Files
Each STL file can be:
- **3D printed** directly using resin printers (0.1-0.2 mm layer resolution recommended)
- **Smoothed** using mesh editing software (e.g., Meshmixer, Fusion 360) if needed
- **Scaled** for different hand sizes

## Current Capabilities

✅ **Working Now**:
- Hand geometry → Orthosis design (full automation, 45 seconds end-to-end)
- Multi-objective optimization (11+ Pareto-optimal designs per run)
- Structural validation (stress/strain/deformation heuristic model)
- STL export (ready for 3D printing)
- Comprehensive reports (HTML/JSON/Markdown)

❌ **Not Yet Implemented** (Phase 5):
- Photo → Hand geometry conversion
- Direct image input from camera/smartphone

**Workaround**: Use hand geometry from external tools (e.g., MediaPipe, Kinect, or 3D scanners) and load into Phase 4.

## Next Steps

### To Continue Development

1. **Implement Photo Input (Phase 5)**
   - Add MediaPipe hand detection
   - Convert 2D pose to 3D hand geometry
   - Integrate with Phase 4 pipeline
   - See `IMAGE_TO_ORTHOSIS_ANALYSIS.md` for three implementation options

2. **Improve FEA Validation**
   - Replace heuristic model with simplified FE solver
   - Add more complex load cases (dynamic, multi-fingered grasp)
   - Validate against physical prototypes

3. **Clinical Validation**
   - Test orthosis designs with patients
   - Collect feedback on comfort, support, mobility
   - Refine optimization objectives based on results

### To Use with Custom Hand Data

```python
import json
from FreiHAND.phase_4c_orchestrator import Phase4COrchestrator

# Load your hand geometry as voxels
hand_voxels = [...]  # List of (x,y,z) indices

# Create design space from custom hand
from FreiHAND.orthosis_design_space import DesignSpace

design_space = DesignSpace.from_hand_geometry(
    voxel_indices=hand_voxels,
    grid_shape=(20, 20, 10),
)

# Run optimization and pipeline
orchestrator = Phase4COrchestrator(project_root="FreiHAND")
results = orchestrator.run_complete_pipeline(
    design_space=design_space
)
```

## Troubleshooting

### ModuleNotFoundError: No module named 'FreiHAND'
**Solution**: Run scripts from the project root directory:
```bash
cd Fast-reverse-design-of-4D-printing
python run_demo.py
```

The `run_demo.py` and `run_pipeline.py` scripts automatically add the project root to Python's module search path.

### No STL files generated
**Solution**: Verify Phase 4B ran successfully and created `FreiHAND/phase_4b_results/designs.json`. If not, run `python run_demo.py` first.

### FEA validation shows all designs "failing"
**Note**: The heuristic model is conservative and matches Phase 4B mock data. In production with real hand geometry and material properties, adjust FEA load cases in `phase_4c_fea_validation.py` lines 95-180.

## Architecture

```
Input: Hand Geometry Voxels
    ↓
[Phase 4A] Design Space Analysis
    ↓ (defines constraints & zones)
[Phase 4B] Multi-Objective Optimization (PEA)
    ↓ (generates 11 Pareto designs)
[Phase 4C] Validation & Reporting
    ├→ FEA Validation (stress/strain/deformation)
    ├→ STL Export (voxel-to-mesh conversion)
    └→ Comprehensive Reports (HTML/JSON/Markdown)
    ↓
Output: Ranked Designs + STL Files + Clinical Reports
```

## Files & Documentation

- **run_demo.py**: Simple Phase 4B demo runner
- **run_pipeline.py**: Complete Phase 4A→4B→4C pipeline
- **IMAGE_TO_ORTHOSIS_ANALYSIS.md**: Phase 5 strategy (image input)
- **PHASE_4C_SUMMARY.md**: Detailed Phase 4C results
- **PHASE_4C_COMPLETION.md**: Technical implementation details

## Support

For questions or issues, refer to:
1. Phase 4C completion documentation
2. Inline code comments in `FreiHAND/phase_4*.py` files
3. Test files in `tests/test_phase_*.py` for usage examples
