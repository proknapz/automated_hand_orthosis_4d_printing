#!/usr/bin/env python3
"""Complete Phase 4C pipeline runner - works from any directory.

Runs the complete validation, fabrication, and reporting pipeline
on the Phase 4B Pareto-optimal designs.

Run from anywhere:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import and run
from FreiHAND.phase_4c_orchestrator import Phase4COrchestrator

if __name__ == '__main__':
    orchestrator = Phase4COrchestrator(
        project_root=str(PROJECT_ROOT / "FreiHAND"),
        output_dir="phase_4c_output",
        top_designs=None,  # Export all
        stl_format="binary",
    )
    
    results = orchestrator.run_complete_pipeline()
    
    # Summary
    if results['status'] == 'complete':
        print("\n" + "="*70)
        print("SUCCESS: Phase 4C Pipeline Complete!")
        print("="*70)
        print("\nOutput files:")
        print(f"  STL Files:  FreiHAND/phase_4c_output/stl_exports/")
        print(f"  Reports:    FreiHAND/phase_4c_output/reports/")
        print("\nView results:")
        print(f"  HTML Report: Open FreiHAND/phase_4c_output/reports/phase_4c_report.html")
        print(f"  JSON Data:   FreiHAND/phase_4c_output/reports/phase_4c_report.json")
        print("\nNext: Use STL files for 3D printing")
        print("="*70 + "\n")
    else:
        print(f"\nPipeline failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)
