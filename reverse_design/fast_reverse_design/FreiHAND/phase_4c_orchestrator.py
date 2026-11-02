#!/usr/bin/env python3
"""Phase 4C: Complete Validation, Fabrication & Reporting Pipeline.

This script orchestrates the complete Phase 4C workflow:
1. FEA Validation of Pareto-optimal designs
2. STL Export for 3D printing
3. Comprehensive multi-perspective reporting
4. Clinical recommendations and fabrication specifications

Outputs:
- HTML/JSON/Markdown comparison reports
- STL files ready for 3D printing
- FEA validation results with safety margins
- Clinical use case recommendations

Usage:
    python phase_4c_orchestrator.py [--top-designs N] [--stl-format FORMAT]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from phase_4c_fea_validation import FEAValidator, PatientLoadCase, FEAResult
from phase_4c_stl_export import STLExportPipeline
from phase_4c_comprehensive_report import (
    ComprehensiveAnalyzer,
    ReportGenerator,
    DesignRanking,
)


class Phase4COrchestrator:
    """Orchestrates complete Phase 4C pipeline."""
    
    def __init__(
        self,
        project_root: str = ".",
        output_dir: str = "phase_4c_output",
        top_designs: int = None,
        stl_format: str = "binary",
    ):
        """Initialize orchestrator.
        
        Args:
            project_root: Path to FreiHAND project root
            output_dir: Output directory for all Phase 4C results
            top_designs: Export only top N designs; None = all
            stl_format: 'binary' or 'ascii' STL format
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.designs_json = self.project_root / "phase_4b_results" / "designs.json"
        self.top_designs = top_designs
        self.stl_format = stl_format
        
        # Create subdirectories
        self.stl_dir = self.output_dir / "stl_exports"
        self.reports_dir = self.output_dir / "reports"
        self.stl_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_complete_pipeline(self) -> Dict:
        """Run complete Phase 4C pipeline.
        
        Returns:
            Dictionary with results and file paths
        """
        results = {
            'status': 'running',
            'fea_results': None,
            'stl_exports': None,
            'rankings': None,
            'reports': {},
        }
        
        try:
            print("\n" + "=" * 70)
            print("PHASE 4C: COMPLETE VALIDATION & FABRICATION PIPELINE")
            print("=" * 70)
            
            # Step 1: FEA Validation
            print("\n[STEP 1/4] FEA Structural Validation")
            print("-" * 70)
            fea_results, fea_summary = self._run_fea_validation()
            results['fea_results'] = fea_results
            print(f"\n  [OK] Validated {fea_summary['total_designs']} designs")
            print(f"       Average max stress: {fea_summary['avg_max_stress_mpa']:.2f} MPa")
            print(f"       Passing designs: {fea_summary['designs_passing']}/{fea_summary['total_designs']}")
            
            # Step 2: STL Export
            print("\n[STEP 2/4] STL Export for 3D Printing")
            print("-" * 70)
            stl_results = self._run_stl_export()
            results['stl_exports'] = stl_results
            print(f"\n  [OK] Exported {len(stl_results)} STL files")
            print(f"       Location: {self.stl_dir}")
            
            # Step 3: Comprehensive Analysis
            print("\n[STEP 3/4] Comprehensive Analysis & Ranking")
            print("-" * 70)
            rankings = self._run_comprehensive_analysis(fea_results, stl_results)
            results['rankings'] = rankings
            print(f"\n  [OK] Ranked {len(rankings)} designs")
            print(f"       Top design: Rank {rankings[0].rank_in_pareto} "
                  f"(composite score: {rankings[0].composite_score:.3f})")
            print(f"       Recommendation: {rankings[0].clinical_recommendation[:70]}...")
            
            # Step 4: Report Generation
            print("\n[STEP 4/4] Report Generation")
            print("-" * 70)
            reports = self._generate_reports(rankings)
            results['reports'] = reports
            print(f"\n  [OK] Generated {len(reports)} reports")
            for report_type, path in reports.items():
                print(f"       - {report_type}: {path}")
            
            # Summary
            print("\n" + "=" * 70)
            print("PHASE 4C COMPLETE")
            print("=" * 70)
            print("\n[SUMMARY]")
            print(f"  Total designs analyzed: {len(rankings)}")
            print(f"  FEA validation: {fea_summary['designs_passing']} passing, "
                  f"{fea_summary['total_designs'] - fea_summary['designs_passing']} failing")
            print(f"  STL exports: {len(stl_results)} files in {self.stl_dir}")
            print(f"  Reports: {len(reports)} files in {self.reports_dir}")
            print(f"\n  Recommended design: Rank {rankings[0].rank_in_pareto}")
            print(f"  Use case: {rankings[0].clinical_recommendation}")
            
            results['status'] = 'complete'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            print(f"\n[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _run_fea_validation(self) -> Tuple[List[FEAResult], Dict]:
        """Run FEA validation on Pareto front.
        
        Returns:
            Tuple of (FEA results list, summary dict)
        """
        validator = FEAValidator(
            design_space_voxels=300,
            voxel_resolution_mm=2.0,
            material_density_g_cm3=1.2,
            use_fea=False,  # Use heuristic; set to True if Abaqus available
        )
        
        load_case = PatientLoadCase.clinical_standard()
        
        fea_results, fea_summary = validator.validate_pareto_front(
            str(self.designs_json),
            load_case=load_case,
        )
        
        return fea_results, fea_summary
    
    def _run_stl_export(self) -> Dict[int, Dict]:
        """Run STL export for Pareto front.
        
        Returns:
            Dictionary mapping design rank to export metadata
        """
        pipeline = STLExportPipeline(
            output_dir=str(self.stl_dir),
            voxel_resolution_mm=2.0,
            apply_smoothing=True,
            export_format=self.stl_format,
        )
        
        stl_results = pipeline.export_pareto_front(
            str(self.designs_json),
            top_n=self.top_designs,
        )
        
        return stl_results
    
    def _run_comprehensive_analysis(
        self,
        fea_results: List[FEAResult],
        stl_results: Dict,
    ) -> List[DesignRanking]:
        """Run comprehensive analysis and ranking.
        
        Returns:
            List of DesignRanking sorted by composite score
        """
        analyzer = ComprehensiveAnalyzer(str(self.designs_json))
        rankings = analyzer.analyze_complete_pipeline(stl_results, fea_results)
        return rankings
    
    def _generate_reports(self, rankings: List[DesignRanking]) -> Dict[str, str]:
        """Generate all reports.
        
        Args:
            rankings: List of DesignRanking
        
        Returns:
            Dictionary mapping report type to file path
        """
        reports = {}
        
        # HTML report
        html_path = ReportGenerator.generate_html_report(
            rankings,
            output_path=str(self.reports_dir / "phase_4c_report.html"),
        )
        reports['html'] = html_path
        
        # JSON report
        json_path = ReportGenerator.generate_json_report(
            rankings,
            output_path=str(self.reports_dir / "phase_4c_report.json"),
        )
        reports['json'] = json_path
        
        # Markdown report
        md_path = ReportGenerator.generate_markdown_report(
            rankings,
            output_path=str(self.reports_dir / "phase_4c_report.md"),
        )
        reports['markdown'] = md_path
        
        return reports


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 4C: Orthosis Design Validation & Fabrication Pipeline"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to FreiHAND project root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="phase_4c_output",
        help="Output directory for Phase 4C results",
    )
    parser.add_argument(
        "--top-designs",
        type=int,
        default=None,
        help="Export only top N designs (default: all)",
    )
    parser.add_argument(
        "--stl-format",
        type=str,
        choices=["binary", "ascii"],
        default="binary",
        help="STL export format",
    )
    
    args = parser.parse_args()
    
    # Run orchestrator
    orchestrator = Phase4COrchestrator(
        project_root=args.project_root,
        output_dir=args.output_dir,
        top_designs=args.top_designs,
        stl_format=args.stl_format,
    )
    
    results = orchestrator.run_complete_pipeline()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if results['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
