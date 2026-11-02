"""Phase 4C: Comprehensive Reporting and Analysis Module.

Generates detailed comparison reports for all Pareto-optimal orthosis designs,
integrating FEA validation and STL export metadata:

1. Load Phase 4B Pareto designs
2. Perform FEA validation on top candidates
3. Generate STL exports
4. Create multi-perspective comparison analysis
5. Export PDF/HTML/JSON reports
6. Provide clinical recommendations

Report Sections:
- Executive Summary (key findings, recommendations)
- Design Comparison Matrix (all objectives vs. all designs)
- FEA Validation Results (stress, strain, safety margins)
- Fabrication Specifications (print time, material, supports)
- Clinical Recommendations (best use cases for each design)
- Appendices (full design data, raw metrics)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from phase_4c_fea_validation import FEAValidator, PatientLoadCase, FEAResult
from phase_4c_stl_export import STLExportPipeline


@dataclass
class DesignRanking:
    """Ranking of a single design across multiple criteria."""
    
    rank_in_pareto: int
    voxel_count: int
    weight_grams: float
    
    # Phase 4B objectives
    support_quality: float
    movement_preservation: float
    material_efficiency: float
    fabrication_feasibility: float
    
    # Phase 4C FEA validation
    fea_result: Optional[FEAResult] = None
    
    # Phase 4C fabrication
    stl_vertices: Optional[int] = None
    stl_volume_mm3: Optional[float] = None
    
    # Overall scoring
    composite_score: float = 0.0
    clinical_recommendation: str = ""
    
    def compute_composite_score(
        self,
        w_support: float = 0.35,
        w_movement: float = 0.20,
        w_efficiency: float = 0.20,
        w_feasibility: float = 0.15,
        w_fea_pass: float = 0.10,
    ) -> float:
        """Compute weighted composite score.
        
        Combines all Phase 4B and 4C metrics into single score.
        
        Args:
            w_*: Weights for each objective (should sum to 1.0)
        
        Returns:
            Composite score (0-1 scale, higher is better)
        """
        base_score = (
            w_support * self.support_quality +
            w_movement * self.movement_preservation +
            w_efficiency * self.material_efficiency +
            w_feasibility * self.fabrication_feasibility
        )
        
        fea_bonus = 0.0
        if self.fea_result:
            if self.fea_result.overall_pass:
                fea_bonus = w_fea_pass * 1.0
            else:
                fea_bonus = w_fea_pass * 0.5
        
        self.composite_score = base_score + fea_bonus
        return self.composite_score


class ComprehensiveAnalyzer:
    """Analyzes and ranks Pareto-optimal designs."""
    
    def __init__(self, designs_json_path: str):
        """Initialize analyzer.
        
        Args:
            designs_json_path: Path to phase_4b_results/designs.json
        """
        self.designs_json_path = Path(designs_json_path)
        self.designs = self._load_designs()
    
    def _load_designs(self) -> List[Dict]:
        """Load designs from JSON."""
        with open(self.designs_json_path) as f:
            return json.load(f)
    
    def analyze_complete_pipeline(
        self,
        stl_export_results: Dict[int, Dict],
        fea_results: List[FEAResult],
    ) -> List[DesignRanking]:
        """Create complete design rankings.
        
        Args:
            stl_export_results: Output from STLExportPipeline.export_pareto_front()
            fea_results: List of FEAResult from FEAValidator
        
        Returns:
            List of DesignRanking sorted by composite score
        """
        rankings = []
        
        # Map FEA results by rank
        fea_by_rank = {r.design_rank: r for r in fea_results}
        
        for design in self.designs:
            rank = design['rank']
            
            # Get STL export data if available
            stl_data = stl_export_results.get(rank, {})
            
            # Get FEA result if available
            fea = fea_by_rank.get(rank)
            
            ranking = DesignRanking(
                rank_in_pareto=rank,
                voxel_count=design['voxel_count'],
                weight_grams=design['weight_grams'],
                support_quality=design['support_quality'],
                movement_preservation=design['movement_preservation'],
                material_efficiency=design['material_efficiency'],
                fabrication_feasibility=design['fabrication_feasibility'],
                fea_result=fea,
                stl_vertices=stl_data.get('vertices'),
                stl_volume_mm3=stl_data.get('volume_mm3'),
            )
            
            # Compute composite score
            ranking.compute_composite_score()
            
            # Generate clinical recommendation
            ranking.clinical_recommendation = self._generate_recommendation(ranking)
            
            rankings.append(ranking)
        
        # Sort by composite score (descending)
        rankings.sort(key=lambda r: r.composite_score, reverse=True)
        
        return rankings
    
    def _generate_recommendation(self, ranking: DesignRanking) -> str:
        """Generate clinical recommendation for design."""
        recommendations = []
        
        # Support quality
        if ranking.support_quality > 0.85:
            recommendations.append("High support (therapeutic/intensive)")
        elif ranking.support_quality > 0.75:
            recommendations.append("Good support (standard therapy)")
        else:
            recommendations.append("Light support (maintenance/comfort)")
        
        # Movement preservation
        if ranking.movement_preservation > 0.55:
            recommendations.append("High mobility (dynamic therapy)")
        elif ranking.movement_preservation > 0.45:
            recommendations.append("Moderate mobility (functional support)")
        else:
            recommendations.append("Minimal mobility (immobilization)")
        
        # Material efficiency
        if ranking.material_efficiency > 0.90:
            recommendations.append("Ultra-lightweight (minimal print time)")
        elif ranking.material_efficiency > 0.75:
            recommendations.append("Lightweight (efficient fabrication)")
        else:
            recommendations.append("Heavier design (maximum durability)")
        
        # FEA validation
        if ranking.fea_result:
            if ranking.fea_result.overall_pass:
                recommendations.append("PASSES structural validation")
            else:
                recommendations.append("FAILS one or more FEA checks [REVIEW]")
        
        return " | ".join(recommendations)


class ReportGenerator:
    """Generates HTML/JSON/Markdown reports."""
    
    @staticmethod
    def generate_html_report(
        rankings: List[DesignRanking],
        output_path: str = "phase_4c_report.html",
        title: str = "Phase 4C: Orthosis Design Validation & Comparison Report",
    ) -> str:
        """Generate comprehensive HTML report.
        
        Args:
            rankings: Sorted list of DesignRanking
            output_path: Output HTML file path
            title: Report title
        
        Returns:
            Output file path
        """
        html_parts = []
        
        # Header
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1 {{ color: #1f4788; border-bottom: 3px solid #1f4788; padding-bottom: 10px; }}
                h2 {{ color: #2c5aa0; margin-top: 30px; }}
                h3 {{ color: #5a85c0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #1f4788; color: white; padding: 12px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 10px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .pass {{ background-color: #d4edda; color: #155724; font-weight: bold; }}
                .fail {{ background-color: #f8d7da; color: #721c24; font-weight: bold; }}
                .metric {{ color: #0066cc; font-weight: bold; }}
                .section {{ margin-bottom: 40px; }}
                .box {{ border-left: 4px solid #1f4788; padding: 15px; background-color: #f0f4f8; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """)
        
        # Executive Summary
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Executive Summary</h2>")
        
        if rankings:
            top_design = rankings[0]
            html_parts.append(f"""
            <div class='box'>
                <h3>Top Recommendation: Design Rank {top_design.rank_in_pareto}</h3>
                <p><strong>Composite Score:</strong> <span class='metric'>{top_design.composite_score:.3f}</span></p>
                <p><strong>Clinical Use:</strong> {top_design.clinical_recommendation}</p>
                <p><strong>Key Features:</strong></p>
                <ul>
                    <li>Support Quality: {top_design.support_quality:.1%}</li>
                    <li>Movement Preservation: {top_design.movement_preservation:.1%}</li>
                    <li>Material Efficiency: {top_design.material_efficiency:.1%}</li>
                    <li>Weight: {top_design.weight_grams:.2f}g</li>
                </ul>
            </div>
            """)
        
        html_parts.append("""
            <p><strong>Total Designs Analyzed:</strong> <span class='metric'>{}</span></p>
            <p><strong>Analysis Completed:</strong> Phase 4B (Inverse Design) + Phase 4C (FEA Validation + STL Export)</p>
        """.format(len(rankings)))
        
        html_parts.append("</div>")
        
        # Design Comparison Matrix
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Design Comparison Matrix</h2>")
        html_parts.append("""
        <table>
            <tr>
                <th>Rank</th>
                <th>Composite<br>Score</th>
                <th>Support<br>Quality</th>
                <th>Movement<br>Preservation</th>
                <th>Material<br>Efficiency</th>
                <th>Fabrication<br>Feasibility</th>
                <th>Voxel<br>Count</th>
                <th>Weight<br>(g)</th>
                <th>FEA<br>Status</th>
                <th>Recommendation</th>
            </tr>
        """)
        
        for ranking in rankings[:10]:  # Top 10 designs
            fea_status = "PASS" if (ranking.fea_result and ranking.fea_result.overall_pass) else (
                "FAIL" if ranking.fea_result else "N/A"
            )
            fea_class = "pass" if fea_status == "PASS" else ("fail" if fea_status == "FAIL" else "")
            
            html_parts.append(f"""
            <tr>
                <td>{ranking.rank_in_pareto}</td>
                <td class='metric'>{ranking.composite_score:.3f}</td>
                <td>{ranking.support_quality:.1%}</td>
                <td>{ranking.movement_preservation:.1%}</td>
                <td>{ranking.material_efficiency:.1%}</td>
                <td>{ranking.fabrication_feasibility:.1%}</td>
                <td>{ranking.voxel_count}</td>
                <td>{ranking.weight_grams:.2f}</td>
                <td class='{fea_class}'>{fea_status}</td>
                <td style='font-size: 0.9em;'>{ranking.clinical_recommendation[:60]}...</td>
            </tr>
            """)
        
        html_parts.append("</table>")
        html_parts.append("</div>")
        
        # FEA Validation Results
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>FEA Validation Results</h2>")
        
        fea_designs = [r for r in rankings if r.fea_result]
        if fea_designs:
            html_parts.append("""
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Max Stress<br>(MPa)</th>
                    <th>Peak Strain</th>
                    <th>Max Deflection<br>(mm)</th>
                    <th>Support<br>Effectiveness</th>
                    <th>Overall<br>Pass</th>
                </tr>
            """)
            
            for ranking in fea_designs[:10]:
                result = ranking.fea_result
                status_class = "pass" if result.overall_pass else "fail"
                
                html_parts.append(f"""
                <tr>
                    <td>{ranking.rank_in_pareto}</td>
                    <td>{result.max_von_mises_stress_mpa:.2f}</td>
                    <td>{result.peak_strain:.4f}</td>
                    <td>{result.max_deformation_mm:.2f}</td>
                    <td>{result.support_effectiveness:.3f}</td>
                    <td class='{status_class}'>{'PASS' if result.overall_pass else 'FAIL'}</td>
                </tr>
                """)
            
            html_parts.append("</table>")
        else:
            html_parts.append("<p>No FEA results available.</p>")
        
        html_parts.append("</div>")
        
        # Fabrication Specifications
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Fabrication Specifications (STL Export)</h2>")
        
        stl_designs = [r for r in rankings if r.stl_vertices]
        if stl_designs:
            html_parts.append("""
            <table>
                <tr>
                    <th>Rank</th>
                    <th>STL File</th>
                    <th>Vertices</th>
                    <th>Volume<br>(mm3)</th>
                    <th>Est. Weight<br>(g @ 1.2 g/cm3)</th>
                    <th>Est. Print Time<br>(min)</th>
                </tr>
            """)
            
            for ranking in stl_designs[:10]:
                volume_cm3 = ranking.stl_volume_mm3 / 1000 if ranking.stl_volume_mm3 else 0
                est_weight = volume_cm3 * 1.2
                est_print_time = max(5, volume_cm3 * 2)  # Rough estimate
                
                html_parts.append(f"""
                <tr>
                    <td>{ranking.rank_in_pareto}</td>
                    <td>orthosis_rank_{ranking.rank_in_pareto:02d}.stl</td>
                    <td>{ranking.stl_vertices:,}</td>
                    <td>{ranking.stl_volume_mm3:.1f}</td>
                    <td>{est_weight:.2f}</td>
                    <td>{est_print_time:.0f}</td>
                </tr>
                """)
            
            html_parts.append("</table>")
        else:
            html_parts.append("<p>No STL export results available.</p>")
        
        html_parts.append("</div>")
        
        # Clinical Recommendations
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Clinical Recommendations by Use Case</h2>")
        
        html_parts.append("""
        <h3>For Maximum Support (Therapeutic Orthosis)</h3>
        <p>Choose designs with high support quality (>0.85) and passing FEA validation.</p>
        """)
        
        support_designs = [r for r in rankings if r.support_quality > 0.85][:3]
        if support_designs:
            for r in support_designs:
                html_parts.append(f"<p>- Design Rank {r.rank_in_pareto}: {r.clinical_recommendation}</p>")
        
        html_parts.append("""
        <h3>For Lightweight Comfort (Daily Wear)</h3>
        <p>Choose designs with high material efficiency (>0.90) and good movement preservation (>0.50).</p>
        """)
        
        light_designs = [r for r in rankings if r.material_efficiency > 0.90 and r.movement_preservation > 0.50][:3]
        if light_designs:
            for r in light_designs:
                html_parts.append(f"<p>- Design Rank {r.rank_in_pareto}: {r.clinical_recommendation}</p>")
        
        html_parts.append("""
        <h3>For Dynamic Therapy (Intensive Rehabilitation)</h3>
        <p>Choose designs with balanced support and high movement preservation (0.45-0.60).</p>
        """)
        
        dynamic_designs = [r for r in rankings if 0.45 < r.movement_preservation < 0.60][:3]
        if dynamic_designs:
            for r in dynamic_designs:
                html_parts.append(f"<p>- Design Rank {r.rank_in_pareto}: {r.clinical_recommendation}</p>")
        
        html_parts.append("</div>")
        
        # Footer
        html_parts.append("""
        </body>
        </html>
        """)
        
        # Write file
        with open(output_path, 'w') as f:
            f.write("\n".join(html_parts))
        
        return output_path
    
    @staticmethod
    def generate_json_report(
        rankings: List[DesignRanking],
        output_path: str = "phase_4c_report.json",
    ) -> str:
        """Generate JSON report (machine-readable).
        
        Args:
            rankings: Sorted list of DesignRanking
            output_path: Output JSON file path
        
        Returns:
            Output file path
        """
        report_data = {
            'metadata': {
                'title': 'Phase 4C: Orthosis Design Validation & Comparison',
                'total_designs': len(rankings),
            },
            'designs': []
        }
        
        for ranking in rankings:
            design_entry = {
                'rank_in_pareto': ranking.rank_in_pareto,
                'composite_score': round(ranking.composite_score, 4),
                'phase_4b_metrics': {
                    'support_quality': round(ranking.support_quality, 4),
                    'movement_preservation': round(ranking.movement_preservation, 4),
                    'material_efficiency': round(ranking.material_efficiency, 4),
                    'fabrication_feasibility': round(ranking.fabrication_feasibility, 4),
                },
                'phase_4c_fea': None,
                'phase_4c_stl': None,
                'clinical_recommendation': ranking.clinical_recommendation,
            }
            
            # Add FEA data if available
            if ranking.fea_result:
                design_entry['phase_4c_fea'] = ranking.fea_result.summary_dict()
            
            # Add STL data if available
            if ranking.stl_vertices:
                design_entry['phase_4c_stl'] = {
                    'vertices': ranking.stl_vertices,
                    'volume_mm3': round(ranking.stl_volume_mm3, 2) if ranking.stl_volume_mm3 else None,
                    'filename': f"orthosis_rank_{ranking.rank_in_pareto:02d}.stl",
                }
            
            report_data['designs'].append(design_entry)
        
        # Write file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return output_path
    
    @staticmethod
    def generate_markdown_report(
        rankings: List[DesignRanking],
        output_path: str = "phase_4c_report.md",
    ) -> str:
        """Generate Markdown report (human-readable).
        
        Args:
            rankings: Sorted list of DesignRanking
            output_path: Output Markdown file path
        
        Returns:
            Output file path
        """
        lines = []
        
        lines.append("# Phase 4C: Orthosis Design Validation & Comparison Report")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        
        if rankings:
            top_design = rankings[0]
            lines.append(f"**Top Recommendation: Design Rank {top_design.rank_in_pareto}**")
            lines.append(f"- Composite Score: `{top_design.composite_score:.3f}`")
            lines.append(f"- Clinical Use: {top_design.clinical_recommendation}")
            lines.append("")
        
        lines.append(f"Total Designs Analyzed: **{len(rankings)}**")
        lines.append("")
        
        # Design Comparison
        lines.append("## Design Comparison Matrix")
        lines.append("")
        lines.append("| Rank | Score | Support | Movement | Efficiency | Fabric | Voxels | Weight (g) | FEA | Recommendation |")
        lines.append("|------|-------|---------|----------|------------|--------|--------|------------|-----|----------------|")
        
        for ranking in rankings[:10]:
            fea_status = "PASS" if (ranking.fea_result and ranking.fea_result.overall_pass) else (
                "FAIL" if ranking.fea_result else "N/A"
            )
            lines.append(
                f"| {ranking.rank_in_pareto} | {ranking.composite_score:.3f} | "
                f"{ranking.support_quality:.1%} | {ranking.movement_preservation:.1%} | "
                f"{ranking.material_efficiency:.1%} | {ranking.fabrication_feasibility:.1%} | "
                f"{ranking.voxel_count} | {ranking.weight_grams:.2f} | {fea_status} | "
                f"{ranking.clinical_recommendation[:40]}... |"
            )
        
        lines.append("")
        
        # FEA Results
        lines.append("## FEA Validation Results")
        lines.append("")
        
        fea_designs = [r for r in rankings if r.fea_result]
        if fea_designs:
            lines.append("| Rank | Max Stress (MPa) | Peak Strain | Max Deflection (mm) | Support Eff. | Pass |")
            lines.append("|------|-----------------|-------------|---------------------|--------------|------|")
            
            for ranking in fea_designs[:10]:
                result = ranking.fea_result
                status = "PASS" if result.overall_pass else "FAIL"
                lines.append(
                    f"| {ranking.rank_in_pareto} | {result.max_von_mises_stress_mpa:.2f} | "
                    f"{result.peak_strain:.4f} | {result.max_deformation_mm:.2f} | "
                    f"{result.support_effectiveness:.3f} | {status} |"
                )
        
        lines.append("")
        
        # Clinical Recommendations
        lines.append("## Clinical Recommendations")
        lines.append("")
        
        lines.append("### For Maximum Support (Therapeutic)")
        support_designs = [r for r in rankings if r.support_quality > 0.85][:3]
        for r in support_designs:
            lines.append(f"- **Rank {r.rank_in_pareto}**: {r.clinical_recommendation}")
        
        lines.append("")
        lines.append("### For Lightweight Daily Wear")
        light_designs = [r for r in rankings if r.material_efficiency > 0.90 and r.movement_preservation > 0.50][:3]
        for r in light_designs:
            lines.append(f"- **Rank {r.rank_in_pareto}**: {r.clinical_recommendation}")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return output_path


def main():
    """Demo: Generate comprehensive Phase 4C reports."""
    print("[DEMO] Phase 4C Comprehensive Reporting")
    print("=" * 60)
    
    designs_path = Path(__file__).parent / 'phase_4b_results' / 'designs.json'
    
    if not designs_path.exists():
        print(f"[ERROR] Designs file not found: {designs_path}")
        return
    
    # Step 1: FEA Validation
    print("\n[STEP 1] FEA Validation")
    validator = FEAValidator(design_space_voxels=300)
    fea_results, fea_summary = validator.validate_pareto_front(
        str(designs_path),
        load_case=PatientLoadCase.clinical_standard()
    )
    print(f"  Validated designs: {fea_summary['total_designs']}")
    
    # Step 2: STL Export
    print("\n[STEP 2] STL Export")
    pipeline = STLExportPipeline(output_dir="phase_4c_stl_exports")
    stl_results = pipeline.export_pareto_front(str(designs_path), top_n=5)
    print(f"  Exported designs: {len(stl_results)}")
    
    # Step 3: Comprehensive Analysis
    print("\n[STEP 3] Comprehensive Analysis")
    analyzer = ComprehensiveAnalyzer(str(designs_path))
    rankings = analyzer.analyze_complete_pipeline(stl_results, fea_results)
    print(f"  Ranked designs: {len(rankings)}")
    print(f"  Top design: Rank {rankings[0].rank_in_pareto} (score: {rankings[0].composite_score:.3f})")
    
    # Step 4: Generate Reports
    print("\n[STEP 4] Generating Reports")
    
    reports_dir = Path(__file__).parent / "phase_4c_reports"
    reports_dir.mkdir(exist_ok=True)
    
    html_path = ReportGenerator.generate_html_report(
        rankings,
        output_path=str(reports_dir / "report.html")
    )
    print(f"  HTML report: {html_path}")
    
    json_path = ReportGenerator.generate_json_report(
        rankings,
        output_path=str(reports_dir / "report.json")
    )
    print(f"  JSON report: {json_path}")
    
    md_path = ReportGenerator.generate_markdown_report(
        rankings,
        output_path=str(reports_dir / "report.md")
    )
    print(f"  Markdown report: {md_path}")
    
    print("\n[COMPLETE] Phase 4C Pipeline Finished")


if __name__ == '__main__':
    main()
