# =============================================================================
# DDx Results Analyzer - Multi-Perspective Clinical Performance Analysis
# =============================================================================

"""
Specialized analysis tool for DDx evaluation results.
Takes JSON outputs from DDx evaluation and provides multiple analytical perspectives:
- Traditional ML metrics
- Clinical performance assessment  
- Research insights
- Comparative analysis across cases

Usage:
    python ddx_results_analyzer.py --file ddx_results_Case_2_20250716_161914.json
    python ddx_results_analyzer.py --batch /path/to/results_folder/
"""

import json
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import statistics

@dataclass
class AnalysisResult:
    """Comprehensive analysis result"""
    case_id: str
    traditional_metrics: Dict[str, float]
    clinical_metrics: Dict[str, float]
    research_metrics: Dict[str, float]
    insights: List[str]
    performance_category: str
    raw_data: Dict[str, Any]

class DDxResultsAnalyzer:
    """Multi-perspective analyzer for DDx evaluation results"""
    
    def __init__(self):
        self.performance_thresholds = {
            'excellent': {'clinical_success': 0.8, 'true_miss': 0.1},
            'good': {'clinical_success': 0.6, 'true_miss': 0.25},
            'fair': {'clinical_success': 0.4, 'true_miss': 0.4},
            'poor': {'clinical_success': 0.3, 'true_miss': 0.6}
        }
    
    def analyze_single_case(self, results_file: str) -> AnalysisResult:
        """Analyze a single DDx results file"""
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        case_id = self._extract_case_id(results_file)
        
        # Extract core metrics - handle both nested and flat JSON structures
        if 'evaluation_result' in data:
            eval_result = data['evaluation_result']
            synthesis_result = data.get('synthesis_result', {})
        else:
            # Flat structure - metrics at top level
            eval_result = data
            synthesis_result = data
        
        tp = data.get('tp_count', eval_result.get('tp_count', 0))
        fp = data.get('fp_count', eval_result.get('fp_count', 0))
        ae = data.get('ae_count', eval_result.get('ae_count', 0))
        tm_sm = data.get('tm_sm_count', eval_result.get('tm_sm_count', 0))
        
        # Handle fn_count correctly - it often includes AE incorrectly
        reported_fn = data.get('fn_count', eval_result.get('fn_count', 0))
        fn = reported_fn - ae  # Remove AE from FN since AE should not be false negatives
        if fn < 0:
            fn = 0  # Can't have negative true misses
        
        # Get total counts - use provided totals if available
        total_gt = data.get('total_gt_diagnoses', tp + ae + fn + tm_sm)
        total_system = data.get('total_team_diagnoses', tp + fp)
        
        # Debug output for verification
        print(f"📊 Extracted metrics for {case_id}:")
        print(f"   TP: {tp}, FP: {fp}, AE: {ae}, TM: {fn}, TM-SM: {tm_sm}")
        print(f"   Total GT: {total_gt}, Total System: {total_system}")
        print(f"   Reported FN (includes AE): {reported_fn}, Corrected TM: {fn}")
        
        # PRIMARY CLINICAL METRICS (Normalized Binary Assessment)
        clinical_metrics = {
            'clinical_success_rate': (tp + ae) / total_gt if total_gt > 0 else 0.0,  # TP + AE = good medicine
            'clinical_failure_rate': fn / total_gt if total_gt > 0 else 0.0,  # TM only = actual failures
            'diagnostic_precision': tp / total_system if total_system > 0 else 0.0,  # Unchanged
            'appropriate_exclusion_rate': ae / total_gt if total_gt > 0 else 0.0,  # Shows reasoning quality
            'symptom_management_capture_rate': tm_sm / total_gt if total_gt > 0 else 0.0,  # Safety net
            'clinical_reasoning_quality': eval_result.get('clinical_reasoning_quality', 0.0)  # From original evaluation
        }
        
        # Traditional ML Metrics (For Comparison Only)
        traditional_metrics = {
            'precision': tp / total_system if total_system > 0 else 0.0,
            'recall': tp / total_gt if total_gt > 0 else 0.0,  # Treats AE as failures (clinically wrong)
            'f1_score': 0.0,  # Will calculate below
            'accuracy': (tp + ae) / (total_gt + fp) if (total_gt + fp) > 0 else 0.0
        }
        traditional_metrics['f1_score'] = (2 * traditional_metrics['precision'] * traditional_metrics['recall']) / \
                                        (traditional_metrics['precision'] + traditional_metrics['recall']) \
                                        if (traditional_metrics['precision'] + traditional_metrics['recall']) > 0 else 0.0
        
        # Research Metrics - handle missing fields gracefully
        research_metrics = {
            'reasoning_thoroughness': data.get('reasoning_thoroughness', eval_result.get('reasoning_thoroughness', 0.0)),
            'diagnostic_safety': data.get('diagnostic_safety', eval_result.get('diagnostic_safety', 0.0)),
            'system_safety_coverage': data.get('system_safety_coverage', eval_result.get('system_safety_coverage', 0.0)),
            'tempo_score': synthesis_result.get('tempo_score', data.get('tempo_score', 0.0)),
            'credibility_weighted_performance': data.get('caa_weight_applied', eval_result.get('caa_weight_applied', 0.0)),
            'clinical_reasoning_quality': data.get('clinical_reasoning_quality', eval_result.get('clinical_reasoning_quality', 0.0))
        }
        
        # Generate insights
        insights = self._generate_insights(tp, fp, ae, fn, tm_sm, total_gt, 
                                         traditional_metrics, clinical_metrics, research_metrics)
        
        # Determine performance category
        performance_category = self._categorize_performance(clinical_metrics)
        
        return AnalysisResult(
            case_id=case_id,
            traditional_metrics=traditional_metrics,
            clinical_metrics=clinical_metrics, 
            research_metrics=research_metrics,
            insights=insights,
            performance_category=performance_category,
            raw_data=data
        )
    
    def _extract_case_id(self, filename: str) -> str:
        """Extract case ID from filename"""
        path = Path(filename)
        name = path.stem
        # Extract case from patterns like "ddx_results_Case_3_20250716_180646" or "Case_2_20250716_161914"
        if 'Case_' in name:
            case_part = name.split('Case_')[1].split('_')[0]
            return f"Case {case_part}"
        elif 'case_' in name.lower():
            case_part = name.lower().split('case_')[1].split('_')[0]
            return f"Case {case_part}"
        return name
    
    def _generate_insights(self, tp: int, fp: int, ae: int, fn: int, tm_sm: int, total_gt: int,
                          traditional: Dict, clinical: Dict, research: Dict) -> List[str]:
        """Generate analytical insights"""
        
        insights = []
        
        # Primary Clinical Assessment (Normalized Binary)
        if clinical['clinical_success_rate'] > 0.8:
            insights.append(f"🎯 EXCELLENT clinical performance ({clinical['clinical_success_rate']:.1%}) - system appropriately handled {tp + ae}/{total_gt} ground truth diagnoses")
        elif clinical['clinical_success_rate'] > 0.6:
            insights.append(f"✅ GOOD clinical performance ({clinical['clinical_success_rate']:.1%}) - system appropriately handled {tp + ae}/{total_gt} ground truth diagnoses")
        else:
            insights.append(f"⚠️ NEEDS IMPROVEMENT - clinical success rate only {clinical['clinical_success_rate']:.1%}")
        
        # Binary Clinical Logic Explanation
        insights.append(f"🔬 CLINICAL BINARY LOGIC: Success = TP({tp}) + AE({ae}) = {tp + ae}/{total_gt} | Failure = TM({fn}) = {fn}/{total_gt}")
        
        # Key Insight: AE as Success
        if ae > 0:
            insights.append(f"✅ CLINICAL REASONING CREDIT: {ae} diagnoses appropriately excluded with evidence-based reasoning - counted as clinical successes")
            insights.append(f"📊 TRADITIONAL vs CLINICAL: Traditional recall ({traditional['recall']:.1%}) vs Clinical success ({clinical['clinical_success_rate']:.1%}) - {clinical['clinical_success_rate'] - traditional['recall']:+.1%} difference due to crediting good exclusions")
        
        # True Clinical Failures
        if fn > 0:
            insights.append(f"❌ ACTUAL FAILURES: {fn} diagnoses completely missed (failure rate: {clinical['clinical_failure_rate']:.1%}) - these represent true diagnostic inadequacy")
        else:
            insights.append("🏆 ZERO FAILURES: All ground truth diagnoses were either matched or appropriately excluded with reasoning")
        
        # Symptom Management Safety Net
        if tm_sm > 0:
            insights.append(f"🏥 SAFETY NET ACTIVATION: {tm_sm} missed diagnoses had symptom management captured - partial mitigation of failures")
        
        # Diagnostic Precision (Unchanged)
        if fp > 3:
            insights.append(f"⚠️ OVERDIAGNOSIS CONCERN: {fp} false positives - suggests clinical equivalence matching issues or overinclusive selection")
        elif fp == 0:
            insights.append("🎯 PERFECT DIAGNOSTIC PRECISION: All proposed diagnoses matched ground truth")
        
        # Clinical Excellence Indicators
        if clinical['clinical_success_rate'] > 0.8 and clinical['clinical_failure_rate'] < 0.2:
            insights.append("🏆 CLINICAL EXCELLENCE: High success rate with minimal true failures - demonstrates strong diagnostic competence")
        
        # Performance Recommendations
        if clinical['clinical_failure_rate'] > 0.3:
            insights.append("🔧 PRIORITY: Reduce true miss rate - focus on improving recognition of ground truth conditions")
        
        if fp > 2 and clinical['diagnostic_precision'] < 0.5:
            insights.append("🔧 SECONDARY: Improve diagnostic precision - refine equivalence matching or selection criteria")
        
        return insights
    
    def _categorize_performance(self, clinical_metrics: Dict) -> str:
        """Categorize overall performance based on clinical success rate"""
        
        clinical_success = clinical_metrics['clinical_success_rate']
        clinical_failure = clinical_metrics['clinical_failure_rate']
        
        if clinical_success >= 0.8 and clinical_failure <= 0.2:
            return "EXCELLENT"
        elif clinical_success >= 0.6 and clinical_failure <= 0.3:
            return "GOOD" 
        elif clinical_success >= 0.4 and clinical_failure <= 0.5:
            return "FAIR"
        else:
            return "POOR"
    
    def generate_report(self, analysis: AnalysisResult) -> str:
        """Generate detailed analysis report"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx CASE ANALYSIS REPORT                             ║
║                              Case: {analysis.case_id:<10}                               ║
║                        Performance: {analysis.performance_category:<10}                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

🏥 PRIMARY CLINICAL METRICS (Normalized Binary Assessment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Clinical Success Rate:          {analysis.clinical_metrics['clinical_success_rate']:>6.1%}  (TP + AE) / Total GT
   Clinical Failure Rate:          {analysis.clinical_metrics['clinical_failure_rate']:>6.1%}  TM only / Total GT
   Diagnostic Precision:           {analysis.clinical_metrics['diagnostic_precision']:>6.1%}  TP / System Diagnoses
   Appropriate Exclusion Rate:     {analysis.clinical_metrics['appropriate_exclusion_rate']:>6.1%}  Good reasoning quality
   Symptom Management Capture:     {analysis.clinical_metrics['symptom_management_capture_rate']:>6.1%}  Safety net activation

📊 TRADITIONAL ML METRICS (For Comparison)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Precision:  {analysis.traditional_metrics['precision']:>6.1%}  (TP / System Diagnoses) - Same as above
   Recall:     {analysis.traditional_metrics['recall']:>6.1%}  (TP / Ground Truth) - Treats AE as failures 
   F1-Score:   {analysis.traditional_metrics['f1_score']:>6.1%}  (Harmonic Mean)
   Accuracy:   {analysis.traditional_metrics['accuracy']:>6.1%}  (Overall Correctness)

🔬 RESEARCH METRICS (Advanced Analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Clinical Reasoning Quality:     {analysis.clinical_metrics['clinical_reasoning_quality']:>6.1%}  Overall reasoning score
   Reasoning Thoroughness:         {analysis.research_metrics['reasoning_thoroughness']:>6.1%}  Evidence-based decisions
   Diagnostic Safety:              {analysis.research_metrics['diagnostic_safety']:>6.1%}  Risk mitigation
   System Safety Coverage:         {analysis.research_metrics['system_safety_coverage']:>6.1%}  Safety net effectiveness  
   Tempo Score:                    {analysis.research_metrics['tempo_score']:>6.1f}  Diagnostic urgency
   Credibility Weighting:          {analysis.research_metrics['credibility_weighted_performance']:>6.1f}  Expert consensus quality

💡 ANALYTICAL INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for i, insight in enumerate(analysis.insights, 1):
            report += f"\n{i:>2}. {insight}"

        report += f"""

📈 METRIC EXPLANATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLINICAL BINARY REFRAMING:
• Clinical Success ({analysis.clinical_metrics['clinical_success_rate']:.1%}): TP + AE = both represent appropriate handling
• Clinical Failure ({analysis.clinical_metrics['clinical_failure_rate']:.1%}): TM only = actual diagnostic inadequacy
• Diagnostic Precision: TP / System Diagnoses = accuracy of proposed diagnoses

KEY INSIGHT: "Appropriately Excluded" diagnoses represent EXCELLENT clinical performance - 
they were considered, evaluated, and correctly ruled out with proper reasoning. This is 
exactly what we want from clinical decision-making.

TRADITIONAL vs CLINICAL PERSPECTIVES:
• Traditional Recall ({analysis.traditional_metrics['recall']:.1%}): Treats AE as failures (clinically wrong)
• Clinical Success ({analysis.clinical_metrics['clinical_success_rate']:.1%}): Credits AE as good medicine (clinically correct)
• Gap: {analysis.clinical_metrics['clinical_success_rate'] - analysis.traditional_metrics['recall']:+.1%} represents the value of appropriate clinical exclusions

PERFORMANCE CATEGORIES (Clinical Binary):
• EXCELLENT: >80% clinical success, <20% clinical failure
• GOOD: >60% clinical success, <30% clinical failure  
• FAIR: >40% clinical success, <50% clinical failure
• POOR: <40% clinical success or >50% clinical failure

ACTIONABLE METRICS:
• Clinical Failure Rate: PRIORITY - represents actual diagnostic inadequacy
• Clinical Success Rate: Overall clinical competence - credits both matches and good exclusions
• Diagnostic Precision: Accuracy of final diagnosis list - unchanged from traditional

CLINICAL LOGIC:
This normalized binary approach reflects how medicine actually works - good clinicians 
consider relevant diagnoses and make evidence-based decisions to include or exclude them. 
Both actions represent clinical competence when done with proper reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report
    
    def batch_analyze(self, results_folder: str) -> List[AnalysisResult]:
        """Analyze multiple result files in a folder"""
        
        results = []
        results_path = Path(results_folder)
        
        for json_file in results_path.glob("*.json"):
            try:
                analysis = self.analyze_single_case(str(json_file))
                results.append(analysis)
                print(f"✅ Analyzed: {json_file.name}")
            except Exception as e:
                print(f"❌ Error analyzing {json_file.name}: {e}")
        
        return results
    
    def generate_batch_summary(self, analyses: List[AnalysisResult]) -> str:
        """Generate summary report for multiple cases"""
        
        if not analyses:
            return "No analyses to summarize."
        
        # Aggregate metrics
        clinical_success_scores = [a.clinical_metrics['clinical_success_rate'] for a in analyses]
        clinical_failure_rates = [a.clinical_metrics['clinical_failure_rate'] for a in analyses]
        traditional_recalls = [a.traditional_metrics['recall'] for a in analyses]
        precisions = [a.traditional_metrics['precision'] for a in analyses]
        
        # Performance distribution
        performance_counts = {}
        for analysis in analyses:
            cat = analysis.performance_category
            performance_counts[cat] = performance_counts.get(cat, 0) + 1
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DDx BATCH ANALYSIS SUMMARY                           ║
║                            {len(analyses)} Cases Analyzed                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 AGGREGATE PERFORMANCE (Primary Clinical Metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Success Rate (TP + AE):
   Mean: {statistics.mean(clinical_success_scores):>6.1%}   Median: {statistics.median(clinical_success_scores):>6.1%}
   Range: {min(clinical_success_scores):>5.1%} - {max(clinical_success_scores):.1%}

Clinical Failure Rate (TM only):  
   Mean: {statistics.mean(clinical_failure_rates):>6.1%}   Median: {statistics.median(clinical_failure_rates):>6.1%}
   Range: {min(clinical_failure_rates):>5.1%} - {max(clinical_failure_rates):.1%}

Diagnostic Precision:
   Mean: {statistics.mean(precisions):>6.1%}   Median: {statistics.median(precisions):>6.1%}
   Range: {min(precisions):>5.1%} - {max(precisions):.1%}

Traditional Recall (For Comparison):
   Mean: {statistics.mean(traditional_recalls):>6.1%}   Median: {statistics.median(traditional_recalls):>6.1%}
   Range: {min(traditional_recalls):>5.1%} - {max(traditional_recalls):.1%}

🏆 PERFORMANCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for category in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']:
            count = performance_counts.get(category, 0)
            percentage = (count / len(analyses)) * 100
            summary += f"\n   {category:<10}: {count:>3} cases ({percentage:>5.1f}%)"

        summary += f"""

📈 KEY INSIGHTS (Clinical Binary Perspective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical Success vs Traditional Recall gap: {statistics.mean(clinical_success_scores) - statistics.mean(traditional_recalls):+.1%}
  This gap represents the clinical value of appropriate diagnostic exclusions.

• Average Clinical Performance: {statistics.mean(clinical_success_scores):.1%} success rate
  This credits both exact matches AND appropriate exclusions with reasoning.

• Average Clinical Failure Rate: {statistics.mean(clinical_failure_rates):.1%}
  This represents only true diagnostic inadequacy (complete misses).

• Best performing case: {max(analyses, key=lambda x: x.clinical_metrics['clinical_success_rate']).case_id} 
  ({max(clinical_success_scores):.1%} clinical success)

• Most challenging case: {min(analyses, key=lambda x: x.clinical_metrics['clinical_success_rate']).case_id}
  ({min(clinical_success_scores):.1%} clinical success)

CLINICAL INSIGHT: The normalized binary approach shows that systems perform better 
clinically than traditional metrics suggest, because good clinical reasoning includes 
both correct inclusions AND appropriate exclusions with evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return summary

def main():
    parser = argparse.ArgumentParser(description='Analyze DDx evaluation results')
    parser.add_argument('--file', type=str, help='Single JSON results file to analyze')
    parser.add_argument('--batch', type=str, help='Folder containing multiple JSON results files')
    parser.add_argument('--output', type=str, help='Output file for report (optional)')
    
    args = parser.parse_args()
    
    if not args.file and not args.batch:
        print("Please specify either --file or --batch option")
        parser.print_help()
        return
    
    analyzer = DDxResultsAnalyzer()
    
    if args.file:
        # Single file analysis
        print(f"🔍 Analyzing file: {args.file}")
        try:
            analysis = analyzer.analyze_single_case(args.file)
            report = analyzer.generate_report(analysis)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"📊 Report saved to: {args.output}")
            else:
                print(report)
        except Exception as e:
            print(f"❌ Error analyzing {args.file}: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.batch:
        # Batch analysis
        analyses = analyzer.batch_analyze(args.batch)
        
        # Generate individual reports
        for analysis in analyses:
            report = analyzer.generate_report(analysis)
            output_file = f"analysis_report_case_{analysis.case_id.replace(' ', '_')}.txt"
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📊 Individual report saved: {output_file}")
        
        # Generate batch summary
        summary = analyzer.generate_batch_summary(analyses)
        summary_file = "batch_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"📊 Batch summary saved: {summary_file}")
        
        print(summary)

if __name__ == "__main__":
    main()

# Example usage:
# python ddx_results_analyzer.py --file ddx_results_Case_3_20250716_180646.json
# python ddx_results_analyzer.py --batch /path/to/results_folder/
# python ddx_results_analyzer.py --file your_case.json --output report.txt
