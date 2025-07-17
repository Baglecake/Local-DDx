
# =============================================================================
# DDx Runner v6 - Complete Pipeline Orchestrator
# =============================================================================

"""
DDx Runner v6: Clean, unified entry point for the complete v6 diagnostic system.
Handles case analysis, rounds execution, synthesis, and evaluation in a single
clean interface.
"""

import time
import traceback
from typing import Dict, List, Optional, Any

# Import all v6 system components
from ddx_core_v6 import DDxSystem
from ddx_rounds_v6 import RoundType
from ddx_synthesis_v6 import SynthesisResult
from ddx_evaluator_v9 import EvaluationResult

class DDxRunner:
    """Complete pipeline orchestrator for the v6 diagnostic system"""

    def __init__(self):
        """Initialize the runner with v6 system components"""
        self.ddx_system = DDxSystem()
        self.is_initialized = False
        self.results_history = []

        # Performance tracking
        self.total_cases_run = 0
        self.successful_cases = 0
        self.last_case_duration = 0.0

    def initialize_system(self) -> bool:
        """Initialize all v6 system components"""
        if self.is_initialized:
            print("✅ System already initialized")
            return True

        print("🚀 Initializing DDx System v6...")
        print("=" * 50)

        try:
            # Initialize core system (models, agents)
            success = self.ddx_system.initialize()

            if success:
                # Add synthesis and rounds capabilities
                self.ddx_system.add_synthesis_manager()

                self.is_initialized = True
                print("\n✅ DDx System v6 Ready!")
                print(f"   🤖 Models loaded: {len(self.ddx_system.model_manager.get_available_models())}")
                print(f"   🏥 System status: Operational")
                return True
            else:
                print("\n❌ Failed to initialize DDx System v6")
                return False

        except Exception as e:
            print(f"\n❌ System initialization error: {e}")
            traceback.print_exc()
            return False

    def run_case(self, case_name: str, case_description: str,
                ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Run a complete diagnostic case through the entire v6 pipeline

        Args:
            case_name: Identifier for the case
            case_description: Clinical case presentation
            ground_truth: Expected diagnoses with supporting evidence

        Returns:
            Dictionary with complete results and performance metrics
        """
        if not self.is_initialized:
            print("❌ System not initialized. Please run initialize_system() first.")
            return {"success": False, "error": "System not initialized"}

        print(f"\n🚀 LAUNCHING CASE: {case_name.upper()}")
        print("=" * 70)

        case_start_time = time.time()
        self.total_cases_run += 1

        try:
            # Step 1: Case Analysis & Agent Generation
            print(f"\n📋 STEP 1: Case Analysis & Dynamic Agent Generation")
            print("-" * 50)

            analysis_result = self.ddx_system.analyze_case(case_description, case_name)

            if not analysis_result.get('success'):
                return self._create_failure_result("Case analysis failed", analysis_result.get('error'))

            agents_generated = analysis_result.get('agents_generated', 0)
            print(f"✅ Generated {agents_generated} specialist agents")

            # Step 2: Execute Complete Diagnostic Sequence
            print(f"\n🔄 STEP 2: Complete Diagnostic Rounds Sequence")
            print("-" * 50)

            rounds_result = self.ddx_system.run_complete_diagnostic_sequence()

            if not rounds_result:
                return self._create_failure_result("Diagnostic rounds failed", "No round results returned")

            successful_rounds = sum(1 for result in rounds_result.values() if result.success)
            total_rounds = len(rounds_result)
            print(f"✅ Completed {successful_rounds}/{total_rounds} diagnostic rounds")

            # Step 3: Synthesis & Evaluation
            print(f"\n🔬 STEP 3: Analysis, Synthesis & Evaluation")
            print("-" * 50)

            analysis_results = self.ddx_system.run_complete_analysis(ground_truth)

            if not analysis_results.get('success'):
                return self._create_failure_result("Analysis failed", "Synthesis or evaluation error")

            # Step 4: Export Results
            print(f"\n📊 STEP 4: Results Export & Summary")
            print("-" * 50)

            json_results = self._export_case_results(case_name, analysis_results, ground_truth)

            # Step 5: Calculate Performance Summary
            case_duration = time.time() - case_start_time
            self.last_case_duration = case_duration
            self.successful_cases += 1

            # Store in history
            case_record = {
                'case_name': case_name,
                'case_description': case_description,
                'ground_truth': ground_truth,
                'analysis_results': analysis_results,
                'json_results': json_results,
                'duration': case_duration,
                'agents_generated': agents_generated,
                'rounds_completed': successful_rounds,
                'timestamp': time.time()
            }
            self.results_history.append(case_record)

            # Display Final Summary
            self._display_case_summary(case_name, analysis_results, case_duration)

            print(f"\n🏁 CASE COMPLETED: {case_name.upper()}")
            print("=" * 70)

            return {
                "success": True,
                "case_name": case_name,
                "duration": case_duration,
                "agents_generated": agents_generated,
                "rounds_completed": successful_rounds,
                "analysis_results": analysis_results,
                "json_results": json_results,
                "performance_summary": self._get_performance_summary()
            }

        except Exception as e:
            case_duration = time.time() - case_start_time
            error_msg = f"Pipeline error in {case_name}: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()

            return self._create_failure_result(error_msg, str(e), case_duration)

    def _create_failure_result(self, message: str, error: str, duration: float = 0.0) -> Dict[str, Any]:
        """Create a standardized failure result"""
        print(f"❌ {message}: {error}")

        return {
            "success": False,
            "error": message,
            "error_details": error,
            "duration": duration,
            "case_name": "Unknown",
            "agents_generated": 0,
            "rounds_completed": 0,
            "analysis_results": None,
            "json_results": None
        }

    def _export_case_results(self, case_name: str, analysis_results: Dict[str, Any],
                           ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """Export case results to JSON format"""
        try:
            if hasattr(self.ddx_system.synthesis_manager, 'export_to_json'):
                json_results = self.ddx_system.synthesis_manager.export_to_json(
                    case_id=case_name,
                    analysis_results=analysis_results,
                    ground_truth=ground_truth,
                    save_file=True
                )
                print(f"✅ Results exported to JSON")
                return json_results
            else:
                print(f"⚠️ JSON export not available")
                return {}
        except Exception as e:
            print(f"⚠️ JSON export failed: {e}")
            return {}

    def _display_case_summary(self, case_name: str, analysis_results: Dict[str, Any],
                            duration: float):
        """Display comprehensive case summary"""

        synthesis = analysis_results.get('synthesis_result')
        evaluation = analysis_results.get('evaluation_result')

        if not synthesis or not evaluation:
            print("⚠️ Incomplete results - cannot display full summary")
            return

        print(f"\n🎯 DIAGNOSTIC PERFORMANCE SUMMARY")
        print("=" * 50)

        # Core Results
        print(f"📋 Final Diagnoses: {len(synthesis.final_list)}")
        for i, diagnosis in enumerate(synthesis.final_list, 1):
            print(f"   {i}. {diagnosis}")

        # Performance Metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   ✅ True Positives: {evaluation.tp_count}")
        print(f"   ⚠️ False Positives: {evaluation.fp_count}")
        print(f"   ❌ False Negatives: {evaluation.fn_count}")
        if hasattr(evaluation, 'caa_count') and evaluation.caa_count > 0:
            print(f"   🛡️ Clinical Alternatives: {evaluation.caa_count}")
        if hasattr(evaluation, 'tm_sm_count') and evaluation.tm_sm_count > 0:
            print(f"   🏥 Symptom Mgmt Captures: {evaluation.tm_sm_count}")

        print(f"\n📊 Quality Scores:")
        print(f"   Precision: {evaluation.traditional_precision:.3f}")
        print(f"   Recall: {evaluation.traditional_recall:.3f}")
        print(f"   Clinical Reasoning: {evaluation.clinical_reasoning_quality:.3f}")
        print(f"   Diagnostic Safety: {evaluation.diagnostic_safety:.3f}")
        if hasattr(evaluation, 'system_safety_coverage'):
            print(f"   System Safety: {evaluation.system_safety_coverage:.3f}")

        # Performance Assessment
        performance_tier = self._assess_performance_tier(evaluation)
        print(f"\n🏆 Performance Assessment: {performance_tier}")

        # Timing
        print(f"\n⏱️ Execution Time: {duration:.1f} seconds")

    def _assess_performance_tier(self, evaluation: EvaluationResult) -> str:
        """Assess overall performance tier with AE-aware assessment"""
        if not evaluation:
            return "Unknown"

        # If we have AE diagnoses, calculate clinical success rate using correct formula
        if hasattr(evaluation, 'ae_count') and evaluation.ae_count > 0:
            tp = evaluation.tp_count
            ae = evaluation.ae_count
            tm = evaluation.fn_count  # fn_count is already just TM, not TM + AE
            tm_sm = getattr(evaluation, 'tm_sm_count', 0)
            
            # Clinical Success Rate: (TP + AE) / (TP + AE + TM + TM-SM)
            total_gt = tp + ae + tm + tm_sm
            clinical_success = (tp + ae) / total_gt if total_gt > 0 else 0
            
            return f"🟡 Clinical Analysis Needed (AE={ae}, CS={clinical_success:.1%})"

        # Traditional assessment when no AE diagnoses
        scores = [
            evaluation.traditional_precision,
            evaluation.traditional_recall,
            evaluation.clinical_reasoning_quality,
            evaluation.diagnostic_safety
        ]

        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return "Unknown"

        avg_score = sum(valid_scores) / len(valid_scores)

        if avg_score >= 0.8:
            return "🌟 Excellent"
        elif avg_score >= 0.65:
            return "🟢 Good"
        elif avg_score >= 0.5:
            return "🟡 Moderate"
        elif avg_score >= 0.3:
            return "🟠 Needs Improvement"
        else:
            return "🔴 Poor"

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall runner performance summary"""
        success_rate = (self.successful_cases / self.total_cases_run
                       if self.total_cases_run > 0 else 0.0)

        return {
            "total_cases_run": self.total_cases_run,
            "successful_cases": self.successful_cases,
            "success_rate": success_rate,
            "last_case_duration": self.last_case_duration,
            "average_duration": (sum(case['duration'] for case in self.results_history) /
                                len(self.results_history) if self.results_history else 0.0)
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.is_initialized:
            return {
                "status": "Not Initialized",
                "initialized": False,
                "models_loaded": 0,
                "agents_available": 0
            }

        ddx_status = self.ddx_system.get_system_status()
        runner_performance = self._get_performance_summary()

        return {
            "status": "Operational" if self.is_initialized else "Error",
            "initialized": self.is_initialized,
            "models_loaded": ddx_status.get('models_loaded', 0),
            "available_models": ddx_status.get('available_models', []),
            "current_agents": ddx_status.get('current_agents', 0),
            "case_loaded": ddx_status.get('case_loaded', False),
            "runner_performance": runner_performance,
            "ready_for_cases": self.is_initialized and ddx_status.get('models_loaded', 0) > 0
        }

    def get_case_history(self, include_details: bool = False) -> List[Dict[str, Any]]:
        """Get history of completed cases"""
        if not include_details:
            # Return summary only
            return [{
                'case_name': case['case_name'],
                'duration': case['duration'],
                'agents_generated': case['agents_generated'],
                'rounds_completed': case['rounds_completed'],
                'timestamp': case['timestamp']
            } for case in self.results_history]
        else:
            # Return full details
            return self.results_history

    def clear_history(self):
        """Clear case history and reset performance counters"""
        self.results_history = []
        self.total_cases_run = 0
        self.successful_cases = 0
        self.last_case_duration = 0.0
        print("✅ Case history cleared")

# =============================================================================
# Convenience Functions for Direct Usage
# =============================================================================

def run_single_case(case_name: str, case_description: str,
                   ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Convenience function to run a single case with automatic initialization

    Args:
        case_name: Case identifier
        case_description: Clinical presentation
        ground_truth: Expected diagnoses with evidence

    Returns:
        Complete case results
    """
    runner = DDxRunner()

    if not runner.initialize_system():
        return {"success": False, "error": "Failed to initialize system"}

    return runner.run_case(case_name, case_description, ground_truth)

def run_case_batch(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run multiple cases in batch mode

    Args:
        cases: List of case dictionaries with 'name', 'description', 'ground_truth'

    Returns:
        List of case results
    """
    runner = DDxRunner()

    if not runner.initialize_system():
        return [{"success": False, "error": "Failed to initialize system"}]

    results = []

    for case_data in cases:
        case_name = case_data.get('name', f'Case_{len(results)+1}')
        case_description = case_data.get('description', '')
        ground_truth = case_data.get('ground_truth', {})

        result = runner.run_case(case_name, case_description, ground_truth)
        results.append(result)

        # Brief pause between cases
        time.sleep(1)

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 BATCH SUMMARY: {successful}/{len(cases)} cases successful")

    return results

# =============================================================================
# Example Usage and Testing
# =============================================================================

def test_runner_v6():
    """Test the v6 runner with a sample case"""
    print("🧪 Testing DDx Runner v6")
    print("=" * 50)

    # Sample case data
    test_case_name = "Test Case - Acute Chest Pain"
    test_case_description = """
    A 45-year-old male presents with acute chest pain that began 2 hours ago.
    The pain is crushing, radiates to the left arm, and is associated with
    shortness of breath and diaphoresis. He has a history of diabetes and
    hypertension. ECG shows ST elevation in leads II, III, aVF.
    """
    test_ground_truth = {
        'STEMI': ['ST elevation in leads II, III, aVF', 'acute chest pain'],
        'Acute Coronary Syndrome': ['chest pain', 'cardiac risk factors'],
        'Myocardial Infarction': ['crushing chest pain', 'radiation to left arm']
    }

    try:
        # Test the runner
        runner = DDxRunner()

        # Test initialization
        if not runner.initialize_system():
            print("❌ Runner initialization failed")
            return False

        # Test system status
        status = runner.get_system_status()
        print(f"📊 System Status: {status['status']}")
        print(f"   Models loaded: {status['models_loaded']}")
        print(f"   Ready for cases: {status['ready_for_cases']}")

        # Test case execution
        result = runner.run_case(test_case_name, test_case_description, test_ground_truth)

        if result['success']:
            print(f"\n✅ Runner v6 test successful!")
            print(f"   Duration: {result['duration']:.1f}s")
            print(f"   Agents: {result['agents_generated']}")
            print(f"   Rounds: {result['rounds_completed']}")
            return True
        else:
            print(f"\n❌ Case execution failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Runner test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    test_runner_v6()
