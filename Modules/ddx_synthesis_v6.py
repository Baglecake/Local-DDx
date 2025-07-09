# =============================================================================
# DDx Synthesis v6 - Integrated Advanced Analysis & Evaluation
# =============================================================================

"""
DDx Synthesis v6: Clean rebuild that integrates with v6 architecture while
preserving all sophisticated features (TempoScore, Dr. Reed, credibility weighting)
"""

import time
import json
import statistics
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

# Import v6 core components
from ddx_core_v6 import DDxSystem, DDxAgent, AgentResponse
from ddx_rounds_v6 import RoundResult, RoundType, RoundOrchestrator
from ddx_utils import extract_diagnoses, validate_medical_response
from ddx_evaluator_v6 import EnhancedClinicalEvaluator

# =============================================================================
# 1. TempoScore Calculation System (Preserved from original)
# =============================================================================

@dataclass
class TempoScoreMetrics:
    """Metrics for calculating TempoScore"""
    round_type: str
    unique_diagnoses: int = 0
    high_value_interactions: int = 0
    minimum_expected_interactions: int = 0
    symbolic_curvature: float = 0.0
    tempo_score: float = 0.0
    calculation_method: str = ""

class TempoScoreCalculator:
    """Calculate TempoScore for different round types"""

    def __init__(self):
        self.round_scores = {}

    def calculate_round_tempo(self, round_result: RoundResult) -> TempoScoreMetrics:
        """Calculate TempoScore based on round type and results"""

        if round_result.round_type == RoundType.TEAM_INDEPENDENT_DIFFERENTIALS:
            return self._calculate_independent_tempo(round_result)
        elif round_result.round_type == RoundType.REFINEMENT_AND_JUSTIFICATION:
            return self._calculate_debate_tempo(round_result)
        else:
            # Default tempo for other rounds
            return TempoScoreMetrics(
                round_type=round_result.round_type.value,
                tempo_score=1.0,
                calculation_method="Default tempo score"
            )

    def _calculate_independent_tempo(self, round_result: RoundResult) -> TempoScoreMetrics:
        """Calculate TempoScore for Independent Differentials round"""

        unique_diagnoses = round_result.metadata.get('total_diagnoses', 0)

        # Formula: TempoScore = 1.0 + (0.1 × Number of unique diagnoses)
        tempo_score = 1.0 + (0.1 * unique_diagnoses)

        return TempoScoreMetrics(
            round_type="Independent Differentials",
            unique_diagnoses=unique_diagnoses,
            tempo_score=tempo_score,
            calculation_method=f"1.0 + (0.1 × {unique_diagnoses}) = {tempo_score:.1f}"
        )

    def _calculate_debate_tempo(self, round_result: RoundResult) -> TempoScoreMetrics:
        """Calculate TempoScore for Debate round using Symbolic Curvature"""

        high_value_interactions = 0
        minimum_expected_interactions = len(round_result.participants)

        # Count high-value interactions from responses
        for response in round_result.responses.values():
            if self._is_high_value_interaction(response.content):
                high_value_interactions += 1

        # Calculate Symbolic Curvature
        symbolic_curvature = (high_value_interactions / minimum_expected_interactions
                            if minimum_expected_interactions > 0 else 0)

        # Formula: TempoScore = Symbolic Curvature + 0.5 (capped at 2.5)
        tempo_score = min(symbolic_curvature + 0.5, 2.5)

        return TempoScoreMetrics(
            round_type="Debate/Refinement",
            high_value_interactions=high_value_interactions,
            minimum_expected_interactions=minimum_expected_interactions,
            symbolic_curvature=symbolic_curvature,
            tempo_score=tempo_score,
            calculation_method=f"min({symbolic_curvature:.2f} + 0.5, 2.5) = {tempo_score:.1f}"
        )

    def _is_high_value_interaction(self, content: str) -> bool:
        """Determine if an interaction is high-value based on content analysis"""
        content_lower = content.lower()

        # High-value interaction indicators
        high_value_patterns = [
            'evidence', 'support', 'challenge', 'disagree', 'alternative',
            'clinical reasoning', 'risk factor', 'contradict', 'position',
            'recommendations', 'differential'
        ]

        pattern_count = sum(1 for pattern in high_value_patterns if pattern in content_lower)

        # Check for structured format markers
        has_structured_format = any(marker in content for marker in [
            '**Your Position**', '**Clinical Evidence**', '**Reasoning**',
            '**Alternative Considerations**', '**Final Recommendations**'
        ])

        # High-value if has multiple reasoning patterns or structured format
        return pattern_count >= 3 or has_structured_format

# =============================================================================
# 2. Dr. Reed Assessment System (Preserved from original)
# =============================================================================

@dataclass
class SpecialistAssessment:
    """Assessment data for a specialist"""
    agent_name: str
    specialty: str
    relevance_weight: float
    round_scores: Dict[str, Dict] = field(default_factory=dict)
    final_score: float = 0.0

class DrReedAssessment:
    """Dr. Reed assessment system for evaluating specialist performance"""

    def __init__(self):
        self.assessments = {}
        self.tempo_scores = {}

    def assess_specialists(self, agents: List[DDxAgent],
                         round_results: Dict[RoundType, RoundResult],
                         tempo_calculator: TempoScoreCalculator) -> Dict[str, Any]:
        """Conduct Dr. Reed assessment of all specialists"""

        print("\n🏥 DR. REED'S ASSESSMENT")
        print("=" * 50)

        # Calculate TempoScores for each round
        self._calculate_round_tempo_scores(round_results, tempo_calculator)

        # Assess each specialist
        for agent in agents:
            assessment = self._assess_individual_specialist(agent, round_results)
            self.assessments[agent.name] = assessment

        # Generate final score table
        score_table = self._generate_score_table()

        print(f"\n📊 SPECIALIST PERFORMANCE SCORES:")
        for i, (agent_name, score) in enumerate(score_table, 1):
            print(f"   {i}. {agent_name}: {score:.1f}/120")

        return {
            'tempo_scores': self.tempo_scores,
            'specialist_assessments': self.assessments,
            'final_score_table': score_table
        }

    def _calculate_round_tempo_scores(self, round_results: Dict[RoundType, RoundResult],
                                    tempo_calculator: TempoScoreCalculator):
        """Calculate TempoScore for each completed round"""

        print("📊 Calculating TempoScores...")

        for round_type, result in round_results.items():
            if result.success:
                metrics = tempo_calculator.calculate_round_tempo(result)
                self.tempo_scores[round_type.value] = metrics
                print(f"   {round_type.value.title()}: {metrics.tempo_score:.1f}")
                print(f"      Calculation: {metrics.calculation_method}")

    def _assess_individual_specialist(self, agent: DDxAgent,
                                    round_results: Dict[RoundType, RoundResult]) -> SpecialistAssessment:
        """Assess an individual specialist's performance"""

        # Determine relevance weight based on case relevance score
        relevance_weight = min(agent.config.case_relevance_score / 15.0, 1.0)

        assessment = SpecialistAssessment(
            agent_name=agent.name,
            specialty=agent.specialty.value,
            relevance_weight=relevance_weight
        )

        # Assess each round the specialist participated in
        total_score = 0
        round_count = 0

        for round_type, result in round_results.items():
            if result.success and agent.name in result.responses:
                response = result.responses[agent.name]
                round_score = self._score_round_performance(
                    agent.name, response, round_type.value
                )
                assessment.round_scores[round_type.value] = round_score
                total_score += round_score['final_score']
                round_count += 1

        # Calculate final score
        assessment.final_score = total_score / round_count if round_count > 0 else 0

        return assessment

    def _score_round_performance(self, agent_name: str, response: AgentResponse,
                               round_name: str) -> Dict:
        """Score a specialist's performance in a specific round"""

        content = response.content

        # Score based on round type and content quality
        if round_name == "team_independent_differentials":
            insight_score = self._score_diagnostic_insight(response)
            synthesis_score = self._score_diagnostic_breadth(response)
            action_score = self._score_diagnostic_clarity(response)
        else:
            insight_score = self._score_content_insight(content)
            synthesis_score = self._score_content_synthesis(content)
            action_score = self._score_content_action(content)

        # Calculate Base_DES: (2.5 × Insight) + (1.5 × Synthesis) + (1.0 × Action)
        base_des = (2.5 * insight_score) + (1.5 * synthesis_score) + (1.0 * action_score)

        professional_valence = self._assess_professional_valence(content, round_name)
        final_score = base_des * professional_valence

        return {
            'insight_score': insight_score,
            'synthesis_score': synthesis_score,
            'action_score': action_score,
            'base_des': base_des,
            'professional_valence': professional_valence,
            'final_score': final_score
        }

    def _score_diagnostic_insight(self, response: AgentResponse) -> float:
        """Score diagnostic insight based on structured data quality"""
        if response.structured_data:
            num_diagnoses = len(response.structured_data)
            total_evidence = sum(len(evidence) for evidence in response.structured_data.values())
            avg_evidence = total_evidence / num_diagnoses if num_diagnoses > 0 else 0

            # Score: diagnoses + evidence quality
            score = min((num_diagnoses * 3) + (avg_evidence * 2), 20.0)
            return score
        else:
            return self._score_content_insight(response.content)

    def _score_diagnostic_breadth(self, response: AgentResponse) -> float:
        """Score diagnostic breadth"""
        if response.structured_data:
            return min(len(response.structured_data) * 4, 20.0)
        else:
            return self._score_content_synthesis(response.content)

    def _score_diagnostic_clarity(self, response: AgentResponse) -> float:
        """Score diagnostic clarity and actionability"""
        if response.structured_data:
            # Check if primary diagnosis has good evidence
            if response.structured_data:
                first_diag_evidence = next(iter(response.structured_data.values()), [])
                return 20.0 if len(first_diag_evidence) >= 2 else 10.0
        return self._score_content_action(response.content)

    def _score_content_insight(self, content: str) -> float:
        """Score content insight for non-differential rounds"""
        content_lower = content.lower()
        insight_indicators = [
            'evidence', 'diagnosis', 'symptoms', 'findings', 'clinical',
            'pathophysiology', 'risk factors', 'differential', 'presentation'
        ]
        indicator_count = sum(1 for indicator in insight_indicators if indicator in content_lower)
        return min(indicator_count * 2.5, 20.0)

    def _score_content_synthesis(self, content: str) -> float:
        """Score content synthesis quality"""
        content_lower = content.lower()
        synthesis_indicators = [
            'integrat', 'combin', 'togeth', 'consid', 'balanc', 'weigh'
        ]
        indicator_count = sum(1 for indicator in synthesis_indicators if indicator in content_lower)
        return min(indicator_count * 4, 20.0)

    def _score_content_action(self, content: str) -> float:
        """Score content actionability"""
        content_lower = content.lower()
        action_indicators = [
            'recommend', 'suggest', 'should', 'need', 'action', 'next', 'plan'
        ]
        indicator_count = sum(1 for indicator in action_indicators if indicator in content_lower)
        return min(indicator_count * 3, 20.0)

    def _assess_professional_valence(self, content: str, round_name: str) -> float:
        """Assess epistemic conduct quality"""
        content_lower = content.lower()

        # Look for evidence of elevating team performance
        elevating_indicators = [
            'evidence', 'research', 'studies', 'guidelines', 'consider',
            'alternative', 'question', 'challenge', 'disagree'
        ]

        elevating_count = sum(1 for indicator in elevating_indicators if indicator in content_lower)

        # Determine valence based on contribution quality
        if elevating_count >= 4 and len(content) > 200:
            return 1.2  # Significantly elevated team performance
        elif elevating_count >= 2 and len(content) > 100:
            return 1.0  # Professional contribution
        elif len(content) > 50:
            return 0.8  # Basic contribution
        else:
            return 0.6  # Minimal contribution

    def _generate_score_table(self) -> List[Tuple[str, float]]:
        """Generate final score table sorted by performance"""
        score_table = []
        for agent_name, assessment in self.assessments.items():
            score_table.append((agent_name, assessment.final_score))

        # Sort by score (highest first)
        score_table.sort(key=lambda x: x[1], reverse=True)
        return score_table

# =============================================================================
# 3. Diagnosis Synthesis System (Enhanced for v6)
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of diagnosis synthesis"""
    provisional_list: List[str]
    final_list: List[str]
    credibility_scores: Dict[str, float]
    selection_reasoning: Dict[str, str]
    consolidation_applied: bool = False

class DiagnosisSynthesizer:
    """Synthesizes final diagnosis from round results and assessments"""

    def __init__(self):
        self.synthesis_result = None

    def synthesize_final_diagnosis(self, round_results: Dict[RoundType, RoundResult],
                                 reed_assessment: Dict[str, Any]) -> SynthesisResult:
        """Synthesize final diagnosis using round results and credibility scores"""

        print("\n🔄 DIAGNOSIS SYNTHESIS")
        print("=" * 50)

        # Extract master diagnosis list from independent differentials
        master_list = self._extract_master_diagnosis_list(round_results)
        print(f"📋 Master diagnosis list: {len(master_list)} diagnoses")

        # Get credibility scores from Reed assessment
        credibility_scores = self._extract_credibility_scores(reed_assessment)

        # Apply credibility-weighted selection
        provisional_list = self._apply_credibility_selection(master_list, credibility_scores, round_results)
        print(f"📊 Provisional selection: {len(provisional_list)} diagnoses")

        # Apply consolidation based on tempo score
        final_list, consolidation_applied = self._apply_consolidation(
            provisional_list, reed_assessment.get('tempo_scores', {})
        )

        print(f"✅ Final diagnosis list: {len(final_list)} diagnoses")
        for i, diagnosis in enumerate(final_list, 1):
            print(f"   {i}. {diagnosis}")

        result = SynthesisResult(
            provisional_list=provisional_list,
            final_list=final_list,
            credibility_scores=credibility_scores,
            selection_reasoning=self._generate_selection_reasoning(
                master_list, provisional_list, final_list
            ),
            consolidation_applied=consolidation_applied
        )

        self.synthesis_result = result
        return result

    def _extract_master_diagnosis_list(self, round_results: Dict[RoundType, RoundResult]) -> List[str]:
        """Extract all unique diagnoses from independent differentials"""
        all_diagnoses = set()

        if RoundType.TEAM_INDEPENDENT_DIFFERENTIALS in round_results:
            result = round_results[RoundType.TEAM_INDEPENDENT_DIFFERENTIALS]
            if result.success and 'all_diagnoses' in result.metadata:
                # Extract diagnosis names from structured data
                for diag_data in result.metadata['all_diagnoses']:
                    if isinstance(diag_data, dict) and 'diagnosis' in diag_data:
                        all_diagnoses.add(diag_data['diagnosis'])
                    elif isinstance(diag_data, str):
                        all_diagnoses.add(diag_data)

        return sorted(list(all_diagnoses))

    def _extract_credibility_scores(self, reed_assessment: Dict[str, Any]) -> Dict[str, float]:
        """Extract credibility scores from Reed assessment"""
        credibility_scores = {}

        if 'specialist_assessments' in reed_assessment:
            for agent_name, assessment in reed_assessment['specialist_assessments'].items():
                credibility_scores[agent_name] = assessment.final_score

        return credibility_scores

    def _apply_credibility_selection(self, master_list: List[str],
                                   credibility_scores: Dict[str, float],
                                   round_results: Dict[RoundType, RoundResult]) -> List[str]:
        """Apply credibility-weighted preferential voting selection"""

        print(f"🗳️ Applying credibility-weighted preferential voting selection...")

        # Extract preferential voting data
        borda_scores = {}
        all_rankings = {}

        if RoundType.POST_DEBATE_VOTING in round_results:
            voting_result = round_results[RoundType.POST_DEBATE_VOTING]
            if voting_result.success:
                borda_scores = voting_result.metadata.get('borda_scores', {})
                all_rankings = voting_result.metadata.get('all_rankings', {})
                print(f"   ✅ Found preferential voting data for {len(borda_scores)} diagnoses")
            else:
                print(f"   ⚠️ Preferential voting round failed")
                return self._fallback_selection(master_list, credibility_scores, round_results)
        else:
            print(f"   ⚠️ No preferential voting round found")
            return self._fallback_selection(master_list, credibility_scores, round_results)

        if not borda_scores:
            print(f"   ⚠️ No Borda scores available")
            return self._fallback_selection(master_list, credibility_scores, round_results)

        provisional_list = []

        # Apply credibility weighting to Borda scores
        print(f"\n📊 Applying credibility-weighted scoring:")

        credibility_weighted_scores = {}

        for diagnosis, base_score in borda_scores.items():
            weighted_score = 0
            supporting_agents = []

            # Find which agents ranked this diagnosis and their positions
            for agent_name, ranking in all_rankings.items():
                if diagnosis in ranking:
                    position = ranking.index(diagnosis) + 1  # 1-indexed position
                    agent_credibility = credibility_scores.get(agent_name, 1.0)

                    # Calculate position-based points (higher position = fewer points)
                    max_rank = len(ranking)
                    position_points = max_rank - position + 1

                    # Weight by credibility
                    contribution = position_points * agent_credibility
                    weighted_score += contribution
                    supporting_agents.append(f"{agent_name}(#{position}:{agent_credibility:.1f})")

            if weighted_score > 0:
                credibility_weighted_scores[diagnosis] = weighted_score
                agent_details = ", ".join(supporting_agents)
                print(f"   {diagnosis}: {weighted_score:.1f} pts [{agent_details}]")

        # Sort by credibility-weighted score and apply selection threshold
        sorted_diagnoses = sorted(
            credibility_weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Dynamic threshold based on score distribution
        if sorted_diagnoses:
            max_score = sorted_diagnoses[0][1]
            min_threshold = max_score * 0.3  # At least 30% of max score

            print(f"\n🎯 Selection criteria (min threshold: {min_threshold:.1f}):")

            for diagnosis, score in sorted_diagnoses:
                if score >= min_threshold and len(provisional_list) < 10:  # Max 10 diagnoses
                    provisional_list.append(diagnosis)
                    percentage = (score / max_score) * 100
                    print(f"   ✅ {diagnosis}: {score:.1f} pts ({percentage:.0f}% of max)")
                else:
                    print(f"   ❌ {diagnosis}: {score:.1f} pts (below threshold)")

        print(f"\n🗳️ Selected {len(provisional_list)} diagnoses via credibility-weighted preferential voting")

        return provisional_list

    def _fallback_selection(self, master_list: List[str],
                          credibility_scores: Dict[str, float],
                          round_results: Dict[RoundType, RoundResult]) -> List[str]:
        """Fallback selection when voting data unavailable"""
        print(f"🔄 Using fallback selection based on agent credibility...")

        # Get diagnoses from independent differentials round
        diagnosis_supporters = {}

        if RoundType.TEAM_INDEPENDENT_DIFFERENTIALS in round_results:
            diff_result = round_results[RoundType.TEAM_INDEPENDENT_DIFFERENTIALS]

            for agent_name, response in diff_result.responses.items():
                if response.structured_data:
                    for diagnosis in response.structured_data.keys():
                        if diagnosis not in diagnosis_supporters:
                            diagnosis_supporters[diagnosis] = []
                        diagnosis_supporters[diagnosis].append(agent_name)

        # Score diagnoses by credibility of supporters
        diagnosis_scores = {}
        for diagnosis, supporters in diagnosis_supporters.items():
            total_credibility = sum(credibility_scores.get(supporter, 50.0) for supporter in supporters)
            diagnosis_scores[diagnosis] = total_credibility

        # Select top diagnoses
        sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda x: x[1], reverse=True)

        provisional_list = []
        for diagnosis, score in sorted_diagnoses[:7]:  # Top 7
            provisional_list.append(diagnosis)
            print(f"   ✅ {diagnosis}: {score:.1f} credibility points")

        return provisional_list

    def _apply_consolidation(self, provisional_list: List[str],
                           tempo_scores: Dict[str, Any]) -> Tuple[List[str], bool]:
        """Apply semantic consolidation based on tempo scores"""

        # Get refinement tempo score
        refinement_tempo = 1.0
        for round_name, metrics in tempo_scores.items():
            if 'refinement' in round_name.lower():
                if hasattr(metrics, 'tempo_score'):
                    refinement_tempo = metrics.tempo_score
                break

        consolidation_applied = False

        if refinement_tempo > 1.5:
            print(f"🔥 High tempo ({refinement_tempo:.1f}) - applying consolidation")
            final_list = self._consolidate_similar_diagnoses(provisional_list)
            consolidation_applied = True
        else:
            print(f"📝 Standard tempo ({refinement_tempo:.1f}) - minimal consolidation")
            final_list = list(set(provisional_list))  # Simple deduplication

        return final_list, consolidation_applied

    def _consolidate_similar_diagnoses(self, provisional_list: List[str]) -> List[str]:
        """Consolidate semantically similar diagnoses"""
        final_list = []
        processed = set()

        for diagnosis in provisional_list:
            if diagnosis in processed:
                continue

            # Check for similar diagnoses to consolidate
            similar_found = False
            diagnosis_lower = diagnosis.lower()

            for other in provisional_list:
                if other != diagnosis and other not in processed:
                    other_lower = other.lower()

                    # Consolidation rules
                    if self._should_consolidate(diagnosis_lower, other_lower):
                        # Keep the more specific one
                        if len(diagnosis) > len(other):
                            final_list.append(diagnosis)
                        else:
                            final_list.append(other)
                        processed.add(diagnosis)
                        processed.add(other)
                        similar_found = True
                        break

            if not similar_found:
                final_list.append(diagnosis)
                processed.add(diagnosis)

        return final_list

    def _should_consolidate(self, diag1: str, diag2: str) -> bool:
        """Determine if two diagnoses should be consolidated"""
        # Consolidate myocardial infarction variants
        mi_terms = ['myocardial infarction', 'stemi', 'acute coronary syndrome']
        if any(term in diag1 for term in mi_terms) and any(term in diag2 for term in mi_terms):
            return True

        # Consolidate Dengue variants
        dengue_terms = ['dengue fever', 'dengue']
        if any(term in diag1 for term in dengue_terms) and any(term in diag2 for term in dengue_terms):
            return True

        # Add more consolidation rules as needed
        return False

    def _generate_selection_reasoning(self, master_list: List[str],
                                    provisional_list: List[str],
                                    final_list: List[str]) -> Dict[str, str]:
        """Generate reasoning for selection decisions"""
        return {
            'master_list_size': f"{len(master_list)} total diagnoses from specialists",
            'provisional_selection': f"{len(provisional_list)} diagnoses selected based on credibility weighting",
            'final_selection': f"{len(final_list)} diagnoses after consolidation",
            'selection_criteria': "Credibility-weighted preferential voting with consolidation"
        }

# =============================================================================
# 4. Synthesis & Evaluation Manager (v6 Integration)
# =============================================================================

class SynthesisEvaluationManager:
    """Manages complete synthesis and evaluation pipeline for v6"""

    def __init__(self, ddx_system: DDxSystem):
        self.ddx_system = ddx_system
        self.tempo_calculator = TempoScoreCalculator()
        self.dr_reed = DrReedAssessment()
        self.synthesizer = DiagnosisSynthesizer()

    def run_complete_analysis(self, round_results: Dict[RoundType, RoundResult],
                            ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Run complete synthesis and evaluation analysis"""

        print(f"\n🔬 COMPLETE ANALYSIS PIPELINE")
        print("=" * 60)

        # Step 1: Dr. Reed Assessment
        reed_results = self.dr_reed.assess_specialists(
            self.ddx_system.current_agents, round_results, self.tempo_calculator
        )

        # Step 2: Diagnosis Synthesis
        synthesis_result = self.synthesizer.synthesize_final_diagnosis(
            round_results, reed_results
        )

        # Step 3: Evaluation (if ground truth provided)
        evaluation_result = None
        if ground_truth:
            # Import evaluation module when needed
            try:
                from ddx_evaluator_v6 import EnhancedClinicalEvaluator
                evaluator = EnhancedClinicalEvaluator(self.ddx_system.model_manager)

                # Set transcript for context-aware evaluation
                try:
                    orchestrator = getattr(self.ddx_system, 'round_orchestrator', None)
                    if (orchestrator is not None and
                        hasattr(orchestrator, 'full_transcript')):
                        evaluator.set_transcript(orchestrator.full_transcript)
                        print(f"📋 Transcript passed to evaluator for context analysis")
                except Exception as e:
                    print(f"⚠️ Could not set transcript: {e}")

                evaluation_result = evaluator.evaluate_diagnosis(
                    synthesis_result,
                    ground_truth,
                    round_results
                )
            except ImportError:
                print("⚠️ Enhanced evaluator not available, skipping evaluation")

        return {
            'reed_assessment': reed_results,
            'synthesis_result': synthesis_result,
            'evaluation_result': evaluation_result,
            'success': True
        }

    def export_to_json(self, case_id: str, analysis_results: Dict[str, Any],
                      ground_truth: Dict[str, List[str]], save_file: bool = True) -> Dict[str, Any]:
        """Export results to JSON format"""

        json_results = self._build_json_results(case_id, analysis_results, ground_truth)

        if save_file:
            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ddx_results_{case_id.replace(' ', '_').replace(':', '')}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)

            print(f"📊 JSON results saved to: {filename}")

        return json_results

    def _build_json_results(self, case_id: str, analysis_results: Dict[str, Any],
                          ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build structured JSON results"""

        evaluation = analysis_results.get('evaluation_result')
        synthesis = analysis_results.get('synthesis_result')
        reed_assessment = analysis_results.get('reed_assessment')

        if not evaluation:
            return {"error": "No evaluation results available"}

        # Build structured JSON
        json_results = {
            "case_id": case_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluation_method": "Enhanced_DD_Evaluator_v6",

            # Core metrics
            "tp_count": evaluation.tp_count,
            "fp_count": evaluation.fp_count,
            "fn_count": evaluation.fn_count,
            "caa_count": evaluation.caa_count,
            "ae_count": evaluation.ae_count,

            # Totals
            "total_gt_diagnoses": len(ground_truth),
            "total_team_diagnoses": len(synthesis.final_list) if synthesis else 0,

            # Rates
            "tp_rate": evaluation.tp_rate,
            "caa_weight_applied": evaluation.caa_weight_applied,

            # Performance metrics
            "traditional_precision": evaluation.traditional_precision,
            "traditional_recall": evaluation.traditional_recall,
            "clinical_reasoning_quality": evaluation.clinical_reasoning_quality,
            "diagnostic_safety": evaluation.diagnostic_safety,

            # Diagnosis lists
            "final_diagnoses": synthesis.final_list if synthesis else [],
            "ground_truth_diagnoses": list(ground_truth.keys()),

            # Matches and misses
            "true_positives": [{"ground_truth": gt, "system": sys} for gt, sys in evaluation.matched_diagnoses],
            "false_positives": evaluation.false_positives,
            "false_negatives": evaluation.false_negatives,

            # Enhanced features
            "ai_enhanced_matching": True,
            "context_informed_matches": len(evaluation.context_matches) if hasattr(evaluation, 'context_matches') else 0,

            # Credibility scores
            "credibility_scores": synthesis.credibility_scores if synthesis else {},

            # Performance tier
            "weighting_tier": self._determine_performance_tier(evaluation.traditional_recall, evaluation.clinical_reasoning_quality),

            # Specialist performance
            "specialist_performance": self._extract_specialist_performance(reed_assessment) if reed_assessment else {},

            # Round metrics
            "round_completion": self._extract_round_metrics(analysis_results),

            "full_transcript": self._extract_full_transcript(analysis_results),

        }

        return json_results

    def _extract_full_transcript(self, analysis_results: Dict) -> Dict[str, Any]:
        """Extract full transcript for JSON export"""
        try:
            # Get transcript from DDx system
            if (hasattr(self.ddx_system, 'round_orchestrator') and
                hasattr(self.ddx_system.round_orchestrator, 'full_transcript')):
                return self.ddx_system.round_orchestrator.full_transcript
            else:
                return {"note": "Transcript not available"}
        except Exception as e:
            return {"error": f"Could not extract transcript: {str(e)}"}

    def _determine_performance_tier(self, recall: float, clinical_quality: float) -> str:
        """Determine performance tier based on metrics"""
        avg_score = (recall + clinical_quality) / 2

        if avg_score >= 0.8:
            return "Excellent"
        elif avg_score >= 0.6:
            return "Good"
        elif avg_score >= 0.4:
            return "Moderate"
        else:
            return "Needs Improvement"

    def _extract_specialist_performance(self, reed_assessment: Dict) -> Dict[str, Any]:
        """Extract specialist performance summary"""
        if not reed_assessment or 'specialist_assessments' not in reed_assessment:
            return {}

        scores = []
        for name, assessment in reed_assessment['specialist_assessments'].items():
            scores.append({
                "name": name,
                "specialty": assessment.specialty,
                "final_score": assessment.final_score,
                "relevance_weight": assessment.relevance_weight
            })

        # Sort by score
        scores.sort(key=lambda x: x['final_score'], reverse=True)

        return {
            "top_performer": scores[0]['name'] if scores else None,
            "top_score": scores[0]['final_score'] if scores else 0,
            "average_score": sum(s['final_score'] for s in scores) / len(scores) if scores else 0,
            "specialist_rankings": scores
        }

    def _extract_round_metrics(self, analysis_results: Dict) -> Dict[str, Any]:
        """Extract round completion metrics"""
        reed = analysis_results.get('reed_assessment', {})
        tempo_scores = reed.get('tempo_scores', {})

        # Convert tempo scores safely
        tempo_dict = {}
        if tempo_scores:
            for round_name, metrics in tempo_scores.items():
                if metrics and hasattr(metrics, 'tempo_score'):
                    tempo_dict[round_name] = metrics.tempo_score

        # Get debate quality safely
        debate_quality = 0
        for round_name, metrics in tempo_scores.items():
            if 'refinement' in round_name.lower() and hasattr(metrics, 'tempo_score'):
                debate_quality = metrics.tempo_score
                break

        return {
            "rounds_completed": len(tempo_scores) if tempo_scores else 0,
            "tempo_scores": tempo_dict,
            "debate_quality": debate_quality
        }

# =============================================================================
# 5. Integration with DDxSystem v6
# =============================================================================

def integrate_synthesis_with_ddx_system():
    """Extend DDxSystem with synthesis and evaluation capabilities"""

    def add_synthesis_manager(self):
        """Add synthesis and evaluation manager to DDx system"""
        if not hasattr(self, 'synthesis_manager'):
            self.synthesis_manager = SynthesisEvaluationManager(self)

    def run_complete_analysis(self, ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        if not hasattr(self, 'synthesis_manager'):
            self.add_synthesis_manager()

        if not hasattr(self, 'round_orchestrator') or not self.round_orchestrator.round_results:
            raise ValueError("No round results available. Run diagnostic rounds first.")

        return self.synthesis_manager.run_complete_analysis(
            self.round_orchestrator.round_results, ground_truth
        )

    # Add methods to DDxSystem class
    from ddx_core_v6 import DDxSystem
    DDxSystem.add_synthesis_manager = add_synthesis_manager
    DDxSystem.run_complete_analysis = run_complete_analysis

# Apply integration
integrate_synthesis_with_ddx_system()

# =============================================================================
# 6. Testing Function
# =============================================================================

def test_synthesis_system():
    """Test the complete synthesis and evaluation system"""
    print("🧪 Testing Synthesis & Evaluation Integration v6")
    print("=" * 60)

    # Import and use the complete system
    from ddx_core_v6 import DDxSystem

    # Initialize system
    ddx = DDxSystem()
    if not ddx.initialize():
        return False

    # Test case
    test_case = """
    A 45-year-old male presents with acute chest pain that began 2 hours ago.
    The pain is crushing, radiates to the left arm, and is associated with
    shortness of breath and diaphoresis. He has a history of diabetes and
    hypertension. ECG shows ST elevation in leads II, III, aVF.
    """

    # Ground truth for evaluation
    ground_truth = {
        'Myocardial Infarction': ['crushing chest pain', 'ST elevation', 'radiation to left arm'],
        'STEMI': ['ST elevation in leads II, III, aVF', 'acute onset'],
        'Acute Coronary Syndrome': ['chest pain', 'cardiac risk factors', 'ECG changes']
    }

    try:
        # Run complete pipeline
        result = ddx.analyze_case(test_case, "synthesis_test_case_v6")
        if not result['success']:
            print(f"❌ Case analysis failed")
            return False

        # Run diagnostic rounds
        if not hasattr(ddx, 'run_complete_diagnostic_sequence'):
            # Manual integration if not already done
            from ddx_rounds_v6 import integrate_rounds_with_ddx_system
            integration_methods = integrate_rounds_with_ddx_system()
            ddx.add_round_orchestrator = integration_methods['add_round_orchestrator'].__get__(ddx, type(ddx))
            ddx.run_complete_diagnostic_sequence = integration_methods['run_diagnostic_sequence'].__get__(ddx, type(ddx))
            ddx.get_diagnostic_summary = integration_methods['get_diagnostic_summary'].__get__(ddx, type(ddx))

        rounds_results = ddx.run_complete_diagnostic_sequence()

        # Run complete analysis with evaluation
        analysis_results = ddx.run_complete_analysis(ground_truth)

        if analysis_results['success']:
            print(f"\n🎉 Complete Analysis Success!")

            # Show key results
            synthesis = analysis_results['synthesis_result']
            evaluation = analysis_results['evaluation_result']

            print(f"\n📊 Final Results:")
            print(f"   Final Diagnoses: {len(synthesis.final_list)}")
            if evaluation:
                print(f"   Precision: {evaluation.traditional_precision:.3f}")
                print(f"   Recall: {evaluation.traditional_recall:.3f}")
                print(f"   Clinical Quality: {evaluation.clinical_reasoning_quality:.3f}")

            return ddx
        else:
            print(f"❌ Analysis failed")
            return False

    except Exception as e:
        print(f"❌ Synthesis system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the complete system
    system = test_synthesis_system()

    if system:
        print(f"\n🎉 Complete DDx System v6 Ready!")
        print(f"✅ Dynamic generation integrated")
        print(f"✅ Sophisticated rounds system")
        print(f"✅ TempoScore calculation")
        print(f"✅ Dr. Reed assessment")
        print(f"✅ Advanced synthesis")
        print(f"✅ Enhanced evaluation")
