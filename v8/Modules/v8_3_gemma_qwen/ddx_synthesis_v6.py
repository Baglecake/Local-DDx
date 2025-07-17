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
from ddx_evaluator_v9 import EnhancedClinicalEvaluator


# from ddx_evaluator_v8 import ClinicalEvaluator as EnhancedClinicalEvaluator

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

    def get_evaluation_list(self, ground_truth_size: int) -> List[str]:
        """Get top N diagnoses for evaluation where N = ground truth size"""
        return self.final_list[:ground_truth_size]

class DiagnosisSynthesizer:
    """Synthesizes final diagnosis from round results and assessments"""

    def __init__(self):
        self.synthesis_result = None

    def synthesize_final_diagnosis(self, round_results: Dict[RoundType, RoundResult],
                             reed_assessment: Dict[str, Any]) -> SynthesisResult:
        """Synthesize final diagnosis using round results and credibility scores (NO ground truth knowledge)"""

        print("\n🔄 DIAGNOSIS SYNTHESIS")
        print("=" * 50)

        # Extract master diagnosis list from independent differentials
        master_list = self._extract_master_diagnosis_list(round_results)
        print(f"📋 Master diagnosis list: {len(master_list)} diagnoses")

        # Get credibility scores from Reed assessment
        credibility_scores = self._extract_credibility_scores(reed_assessment)

        # Apply credibility-weighted selection (clean, no GT knowledge)
        provisional_list = self._apply_credibility_selection(
            master_list, credibility_scores, round_results
        )
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
        """Apply credibility-weighted preferential voting selection with pre-normalization"""

        print(f"🗳️ Applying credibility-weighted preferential voting selection...")

        # Extract preferential voting data with PRE-NORMALIZATION
        raw_borda_scores = {}
        raw_all_rankings = {}

        if RoundType.POST_DEBATE_VOTING in round_results:
            voting_result = round_results[RoundType.POST_DEBATE_VOTING]
            if voting_result.success:
                raw_borda_scores = voting_result.metadata.get('borda_scores', {})
                raw_all_rankings = voting_result.metadata.get('all_rankings', {})
                print(f"   ✅ Found raw voting data for {len(raw_borda_scores)} diagnoses")

                # PRE-NORMALIZE all diagnosis candidates before vote tallying
                normalized_scores = {}
                normalized_rankings = {}

                # Create normalization mapping
                normalization_map = {}
                for diagnosis in raw_borda_scores.keys():
                    normalized = self._normalize_diagnosis_for_synthesis(diagnosis)
                    if normalized:  # Only keep valid normalized diagnoses
                        normalization_map[diagnosis] = normalized

                # Consolidate scores by normalized name
                for original_diag, score in raw_borda_scores.items():
                    if original_diag in normalization_map:
                        normalized_name = normalization_map[original_diag]
                        normalized_scores[normalized_name] = normalized_scores.get(normalized_name, 0) + score

                # Consolidate rankings by normalized name
                for agent_name, ranking in raw_all_rankings.items():
                    normalized_ranking = []
                    seen_normalized = set()
                    for original_diag in ranking:
                        if original_diag in normalization_map:
                            normalized_name = normalization_map[original_diag]
                            if normalized_name not in seen_normalized:  # Avoid duplicates within same agent
                                normalized_ranking.append(normalized_name)
                                seen_normalized.add(normalized_name)
                    if normalized_ranking:  # Only keep non-empty rankings
                        normalized_rankings[agent_name] = normalized_ranking

                borda_scores = normalized_scores
                all_rankings = normalized_rankings

                print(f"   🔄 Pre-normalized: {len(raw_borda_scores)} → {len(borda_scores)} unique diagnoses")

            else:
                print(f"   ⚠️ Preferential voting round failed")
                return self._fallback_selection(master_list, credibility_scores, round_results)
        else:
            print(f"   ⚠️ No preferential voting round found")
            return self._fallback_selection(master_list, credibility_scores, round_results)

        if not borda_scores:
            print(f"   ⚠️ No Borda scores available after normalization")
            return self._fallback_selection(master_list, credibility_scores, round_results)

        # Apply credibility weighting to normalized scores
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

                    # Calculate median credibility for capping
                    all_credibilities = list(credibility_scores.values())
                    median_credibility = statistics.median(all_credibilities) if all_credibilities else 50.0

                    # Cap credibility weighting to prevent dominance
                    max_credibility = median_credibility * 2.0  # Cap at 2x median
                    capped_credibility = min(agent_credibility, max_credibility)

                    # Weight by capped credibility
                    contribution = position_points * capped_credibility
                    weighted_score += contribution
                    supporting_agents.append(f"{agent_name}(#{position}:{capped_credibility:.1f})")

            if weighted_score > 0:
                credibility_weighted_scores[diagnosis] = weighted_score
                agent_details = ", ".join(supporting_agents)
                print(f"   {diagnosis}: {weighted_score:.1f} pts [{agent_details}]")

        # Sort by credibility-weighted score
        sorted_diagnoses = sorted(
            credibility_weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # FIXED SELECTION SIZE (No Ground Truth Knowledge)
        provisional_list = []

        if sorted_diagnoses:
            # Fixed selection - always take top 6 diagnoses
            target_size = min(6, len(sorted_diagnoses))
            max_score = sorted_diagnoses[0][1] if sorted_diagnoses else 0

            print(f"\n🎯 Selection criteria: Top {target_size} by credibility-weighted score (fixed size)")

            for i, (diagnosis, score) in enumerate(sorted_diagnoses[:target_size]):
                provisional_list.append(diagnosis)
                percentage = (score / max_score) * 100 if max_score > 0 else 0
                rank_suffix = ["st", "nd", "rd"][i] if i < 3 else "th"
                print(f"   ✅ {i+1}{rank_suffix}: {diagnosis} ({score:.1f} pts, {percentage:.0f}% of max)")

            # Show what was excluded for transparency
            if len(sorted_diagnoses) > target_size:
                print(f"\n📋 Excluded diagnoses (rank {target_size+1}+):")
                for i, (diagnosis, score) in enumerate(sorted_diagnoses[target_size:], target_size+1):
                    percentage = (score / max_score) * 100 if max_score > 0 else 0
                    print(f"   ❌ {i}th: {diagnosis} ({score:.1f} pts, {percentage:.0f}% of max)")

        print(f"\n🗳️ Selected {len(provisional_list)} diagnoses via credibility-weighted preferential voting")
        return provisional_list

    def _normalize_diagnosis_for_synthesis(self, diagnosis: str) -> str:
        """Enhanced normalization for synthesis - handles fragmented outputs and variants"""
        import re

        if not isinstance(diagnosis, str):
            return str(diagnosis)

        # Remove numbered list prefixes (1., 2., etc.)
        cleaned = re.sub(r'^\s*\d+[\.\)\-\s]+', '', diagnosis)

        # Remove bullet points and dashes
        cleaned = re.sub(r'^\s*[-–•]\s*', '', cleaned)

        # CRITICAL: Cut off reasoning text that gets appended to diagnosis names
        sentence_endings = [
            r'\s+the patient\s+',
            r'\s+which is\s+',
            r'\s+this\s+',
            r'\s+based on\s+',
            r'\s+given\s+',
            r'\s+considering\s+',
            r'\s+evidence\s+',
            r'\s+with\s+elevated\s+',
            r'\s+the\s+elevated\s+',
            r'\.\s+',  # Period followed by space
        ]

        for pattern in sentence_endings:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[:match.start()].strip()
                break

        # Remove markdown bold formatting
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)

        # Remove trailing dashes and punctuation
        cleaned = re.sub(r'\s*-\s*$', '', cleaned)
        cleaned = re.sub(r'\s*\.\s*$', '', cleaned)

        # Remove leading/trailing quotes
        cleaned = cleaned.strip('"\'')

        # Remove extra whitespace and normalize
        cleaned = cleaned.strip()

        # ENHANCEMENT 1: Fix GENERAL fragmented compound terms that Gemma commonly produces
        # Pattern-based fixes that work across all medical specialties

        # Fix missing hyphens in compound medical terms (GENERAL PATTERN)
        # Pattern: "word1 induced word2" → "word1-induced word2"
        cleaned = re.sub(r'\b(\w+)\s+induced\s+(\w+)', r'\1-induced \2', cleaned, flags=re.IGNORECASE)

        # Pattern: "word1 related word2" → "word1-related word2"
        cleaned = re.sub(r'\b(\w+)\s+related\s+(\w+)', r'\1-related \2', cleaned, flags=re.IGNORECASE)

        # Pattern: "word1 associated word2" → "word1-associated word2"
        cleaned = re.sub(r'\b(\w+)\s+associated\s+(\w+)', r'\1-associated \2', cleaned, flags=re.IGNORECASE)

        # Fix compound terms with missing spaces/hyphens (GENERAL PATTERN)
        # Pattern: "wordsquashedtogether" → "word-squashed together" (if contains known medical compounds)
        medical_compounds = [
            (r'\b(\w+)induced(\w+)', r'\1-induced \2'),
            (r'\b(\w+)related(\w+)', r'\1-related \2'),
            (r'\b(\w+)associated(\w+)', r'\1-associated \2'),
            (r'\b(\w+)dependent(\w+)', r'\1-dependent \2'),
            (r'\b(\w+)mediated(\w+)', r'\1-mediated \2'),
        ]

        for pattern, replacement in medical_compounds:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # ENHANCEMENT 2: Handle standalone incomplete terms by checking context
        # (This will be used later in consolidation to merge incomplete terms)

        # ENHANCEMENT 3: Standardize hyphenation and spacing
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', cleaned)  # Fix hyphenation

        # ENHANCEMENT 4: Apply comprehensive abbreviation mapping
        abbreviation_expansions = {
            'aki': 'acute kidney injury',
            'arf': 'acute renal failure',
            'cin': 'contrast-induced nephropathy',
            'ttp': 'thrombotic thrombocytopenic purpura',
            'tma': 'thrombotic microangiopathy',
            'mi': 'myocardial infarction',
            'chf': 'congestive heart failure',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'uti': 'urinary tract infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'tia': 'transient ischemic attack',
            'cva': 'cerebrovascular accident',
            'gerd': 'gastroesophageal reflux disease',
            'ibd': 'inflammatory bowel disease',
            'ra': 'rheumatoid arthritis',
            'sle': 'systemic lupus erythematosus',
            'ms': 'multiple sclerosis',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'afib': 'atrial fibrillation',
            'af': 'atrial fibrillation',
            'vtach': 'ventricular tachycardia',
            'vt': 'ventricular tachycardia',
            'svt': 'supraventricular tachycardia',
            'ards': 'acute respiratory distress syndrome',
            'dic': 'disseminated intravascular coagulation',
            'hus': 'hemolytic uremic syndrome',
            'sirs': 'systemic inflammatory response syndrome',
            'aids': 'acquired immunodeficiency syndrome',
            'hiv': 'human immunodeficiency virus',

            # NEW ADDITIONS - Crystal arthropathies
            'cppd': 'pseudogout',
            'oa': 'osteoarthritis',

            # NEW ADDITIONS - Pulmonary infections
            'pcp': 'pneumocystis jirovecii pneumonia',
            'pjp': 'pneumocystis jirovecii pneumonia',
            'abpa': 'allergic bronchopulmonary aspergillosis',
            'histo': 'histoplasmosis',
            'cocci': 'coccidioidomycosis',
            'tb': 'tuberculosis',
        }

        # Apply abbreviation mapping (but preserve compound terms)
        cleaned_lower = cleaned.lower()
        if cleaned_lower in abbreviation_expansions:
            cleaned = abbreviation_expansions[cleaned_lower]

        # Convert to title case for consistency
        if cleaned:
            # Use title case but preserve medical abbreviations
            words = cleaned.split()
            titled_words = []
            for word in words:
                if word.upper() in ['AKI', 'CIN', 'TTP', 'TMA', 'MI', 'CHF', 'DM', 'HTN', 'UTI', 'DVT', 'PE', 'TIA', 'CVA', 'GERD', 'IBD', 'RA', 'SLE', 'MS', 'CKD', 'ESRD', 'AF', 'VT', 'SVT', 'ARDS', 'DIC', 'HUS', 'SIRS', 'AIDS', 'HIV']:
                    titled_words.append(word.upper())
                else:
                    titled_words.append(word.title())
            cleaned = ' '.join(titled_words)

        return cleaned

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

        if refinement_tempo >= 1.5:
            print(f"🔥 High tempo ({refinement_tempo:.1f}) - applying consolidation")
            final_list = self._consolidate_similar_diagnoses(provisional_list)
            consolidation_applied = True
        else:
            print(f"📝 Standard tempo ({refinement_tempo:.1f}) - minimal consolidation")
            final_list = list(set(provisional_list))  # Simple deduplication

        return final_list, consolidation_applied

    def _consolidate_similar_diagnoses(self, provisional_list: List[str]) -> List[str]:
        """Enhanced consolidation - handles hierarchical relationships and incomplete terms across all medical specialties"""
        final_list = []
        processed = set()

        # STEP 1: Apply normalization to all diagnoses first
        normalized_diagnoses = []
        for diagnosis in provisional_list:
            normalized = self._normalize_diagnosis_for_synthesis(diagnosis)
            normalized_diagnoses.append((diagnosis, normalized))

        print(f"   🔍 Consolidating {len(normalized_diagnoses)} diagnoses...")

        # STEP 2: Process each diagnosis for consolidation
        for i, (original_diagnosis, normalized_current) in enumerate(normalized_diagnoses):
            if original_diagnosis in processed:
                continue

            # Find all similar/related diagnoses to consolidate with this one
            consolidation_group = [original_diagnosis]

            for j, (other_original, normalized_other) in enumerate(normalized_diagnoses):
                if i != j and other_original not in processed:

                    # Check if these should be consolidated
                    should_consolidate = self._should_consolidate_enhanced(
                        normalized_current, normalized_other,
                        original_diagnosis, other_original
                    )

                    if should_consolidate:
                        consolidation_group.append(other_original)
                        processed.add(other_original)

            # STEP 3: Select the best representative from the consolidation group
            if len(consolidation_group) > 1:
                best_diagnosis = self._select_best_from_group(consolidation_group)
                final_list.append(best_diagnosis)
                processed.add(original_diagnosis)

                # Log the consolidation for debugging
                group_names = [d[:30] + ('...' if len(d) > 30 else '') for d in consolidation_group]
                print(f"   🔄 Consolidated group: {group_names} → '{best_diagnosis[:50]}{'...' if len(best_diagnosis) > 50 else ''}'")

            elif original_diagnosis not in processed:
                # No consolidation needed - add as-is
                final_list.append(original_diagnosis)
                processed.add(original_diagnosis)

        print(f"   ✅ Consolidation complete: {len(provisional_list)} → {len(final_list)} diagnoses")
        return final_list

    def _should_consolidate(self, diag1: str, diag2: str) -> bool:
        """Determine if two diagnoses should be consolidated - GENERALIZED FOR ALL MEDICAL CONDITIONS"""

        # Normalize for comparison
        d1_norm = self._normalize_diagnosis(diag1)
        d2_norm = self._normalize_diagnosis(diag2)

        # Exact match after normalization
        if d1_norm == d2_norm:
            return True

        # Calculate similarity scores
        similarity_score = self._calculate_diagnosis_similarity(d1_norm, d2_norm)

        # Consolidate if similarity is high enough
        return similarity_score >= 0.85  # 85% similarity threshold

    def _calculate_diagnosis_similarity(self, diag1: str, diag2: str) -> float:
        """Calculate similarity between two diagnoses using multiple metrics"""
        import re
        from difflib import SequenceMatcher

        # Method 1: Token overlap similarity
        tokens1 = set(diag1.split())
        tokens2 = set(diag2.split())

        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        token_similarity = intersection / union if union > 0 else 0.0

        # Method 2: String similarity (handles typos and variations)
        string_similarity = SequenceMatcher(None, diag1, diag2).ratio()

        # Method 3: Medical-specific similarity
        medical_similarity = self._calculate_medical_similarity(diag1, diag2)

        # Weighted combination
        final_similarity = (
            token_similarity * 0.5 +      # 50% weight on shared words
            string_similarity * 0.3 +     # 30% weight on string similarity
            medical_similarity * 0.2      # 20% weight on medical patterns
        )

        return final_similarity

    def _calculate_medical_similarity(self, diag1: str, diag2: str) -> float:
        """Calculate medical-specific similarity patterns"""

        # Check for common medical abbreviation patterns
        if self._are_abbreviation_variants(diag1, diag2):
            return 1.0

        # Check for syndrome/disease variations
        if self._are_medical_variants(diag1, diag2):
            return 0.9

        # Check for anatomical variations (e.g., "renal" vs "kidney")
        if self._are_anatomical_variants(diag1, diag2):
            return 0.8

        return 0.0

    def _are_abbreviation_variants(self, diag1: str, diag2: str) -> bool:
        """Check if diagnoses are abbreviation variants of each other"""

        # Common medical abbreviation patterns
        abbreviation_map = {
            'acute kidney injury': ['aki', 'acute renal failure', 'arf'],
            'myocardial infarction': ['mi', 'heart attack'],
            'congestive heart failure': ['chf', 'heart failure'],
            'chronic obstructive pulmonary disease': ['copd'],
            'diabetes mellitus': ['dm', 'diabetes'],
            'hypertension': ['htn', 'high blood pressure'],
            'pneumonia': ['pna'],
            'urinary tract infection': ['uti'],
            'deep vein thrombosis': ['dvt'],
            'pulmonary embolism': ['pe'],
            'transient ischemic attack': ['tia'],
            'cerebrovascular accident': ['cva', 'stroke'],
            'gastroesophageal reflux disease': ['gerd'],
            'inflammatory bowel disease': ['ibd'],
            'rheumatoid arthritis': ['ra'],
            'systemic lupus erythematosus': ['sle', 'lupus'],
            'multiple sclerosis': ['ms'],
            'chronic kidney disease': ['ckd'],
            'end stage renal disease': ['esrd'],
            'acute coronary syndrome': ['acs'],
            'atrial fibrillation': ['afib', 'af'],
            'ventricular tachycardia': ['vtach', 'vt'],
            'supraventricular tachycardia': ['svt'],
            'acute respiratory distress syndrome': ['ards'],
            'disseminated intravascular coagulation': ['dic'],
            'thrombotic thrombocytopenic purpura': ['ttp'],
            'hemolytic uremic syndrome': ['hus'],
            'systemic inflammatory response syndrome': ['sirs'],
            'acquired immunodeficiency syndrome': ['aids'],
            'human immunodeficiency virus': ['hiv'],
        }

        # Check if either diagnosis is an abbreviation of the other
        for full_term, abbreviations in abbreviation_map.items():
            full_normalized = self._normalize_diagnosis(full_term)
            if ((diag1 == full_normalized and diag2 in abbreviations) or
                (diag2 == full_normalized and diag1 in abbreviations) or
                (diag1 in abbreviations and diag2 in abbreviations)):
                return True

        return False

    def _are_medical_variants(self, diag1: str, diag2: str) -> bool:
        """Check for medical terminology variants"""

        # Common medical term substitutions
        medical_variants = [
            (['renal', 'kidney'], 'kidney-related'),
            (['cardiac', 'heart'], 'heart-related'),
            (['pulmonary', 'lung'], 'lung-related'),
            (['hepatic', 'liver'], 'liver-related'),
            (['cerebral', 'brain'], 'brain-related'),
            (['gastric', 'stomach'], 'stomach-related'),
            (['cutaneous', 'skin'], 'skin-related'),
            (['ocular', 'eye'], 'eye-related'),
            (['osseous', 'bone'], 'bone-related'),
            (['vascular', 'vessel'], 'vessel-related'),
        ]

        # Check if diagnoses differ only by medical term variants
        for variants, _ in medical_variants:
            for i, variant1 in enumerate(variants):
                for j, variant2 in enumerate(variants):
                    if i != j:
                        if (variant1 in diag1 and variant2 in diag2 and
                            diag1.replace(variant1, variant2) == diag2):
                            return True

        return False

    def _are_anatomical_variants(self, diag1: str, diag2: str) -> bool:
        """Check for anatomical terminology variants"""

        anatomical_synonyms = {
            'renal': 'kidney',
            'cardiac': 'heart',
            'pulmonary': 'lung',
            'hepatic': 'liver',
            'cerebral': 'brain',
            'gastric': 'stomach',
            'cutaneous': 'skin',
            'ocular': 'eye',
            'osseous': 'bone',
            'vascular': 'blood vessel',
            'neural': 'nerve',
            'muscular': 'muscle',
        }

        # Create reverse mapping
        reverse_synonyms = {v: k for k, v in anatomical_synonyms.items()}
        all_synonyms = {**anatomical_synonyms, **reverse_synonyms}

        # Check if replacing anatomical terms makes diagnoses equivalent
        for medical_term, common_term in all_synonyms.items():
            if (medical_term in diag1 and common_term in diag2 and
                diag1.replace(medical_term, common_term) == diag2):
                return True
            if (common_term in diag1 and medical_term in diag2 and
                diag1.replace(common_term, medical_term) == diag2):
                return True

        return False

    def _normalize_diagnosis(self, diagnosis: str) -> str:
        """Enhanced normalization for all medical diagnoses"""
        import re

        # Convert to lowercase and remove extra whitespace
        normalized = diagnosis.lower().strip()

        # Remove common prefixes/suffixes that don't affect meaning
        prefixes_to_remove = [
            'acute', 'chronic', 'primary', 'secondary', 'idiopathic',
            'essential', 'benign', 'malignant', 'recurrent', 'progressive',
            'mild', 'moderate', 'severe', 'early', 'late', 'stage'
        ]

        suffixes_to_remove = [
            'syndrome', 'disease', 'disorder', 'condition', 'injury',
            'failure', 'insufficiency', 'deficiency', 'dysfunction',
            'inflammation', 'infection', 'neoplasm', 'tumor'
        ]

        # Remove prefixes
        for prefix in prefixes_to_remove:
            pattern = rf'^{prefix}\s+'
            normalized = re.sub(pattern, '', normalized)

        # Remove suffixes (but keep core meaning)
        for suffix in suffixes_to_remove:
            pattern = rf'\s+{suffix}$'
            normalized = re.sub(pattern, '', normalized)

        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)

        # Handle common abbreviations and expansions (expanded list)
        abbreviation_expansions = {
            'aki': 'acute kidney injury',
            'arf': 'acute renal failure',
            'ttp': 'thrombotic thrombocytopenic purpura',
            'tma': 'thrombotic microangiopathy',
            'mi': 'myocardial infarction',
            'acs': 'acute coronary syndrome',
            'hit': 'heparin induced thrombocytopenia',
            'cap': 'community acquired pneumonia',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'uti': 'urinary tract infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'tia': 'transient ischemic attack',
            'cva': 'cerebrovascular accident',
            'gerd': 'gastroesophageal reflux disease',
            'ibd': 'inflammatory bowel disease',
            'ra': 'rheumatoid arthritis',
            'sle': 'systemic lupus erythematosus',
            'ms': 'multiple sclerosis',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'afib': 'atrial fibrillation',
            'af': 'atrial fibrillation',
            'vtach': 'ventricular tachycardia',
            'vt': 'ventricular tachycardia',
            'svt': 'supraventricular tachycardia',
            'ards': 'acute respiratory distress syndrome',
            'dic': 'disseminated intravascular coagulation',
            'hus': 'hemolytic uremic syndrome',
            'sirs': 'systemic inflammatory response syndrome',
            'aids': 'acquired immunodeficiency syndrome',
            'hiv': 'human immunodeficiency virus',
        }

        # Apply abbreviation mapping
        if normalized in abbreviation_expansions:
            normalized = abbreviation_expansions[normalized]

        return normalized.strip()

    def _generate_selection_reasoning(self, master_list: List[str],
                                provisional_list: List[str],
                                final_list: List[str]) -> Dict[str, str]:
        """Generate reasoning for selection decisions"""
        return {
            'master_list_size': f"{len(master_list)} total diagnoses from specialists",
            'provisional_selection': f"{len(provisional_list)} diagnoses selected based on credibility weighting",
            'final_selection': f"{len(final_list)} diagnoses after consolidation",
            'selection_criteria': "Credibility-weighted preferential voting with fixed selection size"
        }

    def _should_consolidate_enhanced(self, diag1_norm: str, diag2_norm: str,
                               diag1_orig: str, diag2_orig: str) -> bool:
        """Enhanced consolidation logic - generalized for all medical specialties"""

        # Rule 1: Exact match after normalization
        if diag1_norm.lower() == diag2_norm.lower():
            return True

        # Rule 2: Hierarchical relationship (specific subsumes general)
        if self._is_hierarchical_relationship(diag1_norm, diag2_norm):
            return True

        # Rule 3: Incomplete term absorption (e.g., "drug" absorbed by "drug-induced vasculitis")
        if self._is_incomplete_term_relationship(diag1_norm, diag2_norm):
            return True

        # Rule 4: Semantic similarity for medical variants
        similarity_score = self._calculate_medical_similarity_enhanced(diag1_norm, diag2_norm)
        if similarity_score >= 0.85:  # 85% similarity threshold
            return True

        return False

    def _is_hierarchical_relationship(self, diag1: str, diag2: str) -> bool:
        """Check if one diagnosis is a specific subtype of another (GENERALIZED)"""

        d1_lower = diag1.lower()
        d2_lower = diag2.lower()

        # Pattern 1: One diagnosis contains the other as a substring (specific contains general)
        # Example: "drug-induced vasculitis" contains "vasculitis"
        if d1_lower in d2_lower or d2_lower in d1_lower:
            return True

        # Pattern 2: Shared core medical term with one being more specific
        # Extract core medical terms (last significant word)
        d1_core = self._extract_core_medical_term(d1_lower)
        d2_core = self._extract_core_medical_term(d2_lower)

        if d1_core == d2_core and d1_core:
            # Same core term - check if one is more specific
            d1_specificity = self._calculate_diagnostic_specificity(d1_lower)
            d2_specificity = self._calculate_diagnostic_specificity(d2_lower)

            # If one is significantly more specific, they should consolidate
            if abs(d1_specificity - d2_specificity) >= 2:
                return True

        # Pattern 3: Common medical hierarchical patterns (GENERALIZED)
        hierarchical_patterns = [
            # Format: (general_term, specific_indicators)
            ('injury', ['acute', 'chronic', 'drug-induced', 'contrast-induced', 'ischemic', 'traumatic']),
            ('nephropathy', ['diabetic', 'contrast-induced', 'drug-induced', 'hypertensive']),
            ('vasculitis', ['drug-induced', 'systemic', 'necrotizing', 'hypersensitivity']),
            ('pneumonia', ['community-acquired', 'hospital-acquired', 'aspiration', 'viral', 'bacterial']),
            ('cardiomyopathy', ['dilated', 'hypertrophic', 'restrictive', 'ischemic', 'drug-induced']),
            ('arthritis', ['rheumatoid', 'psoriatic', 'septic', 'reactive', 'osteo']),
            ('anemia', ['iron-deficiency', 'sickle-cell', 'aplastic', 'hemolytic']),
            ('diabetes', ['type-1', 'type-2', 'gestational', 'drug-induced']),
            ('hypertension', ['essential', 'secondary', 'pulmonary', 'portal']),
            ('failure', ['heart', 'kidney', 'liver', 'respiratory', 'acute', 'chronic']),
        ]

        for general_term, specific_indicators in hierarchical_patterns:
            # Check if both diagnoses relate to this medical category
            d1_has_general = general_term in d1_lower
            d2_has_general = general_term in d2_lower

            if d1_has_general and d2_has_general:
                # Check specificity patterns
                d1_has_specific = any(indicator in d1_lower for indicator in specific_indicators)
                d2_has_specific = any(indicator in d2_lower for indicator in specific_indicators)

                if d1_has_specific != d2_has_specific:  # One is specific, one is general
                    return True

        return False

    def _is_incomplete_term_relationship(self, diag1: str, diag2: str) -> bool:
        """Check if one diagnosis is an incomplete fragment of another"""

        d1_lower = diag1.lower().strip()
        d2_lower = diag2.lower().strip()

        # Rule 1: Single word absorption
        # Example: "drug" should be absorbed by "drug-induced vasculitis"
        if (len(d1_lower.split()) == 1 and len(d2_lower.split()) > 1 and d1_lower in d2_lower):
            return True
        if (len(d2_lower.split()) == 1 and len(d1_lower.split()) > 1 and d2_lower in d1_lower):
            return True

        # Rule 2: Incomplete compound term absorption
        # Example: "kidney injury" should be absorbed by "acute kidney injury"
        d1_words = set(d1_lower.split())
        d2_words = set(d2_lower.split())

        # If one diagnosis's words are completely contained in the other
        if d1_words.issubset(d2_words) and len(d1_words) < len(d2_words):
            return True
        if d2_words.issubset(d1_words) and len(d2_words) < len(d1_words):
            return True

        # Rule 3: Medical abbreviation fragments
        # Example: "aki" should be absorbed by "acute kidney injury"
        medical_abbreviations = {
            'aki': ['acute', 'kidney', 'injury'],
            'mi': ['myocardial', 'infarction'],
            'chf': ['congestive', 'heart', 'failure'],
            'copd': ['chronic', 'obstructive', 'pulmonary', 'disease'],
            'uti': ['urinary', 'tract', 'infection'],
            'dvt': ['deep', 'vein', 'thrombosis'],
            'pe': ['pulmonary', 'embolism'],
            'tia': ['transient', 'ischemic', 'attack'],
        }

        for abbrev, full_words in medical_abbreviations.items():
            if ((d1_lower == abbrev and all(word in d2_lower for word in full_words)) or
                (d2_lower == abbrev and all(word in d1_lower for word in full_words))):
                return True

        return False

    def _extract_core_medical_term(self, diagnosis: str) -> str:
        """Extract the core medical term from a diagnosis (typically the last significant medical word)"""

        # Common medical suffixes that indicate core conditions
        medical_suffixes = [
            'itis', 'osis', 'pathy', 'trophy', 'plasia', 'genic', 'lysis',
            'megaly', 'scopy', 'tomy', 'ectomy', 'plasty', 'rrhea', 'algia',
            'syndrome', 'disease', 'disorder', 'injury', 'failure', 'dysfunction'
        ]

        words = diagnosis.split()

        # Look for words with medical suffixes
        for word in reversed(words):  # Start from the end
            if any(word.endswith(suffix) for suffix in medical_suffixes):
                return word

        # If no medical suffix found, return the last significant word (not articles/prepositions)
        stop_words = {'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'the', 'a', 'an'}
        for word in reversed(words):
            if word.lower() not in stop_words and len(word) > 2:
                return word

        return ''

    def _calculate_diagnostic_specificity(self, diagnosis: str) -> int:
        """Calculate how specific a diagnosis is (higher number = more specific)"""

        specificity_score = 0

        # More words generally indicate higher specificity
        specificity_score += len(diagnosis.split())

        # Specific medical modifiers increase specificity
        specific_modifiers = [
            'acute', 'chronic', 'drug-induced', 'contrast-induced', 'idiopathic',
            'primary', 'secondary', 'autoimmune', 'infectious', 'ischemic',
            'hemorrhagic', 'thrombotic', 'embolic', 'inflammatory', 'degenerative'
        ]

        for modifier in specific_modifiers:
            if modifier in diagnosis:
                specificity_score += 2

        # Anatomical specificity
        anatomical_terms = [
            'renal', 'cardiac', 'pulmonary', 'hepatic', 'cerebral', 'gastric',
            'cutaneous', 'ocular', 'osseous', 'vascular', 'neural', 'muscular'
        ]

        for term in anatomical_terms:
            if term in diagnosis:
                specificity_score += 1

        return specificity_score

    def _calculate_medical_similarity_enhanced(self, diag1: str, diag2: str) -> float:
        """Enhanced medical similarity calculation for all specialties"""
        from difflib import SequenceMatcher

        # Method 1: Token overlap (medical terms)
        tokens1 = set(diag1.lower().split())
        tokens2 = set(diag2.lower().split())

        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        token_similarity = intersection / union if union > 0 else 0.0

        # Method 2: String similarity (handles typos/variants)
        string_similarity = SequenceMatcher(None, diag1.lower(), diag2.lower()).ratio()

        # Method 3: Medical core term similarity
        core1 = self._extract_core_medical_term(diag1)
        core2 = self._extract_core_medical_term(diag2)
        core_similarity = 1.0 if core1 == core2 and core1 else 0.0

        # Weighted combination
        final_similarity = (
            token_similarity * 0.5 +      # 50% weight on shared medical terms
            string_similarity * 0.3 +     # 30% weight on string similarity
            core_similarity * 0.2         # 20% weight on core medical term match
        )

        return final_similarity

    def _select_best_from_group(self, consolidation_group: List[str]) -> str:
        """Select the best representative diagnosis from a consolidation group"""

        if len(consolidation_group) == 1:
            return consolidation_group[0]

        # Scoring criteria for selecting the best diagnosis
        best_diagnosis = consolidation_group[0]
        best_score = 0

        for diagnosis in consolidation_group:
            score = 0
            diagnosis_lower = diagnosis.lower()

            # Prefer more specific diagnoses (higher word count)
            score += len(diagnosis.split())

            # Prefer diagnoses with medical modifiers
            specific_modifiers = ['acute', 'chronic', 'drug-induced', 'contrast-induced', 'primary', 'secondary']
            score += sum(2 for modifier in specific_modifiers if modifier in diagnosis_lower)

            # Prefer properly formatted diagnoses (with hyphens)
            if '-' in diagnosis:
                score += 3

            # Prefer complete medical terms over abbreviations
            if not any(word.isupper() for word in diagnosis.split()):
                score += 2

            # Penalize very short or incomplete terms
            if len(diagnosis.split()) < 2:
                score -= 5

            # Prefer longer, more descriptive diagnoses
            score += len(diagnosis) * 0.1

            if score > best_score:
                best_score = score
                best_diagnosis = diagnosis

        return best_diagnosis

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
        # In the run_complete_analysis method, update this line:
        synthesis_result = self.synthesizer.synthesize_final_diagnosis(
            round_results, reed_results  # Remove ground_truth parameter
        )

        # Step 3: Evaluation (if ground truth provided)
        evaluation_result = None
        if ground_truth:
            # Import evaluation module when needed
            try:
                from ddx_evaluator_v9 import EnhancedClinicalEvaluator

                evaluator = EnhancedClinicalEvaluator(self.ddx_system.model_manager)

                # CONNECT THE EVALUATOR AGENT TO THE EQUIVALENCE AGENT
                evaluator_agent = None
                for agent in self.ddx_system.current_agents:
                    if "Clinical Evaluator" in agent.name or "Evaluator" in agent.name:
                        evaluator_agent = agent
                        break

                if evaluator_agent:
                    evaluator.equivalence_agent.set_evaluator_agent(evaluator_agent)
                    print(f"✅ Connected evaluator: {evaluator_agent.name}")
                else:
                    print("⚠️ No evaluator agent found in team")

                # Set transcript for context-aware evaluation
                # Get transcript for v8 evaluator
                full_transcript = None
                try:
                    orchestrator = getattr(self.ddx_system, 'round_orchestrator', None)
                    if (orchestrator is not None and
                        hasattr(orchestrator, 'full_transcript')):
                        full_transcript = orchestrator.full_transcript
                        print(f"📋 Transcript available for v8 evaluator context analysis")
                except Exception as e:
                    print(f"⚠️ Could not get transcript: {e}")

                evaluation_result = evaluator.evaluate_diagnosis(
                    synthesis_result,
                    ground_truth,
                    round_results,
                    full_transcript  # Pass transcript directly to v8 evaluator
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

        # Clean up models after evaluation
        if hasattr(ddx, 'agent_generator'):
            ddx.agent_generator.cleanup_models_after_evaluation()

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
