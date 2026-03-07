
# =============================================================================
# DDx Evaluation v6 - AI-Enhanced Clinical Evaluation with v6 Integration
# =============================================================================

"""
DDx Evaluation v6: Clean rebuild that integrates with v6 architecture while
preserving all sophisticated AI-enhanced evaluation features
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Import v6 core components
from ddx_core_v6 import DDxAgent, AgentResponse, ModelManager
from ddx_rounds_v6 import RoundType, RoundResult
from vllm import SamplingParams

# =============================================================================
# Enhanced Data Structures for v6
# =============================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result with v6 enhancements"""
    # Core metrics
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    caa_count: int = 0
    ae_count: int = 0
    tm_sm_count: int = 0  # v6 addition for symptom management captures

    # Matched data
    matched_diagnoses: List[Tuple[str, str]] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    clinical_alternatives: List[str] = field(default_factory=list)
    appropriately_excluded: List[str] = field(default_factory=list)
    symptom_management_captures: List[str] = field(default_factory=list)

    # Performance metrics
    traditional_precision: float = 0.0
    traditional_recall: float = 0.0
    clinical_reasoning_quality: float = 0.0
    diagnostic_safety: float = 0.0
    reasoning_thoroughness: float = 0.0
    system_safety_coverage: float = 0.0  # v6 addition
    tp_rate: float = 0.0
    caa_weight_applied: float = 0.0

    # Enhanced analysis
    reasoning_analysis: Dict[str, Any] = field(default_factory=dict)
    context_matches: List[Dict[str, str]] = field(default_factory=list)
    symptom_management_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EquivalenceResult:
    """Result of clinical equivalence evaluation"""
    is_equivalent: bool
    confidence: float
    reasoning: str
    context_evidence: Optional[str] = None
    alternative_terms_found: List[str] = field(default_factory=list)

# =============================================================================
# Clinical Equivalence Agent (Enhanced for v6)
# =============================================================================

class ClinicalEquivalenceAgent:
    """Clinical evaluator that uses pre-generated evaluator agent"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.evaluation_cache = {}
        self.evaluator_agent = None
        print("✅ Clinical Equivalence Agent initialized for pre-generated evaluator")

    def set_evaluator_agent(self, evaluator_agent):
        """Set the pre-generated evaluator agent"""
        self.evaluator_agent = evaluator_agent
        print(f"✅ Using pre-generated evaluator: {evaluator_agent.name}")

    def evaluate_diagnosis_match(self, system_diagnosis: str, ground_truth_diagnosis: str,
                              transcript: Optional[Dict] = None) -> EquivalenceResult:
        """Evaluate using the pre-generated evaluator agent"""
        
        if not self.evaluator_agent:
            # Fallback only
            is_equivalent = system_diagnosis.lower().strip() == ground_truth_diagnosis.lower().strip()
            return EquivalenceResult(
                is_equivalent=is_equivalent,
                confidence=1.0 if is_equivalent else 0.0,
                reasoning="Fallback exact match (no evaluator agent)",
                context_evidence=None
            )

        # Use the pre-generated agent - NO MODEL LOADING!
        prompt = self._create_equivalence_prompt(system_diagnosis, ground_truth_diagnosis)
        response = self.evaluator_agent.generate_response(prompt, "clinical_evaluation")
        
        return self._parse_equivalence_response(response.content, system_diagnosis, ground_truth_diagnosis)

    def _extract_transcript_context(self, diagnosis: str, transcript: Dict) -> Optional[str]:
        """Extract relevant context from v6 transcript for diagnosis"""
        if not transcript or 'rounds' not in transcript:
            return None

        diagnosis_lower = diagnosis.lower()
        context_snippets = []

        # Search through v6 transcript structure
        for round_name, round_data in transcript['rounds'].items():
            if 'responses' not in round_data:
                continue

            for agent_name, response_data in round_data['responses'].items():
                content = response_data.get('content', '').lower()

                # Look for diagnosis mentions or related terms
                if any(term in content for term in [
                    diagnosis_lower,
                    diagnosis_lower.replace(' ', ''),
                    diagnosis_lower.replace('disease', '').strip(),
                    diagnosis_lower.replace('syndrome', '').strip()
                ]):
                    # Extract surrounding context (up to 200 chars)
                    snippet = self._extract_snippet_around_term(content, diagnosis_lower)
                    if snippet:
                        context_snippets.append(f"{agent_name}: {snippet}")

        return " | ".join(context_snippets[:3]) if context_snippets else None

    def _extract_snippet_around_term(self, text: str, term: str, window: int = 100) -> str:
        """Extract a snippet of text around a term"""
        pos = text.find(term)
        if pos == -1:
            return ""

        start = max(0, pos - window)
        end = min(len(text), pos + len(term) + window)

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _create_equivalence_prompt(self, sys_diag: str, gt_diag: str,
                                 context: Optional[str] = None) -> str:
        """Create prompt for clinical equivalence evaluation"""
        context_section = ""
        if context:
            context_section = f"""
TRANSCRIPT CONTEXT:
The following discussion occurred during the diagnostic process:
{context}
"""

        return f"""You are a clinical evaluation expert. Assess whether these two medical diagnoses are clinically equivalent or represent the same medical condition.

Consider these as EQUIVALENT:
- Exact matches
- Medical abbreviations vs. full terms (e.g., "COPD" = "chronic obstructive pulmonary disease")
- Synonymous medical terms (e.g., "myocardial infarction" = "heart attack")
- Specificity variations (e.g., "diabetes" = "diabetes mellitus", "pneumonia" = "bacterial pneumonia")
- Alternative medical terminology for the same condition
- Different phrasing of the same underlying pathology
- Formal vs. colloquial medical terms
- ICD-10 vs. clinical descriptive terms

Consider these as DIFFERENT:
- Different organ systems or anatomical locations
- Different underlying pathophysiology or disease mechanisms  
- Different stages or severity levels of disease
- Complications vs. primary conditions
- Acute vs. chronic versions (unless clearly the same condition)
- Different etiologies of similar presentations

If transcript context shows the team discussed the ground truth concept using different terminology, consider this as evidence for equivalence.

Respond exactly in this format:
EQUIVALENT: YES/NO
CONFIDENCE: [0.0-1.0]
REASONING: [Brief clinical justification citing any transcript evidence]

System Diagnosis: "{sys_diag}"
Ground Truth Diagnosis: "{gt_diag}"
{context_section}
Are these clinically equivalent?"""

    def _parse_equivalence_response(self, content: str, sys_diag: str, gt_diag: str,
                                  context: Optional[str] = None) -> EquivalenceResult:
        """Parse the agent's equivalence evaluation response"""
        is_equivalent = False
        confidence = 0.0
        reasoning = "No reasoning provided"

        try:
            # Parse response lines
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('EQUIVALENT:'):
                    equivalent_text = line.split(':', 1)[1].strip().upper()
                    is_equivalent = equivalent_text == 'YES'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.8 if is_equivalent else 0.2
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()

        except Exception as e:
            print(f"⚠️ Failed to parse equivalence response: {e}")
            # Fallback to simple string matching
            is_equivalent = sys_diag.lower().strip() == gt_diag.lower().strip()
            confidence = 1.0 if is_equivalent else 0.0
            reasoning = "Fallback exact match due to parsing error"

        return EquivalenceResult(
            is_equivalent=is_equivalent,
            confidence=confidence,
            reasoning=reasoning,
            context_evidence=context,
            alternative_terms_found=[]
        )

# =============================================================================
# Enhanced Clinical Evaluator v6
# =============================================================================

class EnhancedClinicalEvaluator:
    """Enhanced evaluator with v6 integration and comprehensive analysis"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.equivalence_agent = ClinicalEquivalenceAgent(model_manager)
        self.transcript = None

    def set_transcript(self, transcript: Dict):
        """Set the diagnostic transcript for context-aware evaluation"""
        self.transcript = transcript

    def evaluate_diagnosis(self, synthesis_result, ground_truth: Dict[str, List[str]],
                          round_results: Optional[Dict] = None) -> EvaluationResult:
        """Comprehensive evaluation with v6 enhancements"""

        print(f"\n📊 CLINICAL EVALUATION v6 (AI-Enhanced)")
        print("=" * 50)

        system_diagnoses = set(synthesis_result.final_list)
        gt_diagnoses = set(ground_truth.keys())

        print(f"🤖 System diagnoses: {len(system_diagnoses)}")
        print(f"🎯 Ground truth: {len(gt_diagnoses)}")
        print(f"🧠 Using Clinical Equivalence Agent for matching...")
        if self.transcript:
            print(f"📋 Transcript available for context analysis")

        # Step 1: AI-Enhanced Matching
        matched_pairs = []
        matched_system = set()
        matched_gt = set()
        context_matches = []

        for sys_diag in system_diagnoses:
            best_match = None
            best_result = None

            for gt_diag in gt_diagnoses:
                if gt_diag in matched_gt:
                    continue  # Already matched

                # Use AI agent to evaluate equivalence
                print(f"   🔍 Evaluating: '{sys_diag}' vs '{gt_diag}'")
                eval_result = self.equivalence_agent.evaluate_diagnosis_match(
                    sys_diag, gt_diag, self.transcript
                )

                if eval_result.is_equivalent:
                    best_match = gt_diag
                    best_result = eval_result
                    print(f"      ✅ MATCH ({eval_result.confidence:.2f}): {eval_result.reasoning}")
                    break
                else:
                    print(f"      ❌ Different ({eval_result.confidence:.2f}): {eval_result.reasoning}")

            if best_match and best_result:  # ✅ FIXED: Ensure both exist
                  matched_pairs.append((best_match, sys_diag))
                  matched_system.add(sys_diag)
                  matched_gt.add(best_match)

                  # Store context information
                  if best_result.context_evidence:
                      context_matches.append({
                          'system': sys_diag,
                          'ground_truth': best_match,
                          'context': best_result.context_evidence
                      })

        # Step 2: Can't Miss Analysis with Frequency-Based Promotion
        caa_diagnoses = []
        cant_miss_matches = []

        if round_results:
            cant_miss_diagnoses = self._extract_cant_miss_diagnoses(round_results)

            if cant_miss_diagnoses:
                print(f"\n🚨 Checking Can't Miss diagnoses for promotion/CAA classification:")
                
                # Count frequency of each diagnosis (normalize names for proper aggregation)
                cant_miss_frequency = {}
                name_mapping = {}
                
                for cant_miss_diag in cant_miss_diagnoses:
                    diag_name = cant_miss_diag['diagnosis']
                    # Normalize: remove asterisks, extra spaces, standardize case
                    normalized_name = diag_name.strip('*').strip().lower()
                    cant_miss_frequency[normalized_name] = cant_miss_frequency.get(normalized_name, 0) + 1
                    
                    # Keep mapping to original name for display
                    if normalized_name not in name_mapping:
                        name_mapping[normalized_name] = diag_name

                # Process each unique diagnosis (avoid duplicates)
                processed_diagnoses = set()
                
                for cant_miss_diag in cant_miss_diagnoses:
                    diag_name = cant_miss_diag['diagnosis']
                    normalized_name = diag_name.strip('*').strip().lower()
                    specialist = cant_miss_diag.get('specialist', 'Unknown')
                    frequency = cant_miss_frequency[normalized_name]
                    
                    # Skip if we've already processed this normalized diagnosis
                    if normalized_name in processed_diagnoses:
                        continue
                    processed_diagnoses.add(normalized_name)

                    print(f"   🔍 Can't Miss: '{diag_name}' by {specialist} (mentioned {frequency} times)")

                    # Check if this Can't Miss diagnosis matches any unmatched ground truth
                    for gt_diag in gt_diagnoses:
                        if gt_diag in matched_gt:
                            continue  # Already matched as TP

                        # Use AI agent to check equivalence
                        eval_result = self.equivalence_agent.evaluate_diagnosis_match(
                            diag_name, gt_diag, self.transcript
                        )

                        if eval_result.is_equivalent:
                            # ✅ FIXED: Check frequency for promotion to TP
                            if frequency >= 3:  # Threshold: 3+ mentions → promote to TP
                                matched_pairs.append((gt_diag, diag_name))
                                matched_system.add(diag_name)
                                matched_gt.add(gt_diag)
                                print(f"      🎯 PROMOTED TO TP: {diag_name} ↔ {gt_diag} (safety net capture)")
                                print(f"         Frequency: {frequency} mentions, promoted to True Positive")
                            else:
                                # Still treat as CAA if mentioned but not frequently enough
                                caa_diagnoses.append(diag_name)
                                cant_miss_matches.append({
                                    'cant_miss_diagnosis': diag_name,
                                    'ground_truth': gt_diag,
                                    'specialist': specialist,
                                    'reasoning': eval_result.reasoning
                                })
                                matched_gt.add(gt_diag)
                                print(f"      ✅ CAA MATCH: {diag_name} ↔ {gt_diag}")
                                print(f"         Reasoning: {eval_result.reasoning}")
                            break
                    else:
                        print(f"      ℹ️ No GT match for Can't Miss diagnosis")

        # Step 3: Symptom Management Analysis (v6 Enhancement)
        symptom_management_captures = []
        tm_sm_matches = []

        if round_results:
            symptom_mgmt_data = self._extract_symptom_management(round_results)

            if symptom_mgmt_data:
                print(f"\n🏥 Checking Symptom Management for TM-SM classification:")

                for gt_diag in (gt_diagnoses - matched_gt):
                    sm_match = self._analyze_symptom_management_coverage(gt_diag, symptom_mgmt_data)

                    if sm_match:
                        symptom_management_captures.append(gt_diag)
                        tm_sm_matches.append({
                            'ground_truth': gt_diag,
                            'symptom_intervention': sm_match['intervention'],
                            'clinical_rationale': sm_match['rationale']
                        })
                        matched_gt.add(gt_diag)  # Mark as handled by symptom management
                        print(f"      🏥 TM-SM: {gt_diag} covered by '{sm_match['intervention']}'")

        # Step 4: Reasoning Analysis for Remaining Unmatched
        appropriately_excluded = []
        true_misses = []

        for gt_diag in (gt_diagnoses - matched_gt):
            exclusion_evidence = self._search_for_exclusion_evidence(gt_diag, round_results)

            if exclusion_evidence:
                appropriately_excluded.append(gt_diag)
                print(f"      ✅ AE: {gt_diag} - {exclusion_evidence}")
            else:
                true_misses.append(gt_diag)
                print(f"      ❌ TM: {gt_diag} - No evidence of consideration")

        # Step 5: Calculate Enhanced Metrics
        result = self._calculate_enhanced_metrics(
            matched_pairs, system_diagnoses - matched_system, true_misses,
            caa_diagnoses, appropriately_excluded, symptom_management_captures,
            len(gt_diagnoses), context_matches, tm_sm_matches
        )

        # Display comprehensive results
        self._display_evaluation_results(result, cant_miss_matches, tm_sm_matches)

        return result

    def _extract_cant_miss_diagnoses(self, round_results: Dict) -> List[Dict[str, str]]:
        """Extract Can't Miss diagnoses from v6 round results"""
        cant_miss_diagnoses = []

        if RoundType.CANT_MISS in round_results:
            cant_miss_result = round_results[RoundType.CANT_MISS]
            if cant_miss_result.success and 'critical_diagnoses' in cant_miss_result.metadata:
                for critical_diag in cant_miss_result.metadata['critical_diagnoses']:
                    cant_miss_diagnoses.append({
                        'diagnosis': critical_diag['diagnosis'],
                        'specialist': f"Critical Care Team ({critical_diag.get('mention_count', 1)} mentions)"
                    })

        return cant_miss_diagnoses

    def _extract_symptom_management(self, round_results: Dict) -> Optional[Dict]:
        """Extract symptom management data from v6 round results"""

        if RoundType.SYMPTOM_MANAGEMENT in round_results:
            sm_result = round_results[RoundType.SYMPTOM_MANAGEMENT]
            if sm_result.success:
                return sm_result.metadata

        return None

    def _analyze_symptom_management_coverage(self, gt_diagnosis: str, symptom_data: Dict) -> Optional[Dict]:
        """Analyze if symptom management covers a ground truth diagnosis"""

        if not symptom_data or 'interventions' not in symptom_data:
            return None

        # Define symptom-to-diagnosis mappings
        symptom_mappings = {
            'dehydration': ['acute kidney injury', 'diabetic nephropathy', 'renal failure'],
            'chest pain': ['myocardial infarction', 'acute coronary syndrome', 'stemi'],
            'shortness of breath': ['heart failure', 'pulmonary embolism', 'pneumonia'],
            'hyperthermia': ['infection', 'sepsis', 'pneumonia'],
            'nausea': ['diabetic ketoacidosis', 'kidney disease'],
            'hypertension': ['hypertensive emergency', 'stroke']
        }

        gt_lower = gt_diagnosis.lower()

        # Check if any symptom management intervention addresses this diagnosis
        for intervention in symptom_data['interventions']:
            intervention_lower = intervention.lower()

            for symptom, related_diagnoses in symptom_mappings.items():
                if symptom in intervention_lower:
                    if any(related_diag in gt_lower for related_diag in related_diagnoses):
                        return {
                            'intervention': intervention,
                            'rationale': f"Symptom management addresses {symptom} which is relevant to {gt_diagnosis}"
                        }

        return None

    def _search_for_exclusion_evidence(self, gt_diagnosis: str, round_results: Dict) -> Optional[str]:
        """Search for evidence that a diagnosis was appropriately excluded"""

        if not round_results:
            return None

        # Search refinement round for exclusion evidence
        if RoundType.REFINEMENT_AND_JUSTIFICATION in round_results:
            refinement_result = round_results[RoundType.REFINEMENT_AND_JUSTIFICATION]

            for agent_name, response in refinement_result.responses.items():
                content = response.content.lower()
                gt_lower = gt_diagnosis.lower()

                # Look for exclusion patterns
                exclusion_patterns = [
                    f"exclude.*{gt_lower}",
                    f"{gt_lower}.*unlikely",
                    f"rule out.*{gt_lower}",
                    f"{gt_lower}.*tier 4",
                    f"{gt_lower}.*less likely"
                ]

                for pattern in exclusion_patterns:
                    if re.search(pattern, content):
                        return f"Considered but excluded by {agent_name}"

        return None

    def _calculate_enhanced_metrics(self, matched_pairs: List[Tuple[str, str]],
                                  false_positives: Set[str], true_misses: List[str],
                                  caa_diagnoses: List[str], appropriately_excluded: List[str],
                                  symptom_management_captures: List[str], total_gt: int,
                                  context_matches: List[Dict], tm_sm_matches: List[Dict]) -> EvaluationResult:
        """Calculate comprehensive v6 metrics"""

        tp_count = len(matched_pairs)
        fp_count = len(false_positives)
        fn_count = len(true_misses)
        caa_count = len(caa_diagnoses)
        ae_count = len(appropriately_excluded)
        tm_sm_count = len(symptom_management_captures)

        # Calculate rates
        tp_rate = tp_count / total_gt if total_gt > 0 else 0.0

        # Dynamic CAA weighting
        if tp_rate >= 0.7:
            caa_weight = 0.7
        elif tp_rate >= 0.5:
            caa_weight = 0.3
        elif tp_rate >= 0.3:
            caa_weight = 0.0
        else:
            caa_weight = -0.2

        # Traditional metrics
        total_system = tp_count + fp_count + caa_count
        precision = tp_count / total_system if total_system > 0 else 0.0
        recall = tp_count / total_gt if total_gt > 0 else 0.0

        # Enhanced metrics
        total_denominator = tp_count + caa_count + ae_count + fn_count + fp_count
        clinical_reasoning_quality = ((tp_count + (caa_weight * caa_count) + ae_count) / total_denominator
                                    if total_denominator > 0 else 0.0)

        diagnostic_safety = ((tp_count + max(0, caa_weight * caa_count)) / (tp_count + caa_count + fp_count)
                           if (tp_count + caa_count + fp_count) > 0 else 0.0)

        reasoning_thoroughness = ((tp_count + ae_count) / (tp_count + fn_count + ae_count)
                                if (tp_count + fn_count + ae_count) > 0 else 0.0)

        # v6 Enhancement: System Safety Coverage
        system_safety_coverage = ((tp_count + tm_sm_count) / total_gt
                                if total_gt > 0 else 0.0)

        return EvaluationResult(
            tp_count=tp_count,
            fp_count=fp_count,
            fn_count=fn_count,
            caa_count=caa_count,
            ae_count=ae_count,
            tm_sm_count=tm_sm_count,
            matched_diagnoses=matched_pairs,
            false_positives=list(false_positives),
            false_negatives=true_misses,
            clinical_alternatives=caa_diagnoses,
            appropriately_excluded=appropriately_excluded,
            symptom_management_captures=symptom_management_captures,
            traditional_precision=precision,
            traditional_recall=recall,
            clinical_reasoning_quality=clinical_reasoning_quality,
            diagnostic_safety=diagnostic_safety,
            reasoning_thoroughness=reasoning_thoroughness,
            system_safety_coverage=system_safety_coverage,
            tp_rate=tp_rate,
            caa_weight_applied=caa_weight,
            context_matches=context_matches,
            symptom_management_analysis={'tm_sm_matches': tm_sm_matches}
        )

    def _display_evaluation_results(self, result: EvaluationResult,
                                  cant_miss_matches: List[Dict],
                                  tm_sm_matches: List[Dict]):
        """Display comprehensive evaluation results"""

        print(f"\n🧠 AI-Enhanced Matching Results:")
        print(f"✅ True Positives: {result.tp_count}")
        for gt, sys in result.matched_diagnoses:
            print(f"   • {gt} ↔ {sys}")

        if result.caa_count > 0:
            print(f"\n🛡️ Clinically Appropriate Alternatives (CAA): {result.caa_count}")
            for match in cant_miss_matches:
                print(f"   • {match['ground_truth']} ← Can't Miss: {match['cant_miss_diagnosis']}")
                print(f"     Reasoning: {match['reasoning']}")
            print(f"   CAA Weight Applied: {result.caa_weight_applied:+.1f}")

        if result.tm_sm_count > 0:
            print(f"\n🏥 Symptom Management Captures (TM-SM): {result.tm_sm_count}")
            for match in tm_sm_matches:
                print(f"   • {match['ground_truth']} covered by {match['symptom_intervention']}")

        if result.ae_count > 0:
            print(f"\n✅ Appropriately Excluded (AE): {result.ae_count}")
            for ae_diag in result.appropriately_excluded:
                print(f"   • {ae_diag}")

        if result.context_matches:
            print(f"\n📋 Context-Informed Matches:")
            for match in result.context_matches:
                print(f"   • {match['ground_truth']} found via transcript context")

        print(f"\n⚠️ False Positives: {result.fp_count}")
        for fp in result.false_positives:
            print(f"   • {fp}")

        print(f"\n❌ True Misses: {result.fn_count}")
        for fn in result.false_negatives:
            print(f"   • {fn}")

        print(f"\n📈 Enhanced Metrics v6:")
        print(f"   Traditional Precision: {result.traditional_precision:.3f}")
        print(f"   Traditional Recall: {result.traditional_recall:.3f}")
        print(f"   Clinical Reasoning Quality: {result.clinical_reasoning_quality:.3f}")
        print(f"   Diagnostic Safety: {result.diagnostic_safety:.3f}")
        print(f"   Reasoning Thoroughness: {result.reasoning_thoroughness:.3f}")
        print(f"   System Safety Coverage: {result.system_safety_coverage:.3f}")
        print(f"   TP Rate: {result.tp_rate:.3f}")

# =============================================================================
# Legacy Compatibility & Integration
# =============================================================================

class ClinicalEvaluator(EnhancedClinicalEvaluator):
    """Legacy compatibility class"""

    def __init__(self, model_manager=None):
        if model_manager:
            super().__init__(model_manager)
        else:
            # Fallback for backward compatibility
            self.equivalence_agent = None
            self.transcript = None

    def evaluate_diagnosis(self, synthesis_result, ground_truth: Dict[str, List[str]],
                          round_results: Optional[Dict] = None) -> EvaluationResult:
        """Evaluate with enhanced v6 features when available"""

        if self.equivalence_agent:
            # Use enhanced evaluation
            return super().evaluate_diagnosis(synthesis_result, ground_truth, round_results)
        else:
            # Fallback to simple evaluation
            return self._simple_evaluation(synthesis_result, ground_truth)

    def _simple_evaluation(self, synthesis_result, ground_truth: Dict[str, List[str]]) -> EvaluationResult:
        """Simple fallback evaluation without AI enhancement"""
        system_diagnoses = set(synthesis_result.final_list)
        gt_diagnoses = set(ground_truth.keys())

        # Simple string matching
        matched_pairs = []
        matched_system = set()
        matched_gt = set()

        for sys_diag in system_diagnoses:
            for gt_diag in gt_diagnoses:
                if gt_diag in matched_gt:
                    continue

                if self._are_clinically_equivalent(sys_diag, gt_diag):
                    matched_pairs.append((gt_diag, sys_diag))
                    matched_system.add(sys_diag)
                    matched_gt.add(gt_diag)
                    break

        # Calculate basic metrics
        tp_count = len(matched_pairs)
        fp_count = len(system_diagnoses - matched_system)
        fn_count = len(gt_diagnoses - matched_gt)

        precision = tp_count / len(system_diagnoses) if system_diagnoses else 0.0
        recall = tp_count / len(gt_diagnoses) if gt_diagnoses else 0.0

        return EvaluationResult(
            tp_count=tp_count,
            fp_count=fp_count,
            fn_count=fn_count,
            matched_diagnoses=matched_pairs,
            false_positives=list(system_diagnoses - matched_system),
            false_negatives=list(gt_diagnoses - matched_gt),
            traditional_precision=precision,
            traditional_recall=recall,
            clinical_reasoning_quality=precision,  # Simplified
            diagnostic_safety=precision,
            reasoning_thoroughness=recall,
            tp_rate=recall,
            caa_weight_applied=0.0
        )

    def _are_clinically_equivalent(self, sys_diag: str, gt_diag: str) -> bool:
        """Legacy method for backward compatibility"""
        if self.equivalence_agent:
            result = self.equivalence_agent.evaluate_diagnosis_match(sys_diag, gt_diag, self.transcript)
            return result.is_equivalent
        else:
            # Simple string matching fallback
            return sys_diag.lower().strip() == gt_diag.lower().strip()

# =============================================================================
# Testing Function
# =============================================================================

def test_evaluation_v6():
    """Test the enhanced evaluation system"""
    print("🧪 Testing Enhanced Clinical Evaluation v6")
    print("=" * 50)

    # Mock test data
    from dataclasses import dataclass

    @dataclass
    class MockSynthesis:
        final_list: List[str]

    synthesis = MockSynthesis(final_list=[
        'Acute Myocardial Infarction',
        'Type 2 Diabetes Mellitus',
        'Acute Coronary Syndrome'
    ])

    ground_truth = {
        'STEMI': ['ST elevation', 'chest pain'],
        'Acute Coronary Syndrome': ['chest pain', 'ECG changes'],
        'Diabetic Nephropathy': ['elevated creatinine', 'diabetes history']
    }

    # Test without model manager (fallback mode)
    evaluator = ClinicalEvaluator()
    result = evaluator.evaluate_diagnosis(synthesis, ground_truth)

    print(f"✅ Evaluation completed:")
    print(f"   TP: {result.tp_count}, FP: {result.fp_count}, FN: {result.fn_count}")
    print(f"   Precision: {result.traditional_precision:.3f}")
    print(f"   Recall: {result.traditional_recall:.3f}")

    return True

if __name__ == "__main__":
    test_evaluation_v6()
