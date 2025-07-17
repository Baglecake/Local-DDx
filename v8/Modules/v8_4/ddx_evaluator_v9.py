# =============================================================================
# DDx Evaluator v9 - Deterministic Clinical Equivalence Engine
# =============================================================================

"""
DDx Evaluator v9: Deterministic rule-based evaluation replicating the original
chatstorm design. Provides reproducible, transparent, research-grade evaluation.
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from ddx_core_v6 import ModelManager, DDxAgent
# =============================================================================
# 1. Clinical Equivalence Rules Engine
# =============================================================================

class AIEquivalenceMatcher:
    """AI-based diagnosis equivalence matching"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.evaluator_agent = None
        self.match_cache = {}

    def set_evaluator_agent(self, evaluator_agent: DDxAgent):
        """Set evaluator agent for equivalence matching"""
        self.evaluator_agent = evaluator_agent

    def are_diagnoses_equivalent(self, system_diagnosis: str, ground_truth_diagnosis: str) -> Dict[str, Any]:
        """Determine if two diagnoses are clinically equivalent with master key protection"""

        # ROBUSTNESS LAYER: Master key detection
        if self._detect_master_key_vulnerability(system_diagnosis, ground_truth_diagnosis):
            print(f"⚠️ Master key vulnerability detected - returning non-equivalent")
            print(f"   System: '{system_diagnosis}' | GT: '{ground_truth_diagnosis}'")
            return {
                'equivalent': False,
                'confidence': 0.0,
                'reasoning': 'Master key pattern detected - rejected for safety'
            }

        # Check cache first
        cache_key = f"{system_diagnosis.lower()}||{ground_truth_diagnosis.lower()}"
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        if not self.evaluator_agent:
            # Simple fallback
            is_equivalent = system_diagnosis.lower().strip() == ground_truth_diagnosis.lower().strip()
            result = {
                'equivalent': is_equivalent,
                'confidence': 1.0 if is_equivalent else 0.0,
                'reasoning': 'Exact string match (fallback)'
            }
            self.match_cache[cache_key] = result
            return result

        # AI evaluation
        equivalence_prompt = f"""
    Are these two medical diagnoses clinically equivalent?

    System Diagnosis: "{system_diagnosis}"
    Ground Truth Diagnosis: "{ground_truth_diagnosis}"

    Consider these as EQUIVALENT:
    - Exact matches
    - Medical abbreviations vs. full terms (e.g., "AKI" = "Acute Kidney Injury")
    - Synonymous medical terms (e.g., "Myocardial Infarction" = "Heart Attack")
    - Subtype relationships (e.g., "STEMI" is a type of "Myocardial Infarction")

    Respond with JSON:
    {{
        "equivalent": true/false,
        "confidence": 0.8,
        "reasoning": "brief clinical justification"
    }}
    """

        try:
            response = self.evaluator_agent.generate_response(
                equivalence_prompt, "equivalence_evaluation"
            )

            result = self._parse_equivalence_response(response.content)
            self.match_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"⚠️ AI equivalence check failed: {e}")
            # Fallback to simple matching
            is_equivalent = system_diagnosis.lower().strip() == ground_truth_diagnosis.lower().strip()
            result = {
                'equivalent': is_equivalent,
                'confidence': 0.5,
                'reasoning': f'Fallback match due to error: {str(e)}'
            }
            self.match_cache[cache_key] = result
            return result

    def _parse_equivalence_response(self, content: str) -> Dict[str, Any]:
        """Parse equivalence evaluation response"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return {
                    'equivalent': bool(data.get('equivalent', False)),
                    'confidence': float(data.get('confidence', 0.5)),
                    'reasoning': data.get('reasoning', 'AI evaluation')
                }
        except:
            pass

        # Fallback parsing
        content_lower = content.lower()
        equivalent = 'true' in content_lower and 'equivalent' in content_lower

        return {
            'equivalent': equivalent,
            'confidence': 0.6,
            'reasoning': 'Parsed from natural language response'
        }

    def _detect_master_key_vulnerability(self, system_diagnosis: str, gt_diagnosis: str) -> bool:
        """Detect potential master key attacks that could fool LLM judges"""

        # Master key patterns from the paper - adapted for medical context
        medical_master_keys = [
            # Non-word symbols
            " ", ".", ",", ":",

            # Medical reasoning openers (most dangerous)
            "based on the clinical presentation",
            "the differential diagnosis includes",
            "given the patient's history",
            "clinical reasoning suggests",
            "the most likely diagnosis",
            "considering the symptoms",
            "differential diagnosis",
            "clinical assessment",
            "diagnostic evaluation",

            # Generic reasoning openers
            "thought process:", "solution", "let's solve this step by step",
            "to solve this problem", "we need to consider"
        ]

        # Check if either diagnosis is just a master key pattern
        system_lower = system_diagnosis.lower().strip()
        gt_lower = gt_diagnosis.lower().strip()

        for key in medical_master_keys:
            # Exact match or starts with pattern
            if (system_lower == key.lower() or
                system_lower.startswith(key.lower()) or
                gt_lower == key.lower() or
                gt_lower.startswith(key.lower())):
                return True

        # Check for very short responses (< 5 chars) that aren't real diagnoses
        if len(system_lower) < 5 or len(gt_lower) < 5:
            return True

        return False


class ClinicalEquivalenceEngine:
    """Deterministic clinical equivalence matching based on chatstorm rules"""

    def __init__(self):
        """Initialize clinical equivalence engine with AI fallback support"""
        self.medical_synonyms = self._build_medical_synonyms()
        self.clinical_hierarchies = self._build_clinical_hierarchies()
        self.ai_matcher = None  # Will be set by parent evaluator if available

    def _build_medical_synonyms(self) -> Dict[str, Set[str]]:
        """Build comprehensive medical synonym dictionary"""
        synonyms = {
            # Cardiovascular
            'myocardial infarction': {'mi', 'heart attack', 'acute mi'},
            'acute coronary syndrome': {'acs', 'acute coronary syndrome'},
            'congestive heart failure': {'chf', 'heart failure', 'cardiac failure'},
            'atrial fibrillation': {'afib', 'af', 'atrial fibrillation'},

            # Renal/Nephrology
            'acute kidney injury': {'aki', 'acute renal failure', 'arf'},
            'chronic kidney disease': {'ckd', 'chronic renal failure', 'crf'},
            'contrast-induced nephropathy': {'cin', 'contrast nephropathy', 'contrast-induced aki'},

            # Respiratory
            'chronic obstructive pulmonary disease': {'copd'},
            'community-acquired pneumonia': {'cap', 'pneumonia'},
            'acute respiratory distress syndrome': {'ards'},
            'pulmonary embolism': {'pe'},
            'deep vein thrombosis': {'dvt'},

            # Endocrine
            'diabetes mellitus': {'dm', 'diabetes'},
            'diabetic ketoacidosis': {'dka'},
            'type 1 diabetes': {'t1dm', 'iddm'},
            'type 2 diabetes': {'t2dm', 'niddm'},

            # Hematology - EXPANDED
            'thrombotic thrombocytopenic purpura': {'ttp'},
            'thrombotic microangiopathy': {'tma'},
            'disseminated intravascular coagulation': {'dic'},
            'hemolytic uremic syndrome': {'hus'},
            'hemolytic anemia': {'ha', 'hemolytic anemia'},
            'microangiopathic hemolytic anemia': {'maha', 'microangiopathic ha'},
            'autoimmune hemolytic anemia': {'aiha', 'autoimmune ha'},
            'paroxysmal nocturnal hemoglobinuria': {'pnh'},
            'sickle cell disease': {'scd', 'sickle cell anemia'},
            'aplastic anemia': {'aplastic anemia'},
            'iron deficiency anemia': {'ida', 'iron deficiency'},
            'megaloblastic anemia': {'b12 deficiency', 'folate deficiency'},
            'spherocytosis': {'hereditary spherocytosis'},
            'thrombocytopenia': {'low platelets'},
            'immune thrombocytopenic purpura': {'itp', 'idiopathic thrombocytopenic purpura'},
            'heparin induced thrombocytopenia': {'hit'},
            'antiphospholipid syndrome': {'aps', 'antiphospholipid antibody syndrome'},

            # Rheumatology
            'systemic lupus erythematosus': {'sle', 'lupus'},
            'rheumatoid arthritis': {'ra'},
            'ankylosing spondylitis': {'as'},

            # Crystal arthropathies and joint disorders
            'pseudogout': {
                'cppd', 'calcium pyrophosphate deposition disease',
                'calcium-pyrophosphate disease', 'cppd arthropathy',
                'chondrocalcinosis', 'cppd deposition disease'
            },
            'gout': {'gouty arthritis', 'uric acid arthropathy'},
            'osteoarthritis': {'oa', 'degenerative joint disease'},

            # Infectious/Pulmonary conditions
            'pneumocystis jirovecii pneumonia': {
                'pneumocystis pneumonia', 'pjp', 'pcp',
                'pneumocystis jiroveci pneumonia'
            },
            'allergic bronchopulmonary aspergillosis': {
                'abpa', 'allergic aspergillosis'
            },
            'acute pulmonary aspergillosis': {
                'invasive aspergillosis', 'aspergillus pneumonia'
            },
            'histoplasmosis': {'histo', 'histoplasma infection'},
            'coccidioidomycosis': {'cocci', 'valley fever'},
            'tuberculosis': {'tb', 'pulmonary tuberculosis'},

            # Neurology
            'transient ischemic attack': {'tia'},
            'cerebrovascular accident': {'cva', 'stroke'},
            'multiple sclerosis': {'ms'},

            # Gastroenterology
            'gastroesophageal reflux disease': {'gerd'},
            'inflammatory bowel disease': {'ibd'},
            'peptic ulcer disease': {'pud'},
            'infectious gastroenteritis': {'gastroenteritis', 'viral gastroenteritis', 'bacterial gastroenteritis'},

            # Infectious Disease
            'urinary tract infection': {'uti'},
            'acquired immunodeficiency syndrome': {'aids'},
            'human immunodeficiency virus': {'hiv'},

            # Other
            'end stage renal disease': {'esrd'},
            'acute tubular necrosis': {'atn'},
            'hypertension': {'htn', 'high blood pressure'},
            'hypotension': {'low blood pressure'},
        }

        # Create bidirectional mapping
        bidirectional = {}
        for canonical, aliases in synonyms.items():
            bidirectional[canonical] = aliases.copy()
            for alias in aliases:
                bidirectional[alias] = {canonical}.union(aliases - {alias})

        return bidirectional

    def _build_clinical_hierarchies(self) -> Dict[str, List[str]]:
        """Build subtype/supertype relationships"""
        return {
            # Supertype: [list of subtypes]
            'myocardial infarction': [
                'stemi', 'nstemi', 'st elevation myocardial infarction',
                'non-st elevation myocardial infarction', 'acute mi'
            ],
            'pneumonia': [
                'community-acquired pneumonia', 'hospital-acquired pneumonia',
                'ventilator-associated pneumonia', 'aspiration pneumonia',
                'bacterial pneumonia', 'viral pneumonia', 'fungal pneumonia',
                'pneumocystis pneumonia', 'pneumocystis jirovecii pneumonia',  # ← ADDED!
                'histoplasmosis', 'coccidioidomycosis'  # ← ADDED!
            ],
            'arthritis': [
                'osteoarthritis', 'rheumatoid arthritis', 'psoriatic arthritis',
                'septic arthritis', 'reactive arthritis', 'gouty arthritis'
            ],
            'acute kidney injury': [
                'contrast-induced nephropathy', 'drug-induced aki', 'ischemic aki',
                'nephrotoxic aki', 'prerenal aki', 'intrinsic aki', 'postrenal aki'
            ],
            'heart failure': [
                'congestive heart failure', 'systolic heart failure',
                'diastolic heart failure', 'acute heart failure', 'chronic heart failure'
            ],
            'diabetes mellitus': [
                'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes',
                'drug-induced diabetes', 'secondary diabetes'
            ],
            'vasculitis': [
                'systemic vasculitis', 'necrotizing vasculitis', 'drug-induced vasculitis',
                'hypersensitivity vasculitis', 'anca-associated vasculitis'
            ],
            'nephropathy': [
                'diabetic nephropathy', 'hypertensive nephropathy', 'contrast-induced nephropathy',
                'drug-induced nephropathy', 'ischemic nephropathy'
            ],
            'aspergillosis': [
                'acute pulmonary aspergillosis', 'invasive aspergillosis',
                'allergic bronchopulmonary aspergillosis', 'abpa',
                'aspergillus pneumonia', 'allergic aspergillosis'
            ],
            'fungal pneumonia': [
                'histoplasmosis', 'coccidioidomycosis', 'aspergillosis',
                'pneumocystis pneumonia', 'blastomycosis', 'cryptococcosis'
            ],
            'endemic mycoses': [
                'histoplasmosis', 'coccidioidomycosis', 'blastomycosis',
                'paracoccidioidomycosis'
            ],
            'opportunistic infections': [
                'pneumocystis pneumonia', 'aspergillosis', 'cryptococcosis',
                'cytomegalovirus', 'toxoplasmosis'
            ],
            'pulmonary infections': [
                'pneumonia', 'tuberculosis', 'histoplasmosis',
                'coccidioidomycosis', 'aspergillosis', 'pneumocystis pneumonia'
            ],

            # HEMATOLOGIC HIERARCHIES
            'hemolytic anemia': [
                'autoimmune hemolytic anemia', 'microangiopathic hemolytic anemia',
                'hereditary spherocytosis', 'sickle cell anemia',
                'paroxysmal nocturnal hemoglobinuria', 'glucose-6-phosphate dehydrogenase deficiency'
            ],
            'thrombotic microangiopathy': [
                'thrombotic thrombocytopenic purpura', 'hemolytic uremic syndrome',
                'atypical hemolytic uremic syndrome', 'complement-mediated ttp'
            ],
            'bleeding disorders': [
                'disseminated intravascular coagulation', 'immune thrombocytopenic purpura',
                'heparin induced thrombocytopenia', 'thrombocytopenia',
                'von willebrand disease', 'hemophilia'
            ],
            'anemia': [
                'iron deficiency anemia', 'megaloblastic anemia', 'hemolytic anemia',
                'aplastic anemia', 'chronic disease anemia', 'sickle cell anemia'
            ],
            'gastroenteritis': [
                'infectious gastroenteritis', 'viral gastroenteritis',
                'bacterial gastroenteritis', 'parasitic gastroenteritis'
            ]
        }

    def normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis for comparison"""
        if not diagnosis or not isinstance(diagnosis, str):
            return ""

        # Convert to lowercase and strip
        normalized = diagnosis.lower().strip()

        # Remove common prefixes that don't affect equivalence
        prefixes_to_remove = ['acute', 'chronic', 'primary', 'secondary', 'idiopathic']
        for prefix in prefixes_to_remove:
            pattern = rf'^{prefix}\s+'
            normalized = re.sub(pattern, '', normalized)

        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def are_clinically_equivalent(self, diag1: str, diag2: str) -> Tuple[bool, str, float]:
        """
        Determine if two diagnoses are clinically equivalent

        Returns:
            (is_equivalent, reasoning, confidence)
        """
        if not diag1 or not diag2:
            return False, "Empty diagnosis", 0.0

        # Normalize both diagnoses
        norm1 = self.normalize_diagnosis(diag1)
        norm2 = self.normalize_diagnosis(diag2)

        # Rule 1: Direct match after normalization
        if norm1 == norm2:
            return True, "Direct match after normalization", 1.0

        # Rule 2: Medical synonym match
        if self._check_synonym_match(norm1, norm2):
            return True, f"Medical synonym match", 0.95

        # Rule 3: Subtype/Supertype match
        hierarchy_result = self._check_hierarchy_match(norm1, norm2)
        if hierarchy_result[0]:
            return True, hierarchy_result[1], 0.90

        # Rule 4: Partial match with high overlap
        overlap_score = self._calculate_word_overlap(norm1, norm2)
        if overlap_score >= 0.8:
            return True, f"High word overlap ({overlap_score:.2f})", overlap_score

        return False, "No clinical equivalence found", 0.0

    def _check_synonym_match(self, norm1: str, norm2: str) -> bool:
        """Check if diagnoses are medical synonyms"""
        # Check direct synonym lookup
        if norm1 in self.medical_synonyms:
            if norm2 in self.medical_synonyms[norm1] or norm2 == norm1:
                return True

        if norm2 in self.medical_synonyms:
            if norm1 in self.medical_synonyms[norm2] or norm1 == norm2:
                return True

        return False

    def _check_hierarchy_match(self, norm1: str, norm2: str) -> Tuple[bool, str]:
        """Check subtype/supertype relationships using scalable algorithms with AI fallback"""

        # Keep existing predefined hierarchies for known cases
        for supertype, subtypes in self.clinical_hierarchies.items():
            supertype_norm = self.normalize_diagnosis(supertype)
            if norm1 == supertype_norm:
                for subtype in subtypes:
                    subtype_norm = self.normalize_diagnosis(subtype)
                    if norm2 == subtype_norm:
                        return True, f"Subtype relationship: {norm2} is a type of {norm1}"
            if norm2 == supertype_norm:
                for subtype in subtypes:
                    subtype_norm = self.normalize_diagnosis(subtype)
                    if norm1 == subtype_norm:
                        return True, f"Subtype relationship: {norm1} is a type of {norm2}"

        # ALGORITHMIC SCALING APPROACH
        # Method 1: Check if one diagnosis contains the other (specificity relationship)
        if norm1 in norm2 or norm2 in norm1:
            longer = norm2 if len(norm2) > len(norm1) else norm1
            shorter = norm1 if len(norm1) < len(norm2) else norm2
            return True, f"Specificity relationship: {longer} is more specific than {shorter}"

        # Method 2: Shared medical root + different modifiers
        words1 = norm1.split()
        words2 = norm2.split()

        # Remove common medical modifiers
        modifiers = {'acute', 'chronic', 'allergic', 'drug-induced', 'contrast-induced',
                    'primary', 'secondary', 'idiopathic', 'autoimmune', 'infectious'}

        core1 = [w for w in words1 if w not in modifiers]
        core2 = [w for w in words2 if w not in modifiers]

        # If cores match but modifiers differ, they're related
        if core1 == core2 and words1 != words2:
            return True, f"Same condition with different modifiers"

        # Method 3: Calculate semantic similarity using edit distance
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()

        if similarity >= 0.8:
            return True, f"High semantic similarity ({similarity:.2f})"

        # AI FALLBACK with master key protection
        if hasattr(self, 'ai_matcher') and self.ai_matcher:
            try:
                # Use protected AI matching from evaluator_v8
                ai_result = self.ai_matcher.are_diagnoses_equivalent(norm1, norm2)
                if ai_result.get('equivalent', False):
                    return True, f"AI semantic match: {ai_result.get('reasoning', 'AI determined equivalence')}"
            except Exception as e:
                print(f"⚠️ AI equivalence check failed: {e}")

        return False, "No match found"

    def _calculate_word_overlap(self, norm1: str, norm2: str) -> float:
        """Calculate word overlap score"""
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

# =============================================================================
# 2. Transcript Search Engine
# =============================================================================

class TranscriptSearchEngine:
    """Search transcript for evidence of diagnosis consideration"""

    def __init__(self):
        self.evidence_keywords = {
            'exclusion': [
                'less likely than', 'more suggestive of', 'ruled out due to',
                'excluded', 'excluded', 'unlikely', 'tier 4', 'tier four',
                'demoted', 'to be demoted', 'not supported by'
            ],
            'consideration': [
                'considered', 'evaluated', 'assessed', 'discussed', 'analyzed',
                'tier 1', 'tier 2', 'tier 3', 'tier one', 'tier two', 'tier three',
                'differential', 'possible', 'potential', 'candidate'
            ],
            'voting': [
                'yes vote', 'no vote', 'unanimous', 'tie-breaker', 'credibility',
                'threshold', 'weighted', 'voting'
            ],
            'synthesizer': [
                'lost tie-breaker', 'did not meet threshold', 'highest yes vote',
                'credibility score', 'excluded', 'included', 'provisional list'
            ]
        }

    def search_diagnosis_evidence(self, diagnosis: str, transcript: str) -> Dict[str, Any]:
        """Search transcript for evidence of diagnosis consideration"""
        if not transcript:
            return {
                'found_evidence': False,
                'evidence_type': 'none',
                'evidence_quotes': [],
                'classification': 'TM',
                'confidence': 0.0
            }

        diagnosis_variants = self._generate_diagnosis_variants(diagnosis)
        evidence_found = []
        evidence_type = 'none'

        # Search for mentions of diagnosis
        for variant in diagnosis_variants:
            quotes = self._find_diagnosis_mentions(variant, transcript)
            evidence_found.extend(quotes)

        if not evidence_found:
            return {
                'found_evidence': False,
                'evidence_type': 'none',
                'evidence_quotes': [],
                'classification': 'TM',
                'confidence': 0.9
            }

        # Classify evidence type
        exclusion_evidence = self._find_exclusion_evidence(evidence_found)
        consideration_evidence = self._find_consideration_evidence(evidence_found)

        if exclusion_evidence:
            classification = 'AE'
            evidence_type = 'exclusion'
            confidence = 0.8
        elif consideration_evidence:
            classification = 'AE'
            evidence_type = 'consideration'
            confidence = 0.7
        else:
            classification = 'TM'
            evidence_type = 'mention_only'
            confidence = 0.6

        return {
            'found_evidence': True,
            'evidence_type': evidence_type,
            'evidence_quotes': evidence_found[:3],  # Limit to top 3
            'classification': classification,
            'confidence': confidence
        }

    def _generate_diagnosis_variants(self, diagnosis: str) -> List[str]:
        """Generate search variants for a diagnosis"""
        variants = [diagnosis.lower()]

        # Add common abbreviations and variations
        abbrev_map = {
            'acute kidney injury': ['aki', 'acute renal failure'],
            'myocardial infarction': ['mi', 'heart attack'],
            'congestive heart failure': ['chf', 'heart failure'],
            # Add more as needed
        }

        diag_lower = diagnosis.lower()
        for full_name, abbrevs in abbrev_map.items():
            if diag_lower == full_name:
                variants.extend(abbrevs)
            elif diag_lower in abbrevs:
                variants.append(full_name)

        return variants

    def _find_diagnosis_mentions(self, diagnosis: str, transcript: str) -> List[str]:
        """Find mentions of diagnosis in transcript"""
        quotes = []
        transcript_lower = transcript.lower()

        # Find all mentions
        start = 0
        while True:
            pos = transcript_lower.find(diagnosis, start)
            if pos == -1:
                break

            # Extract context around mention
            context_start = max(0, pos - 100)
            context_end = min(len(transcript), pos + len(diagnosis) + 100)
            context = transcript[context_start:context_end].strip()

            if context and context not in quotes:
                quotes.append(context)

            start = pos + 1

        return quotes

    def _find_exclusion_evidence(self, evidence_quotes: List[str]) -> List[str]:
        """Find exclusion evidence in quotes"""
        exclusion_evidence = []

        for quote in evidence_quotes:
            quote_lower = quote.lower()
            for keyword in self.evidence_keywords['exclusion']:
                if keyword in quote_lower:
                    exclusion_evidence.append(quote)
                    break

        return exclusion_evidence

    def _find_consideration_evidence(self, evidence_quotes: List[str]) -> List[str]:
        """Find consideration evidence in quotes"""
        consideration_evidence = []

        for quote in evidence_quotes:
            quote_lower = quote.lower()
            for keyword in self.evidence_keywords['consideration']:
                if keyword in quote_lower:
                    consideration_evidence.append(quote)
                    break

        return consideration_evidence

# =============================================================================
# 3. Enhanced Classification System
# =============================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result"""
    # Core metrics (preserved interface)
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    caa_count: int = 0
    ae_count: int = 0
    tm_sm_count: int = 0

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
    system_safety_coverage: float = 0.0
    tp_rate: float = 0.0
    caa_weight_applied: float = 0.0

    # Enhanced analysis
    reasoning_analysis: Dict[str, Any] = field(default_factory=dict)
    context_matches: List[Dict[str, str]] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)

# =============================================================================
# 4. Main Deterministic Evaluator
# =============================================================================

class DeterministicClinicalEvaluator:
    """Main deterministic evaluator using chatstorm design"""

    def __init__(self, model_manager=None):
        """Initialize deterministic evaluator with AI fallback and master key protection"""
        self.equivalence_engine = ClinicalEquivalenceEngine()
        self.transcript_search = TranscriptSearchEngine()

        # Enhanced compatibility layer with AI fallback
        if model_manager:
            self.equivalence_agent = AIEquivalenceMatcher(model_manager)
            # Connect AI matcher to equivalence engine for fallback
            self.equivalence_engine.ai_matcher = self.equivalence_agent
        else:
            self.equivalence_agent = None

        # Missing interface methods for compatibility
        self.transcript = None

    def set_transcript(self, transcript: Dict):
        """Compatibility method - transcript is handled internally"""
        self.transcript = transcript

    def set_evaluator_agent(self, evaluator_agent):
        """Compatibility method - not needed for deterministic evaluation"""
        pass  # No-op for deterministic evaluator

    def evaluate_diagnosis(self, synthesis_result, ground_truth: Dict[str, List[str]],
                          round_results: Optional[Dict] = None,
                          full_transcript: Optional[Dict] = None) -> EvaluationResult:
        """Main evaluation method using deterministic rules"""

        print(f"\n📊 DETERMINISTIC CLINICAL EVALUATION")
        print("=" * 50)

        # Get transcript text for search
        transcript_text = self._extract_transcript_text(round_results, full_transcript)

        # Step 1: Initial matching
        system_diagnoses = set(synthesis_result.final_list)
        gt_diagnoses = set(ground_truth.keys())

        print(f"🎯 System diagnoses: {len(system_diagnoses)}")
        print(f"📋 Ground truth: {len(gt_diagnoses)}")
        print(f"🔍 Using deterministic clinical equivalence rules...")

        # Phase 1: Standard matching process with audit trail
        audit_trail = []
        audit_trail.append("=== AUDIT TRAIL START ===")
        audit_trail.append(f"GT_List: {list(gt_diagnoses)}")
        audit_trail.append(f"Team_List: {list(system_diagnoses)}")

        # Find matches using deterministic rules
        matched_pairs = []
        matched_system = set()
        matched_gt = set()

        audit_trail.append("\n--- MATCHING PROCESS ---")
        for gt_diag in gt_diagnoses:
            match_found = False
            for sys_diag in system_diagnoses:
                if sys_diag in matched_system:
                    continue

                is_equivalent, reasoning, confidence = self.equivalence_engine.are_clinically_equivalent(gt_diag, sys_diag)

                if is_equivalent:
                    matched_pairs.append((gt_diag, sys_diag))
                    matched_system.add(sys_diag)
                    matched_gt.add(gt_diag)
                    match_found = True
                    audit_trail.append(f"✅ MATCH: {gt_diag} ↔ {sys_diag} ({confidence:.2f}) - {reasoning}")
                    print(f"   ✅ MATCH: {gt_diag} ↔ {sys_diag} ({confidence:.2f})")
                    break

            if not match_found:
                audit_trail.append(f"❌ NO MATCH: {gt_diag}")

        # Phase 2: Analyze unmatched diagnoses
        unmatched_gt = gt_diagnoses - matched_gt
        unmatched_system = system_diagnoses - matched_system

        audit_trail.append(f"\nMatched_Pairs: {matched_pairs}")
        audit_trail.append(f"Unmatched_GT: {list(unmatched_gt)}")
        audit_trail.append(f"Unmatched_System: {list(unmatched_system)}")

        # Classify unmatched ground truth
        # Classify unmatched ground truth with TM-SM detection
        appropriately_excluded = []
        true_misses = []
        symptom_management_captures = []

        if transcript_text and unmatched_gt:
            audit_trail.append("\n--- REASONING ANALYSIS ---")
            print(f"\n🔍 Analyzing {len(unmatched_gt)} unmatched ground truth diagnoses:")

            # Extract symptom management interventions first
            symptom_interventions = self._extract_symptom_management_interventions(transcript_text)

            for gt_diag in unmatched_gt:
                evidence = self.transcript_search.search_diagnosis_evidence(gt_diag, transcript_text)

                if evidence['classification'] == 'AE':
                    appropriately_excluded.append(gt_diag)
                    audit_trail.append(f"✅ AE: {gt_diag} - {evidence['evidence_type']}")
                    print(f"   ✅ AE: {gt_diag} - found {evidence['evidence_type']} evidence")
                else:
                    # Check for symptom management capture before classifying as TM
                    sm_capture = self._check_symptom_management_capture(gt_diag, symptom_interventions)

                    if sm_capture['captured']:
                        symptom_management_captures.append(gt_diag)
                        audit_trail.append(f"🏥 TM-SM: {gt_diag} - captured by symptom management: {sm_capture['intervention']}")
                        print(f"   🏥 TM-SM: {gt_diag} - symptom management captured: {sm_capture['intervention']}")
                    else:
                        true_misses.append(gt_diag)
                        audit_trail.append(f"❌ TM: {gt_diag} - no adequate consideration")
                        print(f"   ❌ TM: {gt_diag} - no adequate consideration found")
        else:
            true_misses = list(unmatched_gt)
            audit_trail.append("No transcript available - all unmatched GT classified as TM")

        # Classify unmatched system diagnoses (simplified for now)
        false_positives = list(unmatched_system)  # All unmatched system = FP for now
        clinical_alternatives = []  # Could enhance this later

        # Calculate metrics
        tp_count = len(matched_pairs)
        fp_count = len(false_positives)
        ae_count = len(appropriately_excluded)
        tm_count = len(true_misses)
        tm_sm_count = len(symptom_management_captures)

        # Dynamic CAA weighting
        tp_rate = tp_count / len(gt_diagnoses) if gt_diagnoses else 0.0
        if tp_rate >= 0.7:
            caa_weight = 0.7
        elif tp_rate >= 0.5:
            caa_weight = 0.3
        elif tp_rate >= 0.3:
            caa_weight = 0.0
        else:
            caa_weight = -0.2


        # Traditional metrics
        precision = tp_count / len(system_diagnoses) if system_diagnoses else 0.0
        recall = tp_count / (tp_count + tm_count) if (tp_count + tm_count) > 0 else 0.0


        # Enhanced metrics
        total_denominator = tp_count + ae_count + tm_count + fp_count
        clinical_reasoning_quality = ((tp_count + ae_count) / total_denominator
                                    if total_denominator > 0 else 0.0)

        diagnostic_safety = (tp_count / (tp_count + fp_count)
                           if (tp_count + fp_count) > 0 else 0.0)

        reasoning_thoroughness = ((tp_count + ae_count) / (tp_count + ae_count + tm_count)
                                if (tp_count + ae_count + tm_count) > 0 else 0.0)

        audit_trail.append(f"\n--- FINAL METRICS ---")
        audit_trail.append(f"TP: {tp_count}, FP: {fp_count}, AE: {ae_count}, TM: {tm_count}")
        audit_trail.append(f"TP_Rate: {tp_rate:.3f}, CAA_Weight: {caa_weight}")
        audit_trail.append("=== AUDIT TRAIL END ===")

        # Display results
        print(f"\n📊 Evaluation Results:")
        print(f"   ✅ True Positives: {tp_count}")
        print(f"   ⚠️ False Positives: {fp_count}")
        print(f"   ✅ Appropriately Excluded: {ae_count}")
        print(f"   ❌ True Misses: {tm_count}")
        print(f"   📈 Precision: {precision:.3f}")
        print(f"   📈 Recall: {recall:.3f}")
        print(f"   📈 Clinical Reasoning Quality: {clinical_reasoning_quality:.3f}")

        return EvaluationResult(
            tp_count=tp_count,
            fp_count=fp_count,
            fn_count=tm_count,
            ae_count=ae_count,
            matched_diagnoses=matched_pairs,
            false_positives=false_positives,
            false_negatives=true_misses,
            appropriately_excluded=appropriately_excluded,
            traditional_precision=precision,
            traditional_recall=recall,
            clinical_reasoning_quality=clinical_reasoning_quality,
            diagnostic_safety=diagnostic_safety,
            reasoning_thoroughness=reasoning_thoroughness,
            tp_rate=tp_rate,
            caa_weight_applied=caa_weight,
            tm_sm_count=tm_sm_count,
            symptom_management_captures=symptom_management_captures,
            audit_trail=audit_trail
        )

    def _extract_transcript_text(self, round_results: Optional[Dict],
                                full_transcript: Optional[Dict]) -> str:
        """Extract searchable text from transcript"""
        if not round_results:
            return ""

        transcript_parts = []

        # Extract from round results
        for round_type, result in round_results.items():
            if hasattr(result, 'responses'):
                for agent_name, response in result.responses.items():
                    if hasattr(response, 'content'):
                        transcript_parts.append(f"{agent_name}: {response.content}")

        return "\n".join(transcript_parts)

    def _extract_symptom_management_interventions(self, transcript_text: str) -> List[Dict[str, str]]:
        """Extract symptom management interventions from transcript"""
        interventions = []

        # Look for SYMPTOM_MITIGATION blocks
        mitigation_pattern = r'<SYMPTOM_MITIGATION>(.*?)</SYMPTOM_MITIGATION>'
        matches = re.findall(mitigation_pattern, transcript_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            lines = match.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and (line.startswith('-') or line.startswith('•')):
                    # Parse "- Symptom: Intervention" format
                    parts = line[1:].strip().split(':', 1)
                    if len(parts) == 2:
                        symptom = parts[0].strip()
                        intervention = parts[1].strip()
                        interventions.append({
                            'symptom': symptom,
                            'intervention': intervention,
                            'raw_line': line
                        })

        return interventions

    def _check_symptom_management_capture(self, diagnosis: str, symptom_interventions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Check if a missed diagnosis was captured by symptom management"""

        # Define clinical mappings: diagnosis -> symptoms/interventions that would address it
        clinical_mappings = {
            'cholesterol embolization': [
                'circulation', 'perfusion', 'vascular', 'blood flow', 'ischemia',
                'anticoagulation', 'heparin', 'circulation support'
            ],
            'drug-induced interstitial nephritis': [
                'nephrotoxic', 'medication review', 'drug discontinuation',
                'renal protection', 'nephrology'
            ],
            'contrast-induced nephropathy': [
                'hydration', 'fluid resuscitation', 'renal protection',
                'nephrology', 'kidney function'
            ],
            'acute kidney injury': [
                'hydration', 'fluid resuscitation', 'renal protection',
                'nephrology', 'kidney function', 'dialysis'
            ],
            'diabetic nephropathy': [
                'glucose control', 'diabetes management', 'renal protection',
                'nephrology', 'kidney function'
            ],
            'hypovolemic shock': [
                'fluid resuscitation', 'iv fluids', 'volume expansion',
                'blood pressure support', 'circulation'
            ],
            'sepsis': [
                'antibiotic', 'infection control', 'fluid resuscitation',
                'blood pressure support'
            ],
            'heart failure': [
                'diuretic', 'fluid management', 'cardiac support',
                'blood pressure management'
            ]
        }

        diagnosis_lower = diagnosis.lower()

        # Check if any symptom management intervention addresses this diagnosis
        for intervention_data in symptom_interventions:
            intervention_text = (intervention_data['symptom'] + ' ' + intervention_data['intervention']).lower()

            # Direct mapping check
            for diag_key, intervention_keywords in clinical_mappings.items():
                if diag_key in diagnosis_lower:
                    for keyword in intervention_keywords:
                        if keyword in intervention_text:
                            return {
                                'captured': True,
                                'intervention': intervention_data['raw_line'],
                                'reasoning': f"Symptom management addressed {keyword} which clinically relates to {diagnosis}"
                            }

        return {'captured': False, 'intervention': None, 'reasoning': 'No relevant symptom management found'}

# =============================================================================
# 5. Compatibility Interface
# =============================================================================

class ClinicalEvaluator(DeterministicClinicalEvaluator):
    """Legacy compatibility class that inherits from deterministic evaluator"""

    def __init__(self, model_manager=None):
        super().__init__(model_manager)
        # Add any additional legacy interface requirements

    def set_transcript(self, transcript: Dict):
        """Legacy method - transcript is handled internally by deterministic evaluator"""
        self.transcript = transcript

class EnhancedClinicalEvaluator(DeterministicClinicalEvaluator):
    """Enhanced compatibility class that inherits from deterministic evaluator"""

    def __init__(self, model_manager=None):
        super().__init__(model_manager)

    def set_evaluator_agent(self, evaluator_agent):
        """Compatibility method - not needed for deterministic evaluation"""
        pass

# =============================================================================
# 6. Testing
# =============================================================================

def test_deterministic_evaluator():
    """Test the deterministic evaluator"""
    print("🧪 Testing Deterministic Clinical Evaluator")
    print("=" * 50)

    # Mock synthesis result
    from dataclasses import dataclass

    @dataclass
    class MockSynthesis:
        final_list: List[str]

    synthesis = MockSynthesis(final_list=[
        'Acute Kidney Injury',
        'Community-Acquired Pneumonia',
        'Myocardial Infarction'
    ])

    ground_truth = {
        'AKI': ['elevated creatinine', 'oliguria'],
        'Pneumonia': ['fever', 'cough', 'infiltrate'],
        'STEMI': ['ST elevation', 'chest pain'],
        'Heart Failure': ['dyspnea', 'edema']
    }

    # Test evaluator
    evaluator = DeterministicClinicalEvaluator()
    result = evaluator.evaluate_diagnosis(synthesis, ground_truth)

    print(f"✅ Evaluation completed:")
    print(f"   TP: {result.tp_count}, FP: {result.fp_count}")
    print(f"   AE: {result.ae_count}, TM: {len(result.false_negatives)}")
    print(f"   Precision: {result.traditional_precision:.3f}")
    print(f"   Recall: {result.traditional_recall:.3f}")

    return True

if __name__ == "__main__":
    test_deterministic_evaluator()
