"""
Standalone Deterministic Evaluator for Open-XDDx Benchmarking.

Adapted from v8's ddx_evaluator_v9.py but with NO v6/v7 imports.
Pure deterministic matching: synonym dict, clinical hierarchies,
normalization, word overlap. No LLM-based matching.

Designed to evaluate v10 pipeline output against Open-XDDx ground truth.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Clinical Equivalence Engine
# =============================================================================

class ClinicalEquivalenceEngine:
    """Deterministic clinical equivalence matching (no LLM dependency)"""

    def __init__(self):
        self.medical_synonyms = self._build_medical_synonyms()
        self.clinical_hierarchies = self._build_clinical_hierarchies()

    def _build_medical_synonyms(self) -> Dict[str, Set[str]]:
        """Build comprehensive medical synonym dictionary"""
        synonyms = {
            # Cardiovascular
            'myocardial infarction': {'mi', 'heart attack', 'acute mi'},
            'acute coronary syndrome': {'acs'},
            'congestive heart failure': {'chf', 'heart failure', 'cardiac failure'},
            'atrial fibrillation': {'afib', 'af'},

            # Renal
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

            # Hematology
            'thrombotic thrombocytopenic purpura': {'ttp'},
            'thrombotic microangiopathy': {'tma'},
            'disseminated intravascular coagulation': {'dic'},
            'hemolytic uremic syndrome': {'hus'},
            'hemolytic anemia': {'ha'},
            'microangiopathic hemolytic anemia': {'maha', 'microangiopathic ha'},
            'autoimmune hemolytic anemia': {'aiha', 'autoimmune ha'},
            'paroxysmal nocturnal hemoglobinuria': {'pnh'},
            'sickle cell disease': {'scd', 'sickle cell anemia'},
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
            'pseudogout': {
                'cppd', 'calcium pyrophosphate deposition disease',
                'calcium-pyrophosphate disease', 'cppd arthropathy',
                'chondrocalcinosis', 'cppd deposition disease'
            },
            'gout': {'gouty arthritis', 'uric acid arthropathy'},
            'osteoarthritis': {'oa', 'degenerative joint disease'},

            # Infectious
            'pneumocystis jirovecii pneumonia': {
                'pneumocystis pneumonia', 'pjp', 'pcp',
                'pneumocystis jiroveci pneumonia'
            },
            'allergic bronchopulmonary aspergillosis': {'abpa', 'allergic aspergillosis'},
            'acute pulmonary aspergillosis': {'invasive aspergillosis', 'aspergillus pneumonia'},
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
            'infectious gastroenteritis': {
                'gastroenteritis', 'viral gastroenteritis',
                'bacterial gastroenteritis'
            },

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

        # Build bidirectional mapping
        bidirectional = {}
        for canonical, aliases in synonyms.items():
            bidirectional[canonical] = aliases.copy()
            for alias in aliases:
                bidirectional[alias] = {canonical}.union(aliases - {alias})

        return bidirectional

    def _build_clinical_hierarchies(self) -> Dict[str, List[str]]:
        """Build subtype/supertype relationships"""
        return {
            'myocardial infarction': [
                'stemi', 'nstemi', 'st elevation myocardial infarction',
                'non-st elevation myocardial infarction', 'acute mi'
            ],
            'pneumonia': [
                'community-acquired pneumonia', 'hospital-acquired pneumonia',
                'ventilator-associated pneumonia', 'aspiration pneumonia',
                'bacterial pneumonia', 'viral pneumonia', 'fungal pneumonia',
                'pneumocystis pneumonia', 'pneumocystis jirovecii pneumonia',
                'histoplasmosis', 'coccidioidomycosis'
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
                'diabetic nephropathy', 'hypertensive nephropathy',
                'contrast-induced nephropathy', 'drug-induced nephropathy',
                'ischemic nephropathy'
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
            'hemolytic anemia': [
                'autoimmune hemolytic anemia', 'microangiopathic hemolytic anemia',
                'hereditary spherocytosis', 'sickle cell anemia',
                'paroxysmal nocturnal hemoglobinuria',
                'glucose-6-phosphate dehydrogenase deficiency'
            ],
            'thrombotic microangiopathy': [
                'thrombotic thrombocytopenic purpura', 'hemolytic uremic syndrome',
                'atypical hemolytic uremic syndrome', 'complement-mediated ttp'
            ],
            'bleeding disorders': [
                'disseminated intravascular coagulation',
                'immune thrombocytopenic purpura',
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
            ],
        }

    def normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis for comparison"""
        if not diagnosis or not isinstance(diagnosis, str):
            return ""

        normalized = diagnosis.lower().strip()

        # Remove numbering/bullets
        normalized = re.sub(r'^[\d\.\-\*]+\s*', '', normalized)

        # Remove common prefixes that don't affect equivalence
        for prefix in ['acute', 'chronic', 'primary', 'secondary', 'idiopathic']:
            normalized = re.sub(rf'^{prefix}\s+', '', normalized)

        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def are_clinically_equivalent(self, diag1: str, diag2: str) -> Tuple[bool, str, float]:
        """
        Determine if two diagnoses are clinically equivalent.

        Returns: (is_equivalent, reasoning, confidence)
        4-tier matching: direct -> synonym -> hierarchy -> word overlap
        """
        if not diag1 or not diag2:
            return False, "Empty diagnosis", 0.0

        norm1 = self.normalize_diagnosis(diag1)
        norm2 = self.normalize_diagnosis(diag2)

        # Rule 1: Direct match
        if norm1 == norm2:
            return True, "Direct match after normalization", 1.0

        # Rule 2: Synonym match
        if self._check_synonym_match(norm1, norm2):
            return True, "Medical synonym match", 0.95

        # Rule 3: Hierarchy match (subtype/supertype + substring + core match)
        hierarchy_result = self._check_hierarchy_match(norm1, norm2)
        if hierarchy_result[0]:
            return True, hierarchy_result[1], 0.90

        # Rule 4: Word overlap (Jaccard >= 0.8)
        overlap_score = self._calculate_word_overlap(norm1, norm2)
        if overlap_score >= 0.8:
            return True, f"High word overlap ({overlap_score:.2f})", overlap_score

        return False, "No clinical equivalence found", 0.0

    def _check_synonym_match(self, norm1: str, norm2: str) -> bool:
        """Check if diagnoses are medical synonyms"""
        if norm1 in self.medical_synonyms:
            if norm2 in self.medical_synonyms[norm1]:
                return True
        if norm2 in self.medical_synonyms:
            if norm1 in self.medical_synonyms[norm2]:
                return True
        return False

    def _check_hierarchy_match(self, norm1: str, norm2: str) -> Tuple[bool, str]:
        """Check subtype/supertype relationships + substring containment"""
        # Predefined hierarchies
        for supertype, subtypes in self.clinical_hierarchies.items():
            sup_norm = self.normalize_diagnosis(supertype)
            if norm1 == sup_norm:
                for sub in subtypes:
                    if norm2 == self.normalize_diagnosis(sub):
                        return True, f"Subtype: {norm2} is a type of {norm1}"
            if norm2 == sup_norm:
                for sub in subtypes:
                    if norm1 == self.normalize_diagnosis(sub):
                        return True, f"Subtype: {norm1} is a type of {norm2}"

        # Substring containment
        if norm1 in norm2 or norm2 in norm1:
            longer = norm2 if len(norm2) > len(norm1) else norm1
            shorter = norm1 if len(norm1) < len(norm2) else norm2
            return True, f"Specificity: {longer} contains {shorter}"

        # Core word match (strip modifiers)
        modifiers = {'acute', 'chronic', 'allergic', 'drug-induced',
                     'contrast-induced', 'primary', 'secondary',
                     'idiopathic', 'autoimmune', 'infectious'}
        core1 = [w for w in norm1.split() if w not in modifiers]
        core2 = [w for w in norm2.split() if w not in modifiers]
        if core1 and core2 and core1 == core2:
            return True, "Same condition with different modifiers"

        return False, "No hierarchy match"

    def _calculate_word_overlap(self, norm1: str, norm2: str) -> float:
        """Calculate Jaccard word overlap"""
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Evaluation Data Structures
# =============================================================================

@dataclass
class CaseEvaluation:
    """Evaluation result for a single case"""
    case_index: int
    case_name: str = ""
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    ae_count: int = 0  # Appropriately excluded (considered but not in final list)

    matched_pairs: List[Tuple[str, str]] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    appropriately_excluded: List[str] = field(default_factory=list)

    system_diagnoses: List[str] = field(default_factory=list)
    ground_truth_diagnoses: List[str] = field(default_factory=list)

    recall: float = 0.0
    precision: float = 0.0
    clinical_reasoning_quality: float = 0.0
    diagnostic_safety: float = 0.0
    duration_seconds: float = 0.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class BatchEvaluation:
    """Aggregated evaluation across multiple cases"""
    case_results: List[CaseEvaluation] = field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_ae: int = 0
    macro_recall: float = 0.0
    macro_precision: float = 0.0
    macro_f1: float = 0.0
    macro_crq: float = 0.0
    macro_safety: float = 0.0
    micro_recall: float = 0.0
    micro_precision: float = 0.0
    total_cases: int = 0
    total_duration: float = 0.0


# =============================================================================
# Diagnosis Extraction from v10 Output
# =============================================================================

def extract_system_diagnoses(result: Dict[str, Any]) -> List[str]:
    """
    Extract final diagnoses from v10 pipeline output.

    Checks multiple locations in order of priority:
    1. results.voting_result.ranked_results (Borda-voted, best source)
    2. results.final_diagnoses (synthesized list)
    3. Last round structured_data
    """
    diagnoses = []

    results = result.get('results', result)

    # Priority 1: Voting results (ranked Borda output)
    voting = results.get('voting_result', {})
    if voting:
        ranked = voting.get('ranked_results', [])
        if ranked:
            for item in ranked:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    diagnoses.append(str(item[0]))
                elif isinstance(item, str):
                    diagnoses.append(item)
            if diagnoses:
                return diagnoses[:6]

    # Priority 2: final_diagnoses list
    final = results.get('final_diagnoses', [])
    if final:
        for d in final:
            if isinstance(d, (list, tuple)):
                diagnoses.append(str(d[0]))
            elif isinstance(d, str):
                diagnoses.append(d)
        if diagnoses:
            return diagnoses[:6]

    # Priority 3: Scan structured_data from last rounds
    rounds = results.get('rounds', {})
    for round_name in reversed(list(rounds.keys())):
        round_data = rounds[round_name]
        responses = round_data.get('responses', [])
        for resp in responses:
            sd = resp.get('structured_data', {})
            if isinstance(sd, dict):
                diagnoses.extend(sd.keys())
        if diagnoses:
            break

    return list(dict.fromkeys(diagnoses))[:6]  # Deduplicate, preserve order


def extract_all_considered(result: Dict[str, Any]) -> Set[str]:
    """
    Extract ALL diagnoses mentioned anywhere in the pipeline
    (for AE classification — was it considered even if not in final list?).
    """
    considered = set()
    results = result.get('results', result)

    rounds = results.get('rounds', {})
    for round_data in rounds.values():
        responses = round_data.get('responses', [])
        for resp in responses:
            sd = resp.get('structured_data', {})
            if isinstance(sd, dict):
                considered.update(sd.keys())

    return considered


# =============================================================================
# Benchmark Evaluator
# =============================================================================

class BenchmarkEvaluator:
    """Evaluates v10 pipeline output against Open-XDDx ground truth"""

    def __init__(self):
        self.engine = ClinicalEquivalenceEngine()

    def evaluate_case(self, result: Dict[str, Any],
                      ground_truth: Dict[str, List[str]],
                      case_index: int = 0) -> CaseEvaluation:
        """
        Evaluate a single case.

        Args:
            result: v10 pipeline JSON output (from export_results)
            ground_truth: {diagnosis: [evidence_list]} from Open-XDDx
            case_index: Index in dataset
        """
        evaluation = CaseEvaluation(case_index=case_index)

        # Extract system diagnoses from pipeline output
        sys_diags = extract_system_diagnoses(result)
        gt_diags = list(ground_truth.keys())
        all_considered = extract_all_considered(result)

        evaluation.system_diagnoses = sys_diags
        evaluation.ground_truth_diagnoses = gt_diags
        evaluation.duration_seconds = (
            result.get('results', result).get('total_duration', 0)
        )

        # Case name
        case_data = result.get('case', {})
        evaluation.case_name = case_data.get('name', f"case_{case_index}")

        # Match system diagnoses to ground truth
        matched_gt = set()
        matched_sys = set()

        for gt_diag in gt_diags:
            for sys_diag in sys_diags:
                if sys_diag in matched_sys:
                    continue
                is_eq, reason, conf = self.engine.are_clinically_equivalent(
                    gt_diag, sys_diag
                )
                if is_eq:
                    evaluation.matched_pairs.append((gt_diag, sys_diag))
                    matched_gt.add(gt_diag)
                    matched_sys.add(sys_diag)
                    break

        # True Positives
        evaluation.tp_count = len(evaluation.matched_pairs)

        # False Negatives (ground truth not matched)
        for gt_diag in gt_diags:
            if gt_diag not in matched_gt:
                # Check if it was considered anywhere in pipeline
                was_considered = False
                for considered_diag in all_considered:
                    is_eq, _, _ = self.engine.are_clinically_equivalent(
                        gt_diag, considered_diag
                    )
                    if is_eq:
                        was_considered = True
                        break

                if was_considered:
                    evaluation.appropriately_excluded.append(gt_diag)
                    evaluation.ae_count += 1
                else:
                    evaluation.false_negatives.append(gt_diag)
                    evaluation.fn_count += 1

        # False Positives (system diagnoses not matching any GT)
        for sys_diag in sys_diags:
            if sys_diag not in matched_sys:
                evaluation.false_positives.append(sys_diag)
                evaluation.fp_count += 1

        # Compute metrics
        tp = evaluation.tp_count
        fp = evaluation.fp_count
        fn = evaluation.fn_count
        ae = evaluation.ae_count

        # Recall = TP / (TP + FN)
        evaluation.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Precision = TP / (TP + FP)
        evaluation.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Clinical Reasoning Quality = (TP + 0.5*AE) / (TP + FN + AE)
        evaluation.clinical_reasoning_quality = (
            (tp + 0.5 * ae) / (tp + fn + ae) if (tp + fn + ae) > 0 else 0.0
        )

        # Diagnostic Safety = (TP + AE) / (TP + FN + AE)
        evaluation.diagnostic_safety = (
            (tp + ae) / (tp + fn + ae) if (tp + fn + ae) > 0 else 0.0
        )

        return evaluation

    def evaluate_batch(self, results_dir: str,
                       ground_truths: Dict[int, Dict[str, List[str]]]) -> BatchEvaluation:
        """
        Evaluate all case results in a directory.

        Args:
            results_dir: Directory containing per-case JSON files
            ground_truths: {case_index: {diagnosis: [evidence]}}
        """
        batch = BatchEvaluation()
        json_files = sorted(
            f for f in os.listdir(results_dir) if f.endswith('.json')
            and f.startswith('case_')
        )

        for json_file in json_files:
            filepath = os.path.join(results_dir, json_file)
            try:
                with open(filepath) as f:
                    result = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Skipping {json_file}: {e}")
                continue

            # Extract case index from filename (case_000.json)
            try:
                idx = int(json_file.replace('case_', '').replace('.json', ''))
            except ValueError:
                continue

            gt = ground_truths.get(idx)
            if gt is None:
                print(f"  No ground truth for case {idx}, skipping")
                continue

            case_eval = self.evaluate_case(result, gt, case_index=idx)
            batch.case_results.append(case_eval)

        # Aggregate
        batch.total_cases = len(batch.case_results)
        if batch.total_cases == 0:
            return batch

        batch.total_tp = sum(c.tp_count for c in batch.case_results)
        batch.total_fp = sum(c.fp_count for c in batch.case_results)
        batch.total_fn = sum(c.fn_count for c in batch.case_results)
        batch.total_ae = sum(c.ae_count for c in batch.case_results)
        batch.total_duration = sum(c.duration_seconds for c in batch.case_results)

        # Macro averages
        batch.macro_recall = sum(c.recall for c in batch.case_results) / batch.total_cases
        batch.macro_precision = sum(c.precision for c in batch.case_results) / batch.total_cases
        batch.macro_f1 = sum(c.f1 for c in batch.case_results) / batch.total_cases
        batch.macro_crq = sum(c.clinical_reasoning_quality for c in batch.case_results) / batch.total_cases
        batch.macro_safety = sum(c.diagnostic_safety for c in batch.case_results) / batch.total_cases

        # Micro averages
        micro_tp = batch.total_tp
        micro_fp = batch.total_fp
        micro_fn = batch.total_fn
        batch.micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        batch.micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0

        return batch


# =============================================================================
# Reporting
# =============================================================================

def print_report(batch: BatchEvaluation):
    """Print formatted evaluation report"""
    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 70)

    print(f"\nCases evaluated: {batch.total_cases}")
    print(f"Total duration:  {batch.total_duration:.1f}s "
          f"({batch.total_duration/batch.total_cases:.1f}s/case)" if batch.total_cases > 0 else "")

    print(f"\n--- Aggregate Counts ---")
    print(f"  True Positives:          {batch.total_tp}")
    print(f"  False Positives:         {batch.total_fp}")
    print(f"  False Negatives:         {batch.total_fn}")
    print(f"  Appropriately Excluded:  {batch.total_ae}")

    print(f"\n--- Macro Averages (per-case, then averaged) ---")
    print(f"  Clinical Recall:         {batch.macro_recall:.1%}")
    print(f"  Precision:               {batch.macro_precision:.1%}")
    print(f"  F1 Score:                {batch.macro_f1:.1%}")
    print(f"  Clinical Reasoning Q:    {batch.macro_crq:.1%}")
    print(f"  Diagnostic Safety:       {batch.macro_safety:.1%}")

    print(f"\n--- Micro Averages (pooled TP/FP/FN) ---")
    print(f"  Clinical Recall:         {batch.micro_recall:.1%}")
    print(f"  Precision:               {batch.micro_precision:.1%}")

    # Per-case breakdown
    print(f"\n--- Per-Case Breakdown ---")
    print(f"{'Case':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'AE':>4} "
          f"{'Recall':>8} {'Prec':>8} {'CRQ':>8} {'Safety':>8} {'Time':>7}")
    print("-" * 70)

    for c in batch.case_results:
        print(f"{c.case_index:>6} {c.tp_count:>4} {c.fp_count:>4} "
              f"{c.fn_count:>4} {c.ae_count:>4} "
              f"{c.recall:>8.1%} {c.precision:>8.1%} "
              f"{c.clinical_reasoning_quality:>8.1%} "
              f"{c.diagnostic_safety:>8.1%} "
              f"{c.duration_seconds:>6.1f}s")

    # v7.4 comparison
    print(f"\n--- v7.4 Baseline Comparison ---")
    print(f"  v7.4 Recall:    53.3%")
    print(f"  v10  Recall:    {batch.macro_recall:.1%} "
          f"({'+'if batch.macro_recall > 0.533 else ''}"
          f"{(batch.macro_recall - 0.533)*100:.1f} pp)")
    print(f"  v7.4 Precision: 67.8%")
    print(f"  v10  Precision: {batch.macro_precision:.1%} "
          f"({'+'if batch.macro_precision > 0.678 else ''}"
          f"{(batch.macro_precision - 0.678)*100:.1f} pp)")


def save_report(batch: BatchEvaluation, output_path: str):
    """Save evaluation report as JSON"""
    report = {
        'summary': {
            'total_cases': batch.total_cases,
            'total_tp': batch.total_tp,
            'total_fp': batch.total_fp,
            'total_fn': batch.total_fn,
            'total_ae': batch.total_ae,
            'total_duration_seconds': batch.total_duration,
            'macro_recall': batch.macro_recall,
            'macro_precision': batch.macro_precision,
            'macro_f1': batch.macro_f1,
            'macro_crq': batch.macro_crq,
            'macro_safety': batch.macro_safety,
            'micro_recall': batch.micro_recall,
            'micro_precision': batch.micro_precision,
        },
        'v74_baseline': {
            'recall': 0.533,
            'precision': 0.678,
        },
        'per_case': [
            {
                'case_index': c.case_index,
                'case_name': c.case_name,
                'tp': c.tp_count,
                'fp': c.fp_count,
                'fn': c.fn_count,
                'ae': c.ae_count,
                'recall': c.recall,
                'precision': c.precision,
                'f1': c.f1,
                'crq': c.clinical_reasoning_quality,
                'safety': c.diagnostic_safety,
                'duration': c.duration_seconds,
                'matched_pairs': c.matched_pairs,
                'false_positives': c.false_positives,
                'false_negatives': c.false_negatives,
                'appropriately_excluded': c.appropriately_excluded,
                'system_diagnoses': c.system_diagnoses,
                'ground_truth': c.ground_truth_diagnoses,
            }
            for c in batch.case_results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate v10 benchmark results against Open-XDDx ground truth"
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing per-case JSON results')
    parser.add_argument('--dataset', default='v10_full_pipeline/data/Open-XDDx.xlsx',
                        help='Path to Open-XDDx.xlsx')
    parser.add_argument('--output', default=None,
                        help='Output JSON report path')
    args = parser.parse_args()

    # Load ground truth from dataset
    import ast
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. Install with: pip install pandas openpyxl")
        return

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_excel(args.dataset)

    ground_truths = {}
    for _, row in df.iterrows():
        idx = int(row['Index'])
        interpretation = row['interpretation']
        try:
            gt = ast.literal_eval(interpretation) if isinstance(interpretation, str) else interpretation
        except (ValueError, SyntaxError):
            try:
                gt = json.loads(interpretation)
            except (json.JSONDecodeError, TypeError):
                print(f"  Warning: Could not parse ground truth for case {idx}")
                continue
        ground_truths[idx] = gt

    print(f"Loaded {len(ground_truths)} ground truth cases")

    # Evaluate
    evaluator = BenchmarkEvaluator()
    batch = evaluator.evaluate_batch(args.results_dir, ground_truths)

    # Report
    print_report(batch)

    if args.output:
        save_report(batch, args.output)
    else:
        default_output = os.path.join(args.results_dir, 'evaluation_report.json')
        save_report(batch, default_output)


if __name__ == '__main__':
    main()
