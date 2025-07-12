# =============================================================================
# DDx Utilities - Centralized Helper Functions
# =============================================================================

"""
Centralized utilities for the DDx experiment.
Contains robust, reusable functions used across multiple modules.
"""

import re
import unicodedata
from typing import List, Set, Dict, Any, Optional

# =============================================================================
# Robust Diagnosis Extraction
# =============================================================================

# Compiled regex patterns for efficiency
DIAG_HEAD = re.compile(r"^\s*(?:\d+\.|[-–•])\s*(.+)$", re.M)

# FIXED: More specific medical condition pattern
DISEASE_NGRAM = re.compile(
    r"\b((?:psoriatic|rheumatoid|osteo|ankylosing|acute kidney|diabetic|contrast.*induced|drug.*induced|thrombotic|myocardial|community.*acquired)\s+"
    r"(?:arthritis|spondylitis|injury|nephropathy|infarction|pneumonia|purpura|microangiopathy))\b",
    re.I,
)

# Medical term aliases and standardization
ALIAS = {
    "AKI": "Acute Kidney Injury",
    "CI-AKI": "Contrast Nephropathy",
    "ARF": "Acute Kidney Injury",
    "CIN": "Contrast Nephropathy",
    "TTP": "Thrombotic Thrombocytopenic Purpura",
    "TMA": "Thrombotic Microangiopathy",
    "MI": "Myocardial Infarction",
    "CHF": "Congestive Heart Failure",
    "CKD": "Chronic Kidney Disease",
    "ATN": "Acute Tubular Necrosis",
    "SLE": "Systemic Lupus Erythematosus",
    "ABPA": "Allergic Bronchopulmonary Aspergillosis"
}

def extract_diagnoses(text: str) -> List[str]:
    """
    Return unique diagnoses mentioned as primary claims.

    This is the single, robust diagnostic extractor for the entire DDx system.
    Works on any prose format - no JSON dependency required.

    Args:
        text: Raw response text from any source

    Returns:
        List of unique diagnosis names, deduplicated and standardized
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize text to handle fancy unicode characters
    text = unicodedata.normalize("NFKC", text)

    diagnoses: List[str] = []

    # Method 1: Extract numbered/bulleted list items
    diagnoses.extend([m.group(1).strip(" :") for m in DIAG_HEAD.finditer(text)])

    # Method 2: JSON/dict extraction (preserves existing functionality)
    json_diagnoses = _extract_from_json_blocks(text)
    diagnoses.extend(json_diagnoses)

    # Method 3: Medical n-gram pattern matching
    diagnoses.extend([m.group(1).title() for m in DISEASE_NGRAM.finditer(text)])

    # Method 4: Domain-specific medical patterns
    medical_diagnoses = _extract_medical_patterns(text)
    diagnoses.extend(medical_diagnoses)

    # Standardize aliases
    diagnoses = [ALIAS.get(d, d) for d in diagnoses]

    # Deduplicate while preserving order
    return list(dict.fromkeys([d.strip() for d in diagnoses if d.strip()]))

def _extract_from_json_blocks(text: str) -> List[str]:
    """Extract diagnoses from JSON/dict blocks in text"""
    import json
    import ast

    diagnoses = []

    # Method 1: Look for JSON at the start of response (our new format)
    lines = text.strip().split('\n')
    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        line = line.strip()
        if line.startswith('{') and ':' in line:
            # Try to find the complete JSON block
            json_text = line

            # If it doesn't end with }, look for continuation
            if not line.endswith('}'):
                for j in range(i+1, min(i+10, len(lines))):
                    json_text += ' ' + lines[j].strip()
                    if lines[j].strip().endswith('}'):
                        break

            try:
                data = json.loads(json_text)
                if isinstance(data, dict):
                    return list(data.keys())
            except:
                try:
                    data = ast.literal_eval(json_text)
                    if isinstance(data, dict):
                        return list(data.keys())
                except:
                    continue

    # Method 2: Find JSON-like blocks anywhere in text (fallback)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                diagnoses.extend(list(data.keys()))
        except:
            try:
                data = ast.literal_eval(match)
                if isinstance(data, dict):
                    diagnoses.extend(list(data.keys()))
            except:
                continue

    return diagnoses

def _extract_medical_patterns(text: str) -> List[str]:
    """Extract diagnoses using domain-specific medical patterns - FIXED"""

    # EXACT medical condition names (no wildcards that grab sentences)
    medical_conditions = [
        # Rheumatologic (for current case)
        'psoriatic arthritis', 'rheumatoid arthritis', 'osteoarthritis',
        'ankylosing spondylitis', 'gout', 'pseudogout',

        # Renal
        'acute kidney injury', 'contrast-induced nephropathy', 'diabetic nephropathy',
        'cholesterol embolization', 'drug-induced interstitial nephritis',

        # Cardiovascular
        'myocardial infarction', 'heart failure', 'cardiomyopathy',
        'thrombotic thrombocytopenic purpura', 'thrombotic microangiopathy',

        # Respiratory
        'pneumonia', 'histoplasmosis', 'aspergillosis', 'tuberculosis',
        'community-acquired pneumonia', 'asthma exacerbation',

        # Hematologic
        'cryoglobulinemia', 'polycythemia vera',

        # Systemic
        'systemic lupus erythematosus', 'vasculitis', 'sarcoidosis'
    ]

    found_diagnoses = []
    text_lower = text.lower()

    for condition in medical_conditions:
        if condition in text_lower:
            # Proper title case
            diagnosis = ' '.join(word.capitalize() for word in condition.split())
            if diagnosis not in found_diagnoses:
                found_diagnoses.append(diagnosis)

    return found_diagnoses

# =============================================================================
# Response Validation Utilities
# =============================================================================

def validate_medical_response(response_text: str) -> Dict[str, Any]:
    """
    Validate if a response contains meaningful medical content

    Returns:
        Dict with validation metrics and extracted content
    """
    diagnoses = extract_diagnoses(response_text)

    # Count medical terms
    medical_keywords = [
        'symptom', 'diagnosis', 'patient', 'clinical', 'medical', 'disease',
        'syndrome', 'condition', 'treatment', 'evidence', 'findings'
    ]

    keyword_count = sum(1 for keyword in medical_keywords
                       if keyword in response_text.lower())

    return {
        'diagnoses_found': len(diagnoses),
        'diagnoses': diagnoses,
        'medical_keyword_count': keyword_count,
        'response_length': len(response_text),
        'is_medical_response': len(diagnoses) > 0 or keyword_count >= 3,
        'response_quality': 'high' if len(diagnoses) >= 3 and keyword_count >= 5 else
                          'medium' if len(diagnoses) >= 1 or keyword_count >= 3 else 'low'
    }

# =============================================================================
# Text Cleaning Utilities
# =============================================================================

def clean_response_text(text: str) -> str:
    """Clean response text by removing common model artifacts"""
    if not text:
        return ""

    # Remove common prefixes
    prefixes_to_remove = [
        "Here is my differential diagnosis:",
        "My differential diagnosis is:",
        "Based on the case, here are the diagnoses:",
        "The differential diagnosis includes:",
        "```json", "```python", "```"
    ]

    cleaned = text.strip()
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    # Remove trailing artifacts
    if "```" in cleaned:
        cleaned = cleaned.split("```")[0].strip()

    return cleaned

# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # Test the extractor
    test_cases = [
        '{"acute kidney injury": ["elevated creatinine"], "pneumonia": ["fever", "cough"]}',
        "1. Acute Kidney Injury\n2. Contrast Nephropathy\n3. Drug-induced nephritis",
        "The patient likely has acute kidney injury based on elevated creatinine and oliguria.",
        "Consider AKI, TTP, and CHF in this case."
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test}")
        diagnoses = extract_diagnoses(test)
        print(f"Extracted: {diagnoses}\n")
