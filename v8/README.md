# DDx Experiment v8 - Multi-Model Diagnostic Reasoning System

## Overview

This version implements a deterministic clinical evaluation engine that provides reproducible, research-grade assessment of diagnostic performance using rule-based clinical equivalence matching and evidence-based transcript analysis.

Additionally, this version implements a unified dual instance approach, loading each llm simultaneously. This refinement significantly improves speed of execution, while also enabling batched execution (See below).

## Core Framework Review

**Multi-Model Conservative vs Innovative Reasoning**: The system employs two different LLMs with distinct temperature settings to create specialists with fundamentally different cognitive approaches:
- **Conservative Model**: Lower temperature (0.2-0.35) for systematic, evidence-based reasoning
- **Innovative Model**: Higher temperature (0.6-0.9) for creative, exploratory thinking

This approach enables novel epistemic labor division and collaborative diagnostic emergence.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DDx Runner    │────│   DDx Core v6    │────│  Model Manager  │
│   (Pipeline)    │    │   (Orchestrator) │    │ (Multi-Model)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        │
         │              ┌──────────────────┐               │
         │              │   DDx Rounds v6  │               │
         │              │  (7 Round Types) │               │
         │              └──────────────────┘               │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ DDx Synthesis   │────│ DDx Evaluator v9 │    │ Sliding Context │
│ (TempoScore +   │    │ (Deterministic)  │    │ (Collaboration) │
│  Dr. Reed)      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Module Documentation

### 1. `ddx_core_v6.py` - System Foundation
**Purpose**: Core system architecture with multi-model management and dynamic agent generation.

**Key Classes**:
- `ModelManager`: Handles simultaneous loading of multiple quantized models
- `DDxAgent`: Individual specialist agents with model-specific reasoning styles  
- `DynamicAgentGenerator`: Creates specialist teams dynamically based on case complexity
- `DDxSystem`: Main system orchestrator

**Features**:
- Simultaneous multi-model loading (Conservative + Innovative)
- Dynamic specialty generation (can create ANY medical specialty needed)
- Model-specific temperature and sampling parameter management
- GPU memory optimization for 40GB A100 environments

### 2. `ddx_rounds_v6.py` - Diagnostic Workflow
**Purpose**: Complete implementation of the 7-round diagnostic sequence.

**Round Types**:
1. **Specialized Ranking**: Prioritizes medical specialties by case relevance
2. **Symptom Management**: Immediate triage and stabilization protocols
3. **Team Independent Differentials**: Each specialist provides independent diagnosis
4. **Master List Generation**: Consolidates and deduplicates all proposed diagnoses
5. **Refinement and Justification**: Structured debate with tier placement
6. **Post-Debate Voting**: Preferential voting with credibility weighting
7. **Can't Miss**: Safety check for critical diagnoses

**Features**:
- Sliding context windows for agent collaboration
- Sophisticated JSON extraction from LLM responses
- Automatic consolidation logic based on TempoScore
- Comprehensive transcript generation

### 3. `ddx_synthesis_v6.py` - Advanced Analytics
**Purpose**: Synthesizes final diagnoses using credibility-weighted voting and advanced scoring.

**Key Components**:
- **TempoScore Calculator**: Measures diagnostic complexity and debate intensity
- **Dr. Reed Assessment**: Evaluates individual specialist performance (0-120 scale)
- **Diagnosis Synthesizer**: Applies credibility-weighted preferential voting
- **Dynamic Consolidation**: Semantic merging based on medical relationships

**Metrics**:
- Symbolic Curvature for debate quality measurement
- Professional Valence scoring for epistemic conduct
- Credibility-weighted Borda count voting
- Intelligent diagnosis consolidation with medical knowledge

### 4. `ddx_evaluator_v9.py` - Deterministic Evaluation
**Purpose**: Research-grade evaluation using deterministic clinical equivalence rules.

**Features**:
- **Clinical Equivalence Engine**: 100+ medical synonyms and hierarchies
- **Transcript Search**: Evidence-based classification of missed diagnoses
- **Dynamic CAA Weighting**: Prevents gaming through adaptive scoring
- **Audit Trails**: Complete transparency for research validation

**Classification Types**:
- **TP (True Positive)**: Exact clinical matches (Ground truth semantic match)
- **FP (False Positive)**: Unjustified diagnoses (Ungrounded diagnoses promoted through borda scoring)
- **AE (Appropriately Excluded)**: Considered but reasonably ruled out (Considered but not included in borda counts)
- **TM (True Miss)**: Never adequately considered (Absent from pipeline)
- **CAA (Clinically Appropriate Alternative)**: Defensible alternatives (TP that were not promoted through borda counts)

### 5. `ddx_runner_v6.py` - Pipeline Orchestrator
**Purpose**: Complete end-to-end pipeline for processing diagnostic cases.

**Capabilities**:
- Single case execution with comprehensive analysis
- Batch processing for research datasets (350+ cases)
- JSON export for statistical analysis
- Performance monitoring and error handling
- Multi-model lifecycle management

### 6. `ddx_utils.py` - Robust Utilities
**Purpose**: Centralized utilities for diagnosis extraction and validation.

**Functions**:
- `extract_diagnoses()`: Universal diagnosis extraction from any text format
- `validate_medical_response()`: Quality assessment of LLM responses
- `clean_response_text()`: Removes model artifacts and formatting issues

### 7. `ddx_sliding_context.py` - Agent Collaboration
**Purpose**: Enables sophisticated agent-to-agent interaction through contextual awareness.

**Features**:
- Context filtering by relevance, confidence, and opposition
- Attention guidance for different round types
- Memory management for computational efficiency
- Collaborative emergence through shared reasoning

### 8. `config.yaml` - System Configuration
**Purpose**: Centralized configuration for models, rounds, and evaluation parameters.

**Sections**:
- Model definitions (paths, quantization, sampling parameters)
- Round enablement flags
- Evaluation thresholds
- Synthesis configuration options

## Setup Instructions

### Prerequisites
- Google Colab with A100 GPU (40GB VRAM recommended)
- Python 3.8+
- CUDA-compatible environment

### Installation
```bash
# Install required packages
pip install vllm torch transformers

# Clone/download the DDx modules
# Ensure all .py files are in the same directory as config.yaml
```

### Configuration
1. **Model Setup**: Update `config.yaml` with your preferred model paths:
   ```yaml
   conservative_model:
     model_path: 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4'
     temperature: 0.2
   
   innovative_model:
     model_path: 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4'  
     temperature: 0.8
   ```

2. **Memory Allocation**: Adjust `memory_fraction` based on your GPU:
   - 40GB A100: `0.48` (recommended)
   - 24GB RTX: `0.45`
   - 16GB cards: `0.40`

## Usage Examples

### Single Case Execution
```python
from ddx_runner_v6 import DDxRunner

# Initialize system
runner = DDxRunner()
runner.initialize_system()

# Define case
case_description = """
61-year-old man with decreased urinary output and malaise 
two weeks after cardiac catheterization. History of diabetes.
Creatinine 4.2 mg/dL, mottled feet discoloration.
"""

ground_truth = {
    'acute kidney injury': ['decreased output', 'elevated creatinine'],
    'cholesterol embolization': ['mottled feet', 'post-catheterization']
}

# Run complete analysis
result = runner.run_case("Test Case", case_description, ground_truth)
```

### Batch Processing
```python
# Process multiple cases
cases = [
    {
        'name': 'Case 1',
        'description': '...',
        'ground_truth': {...}
    },
    # ... more cases
]

results = run_case_batch(cases)
```

## Performance Metrics

### Traditional Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

### Enhanced Metrics (Novel)
- **Clinical Reasoning Quality**: Weighted assessment including AE diagnoses
- **Diagnostic Safety**: Risk-adjusted performance measurement
- **Reasoning Thoroughness**: Comprehensive consideration assessment
- **TempoScore**: Debate complexity and interaction quality

### Dynamic Weighting
CAA diagnoses receive adaptive weights based on TP rate:
- TP Rate ≥ 70%: CAA Weight = +0.7 (bonus)
- TP Rate ≥ 50%: CAA Weight = +0.3 (mild bonus)  
- TP Rate ≥ 30%: CAA Weight = 0.0 (neutral)
- TP Rate < 30%: CAA Weight = -0.2 (penalty)

## Research Methodology

### Experimental Design
1. **Multi-Model Innovation**: Conservative vs Innovative reasoning styles
2. **Deterministic Evaluation**: Reproducible results for research validation  
3. **Comprehensive Metrics**: Beyond binary correct/incorrect assessment
4. **Transcript Analysis**: Evidence-based classification of diagnostic reasoning

### Data Export
Results are exported in multiple formats:
- **JSON**: Complete structured data for analysis
- **CSV**: Metrics for statistical processing  
- **Audit Trails**: Full transparency for research validation

### Validation Approach
- Deterministic clinical equivalence rules
- Evidence-based transcript analysis
- Dynamic weighting to prevent gaming
- Comprehensive audit trails

## Known Limitations

1. **Model Dependency**: Performance varies with underlying LLM capabilities
2. **Memory Requirements**: Requires substantial VRAM for simultaneous model loading
3. **Processing Time**: Complex cases may take 5-10 minutes per analysis
4. **Medical Knowledge**: Limited by training data cutoffs and model medical knowledge

## Future Enhancements

- **Specialist Memory**: Persistent learning across cases
- **Dynamic Team Sizing**: Adaptive specialist count based on case complexity
- **Real-time Collaboration**: Live agent-to-agent interaction
- **Medical Knowledge Integration**: External medical database connections


---

**Version**: v8  
**Last Updated**: 2025-01-16  
**Compatibility**: Google Colab A100, vLLM 0.2+, CUDA 11.8+
