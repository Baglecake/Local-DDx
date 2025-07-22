# DDx Experiment v8 - Multi-Model Diagnostic Reasoning System

**v8**
## Overview

This version implements a deterministic clinical evaluation engine that provides reproducible, research-grade assessment of diagnostic performance using rule-based clinical equivalence matching and evidence-based transcript analysis.

Additionally, this version implements a synchronous dual instance approach, loading each llm simultaneously. This refinement significantly improves speed of execution, while also enabling batched execution (See below).

---------------------------------------------------------------------

## Version History

-> **Current** - v8_6: Expanding on v8.5, this version introduces additional prompting to encourage interactivity amongst agents. Implements comorbidity considerations to encourage collaboration between specialists.
```markdown
🎯 ROUND 2: Devil's Advocate Assignment + Direct Challenges
📊 Consensus detected: Hepatic Encephalopathy
😈 Devil's Advocate assigned: Dr. Anya Sharma
📊 Interactive Debate Summary:
   🎯 Direct challenges: 9
   🔄 Position changes: 9
   💬 Evidence citations: 95
   📈 Interaction quality: 7.28
   🔗 Comorbidity discoveries: 9
   🤝 Comorbidity consensus: strong
   📊 Multi-condition likelihood: HIGH - Multiple agents identify concurrent conditions
```

-> v8_5: Building on 8.4 architecture, this version introduces significantly enriched prompting to encourage interactivity amongst agents. Implementation of a "devil's advocate" mechanism acts as a stress test for diagnostic convergence a role assigned to the most credible specialist. Additionally, discourse in the 'refinement and justification' and 'debate rounds' exhibits signs of emergent behaviour within a bounded environment, seen through small fluctuations in experimental outcomes over multiple runs of the same case.

-> v8_4: Synchronous gemma2-qwen2.5 architecture. Significant refactor of refinement and debate rounds to include direct challenges and responses between agents. Debate now directly informs preferential voting round. These additions were reintroduced from earlier iterations using the robust evaluation and metrics traslation of v8.

-> v8_3_gemma_qwen: This setup deploys synchronous Gemma2 and Qwen2.5 models to power its agents. This version also consolidates the evaluator into a single v9 edition. 

-> v8_2_dual_qwen: This setup deploys synchronous dual Qwen2.5 models to power its agents.

-> v8_2_llama_qwen: This setup deploys synchronus llama3 and Qwen2.5 models to power its agents.

-> ddx_results_analyzer.py: This script contains metric normalization analysis to interpret the outputs of the LDDx in comparison to industry and academic benchamrks. It serves as a translator between novel and traditional metrics. Use to interpret results. See the "Reports" directory for batched analyses and individual case reports.

---------------------------------------------------------------------

## Core Framework Review

**Multi-Model Conservative vs Innovative Reasoning**: The system employs two different LLMs with distinct temperature settings to create specialists with fundamentally different cognitive approaches:
- **Conservative Model**: Lower temperature (0.2-0.35) for systematic, evidence-based reasoning
- **Innovative Model**: Higher temperature (0.6-0.9) for creative, exploratory thinking

This approach enables novel epistemic labor division and collaborative diagnostic emergence.

---------------------------------------------------------------------

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
---------------------------------------------------------------------

## Module Documentation

---------------------------------------------------------------------

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
- **AE (Appropriately Excluded)**: Considered but reasonably ruled out (TP considered but not promoted in borda counts)
- **TM (True Miss)**: Never adequately considered (Absent from pipeline)
- **CAA (Clinically Appropriate Alternative)**: Defensible alternatives (Clinical equivalences not promoted through borda counts - uses AI enhanced evaluation)

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

---------------------------------------------------------------------

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

2. **Memory Allocation**: Adjust `memory_fraction` based on GPU:
   - 40GB A100: `0.48` (recommended)
   - 24GB RTX: `0.45`
   - 16GB cards: `0.40`

---------------------------------------------------------------------

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

---------------------------------------------------------------------

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

---------------------------------------------------------------------

## 📊 Performance Metrics

This project uses a combination of traditional ML metrics for benchmarking and a novel, clinically-focused framework for a more valid assessment of diagnostic performance. The translation between these frameworks is handled by the `ddx_results_analyzer.py` script.

### Clinically-Focused Metrics (Primary Evaluation)

Our primary evaluation framework re-categorizes diagnostic outcomes to better reflect real-world clinical competence.

* **Clinical Success Rate**: Measures the percentage of ground truth diagnoses that were correctly handled by the system, either by matching them or by correctly ruling them out with evidence.
    * `Formula: (TP + AE) / (TP + AE + TM)`

* **Clinical Failure Rate**: Measures the percentage of ground truth diagnoses that the system truly missed, representing actual diagnostic inadequacy.
    * `Formula: TM / (TP + AE + TM)`

* **Diagnostic Precision**: Measures the accuracy of the final diagnoses proposed by the system.
    * `Formula: TP / (TP + FP)`

* **Appropriately Excluded (AE)**: A success classification for diagnoses that were considered by the system but reasonably ruled out based on evidence in the transcript. This is a key measure of sound clinical reasoning.

### Traditional Metrics (For Comparison)

These standard metrics are generated for comparison against other benchmarks but are considered less representative of this system's true clinical performance.

* **Precision**: `TP / (TP + FP)`
* **Recall**: `TP / (TP + FN)` (Note: This formula incorrectly penalizes the system by treating Appropriately Excluded (AE) diagnoses as failures).
* **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`

### Advanced System Metrics

* **Reasoning Thoroughness**: Comprehensive assessment of diagnostic consideration based on evidence from the full transcript.
* **Diagnostic Safety**: A risk-adjusted performance metric that evaluates the system's ability to avoid critical misses.
* **TempoScore**: A novel metric that measures the complexity and intensity of the collaborative debate across rounds.
  
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

---------------------------------------------------------------------

## Known Limitations

1. **Model Dependency**: Performance varies with underlying LLM capabilities
2. **Memory Requirements**: Requires substantial VRAM for simultaneous model loading
3. **Processing Time**: Complex cases may take 5-10 minutes per analysis
4. **Medical Knowledge**: Limited by training data cutoffs and model medical knowledge

---------------------------------------------------------------------

**Version**: v8  
**Last Updated**: 2025-01-16  
**Compatibility**: Google Colab A100, vLLM 0.2+, CUDA 11.8+
