# LDDx: Multi-Agent Collaborative Diagnostic System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Type-Research-brightgreen.svg)](https://github.com)

**A multi-agent collaborative AI system for medical differential diagnosis featuring dynamic specialist generation, sliding context windows, and epistemic labor division using local LLMs.**

**v8:** This directory contains the latest version of the DDx. Primary refactor advances from sequential model management, implementing simultaneous model loading for persistent dual instance architecture. Features include precise string matching and backwards compatability for agent enhanced evaluation, as well as enhanced interaction through sliding context. 

- In addition to the features above, Version 8 offers a standardized benchmark to test against the AI enhanced evaluation in the previous iterations using the OPEN-DDx Data set. While a rigorous diagnostic rubric captures the structure of an academic test, the LDDx demonstrates the social dynamics at play in real, complex situations. As a side by side experiment, the LDDx offers a glimpse into the benefits and risks of artificial social intelligence as they apply to the sensitive context of clinical reasoning and diagnostics. See the version 8 README for further details.
  
- Current testing uses synchronous Qwen2.5-7B-Instruct-GPTQ-Int4 models. This specific model was chosen for initial version deployment given its tendency towards strict instruction adherence and capacity to handle extended text.

-> **v8.1:** Implements master key vulnerability checks in line with methods detailed by Zhao et al. (2025). This version hardens AI enhanced evaluation to prevent "hacking" and lenient assessment. Maintains compatability with earlier configs and robust synchronous operation.

**LEGACY VERSION OVERVIEW:**

> Each version directory contains version transcripts and modules. "Transcripts" and "Modules" directories in the main repo contain modules and transcripts from the initial conceptual design (v6.0) only.

**v6:** Developmental Version - Eager: "true"; enforced stability for GPU development purposes. Initial dual model scaffolding with agent generation capabilities and early insights using dual NousResearch/Hermes-2-Pro-Mistral-7B models. Partially compatable with vllm library (missing quantized loading).

**v7:** Fixed metrics and parsing. Multiple operational versions with a variety of configs:

-> **v7.1:** Developmental version - Eager: "true"; enforced stability for GPU development purposes. stable with Hugging Face FSmolLM3 and Microsoft Phi3mini, dual Mistral (v6 config). Partially compatable with vllm library (missing quantized loading).

-> **v7.2:** Production version - Eager: "false"; production ready platform for optimized Hugging Face FSmolLM3 and Microsoft Phi3mini experiments. Compatable with full vllm library.

-> **v7.3:** Production version - Eager: "false"; dual Lamma3 models loaded with v7.2 architecture. This version demonstrates functionality with quantized models. Compatable with full vllm library.

-> **v7.4:** Production version - Eager: "false"; v7.4 architecture with flexible parsing methods for various output formats. This version supports quantized models. Compatable with full vllm library.

-> **v7.5** Production version - Eager: "false"; NEW v7.5 architecture with enhanced parsing, prompt control, and "chattyness" reduction. Supports quantized models with full vllm compatability.

-> **v7.6** Production version - Easger "false"; loaded on v7.5 architecture. This version represents the first iteration mixing quantized and full precision models (Meta-Llama-3-8B-Instruct-GPTQ & Qwen/Qwen2-7B-Instruct). Full vllm compatability.

-> **v7.7** Production version - Easger "false"; NEW v7.7 architecture with a refined evaluation pipeline for precise ground truth matching. Tested on dual instance Qwen2.5-7B-Instruct-GPTQ-Int4. Maintains full vllm compatability.

**VERSIONS 7.2 onwards support interchangeable models from the vllm libray which are loaded in the config.yaml.


**TESTED COMPATABLE MODELS:**

1. Hermes-2-Pro-Mistral-7B * 
2. Nous-Hermes-2-Mistral-7B-DPO *
3. Hugging Face SmolLM3-3B
4. Microsoft Phi-3-mini-4k-instruct
5. Microsoft Phi-4-mini-reasoning
6. Meta-Llama-3-8B-Instruct-GPTQ **
7. Microsoft Diablo-GPT
8. Qwen/Qwen2-7B-Instruct
9. Qwen2.5-7B-Instruct-GPTQ-Int4 **

>*Testing on larger models limited to dual instance approach (v7.3 onwards).

>**Quantized model

## 🚀 Key Innovations

### **Dynamic Specialist Generation**
- **Autonomous agent creation** tailored to each specific case
- **Unlimited specialty diversity** - not constrained by predefined lists
- **AI-driven team composition** based on case complexity and requirements
- **Dual model architecture** (conservative + innovative) for balanced perspectives

### **Sliding Context Windows**
- **True collaborative reasoning** - agents see and respond to each other's insights
- **Context-aware discourse** - later rounds build upon previous team discussions
- **Epistemic labor division** - specialists challenge, support, and synthesize across rounds
- **Intelligent context filtering** - relevant information prioritized by round type and agent specialty

### **Diagnostic Pipeline**
- **7-round diagnostic sequence** following evidence-based medical reasoning
- **Preferential voting with Borda counts** for consensus building
- **TempoScore metrics** for round-by-round performance assessment
- **AI-enhanced clinical evaluation** with equivalence agents

## 🏗️ Architecture

```
DDx System Architecture

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ddx_core_v6   │    │  ddx_rounds_v6  │    │ ddx_synthesis   │
│                 │    │                 │    │                 │
│ • ModelManager  │────│ • 7 Round Types │────│ • TempoScore    │
│ • Agent Gen     │    │ • Sliding Ctx   │    │ • Credibility   │
│ • Dynamic Team  │    │ • Collaboration │    │ • Dr. Reed      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐    ┌─────────────────┐
                    │ ddx_evaluator   │    │   ddx_runner    │
                    │                 │    │                 │
                    │ • AI Matching   │    │ • Pipeline Orch │
                    │ • Clinical Eval │    │ • JSON Export   │
                    │ • v6 Metrics    │    │ • Full Results  │
                    └─────────────────┘    └─────────────────┘
```

## 🔬 Research Significance

This system proposes a multi-agent AI framework for healthcare:

- **First implementation** of sliding context windows in medical AI collaboration
- **Novel approach** to dynamic specialist team generation without constraint to predefined specialties  
- **Advanced consensus mechanisms** using preferential voting and credibility weighting
- **Comprehensive evaluation framework** with AI-enhanced clinical assessment

## 📋 The 7-Round Diagnostic Process

| Round | Purpose | Collaboration Level |
|-------|---------|-------------------|
| **1. Specialized Ranking** | Rank specialist relevance | Individual |
| **2. Symptom Management** | Immediate triage interventions | Individual |
| **3. Team Differentials** | Independent diagnosis generation | **Context-Aware** |
| **4. Master List** | Synthesis of all team diagnoses | **Context-Aware** |
| **5. Refinement & Debate** | Evidence-based collaborative discourse | **Fully Collaborative** |
| **6. Preferential Voting** | Consensus via Borda count voting | **Fully Collaborative** |
| **7. Can't Miss** | Critical diagnosis identification | **Context-Aware** |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: A100, V100, or RTX 4090)
- 16GB+ GPU memory for optimal performance

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/ddx-v6.git
cd ddx-v6

# Install dependencies
pip install vllm torch transformers pyyaml

# Configure models in config.yaml
# Run your first case
python run_example.py
```

### Configuration (v6) - ***See module READMEs for version specs***

Create `config.yaml`:

```yaml
conservative_model:
  name: 'Conservative-Model'
  model_path: 'NousResearch/Nous-Hermes-2-Mistral-7B-DPO'
  memory_fraction: 0.4
  temperature: 0.1

innovative_model:
  name: 'Innovative-Model'  
  model_path: 'NousResearch/Nous-Hermes-2-Mistral-7B-DPO'
  memory_fraction: 0.4
  temperature: 0.9
```
*Adjust config parameters for different reasoning outcomes. Allows for direct insight into the effects of architectural parameter tuning.

## 🚀 Usage

### Basic Case Analysis

```python
from ddx_core_v6 import DDxSystem

# Initialize system
ddx = DDxSystem()
ddx.initialize()

# Define clinical case
case = """
Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus and osteoarthritis of the hips. Prior to admission, his medications were insulin and naproxen. He was also started on aspirin, clopidogrel, and metoprolol after the coronary intervention. His temperature is 38 °C (100.4 °F), pulse is 93/min, and blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet. Laboratory studies show: Hemoglobin count 14 g/dL Leukocyte count 16,400/mm3 Segmented neutrophils 56% Eosinophils 11% Lymphocytes 31% Monocytes 2% Platelet count 260,000/mm3 Erythrocyte sedimentation rate 68 mm/h Serum Urea nitrogen 25 mg/dL Creatinine 4.2 mg/dL Renal biopsy shows intravascular spindle-shaped vacuoles. What is the most likely cause of this patient's symptoms?
"""

# Generate dynamic specialist team
result = ddx.analyze_case(case, "chest_pain_case")
print(f"Generated {len(ddx.current_agents)} specialists")

# Run collaborative diagnostic sequence
round_results = ddx.run_complete_diagnostic_sequence()
print(f"Completed {len(round_results)} rounds with sliding context")
```

### Complete Pipeline with Evaluation (Example, Case 1)

```python
from ddx_runner_v6 import DDxRunner

# 1. Define your case name, description, and ground truth
case_name = "Case 1"

case_description = """
Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus and osteoarthritis of the hips. Prior to admission, his medications were insulin and naproxen. He was also started on aspirin, clopidogrel, and metoprolol after the coronary intervention. His temperature is 38 °C (100.4 °F), pulse is 93/min, and blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet. Laboratory studies show: Hemoglobin count 14 g/dL Leukocyte count 16,400/mm3 Segmented neutrophils 56% Eosinophils 11% Lymphocytes 31% Monocytes 2% Platelet count 260,000/mm3 Erythrocyte sedimentation rate 68 mm/h Serum Urea nitrogen 25 mg/dL Creatinine 4.2 mg/dL Renal biopsy shows intravascular spindle-shaped vacuoles. What is the most likely cause of this patient's symptoms?
"""

ground_truth = {'acute kidney injury (AKI)': ['decreased urinary output', 'malaise', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL'], 'contrast-induced nephropathy': ['Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL'], 'drug-induced interstitial nephritis': ['decreased urinary output and malaise', 'eosinophils 11%', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL', 'Renal biopsy shows intravascular spindle-shaped vacuoles'], 'cholesterol embolization': ['mottled, reticulated purplish discoloration of the feet', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL', 'intravascular spindle-shaped vacuoles'], 'diabetic nephropathy': ['decreased urinary output', 'Serum Urea nitrogen 25 mg/dL', 'Creatinine 4.2 mg/dL']}


# 2. Create an instance of the runner and initialize the system
runner = DDxRunner()
runner.initialize_system()

# 3. Run the complete case with the new, single method
if runner.is_initialized:
    runner.run_case(
        case_name=case_name,
        case_description=case_description,
        ground_truth=ground_truth
    )
```

## 📊 Output & Results

### JSON Export Structure

```json
{
  "case_metadata": {
    "case_name": "acute_mi_case",
    "timestamp": "2025-07-09T20:30:00Z",
    "total_execution_time": 420.5
  },
  "team_composition": {
    "total_agents": 10,
    "dynamic_generation": true,
    "specialties": ["Cardiology", "Emergency Medicine", ...]
  },
  "diagnostic_sequence": {
    "rounds_completed": 7,
    "sliding_context_enabled": true,
    "collaboration_detected": true
  },
  "synthesis_results": {
    "winner": "Acute myocardial infarction (STEMI)",
    "tempo_scores": {...},
    "credibility_weighting": {...}
  },
  "evaluation_metrics": {
    "clinical_recall": 0.95,
    "clinical_precision": 0.88,
    "ai_enhanced_matching": true
  },
  "full_transcript": {
    "rounds": {...},
    "collaboration_evidence": [...]
  }
}
```

### Performance Metrics

- **Clinical Recall**: Percentage of ground truth diagnoses identified
- **Clinical Precision**: Accuracy of team diagnoses
- **TempoScore**: Round-by-round performance assessment
- **Collaboration Index**: Measure of inter-agent discourse quality
- **Context Utilization**: Effectiveness of sliding context window usage

## 🔧 Core Modules

### `ddx_core_v6.py`
- **ModelManager**: Dual-model orchestration with memory management
- **DynamicAgentGenerator**: AI-driven specialist team creation
- **DDxAgent**: Individual agent with collaborative capabilities
- **AgentConfig & AgentResponse**: Enhanced data structures

### `ddx_rounds_v6.py`
- **RoundOrchestrator**: Complete diagnostic sequence management
- **BaseRound**: Abstract framework for all diagnostic rounds
- **7 Specialized Rounds**: Each with unique collaborative requirements
- **Sliding Context Integration**: Context passing and transcript management

### `ddx_synthesis.py`
- **TempoScoreCalculator**: Advanced round performance metrics
- **CredibilityWeighting**: Evidence-based consensus scoring  
- **DrReedAssessment**: Specialist performance evaluation
- **PreferentialVoting**: Borda count consensus mechanism

### `ddx_evaluator_v6.py`
- **ClinicalEquivalenceAgent**: AI-powered diagnosis matching
- **EnhancedEvaluator**: Comprehensive evaluation framework
- **v6 Metrics**: Advanced performance assessment
- **Context-Informed Analysis**: Transcript-aware evaluation

### `ddx_sliding_context.py`
- **SlidingContextManager**: Intelligent context filtering and delivery
- **ContextEntry**: Structured discourse representation
- **Collaboration Filters**: Specialty and round-aware context selection
- **Attention Guidance**: Round-specific collaboration prompting

**v7 repos are backwards compatable. v6 repos are outdated but functional.

## 🧪 Testing & Validation

```bash
# Run system tests
python test_sliding_context.py

# Validate complete pipeline
python validate_pipeline.py

# Performance benchmarking
python benchmark_system.py
```

## 📈 Research Applications

### Medical Education
- **Case-based learning** with multi-perspective analysis
- **Diagnostic reasoning training** through collaborative AI examples
- **Specialty interaction modeling** for interdisciplinary education

### Clinical Decision Support
- **Complex case consultation** with diverse specialist input
- **Consensus building** for difficult diagnoses
- **Evidence synthesis** across multiple clinical perspectives

### AI Research
- **Multi-agent collaboration** in knowledge-intensive domains
- **Context window optimization** for long-form reasoning
- **Epistemic labor division** in AI systems


### Development Setup

```bash
# Development installation
git clone https://github.com/baglecake/ddx-v6.git
cd ddx-v6
pip install -e .[dev]

# Run tests
pytest tests/

# Code formatting
black ddx_*.py
```

### 📚 References
Nori, H., Lee, Y. T., Zhang, S., Carignan, D., Edgar, R., Fusi, N., ... & Horvitz, E. (2023). Can generalist foundation models outcompete special-purpose tuning? A case study in medicine. arXiv. https://arxiv.org/abs/2311.16452

Zhou, S., Lin, M., Ding, S., Zhang, Y., Chen, H., Wang, J., & Liu, Q. (2025). Explainable differential diagnosis with dual-inference large language models. npj Health Systems, 2(1), 12. https://doi.org/10.1038/s44401-025-00015-6

Santoro, A., Lampinen, A., Mathewson, K. W., Lillicrap, T., & Raposo, D. (2022). Symbolic behaviour in artificial intelligence (arXiv Version 2). arXiv. https://doi.org/10.48550/arXiv.2102.03406

Zhao, Y., Liu, H., Yu, D., Kung, S. Y., Mi, H., & Yu, D. (2025). *One token to fool LLM-as-a-judge*. arXiv. https://arxiv.org/abs/2507.08794

## 📚 Citation

If you use LDDx in your research, please cite:

```bibtex
@software{ddx_v6_2025,
  title={DDx v6: Multi-Agent Collaborative Diagnostic System with Sliding Context Windows},
  author={[Silver, Daniel; Fosse, Ethan; Griggs, Brandon; Coburn, Del]},
  year={2025},
  url={https://github.com/yourusername/ddx-v6},
  note={A multi-agent AI system for medical differential diagnosis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


