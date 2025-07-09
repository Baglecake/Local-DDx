# LDDx v6: Multi-Agent Collaborative Diagnostic System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Type-Research-brightgreen.svg)](https://github.com)

**A multi-agent collaborative AI system for medical differential diagnosis featuring dynamic specialist generation, sliding context windows, and epistemic labor division using local LLMs.**

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

### **Advanced Diagnostic Pipeline**
- **7-round diagnostic sequence** following evidence-based medical reasoning
- **Preferential voting with Borda counts** for consensus building
- **TempoScore metrics** for round-by-round performance assessment
- **AI-enhanced clinical evaluation** with equivalence agents

## 🏗️ Architecture

```
DDx v6 System Architecture

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

This system represents **significant advances** in multi-agent AI for healthcare:

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

### Configuration

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

## 🚀 Usage

### Basic Case Analysis

```python
from ddx_core_v6 import DDxSystem

# Initialize system
ddx = DDxSystem()
ddx.initialize()

# Define clinical case
case = """
A 45-year-old male presents with acute chest pain that began 2 hours ago.
The pain is crushing, radiates to the left arm, and is associated with
shortness of breath and diaphoresis. He has a history of diabetes and
hypertension. ECG shows ST elevation in leads II, III, aVF.
"""

# Generate dynamic specialist team
result = ddx.analyze_case(case, "chest_pain_case")
print(f"Generated {len(ddx.current_agents)} specialists")

# Run collaborative diagnostic sequence
round_results = ddx.run_complete_diagnostic_sequence()
print(f"Completed {len(round_results)} rounds with sliding context")
```

### Complete Pipeline with Evaluation

```python
from ddx_runner_v6 import run_complete_ddx_case

# Run full pipeline
results = run_complete_ddx_case(
    case_description=case,
    case_name="acute_mi_case",
    ground_truth_diagnoses=["acute myocardial infarction", "STEMI"],
    export_json=True
)

# Access results
print(f"Consensus diagnosis: {results['synthesis']['winner']}")
print(f"Performance score: {results['evaluation']['clinical_recall']:.2f}")
print(f"Collaboration score: {results['collaboration_metrics']['context_usage']:.2f}")
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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Development installation
git clone https://github.com/yourusername/ddx-v6.git
cd ddx-v6
pip install -e .[dev]

# Run tests
pytest tests/

# Code formatting
black ddx_*.py
```

## 📚 Citation

If you use DDx v6 in your research, please cite:

```bibtex
@software{ddx_v6_2025,
  title={DDx v6: Multi-Agent Collaborative Diagnostic System with Sliding Context Windows},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/ddx-v6},
  note={A sophisticated multi-agent AI system for medical differential diagnosis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **vLLM Team** for efficient LLM inference
- **Anthropic** for AI safety research that inspired collaborative agent design
- **Medical AI Research Community** for foundational work in clinical decision support

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ddx-v6/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ddx-v6/discussions)
- **Email**: your.email@domain.com

---

**⭐ Star this repository if DDx v6 helps your research or clinical work!**

*Built with ❤️ for advancing AI in healthcare*
