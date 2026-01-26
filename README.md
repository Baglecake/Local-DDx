# Local-DDx: Multi-Agent Collaborative Diagnostic System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Local-DDx is a multi-agent AI framework for medical differential diagnosis that addresses diagnostic isolation in resource-limited healthcare settings. The system dynamically generates case-specific specialist teams that collaborate through structured diagnostic rounds, employing sliding context windows to enable genuine epistemic labor division among AI agents.

Unlike single-model approaches, Local-DDx models the collaborative reasoning process of a multidisciplinary medical team—specialists independently analyze cases, engage in evidence-based debate, challenge consensus through a Devil's Advocate mechanism, and synthesize diagnoses through preferential voting. All inference runs locally, ensuring patient data never leaves the clinical environment.

## Key Innovations

### Dynamic Specialist Generation
- **Autonomous agent creation** tailored to each clinical presentation
- **Unlimited specialty diversity** unconstrained by predefined specialty lists
- **Dual model architecture** (conservative + innovative) for balanced diagnostic reasoning

### Sliding Context Windows
- **True collaborative reasoning** where agents see and respond to each other's insights
- **Context-aware discourse** with later rounds building upon team discussions
- **Intelligent context filtering** prioritized by round type and agent specialty

### Structured Diagnostic Pipeline
- **7-round diagnostic sequence** following evidence-based medical reasoning patterns
- **Devil's Advocate mechanism** for stress-testing diagnostic convergence
- **Preferential voting with Borda counts** for consensus building
- **TempoScore metrics** for round-by-round performance assessment

## The 7-Round Diagnostic Process

| Round | Purpose | Collaboration Level |
|-------|---------|---------------------|
| 1. Specialized Ranking | Rank specialist relevance to case | Individual |
| 2. Symptom Management | Immediate triage interventions | Individual |
| 3. Team Differentials | Independent diagnosis generation | Context-Aware |
| 4. Master List | Synthesis of team diagnoses | Context-Aware |
| 5. Refinement & Debate | Evidence-based collaborative discourse | Fully Collaborative |
| 6. Preferential Voting | Consensus via Borda count | Fully Collaborative |
| 7. Can't Miss | Critical diagnosis identification | Context-Aware |

## Architecture

```
Local-DDx System Architecture

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    ddx_core     │    │   ddx_rounds    │    │  ddx_synthesis  │
│                 │    │                 │    │                 │
│ • ModelManager  │────│ • 7 Round Types │────│ • TempoScore    │
│ • Agent Gen     │    │ • Sliding Ctx   │    │ • Credibility   │
│ • Dynamic Team  │    │ • Collaboration │    │ • Dr. Reed      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐    ┌─────────────────┐
                    │  ddx_evaluator  │    │   ddx_runner    │
                    │                 │    │                 │
                    │ • AI Matching   │    │ • Pipeline Orch │
                    │ • Clinical Eval │    │ • JSON Export   │
                    │ • Metrics       │    │ • Full Results  │
                    └─────────────────┘    └─────────────────┘
```

## Installation

### Option A: Full System (CUDA GPU)

For the complete 7-round pipeline with all features:

```bash
git clone https://github.com/yourusername/Local-DDx.git
cd Local-DDx

# Install dependencies
pip install vllm torch transformers pyyaml

# Configure models in v8/config.yaml
cd v8
python ddx_runner.py
```

**Requirements:** CUDA-compatible GPU with 40GB+ VRAM (A100, V100, RTX 4090)

### Option B: Demonstration Version (Apple Silicon)

For a lightweight proof-of-concept on M-series Macs:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b

cd v9_ollama_ui
pip install -r requirements.txt
python app.py
```

**Requirements:** macOS with Apple Silicon, 16GB+ unified memory

See [v9_ollama_ui/README.md](v9_ollama_ui/README.md) for details.

## Usage

### Basic Case Analysis

```python
from ddx_core import DDxSystem

# Initialize system
ddx = DDxSystem()
ddx.initialize()

# Define clinical case
case = """
A 61-year-old man presents two weeks after emergency cardiac
catheterization with decreased urinary output and malaise.
Examination shows mottled, reticulated purplish discoloration
of the feet. Labs show elevated creatinine (4.2 mg/dL) and
eosinophilia (11%). Renal biopsy shows intravascular
spindle-shaped vacuoles.
"""

# Generate specialist team and run diagnosis
result = ddx.analyze_case(case, "renal_case")
round_results = ddx.run_complete_diagnostic_sequence()
```

### Complete Pipeline with Evaluation

```python
from ddx_runner import DDxRunner

runner = DDxRunner()
runner.initialize_system()

runner.run_case(
    case_name="Case 1",
    case_description=case_description,
    ground_truth=ground_truth_dict
)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Clinical Recall | Percentage of ground truth diagnoses identified |
| Clinical Precision | Accuracy of team diagnoses |
| TempoScore | Round-by-round performance assessment |
| Collaboration Index | Measure of inter-agent discourse quality |
| Context Utilization | Effectiveness of sliding context window usage |

## Research Applications

### Medical Education
- Case-based learning with multi-perspective analysis
- Diagnostic reasoning training through collaborative AI demonstration
- Specialty interaction modeling for interdisciplinary education

### Clinical Decision Support
- Complex case consultation with diverse specialist perspectives
- Consensus building for challenging diagnoses
- Evidence synthesis across clinical viewpoints

### AI Research
- Multi-agent collaboration in knowledge-intensive domains
- Context window optimization for long-form reasoning
- Epistemic labor division in AI systems

## Tested Models

| Model | Type | Notes |
|-------|------|-------|
| Qwen2.5-7B-Instruct-GPTQ-Int4 | Quantized | Primary development model |
| Gemma-2-9b-it-AWQ-INT4 | Quantized | Production conservative model |
| Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 | Quantized | Validated |
| llama3.1:8b (Ollama) | Full | v9 demonstration |

## Project Structure

```
Local-DDx/
├── v8/                    # Full production system (vLLM/CUDA)
├── v9_ollama_ui/          # Demonstration version (Ollama/Apple Silicon)
├── Modules/               # Core modules (v6 reference)
├── Transcripts/           # Example diagnostic transcripts
├── METRICS_GUIDE.md       # Detailed metrics documentation
└── CHANGELOG.md           # Version history
```

## References

Nori, H., Lee, Y. T., Zhang, S., et al. (2023). Can generalist foundation models outcompete special-purpose tuning? A case study in medicine. *arXiv*. https://arxiv.org/abs/2311.16452

Zhou, S., Lin, M., Ding, S., et al. (2025). Explainable differential diagnosis with dual-inference large language models. *npj Health Systems*, 2(1), 12. https://doi.org/10.1038/s44401-025-00015-6

Zhao, Y., Liu, H., Yu, D., et al. (2025). One token to fool LLM-as-a-judge. *arXiv*. https://arxiv.org/abs/2507.08794

## Citation

```bibtex
@software{local_ddx_2025,
  title={Local-DDx: Multi-Agent Collaborative Diagnostic System
         with Sliding Context Windows},
  author={Silver, Daniel and Fosse, Ethan and Griggs, Brandon and Coburn, Del},
  year={2025},
  url={https://github.com/yourusername/Local-DDx}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Appendix: Version History

<details>
<summary>Click to expand version history</summary>

### v9 (Current)
- Ollama backend for Apple Silicon compatibility
- Gradio web interface for demonstrations
- Streamlined 3-round POC implementation

### v8
- Production system with full 7-round pipeline
- Sliding context windows for collaborative reasoning
- Devil's Advocate mechanism for diagnostic stress-testing
- TempoScore and credibility-weighted voting
- Synchronous dual-model architecture (Gemma-2 + Qwen2.5)

### v7
- Multiple model configurations validated
- Enhanced parsing and prompt control
- Full vLLM compatibility with quantized models

### v6
- Initial dual-model scaffolding
- Dynamic agent generation capabilities
- Foundation architecture established

</details>
