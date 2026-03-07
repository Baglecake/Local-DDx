# Local-DDx: Multi-Agent Collaborative Diagnostic System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University of Toronto](https://img.shields.io/badge/University%20of%20Toronto-012A5C?style=flat&logoColor=white)](https://www.utoronto.ca)
[![HSIL Hackathon 2026](https://img.shields.io/badge/HSIL%20Hackathon-2026-green.svg)](https://healthsciencesinnovation.ca)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-white.svg)](https://ollama.ai)

## Abstract

Local-DDx is a multi-agent AI framework for medical differential diagnosis designed for **diagnostic isolation in rural and northern healthcare settings**. The system dynamically generates case-specific specialist teams that collaborate through structured diagnostic rounds, employing sliding context windows to enable genuine epistemic labor division among AI agents. All inference runs locally, ensuring patient data never leaves the clinical environment.

Unlike single-model approaches, Local-DDx models the collaborative reasoning process of a multidisciplinary medical team: specialists independently analyze cases, engage in evidence-based debate, challenge consensus through a Devil's Advocate mechanism, and synthesize diagnoses through credibility-weighted preferential voting.

Developed for the **HSIL Hackathon 2026**: *Building High-Value Health Systems: Leveraging AI*.

## Theoretical Grounding

Local-DDx draws on sociological theory to structure agent interaction. Goffman's (1956) distinction between *demeanor* and *deference* maps directly onto the system's architecture: agents demonstrate competence through reasoning quality (demeanor), which the credibility scoring module translates into weighted voting authority (deference). Sacks, Schegloff, and Jefferson's (1974) turn-taking organization structures the 7-round pipeline, ensuring each agent contributes within a defined sequential order. Wu et al. (2025) provide empirical grounding for multi-agent social capability in LLM systems, demonstrating that structured interaction protocols elicit collaborative behaviors absent in single-agent configurations.

## Key Features

- **Dynamic specialist generation:** autonomous agent creation tailored to each clinical presentation
- **7-round diagnostic pipeline:** structured sequence following evidence-based medical reasoning patterns
- **Sliding context windows:** 5 filter types enabling true collaborative reasoning across rounds
- **Credibility-weighted Borda count voting:** Dr. Reed assessment of specialist credibility informs consensus
- **Devil's Advocate mechanism:** stress-testing diagnostic convergence through systematic challenge
- **Dual-model architecture:** conservative (low temp) + innovative (high temp) for balanced reasoning
- **Fully local inference:** all patient data stays on-premises via Ollama

<img width="1691" height="949" alt="image" src="https://github.com/user-attachments/assets/2a6aecbb-580c-4351-9ac1-e480aff4015a" />

## The 7-Round Diagnostic Process

| Round | Purpose | Collaboration Level |
|-------|---------|---------------------|
| 1. Specialized Ranking | Rank specialist relevance to case | Individual |
| 2. Symptom Management | Immediate triage interventions | Individual |
| 3. Team Differentials | Independent diagnosis generation | Context-Aware |
| 4. Master List | Synthesis of team diagnoses | Context-Aware |
| 5. Refinement & Debate | Evidence-based collaborative discourse (3 sub-rounds) | Fully Collaborative |
| 6. Preferential Voting | Consensus via credibility-weighted Borda count | Fully Collaborative |
| 7. Can't Miss | Critical diagnosis identification | Context-Aware |

## Benchmark Results

Evaluated on the [Open-XDDx dataset](https://doi.org/10.1038/s44401-025-00015-6) (570 clinical vignettes, 9 specialties). See [benchmark/README.md](benchmark/README.md) for methodology.

| System | Model | Cases | Clinical Recall |
|--------|-------|-------|-----------------|
| **Local-DDx v10** | Qwen2.5-32B-GPTQ | 399 | **57.6%** |
| Zhou Dual-Inf | GPT-4 | 570 | 53.3% |

Local-DDx v10 outperforms Zhou et al.'s GPT-4 system by **+4.3 percentage points** using a 32B open-weight model on a single A100 GPU. No proprietary APIs required.

## Architecture

```
Local-DDx System Architecture

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    ddx_core      │    │   ddx_rounds    │    │  ddx_synthesis  │
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

## Quick Start

### v10 Full Pipeline (recommended)

Requires macOS with Apple Silicon and 48GB+ unified memory, or equivalent GPU system.

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:32b-instruct-q8_0

cd v10_full_pipeline
source venv/bin/activate
python3 app.py
# → http://localhost:7861
```

### v9 Lightweight Demo

Runs on any Apple Silicon Mac with 16GB+ unified memory.

```bash
ollama pull llama3.1:8b

cd v9_ollama_ui
pip install -r requirements.txt
python3 app.py
# → http://localhost:7860
```

See [v9_ollama_ui/README.md](v9_ollama_ui/README.md) for details.

## Project Structure

```
Local-DDx/
├── v10_full_pipeline/     # Full 7-round pipeline (Ollama/Apple Silicon)
├── v9_ollama_ui/          # Lightweight 3-round demo
├── v8/                    # Original vLLM/CUDA reference implementation
├── benchmark/             # Evaluation infrastructure (Colab A100)
├── archive/               # Historical versions (v6–v7)
├── Transcripts/           # Example diagnostic transcripts
├── docs/                  # GitHub Pages project site
├── CHANGELOG.md           # Version history
└── METRICS_GUIDE.md       # Detailed metrics documentation
```

## References

Goffman, E. (1956). The nature of deference and demeanor. *American Anthropologist*, 58(3), 473–502. https://doi.org/10.1525/aa.1956.58.3.02a00070

Nori, H., Lee, Y. T., Zhang, S., et al. (2023). Can generalist foundation models outcompete special-purpose tuning? A case study in medicine. *arXiv*. https://doi.org/10.48550/arXiv.2311.16452

Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking for conversation. *Language*, 50(4), 696–735. https://doi.org/10.2307/412243

Wu, Y., Xiong, J., & Deng, X. (2025). How social is it? A benchmark for LLMs' capabilities in multi-user multi-turn social agent tasks. *arXiv*. https://doi.org/10.48550/arXiv.2505.04628

Zhou, S., Lin, M., Ding, S., et al. (2025). Explainable differential diagnosis with dual-inference large language models. *npj Health Systems*, 2(1), 12. https://doi.org/10.1038/s44401-025-00015-6

## Citation

```bibtex
@software{local_ddx_2026,
  title={Local-DDx: Multi-Agent Collaborative Diagnostic System
         with Sliding Context Windows},
  author={Coburn, Del and Silver, Daniel and Fosse, Ethan and Griggs, Brandon},
  year={2026},
  url={https://github.com/Baglecake/Local-DDx}
}
```

## License

MIT License — See [LICENSE](LICENSE) for details.
