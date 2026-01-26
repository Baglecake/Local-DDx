# Local-DDx v9: Ollama Edition

**Lightweight demonstration version for Apple Silicon**

This version provides a streamlined proof-of-concept implementation of the Local-DDx multi-agent diagnostic system, designed to run locally on M-series Macs using Ollama for inference.

## Overview

v9_ollama_ui is a portable demonstration of Local-DDx's core architecture:
- **Dynamic specialist generation** - AI assembles case-specific medical teams
- **Multi-round diagnostic reasoning** - Structured collaborative analysis
- **Gradio web interface** - Interactive case submission and visualization
- **Local inference** - All processing happens on-device via Ollama

> **Note:** This is a demonstration version implementing a 3-round diagnostic workflow. For the full 7-round pipeline with sliding context windows, Devil's Advocate mechanisms, and TempoScore metrics, see the [v8 implementation](../v8/).

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Ollama](https://ollama.ai) installed and running
- Python 3.8+
- 16GB+ unified memory recommended

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.ai
# Then pull the recommended model:
ollama pull llama3.1:8b
```

### 2. Install dependencies

```bash
cd v9_ollama_ui
pip install -r requirements.txt
```

### 3. Launch the interface

```bash
python app.py
```

Navigate to `http://127.0.0.1:7860` in your browser.

## Configuration

Edit `config.yaml` to adjust model parameters:

```yaml
conservative_model:
  model_name: "llama3.1:8b"
  temperature: 0.3    # Lower temperature for systematic reasoning

innovative_model:
  model_name: "llama3.1:8b"
  temperature: 0.8    # Higher temperature for exploratory thinking
```

### Tested Models

| Model | Size | Notes |
|-------|------|-------|
| `llama3.1:8b` | 4.9 GB | Recommended - good balance |
| `mistral-nemo:12b` | 7.1 GB | Better reasoning, slower |
| `phi3:mini` | 2.2 GB | Fast iteration for demos |

## Architecture

```
v9_ollama_ui/
├── app.py                      # Gradio web interface
├── config.yaml                 # Model configuration
├── requirements.txt            # Python dependencies
└── Modules/
    ├── inference_backends.py   # Ollama API abstraction
    └── ddx_core_ollama.py      # Core diagnostic system
```

## Diagnostic Rounds (v9 POC)

| Round | Purpose |
|-------|---------|
| 1. Differential | Independent diagnosis generation by each specialist |
| 2. Debate | Collaborative evidence-based discussion |
| 3. Can't-Miss | Critical diagnosis identification |

## Comparison with Full System

| Feature | v9 (Ollama) | v8 (vLLM) |
|---------|-------------|-----------|
| Diagnostic rounds | 3 | 7 |
| Sliding context windows | - | Yes |
| Devil's Advocate mechanism | - | Yes |
| TempoScore metrics | - | Yes |
| Credibility-weighted voting | - | Yes |
| Hardware requirement | Apple Silicon | CUDA GPU |

## License

MIT License - See [LICENSE](../LICENSE)
