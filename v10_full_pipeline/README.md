# Local-DDx v10: Full Pipeline

**Complete 7-round diagnostic pipeline with sliding context windows, structured debate, and credibility-weighted voting.**

This version implements the full feature set from v8, adapted for Ollama on Apple Silicon.

## Features

| Feature | Description |
|---------|-------------|
| **7-Round Pipeline** | Complete diagnostic workflow |
| **Sliding Context Windows** | Agents see and respond to each other |
| **3-Subround Refinement** | Positions → Challenges → Final positions |
| **Preferential Voting** | Borda count with credibility weighting |
| **Dr. Reed Assessment** | Agent credibility scoring |
| **TempoScore Metrics** | Round-by-round performance tracking |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Ollama](https://ollama.ai) installed and running
- Python 3.8+
- 16GB+ unified memory (32GB+ recommended for full pipeline)

## Quick Start

### 1. Install Ollama and model

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b
```

### 2. Install dependencies

```bash
cd v10_full_pipeline
pip install -r requirements.txt
```

### 3. Launch the interface

```bash
python app.py
```

Navigate to `http://127.0.0.1:7861` (note: different port from v9)

## Pipeline Modes

### Full Mode (7 Rounds)

| Round | Purpose |
|-------|---------|
| 1. Specialized Ranking | Prioritize specialist relevance |
| 2. Symptom Management | Immediate triage protocols |
| 3. Team Differentials | Independent diagnosis generation |
| 4. Master List | Consolidate all diagnoses |
| 5. Refinement | 3-subround structured debate |
| 6. Voting | Preferential Borda count |
| 7. Can't Miss | Critical diagnosis safety check |

### Quick Mode (3 Rounds)

Faster demo mode that runs:
- Team Differentials
- Refinement (debate)
- Can't Miss

## Architecture

```
v10_full_pipeline/
├── app.py                      # Gradio web interface
├── config.yaml                 # Model configuration
├── requirements.txt
└── Modules/
    ├── inference_backends.py   # Symlink to v9 (shared)
    ├── ddx_core.py            # Agent framework
    ├── ddx_sliding_context.py # Context window system
    ├── ddx_rounds.py          # All 7 rounds
    ├── ddx_synthesis.py       # TempoScore, credibility, voting
    └── ddx_runner.py          # Main orchestrator
```

## CLI Usage

```bash
# Run test case
cd Modules
python ddx_runner.py --test

# Interactive mode
python ddx_runner.py --interactive
```

## Configuration

Edit `config.yaml` to customize models:

```yaml
# Same model, different temperatures (default)
conservative_model:
  model_name: "llama3.1:8b"
  temperature: 0.3

innovative_model:
  model_name: "llama3.1:8b"
  temperature: 0.8

# True dual-model (better diversity)
# conservative_model:
#   model_name: "llama3.1:8b"
# innovative_model:
#   model_name: "mistral-nemo:12b"
```

## Key Differences from v9

| Aspect | v9 (Demo) | v10 (Full) |
|--------|-----------|------------|
| Rounds | 3 | 7 |
| Context | Basic summary | Filtered sliding windows |
| Debate | Single round | 3 sub-rounds |
| Voting | Simple count | Borda + credibility |
| Metrics | None | TempoScore, Dr. Reed |
| Port | 7860 | 7861 |

## Comparison with v9

v9 remains available as a lightweight demo:
- Faster execution
- Simpler output
- Easier to understand
- Good for quick showcases

v10 is the full implementation:
- Complete diagnostic rigor
- Genuine multi-agent collaboration
- Research-grade output
- Better for detailed analysis

## License

MIT License - See [LICENSE](../LICENSE)
