# Architecture Overview

Local-DDx is a multi-agent diagnostic system built around four core modules that work together through a shared transcript.

## System Diagram

```
                        ┌──────────────────┐
                        │   Case Input     │
                        └────────┬─────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  DynamicAgentGenerator   │
                    │  (ddx_core.py)           │
                    │                          │
                    │  Reads case → asks LLM   │
                    │  to propose 4-6 specs    │
                    │  Assigns model roles     │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         DiagnosticPipeline           │
              │         (ddx_rounds.py)              │
              │                                      │
              │  7 rounds executed in sequence:       │
              │  R1 Ranking → R2 Triage → R3 DDx    │
              │  → R4 Master List → R5 Refinement   │
              │  → R6 Voting → R7 Can't Miss        │
              └───────────┬──────────┬──────────────┘
                          │          │
          ┌───────────────▼──┐  ┌───▼──────────────────┐
          │ TranscriptManager│  │   Synthesis Module    │
          │ (ddx_sliding_    │  │   (ddx_synthesis.py)  │
          │  context.py)     │  │                       │
          │                  │  │  • Dr. Reed Cred.     │
          │ Records entries  │  │  • Borda Voting       │
          │ Filters context  │  │  • TempoScore         │
          │ per round type   │  │  • Consolidation      │
          └──────────────────┘  └───────────────────────┘
```

## Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| **Core** | `ddx_core.py` | Agent framework, model management, dynamic specialist generation |
| **Rounds** | `ddx_rounds.py` | 7-round pipeline execution, round-specific prompts |
| **Sliding Context** | `ddx_sliding_context.py` | Transcript storage, 5 context filters, high-value detection |
| **Synthesis** | `ddx_synthesis.py` | Credibility scoring, Borda voting, TempoScore, diagnosis normalization |
| **Runner** | `ddx_runner.py` | System orchestrator, CLI interface, JSON export |
| **UI** | `app.py` | Gradio web interface with model selection |
| **Backends** | `inference_backends.py` | Ollama/vLLM abstraction (shared with v9 via symlink) |

## Dual-Model Architecture

Both models use the same underlying LLM (default: `qwen2.5:32b-instruct-q8_0`) but with different sampling parameters:

| Parameter | Conservative | Innovative |
|-----------|-------------|------------|
| Temperature | 0.3 | 0.8 |
| Top-p | 0.7 | 0.95 |
| Role | Cautious, evidence-focused | Exploratory, creative |

Specialists are assigned to models in round-robin order during team generation. Each specialist's final temperature is further adjusted by reasoning style (analytical: -0.1, innovative: +0.15, intuitive: +0.1). This creates a spectrum of diagnostic personalities within the team without requiring multiple GPU-loaded models.

The architecture includes hooks for true dual-model support (different base models), configurable via `config.yaml`.

## Dynamic Specialist Generation

The `DynamicAgentGenerator` creates a case-specific team:

1. The conservative model receives the case description and proposes 4-6 specialists as JSON
2. Each proposal includes name, specialty, persona, reasoning style, and focus areas
3. Specialists are mapped to the `SpecialtyType` enum (15 specialties) via keyword matching
4. If LLM proposal fails, a fallback team of 4 generalists is used (Internal Medicine, Emergency Medicine, Cardiology, Infectious Disease)
5. At least one generalist is always requested in the prompt

## Devil's Advocate Mechanism

During Round 5 (Refinement), the pipeline implements adversarial testing through 3 structured sub-rounds:

1. **Initial Positions** - Each specialist states their top 3 diagnoses with confidence levels and supporting evidence
2. **Direct Challenges** - Each specialist is assigned a colleague to challenge (round-robin). They must provide counter-evidence and propose alternatives
3. **Final Positions** - Specialists respond to challenges, marking each diagnosis as MAINTAINED or CHANGED

This structure ensures every position is tested. The round-robin assignment prevents echo chambers — every specialist both challenges and is challenged.

## Data Flow

Each agent response flows through the system as follows:

1. Agent generates response via `DDxAgent.generate_response()`
2. Response is scored for confidence (structured data + medical terms + length)
3. Response is assessed for reasoning quality (basic/standard/high) via marker detection
4. Response is recorded to `TranscriptManager` as a `TranscriptEntry`
5. High-value flag is set if confidence > 0.7 with 2+ reasoning markers, or 4+ markers regardless
6. Subsequent agents receive filtered context from the transcript based on round-specific filter combinations

## Further Reading

- [Sliding Context Windows](sliding-context-windows.md) - Filter types and context management
- [Credibility and Voting](credibility-and-voting.md) - Dr. Reed assessment and Borda count
- [Metrics](metrics.md) - TempoScore, evaluation methodology, performance measures
