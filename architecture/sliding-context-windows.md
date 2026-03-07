# Sliding Context Windows

Sliding context windows transform the multi-agent system from a series of disconnected monologues into a collaborative diagnostic debate. Each agent receives a curated summary of prior team discourse, filtered by round type and relevance, before generating its response.

Implemented in `v10_full_pipeline/Modules/ddx_sliding_context.py`.

## Core Concept

Instead of only seeing the initial case description, each agent is briefed on the evolving state of the team's discussion. This enables agents to:

- React to peer arguments (challenge, support, build upon)
- Avoid redundancy by seeing what has already been said
- Specialize contributions based on gaps or disagreements in the discourse

## Transcript System

All agent responses are recorded in a global `TranscriptManager` as `TranscriptEntry` objects:

```
TranscriptEntry:
  round_type       # e.g., "refinement", "voting"
  round_number     # Sequential within round type
  agent_name       # e.g., "Dr. Emily Carter"
  specialty        # e.g., "Cardiology"
  content          # Full response text
  timestamp        # For ordering
  confidence       # 0.0-1.0, calculated from response quality
  reasoning_quality  # "basic", "standard", or "high"
  is_high_value    # Boolean, set by marker detection
  metadata         # Structured data (diagnoses, votes, etc.)
```

## Five Filter Types

Context is not delivered as raw history. It passes through filters selected per round type.

| Filter | What It Selects | How |
|--------|----------------|-----|
| `RECENT_ROUNDS` | Last 8 entries by timestamp | Recency-based, provides general awareness |
| `HIGH_CONFIDENCE` | Entries with confidence > 0.7 | Surfaces the most well-supported reasoning |
| `OPPOSING_VIEWS` | Entries containing contradiction markers | Keywords: "disagree", "however", "alternatively", "unlikely", "challenge" |
| `KEY_EXCHANGES` | High-value or high-confidence (> 0.8) interactions | Captures the most substantive discourse |
| `SPECIALIST_RELEVANT` | Entries mentioning the agent's specialty | Keyword match on specialty name within content |

Filters are combined via union (not intersection) — an entry matching any active filter is included.

## Round-Specific Filter Combinations

Each round type uses a different combination of filters, tuned to its diagnostic purpose:

| Round | Filters Applied | Rationale |
|-------|----------------|-----------|
| Specialized Ranking | `RECENT_ROUNDS` | Basic awareness of team |
| Symptom Management | `RECENT_ROUNDS` | Triage needs current context |
| Team Differentials | `RECENT_ROUNDS` | Independent reasoning with awareness |
| Master List | `HIGH_CONFIDENCE` + `SPECIALIST_RELEVANT` | Synthesis needs the best ideas, filtered by relevance |
| Refinement | `OPPOSING_VIEWS` + `HIGH_CONFIDENCE` + `KEY_EXCHANGES` | Debate requires disagreements and strong positions |
| Voting | `KEY_EXCHANGES` + `HIGH_CONFIDENCE` | Informed voting based on substantive discourse |
| Can't Miss | `OPPOSING_VIEWS` + `HIGH_CONFIDENCE` | Safety check needs contested and confident positions |

## High-Value Detection

Responses are flagged as high-value based on reasoning markers:

**Markers:** `evidence`, `contradict`, `disagree`, `alternative`, `challenge`, `support`, `clinical reasoning`, `risk factor`, `unlikely`, `more likely`, `less probable`, `consider`, `differential`, `pathophysiology`, `mechanism`, `presentation`, `findings`

**Detection rule:**
- Confidence > 0.7 AND 2+ markers present, OR
- 4+ markers present regardless of confidence

High-value entries are prioritized by the `KEY_EXCHANGES` filter.

## Context Building

Once entries are filtered, the context string is built with a token budget (default: 1,500 characters):

1. Round-specific collaboration guidance is prepended (e.g., "Focus on statements that contradict your prior reasoning")
2. Entries are formatted as: `Dr. Name (Specialty) [HIGH CONF] [KEY]: content preview...`
3. Individual entries are truncated to 200 characters
4. If the budget is exceeded, remaining entries are dropped with a truncation notice
5. The agent's own prior responses are excluded to prevent self-reinforcement

## Collaboration Guidance

Each round type includes a guidance note injected at the top of the context:

| Round | Guidance |
|-------|----------|
| Refinement | "Focus on statements that contradict your prior reasoning, or introduce new evidence. You may reinforce, challenge, or synthesize these perspectives." |
| Voting | "Consider which arguments were most compelling and well-supported by evidence." |
| Can't Miss | "Pay attention to any critical conditions that others may have overlooked." |
| Team Differentials | "Provide your independent differential. You'll collaborate in later rounds." |
