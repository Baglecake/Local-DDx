# Metrics

Local-DDx uses two categories of metrics: collaborative process metrics (assessed during the pipeline) and diagnostic evaluation metrics (assessed against ground truth after completion).

## Collaborative Process Metrics

### TempoScore

TempoScore measures the intellectual momentum of each round. It adapts its calculation to the round's objective.

**Differential round** (Round 3):
```
Score = 1.0 + (0.1 * unique_diagnoses_count)
Cap: 2.5
```
A higher score indicates a more diverse initial hypothesis space from the team.

**Refinement round** (Round 5):
```
Symbolic_Curvature = high_value_interactions / agent_count
Score = min(Symbolic_Curvature + 0.5, 2.5)
```
Based on the ratio of high-value interactions (responses with 2+ reasoning markers and confidence > 0.7) to participants. A higher score indicates more substantive debate.

The refinement TempoScore also tracks:
- **Direct challenges** — occurrences of "challenge", "disagree", "however", "but"
- **Position changes** — occurrences of "convinced", "agree now", "changed my", "reconsidering"
- **Evidence citations** — occurrences of "evidence", "study", "research", "literature", "guideline", "data"

When the refinement TempoScore >= 1.5, more aggressive diagnosis consolidation is triggered in the synthesis step.

### Credibility Score

See [Credibility and Voting](credibility-and-voting.md) for the full Dr. Reed assessment methodology.

## Diagnostic Evaluation Metrics

These metrics are calculated by the benchmark evaluator (`benchmark/ddx_evaluator.py`) against ground truth from the Open-XDDx dataset. The evaluator uses deterministic clinical equivalence matching — no LLM-as-judge.

### Diagnosis Classification

| Category | Abbrev. | Description |
|----------|---------|-------------|
| True Positive | TP | System diagnosis matches a ground truth diagnosis |
| False Positive | FP | System proposed a diagnosis not in ground truth |
| False Negative | FN | System missed a ground truth diagnosis |
| Appropriately Excluded | AE | Ground truth diagnosis was considered during the pipeline but not included in the final list |

### 4-Tier Matching Pipeline

The evaluator determines TP/FP/FN through progressive matching:

| Tier | Score | Method |
|------|-------|--------|
| Direct match | 1.0 | Exact string after 11-step normalization |
| Synonym match | 0.95 | 260+ bidirectional medical synonym groups |
| Hierarchy match | 0.90 | 75+ clinical supertype/subtype trees |
| Word overlap | 0.80 | Jaccard similarity >= 0.8 on content words |

### Normalization Pipeline (11 steps)

1. Lowercase
2. Unicode normalization (NFKD)
3. Apostrophe variant standardization
4. Parenthetical content removal (nested)
5. Dash/slash/en-dash/em-dash to space
6. Stop-word removal (the, a, an, of, in, with, and, or, by, to, for, is, are, was)
7. Whitespace collapse
8. Leading/trailing whitespace strip
9. Abbreviation expansion
10. Possessive removal ('s, s')
11. Trailing punctuation strip

### Performance Scores

**Recall** — What proportion of ground truth diagnoses did the system identify?
```
Recall = TP / (TP + FN)
```

**Precision** — What proportion of system diagnoses were correct?
```
Precision = TP / (TP + FP)
```

**Clinical Reasoning Quality (CRQ)** — Rewards correct diagnoses and appropriately excluded ones.
```
CRQ = (TP + 0.5 * AE) / (TP + FN + AE)
```

**Diagnostic Safety** — How reliably does the system catch conditions requiring attention?
```
Safety = (TP + AE) / (TP + FN + AE)
```

### Benchmark Results

Evaluated on the Open-XDDx dataset (570 cases, 9 specialties). See [benchmark/README.md](../benchmark/README.md).

| System | Model | Cases | Recall | Precision | Safety |
|--------|-------|-------|--------|-----------|--------|
| **Local-DDx v10** | Qwen2.5-32B-GPTQ | 399 | **57.6%** | 49.2% | 67.5% |
| Zhou Dual-Inf | GPT-4 | 570 | 53.3% | — | — |
