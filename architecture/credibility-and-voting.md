# Credibility Scoring and Voting

Local-DDx uses a two-stage consensus mechanism: credibility assessment followed by credibility-weighted preferential voting. This maps directly onto Goffman's (1956) framework — agents earn deference (voting weight) through the quality of their demeanor (reasoning quality) across rounds.

Implemented in `v10_full_pipeline/Modules/ddx_synthesis.py`.

## Dr. Reed Credibility Assessment

The "Dr. Reed" assessment evaluates each specialist's contributions across all rounds to produce a credibility score. This score then weights their vote in Round 6.

### Component Scores

| Component | What It Measures | Formula |
|-----------|-----------------|---------|
| **Insight** | Diagnostic quality | `(num_diagnoses * 3) + (avg_evidence_per_diagnosis * 2)` |
| **Synthesis** | Diagnostic breadth | `num_diagnoses * 4` |
| **Action** | Actionability | 20 if first diagnosis has 2+ evidence items, else 0 |

### Base Diagnostic Excellence Score (DES)

```
Base_DES = (2.5 * Insight) + (1.5 * Synthesis) + (1.0 * Action)
```

### Professional Valence Multiplier

Valence measures whether an agent elevated the team's discourse or contributed minimally. It is assessed by counting elevating indicators in the agent's combined response text:

**Elevating indicators:** `evidence`, `research`, `studies`, `guidelines`, `consider`, `alternative`, `question`, `challenge`, `disagree`, `differential`, `pathophysiology`, `mechanism`

| Indicator Count | Content Length | Valence |
|----------------|---------------|---------|
| 4+ | > 200 chars | 1.2 (significantly elevated) |
| 2+ | > 100 chars | 1.0 (professional) |
| any | > 50 chars | 0.8 (basic) |
| any | < 50 chars | 0.6 (minimal) |

### Final Credibility Score

```
Final_Score = Base_DES * Professional_Valence
```

This score is used to weight Borda votes. An agent who produced more diagnoses with better evidence and engaged more substantively with the discourse will have proportionally more influence on the final consensus.

## Preferential Voting (Borda Count)

Round 6 uses preferential voting where each specialist ranks their top 3 diagnoses.

### Point Allocation

| Rank | Points |
|------|--------|
| 1st Choice | 3 |
| 2nd Choice | 2 |
| 3rd Choice | 1 |

### Credibility Weighting

Each agent's Borda points are multiplied by their credibility score:

```
Weighted_Points = Borda_Points * min(Credibility_Score, 2 * Median_Credibility)
```

The cap at 2x median prevents any single agent from dominating the vote, even if their credibility score is significantly higher than others.

### Vote Extraction

Votes are extracted from agent responses using two methods:
1. **Structured data** — parsed from `<PREFERENTIAL_VOTE>` XML tags
2. **Regex fallback** — patterns like `1st Choice: Diagnosis - Justification` or `1. Diagnosis - Justification`

### Diagnosis Normalization

Before tallying, all voted diagnoses are normalized:
1. Remove numbering and bullets
2. Lowercase
3. Expand abbreviations (MI → Myocardial Infarction, STEMI → ST Elevation Myocardial Infarction, etc.)
4. Remove extra whitespace
5. Convert to title case

This prevents vote splitting between "MI", "Myocardial Infarction", and "STEMI" when they refer to the same condition.

### Consolidation

After voting, similar diagnoses are merged using token overlap (Jaccard similarity >= 0.6) and substring containment checks. The consolidation threshold is influenced by the refinement round's TempoScore — a high-quality debate (TempoScore >= 1.5) triggers more aggressive consolidation, reflecting higher team confidence.

## Theoretical Connection

In Goffman's terms:
- **Demeanor** = how an agent presents itself through reasoning quality, evidence citation, and engagement with peers
- **Deference** = the voting weight assigned by the credibility system

Good demeanor does not claim authority — it earns deference through the quality of conduct. The credibility scoring operationalizes this: agents that cite evidence, challenge assumptions, and propose well-supported differentials earn higher valence, which translates directly into greater influence over the consensus diagnosis.
