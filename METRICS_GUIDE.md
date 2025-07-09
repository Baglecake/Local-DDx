# LDDx v6: Formal Metrics Guide

This document provides a formal definition and explanation of the novel performance metrics used in the LDDx v6 system. These metrics are designed to evaluate not only the diagnostic accuracy but also the quality, safety, and collaborative dynamics of the multi-agent reasoning process.

The metrics are divided into two main categories:
1.  **Collaborative Process Metrics**: Evaluate the quality of the team's interaction during the diagnostic sequence.
2.  **Diagnostic Evaluation Metrics**: Assess the clinical accuracy and safety of the final consensus diagnosis against a ground truth.

---

## 1. Collaborative Process Metrics

These metrics are calculated during and immediately after the diagnostic rounds to assess the internal dynamics of the agent team. They are primarily defined in `ddx_synthesis_v6.py`.

### TempoScore

The **TempoScore** is a dynamic metric that measures the intellectual "energy" or momentum of a given round. It adapts its calculation based on the round's objective.

-   **Independent Differentials Round**: The score reflects the breadth of initial ideas.
    -   **Formula**: `1.0 + (0.1 * number_of_unique_diagnoses)`
    -   **Interpretation**: A higher score indicates a more comprehensive and diverse initial set of hypotheses from the team.

-   **Debate & Refinement Round**: The score reflects the quality and density of the collaborative discourse.
    -   **Calculation**: It is based on **Symbolic Curvature**, which is the ratio of "high-value interactions" to the number of participants.
    -   **High-Value Interactions** are responses containing multiple reasoning keywords (e.g., 'evidence', 'challenge', 'alternative') or a structured argumentative format.
    -   **Formula**: `min(symbolic_curvature + 0.5, 2.5)`
    -   **Interpretation**: A higher score signifies a more robust and evidence-driven debate, where agents are actively engaging with, challenging, and building upon each other's reasoning. A high tempo score in this round can trigger more aggressive diagnosis consolidation in the final synthesis step.

### Dr. Reed's Assessment & Credibility Score

This is an internal performance review conducted by the "Dr. Reed" persona, which evaluates each specialist's contribution to produce a final **`Credibility_Score`**. This score is crucial as it is used to weight each agent's vote during the final diagnosis synthesis.

-   **Core Components**: The assessment is based on three factors:
    -   **Insight Score**: Quality and depth of diagnostic reasoning.
    -   **Synthesis Score**: Ability to integrate multiple data points.
    -   **Action Score**: Clarity and actionability of recommendations.
-   **Professional Valence**: A multiplier that rewards agents for elevating the team's performance (e.g., by citing evidence, challenging assumptions) and penalizes minimal or unhelpful contributions.
    -   **Formula**: `Final Score = (2.5*Insight + 1.5*Synthesis + 1.0*Action) * Professional_Valence`.
    -   **Interpretation**: This system incentivizes high-quality, collaborative behavior, ensuring that the most credible and helpful specialists have a greater influence on the final outcome.

---

## 2. Diagnostic Evaluation Metrics

These metrics are calculated at the end of the pipeline by the `EnhancedClinicalEvaluator` to measure the final output against a ground truth. They are primarily defined in `ddx_evaluator_v6.py`.

### AI-Enhanced Diagnosis Classification

The system moves beyond simple string matching by using a `ClinicalEquivalenceAgent` to classify each diagnosis in the final list. This allows for a more nuanced evaluation that understands clinical context.

| Category                              | Abbreviation | Description                                                                                                                                              |
| :------------------------------------ | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **True Positive** |      **TP** | The system's diagnosis is clinically equivalent to a ground truth diagnosis (e.g., "Heart Attack" vs. "Myocardial Infarction").                           |
| **False Positive** |      **FP** | The system proposed a diagnosis that is not in the ground truth and is not a clinically appropriate alternative.           |
| **False Negative** |      **FN** | The system failed to identify a diagnosis that was in the ground truth. Also referred to as a **True Miss**.                      |
| **Clinically Appropriate Alternative**|     **CAA** | The system identified a diagnosis that, while not the ground truth, is a reasonable alternative supported by evidence in the "Can't Miss" round. |
| **Appropriately Excluded** |      **AE** | The system considered a ground truth diagnosis during the debate but correctly and explicitly ruled it out based on evidence.          |
| **True Miss - Symptom Mgmt. Captured**|    **TM-SM** | The specialists missed a ground truth diagnosis, but the initial triage agent correctly treated its key symptoms (e.g., missed "Dehydration" but gave fluids). |

### Final Performance Scores

These scores provide a holistic view of the system's performance, balancing accuracy with safety and reasoning quality.

-   **`Traditional Recall`**: Standard measure of diagnostic accuracy.
    -   **Formula**: `TP / (Total Ground Truth Diagnoses)`.
    -   **Interpretation**: How many of the correct diagnoses did the system find?

-   **`Clinical Reasoning Quality`**: A holistic score rewarding correct diagnoses, appropriately excluded ones, and clinically acceptable alternatives, while penalizing misses and errors.
    -   **Formula**: `(TP + (CAA_weight * CAA) + AE) / (Total Diagnoses Considered)`.
    -   **Interpretation**: How high was the quality of the overall clinical judgment, beyond just getting the right answer?

-   **`Diagnostic Safety`**: Measures how reliable the final list is for guiding clinical action. It rewards correct and reasonable diagnoses while penalizing incorrect ones that could lead to harm.
    -   **Formula**: `(TP + max(0, CAA_weight * CAA)) / (TP + CAA + FP)`.
    -   **Interpretation**: If a clinician acted on this list, how safe would the outcome be?

-   **`System Safety Coverage`**: A v6 enhancement that measures the entire system's ability to "catch" conditions requiring intervention, including those handled by the initial symptom management round.
    -   **Formula**: `(TP + TM_SM) / (Total Ground Truth Diagnoses)`.
    -   **Interpretation**: What percentage of harmful conditions did the *entire system* (triage + specialists) successfully address, either by diagnosis or by intervention?
