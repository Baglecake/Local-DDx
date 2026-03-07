# LDDx v6: Core System Modules

This directory contains the core modules that power the LDDx v6 system. Each file encapsulates a key component of the multi-agent diagnostic pipeline, from agent creation and round execution to synthesis and evaluation.

## Module Architecture & Responsibilities

The system is designed with a clear separation of concerns, allowing for modular development, testing, and extension.

| Module                      | Primary Responsibility                   | Key Components                                        |
| :-------------------------- | :--------------------------------------- | :---------------------------------------------------- |
| **`ddx_core_v6.py`** | System Foundation & Agent Generation     | `ModelManager`, `DynamicAgentGenerator`, `DDxAgent`       |
| **`ddx_rounds_v6.py`** | Diagnostic Workflow Execution            | `RoundOrchestrator`, 7 Specialized Round Classes      |
| **`ddx_sliding_context.py`**| Collaborative Discourse Management     | `SlidingContextManager`, Context Filtering            |
| **`ddx_synthesis_v6.py`** | Consensus & Performance Analysis         | `TempoScore`, `DrReedAssessment`, `DiagnosisSynthesizer`  |
| **`ddx_evaluator_v6.py`** | Clinical Accuracy & Nuance               | `ClinicalEquivalenceAgent`, `EnhancedClinicalEvaluator` |
| **`ddx_utils.py`** | Common Helper Functions                  | `extract_diagnoses`, `validate_medical_response`      |

---

### 📄 `ddx_core_v6.py`

This is the foundational engine of the entire system.

-   **`ModelManager`**: Manages the loading and orchestration of the dual local LLMs (Conservative and Innovative), handling GPU memory and sampling parameters for stability and performance.
-   **`DynamicAgentGenerator`**: The system's most novel component. It uses an LLM to analyze a clinical case and autonomously design a bespoke team of specialist agents. It is not constrained by a predefined list and can generate any medical subspecialty required by the case.
-   **`DDxAgent`**: Defines the individual AI specialist. Each agent has its own configuration (`AgentConfig`), including a unique persona, reasoning style, and a temperature setting derived from its assigned model and role, allowing for diverse and technically defined agent "temperaments".

### 📄 `ddx_rounds_v6.py`

This module orchestrates the 7-round diagnostic process, turning the static design into an executable workflow.

-   **`RoundOrchestrator`**: Manages the sequential execution of the entire diagnostic pipeline, passing context and the full transcript between rounds.
-   **Specialized Round Classes**: Each of the 7 rounds (e.g., `SpecializedRankingRound`, `RefinementAndJustificationRound`, `PostDebateVotingRound`) is implemented as its own class, containing the logic and prompts for that specific stage of the diagnostic process.
-   **Sliding Context Integration**: This module is responsible for calling the `SlidingContextManager` at the beginning of each relevant round to provide agents with a summary of the prior discourse.

### 📄 `ddx_sliding_context.py`

This module enables true agent collaboration by managing the flow of information between them.

-   **`SlidingContextManager`**: The core of inter-agent communication. It builds a concise, relevant summary of the conversation for each agent before they respond. This prevents agents from acting as isolated monologues and encourages genuine debate.
-   **Intelligent Filtering**: Context is not just a raw transcript. It is filtered based on the current round's objective and the agent's specialty, prioritizing high-confidence statements, opposing views, and key exchanges.

### 📄 `ddx_synthesis_v6.py`

This module analyzes the *quality* of the team's reasoning and synthesizes a final answer.

-   **`TempoScoreCalculator`**: A novel metric that assesses the intellectual "energy" of each round. It measures the diversity of diagnoses in early rounds and the density of high-value interactions in the debate rounds.
-   **`DrReedAssessment`**: An AI-driven persona that evaluates the performance of each specialist, assigning them a `Credibility_Score` based on their insight, clarity, and collaborative conduct.
-   **`DiagnosisSynthesizer`**: The final decision-maker. It uses a credibility-weighted preferential voting system (Borda count) to aggregate the team's votes and produce a final, ranked differential diagnosis.

### 📄 `ddx_evaluator_v6.py`

This module provides a sophisticated, nuanced evaluation of the final diagnosis against a ground truth.

-   **`ClinicalEquivalenceAgent`**: Moves beyond simple keyword matching. This AI agent determines if a system diagnosis (e.g., "heart attack") is clinically equivalent to a ground truth diagnosis (e.g., "myocardial infarction"), allowing for a much fairer and more realistic assessment.
-   **Enhanced Metrics**: The evaluator calculates advanced metrics defined for the project, such as `Clinical_Reasoning_Quality`, `Diagnostic_Safety`, and `System_Safety_Coverage`, which provide a multi-dimensional view of performance.
-   **Context-Aware Analysis**: The evaluator uses the full transcript to find evidence for its judgments, such as identifying when a diagnosis was **Appropriately Excluded (AE)** or when its symptoms were covered by the **Symptom Management (TM-SM)** round.


**Use 'config.yaml' to load models, example:


conservative_model:
  name: 'Conservative-Model-Phi'
  model_path: 'microsoft/Phi-3-mini-4k-instruct'  
...

innovative_model:
  name: 'Innovative-Model-TinyLlama'
  model_path: 'NousResearch/Hermes-2-Pro-Mistral-7B'
...

