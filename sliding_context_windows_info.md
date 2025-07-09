# LDDx v6: Sliding Context Windows

One of the most significant innovations in the LDDx v6 system is the implementation of **Sliding Context Windows**. This mechanism is what elevates the platform from a series of disconnected agent monologues into a truly collaborative and dynamic diagnostic debate.

This document explains the core concept, the technical implementation, and the impact of this feature on the system's reasoning capabilities.

## 1. What is a Sliding Context Window?

A sliding context window is a curated summary of the most recent and relevant parts of the conversation that is provided to an AI agent *before* it generates a response.

Instead of only having access to the initial case file, each agent is given a concise briefing on the evolving state of the team's discourse. This allows agents to:

* **React to Peer Arguments**: Directly challenge, support, or build upon the reasoning of other specialists.
* **Avoid Redundancy**: See what has already been said, preventing repetitive or unhelpful contributions.
* **Engage in True Collaboration**: Participate in a dynamic conversation where ideas are synthesized and refined over time.
* **Facilitate Epistemic Labor Division**: Agents can specialize their contributions based on the gaps or disagreements they observe in the provided context.

This process is managed entirely by the `SlidingContextManager` and is crucial for creating the emergent, collaborative intelligence that the project aims to achieve.

## 2. Technical Workflow: How It Works

The context generation process occurs at the beginning of each collaborative round for every participating agent. Here is the step-by-step workflow:

| Step | Action | Module Responsible |
| :--- | :--- | :--- |
| **1. Gather Transcript** | The `RoundOrchestrator` maintains a complete, unabridged transcript of every agent response from every completed round. | `ddx_rounds_v6.py` |
| **2. Request Context** | As a new round begins, the orchestrator requests a contextual summary for each agent from the `SlidingContextManager`.| `ddx_rounds_v6.py`, `ddx_sliding_context.py`|
| **3. Extract & Structure**| The manager parses the full transcript, converting each prior response into a structured `ContextEntry` containing the statement, agent name, confidence, etc. | `ddx_sliding_context.py` |
| **4. Apply Intelligent Filters** | This is the core of the system. Instead of a simple history, the context is filtered to maximize relevance based on the current round's objective. Filters include: <br> - **High-Value Interactions**: Prioritizing statements with keywords like `evidence`, `disagree`, or `challenge`.<br> - **Opposing Views**: Highlighting arguments that directly contradict the current agent's known positions.<br> - **Specialist-Relevant Content**: Filtering for discourse relevant to the agent's specialty (e.g., providing a cardiologist with context mentioning "cardiac" or "ECG"). | `ddx_sliding_context.py` |
| **5. Build Context String**| The filtered entries are compiled into a final summary. This includes **Attention Guidance**, a round-specific instruction that directs the agent's focus (e.g., *"Focus on statements that contradict your prior reasoning"* ).| `ddx_sliding_context.py` |
| **6. Inject into Prompt**| The final, curated context string is injected into the agent's system prompt before it generates its response. | `ddx_core_v6.py` |

## 3. Example of a Generated Context

Before the **Refinement & Justification Round**, an agent might receive the following context in its prompt:

COLLABORATION GUIDANCE: Focus especially on statements that contradict your prior reasoning, or introduce new evidence. You may reinforce, challenge, or synthesize these perspectives.

RECENT TEAM DISCOURSE:

TEAM_INDEPENDENT_DIFFERENTIALS — Dr. Alexander Park🔥:
I believe this is Acute Tubular Necrosis (ATN) due to the recent cardiac catheterization and the patient's use of nephrotoxic medications. The decreased urinary output is a key indicator.

TEAM_INDEPENDENT_DIFFERENTIALS — Dr. Angela Malik⚡:
However, we must also consider Thrombotic Microangiopathy (TMA) given the patient's history of unstable angina and the presence of intravascular spindle-shaped vacuoles on biopsy.


*(Note: The 🔥 and ⚡ symbols are conceptual indicators of high-confidence or high-value statements identified by the filtering process.)*

## 4. Impact on System Performance

The sliding context window is fundamental to the system's ability to produce high-quality, collaborative reasoning.

* **Deeper Analysis**: It pushes agents beyond their initial hypotheses, forcing them to consider alternatives and justify their positions against counter-arguments.
* **Emergent Synthesis**: Novel insights emerge from the synthesis of different specialist perspectives, which would not be possible if agents operated in isolation.
* **Improved Efficiency**: By seeing the state of the debate, agents can make more targeted and effective contributions, leading to a more efficient diagnostic process.

In summary, the sliding context window transforms the multi-agent system from a simple parallel processor into a cohesive, interactive, and intelligent collaborative team.
