#!/usr/bin/env python3
# =============================================================================
# Local-DDx v10 - Full Pipeline Gradio Interface
# =============================================================================

"""
Gradio web interface for the full 7-round diagnostic pipeline.

Features:
- Full 7-round or quick 3-round modes
- Model selection dropdowns
- Real-time progress updates
- Round-by-round result visualization
- Credibility scores and voting results
- Transcript export
"""

import os
import sys
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Generator

import gradio as gr

# Add Modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modules'))

from ddx_runner import DDxSystem


# =============================================================================
# Global State
# =============================================================================

ddx_system: Optional[DDxSystem] = None
last_result: Optional[Dict[str, Any]] = None
current_models: Dict[str, str] = {}


# =============================================================================
# Ollama Model Discovery
# =============================================================================

def get_available_ollama_models() -> List[str]:
    """Get list of models available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            # Sort by size (larger models first for quality)
            return sorted(models)
    except:
        pass

    # Fallback defaults
    return ["llama3.1:8b", "qwen2.5:32b-instruct-q8_0", "mistral-nemo:12b"]


def get_model_choices() -> List[str]:
    """Get model choices for dropdown"""
    models = get_available_ollama_models()

    # Prioritize certain models at the top
    priority = ["qwen2.5:32b-instruct-q8_0", "mistral-nemo:12b", "llama3.1:8b", "gemma2:latest"]
    sorted_models = []

    for p in priority:
        if p in models:
            sorted_models.append(p)
            models.remove(p)

    sorted_models.extend(models)
    return sorted_models


# =============================================================================
# System Initialization
# =============================================================================

def initialize_system(conservative_model: str, innovative_model: str) -> str:
    """Initialize the DDx system with selected models"""
    global ddx_system, current_models

    # Check if we need to reinitialize
    if ddx_system is not None:
        if (current_models.get('conservative') == conservative_model and
            current_models.get('innovative') == innovative_model):
            return "System ready (already initialized with these models)"

    # Create new system
    ddx_system = DDxSystem()

    # Override config with selected models
    from ddx_core import ModelConfig

    custom_configs = {
        'conservative_model': ModelConfig(
            name='Conservative',
            model_name=conservative_model,
            temperature=0.3,
            top_p=0.7,
            max_tokens=1024,
            role='conservative'
        ),
        'innovative_model': ModelConfig(
            name='Innovative',
            model_name=innovative_model,
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            role='innovative'
        )
    }

    # Initialize with custom configs
    from ddx_core import OllamaModelManager
    from ddx_sliding_context import TranscriptManager
    from ddx_core import DynamicAgentGenerator

    ddx_system.model_manager = OllamaModelManager(custom_configs)

    if not ddx_system.model_manager.initialize():
        ddx_system = None
        return "ERROR: Failed to initialize. Is Ollama running?"

    # Warm up models
    for model_id in ddx_system.model_manager.get_available_models():
        ddx_system.model_manager.load_model(model_id)

    ddx_system.transcript = TranscriptManager()
    ddx_system.agent_generator = DynamicAgentGenerator(
        ddx_system.model_manager, ddx_system.transcript
    )

    current_models = {
        'conservative': conservative_model,
        'innovative': innovative_model
    }

    return f"System initialized with:\n- Conservative: {conservative_model}\n- Innovative: {innovative_model}"


# =============================================================================
# Diagnosis Functions
# =============================================================================

def run_diagnosis(case_text: str, mode: str,
                  conservative_model: str, innovative_model: str) -> Generator:
    """Run diagnosis with progress updates"""
    global ddx_system, last_result

    if not case_text.strip():
        yield "Please enter a clinical case.", "", "", "", ""
        return

    # Initialize with selected models
    yield "Initializing system...", "", "", "", ""
    init_result = initialize_system(conservative_model, innovative_model)
    if "ERROR" in init_result:
        yield init_result, "", "", "", ""
        return

    # Analyze case
    yield f"Generating specialist team using {conservative_model}...", "", "", "", ""

    analysis = ddx_system.analyze_case(case_text, "gradio_case", max_specialists=5)

    if not analysis['success']:
        yield f"Error: {analysis.get('error', 'Unknown error')}", "", "", "", ""
        return

    # Format team info
    team_md = "## Specialist Team\n\n"
    for spec in analysis['specialists']:
        model_badge = "🔵" if "conservative" in spec['model'] else "🟠"
        team_md += f"- **{spec['name']}** - {spec['specialty']} {model_badge}\n"

    team_md += f"\n**Models:** {conservative_model} / {innovative_model}"

    yield "Running diagnosis...", team_md, "", "", ""

    # Run diagnosis based on mode
    if mode == "Full (7 rounds)":
        result = ddx_system.run_full_diagnosis()
    else:
        result = ddx_system.run_quick_diagnosis()

    last_result = result

    # Format results
    diagnoses_md = format_diagnoses(result)
    rounds_md = format_rounds(result)
    credibility_md = format_credibility(result)

    status = f"Diagnosis complete in {result.get('total_duration', 0):.1f}s"

    yield status, team_md, diagnoses_md, rounds_md, credibility_md


def format_diagnoses(result: Dict[str, Any]) -> str:
    """Format final diagnoses as markdown"""
    md = "## Final Diagnoses\n\n"

    # Check for voting result
    voting = result.get('voting_result')
    if voting:
        md += f"**Winner:** {voting.get('winner', 'N/A')}\n\n"
        md += "| Rank | Diagnosis | Score |\n|------|-----------|-------|\n"
        for i, (diag, score) in enumerate(voting.get('ranked', [])[:6], 1):
            md += f"| {i} | {diag} | {score:.1f} |\n"
    elif result.get('final_diagnoses'):
        md += "| Rank | Diagnosis |\n|------|-----------|\n"
        for i, diag in enumerate(result['final_diagnoses'][:6], 1):
            if isinstance(diag, tuple):
                md += f"| {i} | {diag[0]} |\n"
            else:
                md += f"| {i} | {diag} |\n"
    else:
        md += "*No final diagnoses generated*"

    return md


def format_rounds(result: Dict[str, Any]) -> str:
    """Format round-by-round results"""
    md = "## Round Results\n\n"

    rounds = result.get('rounds', {})
    if not rounds:
        return md + "*No round data available*"

    for round_name, round_data in rounds.items():
        responses = round_data.get('responses', [])
        duration = round_data.get('duration', 0)

        md += f"### {round_name.replace('_', ' ').title()}\n"
        md += f"*{len(responses)} responses, {duration:.1f}s*\n\n"

        # Show abbreviated responses
        for resp in responses[:3]:  # Limit to 3 per round
            agent = resp.get('agent_name', 'Unknown')
            specialty = resp.get('specialty', '')
            conf = resp.get('confidence_score', 0)
            content = resp.get('content', '')[:200]

            md += f"**{agent}** ({specialty}) - Confidence: {conf:.0%}\n"
            md += f"> {content}...\n\n"

        if len(responses) > 3:
            md += f"*...and {len(responses) - 3} more responses*\n\n"

        md += "---\n\n"

    return md


def format_credibility(result: Dict[str, Any]) -> str:
    """Format credibility scores"""
    md = "## Agent Credibility (Dr. Reed Assessment)\n\n"

    cred = result.get('credibility_scores', {})
    if not cred:
        return md + "*Credibility scores not available*"

    md += "| Agent | Final Score | Base | Valence |\n"
    md += "|-------|-------------|------|--------|\n"

    # Sort by final score
    sorted_cred = sorted(cred.items(), key=lambda x: x[1].get('final_score', 0), reverse=True)

    for agent_name, scores in sorted_cred:
        final = scores.get('final_score', 0)
        base = scores.get('base_score', 0)
        valence = scores.get('valence', 1.0)
        md += f"| {agent_name} | {final:.1f} | {base:.1f} | {valence:.1f}x |\n"

    return md


# =============================================================================
# Export Function
# =============================================================================

def export_transcript() -> str:
    """Export full transcript to file"""
    global ddx_system, last_result

    if ddx_system is None or last_result is None:
        return "No diagnosis to export. Run a diagnosis first."

    try:
        export_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(export_dir, exist_ok=True)

        filename = f"lddx_v10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(export_dir, filename)

        ddx_system.export_results(filepath)

        return f"Exported to: {filepath}"

    except Exception as e:
        return f"Export failed: {e}"


# =============================================================================
# Example Cases
# =============================================================================

EXAMPLE_CASES = [
    # STEMI
    """A 58-year-old male with history of type 2 diabetes and hypertension presents with crushing substernal chest pain radiating to his left arm that began 90 minutes ago. He is diaphoretic and anxious.

Vitals: BP 165/95, HR 92, RR 22, SpO2 96% on room air
ECG: ST elevation in leads V1-V4 with reciprocal changes in II, III, aVF
Labs: Troponin I elevated at 2.4 ng/mL (normal <0.04)

Physical exam: JVP not elevated, lungs clear, no murmurs, no edema.""",

    # Pulmonary Embolism
    """A 42-year-old female presents with sudden onset dyspnea and pleuritic chest pain. She returned from a 12-hour flight 3 days ago and has been less mobile than usual. She takes oral contraceptives.

Vitals: BP 110/70, HR 112, RR 24, SpO2 91% on room air
ECG: Sinus tachycardia, S1Q3T3 pattern
D-dimer: Elevated at 2.4 mg/L (normal <0.5)

Physical exam: Right calf tender with mild swelling, tachypneic, clear lungs.""",

    # Cholesterol Embolism
    """A 61-year-old man presents two weeks after emergency cardiac catheterization with decreased urinary output and malaise. Examination shows mottled, reticulated purplish discoloration of the feet (livedo reticularis).

Labs: Creatinine 4.2 mg/dL (baseline 1.1), eosinophilia (11%), low complement
Urinalysis: Mild proteinuria, eosinophiluria
Renal biopsy: Intravascular spindle-shaped vacuoles (cholesterol clefts)

History: Recent cardiac catheterization, chronic hypertension, smoking history.""",
]


def get_example_cases():
    """Return example cases for Gradio"""
    return [[case] for case in EXAMPLE_CASES]


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    """Create the Gradio interface"""

    # Get available models
    model_choices = get_model_choices()

    # Set defaults
    default_conservative = "qwen2.5:32b-instruct-q8_0" if "qwen2.5:32b-instruct-q8_0" in model_choices else model_choices[0]
    default_innovative = default_conservative

    with gr.Blocks(
        title="Local-DDx v10",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .prose { max-width: none !important; }
        """
    ) as demo:
        # Header
        gr.Markdown("""
        # Local-DDx v10 - Full Diagnostic Pipeline

        Multi-agent collaborative diagnosis with sliding context windows,
        structured debate, and credibility-weighted voting.
        """)

        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                case_input = gr.Textbox(
                    label="Clinical Case Presentation",
                    placeholder="Enter the clinical case details...",
                    lines=10
                )

                gr.Markdown("### Model Selection")
                with gr.Row():
                    conservative_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=default_conservative,
                        label="Conservative Model (systematic)",
                        info="Lower temperature, evidence-focused"
                    )
                    innovative_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=default_innovative,
                        label="Innovative Model (exploratory)",
                        info="Higher temperature, creative"
                    )

                with gr.Row():
                    mode_select = gr.Radio(
                        choices=["Quick (3 rounds)", "Full (7 rounds)"],
                        value="Quick (3 rounds)",
                        label="Pipeline Mode"
                    )

                with gr.Row():
                    run_btn = gr.Button("Run Diagnosis", variant="primary", size="lg")
                    export_btn = gr.Button("Export Results", size="lg")

                status_box = gr.Textbox(label="Status", interactive=False)

                gr.Examples(
                    examples=get_example_cases(),
                    inputs=[case_input],
                    label="Example Cases"
                )

            # Right column - Results
            with gr.Column(scale=1):
                with gr.Tab("Diagnoses"):
                    diagnoses_output = gr.Markdown()

                with gr.Tab("Team"):
                    team_output = gr.Markdown()

                with gr.Tab("Rounds"):
                    rounds_output = gr.Markdown()

                with gr.Tab("Credibility"):
                    credibility_output = gr.Markdown()

        # Event handlers
        run_btn.click(
            fn=run_diagnosis,
            inputs=[case_input, mode_select, conservative_dropdown, innovative_dropdown],
            outputs=[status_box, team_output, diagnoses_output, rounds_output, credibility_output]
        )

        export_btn.click(
            fn=export_transcript,
            inputs=[],
            outputs=[status_box]
        )

        # Footer
        gr.Markdown("""
        ---
        **Local-DDx v10** - All processing runs locally via Ollama.
        Patient data never leaves your machine.

        🔵 = Conservative model | 🟠 = Innovative model
        """)

    return demo


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point"""
    print("="*60)
    print("Local-DDx v10 - Full Pipeline")
    print("="*60)

    # Show available models
    models = get_available_ollama_models()
    print(f"Available Ollama models: {len(models)}")
    for m in models[:5]:
        print(f"  - {m}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")

    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )


if __name__ == "__main__":
    main()
