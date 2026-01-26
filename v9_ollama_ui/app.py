# =============================================================================
# Local-DDx Gradio Interface
# =============================================================================

"""
Web interface for Local-DDx collaborative diagnostic AI.
Built with Gradio for hackathon demonstration.
"""

import gradio as gr
import time
import sys
import os
from datetime import datetime

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Modules'))

from ddx_core_ollama import DDxSystem


# =============================================================================
# Global System Instance
# =============================================================================

ddx_system = None
last_result = None


def initialize_system():
    """Initialize DDx system on startup"""
    global ddx_system

    if ddx_system is None:
        ddx_system = DDxSystem()
        success = ddx_system.initialize()
        if not success:
            return None, "Failed to initialize. Is Ollama running?"

    return ddx_system, "System ready"


# =============================================================================
# Interface Functions
# =============================================================================

def run_diagnosis(case_description: str, progress=gr.Progress()):
    """Run full diagnostic analysis on a case"""
    global ddx_system, last_result

    if not case_description.strip():
        return (
            "Please enter a clinical case description.",
            "",
            ""
        )

    # Initialize if needed
    if ddx_system is None:
        progress(0, desc="Initializing system...")
        ddx_system, status = initialize_system()
        if ddx_system is None:
            return (
                f"Initialization Error: {status}",
                "",
                ""
            )

    try:
        # Step 1: Analyze case and generate team
        progress(0.1, desc="Analyzing case...")
        analysis = ddx_system.analyze_case(case_description, "ui_case")

        if not analysis['success']:
            return (
                f"Analysis failed: {analysis.get('error', 'Unknown error')}",
                "",
                ""
            )

        team_info = format_team_info(analysis)

        # Step 2: Run diagnostic rounds
        progress(0.3, desc="Running differential diagnosis round...")
        time.sleep(0.5)

        progress(0.5, desc="Running clinical debate round...")
        progress(0.7, desc="Identifying can't-miss diagnoses...")

        result = ddx_system.run_full_diagnosis()

        if not result['success']:
            return (
                f"Diagnosis failed: {result.get('error', 'Unknown error')}",
                team_info,
                ""
            )

        progress(0.9, desc="Synthesizing results...")

        # Store for export
        last_result = {
            'case_description': case_description,
            'round_results': ddx_system.round_results,
            'final_diagnoses': result['final_diagnoses'],
            'analysis': analysis
        }

        # Format outputs
        diagnoses_md = format_diagnoses(result['final_diagnoses'])
        transcript_md = format_full_transcript_md(
            case_description,
            ddx_system.round_results,
            result['final_diagnoses'],
            analysis
        )

        progress(1.0, desc="Complete!")

        return (
            diagnoses_md,
            team_info,
            transcript_md
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"Error: {str(e)}",
            "",
            ""
        )


def format_team_info(analysis: dict) -> str:
    """Format the specialist team information"""
    specialists = analysis.get('specialists', [])

    if not specialists:
        return "No specialists generated"

    md = "### Specialist Team\n\n"
    for i, name in enumerate(specialists, 1):
        md += f"{i}. **{name}**\n"

    md += f"\n*{len(specialists)} specialists generated*"
    return md


def format_diagnoses(diagnoses: list) -> str:
    """Format final diagnoses as markdown"""
    if not diagnoses:
        return "No diagnoses generated"

    md = "## Differential Diagnoses\n\n"

    for i, dx in enumerate(diagnoses, 1):
        confidence = dx.get('confidence', 0)
        confidence_bar = get_confidence_bar(confidence)

        md += f"### {i}. {dx['diagnosis']}\n\n"
        md += f"**Confidence:** {confidence_bar} {confidence:.0%}\n\n"

        evidence = dx.get('evidence', [])
        if evidence:
            md += "**Supporting Evidence:**\n"
            for ev in evidence[:4]:
                md += f"- {ev}\n"
            md += "\n"

        supporters = dx.get('supporters', [])
        if supporters:
            md += f"*Supported by: {', '.join(supporters)}*\n\n"

        md += "---\n\n"

    return md


def get_confidence_bar(confidence: float) -> str:
    """Create a visual confidence bar"""
    filled = int(confidence * 10)
    empty = 10 - filled
    return "[" + "█" * filled + "░" * empty + "]"


def format_full_transcript_md(case_description: str, round_results: dict,
                               final_diagnoses: list, analysis: dict) -> str:
    """Format complete transcript as markdown (full, not truncated)"""
    if not round_results:
        return "No transcript available"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"## Full Diagnostic Transcript\n\n"
    md += f"*Generated: {timestamp}*\n\n"

    # Case
    md += "### Clinical Case\n\n"
    md += f"{case_description.strip()}\n\n"

    # Rounds
    round_names = {
        'differential': 'Round 1: Independent Differentials',
        'debate': 'Round 2: Clinical Debate',
        'cant_miss': 'Round 3: Can\'t-Miss Diagnoses'
    }

    for round_type, responses in round_results.items():
        round_name = round_names.get(round_type, round_type)
        md += f"---\n\n### {round_name}\n\n"

        for resp in responses:
            md += f"#### {resp.agent_name} ({resp.specialty})\n\n"
            md += f"*Confidence: {resp.confidence_score:.0%} | Response time: {resp.response_time:.1f}s*\n\n"

            # Structured diagnoses
            if resp.structured_data:
                md += "**Diagnoses:**\n"
                for dx, evidence in resp.structured_data.items():
                    ev_str = ', '.join(evidence[:3]) if isinstance(evidence, list) else str(evidence)
                    md += f"- **{dx}**: {ev_str}\n"
                md += "\n"

            # Full reasoning (NO TRUNCATION)
            md += "**Full Reasoning:**\n\n"
            md += f"{resp.content}\n\n"

    # Final synthesis
    md += "---\n\n### Final Differential Diagnosis\n\n"
    for i, dx in enumerate(final_diagnoses, 1):
        confidence = dx.get('confidence', 0)
        md += f"{i}. **{dx['diagnosis']}** ({confidence:.0%})\n"
        evidence = dx.get('evidence', [])
        if evidence:
            md += f"   - Evidence: {', '.join(evidence[:3])}\n"

    return md


def get_example_cases():
    """Return example cases for the interface"""
    return [
        ["""A 45-year-old male presents with acute chest pain that began 2 hours ago.
The pain is crushing, radiates to the left arm, and is associated with
shortness of breath and diaphoresis. He has a history of diabetes and
hypertension. ECG shows ST elevation in leads II, III, aVF. Troponin levels are elevated."""],

        ["""A 67-year-old female presents with progressive shortness of breath over 3 weeks,
productive cough with yellow sputum, and fever of 38.5°C. She has a 40 pack-year
smoking history and was diagnosed with COPD 5 years ago. Chest X-ray shows
right lower lobe consolidation. SpO2 is 89% on room air."""],

        ["""A 32-year-old female presents with fatigue, weight gain of 15 pounds over
3 months, constipation, and cold intolerance. She reports her hair has been
thinning and her skin is dry. Physical exam reveals bradycardia (HR 52),
delayed relaxation of deep tendon reflexes, and periorbital edema."""],

        ["""A 58-year-old male with a history of alcohol use disorder presents with
confusion, jaundice, and abdominal distension. He reports dark urine and
clay-colored stools for the past week. Physical exam reveals hepatomegaly,
ascites, spider angiomata, and asterixis. Labs show elevated AST, ALT,
bilirubin, and INR."""]
    ]


# =============================================================================
# Build Interface
# =============================================================================

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Local-DDx",
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.Markdown("""
        # Local-DDx: Collaborative Diagnostic AI

        **Multi-agent differential diagnosis powered by local LLMs**

        This system assembles a dynamic team of AI specialists who collaborate
        through structured diagnostic rounds to analyze clinical cases.

        *Running locally via Ollama - your data never leaves your machine.*
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                case_input = gr.Textbox(
                    label="Clinical Case Presentation",
                    placeholder="Enter the patient presentation, history, symptoms, and any test results...",
                    lines=12,
                    max_lines=20
                )

                with gr.Row():
                    run_btn = gr.Button("Run Diagnosis", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                gr.Markdown("### Example Cases")
                gr.Examples(
                    examples=get_example_cases(),
                    inputs=[case_input],
                    label=""
                )

            # Right column: Output
            with gr.Column(scale=1):
                with gr.Tab("Diagnoses"):
                    diagnoses_output = gr.Markdown(
                        value="*Enter a case and click 'Run Diagnosis' to begin*"
                    )

                with gr.Tab("Team"):
                    team_output = gr.Markdown(
                        value="*Specialist team will appear here*"
                    )

                with gr.Tab("Full Transcript"):
                    transcript_output = gr.Markdown(
                        value="*Full diagnostic reasoning will appear here (not truncated)*"
                    )

        # Footer
        gr.Markdown("""
        ---
        **Local-DDx** | Built for HSIL Hackathon 2026 | Collaborative AI for Rural Health

        *Architecture: Dynamic specialist generation, multi-round clinical reasoning, consensus synthesis*
        """)

        # Event handlers
        run_btn.click(
            fn=run_diagnosis,
            inputs=[case_input],
            outputs=[diagnoses_output, team_output, transcript_output],
            show_progress=True
        )

        clear_btn.click(
            fn=lambda: ("", "*Enter a case and click 'Run Diagnosis' to begin*",
                       "*Specialist team will appear here*",
                       "*Full diagnostic reasoning will appear here (not truncated)*"),
            inputs=[],
            outputs=[case_input, diagnoses_output, team_output, transcript_output]
        )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting Local-DDx Interface...")
    print("=" * 50)

    # Pre-initialize system
    print("Pre-initializing DDx system...")
    ddx_system, status = initialize_system()
    print(f"Status: {status}")

    # Launch interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
