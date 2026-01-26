# =============================================================================
# DDx Core - Ollama Edition
# =============================================================================

"""
DDx Core for Ollama backend: Multi-agent collaborative diagnosis system
adapted for local inference on Apple Silicon via Ollama.

Key changes from vLLM version:
- Uses Ollama REST API instead of vLLM
- Simplified model management (Ollama handles memory)
- Maintains: dynamic specialist generation, multi-model reasoning, sliding context
"""

import yaml
import time
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from inference_backends import (
    OllamaBackend,
    SamplingConfig,
    create_backend
)


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration for Ollama"""
    name: str
    model_name: str  # Ollama model identifier (e.g., "llama3.1:8b")
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    role: str = "balanced"  # "conservative", "innovative", "balanced"

    def to_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )


def load_system_config(config_path: str = "config.yaml") -> Dict[str, ModelConfig]:
    """Load system configuration"""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        print("Loaded config.yaml")
    except FileNotFoundError:
        print("Using default Ollama configuration")
        config_data = {
            'conservative_model': {
                'name': 'Conservative',
                'model_name': 'llama3.1:8b',
                'temperature': 0.3,
                'top_p': 0.7,
                'max_tokens': 1024,
                'role': 'conservative'
            },
            'innovative_model': {
                'name': 'Innovative',
                'model_name': 'llama3.1:8b',
                'temperature': 0.8,
                'top_p': 0.95,
                'max_tokens': 1024,
                'role': 'innovative'
            }
        }

    return {
        key: ModelConfig(**values)
        for key, values in config_data.items()
        if key in ['conservative_model', 'innovative_model']
    }


# =============================================================================
# 2. Model Manager for Ollama
# =============================================================================

class OllamaModelManager:
    """
    Manages multiple model configurations via Ollama.

    Unlike vLLM which loads models into GPU memory, Ollama manages
    model lifecycle automatically. We track configurations and
    which model is "active" for generation.
    """

    def __init__(self, configs: Dict[str, ModelConfig]):
        self.configs = configs
        self.backend = create_backend("ollama")
        self.active_model_id: Optional[str] = None
        self.loaded_models: set = set()

        print(f"Initialized OllamaModelManager with {len(configs)} configurations")

    def initialize(self) -> bool:
        """Initialize and verify Ollama is available"""
        if not self.backend.is_available():
            print("Ollama is not running. Start with: ollama serve")
            return False

        available = self.backend.get_available_models()
        print(f"Ollama available with {len(available)} models")

        # Verify configured models exist
        for config_id, config in self.configs.items():
            if config.model_name not in available:
                print(f"Warning: {config.model_name} not found in Ollama")
                print(f"  Pull with: ollama pull {config.model_name}")

        return True

    def load_model(self, model_id: str) -> bool:
        """Load/warm up a model configuration"""
        if model_id not in self.configs:
            print(f"Unknown model config: {model_id}")
            return False

        config = self.configs[model_id]

        # Load via backend
        if self.backend.load(config.model_name):
            self.active_model_id = model_id
            self.loaded_models.add(model_id)
            return True

        return False

    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model"""
        return self.configs.get(model_id)

    def generate(self, model_id: str, prompt: str,
                 temperature_override: Optional[float] = None) -> str:
        """Generate using a specific model configuration"""
        config = self.configs.get(model_id)
        if not config:
            return f"Error: Unknown model {model_id}"

        # Ensure model is loaded
        if model_id != self.active_model_id:
            self.backend.load(config.model_name)
            self.active_model_id = model_id

        # Create sampling config with optional override
        sampling = config.to_sampling_config()
        if temperature_override is not None:
            sampling.temperature = temperature_override

        return self.backend.generate(prompt, sampling)

    def generate_chat(self, model_id: str, messages: List[Dict[str, str]],
                      temperature_override: Optional[float] = None) -> str:
        """Generate using chat format"""
        config = self.configs.get(model_id)
        if not config:
            return f"Error: Unknown model {model_id}"

        if model_id != self.active_model_id:
            self.backend.load(config.model_name)
            self.active_model_id = model_id

        sampling = config.to_sampling_config()
        if temperature_override is not None:
            sampling.temperature = temperature_override

        return self.backend.generate_chat(messages, sampling)

    def get_available_models(self) -> List[str]:
        """Get list of configured model IDs"""
        return list(self.configs.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get current manager status"""
        return {
            "backend": "ollama",
            "available": self.backend.is_available(),
            "configured_models": list(self.configs.keys()),
            "active_model": self.active_model_id,
            "loaded_models": list(self.loaded_models)
        }


# =============================================================================
# 3. Specialty System
# =============================================================================

class SpecialtyType(Enum):
    """Medical specialties - expandable for dynamic generation"""
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    ENDOCRINOLOGY = "Endocrinology"
    GASTROENTEROLOGY = "Gastroenterology"
    NEUROLOGY = "Neurology"
    NEPHROLOGY = "Nephrology"
    HEMATOLOGY = "Hematology"
    INFECTIOUS_DISEASE = "Infectious Disease"
    EMERGENCY_MEDICINE = "Emergency Medicine"
    CRITICAL_CARE = "Critical Care"
    INTERNAL_MEDICINE = "Internal Medicine"
    RHEUMATOLOGY = "Rheumatology"
    ONCOLOGY = "Oncology"
    SURGERY = "Surgery"
    GENERAL = "General Medicine"


# =============================================================================
# 4. Agent Framework
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for a specialist agent"""
    name: str
    specialty: SpecialtyType
    persona: str
    reasoning_style: str
    temperature: float
    focus_areas: List[str]
    model_id: str = "conservative_model"
    case_relevance_score: float = 0.0


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    specialty: str
    content: str
    structured_data: Optional[Dict] = None
    response_time: float = 0.0
    round_type: str = ""
    confidence_score: float = 0.0


class DDxAgent:
    """
    A specialist agent that generates diagnostic reasoning.

    Uses the model manager for inference while maintaining
    individual personality and expertise configuration.
    """

    def __init__(self, config: AgentConfig, model_manager: OllamaModelManager):
        self.config = config
        self.model_manager = model_manager
        self.conversation_history: List[Dict] = []

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def specialty(self) -> SpecialtyType:
        return self.config.specialty

    def generate_response(self, prompt: str, round_type: str = "general",
                          context: Optional[str] = None) -> AgentResponse:
        """Generate a response for the given prompt"""
        start_time = time.time()

        try:
            # Build messages for chat format
            messages = self._build_messages(prompt, round_type, context)

            # Generate via model manager
            response_content = self.model_manager.generate_chat(
                self.config.model_id,
                messages,
                temperature_override=self.config.temperature
            )

            # Extract structured data if applicable
            structured_data = None
            if round_type in ["differential", "ranking", "voting", "cant_miss"]:
                structured_data = self._extract_structured_data(response_content)

            response_time = time.time() - start_time
            confidence = self._calculate_confidence(response_content)

            response = AgentResponse(
                agent_name=self.name,
                specialty=self.specialty.value,
                content=response_content,
                structured_data=structured_data,
                response_time=response_time,
                round_type=round_type,
                confidence_score=confidence
            )

            # Track history
            self.conversation_history.append({
                'prompt': prompt,
                'response': response_content,
                'round_type': round_type,
                'timestamp': time.time()
            })

            return response

        except Exception as e:
            print(f"Error in {self.name}: {e}")
            return AgentResponse(
                agent_name=self.name,
                specialty=self.specialty.value,
                content=f"Error: {str(e)}",
                response_time=time.time() - start_time,
                round_type=round_type
            )

    def _build_messages(self, user_prompt: str, round_type: str,
                        context: Optional[str]) -> List[Dict[str, str]]:
        """Build chat messages with system prompt"""

        system_prompt = f"""You are {self.name}, a medical specialist in {self.specialty.value}.
{self.config.persona}

Your expertise: {', '.join(self.config.focus_areas)}
Reasoning style: {self.config.reasoning_style}

When providing differential diagnoses, structure your response with:
1. A JSON object containing diagnoses and evidence
2. Followed by your clinical reasoning

JSON format example:
{{"diagnosis_name": ["evidence1", "evidence2"], "another_diagnosis": ["evidence1"]}}"""

        if round_type == "differential":
            system_prompt += """

TASK: Provide 3-5 differential diagnoses from your specialty perspective.
Start with a JSON object, then explain your reasoning."""

        elif round_type == "debate":
            system_prompt += """

TASK: Review your colleagues' diagnoses. Challenge, support, or refine
based on clinical evidence. Be specific about agreements and disagreements."""

        elif round_type == "cant_miss":
            system_prompt += """

TASK: Identify critical diagnoses that cannot be missed from a safety perspective.
Focus on conditions that could cause serious harm if delayed."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add context if available
        if context:
            messages.append({
                "role": "assistant",
                "content": f"Previous team discussion:\n{context}"
            })

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _extract_structured_data(self, content: str) -> Optional[Dict]:
        """Extract JSON diagnoses from response"""
        try:
            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean common issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                data = json.loads(json_str)
                if isinstance(data, dict) and len(data) > 0:
                    return data
        except (json.JSONDecodeError, Exception):
            pass

        return None

    def _calculate_confidence(self, content: str) -> float:
        """Estimate confidence based on response quality"""
        score = 0.5

        # Has structured data
        if self._extract_structured_data(content):
            score += 0.2

        # Uses medical terminology
        medical_terms = len(re.findall(
            r'\b(?:diagnosis|evidence|findings|symptoms|clinical|pathology)\b',
            content, re.IGNORECASE
        ))
        score += min(0.2, medical_terms * 0.02)

        # Has sufficient detail
        if len(content) > 200:
            score += 0.1

        return min(1.0, score)


# =============================================================================
# 5. Dynamic Agent Generator
# =============================================================================

class DynamicAgentGenerator:
    """
    Generates specialist teams dynamically based on case presentation.

    Uses the LLM to analyze cases and propose appropriate specialists,
    maintaining the innovative dual-model approach where possible.
    """

    def __init__(self, model_manager: OllamaModelManager):
        self.model_manager = model_manager

    def generate_agents(self, case_description: str,
                        max_specialists: int = 8) -> List[DDxAgent]:
        """Generate a team of specialists for the case"""
        print(f"\nGenerating specialist team for case...")

        # Get specialist proposals from LLM
        proposals = self._get_specialist_proposals(case_description)

        if not proposals:
            print("Using fallback specialists")
            proposals = self._fallback_proposals()

        # Limit to max
        proposals = proposals[:max_specialists]

        # Create agents
        agents = []
        model_ids = self.model_manager.get_available_models()

        for i, proposal in enumerate(proposals):
            # Alternate between models for diversity
            model_id = model_ids[i % len(model_ids)] if model_ids else "conservative_model"

            try:
                specialty = self._map_specialty(proposal.get('specialty', 'General'))
                temp = self._calculate_temperature(proposal, model_id)

                config = AgentConfig(
                    name=proposal.get('name', f'Dr. Specialist {i+1}'),
                    specialty=specialty,
                    persona=proposal.get('persona', 'Experienced specialist'),
                    reasoning_style=proposal.get('reasoning_style', 'analytical'),
                    temperature=temp,
                    focus_areas=proposal.get('focus_areas', ['general medicine']),
                    model_id=model_id
                )

                agent = DDxAgent(config, self.model_manager)
                agents.append(agent)

                model_type = "Conservative" if "conservative" in model_id else "Innovative"
                print(f"  Created: {agent.name} ({agent.specialty.value}) [{model_type}]")

            except Exception as e:
                print(f"  Failed to create agent: {e}")

        print(f"Generated {len(agents)} specialists")
        return agents

    def _get_specialist_proposals(self, case_description: str) -> List[Dict]:
        """Ask LLM to propose specialists for the case"""

        prompt = f"""Analyze this medical case and propose a team of 4-6 specialists needed for diagnosis.

CASE:
{case_description}

Respond with a JSON array of specialists:
[
  {{
    "name": "Dr. [Realistic Name]",
    "specialty": "[Medical Specialty]",
    "persona": "[Brief description of approach]",
    "reasoning_style": "[analytical/intuitive/systematic/innovative]",
    "focus_areas": ["area1", "area2"],
    "rationale": "[Why this specialist is needed]"
  }}
]

Choose specialists whose expertise matches the case presentation.
Include at least one generalist (Internal Medicine or Emergency Medicine).
"""

        messages = [
            {"role": "system", "content": "You are a medical team coordinator. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use conservative model for team planning
            response = self.model_manager.generate_chat("conservative_model", messages)

            # Extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)

                proposals = json.loads(json_str)
                if isinstance(proposals, list) and len(proposals) >= 2:
                    print(f"  LLM proposed {len(proposals)} specialists")
                    return proposals

        except Exception as e:
            print(f"  Proposal generation failed: {e}")

        return []

    def _fallback_proposals(self) -> List[Dict]:
        """Fallback specialists when LLM generation fails"""
        return [
            {
                "name": "Dr. Sarah Chen",
                "specialty": "Internal Medicine",
                "persona": "Systematic internist with broad diagnostic experience",
                "reasoning_style": "systematic",
                "focus_areas": ["general medicine", "diagnostic reasoning"]
            },
            {
                "name": "Dr. Michael Torres",
                "specialty": "Emergency Medicine",
                "persona": "Rapid assessment specialist focused on acute presentations",
                "reasoning_style": "analytical",
                "focus_areas": ["acute care", "triage"]
            },
            {
                "name": "Dr. Emily Watson",
                "specialty": "Cardiology",
                "persona": "Cardiologist with expertise in complex presentations",
                "reasoning_style": "methodical",
                "focus_areas": ["cardiac disease", "vascular conditions"]
            },
            {
                "name": "Dr. James Park",
                "specialty": "Infectious Disease",
                "persona": "ID specialist attuned to subtle infection patterns",
                "reasoning_style": "intuitive",
                "focus_areas": ["infections", "immunology"]
            }
        ]

    def _map_specialty(self, specialty_str: str) -> SpecialtyType:
        """Map specialty string to enum"""
        specialty_lower = specialty_str.lower()

        mappings = {
            'cardio': SpecialtyType.CARDIOLOGY,
            'heart': SpecialtyType.CARDIOLOGY,
            'pulmon': SpecialtyType.PULMONOLOGY,
            'lung': SpecialtyType.PULMONOLOGY,
            'respiratory': SpecialtyType.PULMONOLOGY,
            'endocrin': SpecialtyType.ENDOCRINOLOGY,
            'gastro': SpecialtyType.GASTROENTEROLOGY,
            'gi': SpecialtyType.GASTROENTEROLOGY,
            'neuro': SpecialtyType.NEUROLOGY,
            'nephro': SpecialtyType.NEPHROLOGY,
            'kidney': SpecialtyType.NEPHROLOGY,
            'renal': SpecialtyType.NEPHROLOGY,
            'hemato': SpecialtyType.HEMATOLOGY,
            'blood': SpecialtyType.HEMATOLOGY,
            'infectious': SpecialtyType.INFECTIOUS_DISEASE,
            'infection': SpecialtyType.INFECTIOUS_DISEASE,
            'emergency': SpecialtyType.EMERGENCY_MEDICINE,
            'critical': SpecialtyType.CRITICAL_CARE,
            'icu': SpecialtyType.CRITICAL_CARE,
            'internal': SpecialtyType.INTERNAL_MEDICINE,
            'rheumat': SpecialtyType.RHEUMATOLOGY,
            'oncol': SpecialtyType.ONCOLOGY,
            'cancer': SpecialtyType.ONCOLOGY,
            'surgery': SpecialtyType.SURGERY,
        }

        for key, value in mappings.items():
            if key in specialty_lower:
                return value

        return SpecialtyType.GENERAL

    def _calculate_temperature(self, proposal: Dict, model_id: str) -> float:
        """Calculate temperature based on model and reasoning style"""
        # Base from model type
        base = 0.3 if 'conservative' in model_id else 0.7

        # Adjust for reasoning style
        style = proposal.get('reasoning_style', 'analytical').lower()
        style_mods = {
            'analytical': -0.1,
            'systematic': -0.1,
            'methodical': -0.1,
            'innovative': +0.15,
            'creative': +0.15,
            'intuitive': +0.1
        }

        modifier = style_mods.get(style, 0.0)

        return max(0.1, min(0.95, base + modifier))


# =============================================================================
# 6. Main DDx System
# =============================================================================

class DDxSystem:
    """
    Main diagnostic system orchestrating agents and rounds.

    Simplified for Ollama while maintaining core collaborative diagnosis.
    """

    def __init__(self):
        self.model_manager: Optional[OllamaModelManager] = None
        self.agent_generator: Optional[DynamicAgentGenerator] = None
        self.current_agents: List[DDxAgent] = []
        self.case_data: Dict = {}
        self.round_results: Dict = {}

    def initialize(self, config_path: str = "config.yaml") -> bool:
        """Initialize the system"""
        print("Initializing DDx System (Ollama Edition)...")

        # Load configuration
        configs = load_system_config(config_path)
        self.model_manager = OllamaModelManager(configs)

        if not self.model_manager.initialize():
            print("Failed to initialize model manager")
            return False

        # Warm up models
        for model_id in self.model_manager.get_available_models():
            self.model_manager.load_model(model_id)

        # Initialize agent generator
        self.agent_generator = DynamicAgentGenerator(self.model_manager)

        print("DDx System ready!")
        return True

    def analyze_case(self, case_description: str,
                     case_name: str = "case") -> Dict[str, Any]:
        """Analyze a case - generate specialists and prepare for diagnosis"""
        print(f"\nAnalyzing case: {case_name}")
        print("=" * 50)

        self.case_data = {
            'description': case_description,
            'name': case_name,
            'timestamp': time.time()
        }

        # Generate specialist team
        self.current_agents = self.agent_generator.generate_agents(
            case_description,
            max_specialists=6
        )

        if not self.current_agents:
            return {'success': False, 'error': 'Failed to generate specialists'}

        return {
            'success': True,
            'case_name': case_name,
            'agents_generated': len(self.current_agents),
            'specialists': [a.name for a in self.current_agents]
        }

    def run_diagnostic_round(self, round_type: str = "differential",
                              context: Optional[str] = None) -> Dict[str, Any]:
        """Run a single diagnostic round with all agents"""
        if not self.current_agents:
            return {'success': False, 'error': 'No agents available'}

        print(f"\nRunning {round_type} round...")
        print("-" * 40)

        results = []
        case_prompt = f"CLINICAL CASE:\n{self.case_data.get('description', '')}"

        for agent in self.current_agents:
            print(f"  {agent.name} analyzing...")
            response = agent.generate_response(case_prompt, round_type, context)
            results.append(response)
            print(f"    -> {len(response.content)} chars, confidence: {response.confidence_score:.2f}")

        self.round_results[round_type] = results

        return {
            'success': True,
            'round_type': round_type,
            'responses': len(results),
            'results': results
        }

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run complete diagnostic workflow"""
        if not self.case_data:
            return {'success': False, 'error': 'No case loaded'}

        # Round 1: Independent differentials
        r1 = self.run_diagnostic_round("differential")

        # Build context from round 1
        context = self._build_round_context(r1.get('results', []))

        # Round 2: Debate and refinement
        r2 = self.run_diagnostic_round("debate", context)

        # Round 3: Can't miss diagnoses
        full_context = context + "\n\n" + self._build_round_context(r2.get('results', []))
        r3 = self.run_diagnostic_round("cant_miss", full_context)

        # Synthesize final diagnosis list
        final_diagnoses = self._synthesize_diagnoses()

        return {
            'success': True,
            'case_name': self.case_data.get('name'),
            'rounds_completed': 3,
            'final_diagnoses': final_diagnoses,
            'round_results': self.round_results
        }

    def _build_round_context(self, responses: List[AgentResponse]) -> str:
        """Build context summary from round responses"""
        context_parts = []

        for resp in responses:
            if resp.structured_data:
                diagnoses = list(resp.structured_data.keys())
                context_parts.append(
                    f"{resp.agent_name} ({resp.specialty}): {', '.join(diagnoses[:3])}"
                )
            else:
                # Extract key points from text
                snippet = resp.content[:200] if resp.content else "No response"
                context_parts.append(f"{resp.agent_name}: {snippet}...")

        return "\n".join(context_parts)

    def _synthesize_diagnoses(self) -> List[Dict[str, Any]]:
        """Synthesize final diagnosis list from all rounds"""
        diagnosis_votes: Dict[str, Dict] = {}

        # Collect all diagnoses with evidence
        for round_type, responses in self.round_results.items():
            for resp in responses:
                if resp.structured_data:
                    for diagnosis, evidence in resp.structured_data.items():
                        diag_key = diagnosis.lower().strip()

                        if diag_key not in diagnosis_votes:
                            diagnosis_votes[diag_key] = {
                                'name': diagnosis,
                                'votes': 0,
                                'evidence': set(),
                                'supporters': []
                            }

                        diagnosis_votes[diag_key]['votes'] += 1
                        diagnosis_votes[diag_key]['supporters'].append(resp.agent_name)

                        if isinstance(evidence, list):
                            diagnosis_votes[diag_key]['evidence'].update(evidence)

        # Sort by votes
        sorted_diagnoses = sorted(
            diagnosis_votes.values(),
            key=lambda x: x['votes'],
            reverse=True
        )

        # Format output
        return [
            {
                'diagnosis': d['name'],
                'confidence': min(1.0, d['votes'] / len(self.current_agents)),
                'evidence': list(d['evidence'])[:5],
                'supporters': d['supporters']
            }
            for d in sorted_diagnoses[:7]
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.model_manager is not None,
            'model_status': self.model_manager.get_status() if self.model_manager else None,
            'agents_loaded': len(self.current_agents),
            'case_loaded': bool(self.case_data),
            'rounds_completed': len(self.round_results)
        }


# =============================================================================
# Testing
# =============================================================================

def test_ddx_system():
    """Test the Ollama-based DDx system"""
    print("Testing DDx System (Ollama Edition)")
    print("=" * 60)

    system = DDxSystem()

    if not system.initialize():
        print("Failed to initialize")
        return False

    # Test case
    test_case = """
    A 45-year-old male presents with acute chest pain that began 2 hours ago.
    The pain is crushing, radiates to the left arm, and is associated with
    shortness of breath and diaphoresis. He has a history of diabetes and
    hypertension. ECG shows ST elevation in leads II, III, aVF.
    Troponin levels are elevated.
    """

    # Analyze case
    analysis = system.analyze_case(test_case, "chest_pain_test")
    if not analysis['success']:
        print(f"Analysis failed: {analysis.get('error')}")
        return False

    print(f"\nGenerated {analysis['agents_generated']} specialists")

    # Run full diagnosis
    result = system.run_full_diagnosis()

    if result['success']:
        print(f"\n{'='*60}")
        print("FINAL DIAGNOSES:")
        print("=" * 60)

        for i, dx in enumerate(result['final_diagnoses'], 1):
            print(f"\n{i}. {dx['diagnosis']}")
            print(f"   Confidence: {dx['confidence']:.0%}")
            print(f"   Evidence: {', '.join(dx['evidence'][:3])}")
            print(f"   Supported by: {', '.join(dx['supporters'])}")

        return True

    return False


if __name__ == "__main__":
    test_ddx_system()
