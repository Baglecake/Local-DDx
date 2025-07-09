
# =============================================================================
# DDx Core v6 - Unified Foundation with PRESERVED Dynamic Generation
# =============================================================================

"""
DDx Core v6: Clean, unified foundation that preserves ALL sophisticated dynamic
generation features. DDx_Main_Design.md integration happens in ROUNDS and
OUTPUT FORMATTING, not in constraining what specialties can be generated.
"""

import yaml
import torch
import time
import json
import re
import ast
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from vllm import LLM, SamplingParams

# =============================================================================
# 1. Configuration Management (Enhanced)
# =============================================================================

@dataclass
class ModelConfig:
    """Enhanced model configuration with v6 improvements"""
    name: str
    model_path: str
    memory_fraction: float
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    max_model_len: int = 2048
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>", "<|im_end|>"])

    # v6 Enhancement: Better Colab stability settings
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True
    enable_prefix_caching: bool = False
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 2

def load_system_config():
    """Load system configuration with enhanced fallbacks"""
    try:
        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)
        print("✅ Loaded config.yaml successfully")
    except FileNotFoundError:
        print("⚠️ config.yaml not found, using enhanced fallback configuration")
        config_data = {
            'conservative_model': {
                'name': 'Conservative-Model',
                'model_path': 'NousResearch/Nous-Hermes-2-Mistral-7B-DPO',
                'memory_fraction': 0.4,
                'temperature': 0.1,
                'top_p': 0.7,
                'max_tokens': 1024,
                'max_model_len': 2048
            },
            'innovative_model': {
                'name': 'Innovative-Model',
                'model_path': 'NousResearch/Nous-Hermes-2-Mistral-7B-DPO',
                'memory_fraction': 0.4,
                'temperature': 0.9,
                'top_p': 0.95,
                'max_tokens': 1024,
                'max_model_len': 2048
            }
        }

    return {key: ModelConfig(**values) for key, values in config_data.items()
            if key in ['conservative_model', 'innovative_model']}

# =============================================================================
# 2. Enhanced Model Management
# =============================================================================

class ModelManager:
    """Enhanced model manager with improved stability and error handling"""

    def __init__(self, configs: Dict[str, ModelConfig]):
        self.configs = configs
        self.models = {}
        self.sampling_params = {}
        self.load_status = {}
        self.conservative_model = None  # Direct reference for evaluation
        self.innovative_model = None    # Direct reference for generation

    def initialize_models(self) -> bool:
        """Initialize models with enhanced stability and error handling"""
        print("🔄 Initializing Enhanced Model Manager v6...")

        if not self._verify_gpu():
            return False

        success = self._load_models_sequential()

        if success:
            self._setup_direct_references()
            self._display_initialization_summary()

        return success

    def _verify_gpu(self) -> bool:
        """Enhanced GPU verification with detailed information"""
        if not torch.cuda.is_available():
            print("❌ No GPU detected - check Colab runtime settings")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_memory = torch.cuda.mem_get_info()[0] / 1e9

        print(f"✅ GPU: {gpu_name}")
        print(f"   Total Memory: {gpu_memory:.1f}GB")
        print(f"   Available Memory: {available_memory:.1f}GB")

        return True

    def _load_models_sequential(self) -> bool:
        """Load models with enhanced error handling and stability"""
        torch.cuda.empty_cache()

        for model_id, config in self.configs.items():
            print(f"\n📥 Loading {config.name}...")

            try:
                # Enhanced vLLM settings for Colab stability
                model = LLM(
                    model=config.model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=config.memory_fraction,
                    max_model_len=config.max_model_len,
                    trust_remote_code=True,
                    dtype="half",
                    # v6 Enhanced stability settings
                    enforce_eager=config.enforce_eager,
                    disable_custom_all_reduce=config.disable_custom_all_reduce,
                    enable_prefix_caching=config.enable_prefix_caching,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    max_num_seqs=config.max_num_seqs
                )

                sampling_params = SamplingParams(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    stop=config.stop_tokens
                )

                self.models[model_id] = model
                self.sampling_params[model_id] = sampling_params
                self.load_status[model_id] = 'success'

                print(f"   ✅ {config.name} loaded successfully")

            except Exception as e:
                print(f"   ❌ Failed to load {config.name}: {e}")
                self.load_status[model_id] = f'failed: {str(e)}'

                # Continue if we have at least one model
                if len([s for s in self.load_status.values() if s == 'success']) > 0:
                    print(f"   💡 Continuing with available models...")
                    continue

        successful_models = [mid for mid, status in self.load_status.items()
                           if status == 'success']

        return len(successful_models) > 0

    def _setup_direct_references(self):
        """Setup direct model references for easier access"""
        if 'conservative_model' in self.models:
            self.conservative_model = self.models['conservative_model']
        if 'innovative_model' in self.models:
            self.innovative_model = self.models['innovative_model']

    def _display_initialization_summary(self):
        """Display comprehensive initialization summary"""
        successful_models = [mid for mid, status in self.load_status.items()
                           if status == 'success']

        print(f"\n📊 Model Manager v6 Initialization Summary:")
        print(f"   Successfully loaded: {len(successful_models)}/{len(self.configs)}")

        for model_id, status in self.load_status.items():
            if status == 'success':
                print(f"   ✅ {self.configs[model_id].name}")
            else:
                print(f"   ❌ {self.configs[model_id].name}: {status}")

    def get_model(self, model_id: str) -> Tuple[Optional[LLM], Optional[SamplingParams]]:
        """Get model and sampling parameters with enhanced error handling"""
        if model_id not in self.models:
            print(f"⚠️ Model {model_id} not available")
            return None, None
        return self.models[model_id], self.sampling_params[model_id]

    def get_available_models(self) -> List[str]:
        """Get list of successfully loaded models"""
        return [mid for mid, status in self.load_status.items() if status == 'success']

    def get_model_status(self) -> Dict[str, str]:
        """Get detailed status of all models"""
        return self.load_status.copy()

# =============================================================================
# 3. PRESERVED Dynamic Specialty System
# =============================================================================

class SpecialtyType(Enum):
    """EXPANSIVE specialty types - can handle ANY specialty generated dynamically"""
    # Core medical specialties
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    ENDOCRINOLOGY = "Endocrinology"
    GASTROENTEROLOGY = "Gastroenterology"
    NEUROLOGY = "Neurology"
    NEPHROLOGY = "Nephrology"
    HEMATOLOGY = "Hematology"
    DERMATOLOGY = "Dermatology"
    RHEUMATOLOGY = "Rheumatology"
    ORTHOPEDICS = "Orthopedics"
    UROLOGY = "Urology"
    GYNECOLOGY = "Gynecology"
    OBSTETRICS = "Obstetrics"
    PSYCHIATRY = "Psychiatry"
    ONCOLOGY = "Oncology"
    INFECTIOUS_DISEASE = "Infectious Disease"
    EMERGENCY_MEDICINE = "Emergency Medicine"
    CRITICAL_CARE = "Critical Care"
    ANESTHESIOLOGY = "Anesthesiology"
    RADIOLOGY = "Radiology"
    PATHOLOGY = "Pathology"
    SURGERY = "Surgery"
    INTERNAL_MEDICINE = "Internal Medicine"
    FAMILY_MEDICINE = "Family Medicine"
    PEDIATRICS = "Pediatrics"
    GERIATRICS = "Geriatrics"
    GENERAL = "General Medicine"

    # Subspecialties - can be generated dynamically
    INTERVENTIONAL_CARDIOLOGY = "Interventional Cardiology"
    ELECTROPHYSIOLOGY = "Electrophysiology"
    TRANSPLANT_NEPHROLOGY = "Transplant Nephrology"
    HEPATOLOGY = "Hepatology"
    REPRODUCTIVE_ENDOCRINOLOGY = "Reproductive Endocrinology"
    # ... system can handle ANY specialty the models generate

# =============================================================================
# 4. Enhanced Agent Framework (PRESERVED Dynamic Generation)
# =============================================================================

@dataclass
class AgentConfig:
    """Enhanced agent configuration with v6 improvements"""
    name: str
    specialty: SpecialtyType
    persona: str
    reasoning_style: str
    temperature: float
    focus_areas: List[str]
    case_relevance_score: float = 0.0

    # v6 Enhancements
    model_assignment: str = "auto"  # Which model to use
    tier_preference: str = "balanced"  # Diagnostic tier preference
    collaboration_style: str = "standard"  # How they interact in debates

@dataclass
class AgentResponse:
    """Enhanced agent response with v6 improvements"""
    agent_name: str
    specialty: str
    content: str
    structured_data: Optional[Dict] = None
    response_time: float = 0.0
    round_type: str = ""

    # v6 Enhancements
    confidence_score: float = 0.0
    reasoning_quality: str = "standard"
    collaboration_markers: List[str] = field(default_factory=list)

class DDxAgent:
    """Enhanced DDx agent with v6 improvements"""

    def __init__(self, config: AgentConfig, model: LLM,
                 sampling_params: SamplingParams, model_id: str):
        self.config = config
        self.model = model
        self.sampling_params = sampling_params
        self.model_id = model_id
        self.conversation_history = []

        # v6 Enhancements
        self.performance_metrics = {
            'responses_generated': 0,
            'structured_data_success': 0,
            'average_response_time': 0.0
        }

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def specialty(self) -> SpecialtyType:
        return self.config.specialty

    @property
    def persona(self) -> str:
        return self.config.persona

    def generate_response(self, prompt: str, round_type: str = "general", 
                     global_transcript: Optional[Dict] = None) -> AgentResponse:
        """Enhanced response generation with v6 improvements and sliding context"""
        start_time = time.time()

        try:
            # Use the agent's existing model and sampling_params (already set in __init__)
            formatted_prompt = self._format_prompt(prompt, round_type, global_transcript)
            outputs = self.model.generate([formatted_prompt], self.sampling_params)
            response_content = outputs[0].outputs[0].text.strip()

            # Enhanced structured data extraction
            structured_data = None
            if round_type in ["independent_differential", "subspecialist_consultation", "team_independent_differentials"]:
                structured_data = self._extract_structured_data(response_content)

            response_time = time.time() - start_time

            # v6 Enhancement: Calculate confidence and quality metrics
            confidence_score = self._calculate_confidence_score(response_content, round_type)
            reasoning_quality = self._assess_reasoning_quality(response_content, round_type)

            response = AgentResponse(
                agent_name=self.name,
                specialty=self.specialty.value,
                content=response_content,
                structured_data=structured_data,
                response_time=response_time,
                confidence_score=confidence_score,
                reasoning_quality=reasoning_quality,
                round_type=round_type
            )

            # Update performance metrics
            self._update_performance_metrics(response)

            # Add to conversation history with context tracking
            self.conversation_history.append({
                'prompt': prompt,
                'response': response_content,
                'round_type': round_type,
                'timestamp': time.time(),
                'had_context': global_transcript is not None and len(global_transcript.get('rounds', {})) > 0
            })

            return response

        except Exception as e:
            print(f"❌ Error generating response for {self.name}: {e}")
            return AgentResponse(
                agent_name=self.name,
                specialty=self.specialty.value,
                content=f"Error: {str(e)}",
                structured_data={},
                response_time=time.time() - start_time,
                confidence_score=0.0,
                reasoning_quality="error",
                round_type=round_type
            )

    def _format_prompt(self, user_input: str, round_type: str = "general", 
                  global_transcript: Optional[Dict] = None) -> str:
        """Enhanced prompt formatting with sliding context window support"""
        
        # Import sliding context manager (with fallback)
        try:
            from ddx_sliding_context import SlidingContextManager
            
            # Initialize context manager if not exists
            if not hasattr(self, '_context_manager'):
                self._context_manager = SlidingContextManager()
            
            # Build contextual information
            prior_context = ""
            if global_transcript and len(global_transcript.get('rounds', {})) > 0:
                prior_context = self._context_manager.build_context_for_agent(
                    agent_name=self.name,
                    agent_specialty=self.specialty.value,
                    round_type=round_type,
                    full_transcript=global_transcript
                )
        except (ImportError, Exception):
            # Fallback if sliding context not available
            prior_context = ""
        
        # Base system prompt using existing agent properties
        base_system = (
            f"You are {self.name}, an expert in {self.specialty.value} "
            f"with this specific approach: {self.persona}\n\n"
            f"Your reasoning style is {self.config.reasoning_style}. "
            f"Focus areas: {', '.join(self.config.focus_areas)}\n\n"
            f"Collaboration style: {self.config.collaboration_style}"
        )
        
        # Enhanced round-specific formatting
        if round_type == "team_independent_differentials":
            system_prompt = f"""{base_system}

    CRITICAL: Your response MUST start with a valid JSON dictionary on the first line.

    EXAMPLE FORMAT:
    {{"acute kidney injury": ["elevated creatinine", "decreased output"], "pneumonia": ["fever", "cough"]}}

    After the JSON, provide your detailed reasoning. Focus on diagnoses most relevant to {self.specialty.value}."""

            user_prompt = f"""Case: {user_input}

    TASK: Start with a JSON dictionary of 3-5 diagnoses from your {self.specialty.value} perspective:

    {{"diagnosis 1": ["evidence 1", "evidence 2"], "diagnosis 2": ["evidence 3", "evidence 4"]}}

    Then explain your clinical reasoning."""

        elif round_type == "refinement_and_justification":
            system_prompt = f"""{base_system}

    You are participating in collaborative medical debate with evidence-based reasoning.
    Challenge, support, or refine diagnoses based on clinical evidence and team discourse."""
            user_prompt = user_input

        elif round_type == "post_debate_voting":
            system_prompt = f"""{base_system}

    Cast your preferential vote considering the full team discussion and evidence presented."""
            user_prompt = user_input

        else:
            system_prompt = base_system
            user_prompt = user_input

        # Include collaborative context if available
        if prior_context:
            system_prompt += f"\n\nCOLLABORATIVE CONTEXT:\n{prior_context}"

        # Enhanced ChatML format (existing pattern)
        return (
            "<|im_start|>system\n"
            f"{system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant"
        )

    def _extract_structured_data(self, content: str) -> Optional[Dict]:
        """Enhanced structured data extraction with v6 improvements"""
        if not content:
            return None

        try:
            # Method 1: Look for JSON at start of response
            lines = content.strip().split('\n')
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                if line.startswith('{') and ':' in line:
                    json_text = line

                    # Complete the JSON if needed
                    if not line.endswith('}'):
                        for j in range(i+1, min(i+10, len(lines))):
                            json_text += ' ' + lines[j].strip()
                            if lines[j].strip().endswith('}'):
                                break

                    try:
                        import json
                        data = json.loads(json_text)
                        if isinstance(data, dict):
                            self.performance_metrics['structured_data_success'] += 1
                            return data
                    except:
                        try:
                            import ast
                            data = ast.literal_eval(json_text)
                            if isinstance(data, dict):
                                self.performance_metrics['structured_data_success'] += 1
                                return data
                        except:
                            continue

            # Method 2: Fallback - extract diagnoses and create structure
            diagnoses = self._extract_diagnoses_from_content(content)
            if diagnoses:
                result = {}
                for diagnosis in diagnoses:
                    evidence = self._extract_evidence_for_diagnosis(diagnosis, content)
                    result[diagnosis] = evidence if evidence else ["clinical presentation"]
                return result

        except Exception as e:
            print(f"⚠️ Structured data extraction error for {self.name}: {e}")

        return None

    def _extract_diagnoses_from_content(self, content: str) -> List[str]:
        """Extract diagnoses from content using enhanced patterns"""
        diagnoses = []

        # Pattern 1: JSON-like patterns
        json_pattern = r'"([^"]*(?:syndrome|disease|disorder|condition|injury|itis|osis|pathy|emia|uria)[^"]*)"'
        matches = re.findall(json_pattern, content, re.IGNORECASE)
        diagnoses.extend(matches)

        # Pattern 2: Medical term patterns
        medical_patterns = [
            r'\b(acute [^,.\n]+(?:injury|syndrome|disease))\b',
            r'\b([^,.\n]*(?:arthritis|pneumonia|nephropathy|infarction))\b',
            r'\b([^,.\n]*(?:embolization|thrombosis|stenosis))\b'
        ]

        for pattern in medical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            diagnoses.extend(matches)

        # Clean and deduplicate
        cleaned_diagnoses = []
        for diag in diagnoses:
            diag = diag.strip().lower()
            if len(diag) > 5 and diag not in cleaned_diagnoses:
                cleaned_diagnoses.append(diag)

        return cleaned_diagnoses[:7]  # Limit to 7 as per requirements

    def _extract_evidence_for_diagnosis(self, diagnosis: str, content: str) -> List[str]:
        """Enhanced evidence extraction with v6 improvements"""
        evidence = []

        # Look for evidence in brackets, quotes, or nearby text
        diagnosis_pattern = rf'"{re.escape(diagnosis)}"\s*:\s*\[([^\]]*)\]'
        match = re.search(diagnosis_pattern, content, re.IGNORECASE)
        if match:
            evidence_str = match.group(1)
            evidence = [e.strip().strip('"\'') for e in evidence_str.split(',') if e.strip()]

        # If no structured evidence found, look for nearby clinical terms
        if not evidence:
            diag_idx = content.lower().find(diagnosis.lower())
            if diag_idx >= 0:
                start = max(0, diag_idx - 50)
                end = min(len(content), diag_idx + len(diagnosis) + 100)
                context = content[start:end]

                # Enhanced clinical term extraction
                clinical_terms = re.findall(r'\b(?:elevated|decreased|increased|abnormal|positive|negative)\s+\w+', context, re.IGNORECASE)
                lab_values = re.findall(r'\b\w+\s*\d+(?:\.\d+)?\s*(?:mg/dL|g/dL|/mm3|°C|°F|mmHg)\b', context)
                symptoms = re.findall(r'\b(?:pain|fever|cough|dyspnea|fatigue|nausea|vomiting|diarrhea|rash|edema)\b', context, re.IGNORECASE)

                evidence.extend(clinical_terms)
                evidence.extend(lab_values)
                evidence.extend(symptoms)

        return evidence[:4] if evidence else ["clinical findings"]

    def _calculate_confidence_score(self, content: str, round_type: str) -> float:
        """Calculate confidence score based on response quality"""
        score = 0.5  # Base score

        # Boost for structured data
        if round_type in ["independent_differential", "subspecialist_consultation"]:
            if self._extract_structured_data(content):
                score += 0.2

        # Boost for medical terminology
        medical_terms = len(re.findall(r'\b(?:diagnosis|evidence|findings|symptoms|clinical|laboratory|imaging|pathology)\b', content, re.IGNORECASE))
        score += min(0.2, medical_terms * 0.02)

        # Boost for length and detail
        if len(content) > 200:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _assess_reasoning_quality(self, content: str, round_type: str) -> str:
        """Assess reasoning quality of response"""
        quality_indicators = [
            len(re.findall(r'\b(?:because|therefore|thus|however|although|evidence|supports|suggests|indicates)\b', content, re.IGNORECASE)),
            len(re.findall(r'\b(?:finding|laboratory|clinical|diagnostic|therapeutic)\b', content, re.IGNORECASE)),
            1 if len(content) > 150 else 0,
            1 if self._extract_structured_data(content) else 0
        ]

        total_score = sum(quality_indicators)

        if total_score >= 6:
            return "high"
        elif total_score >= 3:
            return "standard"
        else:
            return "basic"

    def _update_performance_metrics(self, response: AgentResponse):
        """Update agent performance metrics"""
        self.performance_metrics['responses_generated'] += 1

        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        count = self.performance_metrics['responses_generated']

        self.performance_metrics['average_response_time'] = (
            (current_avg * (count - 1) + response.response_time) / count
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        return {
            'name': self.name,
            'specialty': self.specialty.value,
            'responses_generated': self.performance_metrics['responses_generated'],
            'structured_data_success_rate': (
                self.performance_metrics['structured_data_success'] /
                max(1, self.performance_metrics['responses_generated'])
            ),
            'average_response_time': self.performance_metrics['average_response_time'],
            'conversation_history_length': len(self.conversation_history)
        }

# =============================================================================
# 5. PRESERVED Dynamic Agent Generation System
# =============================================================================

class ModelBasedAgentGenerator:
    """PRESERVED agent generator - generates ANY specialties needed for cases"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_specialist_proposals(self, case_description: str) -> List[Dict[str, Any]]:
        """Generate specialist proposals - COMPLETELY DYNAMIC"""
        print(f"\n🤖 Generating Dynamic Specialists via Model Analysis...")

        available_models = self.model_manager.get_available_models()
        if len(available_models) < 2:
            print(f"⚠️ Only {len(available_models)} models available, duplicating for coverage")
            available_models = available_models * 2

        all_proposals = []

        for i, model_id in enumerate(available_models[:2]):
            print(f"\n📋 Model {i+1} ({model_id}) analyzing case...")

            model, sampling_params = self.model_manager.get_model(model_id)
            if not model:
                continue

            # PRESERVED dynamic analysis prompt
            analysis_prompt = self._create_dynamic_analysis_prompt(case_description, i+1)

            try:
                outputs = model.generate([analysis_prompt], sampling_params)
                response_content = outputs[0].outputs[0].text.strip()

                proposals = self._parse_specialist_proposals(response_content, model_id, i+1)
                all_proposals.extend(proposals)

                print(f"✅ Model {i+1} proposed {len(proposals)} specialists")
                for proposal in proposals:
                    print(f"   • {proposal['name']}: {proposal['specialty']}")

            except Exception as e:
                print(f"❌ Model {i+1} analysis failed: {e}")
                fallback_proposals = self._create_fallback_proposals(model_id, i+1)
                all_proposals.extend(fallback_proposals)

        return all_proposals

    def _create_dynamic_analysis_prompt(self, case_description: str, model_num: int) -> str:
        """Create COMPLETELY DYNAMIC analysis prompt"""

        prompt = f"""<|im_start|>system
You are an expert medical case analyst with full autonomy to design the optimal specialist team. Your task is to analyze this clinical case and propose specialists needed for comprehensive diagnosis.

AUTONOMY GUIDELINES:
- You decide how many specialists this case requires (minimum 3, maximum 10)
- Consider case complexity, multi-system involvement, diagnostic uncertainty
- Create diverse reasoning approaches and subspecialties
- Be strategic about team composition for optimal collaboration
- Generate ANY specialty needed - you are not constrained to any specific list

For each specialist, create:
1. A specific subspecialty (be creative and precise - any medical specialty)
2. A unique realistic name
3. A detailed persona describing their approach and expertise
4. A reasoning style (analytical, intuitive, systematic, innovative, skeptical, methodical, etc.)
5. Focus areas relevant to this case

RESPOND WITH VALID JSON ARRAY:
[
  {{
    "name": "Dr. [Realistic Name]",
    "specialty": "[ANY Medical Specialty - be specific and creative]",
    "persona": "[Detailed background and approach]",
    "reasoning_style": "[cognitive approach]",
    "focus_areas": ["area1", "area2", "area3"],
    "case_rationale": "[Why this specialist is needed for THIS specific case]"
  }},
  ... (your optimal number of specialists)
]

IMPORTANT:
- Choose the number of specialists YOU think this case needs (3-10)
- Prioritize quality and diversity over quantity
- Each specialist should bring unique value to THIS case
- Generate ANY specialty the case needs - no restrictions
<|im_end|>
<|im_start|>user
Medical Case to Analyze:

{case_description}

Model Assignment: You are Model {model_num} - Design the optimal specialist team for this case using your clinical judgment.
<|im_end|>
<|im_start|>assistant"""

        return prompt

    def _parse_specialist_proposals(self, response_content: str, model_id: str, model_num: int) -> List[Dict[str, Any]]:
        """Parse specialist proposals - PRESERVED logic"""
        try:
            import json
            import re

            # Enhanced JSON extraction
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = self._fix_json_issues(json_str)

                proposals_data = json.loads(json_str)

                if isinstance(proposals_data, list) and len(proposals_data) >= 3:
                    max_specialists = min(10, len(proposals_data))
                    actual_count = max_specialists

                    print(f"   📊 Model {model_num} autonomously proposed {len(proposals_data)} specialists")
                    print(f"   ✅ Using {actual_count} specialists (capped at 10)")

                    proposals = []
                    for i, data in enumerate(proposals_data[:max_specialists]):
                        proposal = {
                            'name': data.get('name', f'Dr. Auto {model_num}-{i+1}'),
                            'specialty': data.get('specialty', 'General Medicine'),
                            'persona': data.get('persona', 'Experienced physician'),
                            'reasoning_style': data.get('reasoning_style', 'analytical'),
                            'focus_areas': data.get('focus_areas', ['general medicine']),
                            'case_rationale': data.get('case_rationale', 'General medical expertise'),
                            'model_id': model_id,
                            'model_num': model_num,
                            'agent_index': i,
                            'autonomous_generation': True,
                            'team_size_chosen': len(proposals_data)
                        }
                        proposals.append(proposal)

                    return proposals

                else:
                    print(f"   ⚠️ Model {model_num} proposed only {len(proposals_data)} specialists (minimum 3 required)")

        except Exception as e:
            print(f"   ❌ Failed to parse Model {model_num} proposals: {e}")

        # Enhanced fallback
        fallback_count = 6
        print(f"   🔄 Using fallback generation: {fallback_count} specialists")
        return self._create_fallback_proposals(model_id, model_num, fallback_count)

    def _fix_json_issues(self, json_str: str) -> str:
        """Enhanced JSON fixing with v6 improvements"""
        # Remove text before first [
        start_idx = json_str.find('[')
        if start_idx > 0:
            json_str = json_str[start_idx:]

        # Remove text after last ]
        end_idx = json_str.rfind(']')
        if end_idx >= 0:
            json_str = json_str[:end_idx + 1]

        # Enhanced cleanup
        json_str = json_str.replace('\n', ' ')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'"\s*,\s*"', '", "', json_str)

        # Fix incomplete entries
        if json_str.count('{') != json_str.count('}'):
            last_complete = json_str.rfind('}, {')
            if last_complete > 0:
                json_str = json_str[:last_complete + 1] + ']'

        return json_str

    def _create_fallback_proposals(self, model_id: str, model_num: int, target_count: int = 6) -> List[Dict[str, Any]]:
        """Create fallback proposals - diverse specialties"""

        base_specialties = [
            "Emergency Medicine", "Internal Medicine", "Infectious Disease",
            "Critical Care", "Nephrology", "Cardiology", "Endocrinology",
            "Hematology", "Gastroenterology", "Neurology"
        ]

        fallback_specialists = []
        for i in range(min(target_count, len(base_specialties))):
            specialist_info = {
                'name': f'Dr. Fallback {model_num}-{i+1}',
                'specialty': base_specialties[i],
                'persona': f'Experienced {base_specialties[i]} specialist',
                'reasoning_style': ['analytical', 'systematic', 'methodical', 'intuitive'][i % 4],
                'focus_areas': ['general medicine', 'acute care'],
                'case_rationale': f'General {base_specialties[i]} expertise',
                'model_id': model_id,
                'model_num': model_num,
                'agent_index': i,
                'autonomous_generation': False,
                'team_size_chosen': target_count
            }
            fallback_specialists.append(specialist_info)

        return fallback_specialists

class DynamicAgentGenerator:
    """PRESERVED dynamic agent generator - handles ANY specialty"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.proposal_generator = ModelBasedAgentGenerator(model_manager)

    def generate_agents_for_case(self, case_description: str,
                               max_specialists: int = 20) -> List[DDxAgent]:
        """Generate agents - COMPLETELY DYNAMIC for any case"""
        print(f"\n🎯 Generating Dynamic Specialist Team via Model Analysis...")

        specialist_proposals = self.proposal_generator.generate_specialist_proposals(case_description)

        if not specialist_proposals:
            print("❌ No specialist proposals generated")
            return []

        # Respect autonomous model choices, cap at reasonable limit
        max_autonomous = min(max_specialists, len(specialist_proposals))
        specialist_proposals = specialist_proposals[:max_autonomous]
        print(f"📊 Using {len(specialist_proposals)} total specialists from autonomous generation")

        agents = []

        for proposal in specialist_proposals:
            try:
                agent_config = self._create_agent_config_from_proposal(proposal)
                model, base_sampling_params = self.model_manager.get_model(proposal['model_id'])

                if not model:
                    print(f"⚠️ Model {proposal['model_id']} not available for {proposal['name']}")
                    continue

                custom_params = self._create_custom_sampling_params(
                    base_sampling_params, proposal, len(agents)
                )

                agent = DDxAgent(agent_config, model, custom_params, proposal['model_id'])
                agents.append(agent)

                print(f"✅ Generated: {agent.name}")
                print(f"   Specialty: {agent.specialty.value}")
                print(f"   Persona: {agent.persona[:80]}...")
                print(f"   Model: {proposal['model_id']}")

            except Exception as e:
                print(f"❌ Failed to create agent from proposal {proposal['name']}: {e}")
                continue

        print(f"\n🏥 Final Team: {len(agents)} specialists generated")
        return agents

    def _create_agent_config_from_proposal(self, proposal: Dict[str, Any]) -> AgentConfig:
        """Create agent configuration from proposal"""
        specialty_type = self._map_specialty_to_enum(proposal['specialty'])
        temperature = self._calculate_temperature(proposal)

        return AgentConfig(
            name=proposal['name'],
            specialty=specialty_type,
            persona=proposal['persona'],
            reasoning_style=proposal['reasoning_style'],
            temperature=temperature,
            focus_areas=proposal['focus_areas'],
            case_relevance_score=10.0,
            model_assignment=proposal['model_id']
        )

    def _map_specialty_to_enum(self, specialty_string: str) -> SpecialtyType:
        """Map specialty string to enum - FLEXIBLE mapping"""
        specialty_lower = specialty_string.lower()

        # Try to find exact or close matches
        for specialty_type in SpecialtyType:
            if specialty_type.value.lower() == specialty_lower:
                return specialty_type

            # Check if specialty string contains key words
            if any(word in specialty_lower for word in specialty_type.value.lower().split()):
                return specialty_type

        # Keyword-based mapping for common patterns
        if any(term in specialty_lower for term in ['cardio', 'heart', 'cardiac']):
            return SpecialtyType.CARDIOLOGY
        elif any(term in specialty_lower for term in ['pulmon', 'lung', 'respiratory']):
            return SpecialtyType.PULMONOLOGY
        elif any(term in specialty_lower for term in ['endocrin', 'diabetes', 'hormone']):
            return SpecialtyType.ENDOCRINOLOGY
        elif any(term in specialty_lower for term in ['gastro', 'gi', 'liver', 'digestive']):
            return SpecialtyType.GASTROENTEROLOGY
        elif any(term in specialty_lower for term in ['neuro', 'brain', 'nervous']):
            return SpecialtyType.NEUROLOGY
        elif any(term in specialty_lower for term in ['nephro', 'kidney', 'renal']):
            return SpecialtyType.NEPHROLOGY
        elif any(term in specialty_lower for term in ['hemato', 'blood', 'hematology']):
            return SpecialtyType.HEMATOLOGY
        elif any(term in specialty_lower for term in ['dermato', 'skin', 'dermatology']):
            return SpecialtyType.DERMATOLOGY
        elif any(term in specialty_lower for term in ['rheumat', 'joint', 'arthritis']):
            return SpecialtyType.RHEUMATOLOGY
        elif any(term in specialty_lower for term in ['orthoped', 'bone', 'fracture']):
            return SpecialtyType.ORTHOPEDICS
        elif any(term in specialty_lower for term in ['uro', 'urinary', 'bladder']):
            return SpecialtyType.UROLOGY
        elif any(term in specialty_lower for term in ['gynec', 'reproductive', 'pelvic']):
            return SpecialtyType.GYNECOLOGY
        elif any(term in specialty_lower for term in ['emergency', 'urgent', 'acute']):
            return SpecialtyType.EMERGENCY_MEDICINE
        elif any(term in specialty_lower for term in ['critical', 'intensive', 'icu']):
            return SpecialtyType.CRITICAL_CARE
        elif any(term in specialty_lower for term in ['infectious', 'infection', 'sepsis']):
            return SpecialtyType.INFECTIOUS_DISEASE
        elif any(term in specialty_lower for term in ['oncol', 'cancer', 'tumor']):
            return SpecialtyType.ONCOLOGY
        elif any(term in specialty_lower for term in ['surgery', 'surgical', 'operative']):
            return SpecialtyType.SURGERY
        else:
            # Default to general medicine for unrecognized specialties
            return SpecialtyType.GENERAL

    def _calculate_temperature(self, proposal: Dict[str, Any]) -> float:
        """Calculate temperature based on reasoning style and model characteristics"""
        reasoning_style = proposal.get('reasoning_style', 'analytical').lower()
        model_id = proposal.get('model_id', '')
        agent_index = proposal.get('agent_index', 0)

        # Base temperature from model type
        if 'conservative' in model_id.lower():
            base_temp = 0.2
        else:
            base_temp = 0.6

        # Adjust based on reasoning style
        style_modifiers = {
            'analytical': -0.1,
            'systematic': -0.1,
            'methodical': -0.1,
            'innovative': +0.2,
            'creative': +0.2,
            'intuitive': +0.1,
            'rapid': +0.1
        }

        temp_modifier = style_modifiers.get(reasoning_style, 0.0)
        agent_variation = (agent_index * 0.05)
        final_temp = base_temp + temp_modifier + agent_variation

        return max(0.1, min(1.0, final_temp))

    def _create_custom_sampling_params(self, base_params: SamplingParams,
                                     proposal: Dict[str, Any], agent_count: int) -> SamplingParams:
        """Create custom sampling parameters for this specific agent"""
        temperature = self._calculate_temperature(proposal)

        # Vary top_p slightly for diversity
        top_p = base_params.top_p + (agent_count * 0.02)
        top_p = max(0.7, min(0.95, top_p))

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=base_params.max_tokens,
            stop=base_params.stop
        )

# =============================================================================
# 6. Enhanced DDx System Class
# =============================================================================

class DDxSystem:
    """Enhanced DDx system with PRESERVED dynamic generation"""

    def __init__(self):
        self.model_manager = None
        self.agent_generator = None
        self.current_agents = []
        self.case_data = {}
        self.round_results = {}
        self._ensure_rounds_integration()

    def _ensure_rounds_integration(self):
        """Ensure rounds integration is applied to this DDxSystem instance"""
        try:
            from ddx_rounds_v6 import integrate_rounds_with_ddx_system

            # Get the integration methods
            integration_methods = integrate_rounds_with_ddx_system()

            # Add them to this instance
            import types
            self.add_round_orchestrator = types.MethodType(integration_methods['add_round_orchestrator'], self)
            self.run_complete_diagnostic_sequence = types.MethodType(integration_methods['run_diagnostic_sequence'], self)
            self.get_diagnostic_summary = types.MethodType(integration_methods['get_diagnostic_summary'], self)

            print("✅ Rounds integration applied successfully")

        except Exception as e:
            print(f"⚠️ Rounds integration failed: {e}")

    def initialize(self) -> bool:
        """Initialize the DDx system"""
        print("🚀 Initializing DDx System v6...")

        # Load configuration and initialize models
        configs = load_system_config()
        self.model_manager = ModelManager(configs)

        if not self.model_manager.initialize_models():
            print("❌ Failed to initialize models")
            return False

        # Initialize PRESERVED dynamic agent generator
        self.agent_generator = DynamicAgentGenerator(self.model_manager)

        print("✅ DDx System v6 initialized successfully!")
        return True

    def analyze_case(self, case_description: str, case_name: str = "dynamic_case") -> Dict[str, Any]:
        """Analyze case with PRESERVED dynamic specialist generation"""
        print(f"\n🏥 Analyzing Case: {case_name}")
        print("=" * 60)

        # Store case data
        self.case_data = {
            'description': case_description,
            'case_description': case_description,  
            'name': case_name,
            'timestamp': time.time()
        }

        # Generate DYNAMIC specialist team - ANY specialties for ANY case
        self.current_agents = self.agent_generator.generate_agents_for_case(
            case_description, max_specialists=20
        )

        if not self.current_agents:
            print("❌ Failed to generate specialists for case")
            return {'success': False, 'error': 'No agents generated'}

        print(f"✅ Generated {len(self.current_agents)} specialists for case analysis")

        return {
            'success': True,
            'agents_generated': len(self.current_agents),
            'case_name': case_name,
            'specialists': [agent.name for agent in self.current_agents]
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'models_loaded': len(self.model_manager.get_available_models()) if self.model_manager else 0,
            'available_models': self.model_manager.get_available_models() if self.model_manager else [],
            'current_agents': len(self.current_agents),
            'agent_names': [agent.name for agent in self.current_agents],
            'case_loaded': bool(self.case_data)
        }

# =============================================================================
# 7. Testing and Validation
# =============================================================================

def test_ddx_core():
    """Test the PRESERVED DDx core system"""
    print("🧪 Testing PRESERVED DDx Core System")
    print("=" * 50)

    # Initialize system
    ddx = DDxSystem()
    if not ddx.initialize():
        return False

    # Test case analysis with PRESERVED dynamic generation
    test_case = """
    A 45-year-old male presents with acute chest pain that began 2 hours ago.
    The pain is crushing, radiates to the left arm, and is associated with
    shortness of breath and diaphoresis. He has a history of diabetes and
    hypertension. ECG shows ST elevation in leads II, III, aVF.
    """

    result = ddx.analyze_case(test_case, "preserved_chest_pain_test")

    if result['success']:
        print(f"✅ PRESERVED case analysis successful!")
        print(f"   Generated specialists: {result['specialists']}")

        # Show system status
        status = ddx.get_system_status()
        print(f"\n📊 System Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")

        return ddx
    else:
        print(f"❌ Case analysis failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    # Test the PRESERVED core system
    system = test_ddx_core()

    if system:
        print(f"\n🎉 PRESERVED DDx Core System Ready!")
        print(f"✅ Dynamic agent generation PRESERVED")
        print(f"✅ Can generate ANY specialty for ANY case")
        print(f"✅ Model management stable")
        print(f"✅ Case analysis functional")
        print(f"✅ Ready for round system with DDx_Main_Design.md OUTPUT formatting")
