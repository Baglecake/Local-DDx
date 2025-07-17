
# =============================================================================
# DDx Core v6 - Unified Foundation with PRESERVED Dynamic Generation
# =============================================================================

"""
DDx Core v6: Clean, unified foundation that preserves ALL sophisticated dynamic
generation features. DDx_Main_Design.md integration happens in ROUNDS and
OUTPUT FORMATTING, not in constraining what specialties can be generated.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
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
    dtype: str = 'auto'
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>", "<|im_end|>"])
    enforce_eager: bool = True
    max_num_seqs: int = 2
    quantization: Optional[str] = None

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
    """MULTI-MODEL manager - loads and manages multiple models simultaneously"""

    def __init__(self, configs: Dict[str, ModelConfig]):
        self.configs = configs
        self.active_models: Dict[str, LLM] = {}  # Multiple models by ID
        self.active_sampling_params: Dict[str, SamplingParams] = {}  # Params by ID
        self.model_load_order: List[str] = []  # Track load order for cleanup
        print("✅ Initialized Multi-Model ModelManager for simultaneous loading.")

    def _verify_gpu(self) -> bool:
        """Enhanced GPU verification with memory reporting"""
        if not torch.cuda.is_available():
            print("❌ No GPU detected - check Colab runtime settings")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_memory = torch.cuda.mem_get_info()[0] / 1e9

        print(f"✅ GPU: {gpu_name}")
        print(f"   Total VRAM: {gpu_memory:.1f}GB")
        print(f"   Available: {available_memory:.1f}GB")

        # Check if we can fit multiple models
        estimated_per_model = 20.0  # GB per model (conservative estimate)
        max_models = int(available_memory / estimated_per_model)

        print(f"   Estimated capacity: {max_models} models simultaneously")

        if available_memory < estimated_per_model:
            print("⚠️ Low VRAM - may need sequential loading")
        elif available_memory >= estimated_per_model * 2:
            print("✅ Sufficient VRAM for multi-model loading")

        return True

    def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """Load a model WITHOUT unloading existing ones"""

        # Check if already loaded
        if model_id in self.active_models and not force_reload:
            print(f"✅ Model '{model_id}' is already loaded and active.")
            return True

        if not self._verify_gpu():
            return False

        config = self.configs.get(model_id)
        if not config:
            print(f"❌ Config for model '{model_id}' not found.")
            return False

        print(f"\n📥 Loading {config.name} (keeping existing models active)...")
        try:
            # Check available memory before loading
            available_memory = torch.cuda.mem_get_info()[0] / 1e9
            print(f"   Available VRAM before load: {available_memory:.1f}GB")

            # Build the arguments for the LLM class dynamically
            llm_args = {
                "model": config.model_path,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": config.memory_fraction,
                "max_model_len": config.max_model_len,
                "trust_remote_code": True,
                "dtype": config.dtype,
                "enforce_eager": config.enforce_eager,
                "max_num_seqs": config.max_num_seqs
            }

            # Only add the quantization parameter if it's specified in the config
            if config.quantization:
                llm_args["quantization"] = config.quantization
                print(f"   ⚙️ Applying quantization: {config.quantization}")

            # Load the model
            model = LLM(**llm_args)

            enhanced_stop_tokens = config.stop_tokens + [
                "\n\nThe consolidated",
                "\n\nassistant",
                "assistant\n\n",
                "</CONSOLIDATION_NOTES>\n\n",
                "</MASTER DDX LIST>\n\n",
                "\n\nNote: The ranking",
            ]

            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                stop=enhanced_stop_tokens
            )

            # Store in multi-model dictionaries
            self.active_models[model_id] = model
            self.active_sampling_params[model_id] = sampling_params

            # Track load order for cleanup
            if model_id not in self.model_load_order:
                self.model_load_order.append(model_id)

            # Check memory after loading
            available_memory_after = torch.cuda.mem_get_info()[0] / 1e9
            used_memory = available_memory - available_memory_after

            print(f"   ✅ {config.name} loaded successfully.")
            print(f"   📊 VRAM used: {used_memory:.1f}GB, remaining: {available_memory_after:.1f}GB")
            print(f"   🔢 Total models loaded: {len(self.active_models)}")

            return True

        except Exception as e:
            print(f"   ❌ Failed to load {config.name}: {e}")
            # Clean up partial load
            if model_id in self.active_models:
                del self.active_models[model_id]
            if model_id in self.active_sampling_params:
                del self.active_sampling_params[model_id]
            return False

    def get_model(self, model_id: str) -> Tuple[Optional[LLM], Optional[SamplingParams]]:
        """Get a specific model by ID"""
        if model_id not in self.active_models:
            print(f"⚠️ Model '{model_id}' not loaded, attempting to load...")
            if not self.load_model(model_id):
                return None, None

        return self.active_models.get(model_id), self.active_sampling_params.get(model_id)

    def get_active_model(self) -> Tuple[Optional[LLM], Optional[SamplingParams]]:
        """Get the most recently loaded model (for backward compatibility)"""
        if not self.active_models:
            return None, None

        # Return the most recently loaded model
        latest_model_id = self.model_load_order[-1]
        return self.active_models[latest_model_id], self.active_sampling_params[latest_model_id]

    @property
    def active_model_id(self) -> Optional[str]:
        """Get the most recently loaded model ID (for backward compatibility)"""
        return self.model_load_order[-1] if self.model_load_order else None

    def unload_model(self, model_id: Optional[str] = None):
        """Unload a specific model or the most recent one"""
        if model_id is None:
            # Unload most recent model (backward compatibility)
            if not self.model_load_order:
                print("   ℹ️ No models to unload")
                return
            model_id = self.model_load_order[-1]

        if model_id not in self.active_models:
            print(f"   ℹ️ Model '{model_id}' not currently loaded")
            return

        model_name = self.configs[model_id].name if model_id in self.configs else model_id
        print(f"\n📤 Unloading {model_name}...")

        # Get memory before unload
        available_before = torch.cuda.mem_get_info()[0] / 1e9

        try:
            # Try proper shutdown
            model = self.active_models[model_id]
            try:
                model.shutdown()  # type: ignore
                print("   🔧 vLLM shutdown() called successfully")
            except AttributeError:
                try:
                    model.close()  # type: ignore
                    print("   🔧 vLLM close() called successfully")
                except AttributeError:
                    print("   ⚠️ No shutdown/close method found - partial cleanup only")
            except Exception as e:
                print(f"   ⚠️ Shutdown failed: {e}")

            # Remove from tracking
            del self.active_models[model_id]
            del self.active_sampling_params[model_id]
            self.model_load_order.remove(model_id)

            # Standard cleanup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Check memory recovery
            available_after = torch.cuda.mem_get_info()[0] / 1e9
            freed_memory = available_after - available_before

            print(f"   ✅ {model_name} unloaded. VRAM freed: {freed_memory:.1f}GB")
            print(f"   📊 Remaining models: {len(self.active_models)} ({list(self.active_models.keys())})")

        except Exception as e:
            print(f"   ❌ Error during unload: {e}")

    def unload_all_models(self):
        """Unload all models"""
        print(f"\n🧹 Unloading all {len(self.active_models)} models...")

        # Unload in reverse order
        for model_id in self.model_load_order[::-1]:
            self.unload_model(model_id)

        # Final cleanup
        self.active_models.clear()
        self.active_sampling_params.clear()
        self.model_load_order.clear()

        print("✅ All models unloaded")

    def get_available_models(self) -> List[str]:
        """Get list of configured models"""
        return list(self.configs.keys())

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.active_models.keys())

    def initialize_models(self) -> bool:
        """Initialize system - verify configs and GPU"""
        print("🚀 Initializing Multi-Model Manager...")

        if not self._verify_gpu():
            return False

        print(f"📋 Configured models: {list(self.configs.keys())}")
        print("✅ Multi-model loading system ready.")
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        available_memory = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0

        return {
            'gpu_available': torch.cuda.is_available(),
            'available_memory_gb': available_memory,
            'loaded_models': list(self.active_models.keys()),
            'configured_models': list(self.configs.keys()),
            'loading_mode': 'multi-model',
            'models_count': len(self.active_models)
        }


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
            if round_type in ["independent_differential", "subspecialist_consultation", "team_independent_differentials", "specialized_ranking", "symptom_management", "refinement_and_justification", "cant_miss"]:
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

    def _format_prompt(self, user_input: str, round_type: str = "general", global_transcript: Optional[Dict] = None) -> str:
        """Enhanced prompt formatting with improved JSON instructions for Llama3"""

        # Build sliding context if available
        prior_context = ""
        try:
            from ddx_sliding_context import SlidingContextManager
            if not hasattr(self, '_context_manager'):
                self._context_manager = SlidingContextManager()

            if global_transcript and len(global_transcript.get('rounds', {})) > 0:
                prior_context = self._context_manager.build_context_for_agent(
                    self.name, self.specialty.value, round_type, global_transcript
                )
        except (ImportError, Exception):
            prior_context = ""

        base_system = f"""You are {self.name}, a medical specialist in {self.specialty.value}.
    {self.persona}

    Your expertise: {', '.join(self.config.focus_areas)}
    Reasoning style: {self.config.reasoning_style}"""

        # CRITICAL: Add JSON formatting instructions for structured data rounds
        json_instruction = """
    IMPORTANT: When providing differential diagnoses, start your response with a JSON object in this EXACT format:
    {"diagnosis_name": ["evidence1", "evidence2"], "diagnosis_name": ["evidence1", "evidence2"]}

    After the JSON, provide your detailed clinical reasoning. The JSON must be valid and complete on its own."""

        if round_type in ["independent_differential", "subspecialist_consultation", "team_independent_differentials", "specialized_ranking", "symptom_management", "refinement_and_justification", "cant_miss"]:
            system_prompt = f"""{base_system}

    {json_instruction}

    EXAMPLE FORMAT:
    {{"acute kidney injury": ["elevated creatinine", "decreased output"], "pneumonia": ["fever", "cough"]}}

    After the JSON, provide your detailed reasoning. Focus on diagnoses most relevant to {self.specialty.value}."""

            user_prompt = f"""Case: {user_input}

    TASK: Start with a JSON dictionary of 3-5 diagnoses from your {self.specialty.value} perspective:

    {{"diagnosis 1": ["evidence 1", "evidence 2"], "diagnosis 2": ["evidence 3", "evidence 4"]}}

    Then explain your clinical reasoning."""

        elif round_type == "refinement_and_justification":
            system_prompt = f"""{base_system}

    {json_instruction}

    You are participating in collaborative medical debate with evidence-based reasoning.
    Challenge, support, or refine diagnoses based on clinical evidence and team discourse.

    For refinement rounds, use this JSON format:
    {{"refined_diagnosis": {{"tier": "1", "confidence": "high", "evidence": ["evidence1", "evidence2"], "reasoning": "detailed reasoning"}}}}"""

            user_prompt = user_input

        elif round_type == "post_debate_voting":
            system_prompt = f"""{base_system}

    Cast your preferential vote considering the full team discussion and evidence presented.

    Use this JSON format for voting:
    {{"vote_ranking": ["diagnosis1", "diagnosis2", "diagnosis3"], "rationale": "your reasoning"}}"""

            user_prompt = user_input

        elif round_type == "cant_miss":
            system_prompt = f"""{base_system}

    {json_instruction}

    Identify critical diagnoses that cannot be missed from a safety perspective.

    Use this JSON format:
    {{"critical_diagnosis": {{"urgency": "high", "consequences": "severe", "evidence": ["evidence1", "evidence2"]}}}}"""

            user_prompt = user_input

        else:
            system_prompt = base_system
            user_prompt = user_input

        # Include collaborative context if available
        if prior_context:
            system_prompt += f"\n\nCOLLABORATIVE CONTEXT:\n{prior_context}"

        # Enhanced ChatML format with clearer JSON instructions
        return (
            "<|im_start|>system\n"
            f"{system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "I'll provide my response starting with the requested JSON format:\n\n"
        )

    def _extract_structured_data(self, content: str) -> Optional[Dict]:
        """Enhanced structured data extraction with robust JSON parsing for Llama3 chattiness"""
        if not content:
            return None

        try:
            # Method 1: Enhanced JSON extraction - finds JSON anywhere in response
            json_objects = self._find_json_objects(content)

            for json_obj in json_objects:
                if isinstance(json_obj, dict) and len(json_obj) > 0:
                    # Validate it looks like a medical differential
                    if self._is_valid_medical_json(json_obj):
                        # Clean diagnosis names to remove numbering prefixes
                        cleaned_json = self._clean_diagnosis_names(json_obj)
                        self.performance_metrics['structured_data_success'] += 1
                        print(f"   ✅ {self.name}: Successfully extracted {len(cleaned_json)} diagnoses from JSON")
                        return cleaned_json

            # Method 2: Look for JSON-like patterns in lines (original approach enhanced)
            lines = content.strip().split('\n')
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                if line.startswith('{') and ':' in line:
                    json_text = line

                    # Try to complete the JSON if it spans multiple lines
                    if not line.endswith('}'):
                        for j in range(i+1, min(i+15, len(lines))):  # Look ahead max 15 lines
                            json_text += ' ' + lines[j].strip()
                            if lines[j].strip().endswith('}'):
                                break

                    try:
                        import json
                        data = json.loads(json_text)
                        if isinstance(data, dict) and len(data) > 0:
                            if self._is_valid_medical_json(data):
                                self.performance_metrics['structured_data_success'] += 1
                                print(f"   ✅ {self.name}: Successfully extracted {len(data)} diagnoses from multi-line JSON")
                                return data
                    except:
                        continue

            # Method 3: Fallback - extract diagnoses and create structure
            print(f"   ⚠️ {self.name}: JSON parsing failed, using diagnosis extraction fallback")
            diagnoses = self._extract_diagnoses_from_content(content)
            if diagnoses:
                result = {}
                for diagnosis in diagnoses:
                    evidence = self._extract_evidence_for_diagnosis(diagnosis, content)
                    result[diagnosis] = evidence if evidence else ["clinical presentation"]

                if len(result) > 0:
                    print(f"   🔄 {self.name}: Fallback extracted {len(result)} diagnoses")
                    return result

        except Exception as e:
            print(f"   ❌ {self.name}: Structured data extraction error: {e}")

        print(f"   ❌ {self.name}: No structured data extracted")
        return None

    def _find_json_objects(self, content: str) -> List[Dict]:
        """Find all potential JSON objects in content, handling Llama3 chattiness - COMPLETELY REWRITTEN"""
        import re
        import json

        json_objects = []

        # Method 1: Find JSON objects using improved regex patterns
        # Pattern for complete JSON objects with any content inside
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON objects
            r'\{[^}]+\}',  # Simple JSON objects
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    cleaned = self._clean_json_string(match)
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict) and len(obj) > 0:
                        json_objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    continue

        # Method 2: Line-by-line JSON building for multi-line responses
        lines = content.split('\n')
        current_json = ""
        brace_count = 0
        in_json = False

        for line in lines:
            line = line.strip()

            # Start of JSON detected
            if '{' in line and ':' in line and not in_json:
                in_json = True
                current_json = line
                brace_count = line.count('{') - line.count('}')

                # Check if it's a complete single-line JSON
                if brace_count == 0:
                    try:
                        cleaned = self._clean_json_string(current_json)
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict) and len(obj) > 0:
                            json_objects.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    in_json = False
                    current_json = ""

            elif in_json:
                # Continue building JSON
                current_json += " " + line
                brace_count += line.count('{') - line.count('}')

                # JSON is complete
                if brace_count == 0:
                    try:
                        cleaned = self._clean_json_string(current_json)
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict) and len(obj) > 0:
                            json_objects.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    in_json = False
                    current_json = ""

                # Safety break for runaway JSON
                elif len(current_json) > 5000:
                    in_json = False
                    current_json = ""

        # Method 3: Aggressive pattern matching for JSON-like structures
        # Find anything that looks like {"key": "value"} or {"key": ["item1", "item2"]}
        aggressive_pattern = r'\{[^{}]*"[^"]*"[^{}]*:[^{}]*(?:"[^"]*"|(?:\[[^\]]*\]))[^{}]*\}'
        aggressive_matches = re.findall(aggressive_pattern, content, re.DOTALL)

        for match in aggressive_matches:
            try:
                cleaned = self._clean_json_string(match)
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and len(obj) > 0:
                    # Check if this is a duplicate
                    if not any(self._dict_similar(obj, existing) for existing in json_objects):
                        json_objects.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue

        # Method 4: Handle code block JSON (```json ... ```)
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        code_matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in code_matches:
            try:
                cleaned = self._clean_json_string(match)
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and len(obj) > 0:
                    json_objects.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue

        return json_objects

    def _dict_similar(self, dict1: Dict, dict2: Dict) -> bool:
        """Check if two dictionaries are similar enough to be considered duplicates"""
        if len(dict1) != len(dict2):
            return False

        # Check if they have mostly the same keys
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        overlap = len(keys1.intersection(keys2))

        return overlap / max(len(keys1), len(keys2)) > 0.8

    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to handle common Llama3 formatting issues"""
        import re

        # Remove leading/trailing whitespace
        json_str = json_str.strip()

        # Remove common conversational preambles that Llama3 adds
        preambles = [
            r".*?(?=\{)",  # Remove everything before first {
            r"(?i)here\s+is.*?:",
            r"(?i)based\s+on.*?:",
            r"(?i)certainly.*?:",
            r"(?i)the\s+(?:json|diagnoses?).*?:",
            r"(?i)my\s+(?:differential\s+)?diagnos[ie]s.*?:",
            r"(?i)i\s+would\s+suggest.*?:",
            r"(?i)the\s+following.*?:",
        ]

        for pattern in preambles:
            json_str = re.sub(pattern, "", json_str, flags=re.IGNORECASE | re.DOTALL)

        # Find the actual JSON boundaries more precisely
        first_brace = json_str.find('{')
        if first_brace > 0:
            json_str = json_str[first_brace:]

        # Find the last closing brace that matches the structure
        brace_count = 0
        last_valid_pos = -1

        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:  # Found matching closing brace
                    last_valid_pos = i + 1
                    break

        if last_valid_pos > 0:
            json_str = json_str[:last_valid_pos]

        # Remove trailing conversational text after JSON
        postambles = [
            r'\}\s*\.?\s*(These|This|The above|As you can see|Note that).*$',
            r'\}\s*\.?\s*[A-Z][a-z].*$',  # Remove sentences after JSON
        ]

        for pattern in postambles:
            json_str = re.sub(pattern, '}', json_str, flags=re.IGNORECASE | re.DOTALL)

        # Fix common JSON formatting issues
        # Remove newlines within the JSON
        json_str = re.sub(r'\n+', ' ', json_str)

        # Fix spacing around colons and commas
        json_str = re.sub(r'\s*:\s*', ': ', json_str)
        json_str = re.sub(r'\s*,\s*', ', ', json_str)

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix quote issues
        json_str = re.sub(r'(["\'])\s*,\s*(["\'])', r'\1, \2', json_str)

        # Handle cases where values aren't properly quoted
        # Fix unquoted string values (but not arrays)
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9\s]*[a-zA-Z0-9])\s*([,}])', r': "\1"\2', json_str)

        # Fix double-quoted strings that got broken
        json_str = re.sub(r'"\s*"\s*', '", "', json_str)

        # Remove extra spaces
        json_str = re.sub(r'\s+', ' ', json_str)

        return json_str.strip()

    def _is_valid_medical_json(self, json_obj: Dict) -> bool:
        """Validate that JSON object looks like a medical differential diagnosis"""
        if not isinstance(json_obj, dict) or len(json_obj) == 0:
            return False

        # Check for medical terms in keys
        medical_indicators = [
            'syndrome', 'disease', 'disorder', 'condition', 'injury', 'itis', 'osis',
            'pathy', 'emia', 'uria', 'cardia', 'arthritis', 'pneumonia', 'infarction',
            'fracture', 'cancer', 'tumor', 'infection', 'sepsis'
        ]

        for key in json_obj.keys():
            key_lower = key.lower()
            if any(indicator in key_lower for indicator in medical_indicators):
                return True

            # Check if values look like medical evidence
            values = json_obj[key]
            if isinstance(values, list) and len(values) > 0:
                return True  # Assume list of evidence is medical

        # If we have any diagnoses with reasonable length, accept it
        return len(json_obj) >= 1 and all(len(str(k)) > 3 for k in json_obj.keys())


    def _extract_diagnoses_from_content(self, content: str) -> List[str]:
        """FIXED: Extract diagnoses from content with proper normalization"""
        import re

        diagnoses = []

        # Pattern 1: Numbered lists (common in LLM responses)
        numbered_pattern = r'^\s*\d+[\.\)\-\s]+(.+)$'
        for line in content.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                diagnosis = match.group(1).strip()
                # Clean up common artifacts
                diagnosis = re.sub(r'\s*-.*$', '', diagnosis)  # Remove " - explanation" parts
                if len(diagnosis) > 3:  # Valid length
                    diagnoses.append(diagnosis)

        # Pattern 2: JSON-like structures
        json_pattern = r'"([^"]*(?:syndrome|disease|disorder|condition|injury|itis|osis|pathy)[^"]*)"'
        matches = re.findall(json_pattern, content, re.IGNORECASE)
        diagnoses.extend(matches)

        # Pattern 3: Medical term patterns
        medical_patterns = [
            r'\b(acute [^,.\n]+(?:injury|syndrome|disease))\b',
            r'\b([^,.\n]*(?:arthritis|pneumonia|nephropathy|infarction))\b',
            r'\b([^,.\n]*(?:embolization|thrombosis|stenosis))\b'
        ]

        for pattern in medical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            diagnoses.extend(matches)

        # Clean and deduplicate - NORMALIZE HERE
        cleaned_diagnoses = []
        for diag in diagnoses:
            # Strip any remaining numbering prefixes
            clean_diag = re.sub(r'^\d+[\.\)\-\s]+', '', diag.strip())
            clean_diag = clean_diag.lower().strip()

            if len(clean_diag) > 5 and clean_diag not in cleaned_diagnoses:
                cleaned_diagnoses.append(clean_diag)

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

    def _clean_diagnosis_names(self, json_obj: Dict) -> Dict:
        """Clean diagnosis names to remove numbering prefixes and normalize formatting"""
        import re

        if not isinstance(json_obj, dict):
            return json_obj

        cleaned_obj = {}

        for key, value in json_obj.items():
            # Clean the diagnosis name (key)
            cleaned_key = self._normalize_diagnosis_name(key)

            # Keep the value (evidence list) as-is
            cleaned_obj[cleaned_key] = value

        return cleaned_obj

    def _normalize_diagnosis_name(self, diagnosis_name: str) -> str:
        """Normalize a diagnosis name by removing prefixes and standardizing format"""
        import re

        if not isinstance(diagnosis_name, str):
            return str(diagnosis_name)

        # Remove numbered list prefixes (1., 2., etc.)
        cleaned = re.sub(r'^\s*\d+[\.\)\-\s]+', '', diagnosis_name)

        # Remove bullet points and dashes
        cleaned = re.sub(r'^\s*[-–•]\s*', '', cleaned)

        # Remove extra whitespace
        cleaned = cleaned.strip()

        # Standardize common abbreviations (but keep original if not recognized)
        abbreviation_map = {
            'aki': 'Acute Kidney Injury',
            'tma': 'Thrombotic Microangiopathy',
            'ttp': 'Thrombotic Thrombocytopenic Purpura',
            'mi': 'Myocardial Infarction',
            'pe': 'Pulmonary Embolism',
            'dvt': 'Deep Vein Thrombosis',
            'cad': 'Coronary Artery Disease',
            'chf': 'Congestive Heart Failure',
            'copd': 'Chronic Obstructive Pulmonary Disease',
            'dm': 'Diabetes Mellitus',
            'htn': 'Hypertension'
        }

        # Check if the cleaned name (lowercase) matches any abbreviation
        cleaned_lower = cleaned.lower().strip('()')
        if cleaned_lower in abbreviation_map:
            return abbreviation_map[cleaned_lower]

        # If no abbreviation match, return the cleaned original
        return cleaned

# =============================================================================
# 5. PRESERVED Dynamic Agent Generation System
# =============================================================================

class ModelBasedAgentGenerator:
    """PRESERVED agent generator - generates ANY specialties needed for cases"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_specialist_proposals(self, case_description: str) -> List[Dict[str, Any]]:
        """Generate specialist proposals using SEQUENTIAL model loading"""
        print(f"\n🤖 Generating Dynamic Specialists via Sequential Model Analysis...")
        all_proposals = []

        # Get all configured models
        available_models = self.model_manager.get_available_models()
        print(f"📋 Will analyze with {len(available_models)} models sequentially")

        # Process each model one at a time
        for model_num, model_id in enumerate(available_models, 1):
            print(f"\n🔄 Processing Model {model_num}/{len(available_models)}: {model_id}")

            # Load this specific model
            if not self.model_manager.load_model(model_id):
                print(f"⚠️ Skipping model '{model_id}' due to load failure.")
                continue

            model, sampling_params = self.model_manager.get_active_model()
            if not model:
                print(f"⚠️ Model '{model_id}' not available after loading.")
                continue

            # Generate proposals with this model
            try:
                print(f"   📝 Model {model_num} analyzing case...")
                analysis_prompt = self._create_dynamic_analysis_prompt(case_description, model_num)

                outputs = model.generate([analysis_prompt], sampling_params)
                response_content = outputs[0].outputs[0].text.strip()

                # Parse proposals from this model
                model_proposals = self._parse_specialist_proposals(response_content, model_id, model_num)
                all_proposals.extend(model_proposals)

                print(f"   ✅ Model {model_num} generated {len(model_proposals)} specialist proposals")

            except Exception as e:
                print(f"   ❌ Model {model_num} analysis failed: {e}")

            # Unload model to free VRAM for next one
            # self.model_manager.unload_model()
            # print(f"   🗑️ Model {model_num} unloaded")

        print(f"\n🎯 Total proposals generated: {len(all_proposals)}")

        # Add fallback if no proposals generated
        if not all_proposals:
            print("⚠️ No autonomous proposals generated - creating fallback specialists")
            all_proposals = self._generate_fallback_specialists()

        return all_proposals

    def _generate_fallback_specialists(self) -> List[Dict[str, Any]]:
        """Generate fallback specialists when autonomous generation fails"""
        base_specialties = [
            "Emergency Medicine", "Internal Medicine", "Cardiology",
            "Pulmonology", "Nephrology", "Endocrinology"
        ]

        fallback_specialists = []
        for i, specialty in enumerate(base_specialties):
            specialist_info = {
                'name': f'Dr. Fallback {i+1}',
                'specialty': specialty,
                'persona': f'Experienced {specialty} specialist',
                'reasoning_style': 'analytical',
                'focus_areas': ['general medicine'],
                'case_rationale': f'General {specialty} expertise',
                'model_id': 'conservative_model',
                'model_num': 1,
                'agent_index': i,
                'autonomous_generation': False,
                'team_size_chosen': len(base_specialties)
            }
            fallback_specialists.append(specialist_info)

        return fallback_specialists

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
        """Enhanced JSON fixing with v6 improvements + Llama3 chattiness fix"""

        # FIRST: Strip all content before the first [
        start_idx = json_str.find('[')
        if start_idx > 0:
            json_str = json_str[start_idx:]

        # CRITICAL FIX: Find the COMPLETE JSON array end
        bracket_count = 0
        json_end = -1

        for i, char in enumerate(json_str):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:  # Found the matching closing bracket
                    json_end = i + 1
                    break

        # AGGRESSIVE: Cut everything after the complete JSON
        if json_end > 0:
            json_str = json_str[:json_end]

        # Rest of your existing cleanup
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
        """Generate agents preserving conservative vs innovative model innovation"""
        print(f"\n🎯 Generating Dynamic Specialist Team via Model Analysis...")

        specialist_proposals = self.proposal_generator.generate_specialist_proposals(case_description)

        if not specialist_proposals:
            print("❌ No specialist proposals generated")
            return []

        max_autonomous = min(max_specialists, len(specialist_proposals))
        specialist_proposals = specialist_proposals[:max_autonomous]
        print(f"📊 Using {len(specialist_proposals)} total specialists from autonomous generation")

        # 🎯 PRESERVE INNOVATION: Group by model to maintain conservative/innovative split
        proposals_by_model = {}
        for proposal in specialist_proposals:
            model_id = proposal['model_id']
            if model_id not in proposals_by_model:
                proposals_by_model[model_id] = []
            proposals_by_model[model_id].append(proposal)

        print(f"🎯 Preserving multi-model innovation: {len(proposals_by_model)} different models")
        for model_id, proposals in proposals_by_model.items():
            model_type = "Conservative" if "conservative" in model_id else "Innovative"
            print(f"   {model_type} model ({model_id}): {len(proposals)} specialists")

        agents = []

        # 🔧 MULTI-MODEL FIX: Load all needed models simultaneously
        print(f"\n🔄 Loading models simultaneously for multi-model innovation...")
        for model_id in proposals_by_model.keys():
            if not self.model_manager.load_model(model_id):
                print(f"⚠️ Skipping model {model_id} due to load failure")
                continue

        # Create agents with their assigned models
        for model_id, model_proposals in proposals_by_model.items():
            print(f"\n🎯 Creating {len(model_proposals)} agents for {model_id}...")

            # Get the specific model for these agents
            model, base_sampling_params = self.model_manager.get_model(model_id)
            if not model:
                print(f"⚠️ Model {model_id} not available, skipping agents")
                continue

            # Create all agents for this model
            for proposal in model_proposals:
                try:
                    agent_config = self._create_agent_config_from_proposal(proposal)

                    custom_params = self._create_custom_sampling_params(
                        base_sampling_params, proposal, len(agents)
                    )

                    # 🎯 PRESERVE: Agent gets its specific assigned model
                    agent = DDxAgent(agent_config, model, custom_params, model_id)
                    agents.append(agent)

                    model_type = "Conservative" if "conservative" in model_id else "Innovative"
                    print(f"✅ Generated: {agent.name} ({model_type})")
                    print(f"   Specialty: {agent.specialty.value}")
                    print(f"   Model: {model_id}")

                except Exception as e:
                    print(f"❌ Failed to create agent {proposal['name']}: {e}")
                    continue

        # ADD EVALUATOR AGENT using innovative model
        print(f"\n🔬 Adding Clinical Evaluator Agent...")
        evaluator_model, evaluator_params = self.model_manager.get_model("innovative_model")
        if evaluator_model:
            evaluator_agent = self._create_evaluator_agent_multi(evaluator_model, evaluator_params, "innovative_model")
            if evaluator_agent:
                agents.append(evaluator_agent)
                print(f"✅ Added: {evaluator_agent.name} (Clinical Evaluator)")

        print(f"\n🏥 Final Team: {len(agents)} agents with preserved multi-model innovation!")
        print(f"🎯 Conservative/Innovative split maintained for novel diagnostic reasoning")

        return agents

    def _create_evaluator_agent_multi(self, model, sampling_params, model_id: str) -> Optional[DDxAgent]:
        """Create evaluator agent for multi-model system"""
        try:
            from vllm import SamplingParams

            evaluator_config = AgentConfig(
                name="Dr. Clinical Evaluator",
                specialty=SpecialtyType.INTERNAL_MEDICINE,
                persona="Expert clinical evaluator specializing in diagnosis equivalence assessment.",
                reasoning_style="analytical",
                temperature=0.2,
                focus_areas=["diagnosis equivalence", "clinical terminology"],
                case_relevance_score=10.0,
                model_assignment=model_id
            )

            evaluator_sampling_params = SamplingParams(
                temperature=0.2,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
                stop=sampling_params.stop
            )

            evaluator_agent = DDxAgent(
                evaluator_config,
                model,
                evaluator_sampling_params,
                model_id
            )

            return evaluator_agent

        except Exception as e:
            print(f"⚠️ Failed to create evaluator agent: {e}")
            return None

    def cleanup_models_after_evaluation(self):
        """Clean up all models after evaluation"""
        print(f"\n🧹 Cleaning up multi-model system after evaluation...")
        self.model_manager.unload_all_models()
        print(f"✅ Multi-model cleanup complete")

    def _create_evaluator_agent(self) -> Optional[DDxAgent]:
        """Create dedicated evaluator agent using current loaded model"""
        try:
            # Use whatever model is currently active (should be the last loaded model)
            model, sampling_params = self.model_manager.get_active_model()
            if not model:
                print("⚠️ No active model for evaluator agent")
                return None

            # FIX: Check if sampling_params is None and provide fallbacks
            if not sampling_params:
                print("⚠️ No sampling params available, using defaults for evaluator")
                evaluator_sampling_params = SamplingParams(
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                    stop=["</s>", "<|im_end|>"]
                )
            else:
                evaluator_sampling_params = SamplingParams(
                    temperature=0.2,  # Low temperature for consistent evaluation
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                    stop=sampling_params.stop
                )

            evaluator_config = AgentConfig(
                name="Dr. Clinical Evaluator",
                specialty=SpecialtyType.INTERNAL_MEDICINE,
                persona="Expert clinical evaluator specializing in diagnosis equivalence assessment with extensive knowledge across all medical specialties. Focuses on determining whether different diagnostic terms represent the same underlying medical condition.",
                reasoning_style="analytical",
                temperature=0.2,
                focus_areas=["diagnosis equivalence", "clinical terminology", "medical classification", "differential diagnosis"],
                case_relevance_score=10.0,
                model_assignment=self.model_manager.active_model_id or "unknown"
            )

            evaluator_agent = DDxAgent(
                evaluator_config,
                model,
                evaluator_sampling_params,
                self.model_manager.active_model_id or "unknown"
            )

            return evaluator_agent

        except Exception as e:
            print(f"⚠️ Failed to create evaluator agent: {e}")
            return None

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
        final_temp = max(0.1, min(1.0, base_temp + temp_modifier + agent_variation))

        # 🆕 NEW: Create detailed temperature log
        temp_log = {
            'agent_name': proposal.get('name', 'Unknown'),
            'model_id': model_id,
            'model_type': 'conservative' if 'conservative' in model_id.lower() else 'innovative',
            'base_temperature': base_temp,
            'reasoning_style': reasoning_style,
            'style_modifier': temp_modifier,
            'agent_index': agent_index,
            'index_variation': agent_variation,
            'final_temperature': final_temp,
            'calculation_breakdown': f"{base_temp} + {temp_modifier} + {agent_variation} = {final_temp} (bounded to {final_temp})"
        }

        # 🆕 NEW: Store temperature log in the proposal for later use
        proposal['temperature_log'] = temp_log

        # 🆕 NEW: Print temperature calculation for immediate feedback
        print(f"   🌡️ {temp_log['agent_name']}: {temp_log['calculation_breakdown']}")

        return final_temp

    def _create_custom_sampling_params(self, base_params: SamplingParams | None, 
                                 proposal: Dict[str, Any], agent_count: int) -> SamplingParams:
        """Create custom sampling parameters for this specific agent"""
        # Handle None base_params
        if base_params is None:
            base_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
                stop=["</s>", "<|im_end|>"]
            )
        
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

    def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about currently active models and agents"""
        info = {
            'models_in_use': getattr(self, '_models_in_use', set()),
            'active_agents': len(getattr(self, '_active_agents', [])),
            'current_model': self.model_manager.active_model_id
        }
        return info

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

        # Check if agent_generator is available
        if not self.agent_generator:
            print("❌ Agent generator not initialized")
            return {'success': False, 'error': 'Agent generator not available'}

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
    if not test_sequential_loading():
        print("❌ Sequential loading test failed")
        return False

    if not ddx.initialize():
        print("❌ Failed to initialize DDx system")
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

def test_sequential_loading():
    """Test the sequential loading system"""
    print("🧪 Testing Sequential Model Loading System...")

    # Load configs
    configs = load_system_config()
    model_manager = ModelManager(configs)

    # Initialize system
    if not model_manager.initialize_models():
        print("❌ System initialization failed")
        return False

    # Test loading each model sequentially
    available_models = model_manager.get_available_models()

    for model_id in available_models:
        print(f"\n🔄 Testing model: {model_id}")

        # Load model
        if model_manager.load_model(model_id):
            print(f"   ✅ {model_id} loaded successfully")

            # Get model reference
            model, params = model_manager.get_active_model()
            if model and params:
                print(f"   ✅ Model reference obtained")
            else:
                print(f"   ❌ Failed to get model reference")
                return False

            # Unload model
            model_manager.unload_model()
            print(f"   ✅ {model_id} unloaded successfully")
        else:
            print(f"   ❌ Failed to load {model_id}")
            return False

    print("\n🎉 Sequential loading test PASSED!")
    return True

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
