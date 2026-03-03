"""
VLLMBackend -- InferenceBackend implementation for vLLM (Colab/GPU).

Uses vLLM's in-process LLM class for inference without an HTTP server.
Designed for batch benchmarking on Colab Pro+ with A100 GPU.
"""

import time
import sys
import os
from typing import Dict, List, Optional, Any

# Add v9 modules to path for the InferenceBackend ABC
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'v9_ollama_ui', 'Modules'))
from inference_backends import InferenceBackend, SamplingConfig


class VLLMBackend(InferenceBackend):
    """
    vLLM-based inference backend for GPU systems (Colab A100).

    Loads model once via vllm.LLM, reuses for all calls.
    Temperature/sampling differ per-call, not per-model-load.
    """

    def __init__(self, model_name: str = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ",
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 4096,
                 quantization: str = "gptq_marlin",
                 dtype: str = "float16",
                 **kwargs):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.dtype = dtype
        self.extra_kwargs = kwargs
        self.llm = None
        self.tokenizer = None
        self.loaded_model_name: Optional[str] = None

    def is_available(self) -> bool:
        """Check if vLLM and CUDA are available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self, model_name: str, **kwargs) -> bool:
        """
        Load model into GPU via vLLM.

        If model is already loaded (same or different name), this is a no-op.
        vLLM does not support hot-swapping models in-process. Both conservative
        and innovative configs point to the same model — only temperature differs.
        """
        if self.llm is not None:
            if self.loaded_model_name == model_name:
                return True
            # Different model requested but can't hot-swap — use what's loaded
            print(f"  vLLM model already loaded ({self.loaded_model_name}), reusing")
            return True

        try:
            from vllm import LLM
            print(f"  Loading {model_name} via vLLM...")
            start = time.time()

            self.llm = LLM(
                model=model_name,
                quantization=self.quantization,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                **self.extra_kwargs
            )

            self.loaded_model_name = model_name
            self.tokenizer = self.llm.get_tokenizer()
            elapsed = time.time() - start
            print(f"  Model loaded in {elapsed:.1f}s")
            return True

        except Exception as e:
            print(f"  vLLM load failed: {e}")
            return False

    def generate(self, prompt: str, config: SamplingConfig) -> str:
        """Generate text from a raw prompt"""
        if self.llm is None:
            return "Error: No model loaded. Call load() first."

        try:
            from vllm import SamplingParams
            params = SamplingParams(**config.to_vllm_params())
            outputs = self.llm.generate([prompt], params, use_tqdm=False)
            return outputs[0].outputs[0].text
        except Exception as e:
            print(f"  vLLM generate error: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]],
                      config: SamplingConfig) -> str:
        """Generate using chat format — primary method for v10 pipeline"""
        if self.llm is None:
            return "Error: No model loaded. Call load() first."

        try:
            from vllm import SamplingParams
            params = SamplingParams(**config.to_vllm_params())

            # Try vLLM's native chat() method first
            try:
                outputs = self.llm.chat(
                    messages=[messages],
                    sampling_params=params,
                    use_tqdm=False,
                )
                return outputs[0].outputs[0].text
            except (TypeError, AttributeError):
                # Fallback: manually apply chat template and use generate()
                if self.tokenizer is None:
                    self.tokenizer = self.llm.get_tokenizer()

                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                outputs = self.llm.generate([prompt], params, use_tqdm=False)
                return outputs[0].outputs[0].text

        except Exception as e:
            print(f"  vLLM chat error: {e}")
            return f"Error: {str(e)}"

    def unload(self) -> None:
        """Release model from GPU memory"""
        if self.llm is not None:
            print(f"  Releasing vLLM model: {self.loaded_model_name}")
            del self.llm
            self.llm = None
            self.tokenizer = None
            self.loaded_model_name = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def get_info(self) -> Dict[str, Any]:
        """Get backend and model info"""
        info = {
            "backend": "vllm",
            "model_name": self.loaded_model_name or self.model_name,
            "loaded": self.llm is not None,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
        except Exception:
            pass
        return info

    def get_available_models(self) -> List[str]:
        """Return list of available models (just the loaded one for vLLM)"""
        if self.loaded_model_name:
            return [self.loaded_model_name]
        return [self.model_name]
