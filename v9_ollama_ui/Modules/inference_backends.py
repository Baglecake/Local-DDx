# =============================================================================
# Inference Backends - Abstraction Layer for LLM Inference
# =============================================================================

"""
Provides a unified interface for different LLM inference backends.
Supports Ollama (primary for M4 Mac) with MLX as future option.
"""

import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SamplingConfig:
    """Backend-agnostic sampling configuration"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>", "<|im_end|>", "<|eot_id|>"])

    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert to Ollama API format"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "stop": self.stop_tokens
        }


class InferenceBackend(ABC):
    """Abstract interface for LLM inference"""

    @abstractmethod
    def load(self, model_name: str, **kwargs) -> bool:
        """Load/prepare a model for inference"""
        pass

    @abstractmethod
    def generate(self, prompt: str, config: SamplingConfig) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload current model (if applicable)"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend and model info"""
        pass


class OllamaBackend(InferenceBackend):
    """
    Ollama-based inference backend for Apple Silicon.

    Uses the Ollama REST API for model management and inference.
    Models are loaded on-demand and cached by Ollama.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_model: Optional[str] = None
        self.model_info: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except requests.exceptions.RequestException as e:
            print(f"Failed to get models: {e}")
        return []

    def load(self, model_name: str, **kwargs) -> bool:
        """
        Prepare a model for inference.

        Ollama loads models on first use, but we can warm it up
        with a small request to ensure it's ready.
        """
        print(f"Preparing model: {model_name}")

        # Check if model exists
        available = self.get_available_models()
        if model_name not in available:
            print(f"Model {model_name} not found. Available: {available}")
            print(f"You can pull it with: ollama pull {model_name}")
            return False

        self.current_model = model_name

        # Warm up the model with a minimal request
        try:
            print(f"   Warming up {model_name}...")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "options": {"num_predict": 1},
                    "stream": False
                },
                timeout=120  # First load can take time
            )

            load_time = time.time() - start_time

            if response.status_code == 200:
                print(f"   Model ready in {load_time:.1f}s")
                self.model_info = {
                    "name": model_name,
                    "load_time": load_time,
                    "status": "ready"
                }
                return True
            else:
                print(f"   Warmup failed: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"   Load failed: {e}")
            return False

    def generate(self, prompt: str, config: SamplingConfig) -> str:
        """Generate text using Ollama API"""
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load() first.")

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.current_model,
                    "prompt": prompt,
                    "options": config.to_ollama_options(),
                    "stream": False
                },
                timeout=300  # 5 min timeout for long generations
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Generation failed: {response.status_code}")
                return f"Error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            print(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def generate_chat(self, messages: List[Dict[str, str]], config: SamplingConfig) -> str:
        """Generate using chat format (for instruction-tuned models)"""
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load() first.")

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.current_model,
                    "messages": messages,
                    "options": config.to_ollama_options(),
                    "stream": False
                },
                timeout=300
            )

            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                print(f"Chat generation failed: {response.status_code}")
                return f"Error: {response.status_code}"

        except requests.exceptions.RequestException as e:
            print(f"Chat generation error: {e}")
            return f"Error: {str(e)}"

    def unload(self) -> None:
        """
        Unload current model from memory.

        Note: Ollama manages model lifecycle automatically, but we can
        force unload by loading a tiny model or using the API.
        """
        if self.current_model:
            print(f"Releasing model: {self.current_model}")
            # Ollama doesn't have explicit unload, but setting keep_alive=0
            # on next request will unload after completion
            self.current_model = None
            self.model_info = {}

    def get_info(self) -> Dict[str, Any]:
        """Get backend and model information"""
        return {
            "backend": "ollama",
            "base_url": self.base_url,
            "available": self.is_available(),
            "current_model": self.current_model,
            "model_info": self.model_info,
            "available_models": self.get_available_models()
        }


# =============================================================================
# Backend Factory
# =============================================================================

def create_backend(backend_type: str = "ollama", **kwargs) -> InferenceBackend:
    """
    Factory function to create inference backends.

    Args:
        backend_type: "ollama" (default) or "mlx" (future)
        **kwargs: Backend-specific configuration

    Returns:
        Configured InferenceBackend instance
    """
    if backend_type == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaBackend(base_url=base_url)
    elif backend_type == "mlx":
        raise NotImplementedError("MLX backend coming soon")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# =============================================================================
# Testing
# =============================================================================

def test_ollama_backend():
    """Test the Ollama backend"""
    print("Testing Ollama Backend")
    print("=" * 50)

    backend = create_backend("ollama")

    # Check availability
    if not backend.is_available():
        print("Ollama is not running. Start it with: ollama serve")
        return False

    print("Ollama is available")

    # Show available models
    info = backend.get_info()
    print(f"Available models: {info['available_models']}")

    # Test with a small model
    test_model = "phi3:mini"  # Small, fast model for testing

    if test_model not in info['available_models']:
        print(f"{test_model} not available. Trying llama3:latest...")
        test_model = "llama3:latest"

    if test_model not in info['available_models']:
        print("No suitable test model found")
        return False

    # Load model
    if not backend.load(test_model):
        print("Failed to load model")
        return False

    # Test generation
    print("\nTesting generation...")
    config = SamplingConfig(temperature=0.7, max_tokens=50)

    response = backend.generate(
        "What are three common symptoms of pneumonia? Be brief.",
        config
    )

    print(f"Response: {response[:200]}...")

    # Test chat format
    print("\nTesting chat format...")
    messages = [
        {"role": "system", "content": "You are a medical specialist. Be concise."},
        {"role": "user", "content": "Name one cardiac emergency."}
    ]

    chat_response = backend.generate_chat(messages, config)
    print(f"Chat response: {chat_response[:200]}...")

    print("\nOllama backend test passed!")
    return True


if __name__ == "__main__":
    test_ollama_backend()
