"""
Ollama LLM wrapper for local inference.

Supports Mistral-7B and Llama-2-7B via langchain-ollama.
Includes a health check to verify Ollama is running and
the requested model is available.
"""

import requests
from langchain_ollama.llms import OllamaLLM

from config import GenerationConfig, LLM_MODELS


class LLMConnectionError(Exception):
    """Raised when Ollama is not reachable or model is unavailable."""
    pass


def check_ollama_health(base_url: str, model_name: str) -> bool:
    """
    Verify that Ollama is running and the requested model is available.

    Args:
        base_url: Ollama API base URL (e.g. http://localhost:11434).
        model_name: Model name as registered in Ollama.

    Returns:
        True if healthy.

    Raises:
        LLMConnectionError if Ollama is unreachable or model is missing.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]

        # Ollama model names may include ":latest" suffix
        available = any(
            model_name in m or m.startswith(model_name)
            for m in models
        )

        if not available:
            raise LLMConnectionError(
                f"Model '{model_name}' not found in Ollama. "
                f"Available models: {models}. "
                f"Run: ollama pull {model_name}"
            )
        return True

    except requests.ConnectionError:
        raise LLMConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Make sure Ollama is running: `ollama serve`"
        )


def create_llm(config: GenerationConfig, skip_health_check: bool = False) -> OllamaLLM:
    """
    Create an OllamaLLM instance from config.

    Args:
        config: GenerationConfig with model name and parameters.
        skip_health_check: If True, skip the Ollama connectivity check.

    Returns:
        Configured OllamaLLM instance.
    """
    model_name = LLM_MODELS.get(config.model_name, config.model_name)

    if not skip_health_check:
        check_ollama_health(config.ollama_base_url, model_name)

    llm = OllamaLLM(
        model=model_name,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        num_predict=config.max_tokens,
    )

    return llm
