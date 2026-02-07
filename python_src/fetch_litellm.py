"""
LiteLLM data fetcher for comprehensive model coverage.
Secondary data source - used for models not in OpenRouter.
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_PATH = Path(__file__).parent.parent / "data" / "models" / "litellm.json"


def fetch_litellm_models() -> dict:
    """
    Fetch model pricing from LiteLLM's comprehensive database.

    Returns dict mapping model_id -> pricing info:
    {
        "gpt-4o": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "context_length": 128000,
            "max_output_tokens": 16384,
            "provider": "openai",
            "supports_vision": True,
            "supports_function_calling": True,
            "source": "litellm"
        },
        ...
    }
    """
    try:
        response = requests.get(LITELLM_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[LiteLLM] Failed to fetch: {e}")
        if CACHE_PATH.exists():
            print(f"[LiteLLM] Loading from cache")
            with open(CACHE_PATH) as f:
                return json.load(f).get("models", {})
        return {}

    models = {}
    for model_id, info in data.items():
        # Skip non-chat models and sample spec
        if model_id == "sample_spec":
            continue
        mode = info.get("mode", "")
        if mode and mode not in ["chat", "completion"]:
            continue

        input_cost = info.get("input_cost_per_token", 0) or 0
        output_cost = info.get("output_cost_per_token", 0) or 0

        # Skip models without pricing
        if input_cost == 0 and output_cost == 0:
            continue

        models[model_id] = {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
            "context_length": info.get("max_input_tokens") or info.get("max_tokens", 0),
            "max_output_tokens": info.get("max_output_tokens", 0),
            "provider": info.get("litellm_provider", ""),
            "supports_vision": info.get("supports_vision", False),
            "supports_function_calling": info.get("supports_function_calling", False),
            "supports_reasoning": info.get("supports_reasoning", False),
            "cache_read_cost": info.get("cache_read_input_token_cost", 0),
            "batch_input_cost": info.get("input_cost_per_token_batches", 0),
            "batch_output_cost": info.get("output_cost_per_token_batches", 0),
            "source": "litellm"
        }

    # Cache the results
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "models": models
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"[LiteLLM] Fetched {len(models)} chat/completion models with pricing")
    return models


# Map LiteLLM model IDs to OpenRouter format
LITELLM_TO_OPENROUTER = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    "o3-mini": "openai/o3-mini",
    "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
    "claude-3-5-haiku-20241022": "anthropic/claude-3.5-haiku",
    "claude-3-opus-20240229": "anthropic/claude-3-opus",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.0-pro": "google/gemini-2.5-pro",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-r1",
}


def normalize_model_id(litellm_id: str) -> str:
    """Convert LiteLLM model ID to OpenRouter format."""
    if litellm_id in LITELLM_TO_OPENROUTER:
        return LITELLM_TO_OPENROUTER[litellm_id]

    # Try to infer provider from ID
    if litellm_id.startswith("gpt-") or litellm_id.startswith("o1") or litellm_id.startswith("o3"):
        return f"openai/{litellm_id}"
    if litellm_id.startswith("claude-"):
        return f"anthropic/{litellm_id}"
    if litellm_id.startswith("gemini-"):
        return f"google/{litellm_id}"
    if litellm_id.startswith("deepseek-"):
        return f"deepseek/{litellm_id}"

    return litellm_id


if __name__ == "__main__":
    models = fetch_litellm_models()
    print(f"\nTotal models: {len(models)}")

    # Show some examples
    examples = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash"]
    for model_id in examples:
        if model_id in models:
            m = models[model_id]
            input_per_mtok = m["input_cost_per_token"] * 1_000_000
            output_per_mtok = m["output_cost_per_token"] * 1_000_000
            print(f"  {model_id}: ${input_per_mtok:.2f}/${output_per_mtok:.2f} per MTok")
