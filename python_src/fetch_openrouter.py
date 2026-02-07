"""
OpenRouter API client for fetching real-time model pricing.
Primary data source for Compute CPI.
"""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

OPENROUTER_API = "https://openrouter.ai/api/v1/models"
CACHE_PATH = Path(__file__).parent.parent / "data" / "models" / "openrouter.json"


def fetch_openrouter_models(api_key: str = None) -> dict:
    """
    Fetch all models from OpenRouter API.

    Returns dict mapping model_id -> pricing info:
    {
        "openai/gpt-4o": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "context_length": 128000,
            "source": "openrouter"
        },
        ...
    }
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(OPENROUTER_API, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[OpenRouter] Failed to fetch: {e}")
        # Try to load from cache
        if CACHE_PATH.exists():
            print(f"[OpenRouter] Loading from cache")
            with open(CACHE_PATH) as f:
                return json.load(f)
        return {}

    models = {}
    for model in data.get("data", []):
        model_id = model.get("id")
        pricing = model.get("pricing", {})

        # Convert string prices to float
        input_cost = float(pricing.get("prompt", 0) or 0)
        output_cost = float(pricing.get("completion", 0) or 0)

        if input_cost > 0 or output_cost > 0:
            models[model_id] = {
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
                "context_length": model.get("context_length", 0),
                "source": "openrouter",
                "name": model.get("name", model_id),
            }

    # Cache the results
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "models": models
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"[OpenRouter] Fetched {len(models)} models with pricing")
    return models


if __name__ == "__main__":
    models = fetch_openrouter_models()
    print(f"\nTotal models: {len(models)}")

    # Show some examples
    examples = ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-pro"]
    for model_id in examples:
        if model_id in models:
            m = models[model_id]
            input_per_mtok = m["input_cost_per_token"] * 1_000_000
            output_per_mtok = m["output_cost_per_token"] * 1_000_000
            print(f"  {model_id}: ${input_per_mtok:.2f}/${output_per_mtok:.2f} per MTok")
