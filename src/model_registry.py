"""
Unified model registry combining multiple data sources.
Handles tier classification and model ID mapping.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from fetch_openrouter import fetch_openrouter_models
from fetch_litellm import fetch_litellm_models, normalize_model_id

TIERS_PATH = Path(__file__).parent.parent / "data" / "models" / "tiers.json"


# Model tier definitions based on capability and pricing
MODEL_TIERS = {
    "budget": {
        "description": "Cheapest throughput models for high-volume, low-complexity tasks",
        "criteria": "< $1/MTok input, fast inference",
        "models": [
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-lite-001",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-haiku-4.5",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    },
    "general": {
        "description": "Balanced cost/capability for typical workloads",
        "criteria": "$1-15/MTok input, strong general benchmarks",
        "models": [
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.7-sonnet",
            "meta-llama/llama-3.1-70b-instruct",
        ]
    },
    "frontier": {
        "description": "Best available capability regardless of cost",
        "criteria": "Top-tier benchmark scores",
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.7-sonnet",
            "google/gemini-2.5-pro",
            "anthropic/claude-opus-4",
        ]
    },
    "reasoning": {
        "description": "Models optimized for complex reasoning and chain-of-thought",
        "criteria": "Explicit reasoning capability, extended thinking",
        "models": [
            "openai/o1",
            "openai/o3-mini",
            "openai/o1-mini",
            "deepseek/deepseek-r1",
            "anthropic/claude-3.7-sonnet:thinking",
        ]
    },
    "longctx": {
        "description": "Models supporting >100K context efficiently",
        "criteria": "Context window >100K tokens",
        "models": [
            "google/gemini-2.5-pro",        # 1M+ context
            "google/gemini-2.0-flash-001",  # 1M context
            "anthropic/claude-3.5-sonnet",  # 200K context
            "anthropic/claude-3.7-sonnet",  # 200K context
            "openai/gpt-4o",                # 128K context
        ]
    }
}


def build_unified_registry(openrouter_key: str = None) -> dict:
    """
    Build unified model registry from all sources.

    Priority: OpenRouter (real-time) > LiteLLM (comprehensive)

    Returns:
    {
        "models": {
            "openai/gpt-4o": { ... pricing and metadata ... },
            ...
        },
        "tiers": {
            "budget": ["openai/gpt-4o-mini", ...],
            ...
        },
        "meta": {
            "sources": ["openrouter", "litellm"],
            "generated_at": "...",
            "total_models": 500
        }
    }
    """
    # Fetch from both sources
    openrouter_models = fetch_openrouter_models(openrouter_key)
    litellm_models = fetch_litellm_models()

    # Merge: OpenRouter takes priority
    unified = {}

    # Add OpenRouter models first (primary source)
    for model_id, info in openrouter_models.items():
        unified[model_id] = info

    # Add LiteLLM models not in OpenRouter
    litellm_added = 0
    for litellm_id, info in litellm_models.items():
        normalized_id = normalize_model_id(litellm_id)
        if normalized_id not in unified:
            unified[normalized_id] = info
            unified[normalized_id]["original_id"] = litellm_id
            litellm_added += 1

    print(f"[Registry] {len(openrouter_models)} from OpenRouter, {litellm_added} additional from LiteLLM")
    print(f"[Registry] Total: {len(unified)} models")

    # Build tier assignments (only include models we have pricing for)
    active_tiers = {}
    for tier_name, tier_info in MODEL_TIERS.items():
        active_models = [m for m in tier_info["models"] if m in unified]
        active_tiers[tier_name] = active_models
        print(f"[Registry] Tier '{tier_name}': {len(active_models)}/{len(tier_info['models'])} models available")

    # Save tier assignments
    TIERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIERS_PATH, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tiers": active_tiers,
            "definitions": {k: {"description": v["description"], "criteria": v["criteria"]}
                           for k, v in MODEL_TIERS.items()}
        }, f, indent=2)

    return {
        "models": unified,
        "tiers": active_tiers,
        "meta": {
            "sources": ["openrouter", "litellm"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_models": len(unified),
            "openrouter_count": len(openrouter_models),
            "litellm_added": litellm_added
        }
    }


def get_tier_models(registry: dict, tier: str) -> list:
    """Get list of model IDs in a tier."""
    return registry.get("tiers", {}).get(tier, [])


def get_model_pricing(registry: dict, model_id: str) -> dict:
    """Get pricing for a specific model."""
    return registry.get("models", {}).get(model_id, {})


def calculate_tier_average(registry: dict, tier: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate average cost for a workload across all models in a tier.

    Args:
        registry: Unified model registry
        tier: Tier name (budget, general, frontier, reasoning, longctx)
        input_tokens: Number of input tokens in workload
        output_tokens: Number of output tokens in workload

    Returns:
        Average cost in dollars for the workload, or None if no models available
    """
    tier_models = get_tier_models(registry, tier)
    if not tier_models:
        return None

    costs = []
    for model_id in tier_models:
        pricing = get_model_pricing(registry, model_id)
        if not pricing:
            continue

        input_cost = pricing.get("input_cost_per_token", 0)
        output_cost = pricing.get("output_cost_per_token", 0)

        if input_cost > 0 or output_cost > 0:
            cost = (input_tokens * input_cost) + (output_tokens * output_cost)
            costs.append(cost)

    if not costs:
        return None

    return sum(costs) / len(costs)


if __name__ == "__main__":
    import os
    registry = build_unified_registry(os.environ.get("OPENROUTER_API_KEY"))

    print(f"\n{'='*60}")
    print("Sample tier costs for Chat/Drafting workload (2K in, 500 out):")
    print("="*60)

    for tier in ["budget", "general", "frontier", "reasoning", "longctx"]:
        avg_cost = calculate_tier_average(registry, tier, 2000, 500)
        if avg_cost:
            print(f"  {tier:12} ${avg_cost*1000:.4f} per 1K workloads")
