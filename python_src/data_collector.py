#!/usr/bin/env python3
"""
Multi-Source Data Collector for LLM Market Intelligence

Collects pricing, quality, and volume data from multiple sources:
- OpenRouter API (pricing, models)
- LiteLLM GitHub (comprehensive pricing)
- llm-prices.com (current + historical pricing)
- Arena leaderboard (quality/ELO - manual updates)

Run daily via GitHub Action to keep data fresh.
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PRICES_DIR = DATA_DIR / "prices"


# =============================================================================
# DATA SOURCES
# =============================================================================

def fetch_openrouter_models() -> dict:
    """
    Fetch model pricing from OpenRouter API.

    Returns dict of model_id -> pricing info
    """
    url = "https://openrouter.ai/api/v1/models"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = {}
        for model in data.get("data", []):
            model_id = model.get("id")
            pricing = model.get("pricing", {})

            if model_id and pricing:
                # Convert per-token to per-MTok
                input_price = float(pricing.get("prompt", 0)) * 1_000_000
                output_price = float(pricing.get("completion", 0)) * 1_000_000

                models[model_id] = {
                    "input_mtok": round(input_price, 4),
                    "output_mtok": round(output_price, 4),
                    "context_length": model.get("context_length"),
                    "name": model.get("name"),
                    "source": "openrouter"
                }

        print(f"[OpenRouter] Fetched {len(models)} models")
        return models

    except Exception as e:
        print(f"[OpenRouter] Error: {e}")
        return {}


def fetch_litellm_prices() -> dict:
    """
    Fetch pricing from LiteLLM's GitHub database.

    Comprehensive coverage of many providers.
    """
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = {}
        for model_id, info in data.items():
            if isinstance(info, dict):
                # LiteLLM uses per-token pricing
                input_price = info.get("input_cost_per_token", 0)
                output_price = info.get("output_cost_per_token", 0)

                if input_price or output_price:
                    # Convert to per-MTok
                    models[model_id] = {
                        "input_mtok": round(float(input_price) * 1_000_000, 4),
                        "output_mtok": round(float(output_price) * 1_000_000, 4),
                        "context_length": info.get("max_tokens") or info.get("max_input_tokens"),
                        "source": "litellm"
                    }

        print(f"[LiteLLM] Fetched {len(models)} models")
        return models

    except Exception as e:
        print(f"[LiteLLM] Error: {e}")
        return {}


def fetch_llm_prices_current() -> dict:
    """
    Fetch current pricing from llm-prices.com (Simon Willison).

    Clean, curated dataset focused on major models.
    """
    url = "https://www.llm-prices.com/current-v1.json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = {}
        for model in data.get("prices", []):
            # Construct a model ID from vendor and id
            vendor = model.get("vendor", "").lower()
            model_name = model.get("id", "").lower()
            model_id = f"{vendor}/{model_name}" if vendor else model_name

            # Prices are already per MTok
            input_price = model.get("input", 0) or 0
            output_price = model.get("output", 0) or 0

            if input_price or output_price:
                models[model_id] = {
                    "input_mtok": round(float(input_price), 4),
                    "output_mtok": round(float(output_price), 4),
                    "name": model.get("name"),
                    "source": "llm-prices"
                }

        print(f"[llm-prices] Fetched {len(models)} models")
        return models

    except Exception as e:
        print(f"[llm-prices] Error: {e}")
        return {}


def fetch_llm_prices_historical() -> list:
    """
    Fetch historical pricing from llm-prices.com.

    Returns list of historical snapshots.
    """
    url = "https://www.llm-prices.com/historical-v1.json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        snapshots = data.get("snapshots", [])
        print(f"[llm-prices-historical] Fetched {len(snapshots)} snapshots")
        return snapshots

    except Exception as e:
        print(f"[llm-prices-historical] Error: {e}")
        return []


def fetch_pricepertoken() -> dict:
    """
    Fetch pricing from pricepertoken.com via their API.

    Returns dict of model_id -> pricing info.
    """
    url = "https://api.pricepertoken.com/api/pricing"

    try:
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; OccupantIndex/1.0)",
            "Accept": "application/json",
        })
        response.raise_for_status()
        data = response.json()

        models = {}

        # The API returns a dict with "results" containing model objects
        model_list = data.get("results", []) if isinstance(data, dict) else data

        for model in model_list:
            if not isinstance(model, dict):
                continue

            # Extract model identifiers
            slug = model.get("slug") or model.get("model_id") or model.get("id", "")
            provider = model.get("provider_name") or model.get("provider", "")
            name = model.get("model_name") or model.get("name", slug)

            # Extract pricing (already per 1M tokens)
            input_price = model.get("input_price_per_1m_tokens") or model.get("input_price", 0)
            output_price = model.get("output_price_per_1m_tokens") or model.get("output_price", 0)

            if slug and (input_price or output_price):
                # Normalize provider name
                provider_normalized = provider.lower().replace(" ", "").replace(".", "")
                provider_map = {
                    "openai": "openai",
                    "anthropic": "anthropic",
                    "google": "google",
                    "mistralai": "mistralai",
                    "mistral": "mistralai",
                    "deepseek": "deepseek",
                    "meta": "meta-llama",
                    "xai": "x-ai",
                    "cohere": "cohere",
                    "amazon": "amazon",
                    "qwen": "qwen",
                    "alibaba": "qwen",
                }
                provider_key = provider_map.get(provider_normalized, provider_normalized)

                model_id = f"{provider_key}/{slug}" if provider_key else slug

                models[model_id] = {
                    "input_mtok": round(float(input_price), 4),
                    "output_mtok": round(float(output_price), 4),
                    "name": name,
                    "context_length": model.get("context_window") or model.get("context_length"),
                    "source": "pricepertoken"
                }

        print(f"[pricepertoken] Fetched {len(models)} models")
        return models

    except Exception as e:
        print(f"[pricepertoken] Error: {e}")
        return {}


# =============================================================================
# DATA AGGREGATION
# =============================================================================

def merge_pricing_sources(*sources) -> dict:
    """
    Merge pricing from multiple sources.

    Priority: OpenRouter > LiteLLM > llm-prices
    (OpenRouter is the primary source, others fill gaps)
    """
    merged = {}

    # Process in reverse priority order so higher priority overwrites
    for source in reversed(sources):
        for model_id, data in source.items():
            if model_id not in merged:
                merged[model_id] = data
            else:
                # Keep existing but note we have multiple sources
                merged[model_id]["also_in"] = data.get("source")

    return merged


def normalize_model_id(model_id: str) -> str:
    """
    Normalize model ID to consistent format.

    Examples:
    - "gpt-4o" -> "openai/gpt-4o"
    - "claude-3-sonnet" -> "anthropic/claude-3-sonnet"
    """
    model_id = model_id.lower().strip()

    # Already has provider prefix
    if "/" in model_id:
        return model_id

    # Infer provider from model name
    provider_hints = {
        "gpt": "openai",
        "o1": "openai",
        "o3": "openai",
        "claude": "anthropic",
        "gemini": "google",
        "llama": "meta-llama",
        "mistral": "mistralai",
        "mixtral": "mistralai",
        "deepseek": "deepseek",
        "qwen": "qwen",
        "command": "cohere",
    }

    for hint, provider in provider_hints.items():
        if model_id.startswith(hint):
            return f"{provider}/{model_id}"

    return model_id


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_prices_snapshot(prices: dict):
    """Save aggregated pricing snapshot."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    snapshot = {
        "date": date_str,
        "generated_at": now.isoformat(),
        "model_count": len(prices),
        "models": prices
    }

    # Save dated snapshot
    snapshot_path = PRICES_DIR / f"prices_{date_str}.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Save as latest
    latest_path = PRICES_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"[Prices] Saved {len(prices)} models to {snapshot_path}")


def save_historical_prices(historical: list):
    """Save historical pricing data."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    path = PRICES_DIR / "historical_external.json"
    with open(path, "w") as f:
        json.dump({
            "source": "llm-prices.com",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "snapshots": historical
        }, f, indent=2)

    print(f"[Historical] Saved {len(historical)} snapshots")


# =============================================================================
# MAIN
# =============================================================================

def collect_all():
    """Run full data collection pipeline."""
    print("=" * 60)
    print("  LLM DATA COLLECTOR")
    print("=" * 60)

    # Phase 1: Fetch from all sources
    print("\n--- FETCHING DATA ---")

    openrouter = fetch_openrouter_models()
    litellm = fetch_litellm_prices()
    llm_prices = fetch_llm_prices_current()
    pricepertoken = fetch_pricepertoken()
    historical = fetch_llm_prices_historical()

    # Phase 2: Merge pricing
    print("\n--- MERGING DATA ---")

    merged = merge_pricing_sources(openrouter, litellm, llm_prices, pricepertoken)
    print(f"[Merged] {len(merged)} unique models")

    # Phase 3: Save
    print("\n--- SAVING DATA ---")

    save_prices_snapshot(merged)
    if historical:
        save_historical_prices(historical)

    # Summary
    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)

    print(f"\nSources:")
    print(f"  OpenRouter: {len(openrouter)} models")
    print(f"  LiteLLM: {len(litellm)} models")
    print(f"  llm-prices: {len(llm_prices)} models")
    print(f"  pricepertoken: {len(pricepertoken)} models")
    print(f"  Historical snapshots: {len(historical)}")
    print(f"\nMerged total: {len(merged)} unique models")

    # Sample top models by price
    sorted_models = sorted(
        [(k, v) for k, v in merged.items() if v.get("output_mtok", 0) > 0],
        key=lambda x: x[1]["output_mtok"],
        reverse=True
    )

    print("\nMost expensive (output):")
    for model_id, data in sorted_models[:5]:
        print(f"  ${data['output_mtok']:.2f}/MTok - {model_id}")

    print("\nCheapest (output):")
    for model_id, data in sorted_models[-5:]:
        print(f"  ${data['output_mtok']:.4f}/MTok - {model_id}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    collect_all()
