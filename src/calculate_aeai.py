#!/usr/bin/env python3
"""
AI Economic Activity Index (AEAI) Calculator

Measures global AI economic activity through a synthetic unit (AIU)
modeled on the IMF's Special Drawing Rights.

Basket composition:
- Token volumes (60% weight): Cross-provider token throughput
- Inferred spend (30% weight): Token volumes × current pricing
- Energy proxy (10% weight): Estimated from token volumes and model efficiency

Base period: February 2025 = 100
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AEAI_DIR = DATA_DIR / "aeai"
RANKINGS_DIR = DATA_DIR / "rankings"
PRICES_DIR = DATA_DIR / "prices"

# Ensure AEAI directory exists
AEAI_DIR.mkdir(parents=True, exist_ok=True)

# Constants for spend estimation
# Typical AI workload mix: ~70% input tokens, ~30% output tokens
INPUT_OUTPUT_RATIO = {"input_weight": 0.7, "output_weight": 0.3}

# Energy estimation factors (Joules per billion tokens)
# Based on published GPU efficiency metrics and datacenter PUE
# These are conservative estimates for 2025 baseline models
ENERGY_PER_BILLION_TOKENS = {
    "budget": 50_000,      # Efficient small models (Flash, Haiku, Mini)
    "general": 150_000,    # Mid-tier models (Sonnet, GPT-4o)
    "frontier": 300_000,   # Large frontier models (Opus, o1)
    "reasoning": 500_000,  # Reasoning models with extended CoT
    "default": 150_000     # Unknown models
}


def load_rankings() -> Dict[str, Any]:
    """Load latest token volume rankings."""
    rankings_path = RANKINGS_DIR / "latest.json"

    if not rankings_path.exists():
        print("[AEAI] Warning: No rankings data found")
        return {"models": {}, "total_tokens": 0}

    with open(rankings_path) as f:
        return json.load(f)


def load_prices() -> Dict[str, Any]:
    """Load latest pricing data."""
    prices_path = PRICES_DIR / "latest.json"

    if not prices_path.exists():
        print("[AEAI] Warning: No pricing data found")
        return {"models": {}}

    with open(prices_path) as f:
        return json.load(f)


def load_model_tiers() -> Dict[str, Any]:
    """Load model tier classifications for energy estimation."""
    tiers_path = DATA_DIR / "models" / "tiers.json"

    if not tiers_path.exists():
        print("[AEAI] Warning: No tier data found")
        return {"tiers": {}}

    with open(tiers_path) as f:
        return json.load(f)


def get_model_tier(model_id: str, tiers_data: Dict) -> str:
    """Determine the tier for a model (for energy estimation)."""
    # Check each tier for this model
    for tier_name, models_in_tier in tiers_data.get("tiers", {}).items():
        # Handle both list and dict structures
        if isinstance(models_in_tier, list):
            if model_id in models_in_tier:
                return tier_name
        elif isinstance(models_in_tier, dict):
            models_list = models_in_tier.get("models", [])
            if model_id in models_list:
                return tier_name

    # Heuristic fallback based on model name
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["mini", "flash", "haiku", "lite", "small"]):
        return "budget"
    elif any(x in model_lower for x in ["o1", "o3", "reasoning", "think", "deepthink"]):
        return "reasoning"
    elif any(x in model_lower for x in ["opus", "pro", "large", "gpt-5", "claude-4"]):
        return "frontier"
    else:
        return "general"


def calculate_weekly_spend(rankings: Dict, prices: Dict, tiers_data: Dict) -> Dict[str, Any]:
    """
    Calculate total weekly spend by multiplying token volumes by prices.

    Returns detailed breakdown by model and aggregated totals.
    """
    model_spend = {}
    total_spend = 0.0
    models_matched = 0
    models_estimated = 0

    rankings_models = rankings.get("models", {})
    price_models = prices.get("models", {})

    for model_id, volume_data in rankings_models.items():
        tokens_weekly = volume_data.get("tokens_weekly", 0)

        # Convert to millions of tokens for pricing
        tokens_millions = tokens_weekly / 1_000_000

        # Look up pricing
        if model_id in price_models:
            price_data = price_models[model_id]
            input_price = price_data.get("input_mtok", 0)
            output_price = price_data.get("output_mtok", 0)

            # Calculate blended price using input/output ratio
            blended_price = (
                INPUT_OUTPUT_RATIO["input_weight"] * input_price +
                INPUT_OUTPUT_RATIO["output_weight"] * output_price
            )

            spend = tokens_millions * blended_price
            models_matched += 1
        else:
            # Estimate using tier average
            tier = get_model_tier(model_id, tiers_data)
            estimated_price = estimate_tier_price(tier, price_models, tiers_data)
            spend = tokens_millions * estimated_price
            models_estimated += 1
            blended_price = estimated_price

        model_spend[model_id] = {
            "tokens_weekly": tokens_weekly,
            "tokens_millions": round(tokens_millions, 2),
            "blended_price_per_mtok": round(blended_price, 3),
            "spend_usd": round(spend, 2),
            "tier": get_model_tier(model_id, tiers_data)
        }

        total_spend += spend

    return {
        "total_spend_usd": round(total_spend, 2),
        "models_matched": models_matched,
        "models_estimated": models_estimated,
        "model_breakdown": model_spend
    }


def estimate_tier_price(tier: str, price_models: Dict, tiers_data: Dict) -> float:
    """Estimate average price for a tier when specific model price is unavailable."""
    tier_prices = []

    # Get models in this tier
    tier_data = tiers_data.get("tiers", {}).get(tier, [])

    # Handle both list and dict structures
    if isinstance(tier_data, list):
        tier_model_ids = tier_data
    elif isinstance(tier_data, dict):
        tier_model_ids = tier_data.get("models", [])
    else:
        tier_model_ids = []

    # Calculate average price for this tier
    for model_id in tier_model_ids:
        if model_id in price_models:
            price_data = price_models[model_id]
            input_price = price_data.get("input_mtok", 0)
            output_price = price_data.get("output_mtok", 0)
            blended = (
                INPUT_OUTPUT_RATIO["input_weight"] * input_price +
                INPUT_OUTPUT_RATIO["output_weight"] * output_price
            )
            tier_prices.append(blended)

    if tier_prices:
        return sum(tier_prices) / len(tier_prices)

    # Fallback defaults by tier
    defaults = {
        "budget": 0.5,
        "general": 4.0,
        "frontier": 10.0,
        "reasoning": 8.0,
        "longctx": 3.0
    }
    return defaults.get(tier, 4.0)


def calculate_energy_proxy(rankings: Dict, tiers_data: Dict) -> Dict[str, Any]:
    """
    Estimate energy consumption based on token volumes and model tiers.

    Returns energy estimates in Joules and GWh.
    """
    total_energy_joules = 0.0
    model_energy = {}

    for model_id, volume_data in rankings.get("models", {}).items():
        tokens_weekly = volume_data.get("tokens_weekly", 0)
        tokens_billions = tokens_weekly / 1_000_000_000

        # Get tier and corresponding energy factor
        tier = get_model_tier(model_id, tiers_data)
        energy_factor = ENERGY_PER_BILLION_TOKENS.get(tier, ENERGY_PER_BILLION_TOKENS["default"])

        # Calculate energy for this model
        energy_joules = tokens_billions * energy_factor

        model_energy[model_id] = {
            "tokens_billions": round(tokens_billions, 3),
            "tier": tier,
            "energy_joules": round(energy_joules, 2),
            "energy_kwh": round(energy_joules / 3_600_000, 2)
        }

        total_energy_joules += energy_joules

    # Convert to more readable units
    total_gwh = total_energy_joules / 3_600_000_000_000

    return {
        "total_energy_joules": round(total_energy_joules, 2),
        "total_energy_gwh": round(total_gwh, 6),
        "model_breakdown": model_energy,
        "methodology": "Estimated from token volumes and tier-based efficiency factors"
    }


def load_aeai_baseline() -> Optional[Dict[str, Any]]:
    """Load the immutable AEAI baseline (February 2025 = 100)."""
    baseline_path = AEAI_DIR / "baseline.json"

    if not baseline_path.exists():
        return None

    with open(baseline_path) as f:
        return json.load(f)


def set_aeai_baseline(tokens: float, spend: float, energy: float) -> Dict[str, Any]:
    """
    Set the immutable baseline for AEAI calculation.

    This should only be called once with February 2025 data.
    """
    baseline_path = AEAI_DIR / "baseline.json"

    if baseline_path.exists():
        print("[AEAI] ERROR: Baseline already exists. Cannot overwrite.")
        return load_aeai_baseline()

    baseline = {
        "baseline_date": "2025-02-01",
        "baseline_set_at": datetime.now(timezone.utc).isoformat(),
        "components": {
            "tokens_weekly": tokens,
            "spend_usd_weekly": spend,
            "energy_gwh_weekly": energy
        },
        "aiu_index": 100.0,
        "note": "This baseline is immutable and represents February 2025 AI economic activity levels."
    }

    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"[AEAI] Baseline set: {baseline_path}")
    return baseline


def calculate_aeai(
    current_tokens: float,
    current_spend: float,
    current_energy: float,
    baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate the AI Economic Activity Index (AIU).

    Each component is normalized to its baseline value (Feb 2025 = 100),
    then weighted: 60% tokens, 30% spend, 10% energy.
    """
    baseline_components = baseline["components"]

    # Calculate component indices (baseline = 100)
    token_index = (current_tokens / baseline_components["tokens_weekly"]) * 100
    spend_index = (current_spend / baseline_components["spend_usd_weekly"]) * 100
    energy_index = (current_energy / baseline_components["energy_gwh_weekly"]) * 100

    # Apply weights: 60/30/10
    aiu_index = (
        0.6 * token_index +
        0.3 * spend_index +
        0.1 * energy_index
    )

    return {
        "aiu_index": round(aiu_index, 2),
        "components": {
            "token_index": round(token_index, 2),
            "spend_index": round(spend_index, 2),
            "energy_index": round(energy_index, 2)
        },
        "weights": {
            "tokens": 0.6,
            "spend": 0.3,
            "energy": 0.1
        },
        "contribution": {
            "tokens": round(0.6 * token_index, 2),
            "spend": round(0.3 * spend_index, 2),
            "energy": round(0.1 * energy_index, 2)
        }
    }


def save_aeai_snapshot(aeai_data: Dict[str, Any]):
    """Save AEAI snapshot with timestamp."""
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y-%m-%d")

    # Save dated snapshot
    snapshot_path = AEAI_DIR / f"aeai_{date_str}.json"
    with open(snapshot_path, "w") as f:
        json.dump(aeai_data, f, indent=2)

    # Save as latest
    latest_path = AEAI_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(aeai_data, f, indent=2)

    print(f"[AEAI] Saved snapshot: {snapshot_path}")


def update_historical(aeai_data: Dict[str, Any]):
    """Append current AEAI data to historical time series."""
    historical_path = AEAI_DIR / "historical.json"

    # Load existing history
    if historical_path.exists():
        with open(historical_path) as f:
            history = json.load(f)
    else:
        history = {"entries": []}

    # Add new entry
    entry = {
        "date": aeai_data["date"],
        "aiu_index": aeai_data["aiu_index"],
        "components": aeai_data["components"],
        "activity": {
            "tokens_weekly": aeai_data["activity"]["tokens_weekly"],
            "spend_usd_weekly": aeai_data["activity"]["spend_usd_weekly"],
            "energy_gwh_weekly": aeai_data["activity"]["energy_gwh_weekly"]
        }
    }

    # Check if entry for this date already exists
    existing_dates = [e["date"] for e in history["entries"]]
    if entry["date"] in existing_dates:
        # Update existing entry
        for i, e in enumerate(history["entries"]):
            if e["date"] == entry["date"]:
                history["entries"][i] = entry
                break
    else:
        # Append new entry
        history["entries"].append(entry)

    # Sort by date
    history["entries"].sort(key=lambda x: x["date"])

    # Save
    with open(historical_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[AEAI] Updated historical series: {len(history['entries'])} entries")


def main():
    """Main AEAI calculation pipeline."""
    print("=" * 70)
    print("  AI ECONOMIC ACTIVITY INDEX (AEAI) CALCULATOR")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    rankings = load_rankings()
    prices = load_prices()
    tiers_data = load_model_tiers()

    model_count = len(rankings.get("models", {}))
    print(f"[1/6] Loaded {model_count} models with volume data")

    # Calculate token volume
    print("\n[2/6] Calculating token volumes...")
    total_tokens = rankings.get("total_tokens", 0)
    print(f"[2/6] Total weekly tokens: {total_tokens:,.0f} ({total_tokens/1e12:.2f}T)")

    # Calculate spend
    print("\n[3/6] Calculating weekly spend...")
    spend_data = calculate_weekly_spend(rankings, prices, tiers_data)
    total_spend = spend_data["total_spend_usd"]
    print(f"[3/6] Total weekly spend: ${total_spend:,.2f}")
    print(f"     Matched models: {spend_data['models_matched']}, Estimated: {spend_data['models_estimated']}")

    # Calculate energy
    print("\n[4/6] Estimating energy consumption...")
    energy_data = calculate_energy_proxy(rankings, tiers_data)
    total_energy = energy_data["total_energy_gwh"]
    print(f"[4/6] Estimated weekly energy: {total_energy:.4f} GWh")

    # Load or set baseline
    print("\n[5/6] Checking baseline...")
    baseline = load_aeai_baseline()

    if baseline is None:
        print("[5/6] No baseline found. Setting current data as baseline (Feb 2025 = 100)")
        baseline = set_aeai_baseline(total_tokens, total_spend, total_energy)
    else:
        print(f"[5/6] Using baseline from {baseline['baseline_date']}")

    # Calculate AEAI
    print("\n[6/6] Calculating AIU index...")
    aeai_result = calculate_aeai(total_tokens, total_spend, total_energy, baseline)
    aiu_index = aeai_result["aiu_index"]
    print(f"[6/6] AIU Index: {aiu_index:.2f}")

    # Compile full result
    timestamp = datetime.now(timezone.utc)

    aeai_data = {
        "date": timestamp.strftime("%Y-%m-%d"),
        "generated_at": timestamp.isoformat(),
        "aiu_index": aiu_index,
        "components": aeai_result["components"],
        "weights": aeai_result["weights"],
        "contribution": aeai_result["contribution"],
        "activity": {
            "tokens_weekly": total_tokens,
            "spend_usd_weekly": total_spend,
            "energy_gwh_weekly": total_energy
        },
        "data_quality": {
            "total_models": model_count,
            "price_matched": spend_data["models_matched"],
            "price_estimated": spend_data["models_estimated"],
            "freshness": rankings.get("fetched_at", "unknown")
        },
        "methodology": {
            "basket": "60% token volumes, 30% inferred spend, 10% energy proxy",
            "baseline": "February 2025 = 100",
            "spend_calculation": "tokens × blended_price (70% input, 30% output)",
            "energy_estimation": "Token volumes × tier-based efficiency factors"
        }
    }

    # Save
    save_aeai_snapshot(aeai_data)
    update_historical(aeai_data)

    # Display summary
    print("\n" + "=" * 70)
    print("  AEAI SUMMARY")
    print("=" * 70)
    print(f"\nAIU Index: {aiu_index:.2f} (Baseline Feb 2025 = 100)")
    print(f"\nComponent Indices:")
    print(f"  Token Volume:  {aeai_result['components']['token_index']:6.2f} (60% weight → {aeai_result['contribution']['tokens']:5.2f})")
    print(f"  Spend:         {aeai_result['components']['spend_index']:6.2f} (30% weight → {aeai_result['contribution']['spend']:5.2f})")
    print(f"  Energy:        {aeai_result['components']['energy_index']:6.2f} (10% weight → {aeai_result['contribution']['energy']:5.2f})")

    print(f"\nActivity Levels:")
    print(f"  Tokens/week:   {total_tokens/1e12:6.2f}T")
    print(f"  Spend/week:    ${total_spend:,.0f}")
    print(f"  Energy/week:   {total_energy:.4f} GWh")

    print(f"\nData Quality:")
    print(f"  Models tracked: {model_count}")
    print(f"  Price coverage: {spend_data['models_matched']}/{model_count} ({100*spend_data['models_matched']/model_count:.1f}% exact)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
