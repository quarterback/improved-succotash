#!/usr/bin/env python3
"""
Historical Backfill Script for Compute CPI.

This script runs ONCE to populate data/historical.json with reconstructed
snapshots from known historical prices. This enables MoM and YoY calculations
to work immediately on launch.

Historical prices sourced from:
- pydantic/genai-prices
- simonw/llm-prices
- Provider announcements
"""

import json
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORICAL_PATH = DATA_DIR / "historical.json"

# =============================================================================
# HISTORICAL PRICE DATA
# =============================================================================
# Prices in $/MTok (million tokens) for input and output

HISTORICAL_PRICES = {
    # February 2025 (approximate)
    "2025-02-01": {
        "openai/gpt-4o": {"input": 5.0, "output": 15.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
        "google/gemini-1.5-flash": {"input": 0.075, "output": 0.3},
        "google/gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "openai/o1-mini": {"input": 3.0, "output": 12.0},
        "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    },
    # May 2025 (approximate - after some price cuts)
    "2025-05-01": {
        "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
        "google/gemini-2.0-flash": {"input": 0.1, "output": 0.4},
        "google/gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "openai/o1-mini": {"input": 1.1, "output": 4.4},
        "deepseek/deepseek-chat": {"input": 0.27, "output": 1.1},
    },
    # August 2025 (approximate)
    "2025-08-01": {
        "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
        "google/gemini-2.0-flash": {"input": 0.1, "output": 0.4},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 10.0},
        "openai/o3-mini": {"input": 1.1, "output": 4.4},
        "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    },
    # November 2025 (approximate)
    "2025-11-01": {
        "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-3.7-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
        "google/gemini-2.0-flash": {"input": 0.1, "output": 0.4},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 10.0},
        "openai/o3-mini": {"input": 1.1, "output": 4.4},
        "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    },
    # January 2026 (recent)
    "2026-01-01": {
        "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-3.7-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku": {"input": 0.8, "output": 4.0},
        "google/gemini-2.0-flash-001": {"input": 0.1, "output": 0.4},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 10.0},
        "openai/o3-mini": {"input": 1.1, "output": 4.4},
        "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    }
}

# =============================================================================
# TIER MAPPINGS FOR HISTORICAL DATA
# =============================================================================
# Map model availability to tiers at different points in time

HISTORICAL_TIERS = {
    "2025-02-01": {
        "budget": ["openai/gpt-4o-mini", "google/gemini-1.5-flash", "deepseek/deepseek-chat"],
        "general": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-flash"],
        "frontier": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-pro"],
        "reasoning": ["openai/o1-mini", "anthropic/claude-3.5-sonnet"],
        "longctx": ["google/gemini-1.5-pro", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
    },
    "2025-05-01": {
        "budget": ["openai/gpt-4o-mini", "google/gemini-2.0-flash", "deepseek/deepseek-chat"],
        "general": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"],
        "frontier": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-pro"],
        "reasoning": ["openai/o1-mini", "anthropic/claude-3.5-sonnet"],
        "longctx": ["google/gemini-1.5-pro", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
    },
    "2025-08-01": {
        "budget": ["openai/gpt-4o-mini", "google/gemini-2.0-flash", "anthropic/claude-3.5-haiku"],
        "general": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"],
        "frontier": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-pro"],
        "reasoning": ["openai/o3-mini", "deepseek/deepseek-r1", "anthropic/claude-3.5-sonnet"],
        "longctx": ["google/gemini-2.5-pro", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
    },
    "2025-11-01": {
        "budget": ["openai/gpt-4o-mini", "google/gemini-2.0-flash", "anthropic/claude-3.5-haiku"],
        "general": ["openai/gpt-4o", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash"],
        "frontier": ["openai/gpt-4o", "anthropic/claude-3.7-sonnet", "google/gemini-2.5-pro"],
        "reasoning": ["openai/o3-mini", "deepseek/deepseek-r1", "anthropic/claude-3.7-sonnet"],
        "longctx": ["google/gemini-2.5-pro", "anthropic/claude-3.7-sonnet", "openai/gpt-4o"],
    },
    "2026-01-01": {
        "budget": ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001", "anthropic/claude-3.5-haiku"],
        "general": ["openai/gpt-4o", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-001"],
        "frontier": ["openai/gpt-4o", "anthropic/claude-3.7-sonnet", "google/gemini-2.5-pro"],
        "reasoning": ["openai/o3-mini", "deepseek/deepseek-r1", "anthropic/claude-3.7-sonnet"],
        "longctx": ["google/gemini-2.5-pro", "anthropic/claude-3.7-sonnet", "openai/gpt-4o"],
    },
}

# =============================================================================
# WORKLOAD DEFINITIONS (must match calculate_cpi.py)
# =============================================================================

WORKLOAD_BASKET = {
    "chat_drafting": {"input_tokens": 2000, "output_tokens": 500, "weight": 0.20, "tier": "general"},
    "summarization": {"input_tokens": 10000, "output_tokens": 500, "weight": 0.25, "tier": "general"},
    "classification": {"input_tokens": 500, "output_tokens": 50, "weight": 0.20, "tier": "budget"},
    "coding": {"input_tokens": 3000, "output_tokens": 1000, "weight": 0.15, "tier": "frontier"},
    "judgment": {"input_tokens": 5000, "output_tokens": 2000, "weight": 0.10, "tier": "reasoning"},
    "long_context": {"input_tokens": 50000, "output_tokens": 1000, "weight": 0.10, "tier": "longctx"},
}

PERSONA_BASKETS = {
    "startup": {
        "workloads": {
            "coding": {"tier": "frontier", "input_tokens": 3000, "output_tokens": 1000, "weight": 0.50},
            "rag_context": {"tier": "longctx", "input_tokens": 20000, "output_tokens": 500, "weight": 0.30},
            "routing": {"tier": "budget", "input_tokens": 500, "output_tokens": 50, "weight": 0.20},
        }
    },
    "agentic": {
        "workloads": {
            "thinking": {"tier": "reasoning", "input_tokens": 5000, "output_tokens": 3000, "weight": 0.70},
            "tool_use": {"tier": "general", "input_tokens": 1000, "output_tokens": 500, "weight": 0.20},
            "final_output": {"tier": "frontier", "input_tokens": 500, "output_tokens": 1000, "weight": 0.10},
        }
    },
    "throughput": {
        "workloads": {
            "extraction": {"tier": "general", "input_tokens": 10000, "output_tokens": 500, "weight": 0.80},
            "classification": {"tier": "budget", "input_tokens": 500, "output_tokens": 50, "weight": 0.20},
        }
    },
}

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def get_model_cost(prices: dict, model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a workload given historical prices."""
    if model_id not in prices:
        return None

    pricing = prices[model_id]
    # Convert from $/MTok to $/token
    input_cost_per_token = pricing["input"] / 1_000_000
    output_cost_per_token = pricing["output"] / 1_000_000

    return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)


def calculate_tier_average(prices: dict, tier_models: list, input_tokens: int, output_tokens: int) -> float:
    """Calculate average cost across models in a tier."""
    costs = []
    for model_id in tier_models:
        cost = get_model_cost(prices, model_id, input_tokens, output_tokens)
        if cost is not None:
            costs.append(cost)

    if not costs:
        return None
    return sum(costs) / len(costs)


def calculate_basket_cost(prices: dict, tiers: dict) -> dict:
    """Calculate costs for the main workload basket."""
    costs = {}

    for workload_name, workload in WORKLOAD_BASKET.items():
        tier = workload["tier"]
        tier_models = tiers.get(tier, [])

        avg_cost = calculate_tier_average(
            prices, tier_models,
            workload["input_tokens"], workload["output_tokens"]
        )

        if avg_cost is not None:
            costs[workload_name] = {
                "cost": avg_cost,
                "weight": workload["weight"]
            }

    return costs


def calculate_weighted_total(basket_costs: dict) -> float:
    """Calculate weighted total basket cost."""
    total = 0.0
    for workload, data in basket_costs.items():
        total += data["cost"] * data["weight"]
    return total


def calculate_subindex_costs(prices: dict, tiers: dict) -> dict:
    """Calculate costs for each subindex."""
    costs = {}

    # $JUDGE - Reasoning tier
    reasoning_cost = calculate_tier_average(prices, tiers.get("reasoning", []), 5000, 2000)
    if reasoning_cost:
        costs["judgment"] = reasoning_cost

    # $LCTX - Long context tier
    longctx_cost = calculate_tier_average(prices, tiers.get("longctx", []), 50000, 1000)
    if longctx_cost:
        costs["longctx"] = longctx_cost

    # $BULK - Budget tier
    budget_cost = calculate_tier_average(prices, tiers.get("budget", []), 2000, 500)
    if budget_cost:
        costs["budget"] = budget_cost

    # $FRONT - Frontier tier
    frontier_cost = calculate_tier_average(prices, tiers.get("frontier", []), 2000, 500)
    if frontier_cost:
        costs["frontier"] = frontier_cost

    return costs


def calculate_persona_costs(prices: dict, tiers: dict) -> dict:
    """Calculate costs for persona baskets."""
    persona_costs = {}

    for persona_name, persona in PERSONA_BASKETS.items():
        total = 0.0
        for workload_name, workload in persona["workloads"].items():
            tier = workload["tier"]
            tier_models = tiers.get(tier, [])

            avg_cost = calculate_tier_average(
                prices, tier_models,
                workload["input_tokens"], workload["output_tokens"]
            )

            if avg_cost is not None:
                total += avg_cost * workload["weight"]

        persona_costs[persona_name] = total

    return persona_costs


def calculate_spreads(prices: dict, tiers: dict) -> dict:
    """Calculate spreads between tiers."""
    spreads = {}

    std_input, std_output = 2000, 500
    total_tokens = std_input + std_output

    budget_cost = calculate_tier_average(prices, tiers.get("budget", []), std_input, std_output)
    frontier_cost = calculate_tier_average(prices, tiers.get("frontier", []), std_input, std_output)
    reasoning_cost = calculate_tier_average(prices, tiers.get("reasoning", []), 5000, 2000)
    longctx_cost = calculate_tier_average(prices, tiers.get("longctx", []), 50000, 1000)

    def to_per_mtok(cost, tokens):
        if cost is None:
            return None
        return (cost / tokens) * 1_000_000

    budget_mtok = to_per_mtok(budget_cost, total_tokens)
    frontier_mtok = to_per_mtok(frontier_cost, total_tokens)
    reasoning_mtok = to_per_mtok(reasoning_cost, 7000)
    longctx_mtok = to_per_mtok(longctx_cost, 51000)

    if frontier_mtok and budget_mtok:
        spreads["cognition_premium"] = round(frontier_mtok - budget_mtok, 2)

    if reasoning_mtok and frontier_mtok:
        spreads["judgment_premium"] = round(reasoning_mtok - frontier_mtok, 2)

    if longctx_mtok and frontier_mtok:
        spreads["context_premium"] = round(longctx_mtok - frontier_mtok, 2)

    return spreads


def calculate_cpi(current: float, baseline: float) -> float:
    """Calculate CPI index value."""
    if baseline == 0:
        return 100.0
    return round((current / baseline) * 100, 1)


# =============================================================================
# MAIN BACKFILL FUNCTION
# =============================================================================

def backfill_historical(baseline_date: str = "2025-02-01", force: bool = False):
    """
    Backfill historical.json with reconstructed snapshots.

    Args:
        baseline_date: Date to use as baseline (CPI = 100)
        force: If True, overwrite existing historical data
    """
    if HISTORICAL_PATH.exists() and not force:
        with open(HISTORICAL_PATH) as f:
            existing = json.load(f)
        if len(existing) > 0:
            print(f"[Backfill] Historical data already exists ({len(existing)} snapshots). Use force=True to overwrite.")
            return existing

    print(f"[Backfill] Starting historical backfill with baseline {baseline_date}")

    # Use February 2025 as the baseline (when prices were highest)
    # This way, current prices will show deflation relative to the launch
    baseline_prices = HISTORICAL_PRICES.get("2025-02-01")
    baseline_tiers = HISTORICAL_TIERS.get("2025-02-01")

    baseline_basket = calculate_basket_cost(baseline_prices, baseline_tiers)
    baseline_total = calculate_weighted_total(baseline_basket)
    baseline_subindices = calculate_subindex_costs(baseline_prices, baseline_tiers)
    baseline_personas = calculate_persona_costs(baseline_prices, baseline_tiers)

    print(f"[Backfill] Baseline total: ${baseline_total:.6f}")

    # Now calculate historical snapshots
    snapshots = []

    for date in sorted(HISTORICAL_PRICES.keys()):
        prices = HISTORICAL_PRICES[date]
        tiers = HISTORICAL_TIERS.get(date, baseline_tiers)

        # Calculate basket costs
        basket_costs = calculate_basket_cost(prices, tiers)
        weighted_total = calculate_weighted_total(basket_costs)

        # Calculate CPI
        cpi_value = calculate_cpi(weighted_total, baseline_total)

        # Calculate subindices
        subindex_costs = calculate_subindex_costs(prices, tiers)
        subindices = {}
        for key, cost in subindex_costs.items():
            baseline_val = baseline_subindices.get(key, cost)
            subindices[key] = calculate_cpi(cost, baseline_val)

        # Calculate persona CPIs
        persona_costs = calculate_persona_costs(prices, tiers)
        personas = {}
        for key, cost in persona_costs.items():
            baseline_val = baseline_personas.get(key, cost)
            personas[key] = calculate_cpi(cost, baseline_val)

        # Calculate spreads
        spreads = calculate_spreads(prices, tiers)

        snapshot = {
            "date": date,
            "cpi": cpi_value,
            "basket_cost": weighted_total,
            "subindices": subindices,
            "spreads": spreads,
            "personas": personas,
            "persona_costs": persona_costs,
            "reconstructed": True,
            "source": "backfill"
        }

        snapshots.append(snapshot)
        print(f"[Backfill] {date}: CPI = {cpi_value:.1f}")

    # Sort by date
    snapshots.sort(key=lambda x: x["date"])

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORICAL_PATH, "w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"[Backfill] Saved {len(snapshots)} historical snapshots to {HISTORICAL_PATH}")

    # Also update baseline.json with persona costs
    baseline_path = DATA_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        baseline_data["persona_costs"] = baseline_personas
        baseline_data["backfill_source"] = "historical reconstruction"

        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"[Backfill] Updated baseline.json with persona costs")

    return snapshots


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    backfill_historical(force=force)
