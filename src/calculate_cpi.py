#!/usr/bin/env python3
"""
Compute CPI - The inflation rate for AI work.

Main calculation script that:
1. Fetches pricing from multiple sources (OpenRouter, LiteLLM)
2. Calculates basket costs for each workload category
3. Computes CPI index and subindices
4. Calculates spreads between tiers
5. Tracks historical data and computes MoM/YoY changes
6. Outputs JSON for the dashboard
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_registry import build_unified_registry, calculate_tier_average, calculate_exchange_rates
from historical import (
    load_baseline, save_baseline, save_snapshot, load_historical,
    get_mom_change, get_yoy_change, get_trend,
    get_persona_mom_change, get_days_since_baseline,
    get_closest_snapshot, calculate_change
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "compute-cpi.json"

# Workload basket definition
WORKLOAD_BASKET = {
    "chat_drafting": {
        "input_tokens": 2000,
        "output_tokens": 500,
        "weight": 0.20,
        "tier": "general",
        "description": "Basic chat interactions and text drafting"
    },
    "summarization": {
        "input_tokens": 10000,
        "output_tokens": 500,
        "weight": 0.25,
        "tier": "general",
        "description": "Document and text summarization"
    },
    "classification": {
        "input_tokens": 500,
        "output_tokens": 50,
        "weight": 0.20,
        "tier": "budget",
        "description": "Text classification and labeling"
    },
    "coding": {
        "input_tokens": 3000,
        "output_tokens": 1000,
        "weight": 0.15,
        "tier": "frontier",
        "description": "Code generation and assistance"
    },
    "judgment": {
        "input_tokens": 5000,
        "output_tokens": 2000,
        "weight": 0.10,
        "tier": "reasoning",
        "description": "Complex reasoning and decision tasks"
    },
    "long_context": {
        "input_tokens": 50000,
        "output_tokens": 1000,
        "weight": 0.10,
        "tier": "longctx",
        "description": "Large document processing"
    }
}

# Persona basket definitions - different CPI weightings for different build patterns
PERSONA_BASKETS = {
    "startup": {
        "name": "Startup Builder",
        "ticker": "$START",
        "description": "Building an AI-first product",
        "workloads": {
            "coding": {
                "tier": "frontier",
                "input_tokens": 3000,
                "output_tokens": 1000,
                "weight": 0.50
            },
            "rag_context": {
                "tier": "longctx",
                "input_tokens": 20000,
                "output_tokens": 500,
                "weight": 0.30
            },
            "routing": {
                "tier": "budget",
                "input_tokens": 500,
                "output_tokens": 50,
                "weight": 0.20
            }
        }
    },
    "agentic": {
        "name": "Agentic Team",
        "ticker": "$AGENT",
        "description": "Running autonomous AI agents",
        "workloads": {
            "thinking": {
                "tier": "reasoning",
                "input_tokens": 5000,
                "output_tokens": 3000,
                "weight": 0.70
            },
            "tool_use": {
                "tier": "general",
                "input_tokens": 1000,
                "output_tokens": 500,
                "weight": 0.20
            },
            "final_output": {
                "tier": "frontier",
                "input_tokens": 500,
                "output_tokens": 1000,
                "weight": 0.10
            }
        }
    },
    "throughput": {
        "name": "Throughput",
        "ticker": "$THRU",
        "description": "High-volume processing at scale",
        "workloads": {
            "extraction": {
                "tier": "general",
                "input_tokens": 10000,
                "output_tokens": 500,
                "weight": 0.80
            },
            "classification": {
                "tier": "budget",
                "input_tokens": 500,
                "output_tokens": 50,
                "weight": 0.20
            }
        }
    }
}

# Launch baseline date for multi-series calculations
LAUNCH_BASELINE_DATE = "2025-02-01"

# Methodology variant weights - different basket compositions
METHODOLOGY_WEIGHTS = {
    "general": {
        "name": "General Purpose",
        "description": "Balanced workload mix for typical usage",
        "weights": {
            "chat_drafting": 0.20,
            "summarization": 0.25,
            "classification": 0.20,
            "coding": 0.15,
            "judgment": 0.10,
            "long_context": 0.10
        }
    },
    "frontier_heavy": {
        "name": "Frontier Heavy",
        "description": "Emphasis on highest-capability models",
        "weights": {
            "chat_drafting": 0.10,
            "summarization": 0.10,
            "classification": 0.05,
            "coding": 0.35,
            "judgment": 0.25,
            "long_context": 0.15
        }
    },
    "budget_heavy": {
        "name": "Budget Optimized",
        "description": "Cost-conscious workload mix",
        "weights": {
            "chat_drafting": 0.30,
            "summarization": 0.25,
            "classification": 0.30,
            "coding": 0.05,
            "judgment": 0.05,
            "long_context": 0.05
        }
    },
    "reasoning_focus": {
        "name": "Reasoning Focus",
        "description": "Heavy emphasis on complex reasoning tasks",
        "weights": {
            "chat_drafting": 0.05,
            "summarization": 0.10,
            "classification": 0.05,
            "coding": 0.20,
            "judgment": 0.45,
            "long_context": 0.15
        }
    },
    "enterprise": {
        "name": "Enterprise Mix",
        "description": "Typical enterprise automation workloads",
        "weights": {
            "chat_drafting": 0.25,
            "summarization": 0.30,
            "classification": 0.25,
            "coding": 0.10,
            "judgment": 0.05,
            "long_context": 0.05
        }
    }
}


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_basket_costs(registry: dict) -> dict:
    """Calculate cost for each workload in the basket."""
    costs = {}

    for workload_name, workload in WORKLOAD_BASKET.items():
        tier = workload["tier"]
        input_tokens = workload["input_tokens"]
        output_tokens = workload["output_tokens"]

        avg_cost = calculate_tier_average(registry, tier, input_tokens, output_tokens)

        if avg_cost is not None:
            costs[workload_name] = {
                "cost": avg_cost,
                "weight": workload["weight"],
                "tier": tier,
                "tokens": {"input": input_tokens, "output": output_tokens}
            }
        else:
            print(f"[Warning] No pricing for tier '{tier}' (workload: {workload_name})")

    return costs


def calculate_weighted_total(basket_costs: dict) -> float:
    """Calculate weighted total basket cost."""
    total = 0.0
    total_weight = 0.0

    for workload, data in basket_costs.items():
        total += data["cost"] * data["weight"]
        total_weight += data["weight"]

    # Normalize if weights don't sum to 1
    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
        total = total / total_weight

    return total


def calculate_cpi(current_cost: float, baseline_cost: float) -> float:
    """Calculate CPI index value (base = 100)."""
    if baseline_cost == 0:
        return 100.0
    return round((current_cost / baseline_cost) * 100, 1)


def calculate_subindex_costs(registry: dict) -> dict:
    """Calculate current costs for each subindex."""
    costs = {}

    # $JUDGE - Reasoning tier (uses judgment workload: 5K in, 2K out)
    reasoning_cost = calculate_tier_average(registry, "reasoning", 5000, 2000)
    if reasoning_cost:
        costs["judgment"] = reasoning_cost

    # $LCTX - Long context tier (uses long_context workload: 50K in, 1K out)
    longctx_cost = calculate_tier_average(registry, "longctx", 50000, 1000)
    if longctx_cost:
        costs["longctx"] = longctx_cost

    # $BULK - Budget tier (standard workload: 2K in, 500 out)
    budget_cost = calculate_tier_average(registry, "budget", 2000, 500)
    if budget_cost:
        costs["budget"] = budget_cost

    # $FRONT - Frontier tier (standard workload: 2K in, 500 out)
    frontier_cost = calculate_tier_average(registry, "frontier", 2000, 500)
    if frontier_cost:
        costs["frontier"] = frontier_cost

    return costs


def calculate_subindices(registry: dict, baseline: dict, current_costs: dict) -> dict:
    """Calculate subindices for different tier segments."""
    subindices = {}
    baseline_subindex = baseline.get("subindex_costs", {})

    # $JUDGE - Reasoning tier
    if "judgment" in current_costs:
        baseline_val = baseline_subindex.get("judgment", current_costs["judgment"])
        subindices["judgment"] = {
            "ticker": "$JUDGE",
            "name": "Judgment CPI",
            "value": calculate_cpi(current_costs["judgment"], baseline_val),
            "cost": current_costs["judgment"],
            "models_count": len(registry.get("tiers", {}).get("reasoning", []))
        }

    # $LCTX - Long context tier
    if "longctx" in current_costs:
        baseline_val = baseline_subindex.get("longctx", current_costs["longctx"])
        subindices["longctx"] = {
            "ticker": "$LCTX",
            "name": "LongContext CPI",
            "value": calculate_cpi(current_costs["longctx"], baseline_val),
            "cost": current_costs["longctx"],
            "models_count": len(registry.get("tiers", {}).get("longctx", []))
        }

    # $BULK - Budget tier
    if "budget" in current_costs:
        baseline_val = baseline_subindex.get("budget", current_costs["budget"])
        subindices["budget"] = {
            "ticker": "$BULK",
            "name": "Budget Tier",
            "value": calculate_cpi(current_costs["budget"], baseline_val),
            "cost": current_costs["budget"],
            "models_count": len(registry.get("tiers", {}).get("budget", []))
        }

    # $FRONT - Frontier tier
    if "frontier" in current_costs:
        baseline_val = baseline_subindex.get("frontier", current_costs["frontier"])
        subindices["frontier"] = {
            "ticker": "$FRONT",
            "name": "Frontier Tier",
            "value": calculate_cpi(current_costs["frontier"], baseline_val),
            "cost": current_costs["frontier"],
            "models_count": len(registry.get("tiers", {}).get("frontier", []))
        }

    return subindices


def calculate_spreads(registry: dict) -> dict:
    """Calculate spreads between tiers (normalized to $/MTok)."""
    spreads = {}

    # Standard workload for normalization
    std_input, std_output = 2000, 500
    total_tokens = std_input + std_output  # For normalization to per-MTok

    budget_cost = calculate_tier_average(registry, "budget", std_input, std_output)
    frontier_cost = calculate_tier_average(registry, "frontier", std_input, std_output)
    reasoning_cost = calculate_tier_average(registry, "reasoning", 5000, 2000)
    longctx_cost = calculate_tier_average(registry, "longctx", 50000, 1000)

    # Normalize to $/MTok (million tokens)
    def to_per_mtok(cost, tokens):
        if cost is None:
            return None
        return (cost / tokens) * 1_000_000

    budget_mtok = to_per_mtok(budget_cost, total_tokens) if budget_cost else None
    frontier_mtok = to_per_mtok(frontier_cost, total_tokens) if frontier_cost else None
    reasoning_mtok = to_per_mtok(reasoning_cost, 7000) if reasoning_cost else None
    longctx_mtok = to_per_mtok(longctx_cost, 51000) if longctx_cost else None

    # $COG-P - Cognition Premium (Frontier vs Budget)
    if frontier_mtok and budget_mtok:
        spreads["cognition_premium"] = {
            "ticker": "$COG-P",
            "name": "Cognition Premium",
            "value": round(frontier_mtok - budget_mtok, 2),
            "unit": "$/1M tokens",
            "description": "Premium for frontier capability over budget tier"
        }

    # $JDG-P - Judgment Premium (Reasoning vs Frontier)
    if reasoning_mtok and frontier_mtok:
        spreads["judgment_premium"] = {
            "ticker": "$JDG-P",
            "name": "Judgment Premium",
            "value": round(reasoning_mtok - frontier_mtok, 2),
            "unit": "$/1M tokens",
            "description": "Premium for reasoning capability over frontier"
        }

    # $CTX-P - Context Premium (LongCtx vs Frontier)
    if longctx_mtok and frontier_mtok:
        spreads["context_premium"] = {
            "ticker": "$CTX-P",
            "name": "Context Premium",
            "value": round(longctx_mtok - frontier_mtok, 2),
            "unit": "$/1M tokens",
            "description": "Premium for long context capability"
        }

    return spreads


def calculate_persona_costs(registry: dict) -> dict:
    """Calculate current costs for each persona basket."""
    persona_costs = {}

    for persona_name, persona in PERSONA_BASKETS.items():
        total = 0.0

        for workload_name, workload in persona["workloads"].items():
            tier = workload["tier"]
            avg_cost = calculate_tier_average(
                registry, tier,
                workload["input_tokens"],
                workload["output_tokens"]
            )

            if avg_cost is not None:
                total += avg_cost * workload["weight"]

        persona_costs[persona_name] = total

    return persona_costs


def generate_insight(cpi_value: float, persona_name: str) -> str:
    """Generate human-readable insight based on CPI value."""
    diff = cpi_value - 100

    if abs(diff) < 1:
        return "Costs unchanged from baseline"

    direction = "cheaper" if diff < 0 else "more expensive"
    pct = abs(round(diff))

    if persona_name == "startup":
        action = "to build" if diff < 0 else "to build"
    elif persona_name == "agentic":
        action = "to run agents" if diff < 0 else "to run agents"
    elif persona_name == "throughput":
        action = "at scale" if diff < 0 else "at scale"
    else:
        action = ""

    return f"{pct}% {direction} {action}".strip()


def calculate_persona_cpis(registry: dict, baseline: dict, current_costs: dict) -> dict:
    """
    Calculate CPI values for each persona basket.

    Returns dict with persona CPI data including MoM changes and insights.
    """
    persona_cpis = {}
    baseline_personas = baseline.get("persona_costs", {})
    days_since_baseline = get_days_since_baseline()

    for persona_name, persona in PERSONA_BASKETS.items():
        current_cost = current_costs.get(persona_name, 0)
        baseline_cost = baseline_personas.get(persona_name, current_cost)

        cpi_value = calculate_cpi(current_cost, baseline_cost)

        # Get MoM change (return null if < 30 days of data)
        mom_change = None
        note = None

        if days_since_baseline < 30:
            note = "Benchmarking period - insufficient historical data"
        else:
            mom_change = get_persona_mom_change(persona_name, cpi_value)

        persona_cpis[persona_name] = {
            "ticker": persona["ticker"],
            "name": persona["name"],
            "description": persona["description"],
            "cpi": cpi_value,
            "mom_change": mom_change,
            "basket_cost": round(current_cost, 6),
            "basket_cost_unit": "$ per standardized workload",
            "insight": generate_insight(cpi_value, persona_name),
            "note": note
        }

    return persona_cpis


# =============================================================================
# MULTI-SERIES INDEX FUNCTIONS
# =============================================================================

def calculate_index_series(current_cost: float, baseline: dict) -> dict:
    """
    Calculate CPI against multiple base periods.

    Returns index values for:
    - launch: Since product launch (LAUNCH_BASELINE_DATE)
    - yoy: Year-over-year (365 days ago)
    - qtd: Quarter-to-date (start of current quarter)
    - mtd: Month-to-date (start of current month)
    - wow: Week-over-week (7 days ago)
    """
    historical = load_historical()
    now = datetime.now(timezone.utc)

    def get_historical_cost(target_date_str: str) -> float:
        """Get basket cost for a specific date from historical data."""
        for snapshot in historical:
            if snapshot["date"] == target_date_str:
                return snapshot.get("basket_cost")
        return None

    def get_closest_historical_cost(days_ago: int, tolerance: int = 7) -> float:
        """Get closest historical cost within tolerance window."""
        snapshot = get_closest_snapshot(days_ago, tolerance)
        if snapshot:
            return snapshot.get("basket_cost")
        return None

    series = {}

    # Launch index - vs LAUNCH_BASELINE_DATE
    launch_cost = get_historical_cost(LAUNCH_BASELINE_DATE)
    if launch_cost:
        series["launch"] = {
            "ticker": "$CPI-L",
            "name": "Since Launch",
            "base_date": LAUNCH_BASELINE_DATE,
            "value": round((current_cost / launch_cost) * 100, 1)
        }

    # YoY - vs 365 days ago
    yoy_cost = get_closest_historical_cost(365, tolerance=30)
    if yoy_cost:
        series["yoy"] = {
            "ticker": "$CPI-Y",
            "name": "Year-over-Year",
            "base_date": (now - timedelta(days=365)).strftime("%Y-%m-%d"),
            "value": round((current_cost / yoy_cost) * 100, 1)
        }

    # QTD - vs start of current quarter
    quarter_start_month = ((now.month - 1) // 3) * 3 + 1
    quarter_start = now.replace(month=quarter_start_month, day=1).strftime("%Y-%m-%d")
    qtd_cost = get_historical_cost(quarter_start)
    if not qtd_cost:
        # Try to find closest snapshot near quarter start
        days_since_quarter_start = (now - datetime.strptime(quarter_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days
        qtd_cost = get_closest_historical_cost(days_since_quarter_start, tolerance=15)
    if qtd_cost:
        series["qtd"] = {
            "ticker": "$CPI-Q",
            "name": "Quarter-to-Date",
            "base_date": quarter_start,
            "value": round((current_cost / qtd_cost) * 100, 1)
        }

    # MTD - vs start of current month
    month_start = now.replace(day=1).strftime("%Y-%m-%d")
    mtd_cost = get_historical_cost(month_start)
    if not mtd_cost:
        days_since_month_start = now.day - 1
        mtd_cost = get_closest_historical_cost(days_since_month_start, tolerance=5)
    if mtd_cost:
        series["mtd"] = {
            "ticker": "$CPI-M",
            "name": "Month-to-Date",
            "base_date": month_start,
            "value": round((current_cost / mtd_cost) * 100, 1)
        }

    # WoW - vs 7 days ago
    wow_cost = get_closest_historical_cost(7, tolerance=3)
    if wow_cost:
        series["wow"] = {
            "ticker": "$CPI-W",
            "name": "Week-over-Week",
            "base_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "value": round((current_cost / wow_cost) * 100, 1)
        }

    return series


def calculate_methodology_variants(basket_costs: dict, baseline: dict) -> dict:
    """
    Calculate CPI using different methodology weightings.

    Each variant applies different weights to the workload basket,
    representing different use case mixes.
    """
    variants = {}

    for variant_key, variant in METHODOLOGY_WEIGHTS.items():
        weighted_total = 0.0
        baseline_total = 0.0

        baseline_costs = baseline.get("basket_costs", {})

        for workload, weight in variant["weights"].items():
            if workload in basket_costs:
                current_cost = basket_costs[workload]["cost"]
                weighted_total += current_cost * weight

                # Get baseline cost for this workload
                baseline_workload = baseline_costs.get(workload, {})
                if isinstance(baseline_workload, dict):
                    baseline_cost = baseline_workload.get("cost", current_cost)
                else:
                    baseline_cost = current_cost
                baseline_total += baseline_cost * weight

        if baseline_total > 0:
            cpi_value = round((weighted_total / baseline_total) * 100, 1)
        else:
            cpi_value = 100.0

        variants[variant_key] = {
            "ticker": f"$CPI-{variant_key[:3].upper()}",
            "name": variant["name"],
            "description": variant["description"],
            "value": cpi_value,
            "basket_cost": round(weighted_total, 6)
        }

    return variants


def calculate_trend_analysis(current_cpi: float) -> dict:
    """
    Calculate trend direction and velocity.

    Returns:
    - direction: "deflating", "stable", "inflating"
    - velocity: Rate of change per month
    - acceleration: Change in velocity
    - forecast: Projected CPI in 30 days at current rate
    """
    historical = load_historical()

    if len(historical) < 2:
        return {
            "direction": "stable",
            "velocity": 0.0,
            "acceleration": 0.0,
            "forecast_30d": current_cpi,
            "confidence": "low",
            "note": "Insufficient data for trend analysis"
        }

    # Get recent snapshots for trend calculation
    recent = sorted(historical, key=lambda x: x["date"], reverse=True)[:5]

    # Calculate velocity (average monthly change)
    changes = []
    for i in range(len(recent) - 1):
        if recent[i].get("cpi") and recent[i+1].get("cpi"):
            # Calculate days between snapshots
            date1 = datetime.strptime(recent[i]["date"], "%Y-%m-%d")
            date2 = datetime.strptime(recent[i+1]["date"], "%Y-%m-%d")
            days_diff = (date1 - date2).days

            if days_diff > 0:
                # Normalize to monthly rate (30 days)
                change = recent[i]["cpi"] - recent[i+1]["cpi"]
                monthly_rate = (change / days_diff) * 30
                changes.append(monthly_rate)

    if not changes:
        return {
            "direction": "stable",
            "velocity": 0.0,
            "acceleration": 0.0,
            "forecast_30d": current_cpi,
            "confidence": "low",
            "note": "Insufficient data for trend analysis"
        }

    velocity = sum(changes) / len(changes)

    # Calculate acceleration (change in velocity)
    if len(changes) >= 2:
        acceleration = changes[0] - changes[-1]
    else:
        acceleration = 0.0

    # Determine direction
    if velocity < -1.0:
        direction = "deflating"
    elif velocity > 1.0:
        direction = "inflating"
    else:
        direction = "stable"

    # Forecast at current rate
    forecast_30d = round(current_cpi + velocity, 1)

    # Confidence based on data availability
    confidence = "high" if len(historical) >= 30 else "medium" if len(historical) >= 7 else "low"

    return {
        "direction": direction,
        "velocity": round(velocity, 2),
        "velocity_unit": "points per month",
        "acceleration": round(acceleration, 2),
        "forecast_30d": forecast_30d,
        "confidence": confidence
    }


def calculate_yield_curve(current_cost: float) -> dict:
    """
    Calculate the "yield curve" of deflation over time.

    Shows how much cost has decreased at different time horizons,
    similar to a bond yield curve but for AI compute deflation.

    Args:
        current_cost: Current weighted basket cost
    """
    historical = load_historical()

    if not historical:
        return {"note": "Insufficient historical data for yield curve"}

    now = datetime.now(timezone.utc)

    if current_cost == 0:
        return {"note": "No current cost available"}

    # Calculate deflation at different time horizons
    horizons = [
        ("1w", 7),
        ("1m", 30),
        ("3m", 90),
        ("6m", 180),
        ("1y", 365)
    ]

    curve = {}

    for label, days in horizons:
        snapshot = get_closest_snapshot(days, tolerance=max(7, days // 10))

        if snapshot and snapshot.get("basket_cost"):
            past_cost = snapshot["basket_cost"]
            # Calculate deflation rate (negative = costs falling, positive = costs rising)
            deflation = ((current_cost - past_cost) / past_cost) * 100
            # Annualize it
            annualized = (deflation / days) * 365 if days > 0 else 0

            curve[label] = {
                "days": days,
                "deflation_pct": round(deflation, 2),
                "annualized_pct": round(annualized, 2),
                "source_date": snapshot["date"]
            }

    return {
        "curve": curve,
        "interpretation": _interpret_yield_curve(curve),
        "generated_at": now.isoformat()
    }


def _interpret_yield_curve(curve: dict) -> str:
    """Generate human-readable interpretation of yield curve."""
    if not curve:
        return "Insufficient data"

    rates = [v.get("annualized_pct", 0) for v in curve.values() if v.get("annualized_pct")]

    if not rates:
        return "No deflation data available"

    avg_rate = sum(rates) / len(rates)

    if avg_rate < -20:
        return "Rapid deflation - AI compute costs falling significantly"
    elif avg_rate < -5:
        return "Moderate deflation - steady decrease in AI compute costs"
    elif avg_rate < 5:
        return "Stable - AI compute costs roughly unchanged"
    elif avg_rate < 20:
        return "Moderate inflation - AI compute costs increasing"
    else:
        return "Significant inflation - rapid increase in AI compute costs"


# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_cpi_report(api_key: str = None) -> dict:
    """Generate complete CPI report."""

    print("=" * 60)
    print("  OCCUPANT - Compute CPI Calculator")
    print("=" * 60)

    # 1. Build unified model registry
    print("\n[1/12] Building model registry...")
    registry = build_unified_registry(api_key)

    # 2. Calculate basket costs
    print("\n[2/12] Calculating basket costs...")
    basket_costs = calculate_basket_costs(registry)
    weighted_total = calculate_weighted_total(basket_costs)

    # 2b. Calculate subindex costs
    subindex_costs = calculate_subindex_costs(registry)

    # 2c. Calculate persona costs
    persona_costs = calculate_persona_costs(registry)

    # 3. Load or create baseline
    print("\n[3/12] Loading baseline...")
    baseline = load_baseline()
    if baseline is None:
        print("[Baseline] No baseline found, setting today's prices as baseline")
        save_baseline(
            basket_costs={k: {"cost": v["cost"], "weight": v["weight"]} for k, v in basket_costs.items()},
            total_weighted=weighted_total,
            subindex_costs=subindex_costs,
            persona_costs=persona_costs
        )
        baseline = load_baseline()

    # 4. Calculate CPI
    print("\n[4/12] Calculating CPI...")
    baseline_total = baseline.get("total_weighted", weighted_total)
    cpi_value = calculate_cpi(weighted_total, baseline_total)

    # 5. Calculate subindices and spreads
    print("\n[5/12] Calculating subindices and spreads...")
    subindices = calculate_subindices(registry, baseline, subindex_costs)
    spreads = calculate_spreads(registry)

    # 6. Calculate exchange rates
    print("\n[6/12] Calculating exchange rates...")
    exchange_rates = calculate_exchange_rates(registry)

    # 7. Calculate persona CPIs
    print("\n[7/12] Calculating persona CPIs...")
    persona_cpis = calculate_persona_cpis(registry, baseline, persona_costs)

    # 8. Get historical changes
    print("\n[8/12] Computing historical changes...")
    mom_change = get_mom_change(cpi_value)
    yoy_change = get_yoy_change(cpi_value)
    trend = get_trend(mom_change)

    # 9. Calculate index series (multi-period)
    print("\n[9/12] Calculating index series...")
    index_series = calculate_index_series(weighted_total, baseline)

    # 10. Calculate methodology variants
    print("\n[10/12] Calculating methodology variants...")
    methodology_variants = calculate_methodology_variants(basket_costs, baseline)

    # 11. Calculate trend analysis
    print("\n[11/12] Analyzing trends...")
    trend_analysis = calculate_trend_analysis(cpi_value)

    # 12. Calculate yield curve
    print("\n[12/12] Building yield curve...")
    yield_curve = calculate_yield_curve(weighted_total)

    # Build output
    now = datetime.now(timezone.utc)
    report = {
        "meta": {
            "generated_at": now.isoformat(),
            "baseline_date": baseline.get("date", now.strftime("%Y-%m-%d")),
            "launch_date": LAUNCH_BASELINE_DATE,
            "data_sources": registry["meta"]["sources"],
            "methodology_version": "2.0",
            "models_count": registry["meta"]["total_models"]
        },
        "compute_cpi": {
            "ticker": "$CPI",
            "name": "Compute CPI",
            "value": cpi_value,
            "mom_change": mom_change,
            "yoy_change": yoy_change,
            "trend": trend,
            "basket_cost": round(weighted_total * 1000, 4),  # Per 1K workloads
            "basket_cost_unit": "$ per 1K standardized workloads"
        },
        "index_series": index_series,
        "methodology_variants": methodology_variants,
        "trend_analysis": trend_analysis,
        "yield_curve": yield_curve,
        "subindices": {k: {kk: vv for kk, vv in v.items() if kk != "cost"}
                       for k, v in subindices.items()},
        "spreads": spreads,
        "exchange_rates": exchange_rates,
        "persona_cpis": persona_cpis,
        "basket_detail": {
            k: {
                "cost": round(v["cost"] * 1000, 4),  # Per 1K workloads
                "weight": v["weight"],
                "cost_unit": "$ per 1K workloads"
            }
            for k, v in basket_costs.items()
        }
    }

    # Save historical snapshot (including persona data)
    save_snapshot({
        "date": now.strftime("%Y-%m-%d"),
        "cpi": cpi_value,
        "basket_cost": weighted_total,
        "subindices": {k: v["value"] for k, v in subindices.items()},
        "spreads": {k: v["value"] for k, v in spreads.items()},
        "personas": {k: v["cpi"] for k, v in persona_cpis.items()},
        "persona_costs": persona_costs
    })

    return report


def output_dashboard_json(report: dict):
    """Write report to dashboard JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDashboard JSON written to {OUTPUT_PATH}")


def display_terminal_report(report: dict):
    """Display report in terminal."""
    cpi = report["compute_cpi"]
    subs = report["subindices"]
    spreads = report["spreads"]
    exchange = report.get("exchange_rates", {})
    personas = report.get("persona_cpis", {})
    index_series = report.get("index_series", {})
    methodology = report.get("methodology_variants", {})
    trend = report.get("trend_analysis", {})
    yield_curve = report.get("yield_curve", {})

    print("\n" + "=" * 60)
    print("  OCCUPANT - Compute CPI")
    print("  The inflation rate for AI work")
    print("=" * 60)

    # Main CPI
    print(f"\n  $CPI       {cpi['value']:.1f}")
    print("  " + "─" * 50)
    print(f"  Base = 100 ({report['meta']['baseline_date']})")

    # Changes
    if cpi.get("mom_change") is not None:
        yoy = cpi.get('yoy_change')
        yoy_str = f"{yoy:+.1f}%" if yoy is not None else "N/A"
        print(f"  MoM: {cpi['mom_change']:+.1f}%  |  YoY: {yoy_str}")

    # Index Series (multi-period)
    if index_series:
        print("\n  INDEX SERIES")
        for key, series in index_series.items():
            print(f"  {series['ticker']:10} {series['value']:>8.1f}    {series['name']} (vs {series['base_date']})")

    # Methodology Variants
    if methodology:
        print("\n  METHODOLOGY VARIANTS")
        for key, var in methodology.items():
            print(f"  {var['ticker']:10} {var['value']:>8.1f}    {var['name']}")

    # Trend Analysis
    if trend:
        direction_symbol = "↓" if trend.get("direction") == "deflating" else "↑" if trend.get("direction") == "inflating" else "→"
        print(f"\n  TREND ANALYSIS")
        print(f"  Direction:    {direction_symbol} {trend.get('direction', 'stable').upper()}")
        print(f"  Velocity:     {trend.get('velocity', 0):+.2f} pts/month")
        print(f"  30d Forecast: {trend.get('forecast_30d', cpi['value']):.1f}")
        print(f"  Confidence:   {trend.get('confidence', 'low').upper()}")

    # Subindices
    print("\n  SUBINDICES")
    for key, sub in subs.items():
        print(f"  {sub['ticker']:10} {sub['value']:>8.1f}    {sub['name']}")

    # Spreads
    print("\n  SPREADS")
    for key, spread in spreads.items():
        print(f"  {spread['ticker']:10} {spread['value']:>+10.2f}    {spread['name']} ({spread['unit']})")

    # Exchange Rates
    if exchange and exchange.get("rates"):
        print("\n  EXCHANGE RATES")
        print(f"  Base: {exchange['base']['ticker']} ({exchange['base']['name']})")
        for tier, rate in exchange["rates"].items():
            print(f"  {rate['ticker']:10} = {rate['rate']:>4} $UTIL")

    # Yield Curve
    if yield_curve and yield_curve.get("curve"):
        print("\n  YIELD CURVE (Annualized Deflation)")
        for horizon, data in yield_curve["curve"].items():
            print(f"  {horizon:5} {data['annualized_pct']:>+8.1f}%")
        if yield_curve.get("interpretation"):
            print(f"  → {yield_curve['interpretation']}")

    # Persona CPIs
    if personas:
        print("\n  BUILD COST INDEX")
        for key, p in personas.items():
            mom = f"{p['mom_change']:+.1f}%" if p.get('mom_change') is not None else "N/A"
            print(f"  {p['ticker']:10} {p['cpi']:>8.1f}    {p['name']} (MoM: {mom})")

    print(f"\n  Last updated: {report['meta']['generated_at']}")
    print(f"  Data sources: {', '.join(report['meta']['data_sources'])}")
    print(f"  Models: {report['meta']['models_count']}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY")

    report = generate_cpi_report(api_key)
    output_dashboard_json(report)
    display_terminal_report(report)
