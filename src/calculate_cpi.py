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
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_registry import build_unified_registry, calculate_tier_average, calculate_exchange_rates
from historical import (
    load_baseline, save_baseline, save_snapshot,
    get_mom_change, get_yoy_change, get_trend,
    get_persona_mom_change, get_days_since_baseline
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
# MAIN REPORT GENERATION
# =============================================================================

def generate_cpi_report(api_key: str = None) -> dict:
    """Generate complete CPI report."""

    print("=" * 60)
    print("  OCCUPANT - Compute CPI Calculator")
    print("=" * 60)

    # 1. Build unified model registry
    print("\n[1/8] Building model registry...")
    registry = build_unified_registry(api_key)

    # 2. Calculate basket costs
    print("\n[2/8] Calculating basket costs...")
    basket_costs = calculate_basket_costs(registry)
    weighted_total = calculate_weighted_total(basket_costs)

    # 2b. Calculate subindex costs
    subindex_costs = calculate_subindex_costs(registry)

    # 2c. Calculate persona costs
    persona_costs = calculate_persona_costs(registry)

    # 3. Load or create baseline
    print("\n[3/8] Loading baseline...")
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
    print("\n[4/8] Calculating CPI...")
    baseline_total = baseline.get("total_weighted", weighted_total)
    cpi_value = calculate_cpi(weighted_total, baseline_total)

    # 5. Calculate subindices and spreads
    print("\n[5/8] Calculating subindices and spreads...")
    subindices = calculate_subindices(registry, baseline, subindex_costs)
    spreads = calculate_spreads(registry)

    # 6. Calculate exchange rates
    print("\n[6/8] Calculating exchange rates...")
    exchange_rates = calculate_exchange_rates(registry)

    # 7. Calculate persona CPIs
    print("\n[7/8] Calculating persona CPIs...")
    persona_cpis = calculate_persona_cpis(registry, baseline, persona_costs)

    # 8. Get historical changes
    print("\n[8/8] Computing historical changes...")
    mom_change = get_mom_change(cpi_value)
    yoy_change = get_yoy_change(cpi_value)
    trend = get_trend(mom_change)

    # Build output
    now = datetime.now(timezone.utc)
    report = {
        "meta": {
            "generated_at": now.isoformat(),
            "baseline_date": baseline.get("date", now.strftime("%Y-%m-%d")),
            "data_sources": registry["meta"]["sources"],
            "methodology_version": "1.1",
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
