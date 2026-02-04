#!/usr/bin/env python3
"""
Occupant Compute CPI Calculator
The inflation rate for AI work.

This script calculates the Compute CPI index by:
1. Pulling live prices from OpenRouter API
2. Calculating basket costs across workload categories
3. Comparing to baseline to generate index values
4. Outputting JSON for the dashboard

No referral fees. Public APIs. Auditable methodology.
"""

import requests
import json
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# OpenRouter API (free read access)
OPENROUTER_API = "https://openrouter.ai/api/v1/models"

# Your API key (get free at openrouter.ai - only needed for live prices)
# For development, you can use the fallback data below
OPENROUTER_API_KEY = None  # Set this or use environment variable

# Baseline period (Compute CPI = 100)
BASELINE_DATE = "2026-02-01"

# =============================================================================
# BASKET DEFINITION
# The "expenditure categories" for AI work
# =============================================================================

WORKLOAD_BASKET = {
    "chat_drafting": {
        "description": "Conversational AI, drafting, Q&A",
        "input_tokens": 2000,
        "output_tokens": 500,
        "weight": 0.20,
        "tier": "general"
    },
    "summarization": {
        "description": "Document summarization, extraction",
        "input_tokens": 10000,
        "output_tokens": 500,
        "weight": 0.25,
        "tier": "general"
    },
    "classification": {
        "description": "Triage, routing, categorization",
        "input_tokens": 500,
        "output_tokens": 50,
        "weight": 0.20,
        "tier": "budget"
    },
    "coding": {
        "description": "Code generation, debugging",
        "input_tokens": 3000,
        "output_tokens": 1000,
        "weight": 0.15,
        "tier": "frontier"
    },
    "judgment": {
        "description": "Reasoning, analysis, decision support",
        "input_tokens": 5000,
        "output_tokens": 2000,
        "weight": 0.10,
        "tier": "reasoning"
    },
    "long_context": {
        "description": "Large document processing, synthesis",
        "input_tokens": 50000,
        "output_tokens": 1000,
        "weight": 0.10,
        "tier": "longctx"
    }
}

# =============================================================================
# MODEL TIERS
# Representative models for each tier (update as landscape changes)
# =============================================================================

MODEL_TIERS = {
    "budget": [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3-5-haiku",
    ],
    "general": [
        "openai/gpt-4o",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3-5-sonnet",
    ],
    "frontier": [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-2.0-pro-exp-02-05",
    ],
    "reasoning": [
        "openai/o1",
        "openai/o3-mini",
        "anthropic/claude-3-5-sonnet",  # For comparison
        "deepseek/deepseek-r1",
    ],
    "longctx": [
        "google/gemini-2.0-pro-exp-02-05",  # 2M context
        "anthropic/claude-3-5-sonnet",  # 200K context
        "openai/gpt-4o",  # 128K context
    ]
}

# =============================================================================
# BASELINE PRICES (February 2026 = 100)
# These are the prices that set CPI = 100
# Update these once at launch, then never change
# =============================================================================

BASELINE_BASKET_COST = None  # Will be set on first run

# Fallback baseline for development (approximate Feb 2026 prices)
FALLBACK_BASELINE_COST = {
    "chat_drafting": 0.0045,      # $0.0045 per workload
    "summarization": 0.0180,      # $0.018 per workload
    "classification": 0.00025,    # $0.00025 per workload
    "coding": 0.0120,             # $0.012 per workload
    "judgment": 0.0850,           # $0.085 per workload (reasoning models expensive)
    "long_context": 0.0750,       # $0.075 per workload
    "total_weighted": 0.0198      # Weighted basket total
}

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_openrouter_models(api_key=None):
    """Fetch current model pricing from OpenRouter."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(OPENROUTER_API, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException as e:
        print(f"Warning: Could not fetch OpenRouter data: {e}")
        return None

def get_model_pricing(models, model_id):
    """Extract pricing for a specific model."""
    for model in models:
        if model.get("id") == model_id:
            pricing = model.get("pricing", {})
            return {
                "input": float(pricing.get("prompt", 0)),   # per token
                "output": float(pricing.get("completion", 0))  # per token
            }
    return None

def calculate_workload_cost(pricing, input_tokens, output_tokens):
    """Calculate cost for a specific workload."""
    if not pricing:
        return None
    input_cost = (input_tokens * pricing["input"])
    output_cost = (output_tokens * pricing["output"])
    return input_cost + output_cost

# =============================================================================
# INDEX CALCULATION
# =============================================================================

def calculate_tier_average_cost(models_data, tier_models, input_tokens, output_tokens):
    """Calculate average cost across models in a tier."""
    costs = []
    for model_id in tier_models:
        pricing = get_model_pricing(models_data, model_id)
        if pricing:
            cost = calculate_workload_cost(pricing, input_tokens, output_tokens)
            if cost is not None:
                costs.append(cost)
    
    if costs:
        return sum(costs) / len(costs)
    return None

def calculate_basket_costs(models_data):
    """Calculate costs for each workload in the basket."""
    basket_costs = {}
    
    for workload_name, workload in WORKLOAD_BASKET.items():
        tier = workload["tier"]
        tier_models = MODEL_TIERS.get(tier, MODEL_TIERS["general"])
        
        cost = calculate_tier_average_cost(
            models_data,
            tier_models,
            workload["input_tokens"],
            workload["output_tokens"]
        )
        
        basket_costs[workload_name] = {
            "cost": cost,
            "weight": workload["weight"],
            "weighted_cost": cost * workload["weight"] if cost else None
        }
    
    # Calculate total weighted basket cost
    weighted_costs = [v["weighted_cost"] for v in basket_costs.values() if v["weighted_cost"]]
    basket_costs["total_weighted"] = sum(weighted_costs) if weighted_costs else None
    
    return basket_costs

def calculate_cpi(current_cost, baseline_cost):
    """Calculate CPI index value (base = 100)."""
    if baseline_cost and baseline_cost > 0:
        return (current_cost / baseline_cost) * 100
    return 100.0

def calculate_subindices(models_data, baseline_costs):
    """Calculate subindices for specific workload categories."""
    subindices = {}
    
    # Judgment CPI (reasoning-heavy workloads)
    judgment_cost = calculate_tier_average_cost(
        models_data,
        MODEL_TIERS["reasoning"],
        WORKLOAD_BASKET["judgment"]["input_tokens"],
        WORKLOAD_BASKET["judgment"]["output_tokens"]
    )
    if judgment_cost:
        subindices["judgment"] = {
            "ticker": "$JUDGE",
            "name": "Judgment CPI",
            "value": calculate_cpi(judgment_cost, baseline_costs.get("judgment", judgment_cost)),
            "cost": judgment_cost
        }
    
    # Long Context CPI
    longctx_cost = calculate_tier_average_cost(
        models_data,
        MODEL_TIERS["longctx"],
        WORKLOAD_BASKET["long_context"]["input_tokens"],
        WORKLOAD_BASKET["long_context"]["output_tokens"]
    )
    if longctx_cost:
        subindices["longctx"] = {
            "ticker": "$LCTX",
            "name": "LongContext CPI",
            "value": calculate_cpi(longctx_cost, baseline_costs.get("long_context", longctx_cost)),
            "cost": longctx_cost
        }
    
    # Budget Tier Index
    budget_cost = calculate_tier_average_cost(
        models_data,
        MODEL_TIERS["budget"],
        2000, 500  # Standard workload
    )
    if budget_cost:
        subindices["budget"] = {
            "ticker": "$BULK",
            "name": "Budget Tier",
            "value": calculate_cpi(budget_cost, baseline_costs.get("chat_drafting", budget_cost) * 0.3),
            "cost": budget_cost
        }
    
    # Frontier Tier Index
    frontier_cost = calculate_tier_average_cost(
        models_data,
        MODEL_TIERS["frontier"],
        2000, 500  # Standard workload
    )
    if frontier_cost:
        subindices["frontier"] = {
            "ticker": "$FRONT",
            "name": "Frontier Tier",
            "value": calculate_cpi(frontier_cost, baseline_costs.get("chat_drafting", frontier_cost)),
            "cost": frontier_cost
        }
    
    return subindices

def calculate_spreads(subindices):
    """Calculate premium spreads between tiers."""
    spreads = {}
    
    # Cognition Premium: Frontier - Budget
    if "frontier" in subindices and "budget" in subindices:
        spread = subindices["frontier"]["cost"] - subindices["budget"]["cost"]
        spreads["cognition_premium"] = {
            "ticker": "$COG-P",
            "name": "Cognition Premium",
            "value": spread * 1_000_000,  # Per 1M tokens (normalized)
            "unit": "$/1M tokens"
        }
    
    # Judgment Premium: Judgment - Frontier
    if "judgment" in subindices and "frontier" in subindices:
        # Normalize to same workload size for comparison
        judgment_normalized = subindices["judgment"]["cost"] / 7000  # 5k+2k tokens
        frontier_normalized = subindices["frontier"]["cost"] / 2500   # 2k+0.5k tokens
        spread = (judgment_normalized - frontier_normalized) * 2500   # Back to standard
        spreads["judgment_premium"] = {
            "ticker": "$JDG-P",
            "name": "Judgment Premium",
            "value": spread * 1_000_000,
            "unit": "$/1M tokens"
        }
    
    # Context Premium: LongCtx - Frontier
    if "longctx" in subindices and "frontier" in subindices:
        # Normalize per token
        longctx_per_token = subindices["longctx"]["cost"] / 51000  # 50k+1k
        frontier_per_token = subindices["frontier"]["cost"] / 2500
        spread = (longctx_per_token - frontier_per_token)
        spreads["context_premium"] = {
            "ticker": "$CTX-P",
            "name": "Context Premium",
            "value": spread * 1_000_000,
            "unit": "$/1M tokens"
        }
    
    return spreads

# =============================================================================
# HISTORICAL TRACKING
# =============================================================================

def load_historical_data(filepath="data/historical.json"):
    """Load historical CPI data for trend calculation."""
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"records": []}

def save_historical_data(data, filepath="data/historical.json"):
    """Save current data point to historical record."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def calculate_changes(current_value, historical_data):
    """Calculate MoM and YoY changes."""
    records = historical_data.get("records", [])
    
    mom_change = None
    yoy_change = None
    
    # Find last month's value
    if len(records) >= 30:  # ~1 month of daily data
        mom_change = ((current_value - records[-30]["cpi"]) / records[-30]["cpi"]) * 100
    
    # Find last year's value
    if len(records) >= 365:
        yoy_change = ((current_value - records[-365]["cpi"]) / records[-365]["cpi"]) * 100
    
    return {
        "mom": round(mom_change, 2) if mom_change else None,
        "yoy": round(yoy_change, 2) if yoy_change else None
    }

# =============================================================================
# MAIN OUTPUT
# =============================================================================

def generate_cpi_report(api_key=None):
    """Generate the full CPI report."""
    
    # Fetch live data
    models_data = fetch_openrouter_models(api_key)
    
    if not models_data:
        print("Using fallback baseline data for development")
        # Return mock data for development
        return {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "baseline_date": BASELINE_DATE,
                "data_source": "fallback",
                "methodology_version": "1.0"
            },
            "compute_cpi": {
                "ticker": "$CPI",
                "value": 100.0,
                "mom_change": None,
                "yoy_change": None
            },
            "subindices": {},
            "spreads": {},
            "note": "Live data unavailable. Set OPENROUTER_API_KEY for real prices."
        }
    
    # Calculate basket costs
    basket_costs = calculate_basket_costs(models_data)
    
    # Use fallback baseline for now (set your actual baseline on launch)
    baseline = FALLBACK_BASELINE_COST
    
    # Calculate main CPI
    current_total = basket_costs.get("total_weighted")
    baseline_total = baseline.get("total_weighted", current_total)
    cpi_value = calculate_cpi(current_total, baseline_total) if current_total else 100.0
    
    # Calculate subindices
    subindices = calculate_subindices(models_data, baseline)
    
    # Calculate spreads
    spreads = calculate_spreads(subindices)
    
    # Load historical and calculate changes
    historical = load_historical_data()
    changes = calculate_changes(cpi_value, historical)
    
    # Build report
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_date": BASELINE_DATE,
            "data_source": "openrouter",
            "methodology_version": "1.0",
            "models_count": len(models_data)
        },
        "compute_cpi": {
            "ticker": "$CPI",
            "name": "Compute CPI",
            "value": round(cpi_value, 1),
            "mom_change": changes["mom"],
            "yoy_change": changes["yoy"],
            "basket_cost": round(current_total * 1000, 4) if current_total else None,
            "basket_cost_unit": "$ per 1K standardized workloads"
        },
        "subindices": {
            k: {
                "ticker": v["ticker"],
                "name": v["name"],
                "value": round(v["value"], 1)
            }
            for k, v in subindices.items()
        },
        "spreads": {
            k: {
                "ticker": v["ticker"],
                "name": v["name"],
                "value": round(v["value"], 2),
                "unit": v["unit"]
            }
            for k, v in spreads.items()
        },
        "basket_detail": {
            k: {
                "cost": round(v["cost"] * 1000, 4) if v.get("cost") else None,
                "weight": v["weight"],
                "cost_unit": "$ per 1K workloads"
            }
            for k, v in basket_costs.items()
            if k != "total_weighted"
        }
    }
    
    return report

def output_dashboard_json(report, filepath="data/compute-cpi.json"):
    """Output JSON for the dashboard to consume."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Dashboard JSON written to {filepath}")

def print_terminal_display(report):
    """Print a terminal-friendly display of the CPI."""
    cpi = report["compute_cpi"]
    
    print("\n" + "=" * 60)
    print("  OCCUPANT - Compute CPI")
    print("  The inflation rate for AI work")
    print("=" * 60)
    
    print(f"\n  $CPI    {cpi['value']:>8.1f}", end="")
    if cpi.get("mom_change"):
        print(f"    {cpi['mom_change']:+.1f}% MoM", end="")
    if cpi.get("yoy_change"):
        print(f"    {cpi['yoy_change']:+.1f}% YoY", end="")
    print(f"\n  {'─' * 50}")
    print(f"  Base = 100 ({report['meta']['baseline_date']})")
    
    if report["subindices"]:
        print(f"\n  SUBINDICES")
        for key, idx in report["subindices"].items():
            print(f"  {idx['ticker']:<10} {idx['value']:>8.1f}    {idx['name']}")
    
    if report["spreads"]:
        print(f"\n  SPREADS")
        for key, spread in report["spreads"].items():
            print(f"  {spread['ticker']:<10} {spread['value']:>+8.2f}    {spread['name']} ({spread['unit']})")
    
    print(f"\n  Last updated: {report['meta']['generated_at']}")
    print(f"  Data source: {report['meta']['data_source']}")
    print("=" * 60 + "\n")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Get API key from environment or use None for fallback
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    
    # Generate report
    report = generate_cpi_report(api_key)
    
    # Output to JSON
    output_dashboard_json(report)
    
    # Print terminal display
    print_terminal_display(report)
