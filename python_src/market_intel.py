#!/usr/bin/env python3
"""
LLM Sabermetrics - Market Intelligence Module

Derived statistics for LLM market analysis, inspired by baseball sabermetrics.
Transforms raw price/volume/quality data into actionable market intelligence.
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

DATA_DIR = Path(__file__).parent.parent / "data"
MARKET_DATA_DIR = DATA_DIR / "market"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceRecord:
    """Supply-side pricing data for a model."""
    model_id: str
    provider: str
    timestamp: str
    input_cost_mtok: float      # $/million tokens input
    output_cost_mtok: float     # $/million tokens output
    blended_cost_mtok: float    # Weighted average (75/25 default)
    context_window: int
    source: str

@dataclass
class VolumeRecord:
    """Demand-side volume data from rankings."""
    model_id: str
    timestamp: str
    period: str                  # "daily", "weekly"
    volume_rank: Optional[int]
    volume_share_pct: Optional[float]
    tokens_total: Optional[int]
    app_count: Optional[int]
    source: str

@dataclass
class QualityRecord:
    """Capability/quality data from benchmarks."""
    model_id: str
    timestamp: str
    arena_elo: Optional[int]
    arena_rank: Optional[int]
    arena_votes: Optional[int]
    mmlu_score: Optional[float]
    humaneval_score: Optional[float]
    source: str

@dataclass
class ModelStats:
    """Computed statistics for a model."""
    model_id: str
    timestamp: str

    # Basic
    blended_cost_mtok: float
    volume_rank: Optional[int]
    quality_rank: Optional[int]

    # Derived - Value
    qap: Optional[float]         # Quality-Adjusted Price
    efficiency: Optional[float]  # Composite efficiency score
    value_rank: Optional[int]

    # Derived - Momentum
    msv: Optional[float]         # Market Share Velocity (% change)
    price_velocity: Optional[float]
    momentum_score: Optional[float]

    # Derived - Market Position
    ppc: Optional[int]           # Price Premium Captured
    qpc: Optional[int]           # Quality Premium Captured


# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_openrouter_rankings() -> dict:
    """
    Fetch model rankings/volume data from OpenRouter.

    Note: OpenRouter doesn't expose volume via API, so this parses
    what's available from their public endpoints.
    """
    # The /api/v1/models endpoint doesn't have volume, but we can
    # extract relative popularity from the rankings page structure
    # For now, return cached/manual data

    rankings_path = MARKET_DATA_DIR / "rankings.json"
    if rankings_path.exists():
        with open(rankings_path) as f:
            return json.load(f)

    return {"models": [], "fetched_at": None, "source": "none"}


def fetch_lmsys_arena() -> dict:
    """
    Fetch ELO ratings from Arena leaderboard.

    Data source: https://arena.ai/leaderboard (formerly LMSYS)
    Falls back to cached local data.
    """
    arena_path = MARKET_DATA_DIR / "arena_elo.json"

    # Load cached data (we maintain this manually for now since
    # arena.ai doesn't expose a public JSON API)
    if arena_path.exists():
        with open(arena_path) as f:
            cached = json.load(f)
            print(f"[Arena] Loaded {len(cached.get('data', {}))} models from cache")
            return cached.get("data", {})

    return {}


def parse_arena_elo(arena_data: dict) -> dict:
    """Parse LMSYS arena data into model_id -> elo mapping."""
    elo_map = {}

    # Handle different data formats LMSYS has used
    if isinstance(arena_data, dict):
        # Format: {"model_name": {"elo": 1234, ...}}
        for model, data in arena_data.items():
            if isinstance(data, dict) and "elo" in data:
                # Normalize model name to our ID format
                model_id = normalize_model_id(model)
                elo_map[model_id] = {
                    "elo": data.get("elo"),
                    "rank": data.get("rank"),
                    "votes": data.get("num_battles") or data.get("votes"),
                    "ci_lower": data.get("ci_lower"),
                    "ci_upper": data.get("ci_upper"),
                }
    elif isinstance(arena_data, list):
        # Format: [{"model": "name", "elo": 1234}, ...]
        for i, entry in enumerate(arena_data):
            if isinstance(entry, dict):
                model = entry.get("model") or entry.get("Model")
                model_id = normalize_model_id(model) if model else None
                if model_id:
                    elo_map[model_id] = {
                        "elo": entry.get("elo") or entry.get("Elo"),
                        "rank": entry.get("rank") or i + 1,
                        "votes": entry.get("num_battles") or entry.get("votes"),
                    }

    return elo_map


def normalize_model_id(name: str) -> str:
    """Normalize various model name formats to our standard ID format."""
    if not name:
        return ""

    name = name.lower().strip()

    # Common mappings
    mappings = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "claude-3-opus": "anthropic/claude-3-opus",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
        "gemini-1.5-pro": "google/gemini-1.5-pro",
        "gemini-1.5-flash": "google/gemini-1.5-flash",
        "gemini-2.0-flash": "google/gemini-2.0-flash-001",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "deepseek-v3": "deepseek/deepseek-chat",
        "deepseek-r1": "deepseek/deepseek-r1",
        "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "o1": "openai/o1",
        "o1-mini": "openai/o1-mini",
        "o3-mini": "openai/o3-mini",
    }

    # Check direct mapping
    for key, value in mappings.items():
        if key in name:
            return value

    # Try to construct provider/model format
    if "/" not in name:
        # Guess provider from name
        if "gpt" in name or name.startswith("o1") or name.startswith("o3"):
            return f"openai/{name}"
        elif "claude" in name:
            return f"anthropic/{name}"
        elif "gemini" in name:
            return f"google/{name}"
        elif "llama" in name:
            return f"meta-llama/{name}"
        elif "deepseek" in name:
            return f"deepseek/{name}"
        elif "qwen" in name:
            return f"qwen/{name}"
        elif "mistral" in name or "mixtral" in name:
            return f"mistralai/{name}"

    return name


# =============================================================================
# STAT CALCULATIONS
# =============================================================================

def calculate_blended_cost(input_cost: float, output_cost: float,
                          input_ratio: float = 0.75) -> float:
    """
    Calculate blended cost per MTok.

    Default 75/25 split reflects typical workloads.
    """
    return (input_cost * input_ratio) + (output_cost * (1 - input_ratio))


def calculate_qap(blended_cost: float, arena_elo: int) -> float:
    """
    Quality-Adjusted Price.

    What are you paying per unit of quality?
    Lower QAP = better value for quality.

    Formula: blended_cost / (elo / 1000)
    """
    if not arena_elo or arena_elo == 0:
        return None
    return round(blended_cost / (arena_elo / 1000), 3)


def calculate_msv(current_share: float, previous_share: float) -> float:
    """
    Market Share Velocity.

    How fast is share changing, normalized for current size?

    Returns: percentage change
    """
    if previous_share == 0:
        return None
    return round((current_share - previous_share) / previous_share * 100, 2)


def calculate_price_velocity(current_price: float, previous_price: float) -> float:
    """
    Price velocity over period.

    Returns: percentage change (negative = deflation)
    """
    if previous_price == 0:
        return None
    return round((current_price - previous_price) / previous_price * 100, 2)


def calculate_momentum(msv: float, elo_velocity: float, price_velocity: float,
                      msv_std: float = 10, elo_std: float = 5, price_std: float = 5) -> float:
    """
    Composite momentum score.

    Combines volume momentum, quality trend, and price trend.

    Formula: (MSV_norm * 0.5) + (ELO_velocity_norm * 0.3) + (-price_velocity_norm * 0.2)

    Positive = good momentum (rising volume, rising quality, falling price)
    """
    if msv is None:
        msv = 0
    if elo_velocity is None:
        elo_velocity = 0
    if price_velocity is None:
        price_velocity = 0

    # Normalize (simple z-score style with assumed std devs)
    msv_norm = msv / msv_std if msv_std else 0
    elo_norm = elo_velocity / elo_std if elo_std else 0
    price_norm = price_velocity / price_std if price_std else 0

    momentum = (msv_norm * 0.5) + (elo_norm * 0.3) + (-price_norm * 0.2)
    return round(momentum, 3)


def calculate_ppc(price_rank: int, volume_rank: int) -> int:
    """
    Price Premium Captured.

    Is volume rank higher or lower than price rank suggests?

    Positive PPC: Model captures more volume than price would suggest (premium brand)
    Negative PPC: Model underperforms vs price position (weak demand)
    """
    if price_rank is None or volume_rank is None:
        return None
    return price_rank - volume_rank


def calculate_efficiency(arena_elo: int, tokens_per_second: float,
                        blended_cost: float) -> float:
    """
    Composite efficiency score.

    Formula: (elo / 1000) * tokens_per_second / blended_cost

    Higher = more capability and speed per dollar
    """
    if not arena_elo or not tokens_per_second or not blended_cost or blended_cost == 0:
        return None
    return round((arena_elo / 1000) * tokens_per_second / blended_cost, 2)


def calculate_category_concentration(category_volumes: dict) -> float:
    """
    Category Concentration Index (Herfindahl-style).

    Is a model a specialist or generalist?

    Returns: 0 to 1 (higher = more concentrated/specialist)
    """
    if not category_volumes:
        return None

    total = sum(category_volumes.values())
    if total == 0:
        return None

    shares = [v / total for v in category_volumes.values()]
    hhi = sum(s ** 2 for s in shares)
    return round(hhi, 3)


def calculate_vwar(model_share: float, baseline_share: float) -> float:
    """
    vWAR - Wins Above Replacement.

    Measures how much more market share a model captures than
    the replacement-level baseline (Gemini Flash = $UTIL).

    Positive vWAR = capturing more than expected
    Negative vWAR = underperforming vs commodity option
    """
    if model_share is None or baseline_share is None:
        return None
    return round(model_share - baseline_share, 3)


def calculate_cognitive_arbitrage(arena_elo: int, blended_cost: float,
                                  avg_elo: float, avg_price: float) -> float:
    """
    Cognitive Arbitrage Score.

    Measures whether a model is over/underpriced for its capability.

    Formula: (elo / avg_elo) / (price / avg_price)

    > 1.0 = underpriced for capability (buy signal)
    < 1.0 = overpriced for capability (avoid)
    = 1.0 = fairly priced

    This compares the model's quality ratio to its price ratio.
    If quality is above average but price is below average, arbitrage > 1.
    """
    if not arena_elo or not blended_cost or blended_cost == 0:
        return None
    if not avg_elo or avg_elo == 0 or not avg_price or avg_price == 0:
        return None

    quality_ratio = arena_elo / avg_elo
    price_ratio = blended_cost / avg_price
    arbitrage = quality_ratio / price_ratio
    return round(arbitrage, 2)


def calculate_completion_ratio(output_tokens: int, input_tokens: int) -> float:
    """
    Completion Ratio - Task signature metric.

    Low (0.1-0.3): Summarization, extraction
    Medium (0.5-1.0): Chat, Q&A
    High (1.5-3.0): Generation, coding, agents

    Tracks what kind of work a model/tier is doing.
    """
    if not input_tokens or input_tokens == 0:
        return None
    return round(output_tokens / input_tokens, 2)


def calculate_stickiness(month5_retention: float, month1_retention: float) -> float:
    """
    Stickiness Index.

    Measures user loyalty / lock-in.

    High stickiness = harder to switch, specialized workflows
    Low stickiness = commodity, easy to replace
    """
    if not month1_retention or month1_retention == 0:
        return None
    return round(month5_retention / month1_retention, 2)


# =============================================================================
# OCCUPANT INDICES
# =============================================================================

def calculate_flow_index(current_volume: int, previous_volume: int,
                        days_between: int = 7) -> dict:
    """
    $FLOW - Token Velocity Index.

    Measures whether compute demand is accelerating or decelerating.

    Returns daily velocity and acceleration.
    """
    if not current_volume or not previous_volume or days_between == 0:
        return {"velocity": None, "acceleration": None}

    daily_velocity = (current_volume - previous_volume) / days_between

    return {
        "velocity": round(daily_velocity, 0),
        "velocity_pct": round((current_volume / previous_volume - 1) * 100, 2) if previous_volume else None,
        "interpretation": "accelerating" if daily_velocity > 0 else "decelerating"
    }


def calculate_switch_index(tier_migrations: dict) -> dict:
    """
    $SWITCH - Cross-tier Migration Index.

    Tracks whether users are trading up or down tiers.

    tier_migrations: {"budget_to_frontier": count, "frontier_to_budget": count, ...}
    """
    if not tier_migrations:
        return {"direction": "stable", "intensity": 0}

    up_migrations = sum(v for k, v in tier_migrations.items() if "to_frontier" in k or "to_reasoning" in k)
    down_migrations = sum(v for k, v in tier_migrations.items() if "to_budget" in k)

    total = up_migrations + down_migrations
    if total == 0:
        return {"direction": "stable", "intensity": 0}

    net = up_migrations - down_migrations
    intensity = abs(net) / total

    return {
        "direction": "premiumizing" if net > 0 else "commoditizing" if net < 0 else "stable",
        "intensity": round(intensity, 2),
        "up_migrations": up_migrations,
        "down_migrations": down_migrations
    }


# =============================================================================
# MARKET ANALYSIS
# =============================================================================

def build_model_stats(registry: dict, arena_data: dict = None,
                     previous_snapshot: dict = None) -> dict:
    """
    Build computed statistics for all models.

    Args:
        registry: Model registry with pricing data
        arena_data: LMSYS arena ELO data
        previous_snapshot: Previous period's stats for velocity calculations

    Returns:
        Dict of model_id -> ModelStats
    """
    stats = {}
    now = datetime.now(timezone.utc).isoformat()

    # Parse arena data
    elo_map = parse_arena_elo(arena_data) if arena_data else {}

    # Get all models with pricing
    models = registry.get("models", {})

    # Calculate blended costs and build initial stats
    price_list = []
    for model_id, model_data in models.items():
        # Handle different pricing field names
        input_cost = model_data.get("input_cost_per_token", 0)
        output_cost = model_data.get("output_cost_per_token", 0)

        # Convert to $/MTok
        input_cost = input_cost * 1_000_000
        output_cost = output_cost * 1_000_000

        if input_cost == 0 and output_cost == 0:
            continue

        blended = calculate_blended_cost(input_cost, output_cost)
        price_list.append((model_id, blended))

        # Get ELO if available
        elo_data = elo_map.get(model_id, {})
        arena_elo = elo_data.get("elo")
        arena_rank = elo_data.get("rank")

        # Calculate QAP
        qap = calculate_qap(blended, arena_elo) if arena_elo else None

        # Calculate velocities if we have previous data
        msv = None
        price_velocity = None
        if previous_snapshot and model_id in previous_snapshot:
            prev = previous_snapshot[model_id]
            if prev.get("volume_share"):
                # Would need current volume share - placeholder
                pass
            if prev.get("blended_cost_mtok"):
                price_velocity = calculate_price_velocity(blended, prev["blended_cost_mtok"])

        stats[model_id] = {
            "model_id": model_id,
            "timestamp": now,
            "blended_cost_mtok": round(blended, 4),
            "arena_elo": arena_elo,
            "arena_rank": arena_rank,
            "qap": qap,
            "price_velocity": price_velocity,
            "msv": msv,
        }

    # Calculate price ranks
    price_list.sort(key=lambda x: x[1])
    for rank, (model_id, _) in enumerate(price_list, 1):
        if model_id in stats:
            stats[model_id]["price_rank"] = rank

    # Calculate value ranks (by QAP, lower is better)
    value_list = [(mid, s["qap"]) for mid, s in stats.items() if s.get("qap")]
    value_list.sort(key=lambda x: x[1])
    for rank, (model_id, _) in enumerate(value_list, 1):
        stats[model_id]["value_rank"] = rank

    # Calculate market averages for cognitive arbitrage
    elos = [s["arena_elo"] for s in stats.values() if s.get("arena_elo")]
    prices = [s["blended_cost_mtok"] for s in stats.values() if s.get("blended_cost_mtok") and s.get("arena_elo")]

    avg_elo = sum(elos) / len(elos) if elos else None
    avg_price = sum(prices) / len(prices) if prices else None

    # Calculate cognitive arbitrage for each model
    # Formula: (elo / avg_elo) / (price / avg_price)
    # > 1.0 means model delivers more quality per dollar than average
    for model_id, s in stats.items():
        if s.get("arena_elo") and s.get("blended_cost_mtok") and avg_elo and avg_price:
            s["arbitrage"] = calculate_cognitive_arbitrage(
                s["arena_elo"], s["blended_cost_mtok"], avg_elo, avg_price
            )
        else:
            s["arbitrage"] = None

    # Calculate PPC for models with both price and volume rank
    # (volume rank would come from rankings data)

    return stats


def format_model_name(model_id: str) -> str:
    """Format model ID for display (e.g., 'anthropic/claude-3.5-sonnet' -> 'Claude 3.5 Sonnet')."""
    if not model_id:
        return "Unknown"

    # Extract model name part
    name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Common formatting rules
    name = name.replace("-", " ").replace("_", " ")

    # Capitalize properly
    words = name.split()
    formatted = []
    for word in words:
        # Keep version numbers as-is
        if any(c.isdigit() for c in word):
            formatted.append(word)
        else:
            formatted.append(word.capitalize())

    return " ".join(formatted)


def generate_market_report(registry: dict, rankings_data: dict = None) -> dict:
    """
    Generate a market intelligence report.

    Returns dict with market summary, top movers, tier analysis, and alerts.
    """
    now = datetime.now(timezone.utc)

    # Fetch quality data
    arena_data = fetch_lmsys_arena()

    # Build model stats
    model_stats = build_model_stats(registry, arena_data)

    # Load rankings data if not provided
    if rankings_data is None:
        rankings_path = DATA_DIR / "rankings" / "latest.json"
        if rankings_path.exists():
            with open(rankings_path) as f:
                rankings_data = json.load(f)

    # Tier analysis
    tier_models = registry.get("tiers", {})
    tier_stats = {}

    for tier_name, model_ids in tier_models.items():
        tier_prices = []
        tier_elos = []

        for model_id in model_ids:
            if model_id in model_stats:
                s = model_stats[model_id]
                if s.get("blended_cost_mtok"):
                    tier_prices.append(s["blended_cost_mtok"])
                if s.get("arena_elo"):
                    tier_elos.append(s["arena_elo"])

        if tier_prices:
            tier_stats[tier_name] = {
                "avg_price_mtok": round(sum(tier_prices) / len(tier_prices), 3),
                "min_price_mtok": round(min(tier_prices), 3),
                "max_price_mtok": round(max(tier_prices), 3),
                "model_count": len(model_ids),
                "avg_elo": round(sum(tier_elos) / len(tier_elos)) if tier_elos else None,
            }

    # Calculate judgment share and provider concentration from rankings
    judgment_share = None
    provider_shares = {}
    total_volume = 0

    # Reasoning model patterns - models optimized for complex reasoning/thinking
    reasoning_patterns = [
        "o1", "o3",           # OpenAI reasoning series
        "r1", "r2",           # DeepSeek reasoning
        "thinking",           # Claude thinking variants
        "reason",             # General reasoning models
        "cot",                # Chain-of-thought models
        "opus",               # Claude Opus (high reasoning)
        "reflect",            # Reflection models
    ]

    if rankings_data and rankings_data.get("models"):
        total_volume = sum(m.get("tokens_weekly", 0) for m in rankings_data["models"].values())

        # Calculate judgment share
        reasoning_volume = 0
        for model_id, vol_data in rankings_data["models"].items():
            tokens = vol_data.get("tokens_weekly", 0)
            lower_id = model_id.lower()

            # Check if it's a reasoning model using pattern matching
            is_reasoning = any(pattern in lower_id for pattern in reasoning_patterns)

            if is_reasoning:
                reasoning_volume += tokens

            # Track provider volumes
            provider = model_id.split("/")[0] if "/" in model_id else "other"
            provider_shares[provider] = provider_shares.get(provider, 0) + tokens

        if total_volume > 0:
            judgment_share = round(reasoning_volume / total_volume * 100, 1)

        # Convert provider volumes to percentages and get top 3
        provider_pcts = {p: round(v / total_volume * 100, 1) for p, v in provider_shares.items()}
        top_providers = sorted(provider_pcts.items(), key=lambda x: x[1], reverse=True)[:3]
        provider_shares = {p: pct for p, pct in top_providers}

    # Generate headline insight
    headline = generate_headline_insight(tier_stats, judgment_share, model_stats)

    # Find top value models (lowest QAP)
    value_leaders = sorted(
        [(mid, s) for mid, s in model_stats.items() if s.get("qap")],
        key=lambda x: x[1]["qap"]
    )[:5]

    # Find arbitrage opportunities (highest arbitrage score = most underpriced)
    arbitrage_leaders = sorted(
        [(mid, s) for mid, s in model_stats.items() if s.get("arbitrage")],
        key=lambda x: x[1]["arbitrage"],
        reverse=True
    )[:5]

    # Build report
    report = {
        "generated_at": now.isoformat(),
        "models_analyzed": len(model_stats),

        "headline": headline,

        "market_pulse": {
            "judgment_share": judgment_share,
            "provider_concentration": provider_shares,
            "total_weekly_tokens": total_volume,
        },

        "tier_analysis": tier_stats,

        "value_leaders": [
            {
                "model": mid,
                "model_name": format_model_name(mid),
                "qap": s["qap"],
                "blended_cost": s["blended_cost_mtok"],
                "arena_elo": s.get("arena_elo"),
                "value_rank": s.get("value_rank"),
            }
            for mid, s in value_leaders
        ],

        "arbitrage_opportunities": [
            {
                "model": mid,
                "model_name": format_model_name(mid),
                "arbitrage": s["arbitrage"],
                "interpretation": "underpriced" if s["arbitrage"] > 1.2 else "fair" if s["arbitrage"] > 0.8 else "overpriced",
                "blended_cost": s["blended_cost_mtok"],
                "arena_elo": s.get("arena_elo"),
            }
            for mid, s in arbitrage_leaders
        ],

        "price_summary": {
            "cheapest": min(model_stats.items(), key=lambda x: x[1].get("blended_cost_mtok", float("inf")))[0] if model_stats else None,
            "most_expensive": max(model_stats.items(), key=lambda x: x[1].get("blended_cost_mtok", 0))[0] if model_stats else None,
        },

        "quality_summary": {
            "highest_elo": max(
                [(mid, s.get("arena_elo") or 0) for mid, s in model_stats.items()],
                key=lambda x: x[1] or 0
            )[0] if model_stats else None,
        },

        "model_stats": model_stats,
    }

    return report


def generate_headline_insight(tier_stats: dict, judgment_share: float, model_stats: dict) -> str:
    """Generate a one-sentence headline insight about the market."""
    insights = []

    # Check reasoning tier trends
    if judgment_share is not None:
        if judgment_share > 10:
            insights.append(f"Reasoning demand strong at {judgment_share}% of volume")
        elif judgment_share > 5:
            insights.append(f"Reasoning models capturing {judgment_share}% market share")

    # Check price trends
    if tier_stats:
        budget_price = tier_stats.get("budget", {}).get("avg_price_mtok")
        frontier_price = tier_stats.get("frontier", {}).get("avg_price_mtok")
        if budget_price and frontier_price:
            spread = frontier_price / budget_price
            if spread > 20:
                insights.append(f"Frontier premium at {spread:.0f}x budget tier")

    # Check for arbitrage opportunities
    high_arb = [s for s in model_stats.values() if (s.get("arbitrage") or 0) > 2.0]
    if len(high_arb) > 2:
        insights.append(f"{len(high_arb)} models showing strong arbitrage signals")

    if insights:
        return ". ".join(insights[:2]) + "."
    return "Market conditions stable. Flash models continue to lead on value."


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_market_snapshot(report: dict):
    """Save market snapshot for historical tracking."""
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot_path = MARKET_DATA_DIR / f"snapshot_{date_str}.json"

    with open(snapshot_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also save as latest.json for frontend
    latest_path = MARKET_DATA_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Market] Saved snapshot to {snapshot_path}")


def load_previous_snapshot(days_ago: int = 7) -> dict:
    """Load a previous snapshot for comparison."""
    target_date = datetime.now(timezone.utc) - timedelta(days=days_ago)

    # Try exact date first, then nearby dates
    for offset in range(0, 4):
        check_date = target_date - timedelta(days=offset)
        date_str = check_date.strftime("%Y-%m-%d")
        snapshot_path = MARKET_DATA_DIR / f"snapshot_{date_str}.json"

        if snapshot_path.exists():
            with open(snapshot_path) as f:
                return json.load(f)

    return {}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model_registry import build_unified_registry
    import os

    print("=" * 60)
    print("  LLM SABERMETRICS - Market Intelligence")
    print("=" * 60)

    # Build registry
    api_key = os.environ.get("OPENROUTER_API_KEY")
    registry = build_unified_registry(api_key)

    # Generate report
    report = generate_market_report(registry)

    # Display results
    print(f"\nModels analyzed: {report['models_analyzed']}")

    print("\n--- TIER ANALYSIS ---")
    for tier, stats in report["tier_analysis"].items():
        print(f"  {tier:12} avg=${stats['avg_price_mtok']:.2f}/MTok  "
              f"(${stats['min_price_mtok']:.2f}-${stats['max_price_mtok']:.2f})  "
              f"n={stats['model_count']}"
              f"  ELO={stats['avg_elo'] or 'N/A'}")

    print("\n--- VALUE LEADERS (Lowest QAP) ---")
    for v in report["value_leaders"]:
        print(f"  {v['model']:40} QAP={v['qap']:.2f}  "
              f"${v['blended_cost']:.2f}/MTok  ELO={v['arena_elo'] or 'N/A'}")

    print("\n--- ARBITRAGE OPPORTUNITIES ---")
    print("  (>1.2 = underpriced, <0.8 = overpriced)")
    for a in report["arbitrage_opportunities"]:
        signal = "BUY" if a['arbitrage'] > 1.2 else "FAIR" if a['arbitrage'] > 0.8 else "SELL"
        print(f"  {a['model']:40} arb={a['arbitrage']:.2f} [{signal:4}]  "
              f"${a['blended_cost']:.2f}/MTok  ELO={a['arena_elo'] or 'N/A'}")

    # Save snapshot
    save_market_snapshot(report)

    print("\n" + "=" * 60)
