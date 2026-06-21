#!/usr/bin/env python3
"""
DeepSWE capability source for Market Intelligence.

Occupant's quality signal currently rests on a single source — Arena ELO
(data/market/arena_elo.json). That measures general preference, not whether a
model can actually ship working code. DeepSWE adds a second, independent
capability source so the quality metrics don't depend on one leaderboard.

It is NOT a standalone index. This writes a capability source file
(data/market/deepswe.json) that market_intel.py joins onto model stats, the same
way it consumes arena_elo.json.

Source:
- DeepSWE Leaderboard: https://deepswe.datacurve.ai/
  113 original tasks across 91 repositories in 5 languages. Tasks are newly
  authored (not scraped, so they can't leak into training data) and verifiers
  grade actual software behavior, not implementation specifics.

The leaderboard is rendered client-side with no stable JSON endpoint, so — as
fetch_bls.py does for OEWS — the published v1.1 results are recorded here with
full provenance.

Derived metric:
- cost_per_solved_task = avg_cost / pass@1 — expected dollars to obtain one
  passing solution. The quality-adjusted price of agentic software work: a model
  that is cheap per run but fails often can cost more per *solved* task than an
  expensive one that usually succeeds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MARKET_DATA_DIR = PROJECT_ROOT / "data" / "market"

SOURCE = "DeepSWE Leaderboard"
SOURCE_URL = "https://deepswe.datacurve.ai/"
BENCHMARK_VERSION = "v1.1"

BENCHMARK = {
    "tasks": 113,
    "repositories": 91,
    "languages": 5,
    "models_evaluated": 9,
    "note": (
        "Tasks are newly authored rather than scraped from existing projects, span "
        "diverse codebases, demand ~5.5x more code than comparable benchmarks, and "
        "are graded by verifiers that test actual software behavior, not "
        "implementation specifics."
    ),
}

# Provider display name -> registry/OpenRouter slug, so the capability records key
# on the same model_id space as pricing (provider/model) and join cleanly.
PROVIDER_SLUG = {
    "Anthropic": "anthropic",
    "OpenAI": "openai",
    "Google": "google",
    "Zhipu AI": "z-ai",
    "Moonshot": "moonshotai",
}

# Published v1.1 results, recorded verbatim with provider attribution.
# (model, run config, provider, pass@1 %, CI ±%, avg cost $)
RESULTS_RAW = [
    ("claude-fable-5",    "max",    "Anthropic", 70, 4, 21.63),
    ("gpt-5.5",           "xhigh",  "OpenAI",    67, 6,  7.23),
    ("claude-opus-4.8",   "max",    "Anthropic", 59, 2, 13.22),
    ("gpt-5.4",           "xhigh",  "OpenAI",    52, 2,  5.65),
    ("glm-5.2",           "max",    "Zhipu AI",  44, 2,  3.92),
    ("gemini-3.5-flash",  "medium", "Google",    37, 2,  7.34),
    ("kimi-k2.7-code",    None,     "Moonshot",  31, 1,  2.82),
    ("claude-sonnet-4.6", "high",   "Anthropic", 30, 4,  5.52),
    ("gemini-3.1-pro",    "high",   "Google",    12, 2,  9.48),
]


def build_capability_records() -> dict:
    """model_id (provider/model) -> capability record with cost_per_solved_task."""
    data = {}
    for model, config, provider, pass1, ci, cost in RESULTS_RAW:
        slug = PROVIDER_SLUG.get(provider, provider.lower().replace(" ", "-"))
        model_id = f"{slug}/{model}"
        data[model_id] = {
            "model": model,
            "provider": provider,
            "config": config,
            "swe_pass_at_1": pass1,
            "swe_pass_at_1_ci": ci,
            "swe_avg_cost_usd": cost,
            # Expected cost to land one passing solution.
            "cost_per_solved_task_usd": round(cost / (pass1 / 100), 2),
        }
    return data


def build_payload() -> dict:
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "benchmark_version": BENCHMARK_VERSION,
        "note": "Coding-capability source for Market Intelligence — not a standalone index.",
        "benchmark": BENCHMARK,
        "data": build_capability_records(),
    }


def write_outputs(payload: dict) -> None:
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKET_DATA_DIR / "deepswe.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[DeepSWE] Wrote {path.relative_to(PROJECT_ROOT)} "
          f"({len(payload['data'])} capability records)")


if __name__ == "__main__":
    payload = build_payload()
    write_outputs(payload)
    best_value = min(payload["data"].values(), key=lambda r: r["cost_per_solved_task_usd"])
    best_cap = max(payload["data"].values(), key=lambda r: r["swe_pass_at_1"])
    print(f"[DeepSWE] capability leader: {best_cap['model']} ({best_cap['swe_pass_at_1']}% pass@1)")
    print(f"[DeepSWE] value leader: {best_value['model']} "
          f"(${best_value['cost_per_solved_task_usd']:.2f}/solved task)")
