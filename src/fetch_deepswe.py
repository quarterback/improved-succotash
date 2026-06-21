#!/usr/bin/env python3
"""
DeepSWE benchmark pipeline — the capability signal.

Occupant's three indices price AI work ($CPI), its volume ($AIU), and its labor
pressure ($LDI). DeepSWE adds the missing fourth axis: capability. It measures
whether frontier coding agents can actually complete real software-engineering
tasks, and what each attempt costs.

Source:
- DeepSWE Leaderboard: https://deepswe.datacurve.ai/
  113 original tasks across 91 repositories in 5 languages. Tasks are newly
  authored (not scraped from existing projects, so they can't leak into training
  data), and verifiers test actual software behavior rather than implementation
  details.

The leaderboard is rendered client-side, so there is no stable JSON endpoint to
scrape. Following the same static-fallback convention used by fetch_bls.py, the
published v1.1 results are recorded here with full provenance and re-derived into
the metrics Occupant cares about.

Derived metric:
- cost_per_solved_task = avg_cost / pass@1 — the expected dollars spent to obtain
  one passing solution. This is the quality-adjusted price of agentic software
  work: a cheap model that fails often can cost more per *solved* task than an
  expensive model that usually succeeds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEEPSWE_DIR = DATA_DIR / "deepswe"

SOURCE_URL = "https://deepswe.datacurve.ai/"
BENCHMARK_VERSION = "v1.1"

# Benchmark-wide facts, from the DeepSWE leaderboard.
BENCHMARK_META = {
    "tasks": 113,
    "repositories": 91,
    "languages": 5,
    "models_evaluated": 9,
    "methodology_note": (
        "Tasks are newly authored rather than scraped from existing projects, span "
        "diverse codebases, demand ~5.5x more code than comparable benchmarks, and "
        "are graded by verifiers that test actual software behavior, not "
        "implementation specifics."
    ),
}

# Published v1.1 leaderboard, recorded verbatim with provider attribution.
# Each row: model id, run config, provider, pass@1 (%), CI (±%), avg cost ($),
# output tokens, agent steps.
LEADERBOARD_RAW = [
    ("claude-fable-5",      "max",    "Anthropic", 70, 4, 21.63, 119_000,  88),
    ("gpt-5.5",             "xhigh",  "OpenAI",    67, 6,  7.23,  46_000,  82),
    ("claude-opus-4.8",     "max",    "Anthropic", 59, 2, 13.22, 135_000, 120),
    ("gpt-5.4",             "xhigh",  "OpenAI",    52, 2,  5.65,  71_000,  70),
    ("glm-5.2",             "max",    "Zhipu AI",  44, 2,  3.92,  78_000, 129),
    ("gemini-3.5-flash",    "medium", "Google",    37, 2,  7.34, 276_000,  86),
    ("kimi-k2.7-code",      None,     "Moonshot",  31, 1,  2.82,  59_000, 149),
    ("claude-sonnet-4.6",   "high",   "Anthropic", 30, 4,  5.52,  76_000, 134),
    ("gemini-3.1-pro",      "high",   "Google",    12, 2,  9.48, 196_000,  81),
]


def build_leaderboard() -> list:
    """Sort by pass@1 (desc) and attach the derived cost-per-solved-task metric."""
    rows = []
    for model, config, provider, pass1, ci, cost, out_tokens, steps in LEADERBOARD_RAW:
        rows.append({
            "model": model,
            "config": config,
            "provider": provider,
            "pass_at_1": pass1,
            "pass_at_1_ci": ci,
            "avg_cost_usd": cost,
            "output_tokens": out_tokens,
            "steps": steps,
            # Expected cost to land one passing solution.
            "cost_per_solved_task_usd": round(cost / (pass1 / 100), 2),
        })
    rows.sort(key=lambda r: r["pass_at_1"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def build_payload() -> dict:
    leaderboard = build_leaderboard()

    # Best value = lowest cost per *solved* task.
    best_value = min(leaderboard, key=lambda r: r["cost_per_solved_task_usd"])
    best_pass = max(leaderboard, key=lambda r: r["pass_at_1"])

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "DeepSWE Leaderboard",
            "source_url": SOURCE_URL,
            "benchmark_version": BENCHMARK_VERSION,
            **BENCHMARK_META,
        },
        "headline": (
            f"{best_pass['model']} leads at {best_pass['pass_at_1']}% pass@1; "
            f"{best_value['model']} is the value leader at "
            f"${best_value['cost_per_solved_task_usd']:.2f} per solved task."
        ),
        "leaderboard": leaderboard,
    }


def write_outputs(payload: dict) -> None:
    DEEPSWE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    latest_path = DEEPSWE_DIR / "latest.json"
    snapshot_path = DEEPSWE_DIR / f"deepswe_{today}.json"

    for path in (latest_path, snapshot_path):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[DeepSWE] Wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    payload = build_payload()
    write_outputs(payload)
    print(f"\n[DeepSWE] {payload['headline']}")
    for r in payload["leaderboard"]:
        print(
            f"  {r['rank']:>2}. {r['model']:<18} {r['pass_at_1']:>3}%  "
            f"${r['avg_cost_usd']:>6.2f}/run  "
            f"${r['cost_per_solved_task_usd']:>6.2f}/solved"
        )
