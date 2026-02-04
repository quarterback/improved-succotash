#!/usr/bin/env python3
"""
OpenRouter Rankings Scraper

Fetches weekly token volume and market share data from OpenRouter's
public rankings page. This data enables:
- Market Share Velocity (MSV)
- vWAR calculations
- Token flow indices

The rankings data is embedded in the page as React Server Components
payload with structure: {"x": "date", "ys": {"model_id": token_count}}
"""

import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RANKINGS_DIR = DATA_DIR / "rankings"

OPENROUTER_RANKINGS_URL = "https://openrouter.ai/rankings"


def fetch_rankings_page() -> str:
    """Fetch raw HTML from OpenRouter rankings page."""
    req = urllib.request.Request(
        OPENROUTER_RANKINGS_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; OccupantIndex/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"[Rankings] Failed to fetch: {e}")
        return None


def parse_rankings_html(html: str) -> dict:
    """
    Parse rankings data from HTML.

    OpenRouter embeds the data as RSC (React Server Components) payload
    with structure: {"x": "date", "ys": {"model_id": token_count}}
    """
    rankings = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "openrouter_rankings",
        "weekly_data": [],
        "models": {}
    }

    # Look for the RSC payload pattern with chart data
    # Pattern: "data":[{"x":"date","ys":{...}}]
    pattern = r'"data":\s*\[\s*\{[^]]+\}\s*\]'
    matches = re.findall(pattern, html)

    if not matches:
        # Try alternate pattern for individual data points
        pattern = r'\{"x":"[^"]+","ys":\{[^}]+\}\}'
        matches = re.findall(pattern, html)

    print(f"[Rankings] Found {len(matches)} data chunks")

    # Parse each match
    all_model_volumes = {}

    for match in matches:
        try:
            # Wrap in proper JSON structure if needed
            if not match.startswith('{"data"'):
                match = '{"data":[' + match + ']}'

            data = json.loads(match)
            data_points = data.get("data", [])

            for point in data_points:
                date = point.get("x", "")
                ys = point.get("ys", {})

                if date and ys:
                    rankings["weekly_data"].append({
                        "date": date.split("T")[0] if "T" in date else date,
                        "models": ys
                    })

                    # Aggregate model volumes
                    for model_id, volume in ys.items():
                        if model_id not in all_model_volumes:
                            all_model_volumes[model_id] = 0
                        all_model_volumes[model_id] += volume

        except json.JSONDecodeError:
            continue

    # Get latest week's data
    if rankings["weekly_data"]:
        # Sort by date and get most recent
        rankings["weekly_data"].sort(key=lambda x: x["date"], reverse=True)
        latest = rankings["weekly_data"][0]

        # Calculate market share for latest week
        total_tokens = sum(latest["models"].values())

        for model_id, volume in latest["models"].items():
            share = (volume / total_tokens * 100) if total_tokens > 0 else 0
            rankings["models"][model_id] = {
                "tokens_weekly": volume,
                "market_share_pct": round(share, 3),
                "tokens_formatted": format_tokens(volume)
            }

        rankings["latest_date"] = latest["date"]
        rankings["total_tokens"] = total_tokens

    # Also try to extract from raw patterns if structured parsing failed
    if not rankings["models"]:
        print("[Rankings] Trying raw pattern extraction...")
        rankings = extract_raw_patterns(html, rankings)

    return rankings


def extract_raw_patterns(html: str, rankings: dict) -> dict:
    """
    Fallback: Extract model volumes from raw text patterns.

    OpenRouter embeds data in RSC payload with double-escaped quotes:
    \\"anthropic/claude-4.5-sonnet-20250929\\":288995851600
    """
    # Pattern for double-escaped quotes: \"model_id\":number
    # The backslash is escaped in the HTML, so we match \\\"
    pattern = r'\\\\?"([a-z0-9_-]+/[a-z0-9._-]+)\\\\?":\s*(\d{8,})'
    matches = re.findall(pattern, html, re.IGNORECASE)

    if matches:
        print(f"[Rankings] Found {len(matches)} raw model:volume pairs")

        # Group by model and take the largest value (latest week)
        model_volumes = {}
        for model_id, volume_str in matches:
            volume = int(volume_str)
            if model_id not in model_volumes or volume > model_volumes[model_id]:
                model_volumes[model_id] = volume

        # Calculate shares
        total = sum(model_volumes.values())
        for model_id, volume in model_volumes.items():
            share = (volume / total * 100) if total > 0 else 0
            rankings["models"][model_id] = {
                "tokens_weekly": volume,
                "market_share_pct": round(share, 3),
                "tokens_formatted": format_tokens(volume)
            }

        rankings["total_tokens"] = total

    return rankings


def format_tokens(tokens: int) -> str:
    """Format token count for display (e.g., 1.2B, 500M)."""
    if tokens >= 1_000_000_000_000:
        return f"{tokens / 1_000_000_000_000:.1f}T"
    elif tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.1f}B"
    elif tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def save_rankings_snapshot(rankings: dict):
    """Save rankings snapshot with date stamp."""
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot_path = RANKINGS_DIR / f"rankings_{date_str}.json"

    with open(snapshot_path, "w") as f:
        json.dump(rankings, f, indent=2)

    # Also save as latest
    latest_path = RANKINGS_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"[Rankings] Saved to {snapshot_path}")


def load_previous_rankings(days_ago: int = 7) -> dict:
    """Load previous rankings for comparison."""
    from datetime import timedelta

    target_date = datetime.now(timezone.utc) - timedelta(days=days_ago)

    for offset in range(0, 4):
        check_date = target_date - timedelta(days=offset)
        date_str = check_date.strftime("%Y-%m-%d")
        path = RANKINGS_DIR / f"rankings_{date_str}.json"

        if path.exists():
            with open(path) as f:
                return json.load(f)

    return {}


def calculate_msv(current: dict, previous: dict) -> dict:
    """Calculate Market Share Velocity for each model."""
    msv_data = {}

    current_models = current.get("models", {})
    previous_models = previous.get("models", {})

    for model_id, curr in current_models.items():
        curr_share = curr.get("market_share_pct", 0)

        if model_id in previous_models:
            prev_share = previous_models[model_id].get("market_share_pct", 0)

            if prev_share > 0:
                msv = ((curr_share - prev_share) / prev_share) * 100
                msv_data[model_id] = {
                    "current_share": curr_share,
                    "previous_share": prev_share,
                    "msv": round(msv, 2),
                    "direction": "up" if msv > 2 else "down" if msv < -2 else "stable"
                }

    return msv_data


def main():
    print("=" * 60)
    print("  OPENROUTER RANKINGS SCRAPER")
    print("=" * 60)

    # Fetch page
    print("\n[1/3] Fetching rankings page...")
    html = fetch_rankings_page()

    if not html:
        print("[Error] Failed to fetch page")
        return

    print(f"[1/3] Got {len(html):,} bytes")

    # Parse rankings
    print("\n[2/3] Parsing rankings data...")
    rankings = parse_rankings_html(html)

    model_count = len(rankings.get("models", {}))
    print(f"[2/3] Extracted {model_count} models with volume data")

    # Calculate MSV vs previous week
    print("\n[3/3] Calculating changes...")
    previous = load_previous_rankings(7)
    if previous and previous.get("models"):
        msv_data = calculate_msv(rankings, previous)
        rankings["msv"] = msv_data
        print(f"[3/3] Calculated MSV for {len(msv_data)} models")
    else:
        print("[3/3] No previous data for MSV calculation")

    # Save
    save_rankings_snapshot(rankings)

    # Display summary
    print("\n" + "=" * 60)
    print("  RANKINGS SUMMARY")
    print("=" * 60)

    if rankings.get("models"):
        print(f"\nModels tracked: {len(rankings['models'])}")
        if rankings.get("total_tokens"):
            print(f"Total weekly tokens: {format_tokens(rankings['total_tokens'])}")
        if rankings.get("latest_date"):
            print(f"Data date: {rankings['latest_date']}")

        # Top 10 by volume
        sorted_models = sorted(
            rankings["models"].items(),
            key=lambda x: x[1].get("tokens_weekly", 0),
            reverse=True
        )

        print("\nTop 10 by volume:")
        for model_id, data in sorted_models[:10]:
            share = data.get("market_share_pct", 0)
            tokens = data.get("tokens_formatted", "N/A")
            print(f"  {model_id:45} {tokens:>8}  ({share:.1f}%)")

        # Top gainers (MSV)
        if rankings.get("msv"):
            gainers = sorted(
                rankings["msv"].items(),
                key=lambda x: x[1]["msv"],
                reverse=True
            )[:5]

            print("\nTop gainers (MSV):")
            for model_id, data in gainers:
                print(f"  {model_id:45} {data['msv']:+.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
