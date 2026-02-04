#!/usr/bin/env python3
"""
OpenRouter Rankings Scraper

Fetches weekly token volume and market share data from OpenRouter's
public rankings page. This data enables:
- Market Share Velocity (MSV)
- vWAR calculations
- Token flow indices
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

    Note: This is fragile and depends on the page structure.
    OpenRouter's page is JavaScript-rendered, so we need to look for
    embedded JSON data or pre-rendered content.
    """
    rankings = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "openrouter_rankings",
        "models": []
    }

    # Look for embedded JSON data (common pattern for Next.js/React apps)
    # Pattern: __NEXT_DATA__ or similar
    json_patterns = [
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
        r'"rankings":\s*(\[.*?\])',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, html, re.DOTALL)
        if matches:
            try:
                data = json.loads(matches[0])
                print(f"[Rankings] Found embedded JSON data")

                # Navigate to rankings data - structure varies
                if isinstance(data, dict):
                    # Try common paths
                    if 'props' in data:
                        page_props = data.get('props', {}).get('pageProps', {})
                        if 'rankings' in page_props:
                            rankings['raw_data'] = page_props['rankings']
                        elif 'models' in page_props:
                            rankings['raw_data'] = page_props['models']
                    elif 'rankings' in data:
                        rankings['raw_data'] = data['rankings']
                elif isinstance(data, list):
                    rankings['raw_data'] = data

                break
            except json.JSONDecodeError:
                continue

    # If no JSON found, try parsing visible text patterns
    if 'raw_data' not in rankings:
        # Look for model names and token counts in text
        # Pattern: "model-name ... 1.2B tokens" or similar
        model_patterns = [
            r'([\w-]+/[\w.-]+)\s+.*?(\d+(?:\.\d+)?[BMK]?)\s*tokens',
        ]

        for pattern in model_patterns:
            matches = re.findall(pattern, html)
            if matches:
                rankings['models'] = [
                    {"model": m[0], "tokens_text": m[1]}
                    for m in matches
                ]
                break

    return rankings


def parse_token_count(text: str) -> int:
    """Parse token count text like '1.2B' or '500M' to integer."""
    text = text.upper().strip()

    multipliers = {
        'T': 1_000_000_000_000,
        'B': 1_000_000_000,
        'M': 1_000_000,
        'K': 1_000,
    }

    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except ValueError:
                return 0

    try:
        return int(float(text))
    except ValueError:
        return 0


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


def calculate_volume_changes(current: dict, previous: dict) -> dict:
    """Calculate week-over-week volume changes."""
    changes = {}

    current_models = {m.get('model'): m for m in current.get('models', [])}
    previous_models = {m.get('model'): m for m in previous.get('models', [])}

    for model_id, curr in current_models.items():
        if model_id in previous_models:
            prev = previous_models[model_id]

            curr_tokens = parse_token_count(curr.get('tokens_text', '0'))
            prev_tokens = parse_token_count(prev.get('tokens_text', '0'))

            if prev_tokens > 0:
                pct_change = ((curr_tokens - prev_tokens) / prev_tokens) * 100
                changes[model_id] = {
                    "current_tokens": curr_tokens,
                    "previous_tokens": prev_tokens,
                    "change_pct": round(pct_change, 2),
                    "direction": "up" if pct_change > 0 else "down" if pct_change < 0 else "flat"
                }

    return changes


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

    print(f"[1/3] Got {len(html)} bytes")

    # Parse rankings
    print("\n[2/3] Parsing rankings data...")
    rankings = parse_rankings_html(html)

    model_count = len(rankings.get('models', []))
    if rankings.get('raw_data'):
        print(f"[2/3] Found embedded JSON data")
    else:
        print(f"[2/3] Parsed {model_count} models from HTML")

    # Calculate changes vs previous week
    print("\n[3/3] Calculating changes...")
    previous = load_previous_rankings(7)
    if previous:
        changes = calculate_volume_changes(rankings, previous)
        rankings['weekly_changes'] = changes
        print(f"[3/3] Calculated changes for {len(changes)} models")
    else:
        print("[3/3] No previous data for comparison")

    # Save
    save_rankings_snapshot(rankings)

    # Display summary
    print("\n" + "=" * 60)
    print("  RANKINGS SUMMARY")
    print("=" * 60)

    if rankings.get('models'):
        print(f"\nModels tracked: {len(rankings['models'])}")
        print("\nTop models:")
        for m in rankings['models'][:5]:
            print(f"  {m.get('model', 'unknown'):40} {m.get('tokens_text', 'N/A')}")

    if rankings.get('weekly_changes'):
        gainers = sorted(
            rankings['weekly_changes'].items(),
            key=lambda x: x[1]['change_pct'],
            reverse=True
        )[:5]

        print("\nTop gainers (WoW):")
        for model, data in gainers:
            print(f"  {model:40} {data['change_pct']:+.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
