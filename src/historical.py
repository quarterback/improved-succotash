"""
Historical data tracking for Compute CPI.
Manages baseline, daily snapshots, and MoM/YoY calculations.
"""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
BASELINE_PATH = DATA_DIR / "baseline.json"
HISTORICAL_PATH = DATA_DIR / "historical.json"


def load_baseline() -> Optional[dict]:
    """Load the baseline data. Returns None if not set."""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def save_baseline(basket_costs: dict, total_weighted: float, subindex_costs: dict = None, date: str = None):
    """
    Save baseline prices. This should only be done once at launch.
    After setting, the baseline is immutable.
    """
    if BASELINE_PATH.exists():
        print(f"[Baseline] Already exists, not overwriting")
        return False

    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    baseline = {
        "date": date,
        "basket_costs": basket_costs,
        "total_weighted": total_weighted,
        "subindex_costs": subindex_costs or {},
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "note": "Baseline is immutable after initial setting"
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"[Baseline] Saved baseline for {date} (total_weighted: {total_weighted:.6f})")
    return True


def load_historical() -> list:
    """Load historical snapshots."""
    if not HISTORICAL_PATH.exists():
        return []
    with open(HISTORICAL_PATH) as f:
        return json.load(f)


def save_snapshot(snapshot: dict):
    """
    Append a daily snapshot to historical data.

    Snapshot format:
    {
        "date": "2026-02-04",
        "cpi": 100.0,
        "basket_cost": 0.0409,
        "subindices": {"judgment": 100.0, ...},
        "spreads": {"cognition_premium": 13.3, ...}
    }
    """
    history = load_historical()

    # Check if we already have data for this date
    date = snapshot.get("date")
    existing_dates = {h["date"] for h in history}

    if date in existing_dates:
        # Update existing entry
        history = [h if h["date"] != date else snapshot for h in history]
        print(f"[Historical] Updated snapshot for {date}")
    else:
        history.append(snapshot)
        print(f"[Historical] Added new snapshot for {date}")

    # Sort by date
    history.sort(key=lambda x: x["date"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORICAL_PATH, "w") as f:
        json.dump(history, f, indent=2)

    return history


def get_snapshot_for_date(target_date: str) -> Optional[dict]:
    """Get historical snapshot for a specific date."""
    history = load_historical()
    for snapshot in history:
        if snapshot["date"] == target_date:
            return snapshot
    return None


def get_snapshot_days_ago(days: int) -> Optional[dict]:
    """Get snapshot from N days ago."""
    target = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return get_snapshot_for_date(target)


def calculate_change(current: float, previous: float) -> Optional[float]:
    """Calculate percentage change."""
    if previous is None or previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 2)


def get_mom_change(current_cpi: float) -> Optional[float]:
    """Get month-over-month change (30 days)."""
    snapshot = get_snapshot_days_ago(30)
    if snapshot and snapshot.get("cpi"):
        return calculate_change(current_cpi, snapshot["cpi"])
    return None


def get_yoy_change(current_cpi: float) -> Optional[float]:
    """Get year-over-year change (365 days)."""
    snapshot = get_snapshot_days_ago(365)
    if snapshot and snapshot.get("cpi"):
        return calculate_change(current_cpi, snapshot["cpi"])
    return None


def get_trend(mom_change: Optional[float]) -> str:
    """Determine trend based on MoM change."""
    if mom_change is None:
        return "stable"
    if mom_change > 1:
        return "up"
    if mom_change < -1:
        return "down"
    return "stable"


if __name__ == "__main__":
    # Demo
    print("Baseline:", load_baseline())
    print("Historical snapshots:", len(load_historical()))
