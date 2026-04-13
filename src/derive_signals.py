#!/usr/bin/env python3
"""
LDI Derived Signals

Computes second-order signals from the historical LDI time series and
merges them into latest.json under a "derived_signals" block. These are
fully derived from existing data — no new external sources.

Signals:

1. substitution_velocity — slope of substitution_rate_pct over time,
   expressed as percentage-points per month. Composite + per-workload.
   Computed via least-squares fit on the last `WINDOW_DAYS` of history
   (90 by default). Negative velocity is meaningful too: it would mean
   the procurement signal is regressing.

2. substitution_acceleration — change in velocity between the recent
   half of the window and the prior half. Sign tells you whether
   substitution is speeding up or settling.

A note on epistemics: the bulk of the historical series is reconstructed
(see backfill_ldi.py — sub_rate is interpolated from monthly anchors).
That means velocity computed across the reconstructed segment reflects
the *shape of the anchor curve*, not measurement. Live entries get
velocity computed against neighboring entries the same way; consumers
can read `samples_live` to see how much real signal is in the slope.
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"
LATEST_PATH = LDI_DIR / "latest.json"
HISTORICAL_PATH = LDI_DIR / "historical.json"

WINDOW_DAYS = 90


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def linear_slope(points):
    """
    Least-squares slope of y vs x. Returns slope or None if <2 points
    or all x identical.
    """
    n = len(points)
    if n < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def velocity_from_series(entries, key_path, anchor_date):
    """
    entries: list of dicts each with "date" plus value at key_path
             (key_path is a tuple of nested keys).
    Returns: dict { monthly_pp, samples, samples_live, window_days }
    """
    anchor = parse_date(anchor_date) if isinstance(anchor_date, str) else anchor_date
    cutoff = anchor

    points = []
    live_count = 0
    for e in entries:
        try:
            d = parse_date(e["date"])
        except Exception:
            continue
        if (cutoff - d).days < 0 or (cutoff - d).days > WINDOW_DAYS:
            continue
        v = e
        for k in key_path:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        if v is None:
            continue
        # x = days since the earliest point in the window (computed later)
        points.append((d, float(v), bool(not e.get("reconstructed"))))

    if len(points) < 2:
        return {
            "monthly_pp": None,
            "samples": len(points),
            "samples_live": sum(1 for _, _, live in points if live),
            "window_days": WINDOW_DAYS,
        }

    base = min(p[0] for p in points)
    xy = [((p[0] - base).days, p[1]) for p in points]
    slope_per_day = linear_slope(xy)
    if slope_per_day is None:
        monthly = None
    else:
        # Convert slope (pp per day) to pp per 30-day month
        monthly = round(slope_per_day * 30, 4)

    return {
        "monthly_pp": monthly,
        "samples": len(points),
        "samples_live": sum(1 for _, _, live in points if live),
        "window_days": WINDOW_DAYS,
    }


def acceleration_from_series(entries, key_path, anchor_date):
    """
    Compare velocity over the recent half of the window to velocity over
    the prior half. Returns the delta (recent - prior), pp/month/month.
    """
    anchor = parse_date(anchor_date) if isinstance(anchor_date, str) else anchor_date
    half = WINDOW_DAYS // 2

    def slope_in_window(start_offset, end_offset):
        pts = []
        for e in entries:
            try:
                d = parse_date(e["date"])
            except Exception:
                continue
            age = (anchor - d).days
            if age < start_offset or age > end_offset:
                continue
            v = e
            for k in key_path:
                if not isinstance(v, dict) or k not in v:
                    v = None
                    break
                v = v[k]
            if v is None:
                continue
            pts.append((d, float(v)))
        if len(pts) < 2:
            return None
        base = min(p[0] for p in pts)
        xy = [((p[0] - base).days, p[1]) for p in pts]
        s = linear_slope(xy)
        if s is None:
            return None
        return s * 30  # pp/month

    recent = slope_in_window(0, half)
    prior = slope_in_window(half, WINDOW_DAYS)
    if recent is None or prior is None:
        return None
    return round(recent - prior, 4)


def compute_derived_signals(latest, history):
    entries = history.get("entries", [])
    if not entries:
        return {}

    anchor_date = latest.get("date") or entries[-1].get("date")

    composite_vel = velocity_from_series(entries, ("substitution_rate_pct",), anchor_date)
    composite_accel = acceleration_from_series(entries, ("substitution_rate_pct",), anchor_date)

    per_workload = {}
    for wid in latest.get("workloads", {}):
        v = velocity_from_series(entries, ("workloads", wid, "substitution_rate_pct"), anchor_date)
        a = acceleration_from_series(entries, ("workloads", wid, "substitution_rate_pct"), anchor_date)
        per_workload[wid] = {
            "velocity_monthly_pp": v["monthly_pp"],
            "acceleration_monthly_pp_per_month": a,
            "samples": v["samples"],
            "samples_live": v["samples_live"],
        }

    return {
        "substitution_velocity": {
            "composite_monthly_pp": composite_vel["monthly_pp"],
            "composite_acceleration_monthly_pp_per_month": composite_accel,
            "window_days": WINDOW_DAYS,
            "samples": composite_vel["samples"],
            "samples_live": composite_vel["samples_live"],
            "per_workload": per_workload,
            "method": (
                "Least-squares slope of substitution_rate_pct vs date over the last "
                f"{WINDOW_DAYS} days. Acceleration is the difference between the "
                "slope on the recent half of the window and the prior half. "
                "Reconstructed entries dominate today; samples_live shows how many "
                "live entries fed the slope."
            ),
            "interpretation": (
                "Positive velocity = substitution rate is rising (more procurement-level "
                "evidence of AI displacement over time). Positive acceleration = it's "
                "rising faster than it was a window ago."
            ),
        }
    }


def run():
    print("=" * 60)
    print("  LDI DERIVED SIGNALS")
    print("=" * 60)

    if not LATEST_PATH.exists():
        print(f"[DERIVE] ERROR: {LATEST_PATH} not found")
        return
    if not HISTORICAL_PATH.exists():
        print(f"[DERIVE] ERROR: {HISTORICAL_PATH} not found")
        return

    with open(LATEST_PATH) as f:
        latest = json.load(f)
    with open(HISTORICAL_PATH) as f:
        history = json.load(f)

    derived = compute_derived_signals(latest, history)
    latest["derived_signals"] = derived

    with open(LATEST_PATH, "w") as f:
        json.dump(latest, f, indent=2)

    sv = derived.get("substitution_velocity", {})
    print(f"[DERIVE] composite velocity: {sv.get('composite_monthly_pp')} pp/month "
          f"(samples={sv.get('samples')}, live={sv.get('samples_live')})")
    print(f"[DERIVE] composite acceleration: "
          f"{sv.get('composite_acceleration_monthly_pp_per_month')} pp/month/month")
    for wid, w in sv.get("per_workload", {}).items():
        print(f"[DERIVE]   {wid}: vel={w['velocity_monthly_pp']} pp/mo "
              f"(samples={w['samples']}, live={w['samples_live']})")

    return derived


if __name__ == "__main__":
    run()
