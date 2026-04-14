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


def compute_wage_income_displaced(latest):
    """
    wage_income_displaced = (sub_rate / 100) × annual_volume × human_cost_per_unit

    Distinct from cost_displacement (which is the human-vs-AI gap × volume,
    i.e. structural potential at 100% substitution). Wage income displaced is
    the *realized* labor cost that is being shifted away from human payrolls
    based on the observed substitution signal — what you'd expect to see in
    the wage line, not the savings line.
    """
    workloads = latest.get("workloads", {})
    per_workload = {}
    composite = 0.0

    for wid, w in workloads.items():
        sub_pct = (w.get("substitution_rate", {}) or {}).get("rate_pct") or 0.0
        sub_frac = sub_pct / 100.0
        vol = (w.get("volume", {}) or {}).get("annual_units") or 0
        hcost = (w.get("human_cost", {}) or {}).get("cost_per_unit") or 0.0
        wage_displaced = sub_frac * vol * hcost
        per_workload[wid] = round(wage_displaced, 2)
        composite += wage_displaced

    return {
        "composite_annual_usd": round(composite, 2),
        "per_workload_annual_usd": per_workload,
        "formula": "wage_income_displaced = (sub_rate/100) × annual_volume × human_cost_per_unit",
        "interpretation": (
            "Realized labor income shifted away from human payrolls per the observed "
            "substitution rate. Distinct from cost_displacement (which is the "
            "structural gap at 100% substitution)."
        ),
    }


def compute_contract_vs_fte_split(latest, fpds_data, bls_data):
    """
    For each workload, compare contractor obligations (FPDS PSC totals) to
    in-house wage spend (FedScope FTE × BLS annual wage). Both are "spending
    on the workload" but in different ledger lines; the split tells you
    which substitution lever is closer to the surface.

    contract_share = contractor_spend / (contractor_spend + fte_wage_bill)
    fte_share      = fte_wage_bill   / (contractor_spend + fte_wage_bill)

    Caveat: PSC totals are government-wide and broader than the workload, so
    the contract side over-attributes. Used directionally.
    """
    fpds_workloads = (fpds_data or {}).get("workloads", {})
    bls_workloads = (bls_data or {}).get("workloads", {})
    workloads = latest.get("workloads", {})
    per_workload = {}

    composite_contract = 0.0
    composite_fte = 0.0

    for wid, w in workloads.items():
        contractor = (fpds_workloads.get(wid, {}) or {}).get("current_spend") or 0.0
        fte_current = (w.get("absorption", {}) or {}).get("fte_current")
        annual_wage = (bls_workloads.get(wid, {}) or {}).get("annual_wage") or 0.0
        if fte_current and annual_wage:
            # Use base wage (not fully-loaded) as a conservative estimate; fully-loaded
            # appears elsewhere as cost_per_unit.
            fte_bill = fte_current * annual_wage
        else:
            fte_bill = 0.0

        total = contractor + fte_bill
        if total > 0:
            contract_share = contractor / total
            fte_share = fte_bill / total
        else:
            contract_share = None
            fte_share = None

        per_workload[wid] = {
            "contractor_spend_usd": round(contractor, 2),
            "fte_wage_bill_usd": round(fte_bill, 2),
            "fte_count": fte_current,
            "annual_wage_per_fte": annual_wage,
            "contract_share": round(contract_share, 4) if contract_share is not None else None,
            "fte_share": round(fte_share, 4) if fte_share is not None else None,
        }
        composite_contract += contractor
        composite_fte += fte_bill

    composite_total = composite_contract + composite_fte
    return {
        "composite_contractor_spend_usd": round(composite_contract, 2),
        "composite_fte_wage_bill_usd": round(composite_fte, 2),
        "composite_contract_share": round(composite_contract / composite_total, 4) if composite_total > 0 else None,
        "composite_fte_share": round(composite_fte / composite_total, 4) if composite_total > 0 else None,
        "per_workload": per_workload,
        "formula": "contract_share = contractor_spend / (contractor_spend + fte_count × annual_wage)",
        "caveat": (
            "Contractor spend is from PSC totals (government-wide, broader than the workload), "
            "so the contract side is over-attributed. Directional only."
        ),
    }


def compute_cost_ratio_decomposition(latest, history):
    """
    Decompose the cost_ratio change over the window into the share driven
    by AI cost movement vs the share driven by human cost movement.

    cost_ratio = human / ai
      log(ratio_now / ratio_then) = log(h_now / h_then) - log(a_now / a_then)

    So the "human contribution" to the log-change is +log(h_now/h_then),
    the "AI contribution" is -log(a_now/a_then) (cheaper AI raises the ratio).

    Returns per-workload + composite. For composite, we use volume-weighted
    average of log-ratio changes, then split by share.

    Honest caveat: in the current pipeline the historical series holds human
    cost constant (backfill_ldi.py), so the human contribution will be ~0
    until we have real wage variance over the window. The AI side reflects
    Compute CPI movement. Surface this as-is — the structure stays correct
    once human costs start moving in the live series.
    """
    import math
    entries = list(history.get("entries", []))
    if len(entries) < 2:
        return {"per_workload": {}, "note": "insufficient history"}

    entries.sort(key=lambda e: e["date"])
    anchor_date = parse_date(latest.get("date") or entries[-1]["date"])
    cutoff = anchor_date

    # Window: from window_days ago to anchor_date
    window = [
        e for e in entries
        if (cutoff - parse_date(e["date"])).days <= WINDOW_DAYS
        and (cutoff - parse_date(e["date"])).days >= 0
    ]
    if len(window) < 2:
        return {"per_workload": {}, "note": "no data points in window"}

    earliest = window[0]
    latest_entry = window[-1]

    per_workload = {}
    for wid in latest.get("workloads", {}):
        e_then = (earliest.get("workloads", {}) or {}).get(wid)
        e_now = (latest_entry.get("workloads", {}) or {}).get(wid)
        if not e_then or not e_now:
            per_workload[wid] = None
            continue
        h_then = e_then.get("human_cost_per_unit") or 0
        h_now = e_now.get("human_cost_per_unit") or 0
        a_then = e_then.get("ai_cost_per_unit") or 0
        a_now = e_now.get("ai_cost_per_unit") or 0
        if h_then <= 0 or h_now <= 0 or a_then <= 0 or a_now <= 0:
            per_workload[wid] = None
            continue

        ratio_then = h_then / a_then
        ratio_now = h_now / a_now
        log_change = math.log(ratio_now / ratio_then)
        human_log = math.log(h_now / h_then)
        ai_log = -math.log(a_now / a_then)  # cheaper AI raises ratio

        if abs(log_change) > 1e-9:
            human_share = human_log / log_change
            ai_share = ai_log / log_change
        else:
            human_share = None
            ai_share = None

        per_workload[wid] = {
            "ratio_then": round(ratio_then, 1),
            "ratio_now": round(ratio_now, 1),
            "ratio_change_pct": round((ratio_now / ratio_then - 1) * 100, 2),
            "human_cost_change_pct": round((h_now / h_then - 1) * 100, 4),
            "ai_cost_change_pct": round((a_now / a_then - 1) * 100, 4),
            "human_share_of_log_change": round(human_share, 4) if human_share is not None else None,
            "ai_share_of_log_change": round(ai_share, 4) if ai_share is not None else None,
            "from_date": earliest["date"],
            "to_date": latest_entry["date"],
        }

    # Composite: volume-weighted log-ratio change, then split share.
    workloads = latest.get("workloads", {})
    total_volume = sum(
        (w.get("volume", {}) or {}).get("annual_units", 0)
        for w in workloads.values()
    ) or 1
    composite_log_change = 0.0
    composite_human_log = 0.0
    composite_ai_log = 0.0
    for wid, info in per_workload.items():
        if not info:
            continue
        vol = (workloads.get(wid, {}).get("volume", {}) or {}).get("annual_units", 0)
        weight = vol / total_volume
        log_change = math.log((1 + info["ratio_change_pct"] / 100) or 1)
        human_log = math.log((1 + info["human_cost_change_pct"] / 100) or 1)
        ai_log = -math.log((1 + info["ai_cost_change_pct"] / 100) or 1)
        composite_log_change += log_change * weight
        composite_human_log += human_log * weight
        composite_ai_log += ai_log * weight

    if abs(composite_log_change) > 1e-9:
        comp_human_share = composite_human_log / composite_log_change
        comp_ai_share = composite_ai_log / composite_log_change
    else:
        comp_human_share = None
        comp_ai_share = None

    return {
        "window_days": WINDOW_DAYS,
        "from_date": earliest["date"],
        "to_date": latest_entry["date"],
        "composite_ratio_change_pct": round((math.exp(composite_log_change) - 1) * 100, 2),
        "composite_human_share": round(comp_human_share, 4) if comp_human_share is not None else None,
        "composite_ai_share": round(comp_ai_share, 4) if comp_ai_share is not None else None,
        "per_workload": per_workload,
        "method": (
            "log(ratio_now/ratio_then) = log(h_now/h_then) - log(a_now/a_then). "
            "Each side's share = its log term divided by total log change. "
            "Composite = volume-weighted log changes."
        ),
        "caveat": (
            "Reconstructed history holds human cost constant, so human_share is ~0 "
            "until live wage data accumulates. AI side reflects Compute CPI movement."
        ),
    }


def compute_derived_signals(latest, history, fpds_data=None, bls_data=None):
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
        },
        "wage_income_displaced": compute_wage_income_displaced(latest),
        "contract_vs_fte_split": compute_contract_vs_fte_split(latest, fpds_data, bls_data),
        "cost_ratio_decomposition": compute_cost_ratio_decomposition(latest, history),
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

    fpds_path = LDI_DIR / "fpds_output.json"
    bls_path = LDI_DIR / "bls_output.json"
    fpds_data = json.load(open(fpds_path)) if fpds_path.exists() else None
    bls_data = json.load(open(bls_path)) if bls_path.exists() else None

    derived = compute_derived_signals(latest, history, fpds_data, bls_data)
    latest["derived_signals"] = derived

    with open(LATEST_PATH, "w") as f:
        json.dump(latest, f, indent=2)

    sv = derived.get("substitution_velocity", {})
    print(f"[DERIVE] composite velocity: {sv.get('composite_monthly_pp')} pp/month "
          f"(samples={sv.get('samples')}, live={sv.get('samples_live')})")
    print(f"[DERIVE] composite acceleration: "
          f"{sv.get('composite_acceleration_monthly_pp_per_month')} pp/month/month")

    wid = derived.get("wage_income_displaced", {})
    print(f"[DERIVE] wage_income_displaced (composite): "
          f"${wid.get('composite_annual_usd', 0):,.0f}/yr")

    cf = derived.get("contract_vs_fte_split", {})
    cs = cf.get("composite_contract_share")
    fs = cf.get("composite_fte_share")
    if cs is not None:
        print(f"[DERIVE] contract vs FTE split (composite): "
              f"{cs*100:.1f}% contract / {fs*100:.1f}% in-house wages")

    cd = derived.get("cost_ratio_decomposition", {})
    if cd.get("composite_ratio_change_pct") is not None:
        print(f"[DERIVE] cost-ratio change ({cd.get('from_date')} → {cd.get('to_date')}): "
              f"{cd.get('composite_ratio_change_pct')}% "
              f"(human share: {cd.get('composite_human_share')}, "
              f"AI share: {cd.get('composite_ai_share')})")

    return derived


if __name__ == "__main__":
    run()
