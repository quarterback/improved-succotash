#!/usr/bin/env python3
"""
LDI Historical Backfill

Reconstructs a daily LDI time series from the existing Compute CPI historical
snapshots in data/historical.json. For each historical CPI date:

  - AI cost per unit at date d = (current basket cost) × (cpi_d / cpi_today)
    (scales the workload-tier AI prices proportionally with the CPI level)
  - Human cost per unit, volumes, absorption — held constant from the latest
    LDI snapshot. These inputs are annual data and don't move daily.
  - Substitution rate — interpolated from a small monthly anchor table.
    The FPDS signal is annual; the interpolation is a transparent stand-in
    so the chart shows the trajectory implied by the procurement story until
    we have multi-FY actuals.

Output: data/ldi/historical.json with one entry per historical CPI date.
Existing entries from the live pipeline are preserved (more authoritative
than reconstructions); reconstructed entries are flagged with
"reconstructed": true so consumers can filter.
"""

import json
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"
CPI_HISTORICAL_PATH = DATA_DIR / "historical.json"
LDI_HISTORICAL_PATH = LDI_DIR / "historical.json"
LDI_LATEST_PATH = LDI_DIR / "latest.json"


# Monthly substitution-rate anchors. Linearly interpolated between dates.
# Anchored to the live FY2024 sub_rate of ~2.2% in April 2026, decayed back
# to near-zero in early 2025 when AI procurement was nascent. These are
# reconstructions, not measurements — flagged in the output.
SUB_RATE_ANCHORS = [
    ("2025-02-01", 0.30),
    ("2025-05-01", 0.55),
    ("2025-08-01", 0.92),
    ("2025-11-01", 1.40),
    ("2026-02-01", 1.85),
    ("2026-04-13", 2.21),
]

# Per-workload sub-rate anchors (latest values). Held proportional to composite.
PER_WORKLOAD_LATEST_SUB = {
    "snap_eligibility": 0.0,
    "unemployment_claims": 0.0,
    "call_center_triage": 3.99,
    "document_summarization": 0.0,
    "it_help_desk": 0.0,
    "foia_processing": 0.0,
    "uscis_form_intake": 0.0,
    "irs_notice_generation": 0.0,
    "va_appointment_scheduling": 0.0,
}


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def interpolate_sub_rate(target_date):
    target = parse_date(target_date) if isinstance(target_date, str) else target_date
    anchors = [(parse_date(d), v) for d, v in SUB_RATE_ANCHORS]
    if target <= anchors[0][0]:
        return anchors[0][1]
    if target >= anchors[-1][0]:
        return anchors[-1][1]
    for (d1, v1), (d2, v2) in zip(anchors, anchors[1:]):
        if d1 <= target <= d2:
            span = (d2 - d1).days
            if span == 0:
                return v2
            frac = (target - d1).days / span
            return round(v1 + frac * (v2 - v1), 3)
    return anchors[-1][1]


def main():
    if not CPI_HISTORICAL_PATH.exists():
        print(f"[ERROR] CPI historical not found at {CPI_HISTORICAL_PATH}")
        return
    if not LDI_LATEST_PATH.exists():
        print(f"[ERROR] LDI latest.json not found at {LDI_LATEST_PATH}")
        return

    with open(CPI_HISTORICAL_PATH) as f:
        cpi_history = json.load(f)
    with open(LDI_LATEST_PATH) as f:
        latest = json.load(f)

    cpi_today = latest["meta"]["cpi_value"]
    workloads_latest = latest["workloads"]

    # Map workload_id -> current AI cost per unit (the value at cpi_today)
    ai_cost_today = {
        wid: w["ai_cost"]["cost_per_unit"]
        for wid, w in workloads_latest.items()
    }
    human_cost = {
        wid: w["human_cost"]["cost_per_unit"]
        for wid, w in workloads_latest.items()
    }
    volume = {
        wid: w["volume"]["annual_units"]
        for wid, w in workloads_latest.items()
    }
    absorption = {
        wid: w["absorption"]["classification"]
        for wid, w in workloads_latest.items()
    }

    # Load existing LDI history (preserve live entries, replace reconstructions)
    if LDI_HISTORICAL_PATH.exists():
        with open(LDI_HISTORICAL_PATH) as f:
            existing = json.load(f)
    else:
        existing = {"entries": []}

    live_entries_by_date = {
        e["date"]: e
        for e in existing.get("entries", [])
        if not e.get("reconstructed")
    }

    new_entries = []
    for snap in cpi_history:
        snap_date = snap.get("date")
        cpi = snap.get("cpi")
        if not snap_date or not cpi:
            continue

        # Live entry takes precedence
        if snap_date in live_entries_by_date:
            new_entries.append(live_entries_by_date[snap_date])
            continue

        scale = cpi / cpi_today  # AI cost scales with CPI level

        composite_sub = interpolate_sub_rate(snap_date)
        # Distribute the composite proportionally to the latest per-workload pattern
        latest_sum = sum(PER_WORKLOAD_LATEST_SUB.values())
        latest_composite = sum(
            w["substitution_rate"]["rate_pct"] * (volume[wid] / sum(volume.values()))
            for wid, w in workloads_latest.items()
        ) or 1.0
        composite_scale = composite_sub / latest_composite if latest_composite else 0

        per_workload = {}
        total_displacement = 0.0
        cost_ratios = []
        for wid in workloads_latest:
            ai_cost_d = round(ai_cost_today[wid] * scale, 6)
            sub_pct = round(PER_WORKLOAD_LATEST_SUB[wid] * composite_scale, 3)
            per_unit_disp = human_cost[wid] - ai_cost_d
            total_displacement += per_unit_disp * volume[wid]
            if ai_cost_d > 0:
                cost_ratios.append(human_cost[wid] / ai_cost_d)
            per_workload[wid] = {
                "substitution_rate_pct": sub_pct,
                "human_cost_per_unit": human_cost[wid],
                "ai_cost_per_unit": ai_cost_d,
                "absorption": absorption[wid],
            }

        avg_ratio = round(sum(cost_ratios) / len(cost_ratios), 0) if cost_ratios else 0

        new_entries.append({
            "date": snap_date,
            "substitution_rate_pct": composite_sub,
            "cost_displacement_usd": round(total_displacement, 2),
            "avg_cost_ratio": avg_ratio,
            "workload_count": len(workloads_latest),
            "cpi_value": cpi,
            "reconstructed": True,
            "reconstruction_method": (
                "AI cost scaled by CPI ratio against current snapshot. "
                "Substitution rate interpolated from monthly anchors. "
                "Human cost / volume / absorption held constant."
            ),
            "workloads": per_workload,
        })

    # Preserve any live entries whose date isn't in the CPI history (defensive)
    cpi_dates = {snap.get("date") for snap in cpi_history}
    for d, e in live_entries_by_date.items():
        if d not in cpi_dates:
            new_entries.append(e)

    new_entries.sort(key=lambda x: x["date"])

    output = {"entries": new_entries}
    with open(LDI_HISTORICAL_PATH, "w") as f:
        json.dump(output, f, indent=2)

    reconstructed = sum(1 for e in new_entries if e.get("reconstructed"))
    live = len(new_entries) - reconstructed
    print(f"[BACKFILL] Wrote {len(new_entries)} entries to {LDI_HISTORICAL_PATH}")
    print(f"[BACKFILL]   {live} live, {reconstructed} reconstructed")
    print(f"[BACKFILL]   span: {new_entries[0]['date']} -> {new_entries[-1]['date']}")


if __name__ == "__main__":
    main()
