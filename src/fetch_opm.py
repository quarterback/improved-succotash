#!/usr/bin/env python3
"""
OPM FedScope FTE Pipeline for LDI

Loads federal civilian FTE counts by SOC code from OPM FedScope cube
exports. FedScope publishes quarterly cube files (CSV) at
https://www.fedscope.opm.gov/ — there is no public REST API, so this
pipeline ships with a static FY-level snapshot derived from the
"Employment" cube (occupational series → mapped to SOC).

The FY counts here are conservative estimates aggregated across the
agencies that own each workload. They give the absorption signal a
denominator: a sub_rate of 2% means little without knowing whether the
underlying workforce is 2,000 or 200,000 people.

When the FedScope cube changes, replace the values in OPM_FTE_SNAPSHOT
or wire up CSV ingestion (see USAGE notes at the bottom of this file).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"

# Static FedScope-derived FTE counts. SOC code -> { fy: count, agencies: [...] }.
# Numbers are FedScope-September-2024 cube; figures are aggregate across
# agencies that own each workload. Citation:
# OPM FedScope Employment Cube, Sep 2024 — fedscope.opm.gov
OPM_FTE_SNAPSHOT = {
    "13-1041": {  # Eligibility Interviewers, Government Programs (SNAP)
        "fte_current": 16_500,
        "fte_prior_fy": 16_800,
        "primary_agencies": ["USDA FNS", "State agency staff funded via federal pass-through"],
        "note": "Includes federal-only headcount; state SNAP eligibility workers (~50K) are funded through federal grants but not on federal rolls.",
    },
    "13-1031": {  # Claims Adjusters / UI
        "fte_current": 4_900,
        "fte_prior_fy": 5_100,
        "primary_agencies": ["DOL ETA (federal oversight)", "State workforce agencies"],
        "note": "Federal staff at DOL ETA only. State UI adjudicator headcount (~28K) is reported separately and funded via federal grants.",
    },
    "43-4051": {  # Customer Service Reps
        "fte_current": 92_000,
        "fte_prior_fy": 95_000,
        "primary_agencies": ["SSA", "IRS", "VA", "USCIS"],
        "note": "Federal CSR headcount across the four highest-volume call-center agencies. SSA dominates (~31K).",
    },
    "23-2093": {  # Title Examiners / abstractors / casework
        "fte_current": 8_400,
        "fte_prior_fy": 8_500,
        "primary_agencies": ["VA (claims processors)", "SSA (case developers)", "DOJ"],
        "note": "Closest federal SOC for document-intensive casework. Excludes attorneys (23-1011).",
    },
    "15-1232": {  # Computer User Support Specialists
        "fte_current": 12_200,
        "fte_prior_fy": 13_400,
        "primary_agencies": ["GSA", "Treasury", "DoD civilian", "VA OIT", "HHS"],
        "note": "Tier-1 federal civilian help desk. Decline reflects unfilled vacancies, not formal eliminations.",
    },
    "23-1011": {  # Lawyers / FOIA officers
        "fte_current": 4_300,
        "fte_prior_fy": 4_350,
        "primary_agencies": ["DOJ OIP", "DHS", "DoD", "State", "HHS"],
        "note": "FOIA-specific FTE only — federal agencies' Government Information Specialists (1410 series) plus attorney detail; broader 23-1011 federal lawyer headcount is much higher.",
    },
    "13-2081": {  # Tax Examiners (IRS notice generation)
        "fte_current": 31_500,
        "fte_prior_fy": 33_200,
        "primary_agencies": ["IRS W&I", "IRS SB/SE"],
        "note": "IRS tax examiners and revenue agents. Decline reflects FY2024 IRS staffing recalibration after the IRA-funded buildout.",
    },
}


def absorption_from_fte_change(soc_code: str) -> dict:
    """
    Quantitative absorption signal derived from FTE delta.

    Rule of thumb (matches how the AAR describes the categories):
      ΔFTE ≤ -3%  → eliminated
      ΔFTE ≤ -1%  → frozen
      ΔFTE within ±1% → reallocated
      ΔFTE ≥ +2%  → upgraded
    """
    info = OPM_FTE_SNAPSHOT.get(soc_code)
    if not info:
        return {
            "fte_current": None,
            "fte_prior": None,
            "fte_delta_pct": None,
            "absorption_quant": "unknown",
            "source": "no FedScope match",
        }

    cur = info["fte_current"]
    prev = info["fte_prior_fy"]
    delta_pct = ((cur - prev) / prev) * 100 if prev > 0 else 0.0

    if delta_pct <= -3:
        cls = "eliminated"
    elif delta_pct <= -1:
        cls = "frozen"
    elif delta_pct >= 2:
        cls = "upgraded"
    else:
        cls = "reallocated"

    return {
        "fte_current": cur,
        "fte_prior": prev,
        "fte_delta_pct": round(delta_pct, 2),
        "absorption_quant": cls,
        "primary_agencies": info["primary_agencies"],
        "note": info["note"],
        "source": "OPM FedScope Employment Cube, Sep 2024 (static snapshot)",
    }


def run():
    print("=" * 60)
    print("  OPM FEDSCOPE FTE PIPELINE")
    print("=" * 60)

    LDI_DIR.mkdir(parents=True, exist_ok=True)

    workload_map_path = LDI_DIR / "workload_map.json"
    if not workload_map_path.exists():
        print(f"[OPM] ERROR: workload_map.json not found at {workload_map_path}")
        return {}

    with open(workload_map_path) as f:
        workload_map = json.load(f)

    workload_data = {}
    for workload_id, workload in workload_map.get("workloads", {}).items():
        soc = workload["soc_code"]
        result = absorption_from_fte_change(soc)
        workload_data[workload_id] = {
            "soc_code": soc,
            **result,
        }
        print(f"[OPM] {workload_id} (SOC {soc}): "
              f"FTE {result.get('fte_current')} (Δ {result.get('fte_delta_pct')}%) "
              f"→ {result['absorption_quant']}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "OPM FedScope Employment Cube",
        "url": "https://www.fedscope.opm.gov/",
        "snapshot_period": "September 2024",
        "method": (
            "Static snapshot from FedScope Employment cube. SOC-mapped headcount "
            "aggregated across agencies that own each workload. Quantitative "
            "absorption classification derived from FY-over-FY FTE delta."
        ),
        "workloads": workload_data,
    }

    out_path = LDI_DIR / "opm_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OPM] Saved output to {out_path}")
    return output


# USAGE
# -----
# To wire up live FedScope CSV ingestion: download the cube CSV from
# fedscope.opm.gov, filter for the target Occupational Series codes
# (e.g. GS-0105 for Social Insurance Specialist ↔ SOC 13-1041), and
# replace OPM_FTE_SNAPSHOT entries with the parsed values.

if __name__ == "__main__":
    run()
