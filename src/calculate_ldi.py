#!/usr/bin/env python3
"""
Labor Displacement Index (LDI) Calculator

Mirrors the structure of calculate_cpi.py.

Steps:
1. Load workload map, BLS output, and FPDS output
2. Pull AI cost per unit from existing CPI basket data
3. Compute substitution rate, cost displacement, and absorption classification
4. Output data/ldi/latest.json and append to data/ldi/historical.json

Three computable signals:
- Substitution rate: % of human workload being substituted by AI (proxy from FPDS)
- Cost displacement: (human_cost_per_unit - ai_cost_per_unit) × volume
- Absorption classification: eliminated / reallocated / frozen (inferred from JOLTS + FPDS)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"

sys.path.insert(0, str(Path(__file__).parent))

from fetch_bls import run as run_bls
from fetch_fpds import run as run_fpds
from fetch_opm import run as run_opm
from derive_signals import run as run_derive


AI_COST_MAP = {
    "classification": {
        "description": "Classification tasks — budget tier AI",
        "cpi_workload": "classification",
        "cost_per_unit": 0.000257,
        "cost_unit": "USD per workload unit",
        "source": "Compute CPI basket: classification workload (500 in + 50 out tokens, budget tier)"
    },
    "chat_drafting": {
        "description": "Chat / triage tasks — general tier AI",
        "cpi_workload": "chat_drafting",
        "cost_per_unit": 0.006225,
        "cost_unit": "USD per workload unit",
        "source": "Compute CPI basket: chat_drafting workload (2000 in + 500 out tokens, general tier)"
    },
    "summarization": {
        "description": "Document summarization — general tier AI",
        "cpi_workload": "summarization",
        "cost_per_unit": 0.018225,
        "cost_unit": "USD per workload unit",
        "source": "Compute CPI basket: summarization workload (10000 in + 500 out tokens, general tier)"
    }
}

VOLUME_ESTIMATES = {
    "snap_eligibility": {
        "annual_units": 41_000_000,
        "source": "USDA FNS FY2024: 41M SNAP participants, ~1 eligibility review/participant/year"
    },
    "unemployment_claims": {
        "annual_units": 21_000_000,
        "source": "DOL ETA FY2024: ~21M initial UI claims filed"
    },
    "call_center_triage": {
        "annual_units": 100_000_000,
        "source": "SSA + IRS + VA combined call volume estimate FY2024"
    },
    "document_summarization": {
        "annual_units": 750_000,
        "source": "VA disability claims 250K/yr + SSA hearings 500K/yr"
    },
    "it_help_desk": {
        "annual_units": 18_000_000,
        "source": "Federal civilian workforce 2.1M × ~8.5 tickets/employee/year (ITSM benchmark)"
    },
    "foia_processing": {
        "annual_units": 1_500_000,
        "source": "DOJ OIP FOIA Annual Report FY2023: ~1.5M federal FOIA requests received government-wide"
    },
    "uscis_form_intake": {
        "annual_units": 8_000_000,
        "source": "USCIS FY2024 quarterly data: ~8M form filings annually across all benefit categories"
    },
    "irs_notice_generation": {
        "annual_units": 200_000_000,
        "source": "IRS Data Book FY2023: ~200M pieces of correspondence (math-error, balance-due, identity-verification, routine response notices)"
    },
    "va_appointment_scheduling": {
        "annual_units": 130_000_000,
        "source": "VA Health Care FY2023 utilization: 130M outpatient appointments (each with ≥1 scheduling touch)"
    }
}

ABSORPTION_RULES = {
    "snap_eligibility": {
        "classification": "reallocated",
        "reasoning": (
            "SNAP caseload growing while workforce is flat. Workers being redirected to "
            "complex cases and fraud detection as routine processing automates. "
            "JOLTS government sector: low separations, moderate reallocation signal."
        )
    },
    "unemployment_claims": {
        "classification": "frozen",
        "reasoning": (
            "DOL UI staffing flat since 2022 post-pandemic normalization. "
            "No new hires for claims processing roles despite rising caseloads. "
            "Vacancy rate >6 months in multiple state UI agencies. Frozen classification."
        )
    },
    "call_center_triage": {
        "classification": "reallocated",
        "reasoning": (
            "IRS and SSA piloting AI-assisted call routing. Frontline agents being "
            "redirected to complex case escalations. Headcount flat while call volume rises — "
            "efficiency gain absorbed, not reflected in separations yet."
        )
    },
    "document_summarization": {
        "classification": "reallocated",
        "reasoning": (
            "VA deploying AI-assisted claims development tools. Legal staff redirected "
            "from mechanical summarization to judgment-intensive review. "
            "Reclassification signal: GS-7 to GS-9 band for remaining roles."
        )
    },
    "it_help_desk": {
        "classification": "frozen",
        "reasoning": (
            "Federal IT help desk positions frozen in multiple agencies since FY2023. "
            "AI-assisted ticketing tools deployed (ServiceNow AI, AWS Connect). "
            "No backfill of vacated Tier-1 positions. Budget justification omits replacement headcount."
        )
    },
    "foia_processing": {
        "classification": "reallocated",
        "reasoning": (
            "FOIA officer headcount roughly flat while request volume grows 5-7%/yr. "
            "AI-assisted redaction (DoD's RELYANT, State Department's FOIA Online) "
            "redirects officers from mechanical redaction to exemption-decision work."
        )
    },
    "uscis_form_intake": {
        "classification": "reallocated",
        "reasoning": (
            "USCIS Immigration Services Officers seeing case-mix shift toward complex "
            "adjudications as ELIS automates routine intake. Backlog still growing — "
            "absorption is into harder cases, not headcount cuts."
        )
    },
    "irs_notice_generation": {
        "classification": "frozen",
        "reasoning": (
            "IRS hiring restraint after FY2024 staffing recalibration. Notice generation "
            "automation reduces need for manual correspondence drafting; vacated routine "
            "correspondence positions not backfilled."
        )
    },
    "va_appointment_scheduling": {
        "classification": "reallocated",
        "reasoning": (
            "VHA Medical Support Assistants being redirected to in-person patient navigation "
            "as AI voice scheduling agents handle routine slot-finding. Headcount stable; "
            "task mix shifting toward complex care coordination."
        )
    }
}

ABSORPTION_COLOR = {
    "eliminated": "danger",
    "reallocated": "warning",
    "frozen": "accent",
    "upgraded": "success"
}


def load_bls_output() -> dict:
    path = LDI_DIR / "bls_output.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_fpds_output() -> dict:
    path = LDI_DIR / "fpds_output.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_opm_output() -> dict:
    path = LDI_DIR / "opm_output.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_workload_map() -> dict:
    path = LDI_DIR / "workload_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_cpi_data() -> dict:
    path = DATA_DIR / "compute-cpi.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_ai_cost_from_cpi(cpi_data: dict, ai_task_mapping: str) -> dict:
    """
    Pull AI cost per unit from existing CPI basket data.
    Falls back to static AI_COST_MAP if CPI data unavailable.
    """
    default = AI_COST_MAP.get(ai_task_mapping, AI_COST_MAP["classification"])

    if not cpi_data:
        return default

    basket = cpi_data.get("basket_detail", {})
    cpi_value = cpi_data.get("compute_cpi", {}).get("value", 100.0)

    cpi_workload = default["cpi_workload"]
    basket_entry = basket.get(cpi_workload)

    if not basket_entry:
        return default

    cost_per_1k = basket_entry.get("cost", 0)
    cost_per_unit = cost_per_1k / 1000

    return {
        "description": default["description"],
        "cpi_workload": cpi_workload,
        "cost_per_unit": round(cost_per_unit, 6),
        "cost_unit": "USD per workload unit",
        "source": f"Compute CPI basket live data (CPI={cpi_value})",
        "cpi_value": cpi_value
    }


def compute_displacement(human_cost: float, ai_cost: float, annual_units: int) -> dict:
    """
    Compute cost displacement.

    Formula:
    displacement = (human_cost_per_unit - ai_cost_per_unit) × annual_volume
    """
    displacement_per_unit = human_cost - ai_cost
    total_displacement = displacement_per_unit * annual_units
    displacement_ratio = (displacement_per_unit / human_cost) if human_cost > 0 else 0

    return {
        "displacement_per_unit": round(displacement_per_unit, 6),
        "annual_volume": annual_units,
        "total_annual_displacement": round(total_displacement, 2),
        "displacement_ratio": round(displacement_ratio, 4),
        "displacement_pct": round(displacement_ratio * 100, 2)
    }


def calculate_composite_ldi(workloads: dict) -> dict:
    """
    Calculate composite LDI metrics across all pilot workloads.

    Two distinct signals:

    1. substitution_rate — weighted average of FPDS-derived substitution proxies
       (the observable signal: what fraction of these task categories appears to be
       routing to AI based on procurement data). This is the composite LDI value.

    2. cost_displacement — total dollar gap between human and AI cost at current
       volumes. This is a structural/potential figure, not a realized one.

    These are separated intentionally. The cost gap tells you why substitution
    will happen. The substitution rate tells you how much has happened so far.
    The two should never be presented as the same number.
    """
    total_volume = sum(
        w.get("volume", {}).get("annual_units", 0)
        for w in workloads.values()
    )
    if total_volume == 0:
        return {"substitution_rate": 0.0, "total_cost_displacement_usd": 0.0}

    # Signal 1: Substitution rate (volume-weighted average of FPDS proxy)
    weighted_sub_rate = 0.0
    total_cost_displacement = 0.0
    total_human_cost_annual = 0.0
    total_ai_cost_annual = 0.0

    # Also compute unweighted avg cost differential (per unit, across workloads)
    cost_ratios = []

    for workload in workloads.values():
        vol = workload.get("volume", {}).get("annual_units", 0)
        sub = workload.get("substitution_rate", {})
        disp = workload.get("displacement", {})
        human_cost = workload.get("human_cost", {}).get("cost_per_unit", 0)
        ai_cost = workload.get("ai_cost", {}).get("cost_per_unit", 0)

        weight = vol / total_volume
        weighted_sub_rate += sub.get("rate", 0) * weight
        total_cost_displacement += disp.get("total_annual_displacement", 0)
        total_human_cost_annual += human_cost * vol
        total_ai_cost_annual += ai_cost * vol

        if ai_cost > 0:
            cost_ratios.append(human_cost / ai_cost)

    avg_cost_ratio = round(sum(cost_ratios) / len(cost_ratios), 0) if cost_ratios else 0

    sub_rate_pct = round(weighted_sub_rate * 100, 2)

    return {
        "substitution_rate": sub_rate_pct,
        "substitution_rate_label": "Estimated substitution rate (FPDS proxy)",
        "substitution_rate_interpretation": (
            f"Approximately {sub_rate_pct:.1f}% of task volume in these workload categories "
            "shows procurement-level evidence of AI substitution (contractor spend decline + "
            "AI procurement growth, FY2023 vs FY2024)."
        ),
        "cost_displacement_usd_annual": round(total_cost_displacement, 2),
        "cost_displacement_label": "Structural cost displacement (potential)",
        "cost_displacement_interpretation": (
            "If AI handled 100% of these workloads, the cost differential at current pricing "
            "would be this amount per year. This is technical potential, not realized savings."
        ),
        "avg_cost_ratio": avg_cost_ratio,
        "total_human_cost_annual": round(total_human_cost_annual, 2),
        "total_ai_cost_annual": round(total_ai_cost_annual, 2),
        "pilot_workload_count": len(workloads),
        "total_volume_units": total_volume
    }


def run_pipeline(force_refresh: bool = False):
    """
    Full LDI pipeline:
    1. Run BLS fetch (or load cached)
    2. Run FPDS fetch (or load cached)
    3. Load workload map and CPI data
    4. Compute metrics for each workload
    5. Save latest.json and append to historical.json
    """
    print("=" * 60)
    print("  LABOR DISPLACEMENT INDEX CALCULATOR")
    print("=" * 60)

    LDI_DIR.mkdir(parents=True, exist_ok=True)

    bls_path = LDI_DIR / "bls_output.json"
    fpds_path = LDI_DIR / "fpds_output.json"

    if force_refresh or not bls_path.exists():
        print("\n--- Running BLS pipeline ---")
        run_bls()
    else:
        print(f"\n[LDI] Using cached BLS data ({bls_path})")

    if force_refresh or not fpds_path.exists():
        print("\n--- Running FPDS pipeline ---")
        run_fpds()
    else:
        print(f"[LDI] Using cached FPDS data ({fpds_path})")

    opm_path = LDI_DIR / "opm_output.json"
    if force_refresh or not opm_path.exists():
        print("\n--- Running OPM FedScope pipeline ---")
        run_opm()
    else:
        print(f"[LDI] Using cached OPM data ({opm_path})")

    workload_map = load_workload_map()
    bls_data = load_bls_output()
    fpds_data = load_fpds_output()
    opm_data = load_opm_output()
    cpi_data = load_cpi_data()

    if not workload_map:
        print("[LDI] ERROR: No workload map found")
        return

    if not bls_data:
        print("[LDI] ERROR: No BLS data found")
        return

    print("\n--- Computing LDI metrics ---")

    workload_results = {}
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    for workload_id, workload_def in workload_map["workloads"].items():
        print(f"\n[LDI] Processing: {workload_id}")

        bls_workload = bls_data.get("workloads", {}).get(workload_id)
        fpds_workload = fpds_data.get("workloads", {}).get(workload_id, {})
        opm_workload = opm_data.get("workloads", {}).get(workload_id, {})
        volume_info = VOLUME_ESTIMATES.get(workload_id, {})
        absorption_info = ABSORPTION_RULES.get(workload_id, {})

        if not bls_workload:
            print(f"  [Warning] No BLS data for {workload_id}, skipping")
            continue

        human_cost = bls_workload["cost_per_unit"]
        ai_task_mapping = workload_def.get("ai_task_mapping", "classification")
        ai_cost_info = get_ai_cost_from_cpi(cpi_data, ai_task_mapping)
        ai_cost = ai_cost_info["cost_per_unit"]

        annual_units = volume_info.get("annual_units", 1_000_000)
        displacement = compute_displacement(human_cost, ai_cost, annual_units)

        sub_rate = fpds_workload.get("substitution_rate_proxy", 0.0)

        print(f"  Human cost: ${human_cost:.4f}/unit | AI cost: ${ai_cost:.6f}/unit")
        print(f"  Displacement: ${displacement['displacement_per_unit']:.4f}/unit "
              f"({displacement['displacement_pct']:.1f}%)")
        print(f"  Sub rate: {sub_rate:.3f} | Absorption: {absorption_info.get('classification', 'unknown')}")

        workload_results[workload_id] = {
            "id": workload_id,
            "name": workload_def["name"],
            "soc_code": workload_def["soc_code"],
            "soc_title": workload_def["soc_title"],
            "human_cost": {
                "cost_per_unit": human_cost,
                "cost_per_unit_unit": "USD per workload unit",
                "fully_loaded_hourly": bls_workload.get("fully_loaded_hourly"),
                "annual_wage": bls_workload.get("annual_wage"),
                "minutes_per_unit": bls_workload.get("minutes_per_unit"),
                "source": bls_workload.get("wage_source")
            },
            "ai_cost": {
                "cost_per_unit": ai_cost,
                "cost_per_unit_unit": "USD per workload unit",
                "task_type": ai_task_mapping,
                "source": ai_cost_info.get("source")
            },
            "displacement": displacement,
            "substitution_rate": {
                "rate": sub_rate,
                "rate_pct": round(sub_rate * 100, 2),
                "source": "FPDS procurement signal (contractor spend decline + AI procurement growth)",
                "current_fy_spend": fpds_workload.get("current_spend"),
                "yoy_change_pct": fpds_workload.get("yoy_change_pct"),
                "ai_procurement_current": fpds_workload.get("ai_procurement_current"),
                "ai_procurement_prior": fpds_workload.get("ai_procurement_prior"),
                "ai_procurement_yoy": fpds_workload.get("ai_procurement_yoy_change_pct"),
                "ai_growth_rate": fpds_workload.get("ai_growth_rate_used"),
                "ai_growth_source": fpds_workload.get("ai_growth_source"),
            },
            "absorption": {
                "classification": absorption_info.get("classification", "unknown"),
                "reasoning": absorption_info.get("reasoning", ""),
                "color": ABSORPTION_COLOR.get(absorption_info.get("classification", ""), "accent"),
                "fte_current": opm_workload.get("fte_current"),
                "fte_prior": opm_workload.get("fte_prior"),
                "fte_delta_pct": opm_workload.get("fte_delta_pct"),
                "absorption_quant": opm_workload.get("absorption_quant"),
                "primary_agencies": opm_workload.get("primary_agencies"),
                "fte_source": opm_workload.get("source"),
            },
            "volume": {
                "annual_units": annual_units,
                "source": volume_info.get("source", "estimated")
            }
        }

    composite = calculate_composite_ldi(workload_results)

    # Per-agency AI procurement rollup (top N for surface area)
    ai_summary = fpds_data.get("ai_procurement_summary", {})
    agency_rollup_full = ai_summary.get("by_awarding_agency", []) or []
    top_agencies = agency_rollup_full[:10]

    output = {
        "date": date_str,
        "generated_at": now.isoformat(),
        "substitution_rate": composite["substitution_rate"],
        "substitution_rate_interpretation": composite["substitution_rate_interpretation"],
        "cost_displacement": {
            "annual_usd": composite["cost_displacement_usd_annual"],
            "label": composite["cost_displacement_label"],
            "interpretation": composite["cost_displacement_interpretation"],
            "avg_cost_ratio": composite["avg_cost_ratio"],
            "total_human_cost_annual": composite["total_human_cost_annual"],
            "total_ai_cost_annual": composite["total_ai_cost_annual"]
        },
        "summary": {
            "substitution_rate_pct": composite["substitution_rate"],
            "cost_displacement_usd": composite["cost_displacement_usd_annual"],
            "avg_cost_ratio": composite["avg_cost_ratio"],
            "pilot_workload_count": composite["pilot_workload_count"],
            "total_volume_units": composite["total_volume_units"],
            "avg_human_cost_per_unit": round(
                sum(w["human_cost"]["cost_per_unit"] for w in workload_results.values())
                / len(workload_results), 4
            ) if workload_results else 0,
            "avg_ai_cost_per_unit": round(
                sum(w["ai_cost"]["cost_per_unit"] for w in workload_results.values())
                / len(workload_results), 6
            ) if workload_results else 0
        },
        "workloads": workload_results,
        "ai_procurement_by_agency": {
            "fy_current": fpds_data.get("fiscal_years", {}).get("current"),
            "fy_prior": fpds_data.get("fiscal_years", {}).get("prior"),
            "total_agencies": len(agency_rollup_full),
            "top_agencies": top_agencies,
            "source": "USAspending.gov spending_by_award keyword search, deduped per award_id, grouped by Awarding Agency",
        },
        "meta": {
            "sources": ["BLS OEWS", "BLS ECEC", "USAspending.gov FPDS", "OPM FedScope", "Compute CPI basket"],
            "pilot_workload_ids": list(workload_results.keys()),
            "cpi_value": cpi_data.get("compute_cpi", {}).get("value"),
            "methodology_version": "1.1",
            "note": (
                "substitution_rate is the observable signal (FPDS procurement proxy). "
                "cost_displacement is structural/technical potential at 100% substitution — "
                "not realized savings. These are distinct measurements."
            )
        }
    }

    latest_path = LDI_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[LDI] Saved latest.json to {latest_path}")

    historical_path = LDI_DIR / "historical.json"
    if historical_path.exists():
        with open(historical_path) as f:
            history = json.load(f)
    else:
        history = {"entries": []}

    history_entry = {
        "date": date_str,
        "substitution_rate_pct": composite["substitution_rate"],
        "cost_displacement_usd": composite["cost_displacement_usd_annual"],
        "avg_cost_ratio": composite["avg_cost_ratio"],
        "workload_count": composite["pilot_workload_count"],
        "workloads": {
            wid: {
                "substitution_rate_pct": w["substitution_rate"]["rate_pct"],
                "human_cost_per_unit": w["human_cost"]["cost_per_unit"],
                "ai_cost_per_unit": w["ai_cost"]["cost_per_unit"],
                "absorption": w["absorption"]["classification"]
            }
            for wid, w in workload_results.items()
        }
    }

    existing_dates = [e["date"] for e in history.get("entries", [])]
    if date_str in existing_dates:
        history["entries"] = [
            history_entry if e["date"] == date_str else e
            for e in history["entries"]
        ]
        print(f"[LDI] Updated historical entry for {date_str}")
    else:
        history["entries"].append(history_entry)
        print(f"[LDI] Added historical entry for {date_str}")

    history["entries"].sort(key=lambda x: x["date"])

    with open(historical_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[LDI] Saved historical.json ({len(history['entries'])} entries)")

    print("\n--- Computing derived signals (velocity, acceleration) ---")
    run_derive()

    print("\n" + "=" * 60)
    print("  LDI CALCULATION COMPLETE")
    print("=" * 60)
    print(f"\nSubstitution rate (FPDS proxy): {composite['substitution_rate']:.2f}%")
    print(f"Structural cost displacement: ${composite['cost_displacement_usd_annual']:,.0f}/yr (potential at 100% substitution)")
    print(f"Avg cost ratio: {composite['avg_cost_ratio']:.0f}x (human vs AI per unit)")
    print(f"Workloads computed: {composite['pilot_workload_count']}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Labor Displacement Index")
    parser.add_argument("--refresh", action="store_true", help="Force refresh of BLS and FPDS data")
    args = parser.parse_args()
    run_pipeline(force_refresh=args.refresh)
