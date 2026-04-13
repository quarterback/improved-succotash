#!/usr/bin/env python3
"""
FPDS Procurement Pipeline for LDI

Pulls contract spend from USAspending.gov's public API, aggregated by PSC codes
in the workload map. Computes year-over-year contractor spend change per workload
category as the procurement substitution signal.

Raw API responses (actual HTTP payload + response) are cached in fpds_raw.json.
Derived substitution signals are in fpds_output.json.

IMPORTANT MEASUREMENT LIMITATIONS:
- PSC codes are broad categories, not workload-specific. R499 ($32B/yr) covers
  all "Professional Support: Other" across the federal government. Inferring
  workload-specific substitution from PSC-level data is a rough proxy.
- The AI procurement component (what portion of decline is AI-driven) requires
  keyword-level contract title/description analysis that is not in this pipeline.
  The contractor spend signal (60% weight) is from real API data.
  The AI component is currently zeroed out pending keyword-search implementation.
"""

import json
import requests
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"

USASPENDING_BASE = "https://api.usaspending.gov/api/v2"

CURRENT_FY = 2024
PRIOR_FY = 2023


def fetch_psc_totals_for_year(fiscal_year: int) -> dict:
    """
    Fetch total obligated spending per PSC code from USAspending.gov
    for the given fiscal year (Oct 1 to Sep 30).

    Returns dict keyed by PSC code with amount, name, and request metadata.
    Paginates up to 5 pages (500 PSC codes) to find pilot codes.
    """
    url = f"{USASPENDING_BASE}/search/spending_by_category/psc/"
    start = f"{fiscal_year - 1}-10-01"
    end = f"{fiscal_year}-09-30"

    results_by_psc = {}
    api_responses = []
    page = 1

    TARGET_PSCS = {"D302", "D399", "R408", "R499", "Q999", "U099", "F999"}

    print(f"[FPDS] Fetching FY{fiscal_year} PSC spend (looking for: {TARGET_PSCS})")

    while True:
        payload = {
            "filters": {
                "time_period": [{"start_date": start, "end_date": end}]
            },
            "limit": 100,
            "page": page
        }

        raw_record = {
            "request": {
                "url": url,
                "method": "POST",
                "payload": payload,
                "fiscal_year": fiscal_year
            }
        }

        try:
            response = requests.post(
                url, json=payload, timeout=30,
                headers={"Content-Type": "application/json"}
            )
            raw_record["response"] = {
                "status_code": response.status_code,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "body": response.json() if response.ok else {"error": response.text[:500]}
            }
            api_responses.append(raw_record)

            if not response.ok:
                print(f"[FPDS] FY{fiscal_year} page {page} error: {response.status_code}")
                break

            data = response.json()
            page_results = data.get("results", [])

            for item in page_results:
                code = item.get("code")
                if code in TARGET_PSCS:
                    results_by_psc[code] = {
                        "amount": item.get("amount", 0),
                        "name": item.get("name", ""),
                        "found_on_page": page
                    }

            has_next = data.get("page_metadata", {}).get("hasNext", False)
            found_all = TARGET_PSCS.issubset(set(results_by_psc.keys()))

            if not has_next or found_all or page >= 5:
                break
            page += 1
            time.sleep(0.25)

        except Exception as e:
            raw_record["response"] = {
                "status_code": None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
            api_responses.append(raw_record)
            print(f"[FPDS] FY{fiscal_year} page {page} exception: {e}")
            break

    found = set(results_by_psc.keys())
    missing = TARGET_PSCS - found
    if missing:
        print(f"[FPDS] FY{fiscal_year}: {len(found)} of {len(TARGET_PSCS)} PSC codes found. Missing: {missing}")
    else:
        print(f"[FPDS] FY{fiscal_year}: all {len(TARGET_PSCS)} target PSC codes found")

    return {
        "fiscal_year": fiscal_year,
        "psc_totals": results_by_psc,
        "api_responses": api_responses
    }


def aggregate_psc_spend(psc_codes: list, psc_totals: dict) -> float:
    """Sum obligated amounts across a list of PSC codes."""
    return sum(psc_totals.get(psc, {}).get("amount", 0) for psc in psc_codes)


def compute_substitution_rate(current_spend: float, prior_spend: float) -> float:
    """
    Compute the substitution rate proxy from contractor spend YoY change.

    Formula (contractor-spend component only):
        contractor_decline_rate = |Δ spend| / prior_spend (only if declining)
        sub_rate = contractor_decline_rate × 0.6

    NOTE: The AI procurement growth component (0.4 weight) is set to 0 pending
    keyword-level contract search implementation. PSC-level data cannot isolate
    AI vs. non-AI procurement without contract title/description filtering.
    """
    if prior_spend <= 0:
        return 0.0

    yoy_pct = ((current_spend - prior_spend) / prior_spend) * 100
    decline_rate = max(0.0, -yoy_pct / 100)

    sub_rate = decline_rate * 0.6
    return round(min(sub_rate, 1.0), 4)


def run():
    """Main entry point for FPDS procurement pipeline."""
    print("=" * 60)
    print("  FPDS PROCUREMENT PIPELINE")
    print("=" * 60)

    LDI_DIR.mkdir(parents=True, exist_ok=True)

    workload_map_path = LDI_DIR / "workload_map.json"
    if not workload_map_path.exists():
        print(f"[FPDS] ERROR: workload_map.json not found at {workload_map_path}")
        return {}

    with open(workload_map_path) as f:
        workload_map = json.load(f)

    fy2024_result = fetch_psc_totals_for_year(CURRENT_FY)
    fy2023_result = fetch_psc_totals_for_year(PRIOR_FY)

    psc_2024 = fy2024_result["psc_totals"]
    psc_2023 = fy2023_result["psc_totals"]

    raw_output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "USAspending.gov FPDS — spending_by_category/psc endpoint",
        "api_endpoint": f"{USASPENDING_BASE}/search/spending_by_category/psc/",
        "fiscal_years": {"current": CURRENT_FY, "prior": PRIOR_FY},
        "notes": (
            "Raw API response payloads below. Each api_responses entry contains the full "
            "request payload and HTTP response body as returned by USAspending.gov. "
            "The psc_totals dict shows total obligated spending per PSC code for FY2024 and FY2023. "
            "These are government-wide totals for each PSC category, not workload-specific."
        ),
        "fy2024": {
            "psc_totals": psc_2024,
            "api_responses": fy2024_result["api_responses"]
        },
        "fy2023": {
            "psc_totals": psc_2023,
            "api_responses": fy2023_result["api_responses"]
        }
    }

    raw_path = LDI_DIR / "fpds_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n[FPDS] Saved raw API responses to {raw_path}")

    print("\n--- Computing per-workload substitution signals ---")
    workload_data = {}

    for workload_id, workload in workload_map.get("workloads", {}).items():
        psc_codes = workload.get("fpds_psc_codes", [])

        current_spend = aggregate_psc_spend(psc_codes, psc_2024)
        prior_spend = aggregate_psc_spend(psc_codes, psc_2023)

        live_signal = current_spend > 0 or prior_spend > 0

        yoy_pct = 0.0
        if prior_spend > 0:
            yoy_pct = ((current_spend - prior_spend) / prior_spend) * 100

        sub_rate = compute_substitution_rate(current_spend, prior_spend)

        missing_pscs = [p for p in psc_codes if p not in psc_2024]

        workload_data[workload_id] = {
            "psc_codes": psc_codes,
            "psc_codes_resolved": {
                p: {"fy2024": psc_2024.get(p, {}).get("amount", 0), "fy2023": psc_2023.get(p, {}).get("amount", 0)}
                for p in psc_codes
            },
            "current_fy": CURRENT_FY,
            "prior_fy": PRIOR_FY,
            "current_spend": round(current_spend, 2),
            "prior_spend": round(prior_spend, 2),
            "yoy_change_pct": round(yoy_pct, 2),
            "ai_procurement_spend": None,
            "ai_procurement_notes": (
                "AI-specific procurement share not computable from PSC-level totals. "
                "Requires contract-level keyword search (award title/description) on "
                "USAspending.gov — not yet implemented. AI component set to 0 in sub_rate."
            ),
            "substitution_rate_proxy": sub_rate,
            "source": "USAspending.gov API (live)" if live_signal else "no data",
            "missing_psc_codes": missing_pscs,
            "substitution_rate_methodology": (
                "sub_rate = contractor_decline_rate × 0.6 + 0 (AI component pending). "
                "contractor_decline_rate = max(0, -yoy_change_pct/100). "
                "Only applied when contractor spend is declining. "
                "Note: PSC categories are government-wide and broader than the specific workloads."
            )
        }

        print(f"[FPDS] {workload_id}: current=${current_spend:,.0f}, "
              f"prior=${prior_spend:,.0f}, yoy={yoy_pct:.1f}%, sub_rate={sub_rate:.3f}"
              + (" [missing some PSCs]" if missing_pscs else ""))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "USAspending.gov FPDS",
        "fiscal_years": {"current": CURRENT_FY, "prior": PRIOR_FY},
        "psc_spend_summary": {
            "fy2024": {code: info.get("amount", 0) for code, info in psc_2024.items()},
            "fy2023": {code: info.get("amount", 0) for code, info in psc_2023.items()}
        },
        "workloads": workload_data
    }

    output_path = LDI_DIR / "fpds_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[FPDS] Saved output to {output_path}")

    print("\n" + "=" * 60)
    print("  FPDS PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nProcessed {len(workload_data)} workloads")
    for wid, data in workload_data.items():
        print(f"  {wid}: sub_rate={data['substitution_rate_proxy']:.3f} "
              f"(yoy={data['yoy_change_pct']:.1f}%)")

    return output


if __name__ == "__main__":
    run()
