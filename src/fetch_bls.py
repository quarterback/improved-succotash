#!/usr/bin/env python3
"""
BLS Human Cost Pipeline for LDI

Fetches OEWS wage data from BLS public API for the SOC codes in the workload map.
Falls back to static May 2023 OEWS data if the API is unavailable.

Raw API request payload and full HTTP response body are cached in bls_raw.json.
Computed costs ($/unit per workload) are in bls_output.json.

Data sources:
- BLS OEWS (May 2023): https://www.bls.gov/oes/tables.htm
- BLS ECEC (Q4 2024): https://www.bls.gov/ect/

Computes:
- Fully loaded $/hour = wage × ECEC overhead factor (1.385 × 1.05)
- $/unit = fully loaded hourly rate / 60 × minutes per unit (from O*NET)
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LDI_DIR = DATA_DIR / "ldi"

BLS_API_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

OEWS_SERIES_MAP = {
    "13-1041": "OEWS000000000000013104100000000W",
    "13-1031": "OEWS000000000000013103100000000W",
    "43-4051": "OEWS000000000000043405100000000W",
    "23-2093": "OEWS000000000000023209300000000W",
    "15-1232": "OEWS000000000000015123200000000W",
}

STATIC_OEWS_WAGES = {
    "13-1041": {"annual_wage": 47_930, "year": "2023",
                "source_url": "https://www.bls.gov/oes/tables.htm",
                "source_doc": "BLS OEWS May 2023 — National estimates for SOC 13-1041 (Eligibility Interviewers, Government Programs)"},
    "13-1031": {"annual_wage": 72_040, "year": "2023",
                "source_url": "https://www.bls.gov/oes/tables.htm",
                "source_doc": "BLS OEWS May 2023 — National estimates for SOC 13-1031 (Claims Adjusters, Examiners, and Investigators)"},
    "43-4051": {"annual_wage": 42_040, "year": "2023",
                "source_url": "https://www.bls.gov/oes/tables.htm",
                "source_doc": "BLS OEWS May 2023 — National estimates for SOC 43-4051 (Customer Service Representatives)"},
    "23-2093": {"annual_wage": 58_630, "year": "2023",
                "source_url": "https://www.bls.gov/oes/tables.htm",
                "source_doc": "BLS OEWS May 2023 — National estimates for SOC 23-2093 (Title Examiners, Abstractors, and Searchers)"},
    "15-1232": {"annual_wage": 60_530, "year": "2023",
                "source_url": "https://www.bls.gov/oes/tables.htm",
                "source_doc": "BLS OEWS May 2023 — National estimates for SOC 15-1232 (Computer User Support Specialists)"},
}

ECEC_BENEFIT_MULTIPLIER = 1.385
ECEC_OVERHEAD_FACTOR = 1.05
ECEC_COMBINED = round(ECEC_BENEFIT_MULTIPLIER * ECEC_OVERHEAD_FACTOR, 4)

ECEC_SOURCE = (
    f"BLS ECEC Q4 2024 (https://www.bls.gov/ect/): Government worker wages are approximately "
    f"72.2% of total compensation (implies benefit multiplier = 1/0.722 = {ECEC_BENEFIT_MULTIPLIER}). "
    f"Additional overhead factor {ECEC_OVERHEAD_FACTOR}× applied for training, equipment, and facilities. "
    f"Combined fully loaded factor = {ECEC_BENEFIT_MULTIPLIER} × {ECEC_OVERHEAD_FACTOR} = {ECEC_COMBINED}. "
    f"Note: ECEC parameters are static constants derived from published ECEC tables, not fetched via API. "
    f"ECEC API does not provide a machine-readable series for the government benefit cost share."
)


def fetch_bls_oews(soc_codes: list, api_key: str = None) -> dict:
    """
    Attempt to fetch OEWS annual wage data from the BLS public API.

    Returns a dict with:
      - 'wage_data': dict keyed by SOC code (or {} on failure)
      - 'api_log': dict with the raw request payload and full response body
    """
    api_log = {
        "endpoint": BLS_API_BASE,
        "request_at": datetime.now(timezone.utc).isoformat(),
        "status": None,
        "status_code": None,
        "request_payload": None,
        "response_body": None,
        "error": None
    }

    series_ids = []
    soc_to_series = {}
    for soc in soc_codes:
        if soc in OEWS_SERIES_MAP:
            sid = OEWS_SERIES_MAP[soc]
            series_ids.append(sid)
            soc_to_series[sid] = soc

    payload = {
        "seriesid": series_ids,
        "startyear": "2023",
        "endyear": "2024"
    }
    if api_key:
        payload["registrationkey"] = api_key

    api_log["request_payload"] = payload

    try:
        response = requests.post(
            BLS_API_BASE, json=payload,
            headers={"Content-type": "application/json"},
            timeout=30
        )
        api_log["status_code"] = response.status_code
        api_log["response_body"] = response.json()

        data = api_log["response_body"]
        api_status = data.get("status")
        api_log["status"] = api_status

        if api_status != "REQUEST_SUCCEEDED":
            api_log["error"] = f"API returned status: {api_status}"
            msgs = data.get("message", [])
            if msgs:
                api_log["api_messages"] = msgs
            print(f"[BLS] API returned non-success: {api_status} — {msgs}")
            return {"wage_data": {}, "api_log": api_log}

        wage_data = {}
        for series in data.get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            soc = soc_to_series.get(sid)
            if not soc:
                continue
            for entry in series.get("data", []):
                if entry.get("period") == "A01":
                    try:
                        annual_wage = float(entry["value"].replace(",", ""))
                        wage_data[soc] = {
                            "annual_wage": annual_wage,
                            "year": entry.get("year"),
                            "source": "BLS OEWS API",
                            "series_id": sid
                        }
                        print(f"[BLS-OEWS] {soc}: ${annual_wage:,.0f}/year (API)")
                    except (ValueError, KeyError) as e:
                        print(f"[BLS-OEWS] Parse error for {soc}: {e}")
                    break

        api_log["wage_records_found"] = len(wage_data)
        if not wage_data:
            api_log["error"] = (
                "API returned REQUEST_SUCCEEDED but no annual (period=A01) data found "
                "for the requested OEWS series IDs. Series ID format may need updating."
            )
        return {"wage_data": wage_data, "api_log": api_log}

    except Exception as e:
        api_log["status"] = "ERROR"
        api_log["error"] = str(e)
        print(f"[BLS] API exception: {e}")
        return {"wage_data": {}, "api_log": api_log}


def compute_fully_loaded_costs(oews_data: dict, workload_map: dict) -> dict:
    """
    Compute fully loaded $/hour and $/unit for each workload.

    Formula:
    - hourly_wage = annual_wage / 2080 (assuming 2080 work hours/year)
    - fully_loaded_hourly = hourly_wage × ECEC_BENEFIT_MULTIPLIER × ECEC_OVERHEAD_FACTOR
    - cost_per_unit = fully_loaded_hourly / 60 × minutes_per_unit
    """
    output = {}
    for workload_id, workload in workload_map.get("workloads", {}).items():
        soc = workload["soc_code"]
        if soc not in oews_data:
            print(f"[BLS] No wage data for {workload_id} (SOC {soc}), skipping")
            continue

        wage_info = oews_data[soc]
        annual_wage = wage_info["annual_wage"]
        minutes_per_unit = workload["minutes_per_unit"]

        hourly_wage = annual_wage / 2080
        fully_loaded_hourly = hourly_wage * ECEC_BENEFIT_MULTIPLIER * ECEC_OVERHEAD_FACTOR
        cost_per_unit = (fully_loaded_hourly / 60) * minutes_per_unit

        output[workload_id] = {
            "soc_code": soc,
            "soc_title": workload["soc_title"],
            "annual_wage": annual_wage,
            "wage_year": wage_info["year"],
            "wage_source": wage_info.get("source", "BLS OEWS"),
            "hourly_wage": round(hourly_wage, 4),
            "ecec_multiplier": ECEC_BENEFIT_MULTIPLIER,
            "overhead_factor": ECEC_OVERHEAD_FACTOR,
            "ecec_source": ECEC_SOURCE,
            "fully_loaded_hourly": round(fully_loaded_hourly, 4),
            "minutes_per_unit": minutes_per_unit,
            "cost_per_unit": round(cost_per_unit, 4),
            "cost_per_unit_unit": "USD per workload unit",
            "methodology": (
                f"hourly_wage = ${annual_wage:,} / 2080 = ${hourly_wage:.4f}; "
                f"fully_loaded = ${hourly_wage:.4f} × {ECEC_BENEFIT_MULTIPLIER} (ECEC) "
                f"× {ECEC_OVERHEAD_FACTOR} (overhead) = ${fully_loaded_hourly:.4f}/hr; "
                f"cost_per_unit = ${fully_loaded_hourly:.4f} / 60 × {minutes_per_unit} min = ${cost_per_unit:.4f}"
            )
        }
        print(f"[BLS] {workload_id}: ${cost_per_unit:.4f}/unit (${fully_loaded_hourly:.2f}/hr fully loaded)")

    return output


def run():
    """Main entry point for BLS human cost pipeline."""
    print("=" * 60)
    print("  BLS HUMAN COST PIPELINE")
    print("=" * 60)

    LDI_DIR.mkdir(parents=True, exist_ok=True)

    workload_map_path = LDI_DIR / "workload_map.json"
    if not workload_map_path.exists():
        print(f"[BLS] ERROR: workload_map.json not found at {workload_map_path}")
        return {}

    with open(workload_map_path) as f:
        workload_map = json.load(f)

    soc_codes = list({w["soc_code"] for w in workload_map["workloads"].values()})
    print(f"\n[BLS] Attempting BLS OEWS API for SOC codes: {soc_codes}")

    api_result = fetch_bls_oews(soc_codes)
    wage_data = api_result["wage_data"]
    api_log = api_result["api_log"]

    fallback_used = not bool(wage_data)
    if fallback_used:
        print("[BLS] API unavailable — using static OEWS May 2023 data")
        for soc, info in STATIC_OEWS_WAGES.items():
            wage_data[soc] = {
                "annual_wage": info["annual_wage"],
                "year": info["year"],
                "source": "BLS OEWS May 2023 (static fallback)",
                "source_url": info["source_url"],
                "source_doc": info["source_doc"]
            }
            print(f"[BLS-OEWS] {soc}: ${info['annual_wage']:,.0f}/year (static May 2023)")

    raw_output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "BLS OEWS",
        "api_attempt": {
            "endpoint": BLS_API_BASE,
            "series_ids_requested": OEWS_SERIES_MAP,
            "api_log": api_log,
            "api_reachable": api_log.get("status_code") is not None,
            "api_data_returned": not fallback_used,
            "note": (
                "api_reachable=True means HTTP connection succeeded. "
                "api_data_returned=True means wage values were extracted from the response. "
                "If reachable but no data, the OEWS series ID format may need updating."
            )
        },
        "static_fallback": {
            "used": fallback_used,
            "reason": "BLS API unavailable or returned non-success" if fallback_used else "not used",
            "data": STATIC_OEWS_WAGES if fallback_used else {},
            "citation": "Bureau of Labor Statistics, Occupational Employment and Wage Statistics, May 2023. https://www.bls.gov/oes/tables.htm"
        },
        "ecec_parameters": {
            "multiplier": ECEC_BENEFIT_MULTIPLIER,
            "overhead": ECEC_OVERHEAD_FACTOR,
            "combined_factor": round(ECEC_BENEFIT_MULTIPLIER * ECEC_OVERHEAD_FACTOR, 4),
            "source": ECEC_SOURCE
        },
        "wage_data_used": wage_data
    }

    raw_path = LDI_DIR / "bls_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n[BLS] Saved raw data (with API log) to {raw_path}")

    print("\n--- Computing fully loaded costs ---")
    cost_output = compute_fully_loaded_costs(wage_data, workload_map)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "BLS OEWS + ECEC",
        "api_data": not fallback_used,
        "ecec_multiplier": ECEC_BENEFIT_MULTIPLIER,
        "ecec_overhead": ECEC_OVERHEAD_FACTOR,
        "ecec_notes": ECEC_SOURCE,
        "workloads": cost_output
    }

    output_path = LDI_DIR / "bls_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[BLS] Saved cost output to {output_path}")

    print("\n" + "=" * 60)
    print("  BLS PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nProcessed {len(cost_output)} workloads")
    for wid, data in cost_output.items():
        print(f"  {wid}: ${data['cost_per_unit']:.4f}/unit")

    return output


if __name__ == "__main__":
    run()
