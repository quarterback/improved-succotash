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

# Keywords used to isolate AI-related contract awards in USAspending.gov.
# Search runs against award title and description. Tuned to be specific enough
# to reduce false positives (e.g. plain "AI" alone matches too many things —
# we require it to appear with a related token, or use multi-word phrases).
AI_KEYWORD_QUERIES = [
    "artificial intelligence",
    "machine learning",
    "large language model",
    "generative ai",
    "natural language processing",
    "OpenAI",
    "Anthropic",
    "Palantir AIP",
    "ServiceNow AI",
    "AI chatbot",
    "AI assistant",
    "AI platform",
    "LLM",
]

# Keyword-to-workload mapping. When an AI-related award matches one of these
# keywords, we attribute it to the listed workload(s) for the per-workload
# AI procurement signal. Keywords not in the map count toward the global
# AI total but are not attributed to a specific workload.
AI_KEYWORD_WORKLOAD_HINTS = {
    "snap eligibility": ["snap_eligibility"],
    "supplemental nutrition": ["snap_eligibility"],
    "unemployment insurance": ["unemployment_claims"],
    "ui adjudication": ["unemployment_claims"],
    "claims adjudication": ["unemployment_claims"],
    "contact center": ["call_center_triage"],
    "call center": ["call_center_triage"],
    "ivr": ["call_center_triage"],
    "chatbot": ["call_center_triage", "it_help_desk"],
    "document summarization": ["document_summarization"],
    "case file": ["document_summarization"],
    "claims processing": ["unemployment_claims", "document_summarization"],
    "help desk": ["it_help_desk"],
    "tier 1 support": ["it_help_desk"],
    "itsm": ["it_help_desk"],
    "servicenow": ["it_help_desk"],
}


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


def fetch_ai_award_totals_for_year(fiscal_year: int, max_pages_per_query: int = 3) -> dict:
    """
    Search USAspending.gov for AI-related contract awards in the given fiscal year.

    Hits the spending_by_award endpoint with each AI keyword in
    AI_KEYWORD_QUERIES, paginates a few pages, and aggregates obligated
    amounts. Awards are deduped by award_id so a contract that matches
    multiple keywords is counted once.

    Returns:
        {
            "fiscal_year": int,
            "total_obligated": float,
            "award_count": int,
            "by_keyword": {keyword: {"obligated": float, "awards": int}},
            "workload_attribution": {workload_id: float},
            "api_responses": [...]
        }
    """
    url = f"{USASPENDING_BASE}/search/spending_by_award/"
    start = f"{fiscal_year - 1}-10-01"
    end = f"{fiscal_year}-09-30"

    print(f"[FPDS-AI] Searching FY{fiscal_year} for AI-related awards "
          f"({len(AI_KEYWORD_QUERIES)} keywords)")

    seen_award_ids = set()
    awards_by_id = {}  # award_id -> {amount, matched_keywords}
    by_keyword = {}
    api_responses = []

    for keyword in AI_KEYWORD_QUERIES:
        kw_obligated = 0.0
        kw_count = 0
        page = 1

        while page <= max_pages_per_query:
            payload = {
                "filters": {
                    "keywords": [keyword],
                    "time_period": [{"start_date": start, "end_date": end}],
                    "award_type_codes": ["A", "B", "C", "D"],  # contract IDV/award types
                },
                "fields": [
                    "Award ID",
                    "Recipient Name",
                    "Award Amount",
                    "Description",
                    "Awarding Agency",
                ],
                "limit": 100,
                "page": page,
                "sort": "Award Amount",
                "order": "desc",
            }

            raw_record = {
                "request": {
                    "url": url,
                    "method": "POST",
                    "payload": payload,
                    "fiscal_year": fiscal_year,
                    "keyword": keyword,
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
                    "result_count": None,
                }

                if not response.ok:
                    raw_record["response"]["error"] = response.text[:300]
                    api_responses.append(raw_record)
                    print(f"[FPDS-AI] FY{fiscal_year} '{keyword}' page {page} "
                          f"error: {response.status_code}")
                    break

                data = response.json()
                results = data.get("results", [])
                raw_record["response"]["result_count"] = len(results)
                api_responses.append(raw_record)

                for award in results:
                    award_id = award.get("Award ID") or award.get("internal_id")
                    amount = award.get("Award Amount") or 0
                    if not award_id:
                        continue

                    description = (award.get("Description") or "").lower()
                    recipient = (award.get("Recipient Name") or "").lower()

                    kw_obligated += amount
                    kw_count += 1

                    if award_id not in seen_award_ids:
                        seen_award_ids.add(award_id)
                        awards_by_id[award_id] = {
                            "amount": amount,
                            "matched_keywords": [keyword],
                            "description": description,
                            "recipient": recipient,
                        }
                    else:
                        awards_by_id[award_id]["matched_keywords"].append(keyword)

                has_next = data.get("page_metadata", {}).get("hasNext", False)
                if not has_next or len(results) == 0:
                    break
                page += 1
                time.sleep(0.2)

            except Exception as e:
                raw_record["response"] = {
                    "status_code": None,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                }
                api_responses.append(raw_record)
                print(f"[FPDS-AI] FY{fiscal_year} '{keyword}' exception: {e}")
                break

        by_keyword[keyword] = {"obligated": round(kw_obligated, 2), "awards": kw_count}

    # Total obligated, deduplicated by award_id
    total_obligated = sum(a["amount"] for a in awards_by_id.values())

    # Per-workload attribution: scan deduped award descriptions for hint terms
    workload_attribution = {}
    for award in awards_by_id.values():
        text = (award["description"] + " " + award["recipient"]).lower()
        for hint, workload_ids in AI_KEYWORD_WORKLOAD_HINTS.items():
            if hint in text:
                for wid in workload_ids:
                    workload_attribution[wid] = workload_attribution.get(wid, 0) + award["amount"]

    print(f"[FPDS-AI] FY{fiscal_year}: ${total_obligated:,.0f} across "
          f"{len(awards_by_id)} unique AI-related awards "
          f"({sum(b['awards'] for b in by_keyword.values())} keyword hits)")

    return {
        "fiscal_year": fiscal_year,
        "total_obligated": round(total_obligated, 2),
        "award_count": len(awards_by_id),
        "by_keyword": by_keyword,
        "workload_attribution": {k: round(v, 2) for k, v in workload_attribution.items()},
        "api_responses": api_responses,
    }


def compute_ai_growth_rate(current_ai: float, prior_ai: float) -> float:
    """
    Normalize AI procurement growth into a 0–1 rate.

    Returns YoY growth rate (positive only) capped at 1.0. A doubling of
    AI spend produces 1.0; a 50% increase produces 0.5; a flat or
    declining year produces 0.
    """
    if prior_ai <= 0:
        # If there was effectively no AI procurement last year, any current
        # spend is a step change. Use a saturating function so brand-new
        # AI categories don't trivially produce 1.0 from rounding noise.
        return 1.0 if current_ai > 100_000 else 0.0
    growth = (current_ai - prior_ai) / prior_ai
    return round(min(max(growth, 0.0), 1.0), 4)


def compute_substitution_rate(
    current_spend: float,
    prior_spend: float,
    ai_growth_rate: float = 0.0,
) -> float:
    """
    Compute the substitution rate proxy from two signals:

        sub_rate = contractor_decline_rate × 0.6 + ai_growth_rate × 0.4

    contractor_decline_rate (only if declining):
        max(0, -yoy_change_pct / 100), capped at 1.0

    ai_growth_rate:
        YoY growth in AI-keyword-matched contract obligations, capped at 1.0.
        Computed by compute_ai_growth_rate() from spending_by_award searches.
    """
    if prior_spend <= 0:
        contractor_component = 0.0
    else:
        yoy_pct = ((current_spend - prior_spend) / prior_spend) * 100
        contractor_component = max(0.0, -yoy_pct / 100)
        contractor_component = min(contractor_component, 1.0)

    ai_component = max(0.0, min(ai_growth_rate, 1.0))

    sub_rate = contractor_component * 0.6 + ai_component * 0.4
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

    # AI procurement keyword search (fills the 0.4 weight in the sub_rate formula)
    ai_2024 = fetch_ai_award_totals_for_year(CURRENT_FY)
    ai_2023 = fetch_ai_award_totals_for_year(PRIOR_FY)
    ai_total_growth = compute_ai_growth_rate(
        ai_2024["total_obligated"], ai_2023["total_obligated"]
    )

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
        },
        "ai_keyword_search": {
            "fy2024": {
                "total_obligated": ai_2024["total_obligated"],
                "award_count": ai_2024["award_count"],
                "by_keyword": ai_2024["by_keyword"],
                "workload_attribution": ai_2024["workload_attribution"],
                "api_responses": ai_2024["api_responses"],
            },
            "fy2023": {
                "total_obligated": ai_2023["total_obligated"],
                "award_count": ai_2023["award_count"],
                "by_keyword": ai_2023["by_keyword"],
                "workload_attribution": ai_2023["workload_attribution"],
                "api_responses": ai_2023["api_responses"],
            },
            "yoy_growth_rate": ai_total_growth,
            "keywords_used": AI_KEYWORD_QUERIES,
            "workload_attribution_hints": AI_KEYWORD_WORKLOAD_HINTS,
            "endpoint": f"{USASPENDING_BASE}/search/spending_by_award/",
        },
    }

    raw_path = LDI_DIR / "fpds_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_output, f, indent=2)
    print(f"\n[FPDS] Saved raw API responses to {raw_path}")

    print("\n--- Computing per-workload substitution signals ---")
    workload_data = {}

    ai_attr_2024 = ai_2024["workload_attribution"]
    ai_attr_2023 = ai_2023["workload_attribution"]

    for workload_id, workload in workload_map.get("workloads", {}).items():
        psc_codes = workload.get("fpds_psc_codes", [])

        current_spend = aggregate_psc_spend(psc_codes, psc_2024)
        prior_spend = aggregate_psc_spend(psc_codes, psc_2023)

        live_signal = current_spend > 0 or prior_spend > 0

        yoy_pct = 0.0
        if prior_spend > 0:
            yoy_pct = ((current_spend - prior_spend) / prior_spend) * 100

        # AI procurement signal — per-workload attribution if keywords matched,
        # otherwise fall back to the global AI growth rate so a workload that
        # has no specific keyword hits still gets some signal weight.
        ai_current = ai_attr_2024.get(workload_id, 0.0)
        ai_prior = ai_attr_2023.get(workload_id, 0.0)
        if ai_current > 0 or ai_prior > 0:
            ai_growth = compute_ai_growth_rate(ai_current, ai_prior)
            ai_growth_source = "per-workload keyword attribution"
        else:
            ai_growth = ai_total_growth
            ai_growth_source = "global AI procurement growth (no workload-specific keyword match)"

        ai_growth_pct = None
        if ai_prior > 0:
            ai_growth_pct = round(((ai_current - ai_prior) / ai_prior) * 100, 2)

        sub_rate = compute_substitution_rate(current_spend, prior_spend, ai_growth)

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
            "ai_procurement_current": round(ai_current, 2),
            "ai_procurement_prior": round(ai_prior, 2),
            "ai_procurement_yoy_change_pct": ai_growth_pct,
            "ai_growth_rate_used": ai_growth,
            "ai_growth_source": ai_growth_source,
            "substitution_rate_proxy": sub_rate,
            "source": "USAspending.gov API (live)" if live_signal else "no data",
            "missing_psc_codes": missing_pscs,
            "substitution_rate_methodology": (
                "sub_rate = contractor_decline_rate × 0.6 + ai_growth_rate × 0.4. "
                "contractor_decline_rate = max(0, -yoy_change_pct/100), capped at 1.0. "
                "ai_growth_rate from spending_by_award keyword search "
                "(per-workload if any keyword attribution, else global AI growth). "
                "Note: PSC categories are government-wide and broader than the specific workloads."
            )
        }

        print(f"[FPDS] {workload_id}: contractor=${current_spend:,.0f} (yoy={yoy_pct:.1f}%), "
              f"ai=${ai_current:,.0f} (growth={ai_growth:.3f}, src={ai_growth_source}), "
              f"sub_rate={sub_rate:.3f}"
              + (" [missing some PSCs]" if missing_pscs else ""))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "USAspending.gov FPDS",
        "fiscal_years": {"current": CURRENT_FY, "prior": PRIOR_FY},
        "psc_spend_summary": {
            "fy2024": {code: info.get("amount", 0) for code, info in psc_2024.items()},
            "fy2023": {code: info.get("amount", 0) for code, info in psc_2023.items()}
        },
        "ai_procurement_summary": {
            "fy2024_total": ai_2024["total_obligated"],
            "fy2023_total": ai_2023["total_obligated"],
            "global_yoy_growth_rate": ai_total_growth,
            "fy2024_award_count": ai_2024["award_count"],
            "fy2023_award_count": ai_2023["award_count"],
            "workload_attribution_fy2024": ai_2024["workload_attribution"],
            "workload_attribution_fy2023": ai_2023["workload_attribution"],
            "method": (
                "spending_by_award keyword search across "
                f"{len(AI_KEYWORD_QUERIES)} AI-related keywords. "
                "Awards deduped by award_id; per-workload attribution via "
                "description/recipient text matching against AI_KEYWORD_WORKLOAD_HINTS."
            ),
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
