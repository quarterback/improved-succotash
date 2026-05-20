# Skills & Learning Goals — Occupant

A personal excavation of what this project demonstrates and what it's teaching me.
This is a hobby project (public infrastructure for AI-compute economics, not a product),
but the work is real and the receipts are in the repo.

---

## What I'm Learning (personal-site snippet)

A running list of questions this project is forcing me to answer:

1. **How do you build a defensible price index from partially-priced market data?**
   The Compute CPI covers ~2,200 models, but only ~40% have exact pricing — the rest
   need tier-based imputation. Volume weighting, basket design, and gap handling all
   matter.
2. **What separates a real composite index from a cosmetic one?**
   Working through the AEAI's energy sub-component taught me the "independent signal"
   test: if a proxy moves identically to another component, blending it adds nothing.
3. **How do you measure substitution, not just capability?**
   The LDI shows a ~37,000× cost ratio between human and AI on federal workloads,
   but only ~3.86% actual substitution. Separating *can* from *did* changes the story.
4. **What's the minimum architecture for a credible public dataset?**
   Static site + JSON endpoints + git history as an immutable audit log + no analytics.
   Cheaper, more auditable, and forces methodology to live in code.
5. **How do you write methodology a stranger can audit?**
   Practicing the PRD → calculation code → AAR loop so the math, the choices, and the
   limitations are all legible without me in the room.
6. **When is something public infrastructure vs. a product — and how does that change every decision?**
   Pricing, messaging, license, data collection, even what claims you're allowed to
   make about yourself.

---

## Skills Demonstrated

### Quantitative & economic methodology
- **Custom index design.** Compute CPI built as a volume-weighted basket across six
  workload profiles (classification, summarization, frontier reasoning, etc.) with
  tier-based imputation for unpriced models. See `compute_cpi.py`,
  `scripts/calculate_cpi.py`.
- **Composite indexing.** AEAI as a 60% tokens / 30% spend / 10% energy blend, each
  component independently normalized to a Jan-2025 baseline before recombination.
  Documented in `AEAI_README.md`.
- **Signal-independence reasoning.** Energy sub-index blends a token-derived proxy
  with grid-investment data because the proxy alone fails to add independent signal —
  trade-off explicitly justified in the methodology doc.
- **Cost-per-unit substitution analysis.** LDI maps nine federal workloads (SNAP
  eligibility, UI claims, call-center triage, etc.) onto BLS occupational wages and
  Compute-CPI-priced AI alternatives.
- **Absorption classification.** Eliminated / reallocated / frozen / upgraded states
  derived from JOLTS and FedScope deltas. See `scripts/derive_signals.py`.
- **Decomposition work.** CPI-decoupled cost ratios, contract-vs-FTE splits,
  wage-income displacement — the Axis 4 commit series.

### Data engineering
- **Multi-source ingestion** across OpenRouter, LiteLLM, pricepertoken, BLS, FPDS,
  OPM FedScope, and BloombergNEF, each in its own fetcher module under `scripts/`.
- **Automated pipeline.** GitHub Actions (`.github/workflows/`) regenerates every
  index daily at 06:00 UTC; the daily commit stream is the audit trail.
- **Historical backfill** for retroactively reconstructing index values when sources
  added coverage (`backfill_historical.py`, `backfill_ldi.py`).
- **Data-quality gates.** Price discrepancies above 10% trigger manual review;
  every snapshot carries source-freshness metadata.
- **Model-ID normalization** across OpenRouter, LiteLLM, and LMSYS Arena formats
  (`market_intel.py`).

### Full-stack & web
- **Vanilla JS, no framework.** Async fetch + DOM rendering against `/data/*.json`
  endpoints — deliberately chosen for auditability and longevity.
- **Static-site architecture** with no database; CDN-friendly, cheap to host.
- **Progressive Web App.** Service worker (`sw.js`) precaches pages for offline use,
  manifest for installability.
- **Interactive tooling.** Self-serve calculator (`calculator.html`) computes against
  the live index; D3 chart on the LDI dashboard.
- **Privacy by construction.** Zero analytics, zero third-party scripts, theme stored
  only in `localStorage`.

### Product, writing & PM
- **PRD authorship.** `reports/prd-labor-displacement-index.md` — problem framing,
  scope, derivation receipts, classification rules, quality metrics.
- **Axis-based roadmapping.** Multi-quarter feature planning in numbered phases
  (Axis 4 → procurement rollup, wage-income decomposition; Axis 5 → per-workload
  endpoints, SNAP deep-dive, calculator).
- **After-action reports.** `reports/aar-*.md` — structured analytical writing for
  monthly index review.
- **Competitive messaging audit.** 728-line teardown of a homepage; methodology for
  examining positioning, not just copy.
- **Governance design.** Parachute Commons license with tiered restrictions for
  personal vs. commercial vs. surveillance use.
- **Reputational discipline.** Removed "independent research" language from
  marketing because aggregating public data doesn't qualify. Saying less when less
  is true.

### Domain knowledge
- AI market structure: model tiers, token economics, quality-adjusted pricing,
  arbitrage opportunities.
- Federal procurement: FPDS contract data, OPM FedScope FTE counts, per-agency
  rollups, AI-keyword search across procurement records.
- Labor economics: BLS occupational wages, JOLTS interpretation, contract-vs-FTE
  composition, substitution velocity and acceleration.

---

## Where to look in the repo

| For… | Go to |
| --- | --- |
| Design reasoning and trade-offs | `case-study.md` |
| Index methodology in detail | `AEAI_README.md` |
| Product thinking | `reports/prd-labor-displacement-index.md` |
| Engineering substance | `compute_cpi.py`, `scripts/calculate_*.py` |
| Operational discipline | `.github/workflows/update-data.yml` + the daily commit stream |
| Analytical writing voice | `reports/aar-*.md` |
| Frontend approach | `calculator.html`, `displacement.html`, `sw.js` |

---

## Track record

- 3 published indices (Compute CPI, AEAI, LDI), regenerated daily since Feb 2025.
- ~2,200 AI models tracked across multiple pricing sources.
- 9 federal workloads mapped end-to-end (wage data → procurement signal → cost ratio
  → absorption classification).
- ~$50/month total operating cost.
