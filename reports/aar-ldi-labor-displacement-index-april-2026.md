# After Action Review: Labor Displacement Index — Build and Launch
## April 2026

---

### What This Is

The Labor Displacement Index (LDI) is a new index on the Occupant platform. It measures something the Compute CPI and AEAI do not: **the cost relationship between human labor and AI execution, per unit of government work**.

The Compute CPI tracks how AI costs are changing over time — what it costs to run AI workloads across model tiers. The AEAI tracks how much AI economic activity is happening in aggregate. Neither tells you what happens to the humans doing work that AI can now do cheaper, or by how much.

The LDI is built to surface that question in measurable terms. Not in aggregate. Not nationally. In specific, observable government workloads where the data is legible enough to force into a common unit of analysis: **cost per workload unit**.

---

### The Core Argument

AI inference costs and human labor costs are not in the same galaxy. That's the structural fact the index is designed to quantify.

At current pricing (Compute CPI: 69.1), the average cost gap across five federal workloads is:

- Human: **$16.80 per workload unit** (fully loaded compensation)
- AI: **$0.0050 per workload unit** (inference cost, from Compute CPI basket)
- Average cost ratio: **~49,500x**

If AI handled 100% of these five workloads, the structural cost displacement would be **$2.2 billion per year**. That is not a projection. It is a technical calculation: the cost differential times the volume, if substitution were complete. Actual substitution is 2.21%. The gap between those two numbers is the thing worth watching.

---

### Two Signals, Kept Separate

The index tracks two distinct measurements and refuses to average them. Averaging them would obscure more than it reveals.

**Signal 1 — Cost Differential (structural)**

What the price gap actually is, per workload unit.

```
displacement = (human_cost_per_unit − ai_cost_per_unit) × annual_volume

human_cost_per_unit = (annual_wage / 2080) × 1.454 fully_loaded × (min_per_unit / 60)
ai_cost_per_unit    = Compute CPI basket cost for the relevant task type
```

This is structural. It tells you *why* substitution pressure exists and *what the ceiling is*. It does not tell you how much has shifted.

**Signal 2 — Substitution Rate (observable)**

What procurement data actually shows shifting, measured at the FPDS contract category level.

```
sub_rate = contractor_decline_rate × 0.6 + ai_procurement_growth × 0.4

contractor_decline_rate = max(0, −yoy_change_pct / 100)
Source: USAspending.gov spending_by_category/psc/ — live API, FY2023→FY2024
```

The AI procurement component (40% weight) is currently zeroed — isolating AI-specific awards requires keyword-level search on contract titles and descriptions, which is not yet implemented. The contractor spend signal (60% weight) comes from live USAspending.gov data.

The gap between Signal 1 and Signal 2 is the story. A 49,500x cost ratio and a 2.21% substitution rate in the same sentence is more informative than either number alone.

---

### The Five Pilot Workloads

Workloads were chosen for legibility, not size. Selection criteria: clear SOC anchor, a matching FPDS procurement category, a plausible current-generation AI substitution path, and a publicly observable annual volume.

| Workload | SOC | Human $/unit | AI $/unit | Ratio | Volume | Structural Displacement | Sub Rate | Absorption |
|---|---|---|---|---|---|---|---|---|
| SNAP Eligibility Processing | 13-1041 | $25.13 | $0.000256 | 98,176x | 41M/yr | $1.03B/yr | 0.0% | Reallocated |
| Unemployment Claims Adjudication | 13-1031 | $29.38 | $0.000256 | 114,770x | 21M/yr | $617M/yr | 0.0% | Frozen |
| Call Center Triage | 43-4051 | $3.92 | $0.006225 | 630x | 100M/yr | $391M/yr | 4.0% | Reallocated |
| Document Summarization (Casework) | 23-2093 | $17.08 | $0.018225 | 937x | 750K/yr | $13M/yr | 0.0% | Reallocated |
| Federal IT Help Desk (Tier-1) | 15-1232 | $8.46 | $0.000256 | 33,062x | 18M/yr | $152M/yr | 0.0% | Frozen |

**Total structural potential: $2.21B/yr · Observed substitution rate: 2.21% · Pilot count: 5**

A few things worth noting about this table:

**Call center triage has the lowest cost ratio but the highest observed substitution rate.** At 630x, it's the "least extreme" gap in the set. But it's also the one with the most observable procurement shift (4.0%). This is consistent with the substitution thesis: the workloads that are easiest to substitute (high volume, low judgment, first-tier routing) tend to move first even when the absolute dollar savings per unit are smaller.

**SNAP and UI claims have the largest structural potential but zero observed substitution.** $25.13 and $29.38 per unit, 41M and 21M units annually — but procurement data shows no category-level decline. Growing contractor spend in those categories, in fact. This could mean substitution hasn't started, or that it's happening through internal reallocation rather than contractor reduction, or that the PSC categories are too broad to detect it. Probably all three.

**Document summarization has the smallest structural displacement** ($13M/yr) because the volume is low (750K VA hearing units/yr) even though the cost gap is real. But VA document summarization pilots have shown 60–80% time reduction. The dollar figure understates the operational significance.

---

### How Human Cost Is Computed

Three steps. Each step is documented and auditable.

**Step 1 — Annual wage (BLS OEWS)**
SOC-level mean annual wages from BLS Occupational Employment and Wage Statistics, May 2023. This is the publicly available machine-readable dataset from bls.gov/oes. It gives wages by occupation at the national level, with state and metro breakdowns available.

**Step 2 — Fully loaded compensation (BLS ECEC)**
Wages alone understate true cost. BLS Employer Costs for Employee Compensation (Q4 2024) shows that wages are approximately 69.5% of total compensation for state and local government workers. The rest is benefits: health insurance, retirement contributions, paid leave, and payroll taxes.

Fully loaded multiplier: **1.385 (wage→total compensation) × 1.05 (overhead) = 1.454 combined factor**

Fully loaded hourly = (annual_wage / 2080 hours) × 1.454

Using wages-only would understate the human cost side by roughly 45%, which would materially understate the cost gap and misrepresent substitution pressure.

**Step 3 — Cost per unit (O*NET)**
Minutes per workload unit are derived from O*NET task-level data and cross-referenced with published program benchmarks.

```
cost_per_unit = (fully_loaded_hourly / 60) × minutes_per_unit
```

Minutes per unit by workload:
- SNAP eligibility review: 45 min
- UI claims adjudication: 35 min
- Call center triage (first-tier): 6 min
- Document summarization: 60 min
- IT help desk (Tier-1 ticket): 12 min

---

### How AI Cost Is Computed

AI cost per unit comes directly from the Occupant Compute CPI basket — not estimated independently. Each workload maps to a task type:

- **Classification tasks** (SNAP, UI claims, IT help desk): CPI classification basket
- **Chat/triage tasks** (call center): CPI chat basket
- **Summarization tasks** (document summarization, VA casework): CPI summarization basket

At the current CPI level (69.1), this places AI costs between $0.000256 and $0.018225 per unit depending on task type. As the CPI continues to decline — it has fallen 30.9 points since the 100 baseline — the structural cost gap widens automatically. The LDI is indexed to the Compute CPI on the AI side, which means both indices move together.

---

### How Substitution Rate Is Computed

This is the hardest part and the most important to understand clearly.

The substitution rate is derived from USAspending.gov contract award data, aggregated at the Product/Service Code (PSC) level. Each pilot workload maps to one or more PSC codes — the standard federal procurement taxonomy.

The contractor spend signal measures: does year-over-year contractor spending in the relevant PSC categories show a decline? Decline is interpreted as a proxy for substitution (work moving from contractors to AI tools or internal staff), not as noise.

**FY2023 → FY2024 observed:**

| Workload | PSC Codes | FY2024 Spend | YoY Change | Sub Rate |
|---|---|---|---|---|
| SNAP Eligibility | R408, D302, M1MA | $47.96B | +8.5% | 0.0% |
| UI Claims | R408, D302, D399 | $47.14B | +9.8% | 0.0% |
| Call Center Triage | D302, R408, D399 | $28.72B | **−6.7%** | **4.0%** |
| Document Summarization | R499, D399 | $30.05B | +0.4% | 0.0% |
| IT Help Desk | D302, D399, R499 | $30.05B | +0.4% | 0.0% |

**The 2.21% aggregate substitution rate comes almost entirely from call center triage** (4.0% sub rate on 60% weight of the contractor decline component). The other four workloads show growing category spend, which scores as 0% — not as substitution evidence.

**The most important caveat about this number:** PSC categories are government-wide and far broader than the specific workloads. R408 "Professional: Program Management/Support" covers tens of billions of dollars in spending across every federal agency and function. A YoY decline in that category could reflect budget cuts, contract consolidation, scope changes, or actual substitution. It cannot be cleanly attributed to any specific workload. The substitution rate is a **directional signal, not a precise measurement**. It is the best available proxy from public data. It is not a census.

---

### Absorption: What Happens to the Workers

The absorption classification answers a different question: if substitution is happening, what is the fate of the displaced workers? The index tracks five possible outcomes per workload, inferred from JOLTS, OPM, and budget signals:

- **Eliminated** — separations increase, position abolished
- **Reallocated** — redirected to higher-complexity work, caseload composition shifts
- **Upgraded** — reclassified to higher GS band, new role requirements
- **Contractor reduction** — FPDS non-renewal, spend decline in category
- **Frozen** — vacancy rate increases, no backfill authorized

Current classifications across the five workloads: SNAP (reallocated), UI claims (frozen), call center (reallocated), document summarization (reallocated), IT help desk (frozen).

Frozen and reallocated together cover all five. No eliminations in this dataset. This is consistent with the low observed substitution rate — if only 2.21% of volume has shifted, eliminations would be premature to observe. What we're seeing is the early signal: headcount held flat while volume grows (frozen) or complex caseload growing while routine work automates (reallocated).

---

### Data Sources

| Source | What It Provides | Frequency |
|---|---|---|
| BLS OEWS (bls.gov/oes) | SOC wages by occupation, May 2023 | Annual |
| BLS ECEC (bls.gov/ect) | Compensation structure (wages + benefits), Q4 2024 | Quarterly |
| O*NET Online (onetonline.org) | Task-level minutes per occupation | Annual |
| USAspending.gov FPDS API | Contract awards by PSC, FY2023–FY2024 | Daily (annual comparison used) |
| USDA FNS SNAP data | Annual SNAP participation (volume source) | Annual |
| DOL ETA UI weekly claims | Annual UI claim volumes | Weekly aggregated |
| Occupant Compute CPI | AI cost per unit by task type | Daily |

---

### Pipeline Files

The LDI is produced by three Python scripts running against public APIs:

- **`src/fetch_bls.py`** — Pulls BLS OEWS wage data by SOC code, applies ECEC multiplier, computes fully loaded hourly and per-unit costs. Falls back to static May 2023 values when the live API is unavailable.
- **`src/fetch_fpds.py`** — Queries USAspending.gov spending_by_category/psc/ endpoint for FY2023 and FY2024 award totals by PSC code. Computes YoY change and contractor decline rate per workload.
- **`src/calculate_ldi.py`** — Assembles both signals, maps workloads from `data/ldi/workload_map.json`, pulls AI costs from the Compute CPI, and writes `data/ldi/latest.json`.

The workload map (`data/ldi/workload_map.json`) is the translation layer: it defines the SOC code, PSC codes, O*NET minutes per unit, volume source, and task type for each pilot workload. This is the document that turns procurement codes and occupation codes into a single cost-per-unit comparison.

Output: `data/ldi/latest.json` — updated on each pipeline run.

---

### What This Index Is Not

**It is not a prediction.** The structural cost displacement figure ($2.2B/yr at 100% substitution) is a technical ceiling, not a forecast. Whether and how fast substitution reaches that ceiling depends on procurement decisions, regulatory constraints, workforce agreements, and organizational capacity — none of which this index models.

**It is not a national measurement.** Five workloads in federal government. Not all federal agencies. Not state and local. Not private sector. The five were chosen because the data is legible, not because they represent the whole economy.

**It is not a realized savings number.** The cost displacement column shows what would be displaced if AI handled everything. Actual savings from AI deployment are lower — integration costs, human oversight, error rates, and transition costs are real. The index measures the cost gap, not the net benefit.

**The substitution rate is not a share of tasks done by AI.** It is a procurement signal — contractor spend declining in categories associated with substitutable workloads. This is the best proxy available from public data. It is not a direct measurement of AI task volume.

---

### What We're Watching

**USAspending.gov FY2025 data.** The substitution rate is currently computed on FY2023→FY2024 data. When FY2025 becomes available, it will be the first full year after the major federal AI deployment wave (GSA pilots, VA automation, IRS modernization). If call center triage's -6.7% extends or accelerates, it is a meaningful signal. If the other workloads shift from growing to declining, even more so.

**AI procurement component.** The 40% weight on AI-specific contract awards is currently zeroed. This requires keyword-level search on contract award titles and descriptions — identifying awards to OpenAI, Anthropic, AI-related integrators, or IDIQ vehicles with AI task orders. When this is implemented, the substitution rate will reflect both the contractor decline signal and the AI vendor growth signal, which is the correct formulation.

**Absorption reclassification.** The current absorption classifications are inferred from macro signals (JOLTS sector data, FPDS category trends). As OPM position reclassification data and union contract amendments become more legible, the absorption column will sharpen from five broad categories into a more granular picture of what is actually happening to specific roles.

**The cost ratio as CPI moves.** The average cost ratio of ~49,500x will increase as the Compute CPI declines further. Every point the CPI falls widens the structural gap. At the current trajectory, the ratio will continue climbing while the substitution rate — a function of procurement behavior, not pricing — moves more slowly. The divergence between the widening cost gap and the slow-moving substitution rate is a persistent feature of this measurement problem, not a data error.

---

### Methodology Note

This AAR documents the LDI as launched April 13, 2026. Files added: `displacement.html`, `src/fetch_bls.py`, `src/fetch_fpds.py`, `src/calculate_ldi.py`, `data/ldi/workload_map.json`, `data/ldi/latest.json`, `data/ldi/bls_output.json`, `data/ldi/fpds_output.json`, `data/ldi/historical.json`. The index page is live at `/displacement.html` and linked in the site-wide navigation as LDI.

Human cost inputs use BLS static fallback values (May 2023) for all five workloads — the live OEWS API is available but static values are used for reproducibility until annual refresh logic is validated.

AI cost inputs are live from the Compute CPI at the time of each pipeline run. CPI at time of this writing: 69.1.

Substitution rate uses live USAspending.gov API for FY2023 and FY2024 award totals by PSC. Figures in this document reflect the April 13, 2026 pipeline run.

---

*Occupant Index · April 2026*
*Contact: hello@occupant.ee*
