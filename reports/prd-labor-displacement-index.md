# PRD — Labor Displacement Index (LDI)

**Status:** live · [displacement.html](../displacement.html) · data at
[`/data/ldi/latest.json`](../data/ldi/latest.json)
**Author:** Ron Bronson · Occupant
**Last updated:** April 2026

---

## One-liner

**The LDI is a public, US-only benchmark that measures the cost and staffing
consequences of AI substituting for human labor in specific, observable
government workloads — not in aggregate, not as a forecast, but per workload
unit and per fiscal year, from primary sources.**

---

## Why this exists

The Compute CPI answers "what does a unit of AI cost today?" The AEAI
answers "how much AI economic activity is happening?" Neither answers the
question the rest of the conversation is straining to hold:

> When AI costs $0.0003 to do what a person costs $25 to do, and 41 million
> of those tasks happen per year in one federal program — what actually
> changes for the workers, the contract budget, and the agency?

That question gets answered today by forecasts, consultant decks, and vibes.
The LDI answers it with receipts: BLS wage data, FPDS/USAspending contract
obligations, OPM FedScope FTE counts, Compute CPI pricing. Nine federal
workloads, one common unit (cost per workload unit), one fiscal-year
comparison (FY2023 vs FY2024).

It is deliberately narrow. It is deliberately US-only. It is deliberately
built on data anyone can re-pull.

---

## What it is (and isn't)

**It is:**
- A measurement project. Nine federal workloads resolved into
  `cost_per_unit_human`, `cost_per_unit_ai`, `substitution_rate`, and
  `absorption_classification`.
- A structural gap disclosure. The **cost ratio** (currently ~37,000×
  human:AI per unit across the nine) is the pressure. The **substitution
  rate** (currently 3.86%) is what's realized.
- A daily-updating public artifact. Latest snapshot in `latest.json`,
  per-workload JSON at `/data/ldi/workloads/<id>.json`, flat CSV at
  `/data/ldi/workloads.csv`.
- A small piece of infrastructure that makes the argument legible enough
  to argue with.

**It is not:**
- A savings estimate. The structural cost gap is an upper bound at 100%
  substitution and static AI pricing. It is the pressure, not the outcome.
- A layoff forecast. Absorption classification (eliminated / reallocated /
  frozen / upgraded) is categorical and derived from OPM FedScope + JOLTS
  signals. It describes *where people are going*, not *how many will go*.
- A judgment about whether substitution is good. The point is to make the
  substitution legible so that judgment can happen downstream.
- A private-sector index. Private workloads get covered in
  [`data/ldi/extensions.json`](../data/ldi/extensions.json) as a comparison
  scope but are not in the composite.

---

## The core argument (the story to tell)

Three sentences, in order:

1. **The price gap is structural.** Per unit of routine government
   cognitive work, AI costs roughly four orders of magnitude less than a
   loaded federal employee. That is not a forecast — it is a published
   BLS wage × Compute CPI inference price.
2. **The substitution rate is small but non-zero and accelerating.** The
   FPDS procurement signal (contractor spend decline + AI procurement
   growth) shows ~4% of these task categories already routing toward AI.
   The 90-day velocity is positive and rising.
3. **The absorption signal is where the human story lives.** OPM FedScope
   FTE counts for the nine SOC codes are flat or shrinking while caseloads
   grow. That's the "frozen" and "reallocated" quadrants — the places
   where layoffs haven't happened because backfill has stopped.

These three are intentionally kept as separate numbers in the output, not
averaged. Averaging the structural gap into the substitution rate produces
a number that's more impressive but less true.

---

## Audiences

Named roughly in order of fit.

### 1. Researchers & policy analysts working on AI labor impact

**What they get:** a primary-source-linked dataset they can cite or
replicate. Every number in `latest.json` has a `source` field and a
reproducible pull path. Methodology is in
[`src/calculate_ldi.py`](../src/calculate_ldi.py).

**How to frame it to them:** "Here's the gap, here's the observed
substitution rate, and here's the absorption signal. The numbers update
daily. Fork it or feed your paper with it."

**Likely failure mode:** they ask why it's US-federal only. Answer: because
federal workloads are the only domain where wage, procurement, and
staffing data are all publicly available at the SOC code × agency grain
needed to force a cost-per-unit comparison. Other scopes are extensions,
not the core.

### 2. Journalists covering AI + government / AI + labor

**What they get:** a concrete, attributable number per story. "SNAP
eligibility interviews are currently billed at $25 a unit of human labor
and $0.0003 a unit of AI inference. USDA FNS FTE is flat since 2022."

**How to frame it to them:** "This is the calculator behind the
one-sentence claim. If you want to say AI is displacing government
workers, this tells you which workloads, by how much, and how fast. If
you want to say it isn't, the ratio and velocity numbers will tell you
why that case is getting harder to make."

**Likely failure mode:** they want "X jobs lost." The LDI doesn't produce
that number and shouldn't. The right quote is the ratio × velocity
combination, or the wage-income-displaced figure at the *observed* rate.

### 3. Agency CIOs, GSA, procurement offices

**What they get:** a public yardstick against which to measure their
own internal AI-substitution pilots. "We're running a help-desk pilot
that hits a 30% substitution rate in six months" becomes meaningful only
when the baseline federal rate is 3.86% and visible.

**How to frame it to them:** "This is the number your agency's AI
procurement will be measured against in the next IG report. Here's where
the baseline is today."

### 4. Practitioners building AI-for-government products

**What they get:** a priced target market with a visible substitution
slope. "SNAP eligibility processing is a $1B/yr labor-cost category with
a 4.9M-sub-rate procurement signal that accelerated 0.23 pp/month over
the last 90 days" is the kind of sentence a founder can sharpen a pitch
on — *and* a reader of a founder deck can verify in two clicks.

**How to frame it to them:** "This is the cost-of-status-quo side of your
deck, sourced."

### 5. Peers — researchers, engineers, other indices

**What they get:** a methodology to argue with. The three-pillar structure
(human cost × procurement × absorption) is simple enough to critique and
explicit enough to fork. Extensions in
[`data/ldi/extensions.json`](../data/ldi/extensions.json) are for
collaborators who want to stretch the scope.

**How to frame it to them:** "Here's what I'm measuring and why; here's
where it breaks; here's the seam you'd cut along to extend it."

---

## What "working" looks like

- A peer can find the number (composite sub rate, one workload's cost
  ratio, the per-agency AI procurement total) in ≤ 30 seconds on
  `displacement.html`.
- A researcher can pull the same number from `/data/ldi/latest.json` or
  `/data/ldi/workloads/<id>.json` and cite it with a source chain that
  terminates in a `.gov` or the Compute CPI basket.
- A journalist can quote a specific workload line without misrepresenting
  what was measured, because the page strictly separates structural gap
  from realized substitution.
- The composite number moves over time without manual intervention. The
  historical series at `/data/ldi/historical.json` grows daily.
- A reader can run their own workload through
  [`/displacement/calculator.html`](../displacement/calculator.html) and
  get the same shape of result with the same math.

---

## Methodology at a glance

Three pillars, computed per workload, then composed:

1. **Human cost per unit.** BLS OEWS annual wage for the SOC code →
   hourly × 1.3 loading factor (ECEC) × minutes-per-unit.
2. **AI cost per unit.** Compute CPI basket entry for the task type
   (classification / chat_drafting / summarization), per 1k tokens → per
   unit.
3. **Substitution proxy.** FPDS/USAspending: contractor spend decline
   (FY2024 vs FY2023) × 0.6 + AI procurement growth × 0.4, either
   workload-specific (keyword-matched) or global.
4. **Absorption classification.** OPM FedScope FTE delta + qualitative
   JOLTS signal → one of {eliminated, reallocated, frozen, upgraded}.
5. **Derived signals** (daily): velocity (pp/month, 90-day least-squares
   slope), acceleration (recent-half − prior-half), wage-income-displaced
   (= sub_rate × volume × human_cost), contract-vs-FTE split,
   CPI-decoupled cost-ratio decomposition.

Full spec and source in
[`src/calculate_ldi.py`](../src/calculate_ldi.py),
[`src/derive_signals.py`](../src/derive_signals.py),
[`src/fetch_bls.py`](../src/fetch_bls.py),
[`src/fetch_fpds.py`](../src/fetch_fpds.py),
[`src/fetch_opm.py`](../src/fetch_opm.py).

---

## How to talk about it

Short, accurate phrasings, in rough order of abstraction:

- **One sentence:** *"A US federal benchmark of the cost and staffing gap
  between human and AI labor, per workload unit, updated daily from BLS,
  FPDS, OPM, and Compute CPI."*
- **Tagline:** *"Routine federal cognitive work costs ~37,000× more from
  a human than from an AI. The LDI tracks what's happening because of
  that."*
- **If the room is technical:** three pillars, nine workloads, per-unit
  comparison; two distinct signals (structural gap vs realized
  substitution) kept un-averaged.
- **If the room is skeptical:** every number has a primary-source link.
  Nothing is forecasted. Absorption is categorical, not quantitative.
- **If the room wants a headline:** composite substitution rate (today:
  3.86%) + 90-day velocity (today: +0.23 pp/month) + structural cost gap
  ($4.2B/yr at 100%). Three numbers, three meanings.

Things to avoid saying:

- "The LDI predicts X jobs will be lost." It doesn't predict.
- "The LDI shows AI has saved $4B." It doesn't show savings — it shows a
  structural gap that, at 100% substitution and static prices, would
  equal that. Different claim.
- "Works across sectors." Core index is US-federal. Private/state
  extensions are a separate scope.

---

## Open questions / roadmap

- **Private-sector comparability.** Extensions file has CA Medicaid and
  private paralegals but isn't in the composite. Should there be a
  parallel private composite once wage data is normalized across BLS
  OEWS and private sources?
- **Workload-specific AI procurement.** The USAspending keyword search
  is currently broad ("artificial intelligence", "machine learning",
  "LLM"). A next pass should map those to PSC × SOC so that
  `ai_procurement_by_agency` can be joined per workload.
- **Absorption quantification.** The classification is categorical today.
  OPM FedScope FTE delta already drives it qualitatively; a next step is
  to publish a continuous absorption score per workload.
- **Historical depth.** The historical series currently starts at first
  pipeline run (April 2026) with CPI-scaled backfill for reconstruction.
  Next pass is to pull BLS OEWS and FedScope vintages back to FY2019 so
  the 90-day velocity has a longer context.
- **Expert review.** The methodology is self-published; the goal is
  peer critique, not self-certification. Explicit invitation for
  researchers to break it is part of the framing.

---

## Links

- Page: [`/displacement.html`](../displacement.html)
- Deep dive (per-workload): [`/displacement/snap.html`](../displacement/snap.html)
- Calculator: [`/displacement/calculator.html`](../displacement/calculator.html)
- Composite snapshot: [`/data/ldi/latest.json`](../data/ldi/latest.json)
- Per-workload JSON: [`/data/ldi/workloads/`](../data/ldi/workloads/)
- Flat CSV: [`/data/ldi/workloads.csv`](../data/ldi/workloads.csv)
- Historical: [`/data/ldi/historical.json`](../data/ldi/historical.json)
- After-action review: [`reports/aar-ldi-labor-displacement-index-april-2026.md`](aar-ldi-labor-displacement-index-april-2026.md)
