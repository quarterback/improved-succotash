# Occupant — Public Measurement Infrastructure for the AI Economy

*A project overview. Written to be read cold — by a person, an investor, or another model — with no prior context.*

---

## The one-paragraph version

We talk about AI constantly, but we can't price it. There are public, daily,
methodology-driven indices for soybeans, crude oil, electricity, freight, even
the capacity factor of a wind farm — but for the fastest-moving input in the
economy, AI compute, there is no shared yardstick. Procurement happens on vendor
claims. The macro debate (bubble or deflation? is AI taking jobs?) happens
without a shared number. **Occupant** is the missing instrument panel: three
indices and a set of tools that turn scattered public data into measurements you
can track over time. It tracks the *price* of AI work (`$CPI`), the *volume* of
AI economic activity (`$AIU`), and the *labor* pressure AI puts on real jobs
(`$LDI`) — all rebuilt daily from public sources, all free, no login, no
tracking. The raw material always existed; nobody had assembled it into
instruments. That assembly is the project.

---

## Why it exists

By 2025, AI compute had become one of the largest and fastest-moving line items
in enterprise and government budgets. Prices were falling 30–40% a year. New
models shipped weekly. And yet a procurement officer signing a seven-figure AI
contract had nothing to check the quote against — no equivalent of a commodity
spot price, a CPI print, or a published yield curve.

The contrast is the whole thesis. We have mature, public price discovery for
almost every other input to the economy:

- **Commodities** have spot and futures prices, published continuously.
- **Consumer goods** have the CPI, a basket re-priced every month by a
  government statistical agency.
- **Energy** has wholesale electricity prices, capacity factors, and grid
  investment reports.

AI compute — arguably the input everyone is most anxious about — had none of
this. The data needed to build it *did* exist, scattered across OpenRouter,
LiteLLM, the Bureau of Labor Statistics, federal procurement records, and energy
analysts. It just hadn't been surfaced as instruments. **None of this was being
measured this way.** Occupant exists to do that surfacing: to make implicit
market dynamics explicit, measurable, and public.

It is deliberately *infrastructure, not a product*. No subscriptions, no API
keys, no premium tier, no data capture. Commercial users, government agencies,
and researchers get the same access. The model is closer to a public yield
curve than to a SaaS dashboard.

---

## What it measures: three indices

Each index answers one question that, before this, you couldn't actually look
up.

### 1. `$CPI` — Compute Price Index — *the price signal*

> **Is the price of AI compute actually falling, or is that just frontier hype?**

`$CPI` tracks the price of *work*, not the price of a model. It defines a
standardized basket of workloads and re-prices that basket across capability
tiers, volume-weighted across **2,188 models**, normalized to **February 2025 =
100**.

- **As of mid-May 2026 it sits at ~62.4** — meaning a basket of AI work costs
  roughly 38% less than it did at launch, and it fell ~9.7% in the most recent
  month alone.
- The tiers: **`$BULK`** (commodity models under $1/MTok), **`$FRONT`**
  (frontier capability, ranked by Arena ELO), **`$JUDGE`** (reasoning-intensive,
  o1/o3/R1-class), and **`$LCTX`** (long-context, 128K+).
- It publishes the index across multiple horizons (since-launch, year-over-year,
  quarter- and month-to-date, week-over-week) and methodology variants, plus
  three **Build Cost sub-indices** — **`$START`** (startup builder mix),
  **`$AGENT`** (agentic systems), and **`$THRU`** (high-throughput) — which apply
  fixed, realistic workload mixes so a reader can find the number that matches
  *their* usage pattern.

This is a deflator, the same idea as the consumer CPI: hold the basket fixed,
watch the price move.

### 2. `$AIU` / AEAI — AI Economic Activity Index — *the volume signal*

> **Is the AI market actually growing, or is the spend just chasing flat usage?**

Where `$CPI` measures price, the **AI Economic Activity Index** measures
*volume*, expressed through a synthetic unit called the **AIU**, modeled on the
IMF's Special Drawing Rights. The AIU is a weighted composite:

- **60% token throughput** (usage intensity, from OpenRouter rankings)
- **30% inferred spend** (tokens × blended pricing — economic scale)
- **10% energy proxy** (a *blended* signal: 70% token-derived energy estimate +
  30% external AI-infrastructure capex — itself the equal-weighted average of
  BloombergNEF global grid-investment growth and U.S. Census data-center
  construction spending — so the energy component carries genuinely independent
  information rather than just echoing the token count)

Baseline **February 2025 = 100**. **As of mid-May 2026 the AIU is ~503.65** —
roughly a 5× expansion in activity in about fifteen months, on ~42.7 trillion
tokens/week and ~$210M/week of inferred spend (across the OpenRouter-visible
slice of the market).

The two indices are designed to be read *together*:

> **Total AI spending ≈ AIU × CPI.**

If activity 5×'s while price halves, total spend still rises sharply — but you
can now *decompose* it into efficiency gains (price falling) versus demand
growth (volume rising). That decomposition is exactly what the "is this a
bubble?" debate has been missing.

### 3. `$LDI` — Labor Displacement Index — *the labor signal*

> **When AI is cheaper than a person, how long does it actually take to replace
> one?**

`$LDI` is the most carefully hedged of the three, and intentionally so. It
refuses to report a single scary number. Instead it reports **two numbers that
answer two different questions**, and treats the gap between them as the actual
finding:

- **Cost differential (structural)** — what AI *could* displace at current
  pricing if every unit of work routed to AI tomorrow. A pressure gauge, not an
  outcome. Across the pilot workloads, average human cost is **~$53.56/unit**
  versus **~$0.0062/unit** for AI — a cost ratio on the order of **37,000×**, and
  a structural displacement *potential* of **~$4.2 billion/year**.
- **Substitution rate (observable)** — what procurement records show is
  *actually* shifting. **~3.86%.** Far smaller than the structural number, and
  that distance is the point.

It is built on real, named public data rather than speculation: **BLS** OEWS +
ECEC wage data for human cost, **FPDS / USAspending** contractor-spend trends as
a substitution proxy, **OPM FedScope** FTE counts for what's happening to
headcount, all mapped through **SOC and PSC codes** across **9 pilot federal
workloads**. It even adds an *absorption* dimension — when a workload automates,
are workers cut, frozen, or **reallocated**? — because "cheaper" and "fired" are
not the same event.

The discipline here is the feature: most "AI and jobs" commentary collapses
*could* into *is*. `$LDI` keeps them in separate columns and shows the widening
gap.

---

## The worked example: the SNAP deep dive

The site doesn't just assert the LDI method — it runs it end-to-end on one real
program so the reader can audit every step. The **SNAP Eligibility deep dive**
takes the annual eligibility-interview and redetermination cycle for the
Supplemental Nutrition Assistance Program (SOC 13-1041, *Eligibility
Interviewers, Government Programs*; staffed by USDA FNS and federally funded
state agency workers) and resolves all three pillars to that single workload:

- **Human cost:** ~$25.13 per review (45 minutes at a ~$33.51 fully-loaded
  hourly rate, from a $47,930 BLS annual wage).
- **AI cost:** ~$0.000256 per review, priced from the live `$CPI` basket.
- **Volume:** ~41 million reviews/year (USDA FNS FY2024: ~41M participants, ~1
  review each).
- **Substitution signal:** ~2.1%, from federal procurement trends.
- **Absorption:** classified **"reallocated"** — caseload is growing while the
  workforce is roughly flat (FedScope FTE down ~1.8%), so staff are being
  redirected to complex cases and fraud detection as routine processing
  automates, rather than eliminated outright.

Crucially, the page is honest about its own limits inline: PSC procurement codes
are broad government-wide categories, so the substitution signal is "directional,
not precise." That worked example is the proof the index is methodology, not
hand-waving.

---

## The tools: putting the indices to work

The indices are the readings; the tools let people *act* on them.

- **LLM Cost Calculator** — punch in a monthly spend (or raw token volumes) and a
  primary model, pick a workload profile (startup / agentic / throughput), and
  see how a tiered model mix compares at *live* `$CPI` rates. Answers "should we
  lock in rates now or wait 90 days?"
- **LDI Workload Calculator** — run the human-vs-AI cost differential and
  substitution math on a workload of your own, not just the nine pilots.
- **AI Services Price Reasonableness Worksheet** — a print-ready, FAR-style
  procurement determination sheet for federal AI buyers, built on the `$CPI`
  basket. It translates the index into a document a contracting officer can
  actually fill in, sign, and attach to an RFP evaluation.
- **Market Intelligence** ("sabermetrics for models") — statistics that reprice
  market assumptions: **Quality-Adjusted Price**, **cognitive arbitrage** (where
  you can get ~90% of frontier capability at ~40% of the cost), and tier analysis
  across the model market.

---

## How someone uses it

| If you are… | You use it to… |
|---|---|
| A **procurement officer** | Check a vendor quote against the `$FRONT` baseline; attach the worksheet to an RFP; decide whether to lock rates or wait for deflation. |
| A **budget owner** | Project monthly cost from your real workload mix in the calculator; forecast demand growth with the `$AIU`. |
| An **investor / analyst** | Separate the bubble question into its parts — `$CPI` down *and* `$AIU` up means price cuts are being absorbed by real demand, not propping up flat usage. |
| A **policy or labor researcher** | Use `$LDI` to separate "AI will replace X" rhetoric from the procurement evidence of what's actually shifting. |
| **Another model or agent** | Pull the machine-readable JSON endpoints (`data/*.json`) as a grounded, dated source for AI-market pricing, activity, and labor data. |

---

## How it's built

The engineering choices follow directly from the "infrastructure, not product"
mandate.

- **Static site, no framework, no database.** Vanilla HTML/CSS/JS. The whole
  thing is flat files served by any static host.
- **Python data pipeline** (`src/`): dedicated fetchers for each public source
  (OpenRouter, LiteLLM, BLS, FPDS/USAspending, OPM FedScope, BloombergNEF grid
  data), calculators (`calculate_cpi`, `calculate_aeai`, `calculate_ldi`), a
  model tier registry, signal derivation, and historical backfill.
- **Daily regeneration via GitHub Actions** → recomputes the indices → commits
  fresh JSON. The git history *is* the data lineage: every published number is
  reproducible and auditable from the commit log.
- **Deterministic algorithms, no black box.** No ML, no "secret sauce." The
  indices are explainable formulas with documented weights and baselines.
- **Zero tracking.** No cookies, no analytics, no third-party scripts — only
  client-side localStorage for the theme toggle. Public infrastructure shouldn't
  surveil the people using it to evaluate vendors.
- **Parachute Commons license** — free for personal, educational, and
  non-commercial use; tiered restrictions kick in for commercial scale.

---

## What this project demonstrates

Read as a portfolio piece, Occupant surfaces a specific blend of skills that
rarely sit together:

- **Economic index design.** Baskets, volume weighting, normalization, base
  periods, and synthetic composite units (the AIU's SDR analogy) — applied to a
  domain that had no such instruments.
- **Intellectual honesty as a design constraint.** The structural-vs-observable
  split in `$LDI`; surfacing coverage and freshness metrics rather than hiding
  them; explicitly *removing* "independent" and "transparent" claims because the
  work aggregates public data rather than conducting original research. The
  project under-promises on purpose.
- **Data engineering on messy public sources.** Reconciling price discrepancies
  across OpenRouter and LiteLLM, building SOC↔PSC↔FedScope crosswalks, designing
  fallback hierarchies, and reporting data quality honestly per snapshot.
- **Genuine domain range.** CPI/macro methodology, federal procurement (FAR,
  FPDS, FedRAMP), labor economics (BLS OEWS/ECEC, JOLTS, OPM), and energy (grid
  capital expenditure) — coherently connected.
- **Product and communication judgment.** Killing jargon, designing for a
  thirty-second read by a procurement officer, and maintaining a deliberate
  editorial voice.
- **Pragmatic systems architecture.** Static-site-plus-cron instead of a server;
  git as an audit log; three indices that *compose* (spend ≈ AIU × CPI; the LDI
  consumes the live CPI basket); roughly $50/month to operate.

---

## Honest limitations

Stated plainly, because the project's credibility depends on it:

- **Coverage.** Token and activity data come largely from OpenRouter's public
  rankings — an estimated ~5–10% slice of global inference. Direct API and
  enterprise traffic isn't captured.
- **Pricing.** Roughly 60% of models use tier-based price estimates rather than
  exact matches; spend assumes a 70/30 input/output token mix.
- **Labor signal.** The `$LDI` substitution rate is PSC-level, fiscal-year over
  fiscal-year, and government-wide — directional, not nationally
  representative; many trajectory points are reconstructed from CPI snapshots
  rather than measurement-grade.
- **All figures are snapshots.** The value is in the *trajectory*, not any single
  printed number.

---

## In one line

*Occupant is the price tag, the activity meter, and the labor gauge for AI —
public, daily, and auditable — because a market this large should not run on
vendor claims and vibes.*

---

*Index values cited as of mid-2026: `$CPI` ≈ 62.4, `$AIU` ≈ 503.65, `$LDI`
substitution ≈ 3.86%. All baselines February 2025 = 100. Live data and full
methodology are published on the site and in the repository's `data/` directory.*
