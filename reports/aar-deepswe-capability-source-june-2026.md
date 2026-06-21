# After Action Review: DeepSWE — A Coding-Capability Source for Market Intelligence
## June 2026

---

### The Problem

**Occupant's quality signal rested on a single source, and that source measured the wrong thing for half the questions people bring to the Market Intelligence page.**

The `sabermetrics.html` metrics — Quality-Adjusted Price (QAP), cognitive arbitrage, tier ELO — all derive "quality" from one input: Arena ELO (`data/market/arena_elo.json`). Arena ELO is a general human-preference score. It says nothing about whether a model can actually *ship working code*, which is exactly what a buyer pricing an agentic coding workload wants to know. A model can top the Arena and still flounder on a real repo.

There was also a provenance fragility: a metric built on one leaderboard inherits that leaderboard's blind spots with no second opinion. The quality axis needed corroboration from an independent, task-grounded source.

The trigger was external: the [DeepSWE leaderboard](https://deepswe.datacurve.ai/) — 113 *original* (non-scraped, so leak-resistant) software-engineering tasks across 91 repos in 5 languages, graded by verifiers that test actual behavior rather than implementation specifics. The user pointed at it and said, in effect, *this belongs in the indices' data*.

---

### The Decision That Mattered (read this first)

**The first pass built DeepSWE as a fourth standalone index. That was wrong, and it got reverted.** This is the load-bearing breadcrumb for anyone extending this work.

The initial instinct was to treat DeepSWE as a peer to `$CPI` / `$AIU` / `$LDI`: a dedicated `deepswe.html` page mirroring the leaderboard, a `data/deepswe/` index directory, nav links on every data page, an entry in the `indices.html` tool list. The user rejected this in two sharp corrections:

1. **"Do not add deepswe as an index to Occupant."** Occupant's indices are *original measurement instruments* assembled from public data. DeepSWE is someone else's leaderboard. Re-publishing it as an "index" both misrepresents authorship and dilutes what the word *index* means here.
2. **"Add deepswe to the metrics to help their provenance — don't link to it."** The value isn't a copy of the table. It's (a) a derived *metric* and (b) a second *source* that strengthens the existing quality signal.

The distinction worth internalizing: **Occupant publishes instruments, and consumes sources.** A new external dataset is almost always a *source* feeding an existing instrument, not a new instrument. When in doubt, fold inward; don't add a top-level surface.

What survived the user's filter as genuinely wanted: the derived metric **cost per solved task**.

---

### The Fix

DeepSWE now enters the system the same way Arena ELO does — as a capability source under `data/market/`, joined onto the existing model stats. No page, no index, no nav link.

**`src/fetch_deepswe.py`** records the published v1.1 results and writes **`data/market/deepswe.json`**, a capability source keyed by `model_id` (`provider/model`) so it joins the pricing registry directly. Because the leaderboard renders client-side with no stable JSON endpoint, the results are held as curated static data with full provenance — the same static-fallback convention `fetch_bls.py` uses for OEWS wages. Each record carries pass@1, its CI, average run cost, and the derived metric:

```
cost_per_solved_task = avg_cost / pass@1
```

This is the quality-adjusted price of agentic software work. It penalizes models that are cheap per run but fail often: a model at $3/run and 30% pass costs $10 per *solved* task, the same as a model at $7/run and 70% pass. It is the QAP idea, re-expressed in the units of real task completion instead of token price × ELO.

**`src/market_intel.py`** gained:

- `load_deepswe_capability()` — mirror of `fetch_lmsys_arena()`.
- A join in `build_model_stats()` that attaches `swe_pass_at_1` and `cost_per_solved_task` onto any model that was benchmarked.
- `build_coding_capability()` — a compact digest (capability leader + value leader + benchmark provenance) added to the report. A *digest*, deliberately not a table, to honor "don't mirror the leaderboard."
- An explicit `sources` provenance block listing OpenRouter, Arena, and DeepSWE with what each is used for.

**`sabermetrics.html`** cites DeepSWE in its footer provenance line (`Data from OpenRouter, Arena Leaderboard, DeepSWE`) — text, not a hyperlink, matching the existing non-linked style and the "don't link to it" instruction.

**`.github/workflows/update-data.yml`** runs `fetch_deepswe.py` *before* `market_intel.py`, so the join reads fresh capability data on every daily regeneration.

---

### The Numbers

The v1.1 DeepSWE results, re-derived (cost per solved task = avg cost ÷ pass@1):

| Model | Pass@1 | Cost / run | **Cost / solved** |
|---|---|---|---|
| claude-fable-5 [max] | 70% | $21.63 | $30.90 |
| gpt-5.5 [xhigh] | 67% | $7.23 | $10.79 |
| claude-opus-4.8 [max] | 59% | $13.22 | $22.41 |
| gpt-5.4 [xhigh] | 52% | $5.65 | $10.87 |
| **glm-5.2 [max]** | 44% | $3.92 | **$8.91** |
| gemini-3.5-flash [medium] | 37% | $7.34 | $19.84 |
| kimi-k2.7-code | 31% | $2.82 | $9.10 |
| claude-sonnet-4.6 [high] | 30% | $5.52 | $18.40 |
| gemini-3.1-pro [high] | 12% | $9.48 | $79.00 |

**8 of the 9 benchmarked models matched the live pricing registry** by `model_id`, so their stats now carry the coding-capability fields. (`gemini-3.1-pro` didn't join — a naming mismatch worth fixing if it persists.)

---

### What It Revealed

The metric tells a story raw pass@1 hides. **The capability leader is not the value leader.** `claude-fable-5` wins on accuracy (70%) but costs $30.90 per solved task. `glm-5.2` solves fewer tasks (44%) but at $8.91 per solved task — roughly a third of the cost to land working code. And the cautionary tail: `gemini-3.1-pro` is *cheap per run* yet, at 12% pass, the most expensive way to obtain a solution on the board ($79/solved). Cheap-per-token is a mirage when the work fails.

This is the same lesson the LDI and CPI work keep surfacing in different domains: the headline number (here, pass@1; there, a vendor quote) is not the decision-relevant number. The decision-relevant number is per-*unit-of-outcome*.

---

### Honest Limitations

- **Static source.** DeepSWE renders client-side; the results are curated static data, refreshed by editing `RESULTS_RAW` when a new leaderboard version ships. If a public JSON endpoint appears, swapping in a live fetch is a one-function change.
- **Cost is theirs, not ours.** `avg_cost` comes from DeepSWE's own runs, not the `$CPI` basket. The metric is internally consistent but not re-priced through Occupant's pricing — by design, since these are agentic multi-step runs, not single-call token costs.
- **Thin join coverage.** Only models that are both benchmarked *and* in the pricing registry get enriched stats; the digest stands alone regardless.
- **One benchmark.** It's a second opinion on quality, not ground truth. Coding capability ≠ general capability.

---

### Files Touched

**Added**
- `src/fetch_deepswe.py` — capability-source fetcher
- `data/market/deepswe.json` — the source (committed; deterministic)

**Modified**
- `src/market_intel.py` — loader, join, `coding_capability` digest, `sources` block
- `.github/workflows/update-data.yml` — DeepSWE fetch ordered before market_intel
- `sabermetrics.html` — provenance citation only

**Reverted from the abandoned first pass** (do not resurrect without re-reading "The Decision That Mattered")
- Deleted `deepswe.html`, `data/deepswe/`
- Removed DeepSWE nav links from `aeai.html`, `calculator.html`, `cpi-data.html`, `displacement.html`, `sabermetrics.html`
- Removed the DeepSWE tool entry from `indices.html`

**Deliberately not committed**
- A locally regenerated `data/market/latest.json`. Running `market_intel.py` by hand also rebuilds the model registry and produces ~11k lines of unrelated churn. The integration lives in code; CI will populate `coding_capability` and the per-model fields cleanly on the next daily run.

---

*Branch: `claude/deepswe-datacurve-ueh04g`. Source: DeepSWE Leaderboard v1.1, https://deepswe.datacurve.ai/. Capability data as of June 2026.*
