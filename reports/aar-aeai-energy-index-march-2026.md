# After Action Review: AEAI Energy Index — Adding Grid Investment Signal
## March 2026

---

### The Problem

**The AEAI had a phantom component. The energy index wasn't measuring energy — it was measuring tokens in disguise.**

The energy proxy (10% of the AIU) was computed as token volumes × tier-based efficiency factors. Since those tier factors are constants, the energy index moved in perfect lockstep with token volume. The formula claimed to be weighting three independent signals. It was actually weighting two:

```
AIU (effective) = 0.7 × token_index + 0.3 × spend_index
```

The 10% energy weight was load-bearing in name only. This was documented in the README as a known limitation but hadn't been addressed. The energy component needed an independent external source — something that reflected real-world AI infrastructure buildup rather than re-deriving from the same token data the other components were already using.

---

### The Fix

BloombergNEF's annual *Grid Investment Outlook* is the right source. Global grid investment is a supply-side infrastructure indicator: it tracks what utilities, governments, and private operators are spending to expand power capacity. BNEF explicitly cites data centers as a major driver of accelerating grid demand. That makes it a structurally sound proxy for the physical infrastructure underpinning AI compute — and critically, it is not derived from token volumes.

The energy component is now a **blended index**:

```
energy_index = 0.7 × token_energy_index + 0.3 × grid_investment_index
```

| Sub-component | Blend | Source | Frequency |
|---|---|---|---|
| Token-derived proxy | 70% | OpenRouter tokens × tier efficiency factors | Weekly |
| Grid investment growth | 30% | BloombergNEF global grid capex | Annual (interpolated weekly) |

The grid investment index is normalized to 2025 = 100, consistent with the AEAI baseline. For weekly snapshots, it interpolates linearly between adjacent annual data points. When current-year BNEF data hasn't been published yet, it holds at the most recent available year with a note in the output — no silent fallbacks.

**BNEF data seeded:**

| Year | Global Grid Investment | YoY |
|------|----------------------|-----|
| 2023 | $344B | — |
| 2024 | $403B | +17% |
| 2025 | $470B | +16% (baseline = 100) |

Two consecutive years of double-digit growth. Data centers are the stated demand driver.

---

### The Numbers

As of March 4, 2026:

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Token energy index | 222.47 | 222.47 | — |
| Grid investment index | — | 100.00 | — |
| **Blended energy index** | 222.47 | **185.73** | **-36.74 pts** |
| AIU Index | 233.35 | 229.68 | -3.67 pts |

The AIU shift is modest (energy is only 10% of the basket) but real. The energy index shift is significant: 37 points. That gap is not noise — it's the first clean read on the divergence between AI activity growth and grid infrastructure growth.

---

### What It Revealed

The numbers are telling a story the token-only index couldn't see.

AI token volumes have grown 135% since the January 2025 baseline (token_index: 235.12). Global grid investment in that same period grew 16% — the 2025 BNEF figure of $470B is the baseline, so the grid investment index currently sits at 100.0.

That's a 2.35× increase in AI demand against roughly 1.16× increase in infrastructure investment. The energy index now reflects this decoupling rather than masking it.

This isn't a bug in the methodology. It's an observation about the structure of the AI buildout: **demand is lapping infrastructure**. Grid investment is growing at a strong absolute rate — $470B is not a small number — but AI compute consumption is growing far faster. The blended energy index captures this tension in a way the pure token proxy never could.

When BNEF publishes 2026 figures, the direction of the grid investment index will be the most informative single data point for the energy component. If grid capex accelerates (e.g., +20%+ driven by hyperscaler commitments), the gap narrows. If it holds pace (~16%), the demand-supply gap in the energy index widens further.

---

### Limitations

**Annual lag.** The grid investment index is fixed at 100.0 until BNEF publishes 2026 data. This means the energy component's independent signal is static for much of the year. It moves once annually. This is a known constraint — the alternative (fabricating intra-year estimates without published data) would be worse.

**Attribution is broad.** BNEF's figures cover global grid investment across all sectors: renewables interconnection, transmission expansion, distribution upgrades, and data center-driven load growth. The AI-attributable fraction is not broken out. We're using total grid capex as a proxy for AI infrastructure pressure, which is directionally correct but not precisely attributed.

**Blend weight is a choice.** The 70/30 split between token-derived proxy and grid investment is a methodological judgment, not an empirically derived weighting. It was calibrated to be meaningful without allowing an annual, static data point to dominate a weekly-updating index. This should be revisited as more annual data points accumulate.

---

### What We're Watching

**BNEF 2026 grid investment figures.** This is the most important update. When published (typically mid-year), it will give the grid index its first movement since baseline. If data center demand is driving investment above the 16% trend, the blended energy index will rise faster than the token proxy. If grid capex growth decelerates (permitting constraints, capital allocation shifts), it provides a genuine drag on the energy component independent of token volumes.

**IEA datacenter energy baseline.** The IEA publishes annual datacenter-specific energy consumption data, which would allow more precise attribution than the broad BNEF grid investment figure. Adding this as a third sub-component — or replacing BNEF with it — would improve accuracy. The data is annual, so the structural limitation remains.

**Hyperscaler capex disclosures.** Google, Microsoft, and Amazon all publish infrastructure capex in their quarterly reports. This is higher-frequency (quarterly) and more directly AI-attributable than grid investment. It could eventually replace or supplement the annual BNEF figure for more responsive tracking.

**The demand-infrastructure gap.** If token volumes continue compounding at the current rate (235+ index, 135% above baseline in 14 months) while grid investment grows at 16% annually, the blended energy index will increasingly diverge from the token index over time. That divergence is itself a signal worth tracking — it would indicate persistent infrastructure scarcity relative to AI demand, which has implications for future compute costs and availability.

---

### Methodology Note

This AAR documents the change committed on 2026-03-04 (commit `47f1931`). Changed files: `src/calculate_aeai.py`, `data/aeai/grid_investment.json`, `data/aeai/baseline.json`, `AEAI_README.md`, `aeai.html`. Top-level AEAI formula weights (0.6/0.3/0.1) unchanged. No historical recalculation was performed; baseline grid_investment_index = 100.0 by definition.

Source: BloombergNEF *Grid Investment Outlook*. Numbers as of 2026-03-04 data snapshot.

---

*Occupant Index · March 2026*
*Contact: hello@occupant.ee*
