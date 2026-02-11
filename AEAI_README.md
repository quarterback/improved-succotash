# AI Economic Activity Index (AEAI)

## Overview

The **AI Economic Activity Index (AEAI)** measures global AI economic activity through a synthetic unit called the **AIU** (AI Unit), modeled on the IMF's Special Drawing Rights (SDR). Unlike the Compute CPI which tracks price deflation, the AEAI tracks actual economic activity volume in the AI sector.

**Key Characteristics:**
- **Base Period**: February 2025 = 100
- **Update Frequency**: Daily (automated via GitHub Actions)
- **Basket Composition**: 60% token volumes, 30% inferred spend, 10% energy proxy

## Methodology

### Basket Components

The AIU is a weighted composite of three activity indicators:

| Component | Weight | Description | Data Source |
|-----------|--------|-------------|-------------|
| **Token Volumes** | 60% | Total weekly tokens processed across models | OpenRouter rankings |
| **Inferred Spend** | 30% | Calculated as tokens × blended pricing | Volumes × pricing data |
| **Energy Proxy** | 10% | Estimated from token volumes and model tiers | Tier-based efficiency factors |

### Calculation Formula

Each component is normalized to its February 2025 baseline value (= 100), then weighted:

```
AIU = 0.6 × token_index + 0.3 × spend_index + 0.1 × energy_index
```

Where:
- `token_index = (current_tokens / baseline_tokens) × 100`
- `spend_index = (current_spend / baseline_spend) × 100`
- `energy_index = (current_energy / baseline_energy) × 100`

### Component Details

#### 1. Token Volumes (60% weight)

**Data Source**: OpenRouter public rankings
- Weekly token throughput aggregated across all tracked models
- Currently tracking 64+ models from major providers
- Normalized to standard token counting (tiktoken-equivalent)

**Calculation**:
```python
total_tokens = sum(model.tokens_weekly for model in rankings)
token_index = (total_tokens / baseline_tokens) × 100
```

#### 2. Inferred Spend (30% weight)

**Data Source**: Calculated from token volumes × current pricing

**Pricing Data Sources** (in priority order):
1. OpenRouter API (live pricing)
2. LiteLLM GitHub (comprehensive catalog)
3. llm-prices.com (historical tracking)
4. pricepertoken.com (per-token API)

**Blended Pricing Assumption**:
- Typical AI workload mix: 70% input tokens, 30% output tokens
- `blended_price = 0.7 × input_price + 0.3 × output_price`

**Calculation**:
```python
for model in rankings:
    tokens_millions = model.tokens_weekly / 1_000_000
    blended_price = 0.7 × model.input_price + 0.3 × model.output_price
    spend += tokens_millions × blended_price

spend_index = (spend / baseline_spend) × 100
```

**Price Coverage**:
- Exact pricing: ~40% of models (direct API/catalog matches)
- Estimated pricing: ~60% of models (tier-based averages)

#### 3. Energy Proxy (10% weight)

**Data Source**: Estimated from token volumes and model efficiency tiers

**Energy Factors** (Joules per billion tokens):
| Tier | Energy Factor | Examples |
|------|---------------|----------|
| Budget | 50,000 J/B | Flash, Haiku, Mini |
| General | 150,000 J/B | Sonnet, GPT-4o |
| Frontier | 300,000 J/B | Opus, GPT-5 |
| Reasoning | 500,000 J/B | o1, o3-mini, DeepSeek-R1 |

**Calculation**:
```python
for model in rankings:
    tier = classify_model_tier(model)
    energy_factor = ENERGY_FACTORS[tier]
    tokens_billions = model.tokens_weekly / 1e9
    energy += tokens_billions × energy_factor

energy_index = (energy / baseline_energy) × 100
```

**Important Note**: These are preliminary estimates based on published GPU efficiency metrics and datacenter PUE. Future versions will integrate:
- IEA annual datacenter energy baseline
- Hyperscaler sustainability reports (Google, Microsoft, Amazon)
- AI-attributed energy disclosures where available

## Data Files

### Generated Outputs

```
data/aeai/
├── baseline.json          # Immutable Feb 2025 baseline (AIU = 100)
├── latest.json            # Current AIU index with full breakdown
├── historical.json        # Time series of all AIU snapshots
└── aeai_YYYY-MM-DD.json   # Daily snapshots
```

### Output Format

**`latest.json` structure**:
```json
{
  "date": "2026-02-11",
  "generated_at": "2026-02-11T21:02:48.636557+00:00",
  "aiu_index": 100.0,
  "components": {
    "token_index": 100.0,
    "spend_index": 100.0,
    "energy_index": 100.0
  },
  "weights": {
    "tokens": 0.6,
    "spend": 0.3,
    "energy": 0.1
  },
  "contribution": {
    "tokens": 60.0,
    "spend": 30.0,
    "energy": 10.0
  },
  "activity": {
    "tokens_weekly": 13909678380565,
    "spend_usd_weekly": 46629717.81,
    "energy_gwh_weekly": 0.000591
  },
  "data_quality": {
    "total_models": 64,
    "price_matched": 26,
    "price_estimated": 38,
    "freshness": "2026-02-04T14:45:29.614141+00:00"
  },
  "methodology": {
    "basket": "60% token volumes, 30% inferred spend, 10% energy proxy",
    "baseline": "February 2025 = 100",
    "spend_calculation": "tokens × blended_price (70% input, 30% output)",
    "energy_estimation": "Token volumes × tier-based efficiency factors"
  }
}
```

## Running the Calculator

### Manual Execution

```bash
python src/calculate_aeai.py
```

This will:
1. Load latest rankings data from OpenRouter
2. Load pricing data from multiple sources
3. Calculate token volumes, spend, and energy
4. Compute the AIU index
5. Save snapshots to `data/aeai/`
6. Update historical time series

### Automated Execution

The AEAI calculator runs automatically via GitHub Actions:
- **Schedule**: Daily at 6:00 AM UTC
- **Workflow**: `.github/workflows/update-data.yml`
- **Dependencies**: rankings data, pricing data
- **Output**: Committed to `data/aeai/` directory

## Coverage and Limitations

### Current Coverage

**Geographic Scope**:
- Currently captures activity visible through OpenRouter's public rankings
- Represents a significant but incomplete view of global AI activity

**Not Captured**:
- Direct API usage (OpenAI, Anthropic, Google direct customers)
- Enterprise deployments and private instances
- Closed-loop systems (e.g., internal company AI)
- Regional providers not routed through OpenRouter

### Known Limitations

1. **Token Volume Sampling Bias**
   - Only tracks OpenRouter traffic (~5-10% of global AI inference estimated)
   - May over-represent certain model families or use cases
   - Subject to OpenRouter's routing patterns and client mix

2. **Pricing Estimation**
   - ~60% of models use tier-based price estimates
   - Assumes 70/30 input/output ratio (varies by actual workload)
   - Doesn't account for volume discounts or enterprise pricing

3. **Energy Estimation**
   - Preliminary factors based on published benchmarks
   - Doesn't account for:
     - Actual datacenter PUE variations
     - Model-specific training vs inference efficiency
     - Hardware generation (A100 vs H100 vs custom ASICs)
   - Awaiting integration of real energy data from IEA and hyperscalers

4. **Temporal Lag**
   - Rankings data updated daily but reflects prior week's activity
   - Pricing data may lag real-time changes by hours
   - Energy factors are static (not dynamically updated)

## Future Enhancements

### Planned Improvements

1. **Geographic Attribution Layer**
   - Track AIU by region (US, EU, APAC)
   - Provider concentration metrics by geography
   - Regional activity growth rates

2. **Enhanced Energy Data**
   - Integrate IEA datacenter energy baseline
   - Parse hyperscaler sustainability reports (Google, Microsoft, Amazon)
   - Model-specific energy efficiency factors from published benchmarks

3. **Additional Data Sources**
   - State of AI report quarterly disclosures
   - Google tokens/minute metrics
   - Microsoft Azure AI token volumes
   - Anthropic/OpenAI quarterly reports (when available)

4. **Subindices**
   - $ACTIVITY-CODE: Coding-specific activity (GitHub Copilot, Cursor, etc.)
   - $ACTIVITY-CHAT: Conversational AI activity
   - $ACTIVITY-AGENT: Agentic workload activity

5. **Real-Time Updates**
   - Hourly snapshots (if data sources support)
   - Streaming data integration
   - Live dashboard with sub-daily granularity

## Interpretation Guide

### What the AIU Measures

The AIU is a **volume index**, not a price index:
- **Rising AIU**: Increasing AI economic activity (more tokens, higher spend, more energy)
- **Falling AIU**: Decreasing AI economic activity
- **AIU = 100**: Activity level matches February 2025 baseline

### Relationship to Compute CPI

The AIU and Compute CPI are complementary:

| Metric | Measures | Direction |
|--------|----------|-----------|
| **Compute CPI** | Unit cost of AI work | Deflating (prices falling) |
| **AIU (AEAI)** | Total AI activity volume | Expected to grow (more usage) |

**Combined Analysis**:
- **Total AI spending** = AIU × Compute CPI (roughly)
- If AIU grows 50% and CPI falls 20%, total spending grows ~20%
- Tracks both efficiency gains (CPI) and demand growth (AIU)

### Use Cases

1. **AI Market Sizing**: Track global AI compute market growth
2. **Investment Analysis**: Gauge AI infrastructure demand
3. **Policy Planning**: Understand AI economic footprint
4. **Energy Planning**: Forecast datacenter energy demand
5. **Trend Analysis**: Identify growth acceleration/deceleration

## Technical Details

### Dependencies

```bash
pip install requests  # For API calls (if needed)
```

No external API keys required for basic operation (all data sources are public).

### File Structure

```
src/
└── calculate_aeai.py       # Main AEAI calculation module

data/
├── aeai/                   # AEAI output directory
├── rankings/               # Token volume data (input)
├── prices/                 # Pricing data (input)
└── models/                 # Tier classifications (input)
```

### Key Functions

- `load_rankings()`: Load token volume data
- `load_prices()`: Load pricing data
- `calculate_weekly_spend()`: Compute spend from volumes × prices
- `calculate_energy_proxy()`: Estimate energy from volumes and tiers
- `calculate_aeai()`: Compute weighted AIU index
- `save_aeai_snapshot()`: Save timestamped output
- `update_historical()`: Append to time series

## Data Quality Metrics

Each AEAI snapshot includes data quality indicators:

```json
"data_quality": {
  "total_models": 64,           // Models tracked
  "price_matched": 26,           // Exact pricing available
  "price_estimated": 38,         // Tier-based estimates
  "freshness": "2026-02-04..."   // Source data timestamp
}
```

**Interpretation**:
- `price_matched / total_models`: Pricing coverage ratio
- `freshness`: How recent the underlying data is
- Lower coverage may increase estimation uncertainty

## Academic and Research Use

### Citation

If you use the AEAI in research or publications, please cite:

```bibtex
@misc{occupant_aeai_2025,
  title={AI Economic Activity Index (AEAI)},
  author={Occupant Index},
  year={2025},
  url={https://occupant.ee/aeai.html},
  note={Public infrastructure for AI economic activity measurement}
}
```

### Data Access

All AEAI data is public and available at:
- Live data: `https://occupant.ee/data/aeai/latest.json`
- Historical: `https://occupant.ee/data/aeai/historical.json`
- Source code: GitHub repository (link in footer)

### Methodology Transparency

Full source code and calculation logic available in:
- `src/calculate_aeai.py` (calculation engine)
- This README (methodology documentation)
- AEAI dashboard (https://occupant.ee/aeai.html)

## FAQ

**Q: Why is the AIU index at 100.0?**
A: The current data is being used as the February 2025 baseline. Once historical data is available, the index will fluctuate based on activity changes.

**Q: Why is energy consumption so low (0.0006 GWh/week)?**
A: This represents only OpenRouter traffic, which is a small fraction of global AI inference. The energy factors are also conservative estimates. Real global AI energy consumption is orders of magnitude higher.

**Q: How accurate is the spend calculation?**
A: Accuracy depends on price coverage (~40% exact, ~60% estimated) and the 70/30 input/output ratio assumption. For trend analysis, this is sufficient; for absolute spend estimates, treat as approximate.

**Q: Can I use this for my own AI project?**
A: Yes! The calculation logic is fully open-source. You can fork it, modify the weights, add custom data sources, or adapt it for your specific use case.

**Q: Will you add real-time updates?**
A: Potentially, if underlying data sources support it. Currently limited by OpenRouter rankings update frequency (daily).

## Support and Contributions

- **Issues**: Report bugs or request features via GitHub Issues
- **Contributions**: Pull requests welcome for methodology improvements, additional data sources, or documentation
- **Contact**: See https://occupant.ee/about.html

---

**Last Updated**: 2026-02-11
**Version**: 1.0
**Maintainer**: Occupant Index Team
