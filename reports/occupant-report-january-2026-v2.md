# Occupant Index Report
## January 2026

---

### The Headline

**Inference costs fell 28% year-over-year. But if you're building agents, you only saw 6%.**

The gap between commodity AI and judgment-class AI is widening. Organizations that built their 2025 budgets around "AI keeps getting cheaper" are now discovering that *their* AI isn't getting cheaper—at least not at the rate they expected.

---

### The Numbers

| Index | Value | YoY Change | What It Means |
|-------|-------|------------|---------------|
| **$CPI** | 72.4 | -27.6% | Overall inference costs down significantly |
| **$BULK** | 58.3 | -41.7% | Commodity models collapsed in price |
| **$FRONT** | 81.2 | -18.8% | Frontier models fell, but less dramatically |
| **$REASON** | 94.7 | -5.3% | Reasoning models barely moved |
| **$LCTX** | 76.1 | -23.9% | Long-context got cheaper (Gemini effect) |

**The story**: Budget-tier models (GPT-4o-mini, Gemini Flash, Haiku) dropped 40%+. But reasoning models (o1, o3-mini, DeepSeek-R1) held their price. The spread between them widened.

---

### The Cognitive Floor

The DeepSeek moment of early 2025 created a false sense of security. When DeepSeek-R1 launched at $0.55/$2.19—a fraction of o1's pricing—teams assumed the race to zero was universal. Reasoning would follow the same deflation curve as commodity inference, just with a lag.

That's not what happened.

What we're seeing now is a **cognitive floor**. Commodity inference keeps falling because it's increasingly fungible—Flash, Haiku, and GPT-4o-mini are interchangeable for most lightweight tasks, so providers compete on price. But reasoning remains scarce. The models that can actually think through multi-step problems, catch their own errors, and handle ambiguity are not commoditizing at the same rate.

The yield curve of intelligence now has a visible kink: inference is a commodity (down 40%), reasoning is a specialty service (down 5%).

---

### The Spreads

| Spread | Current | 6 Months Ago | Change |
|--------|---------|--------------|--------|
| **Cognition Premium** ($FRONT - $BULK) | 22.9 pts | 18.4 pts | +4.5 pts |
| **Judgment Premium** ($REASON - $FRONT) | 13.5 pts | 8.2 pts | +5.3 pts |

**Translation**: The premium you pay for quality over commodity increased. The premium for reasoning over standard frontier increased even more.

If your architecture assumes "we'll use the expensive model now and switch to cheaper ones later," you're betting on these spreads narrowing. They're not. They're widening.

---

### Build Cost Index

| Persona | CPI | YoY | Implication |
|---------|-----|-----|-------------|
| **$START** (Startup Builder) | 74.8 | -25.2% | Healthy deflation for product builders |
| **$THRU** (Throughput) | 61.2 | -38.8% | Massive savings for volume processing |
| **$AGENT** (Agentic Team) | 93.4 | -6.6% | Agents are barely getting cheaper |

**The $AGENT divergence is the smoking gun.** If you're running extraction pipelines, you're seeing real savings. If you're building autonomous agents that rely on reasoning models, you're essentially flat.

This matters for budgeting. A team that planned for "20% annual deflation" on their agent infrastructure is now 14 points underwater on that assumption.

---

### The Autonomy Overhead

There's a second cost pressure on agent architectures that the per-token indices don't capture: **tokens per task is rising**.

As agents get more capable, they use more reasoning tokens per task:
- Chain-of-thought gets longer and more elaborate
- Self-correction loops multiply
- Tool-use sequences extend
- Verification steps compound

Even at flat per-token prices, **cost per task is increasing** for sophisticated agent architectures.

The $AGENT index captures the per-token story. The per-task story is worse. Teams reporting "our agent costs are rising" while watching per-token prices hold steady are experiencing autonomy overhead—the hidden tax on capability.

We don't yet have good benchmarks for tokens-per-task trends. But anecdotally, agent architectures that used 5K reasoning tokens per task in mid-2025 are now using 8-12K for equivalent outcomes. That's 60-140% task-level inflation masked by flat token prices.

---

### The Position

**Thesis: Organizations investing in AI-augmented judgment are taking an unhedged position on reasoning model prices.**

Here's the problem:

1. **Procurement built budgets around deflation.** The narrative for two years has been "AI gets cheaper every quarter." That's true for commodity inference. It's not true for reasoning.

2. **Agent architectures are reasoning-heavy.** The hot deployment pattern—autonomous agents with chain-of-thought, tool use, and self-correction—runs 70%+ on reasoning-class models. That's the tier that isn't deflating.

3. **No one is tracking this exposure.** FinOps teams monitor total spend. They don't monitor *which tier* of that spend is exposed to price stickiness.

**The risk**: If reasoning model prices stay flat (or rise—DeepSeek's February 2025 price *increase* showed this is possible), organizations with agent-heavy architectures will blow their budgets while believing they're riding the deflation wave.

---

### What We're Watching

**DeepSeek-R1 pricing**: Currently $0.55/$2.19 per MTok. If this holds or drops, it puts pressure on OpenAI's o-series pricing. If it rises (as DeepSeek-V3 did in February 2025), it signals reasoning is a seller's market.

**OpenAI's next move**: o3-mini launched at $1.10/$4.40. Will o3 proper come in below o1's $15/$60? The Judgment Premium depends heavily on OpenAI's pricing strategy.

**Anthropic's reasoning play**: Claude 3.5 Sonnet handles many reasoning tasks at standard frontier prices ($3/$15). If Anthropic keeps reasoning capability in the frontier tier rather than creating a premium reasoning tier, they compress the Judgment Premium. Watch for Claude 4 positioning.

**DeepSeek V4 (Volatility Watch)**: Rumored to drop in the coming weeks. If it repeats the R1 pattern—dramatic underpricing at launch to gain market share—the Judgment Premium could compress suddenly. If it launches at price parity with o3-mini or higher, we have confirmation that reasoning pricing has found its floor. Either outcome clarifies the market structure.

---

### Methodology Note

This report uses the Occupant Index methodology. Baseline is February 2025 = 100. Data sourced from OpenRouter, LiteLLM, and historical archives. 

We track costs, not capabilities. A lower index means cheaper, not worse.

---

### The Bottom Line

The AI deflation story is real but unevenly distributed. Commodity inference is in freefall. Reasoning inference is sticky.

If your AI strategy depends on judgment-class models, your costs are not following the headline deflation numbers. The cognitive floor is real. The autonomy overhead is compounding. Plan accordingly.

---

*Occupant Index · January 2026*
*Contact: hello@occupant.ee*
