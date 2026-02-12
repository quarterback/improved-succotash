# Building Public Infrastructure for AI Compute Pricing

## The Problem: Opacity in a Critical Market

By early 2025, AI compute had become one of the fastest-moving markets in enterprise technology. Prices were dropping 30-40% annually. New models launched weekly. Organizations were making million-dollar procurement decisions based on scattered pricing data, vendor claims, and gut instinct.

**There was no Bloomberg Terminal for AI compute.**

If you wanted to know whether frontier model pricing was converging, whether reasoning-intensive workloads were getting cheaper, or how your AI budget compared to market benchmarks—you had to build your own tracking system. Or rely on vendors to tell you their prices were competitive.

The market needed **public infrastructure**. Not a SaaS product. Not a vendor-sponsored comparison site. Not a research report published quarterly.

**Real infrastructure:** Free, daily-updated, methodology-driven benchmarks that anyone could use to price AI procurement decisions.

## What We Built

[Occupant](https://occupant.ee) publishes two core instruments:

### 1. Compute Price Index ($CPI)
Tracks **price deflation** across AI model tiers. Like CPI for consumer goods, but for AI compute.

- **$BULK**: Budget-tier models (<$1/MTok) - the commodity layer
- **$FRONT**: Frontier capability models - top performers by Arena ELO
- **$JUDGE**: Reasoning-intensive workloads - o1/o3/R1 class models
- **$LCTX**: Long-context workloads - 128K+ context models

Updated daily from 2,300+ models. Volume-weighted baskets. February 2025 = 100 baseline.

**Use case:** "Should we lock in rates now or wait 90 days? What's the deflation trajectory?"

### 2. AI Activity Index ($AIU / AEAI)
Tracks **economic volume** through the AI market. Token throughput, inferred spend, energy consumption.

- 60% token volumes (usage intensity)
- 30% inferred spend (economic scale)
- 10% energy consumption (infrastructure load)

**Use case:** "Is AI adoption accelerating or plateauing? What's the growth rate?"

### Supporting Tools

- **Calculator**: Estimate monthly costs for specific workload patterns
- **Market Intel**: Quality-Adjusted Pricing (QAP), arbitrage opportunities, value leaders
- **Gov Benchmarks**: Public sector pricing templates with FedRAMP/audit trail requirements

All free. All updated daily. No login, no tracking, no paywalls.

## Technical Approach

### Data Sources
We don't generate pricing data—we **aggregate** it:

- **OpenRouter API**: Primary pricing across 2,300+ models
- **LiteLLM**: Cross-provider validation
- **pricepertoken.com**: Independent verification
- **OpenRouter Rankings**: Volume/market share data
- **Chatbot Arena**: Quality ratings (ELO scores)

Price discrepancies exceeding 10% trigger manual review before publication.

### Methodology
**Compute CPI**: Volume-weighted baskets by tier. Models weighted by actual usage (OpenRouter volume data), not arbitrary "popularity" scores.

**AEAI**: Composite index using principal component analysis on three data streams. Normalized to 100 at February 2025 baseline.

**Build Cost Indices** ($START, $AGENT, $THRU): Fixed workload mixes applied to current tier prices. Track costs relevant to specific use patterns.

### No Black Boxes
We don't publish the calculation code (yet), but methodology changes are documented before implementation. The indices are **deterministic algorithms**—no ML, no "secret sauce," no vendor optimization.

## Design Philosophy

### 1. Public Infrastructure, Not a Product
Occupant doesn't have a business model. It has a **mandate**: make implicit market dynamics explicit and measurable.

We don't sell subscriptions, API keys, or "premium tiers." The data is public. The indices are free. Commercial users, government agencies, researchers—everyone gets the same access.

(Commercial licensing available for derivative products under Parachute Commons.)

### 2. Zero Data Collection
From the [Tardigrade Disclosure](https://occupant.ee/tardigrade.html):

- No cookies
- No analytics
- No tracking pixels
- No third-party scripts
- Only localStorage for theme preference (client-side)

**Why?** Because public infrastructure shouldn't surveil its users. If you're using CPI data to evaluate vendor proposals, that's your business—not ours, not your vendors', not anyone's.

### 3. Parachute Commons License
Content released under [Parachute Commons 1.0](https://occupant.ee/license.html)—a tiered permissions framework:

- ✅ **Free** for personal, educational, non-commercial use
- ⚠️ **Restrictions** for commercial use (scale/automation/org size triggers)
- 🚫 **Prohibited** for surveillance, impersonation, willful attribution violations

Think Creative Commons, but with adaptive restrictions that "deploy protection as it gets closer to the ground."

### 4. No Defensive Positioning
Early drafts claimed "independence," "transparency," "no vendor bias."

We cut all of it.

**Why?** Because we're aggregating public data, not conducting original research. Claiming "independence" when you're pulling from OpenRouter/Arena/LiteLLM is overrated. Better to just explain what the data is and what you can do with it.

No protests. No credibility theater. Just: *here's what we track, updated daily from 2,300+ models.*

## Messaging Revamp: From Abstract to Concrete

### Before
- "Instruments for pricing reality"
- "Decision architecture for institutions navigating AI dependency"
- "We publish what others assume"

Sounds like consulting theater. What does it *actually do*?

### After
- **"The Standard for AI Compute Value"**
- "Real-time benchmarks for AI costs and market activity"
- "Track price deflation, measure economic growth, make data-driven procurement decisions"

**The shift:** From philosophy to utility. From abstract concepts to concrete actions.

### Meta Tags
**Before:**
```
Public benchmarks and decision architecture for
institutions navigating AI dependency. We publish
what others assume.
```

**After:**
```
Real-time benchmarks for AI costs and market activity.
Track price deflation and economic growth. Updated
daily from 2,300+ models.
```

Specific. Scannable. SEO-friendly. Tells you *exactly* what you get.

### About Page
**Before:**
> "We create public instruments that make implicit assumptions explicit and measurable. Indices, frameworks, and analysis that reprice what institutions think is real."

**After:**
> "We aggregate AI pricing and activity data into benchmarks you can track over time. Public data infrastructure for procurement, budgeting, and strategic planning."

**The test:** If a procurement officer lands on the site with 30 seconds to evaluate it, do they understand what it does and why it matters?

Now: Yes.

## Impact & Use Cases

### 1. Procurement Decisions
**Before Occupant:** "Our vendor says this is competitive pricing."

**With Occupant:** "Their quote is 23% above $FRONT baseline. Either negotiate or wait 90 days—deflation trajectory suggests 15% drop by Q3."

### 2. Budget Planning
**Before:** "We'll allocate $500K for AI compute and adjust as needed."

**With Occupant:** Use the calculator with your workload mix (40% bulk, 30% frontier, 20% reasoning, 10% long-context). Get monthly cost projections. Track AEAI to forecast demand growth.

### 3. Market Analysis
**Before:** "AI adoption is growing, probably."

**With Occupant:** AEAI up 34% quarter-over-quarter. Token volumes spiking but energy consumption flat—efficiency gains from new model architectures.

### 4. Government Procurement
Use Gov Benchmarks worksheet with FedRAMP-equivalent pricing, audit trail requirements, and public sector workload assumptions. Submit as part of RFP evaluation.

### 5. API Selection
**Market Intel** shows Quality-Adjusted Pricing (QAP) and arbitrage opportunities. Find models that deliver 90% of frontier performance at 40% of the cost.

## What We Learned

### 1. Kill Your Jargon
Every abstract phrase ("decision architecture," "instruments for pricing reality") was a barrier. When we cut them, clarity emerged.

**The rule:** If you can't explain it to a procurement officer in one sentence, rewrite it.

### 2. Don't Claim What You Can't Prove
We removed "independent" and "transparent" claims because:
- We aggregate public data (not independent research)
- We don't publish calculation code (not fully transparent)

**Better to under-promise than over-claim.** Users care about utility, not positioning.

### 3. Public Infrastructure Doesn't Need a Business Model (Yet)
Occupant costs ~$50/month to run (hosting, domain, API calls). No monetization pressure. No investor deck. No growth targets.

**Result:** Design decisions optimize for **usefulness** instead of engagement, retention, or conversion.

Could we charge for API access? Sure. Do we need to? No. Will we if it scales? Maybe. But starting free removes a huge design constraint.

### 4. Governance Through Culture, Not Just Contracts
Parachute Commons relies on "community reporting, good-faith compliance, and cultural enforcement" as much as legal mechanisms.

**Why it works:** Most misuse isn't malicious—it's ignorance. Clear expectations + low-friction permission requests > adversarial legal frameworks.

### 5. Zero Tracking Is a Feature
Tardigrade Disclosure isn't just compliance theater—it's a **competitive advantage.**

If you're evaluating AI vendors and your usage patterns leak to analytics platforms, that data could flow back to vendors. With Occupant, your research stays private.

**Public infrastructure shouldn't surveil its users.**

## Technical Decisions That Mattered

### 1. Static Site + Daily Regeneration
No database. No server-side rendering. Just:
- Daily cron job fetches pricing data
- Regenerates JSON files
- Deploys updated static site

**Benefits:**
- Fast (CDN everywhere)
- Cheap (hosting costs ~$10/month)
- Resilient (no database to corrupt)
- Auditable (git history = full data lineage)

### 2. Client-Side Rendering for Interactive Tools
Calculator and Market Intel use vanilla JavaScript + Fetch API. No framework bloat.

**Why:** Tools are simple enough that React/Vue overhead isn't justified. Faster page loads, no build pipeline complexity.

### 3. Progressive Web App (PWA)
Service worker caches data for offline access. Add to home screen on mobile.

**Use case:** Procurement officer in a vendor meeting with spotty wifi can still pull up current CPI data.

### 4. Theme Toggle (Light/Dark)
Only localStorage usage on the site. Respects system preference by default.

**Why it matters:** Accessibility. Government users often work in high-security environments with strict display policies.

## What's Next

### Short Term
- **Historical data API**: JSON endpoint for programmatic access
- **Export tools**: CSV/Excel downloads for budget modeling
- **Alerting**: Email notifications when indices move >5% in 7 days

### Medium Term
- **Regional pricing**: Separate indices for US/EU/Asia
- **Provider-specific indices**: Track AWS Bedrock vs. OpenAI vs. Anthropic vs. open source
- **Energy transparency**: kWh per million tokens, carbon intensity by provider

### Long Term
- **Open methodology**: Publish calculation code, allow community contributions
- **Governance framework**: Multi-stakeholder oversight for methodology changes
- **Derivative works**: Encourage others to build tools using CPI/AEAI data

## Conclusion: Infrastructure Over Products

Most AI tools are **products**—designed to extract value through subscriptions, lock-in, or data capture.

Occupant is **infrastructure**—designed to provide common-pool resources that anyone can build on.

The difference:
- **Products** optimize for growth, retention, monetization
- **Infrastructure** optimizes for reliability, accessibility, transparency

We're not trying to be the Bloomberg Terminal for AI. We're trying to be the **yield curve, published daily, for free, because markets work better when pricing is public.**

If you're making AI procurement decisions, you shouldn't have to trust vendor claims or build your own tracking system.

**You should have public benchmarks. Now you do.**

---

## Links
- **Site**: [occupant.ee](https://occupant.ee)
- **License**: [Parachute Commons 1.0](https://occupant.ee/license.html)
- **Disclosure**: [Tardigrade Transparency](https://occupant.ee/tardigrade.html)
- **Contact**: hello@occupant.ee

---

*This case study documents the design, philosophy, and technical decisions behind Occupant—public infrastructure for AI compute pricing. February 2026.*
