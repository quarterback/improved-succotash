# Chief Product Officer / VP of Product
## [Boutique Economic Intelligence Firm] — Remote, ≤15 People

---

### About the Firm

We are a small, mission-driven economic intelligence firm operating at the intersection of AI markets, data infrastructure, and public-good research. Our products are economic indices — not dashboards, not SaaS tools, not AI wrappers. We instrument the AI economy with the same rigor historically applied to commodities, energy, and consumer goods: published methodology, reproducible outputs, zero tracking, no premium tiers.

We operate on a Parachute Commons license. We do not optimize for fundraising optics. We optimize for being correct. Our infrastructure runs for approximately $50/month and produces daily-updated, institutionally-citable economic data. That constraint is a feature, not a limitation — it forces the product to be durable, not dependent.

We are a small team. Everyone here is technical enough to interrogate the numbers and honest enough to publish what they find even when it complicates the narrative. If you have ever deliberately removed a claim from a product because you could not prove it, you will fit here.

---

### The Role

**Title:** Chief Product Officer or VP of Product (negotiated based on experience and firm structure at time of hire)

You would be the first and only dedicated product executive at this firm. You report directly to the founder. You own the full product lifecycle — not just roadmap and prioritization, but methodology design, data source governance, index construction logic, and the public-facing presentation of findings. The product here is as much a published methodology as it is software.

This is not a role for someone who inherits a mature product and optimizes it. This is a role for someone who has already built something like this from nothing and knows what that takes — the early ambiguity, the dead-end data sources, the methodological debates you have with yourself at 11 PM, and the discipline to ship something intellectually honest before it is perfect.

---

### What We Expect You Have Already Built

This is a post-hoc description. The qualifications below are not aspirational — they describe the work. A strong candidate will read this section and recognize themselves in it.

**Economic Index Construction from Scratch**

You have conceived and shipped at least one original economic index — not configured a BI tool, not tuned a pre-existing model, but designed and implemented a new measurement methodology for a domain where no prior standard existed. You know what it means to choose a base period, justify a normalization approach, and defend your weighting decisions in writing to people who will look for holes.

Specifically, you have built or are capable of building:

- **A quality-adjusted price index** across a heterogeneous asset class. You understand hedonic pricing logic and can apply it outside its traditional domains. You know why a raw price comparison across models of wildly different capability is analytically meaningless, and you have built the tier architecture to make the comparison meaningful. ($CPI analog: 2,188+ models, 4 capability tiers — BULK, FRONT, JUDGE, LCTX — and 3 workload sub-indices: $START, $AGENT, $THRU. Base: Feb 2025 = 100. Current reading: ~62.4, a 38% deflationary move.)

- **A composite activity index** using heterogeneous inputs and explicit weighting rationale. You have modeled something analogous to an IMF Special Drawing Rights basket — taken three or more incommensurable signals, normalized each independently, and combined them at defensible weights. You have documented why 60/30/10 (or whatever your weights are) and can explain what would have to be true about the world for those weights to be wrong. ($AIU / AEAI analog: 60% token throughput, 30% inferred spend, 10% energy proxy. Current reading: ~503.65 — a 5× expansion from baseline.)

- **A labor impact or displacement index** that holds the methodological line between structural potential and observed substitution. You have worked with occupational classification systems (SOC codes, PSC codes, or equivalents), government workforce data (BLS OEWS, OPM FedScope, or equivalents), and procurement data (FPDS/USAspending or equivalents). You know how to build a crosswalk and you know why conflating *could replace* with *is replacing* is an analytical error with real policy consequences. ($LDI analog: 9 federal workload categories, structural cost ratio ~37,000× human vs. AI, observable substitution rate ~3.86%.)

**Iterative Methodology Development**

You did not ship a methodology once and call it done. You publish after-action reports. You recalibrate when the data changes in ways that reveal a prior assumption was wrong. You treat index revision as a normal part of the product lifecycle, not an admission of failure. You have a track record of monthly or quarterly reviews where you ask: what did we get right, what did the index miss, and what would we change if we were designing it today?

---

### Technical Footprint

You do not need to be a software engineer. You need to have built something that works and to understand every line of it — or to have collaborated with agents to produce it and to own the result intellectually.

**Data Pipeline Architecture**

- Python 3.11 (or equivalent): fetcher modules for external APIs (model pricing registries, federal procurement databases, labor statistics), calculation engines, and historical backfill scripts. Comfort with ~7,000+ lines of production pipeline code, not as an author necessarily, but as an owner who can audit and modify it.
- Data source integration: OpenRouter, LiteLLM, BLS OEWS/ECEC, FPDS/USAspending, OPM FedScope, BloombergNEF grid investment data, or comparable primary sources. You know what an API rate limit looks like when it breaks your morning run and you know how to handle it without introducing silent data gaps.
- CI/CD philosophy: GitHub Actions (or equivalent) on a daily cron, writing outputs to JSON, with git history serving as the immutable audit log. You understand why the pipeline design matters as much as the index formula — a result you cannot reproduce is not a result.

**Frontend and Data Presentation**

- Static-site delivery: vanilla HTML/CSS/JavaScript, Progressive Web App patterns, offline capability, zero external dependencies, zero tracking. You have strong opinions about why a product that is meant to be trusted should not load 14 third-party scripts.
- Interactive calculators: you have built or overseen the construction of user-facing tools that let practitioners apply your index to their own inputs (cost calculators, pricing reasonableness worksheets, workload absorption classifiers). You know the difference between a calculator that illustrates a methodology and one that obscures it.

**Infrastructure Philosophy**

- Deterministic algorithms over stochastic models. When the answer needs to be auditable and reproducible, you do not reach for ML. You can articulate the design decision.
- Static infrastructure at minimal cost. You have demonstrated that rigor does not require scale, and that a product costing $50/month can produce outputs that institutional users cite and trust.

---

### Analytics and Research Skills

- **Hedonic pricing and quality adjustment** applied outside traditional CPI domains
- **Composite index construction** including normalization, weighting rationale, and sensitivity analysis
- **Labor market crosswalk methodology**: mapping occupational classifications across taxonomies (SOC, PSC, O*NET, or equivalents) to enable cost and substitution analysis
- **Government procurement analytics**: FAR-equivalent pricing reasonableness determinations, FPDS data interpretation, federal workforce cost modeling
- **Cognitive arbitrage analysis**: identifying and quantifying capability-to-cost inefficiencies in compute markets — the ability to characterize what 90% of frontier capability costs at 40% of frontier price
- **Intellectual honesty as a product constraint**: you have removed claims from a product because you could not substantiate them. You treat *could* and *is* as categorically different. You document your assumptions and surface your limitations in the product itself, not buried in a footnote.

---

### Human-Agent Collaboration

This section is where this role diverges from every other product leadership job description you have read this year.

This product was not built by a team of engineers following a product manager's PRD. It was built through iterative human-agent co-development — a working methodology in which a human with deep domain expertise and editorial judgment collaborates with AI agents as thought partners, research accelerants, and implementation partners, across the full arc from concept to shipped product.

We are looking for someone who has already done this and can do it again, better.

**What that means in practice:**

- You have used AI agents not as autocomplete but as collaborators — for index design debate, methodology stress-testing, data source evaluation, and draft-to-production iteration on both code and documentation.
- You maintain intellectual ownership of the output. You are the editorial layer. You know when an agent is hallucinating a data source, compressing a nuance into a falsehood, or proposing a methodology that sounds rigorous and is not. You catch it because you know the domain, not because you ran a check.
- You have built workflows where human judgment gates agent execution. The agent proposes; you decide. You have not outsourced the decision-making — you have accelerated the research, drafting, and prototyping that precedes it.
- You iterate rapidly. Human-agent co-development compresses the gap between "I have an idea for an index" and "the index is live, reproducible, and documented." You have shipped things in days that would have taken a traditional team weeks, without sacrificing analytical integrity.
- You understand the failure modes. You have seen agents confidently produce wrong numbers, cite nonexistent sources, and miss the methodological point entirely. You have built habits and checkpoints that catch these before they reach the product.

This is the emerging primitive of product leadership in an agent-native world: not "manages a team of engineers" but "orchestrates human+agent pipelines to ship high-integrity knowledge products." If you have done this, you know it is a distinct skill. If you have not, this role will teach it to you — but we are hoping you arrive having already learned.

---

### Leadership and Culture Fit

- **You write the roadmap.** There is no roadmap waiting for you. There is a body of work, a set of open questions, and a set of users who have found the product and started relying on it. You decide what comes next and you defend that decision with evidence.
- **You are comfortable with public rigor.** Our methodology is public. Our data is public. Our after-action reports are public. When we are wrong, we say so publicly. You do not treat transparency as a liability.
- **You span disciplines without living in any one.** You can have a real conversation with an economist about index construction, with an engineer about pipeline design, with a policy analyst about labor displacement methodology, and with a procurement officer about FAR compliance. You do not need to be the expert in all of these — but you need to be able to hold the conversation and know when you are out of your depth.
- **You treat methodology as a product artifact.** The index formula, the data source list, the weighting rationale, the known limitations — these are part of the product, not supplementary documentation. You maintain them with the same care as the code.
- **You have a public-good orientation.** You can articulate why this work matters beyond the revenue it generates. You care about whether the index is correct, not just whether it is defensible.
- **You operate at the frontier of a domain that does not have settled answers.** The AI compute market is moving fast enough that the right methodology today may need to be revised next quarter. You are comfortable in that environment and you do not mistake the absence of a prior standard for a reason not to build one.

---

### What We Are Not Looking For

- A generalist product manager who will learn the domain on the job. The domain expertise is the job.
- Someone whose primary credential is having managed large teams. This is a small firm. Headcount is not leverage here; methodology and execution are.
- Someone who needs the roadmap to be handed to them. If you are looking for a defined scope and a sprint cadence, this is not the right environment.
- Someone who is comfortable with AI as a productivity tool but has not yet used it as a co-developer. The agent-collaboration component is not optional.

---

### Compensation Philosophy

We do not publish a salary range in this document because we are negotiating with a specific person for a specific role, not running a volume hiring process. What we will say:

- We pay at or above market for the right person, as defined by the actual difficulty of this work — not the title.
- Equity is on the table. Structure is negotiable.
- We do not have a large team to offer you the social proof of. We offer you the work itself, a stake in something that has institutional traction and no serious comparable, and the autonomy to build the product function from the ground up.

If the work described in this document sounds like the work you have already done — or the work you have been trying to find permission to do — we want to hear from you.

---

*This job description was written post-hoc. The qualifications describe what it took to build what we have built. The ideal candidate reads this and thinks: I have done this. The next candidate thinks: I could do this. We are looking for the first kind.*
