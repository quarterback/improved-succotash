import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

const GovBenchmarks = () => {
  // Government pricing tiers (per seat, per month)
  const pricingTiers = [
    {
      tier: "Light",
      seats: "1-100",
      perSeat: 49,
      features: [
        "Standard models (GPT-3.5, Claude Haiku)",
        "100K tokens/seat/month",
        "Email support",
        "Standard SLA (99% uptime)",
      ],
    },
    {
      tier: "Standard",
      seats: "101-500",
      perSeat: 89,
      features: [
        "General purpose models (GPT-4, Claude Sonnet)",
        "500K tokens/seat/month",
        "Priority email + chat support",
        "Enhanced SLA (99.5% uptime)",
        "Usage analytics",
      ],
    },
    {
      tier: "Heavy",
      seats: "501-2,000",
      perSeat: 149,
      features: [
        "All models including frontier (o1, Claude Opus)",
        "2M tokens/seat/month",
        "24/7 dedicated support",
        "Premium SLA (99.9% uptime)",
        "Advanced analytics + reporting",
        "Custom model fine-tuning",
      ],
    },
    {
      tier: "Enterprise",
      seats: "2,000+",
      perSeat: "Custom",
      features: [
        "Unlimited model access",
        "Custom token allocations",
        "White-glove onboarding",
        "Custom SLA negotiations",
        "Dedicated account team",
        "On-premise deployment options",
        "Federal compliance (FedRAMP, IL5)",
      ],
    },
  ];

  // Benchmark comparison data
  const benchmarks = [
    {
      category: "Knowledge Worker (Light)",
      govRate: 49,
      marketAvg: 75,
      savings: 35,
    },
    {
      category: "Analyst (Standard)",
      govRate: 89,
      marketAvg: 135,
      savings: 34,
    },
    {
      category: "Developer (Heavy)",
      govRate: 149,
      marketAvg: 220,
      savings: 32,
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Header */}
      <section className="pt-32 md:pt-40 pb-16 section-padding border-b border-divider">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Government Pricing
            </span>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
              Public Sector Benchmarks
            </h1>
            <p className="text-lg md:text-xl text-text-secondary max-w-2xl">
              Transparent per-seat pricing for government agencies and public institutions. 
              No hidden costs, no vendor lock-in.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pricing Tiers */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Tier-Based Pricing"
            subtitle="Monthly rates per seat. Volume discounts applied automatically based on total headcount."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {pricingTiers.map((tier, i) => (
              <motion.div
                key={tier.tier}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className={`border border-divider p-6 ${
                  tier.tier === "Standard" ? "bg-surface-elevated/50 ring-2 ring-foreground/20" : ""
                }`}
              >
                <div className="mb-4">
                  <h3 className="text-xl font-bold mb-1">{tier.tier}</h3>
                  <p className="text-xs text-text-tertiary font-mono uppercase tracking-wider">
                    {tier.seats} seats
                  </p>
                </div>

                <div className="mb-6">
                  <div className="flex items-baseline gap-1">
                    {typeof tier.perSeat === "number" ? (
                      <>
                        <span className="text-3xl font-bold tabular-nums">${tier.perSeat}</span>
                        <span className="text-sm text-text-tertiary">/seat/month</span>
                      </>
                    ) : (
                      <span className="text-2xl font-bold">{tier.perSeat}</span>
                    )}
                  </div>
                </div>

                <ul className="space-y-3 text-sm">
                  {tier.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-foreground mt-0.5">✓</span>
                      <span className="text-text-secondary">{feature}</span>
                    </li>
                  ))}
                </ul>

                {tier.tier === "Standard" && (
                  <div className="mt-6">
                    <span className="inline-block px-2 py-1 text-xs font-medium bg-foreground text-background">
                      MOST POPULAR
                    </span>
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          <div className="mt-10 flex gap-4">
            <Button variant="hero" size="lg">
              <Download size={16} />
              Download Rate Card (PDF)
            </Button>
            <Button variant="hero-outline" size="lg">
              Request Custom Quote
            </Button>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Benchmark Comparison */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Benchmark vs. Market Rates"
            subtitle="How government pricing compares to commercial per-seat rates for AI tools."
          />

          <div className="border border-divider">
            <div className="grid grid-cols-4 border-b border-divider bg-secondary/50 py-3 px-4">
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider">Category</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Gov Rate</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Market Avg</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Savings</span>
            </div>
            
            {benchmarks.map((benchmark, i) => (
              <motion.div
                key={benchmark.category}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="grid grid-cols-4 border-b border-divider last:border-b-0 py-4 px-4 hover:bg-secondary/30 transition-colors"
              >
                <span className="font-medium">{benchmark.category}</span>
                <span className="font-mono text-sm tabular-nums text-right font-semibold">
                  ${benchmark.govRate}
                </span>
                <span className="font-mono text-sm tabular-nums text-right text-text-secondary">
                  ${benchmark.marketAvg}
                </span>
                <span className="font-mono text-sm tabular-nums text-right text-index-positive font-semibold">
                  {benchmark.savings}%
                </span>
              </motion.div>
            ))}
          </div>

          <p className="text-sm text-text-tertiary mt-6">
            * Market averages based on commercial AI tool subscriptions (ChatGPT Plus, Claude Pro, GitHub Copilot, etc.) 
            as of {new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}.
          </p>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Budget Worksheet */}
      <section className="py-16 md:py-24 section-padding bg-secondary/30">
        <div className="content-max max-w-3xl">
          <h2 className="text-3xl font-bold mb-6">Budget Planning Worksheet</h2>
          
          <div className="border border-divider bg-background p-8">
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4 pb-4 border-b border-divider">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Total Employees
                  </label>
                  <input 
                    type="number" 
                    className="w-full px-3 py-2 border border-divider bg-background"
                    placeholder="e.g., 500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Expected AI Users (%)
                  </label>
                  <input 
                    type="number" 
                    className="w-full px-3 py-2 border border-divider bg-background"
                    placeholder="e.g., 60"
                  />
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold">Tier Distribution</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-text-tertiary mb-1">Light Users</label>
                    <input 
                      type="number" 
                      className="w-full px-3 py-2 border border-divider bg-background text-sm"
                      placeholder="50%"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-text-tertiary mb-1">Standard Users</label>
                    <input 
                      type="number" 
                      className="w-full px-3 py-2 border border-divider bg-background text-sm"
                      placeholder="40%"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-text-tertiary mb-1">Heavy Users</label>
                    <input 
                      type="number" 
                      className="w-full px-3 py-2 border border-divider bg-background text-sm"
                      placeholder="10%"
                    />
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-divider">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="font-semibold">Estimated Monthly Cost</span>
                  <span className="text-2xl font-bold tabular-nums">$--,---</span>
                </div>
                <div className="flex justify-between items-baseline">
                  <span className="text-sm text-text-tertiary">Annual Budget Required</span>
                  <span className="font-mono font-semibold tabular-nums">$---,---</span>
                </div>
              </div>

              <Button variant="hero" className="w-full mt-6">
                <Download size={16} />
                Download as Excel
              </Button>
            </div>
          </div>

          <div className="mt-8 border-l-2 border-foreground pl-4">
            <h3 className="font-semibold mb-2">Procurement Notes</h3>
            <ul className="text-sm text-text-secondary space-y-2">
              <li>• All government pricing includes federal compliance certifications (FedRAMP, IL4/IL5 available)</li>
              <li>• Volume discounts auto-applied; no negotiation required for standard tiers</li>
              <li>• Month-to-month contracts available; no annual commitment required</li>
              <li>• Usage data exportable for FISMA compliance reporting</li>
              <li>• Data residency options: US-only, specific regions, or on-premise</li>
            </ul>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default GovBenchmarks;
