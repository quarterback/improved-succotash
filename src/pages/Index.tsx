import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import IndexTicker from "@/components/IndexTicker";
import SpreadCard from "@/components/SpreadCard";
import MemoCard from "@/components/MemoCard";
import { ArrowRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchCPIData } from "@/services/cpiData";

const Index = () => {
  // Fetch real CPI data
  const { data: cpiData, isLoading, error } = useQuery({
    queryKey: ['cpi-data'],
    queryFn: fetchCPIData,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Compute derived values from real data
  const indexData = cpiData ? {
    headline: cpiData.compute_cpi.value,
    change: cpiData.compute_cpi.mom_change || 0,
    changePercent: cpiData.compute_cpi.mom_change || 0,
  } : { headline: 0, change: 0, changePercent: 0 };

  // Map subindices from real data (using frontier, bulk, reason)
  const subIndices = cpiData ? [
    { 
      name: cpiData.subindices.frontier.name, 
      value: cpiData.subindices.frontier.value, 
      change: cpiData.subindices.frontier.mom_change, 
      changePercent: cpiData.subindices.frontier.mom_change 
    },
    { 
      name: cpiData.subindices.lctx.name, 
      value: cpiData.subindices.lctx.value, 
      change: cpiData.subindices.lctx.mom_change, 
      changePercent: cpiData.subindices.lctx.mom_change 
    },
    { 
      name: cpiData.subindices.reason.name, 
      value: cpiData.subindices.reason.value, 
      change: cpiData.subindices.reason.mom_change, 
      changePercent: cpiData.subindices.reason.mom_change 
    },
  ] : [];

  // Map spreads from real data
  const spreads = cpiData ? [
    {
      name: cpiData.spreads.cognition_premium.name,
      description: cpiData.spreads.cognition_premium.description,
      value: cpiData.spreads.cognition_premium.value,
      trend: cpiData.spreads.cognition_premium.trend === "widening" ? "up" as const : 
             cpiData.spreads.cognition_premium.trend === "narrowing" ? "down" as const : 
             "stable" as const,
    },
    {
      name: cpiData.spreads.judgment_premium.name, 
      description: cpiData.spreads.judgment_premium.description,
      value: cpiData.spreads.judgment_premium.value,
      trend: cpiData.spreads.judgment_premium.trend === "widening" ? "up" as const : 
             cpiData.spreads.judgment_premium.trend === "narrowing" ? "down" as const : 
             "stable" as const,
    },
  ] : [];

  const recentMemos = [
    {
      number: "OCC-2026-003",
      title: "The Escalation Problem: When AI Can't Say 'I Don't Know'",
      date: "Jan 2026",
      excerpt: "Decision systems require clear escalation paths. Current LLM deployments collapse this into a single probability distribution.",
      href: "/memos/escalation-problem",
      tag: "Trust Engineering",
    },
    {
      number: "OCC-2026-002",
      title: "Compute CPI Q4 Review: Judgment Costs Outpace Moore's Law",
      date: "Dec 2025",
      excerpt: "Despite hardware improvements, the cost of reliable AI judgment increased 12% YoY. Here's what that means for enterprise budgets.",
      href: "/memos/q4-review",
      tag: "Index Report",
    },
  ];

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <div className="animate-pulse text-text-secondary">Loading index data...</div>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <div className="text-text-secondary">Failed to load index data</div>
            <div className="text-sm text-text-tertiary mt-2">{error.message}</div>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="pt-32 md:pt-40 pb-20 md:pb-32 section-padding">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="max-w-4xl"
          >
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-8 text-balance">
              Instruments for
              <br />
              <span className="text-text-secondary">pricing reality.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-text-secondary max-w-2xl mb-10 leading-relaxed">
              Occupant builds public benchmarks and advisory services for institutions 
              navigating AI-dependent decision systems. We publish what others assume.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild variant="hero" size="lg">
                <Link to="/compute-cpi">
                  View Compute CPI
                  <ArrowRight size={18} />
                </Link>
              </Button>
              <Button asChild variant="hero-outline" size="lg">
                <Link to="/work">
                  Advisory Services
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Live Index Ticker */}
      <section className="py-12 bg-ticker-bg">
        <div className="section-padding content-max">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div>
              <span className="font-mono text-xs text-ticker-fg/60 uppercase tracking-wider block mb-2">
                Occupant Index — Live
              </span>
              <div className="flex items-baseline gap-4">
                <span className="text-4xl md:text-5xl font-bold text-ticker-fg tabular-nums">
                  {indexData.headline.toFixed(2)}
                </span>
                <span className={`font-mono text-sm ${indexData.change > 0 ? 'text-index-positive' : 'text-index-negative'}`}>
                  {indexData.change > 0 ? '+' : ''}{indexData.change.toFixed(2)} ({indexData.changePercent > 0 ? '+' : ''}{indexData.changePercent.toFixed(1)}%)
                </span>
              </div>
            </div>
            <Link 
              to="/compute-cpi" 
              className="font-mono text-sm text-ticker-fg/80 hover:text-ticker-fg transition-colors flex items-center gap-2"
            >
              Full dashboard <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </section>

      {/* Section 01 - What We Track */}
      <section className="py-20 md:py-32 section-padding">
        <div className="content-max">
          <SectionHeader
            title="What We Track"
            subtitle="Compute CPI measures the inflation rate of AI work using a market-basket methodology modeled on the Consumer Price Index."
          />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {subIndices.map((index) => (
              <IndexTicker
                key={index.name}
                name={index.name}
                value={index.value}
                change={index.change}
                changePercent={index.changePercent}
              />
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {spreads.map((spread) => (
              <SpreadCard
                key={spread.name}
                name={spread.name}
                description={spread.description}
                value={spread.value}
                trend={spread.trend}
              />
            ))}
          </div>

          <div className="mt-10">
            <Link 
              to="/methodology" 
              className="inline-flex items-center gap-2 font-mono text-sm text-text-secondary hover:text-foreground transition-colors"
            >
              Read methodology →
            </Link>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Section 02 - What We Do */}
      <section className="py-20 md:py-32 section-padding">
        <div className="content-max">
          <SectionHeader
            title="What We Do"
            subtitle="Trust + Decision Engineering for institutions whose judgment now depends on systems they don't control."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="space-y-6"
            >
              <h3 className="text-xl font-semibold">Advisory</h3>
              <p className="text-text-secondary leading-relaxed">
                Embedded work with procurement, platform, and leadership teams. 
                We help organizations build decision architectures that remain 
                auditable and contestable as AI capabilities expand.
              </p>
              <ul className="space-y-3 text-sm text-text-secondary">
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Decision budget frameworks
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Escalation path design
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Governance architecture
                </li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="space-y-6"
            >
              <h3 className="text-xl font-semibold">Research</h3>
              <p className="text-text-secondary leading-relaxed">
                Public instruments and analysis that reprice institutional assumptions. 
                We publish indices, memos, and frameworks for organizations 
                navigating AI dependency.
              </p>
              <ul className="space-y-3 text-sm text-text-secondary">
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Compute CPI index
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Research memos
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-foreground">→</span>
                  Trust assessment frameworks
                </li>
              </ul>
            </motion.div>
          </div>

          <div className="mt-12">
            <Button asChild variant="hero-outline">
              <Link to="/work">
                Learn about our work
                <ArrowRight size={16} />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Section 03 - Recent Memos */}
      <section className="py-20 md:py-32 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Recent Memos"
            subtitle="Analysis and frameworks from Occupant research."
          />

          <div className="space-y-0">
            {recentMemos.map((memo) => (
              <MemoCard
                key={memo.number}
                number={memo.number}
                title={memo.title}
                date={memo.date}
                excerpt={memo.excerpt}
                href={memo.href}
                tag={memo.tag}
              />
            ))}
          </div>

          <div className="mt-10 pt-6 border-t border-divider">
            <Link 
              to="/memos" 
              className="inline-flex items-center gap-2 font-mono text-sm text-text-secondary hover:text-foreground transition-colors"
            >
              All research →
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 md:py-32 section-padding bg-secondary/30">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-2xl"
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Get in touch
            </span>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-6">
              We work with institutions navigating AI dependency.
            </h2>
            <p className="text-text-secondary mb-8 leading-relaxed">
              If your organization is building decision systems that depend on AI judgment, 
              we should talk. We work with government, platforms, and enterprises on trust 
              architecture and governance.
            </p>
            <Button asChild variant="hero" size="lg">
              <Link to="/contact">
                Start a conversation
                <ArrowRight size={18} />
              </Link>
            </Button>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
