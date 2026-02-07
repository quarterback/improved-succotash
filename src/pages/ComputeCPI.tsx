import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import IndexTicker from "@/components/IndexTicker";
import SpreadCard from "@/components/SpreadCard";
import { ArrowRight, Download, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { fetchCPIData, fetchHistoricalData } from "@/services/cpiData";

const ComputeCPI = () => {
  // Fetch real CPI data
  const { data: cpiData, isLoading: cpiLoading, error: cpiError } = useQuery({
    queryKey: ['cpi-data'],
    queryFn: fetchCPIData,
    staleTime: 5 * 60 * 1000,
  });

  // Fetch historical data
  const { data: historicalData, isLoading: histLoading, error: histError } = useQuery({
    queryKey: ['historical-data'],
    queryFn: fetchHistoricalData,
    staleTime: 5 * 60 * 1000,
  });

  const isLoading = cpiLoading || histLoading;
  const error = cpiError || histError;

  // Map data from API
  const headlineIndex = cpiData ? {
    name: cpiData.compute_cpi.name,
    value: cpiData.compute_cpi.value,
    change: cpiData.compute_cpi.mom_change || 0,
    changePercent: cpiData.compute_cpi.mom_change || 0,
    baseDate: cpiData.meta.baseline_date,
  } : { name: "", value: 0, change: 0, changePercent: 0, baseDate: "" };

  const subIndices = cpiData ? Object.values(cpiData.subindices).slice(0, 3).map(sub => ({
    name: sub.name,
    value: sub.value,
    change: sub.mom_change,
    changePercent: sub.mom_change,
    description: sub.description,
  })) : [];

  const spreads = cpiData ? Object.values(cpiData.spreads).map(spread => ({
    name: spread.name,
    description: spread.description,
    value: spread.value,
    trend: spread.trend === "widening" ? "up" as const : 
           spread.trend === "narrowing" ? "down" as const : 
           "stable" as const,
  })) : [];

  const basketComponents = cpiData?.basket_components || [];

  const historicalSeries = historicalData?.historical_series.slice(-8).reverse() || [];

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <div className="animate-pulse text-text-secondary">Loading CPI data...</div>
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
            <div className="text-text-secondary">Failed to load CPI data</div>
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

      {/* Header */}
      <section className="pt-32 md:pt-40 pb-16 section-padding border-b border-divider">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Public Benchmark
            </span>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
              Compute CPI
            </h1>
            <p className="text-lg md:text-xl text-text-secondary max-w-2xl">
              The inflation rate of AI work. Tracking how much it costs to do the same 
              cognitive task over time, using a market-basket methodology.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Live Ticker Board */}
      <section className="py-8 bg-ticker-bg">
        <div className="section-padding content-max">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Headline Index */}
            <div className="lg:col-span-1 border-r-0 lg:border-r border-divider-strong pr-0 lg:pr-6">
              <span className="font-mono text-xs text-ticker-fg/60 uppercase tracking-wider block mb-2">
                Headline Index
              </span>
              <div className="flex items-baseline gap-3">
                <span className="text-5xl md:text-6xl font-bold text-ticker-fg tabular-nums">
                  {headlineIndex.value.toFixed(1)}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <span className={`font-mono text-sm ${headlineIndex.change > 0 ? 'text-index-positive' : 'text-index-negative'}`}>
                  {headlineIndex.change > 0 ? '▲' : '▼'} {Math.abs(headlineIndex.changePercent).toFixed(1)}% MoM
                </span>
              </div>
              <span className="font-mono text-xs text-ticker-fg/40 block mt-3">
                Base = 100 ({headlineIndex.baseDate})
              </span>
            </div>

            {/* Sub-indices */}
            <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4">
              {subIndices.map((index) => (
                <div key={index.name} className="py-3 px-4 bg-ticker-bg border border-divider-strong/30">
                  <span className="font-mono text-xs text-ticker-fg/60 uppercase tracking-wider block mb-2">
                    {index.name}
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-ticker-fg tabular-nums">
                      {index.value.toFixed(1)}
                    </span>
                    <span className={`font-mono text-xs ${index.change > 0 ? 'text-index-positive' : 'text-index-negative'}`}>
                      {index.change > 0 ? '+' : ''}{index.changePercent.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Spreads Panel */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Spreads"
            subtitle="Key differentials that reveal market structure and capability stratification."
          />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
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
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Historical Data */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Historical Series"
            subtitle="Month-over-month index values since baseline."
          />

          <div className="border border-divider">
            {/* Table Header */}
            <div className="grid grid-cols-3 border-b border-divider bg-secondary/50 py-3 px-4">
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider">Period</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Index</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">MoM %</span>
            </div>
            
            {/* Table Rows */}
            {historicalSeries.map((row, i) => (
              <motion.div
                key={row.period}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="grid grid-cols-3 border-b border-divider last:border-b-0 py-3 px-4 hover:bg-secondary/30 transition-colors"
              >
                <span className="font-mono text-sm">{row.period}</span>
                <span className="font-mono text-sm tabular-nums text-right font-medium">{row.value.toFixed(1)}</span>
                <span className={`font-mono text-sm tabular-nums text-right ${(row.mom_change || 0) > 0 ? 'text-index-positive' : 'text-index-negative'}`}>
                  {(row.mom_change || 0) > 0 ? '+' : ''}{(row.mom_change || 0).toFixed(1)}%
                </span>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 flex flex-col sm:flex-row gap-4">
            <Button variant="terminal" size="sm">
              <Download size={14} />
              Download CSV
            </Button>
            <Button variant="terminal" size="sm">
              <ExternalLink size={14} />
              API Access
            </Button>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Market Basket Composition */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Market Basket"
            subtitle="The weighted components that comprise the headline index."
          />

          <div className="space-y-4">
            {basketComponents.map((component, i) => (
              <motion.div
                key={component.category}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="border border-divider p-4 md:p-5"
              >
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h4 className="font-semibold">{component.category}</h4>
                      <span className="font-mono text-xs text-text-tertiary bg-secondary px-2 py-0.5">
                        {component.weight}%
                      </span>
                    </div>
                    <p className="text-sm text-text-secondary">{component.examples}</p>
                  </div>
                  <div className="w-full md:w-48">
                    <div className="h-2 bg-secondary overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: `${component.weight}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.2 + i * 0.05 }}
                        className="h-full bg-foreground"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-10">
            <Link 
              to="/methodology" 
              className="inline-flex items-center gap-2 font-mono text-sm text-text-secondary hover:text-foreground transition-colors"
            >
              Full methodology documentation →
            </Link>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 md:py-24 section-padding bg-secondary/30">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="flex flex-col md:flex-row md:items-center md:justify-between gap-8"
          >
            <div className="max-w-xl">
              <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-4">
                Use Compute CPI in your planning
              </h2>
              <p className="text-text-secondary">
                The index is free for research and editorial use. Enterprise API access 
                and custom analysis available for procurement and FinOps teams.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild variant="hero">
                <Link to="/contact">
                  Get API access
                  <ArrowRight size={16} />
                </Link>
              </Button>
              <Button asChild variant="hero-outline">
                <Link to="/methodology">
                  Read methodology
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default ComputeCPI;
