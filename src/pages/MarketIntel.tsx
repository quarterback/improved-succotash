import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { useQuery } from "@tanstack/react-query";
import { fetchCPIData, fetchRankingsData } from "@/services/cpiData";

const MarketIntel = () => {
  const { data: cpiData, isLoading: cpiLoading } = useQuery({
    queryKey: ['cpi-data'],
    queryFn: fetchCPIData,
    staleTime: 5 * 60 * 1000,
  });

  const { data: rankingsData, isLoading: rankingsLoading } = useQuery({
    queryKey: ['rankings-data'],
    queryFn: fetchRankingsData,
    staleTime: 5 * 60 * 1000,
  });

  const isLoading = cpiLoading || rankingsLoading;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <div className="animate-pulse text-text-secondary">Loading market intelligence...</div>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  const exchangeRates = cpiData?.exchange_rates;
  const marketShare = rankingsData?.rankings.by_market_share || [];
  const qualityAdjusted = rankingsData?.rankings.by_quality_adjusted_price || [];
  const marketVelocity = rankingsData?.rankings.market_velocity;

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
              Market Intelligence
            </span>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
              Sabermetrics for AI Markets
            </h1>
            <p className="text-lg md:text-xl text-text-secondary max-w-2xl">
              Quality-adjusted pricing, cognitive arbitrage opportunities, and market velocity analysis.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Quality-Adjusted Pricing */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Quality-Adjusted Price Rankings"
            subtitle="Models ranked by value: performance per dollar spent. Higher scores indicate better bang for buck."
          />

          <div className="border border-divider">
            <div className="grid grid-cols-5 border-b border-divider bg-secondary/50 py-3 px-4">
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider">Rank</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider col-span-2">Model</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">QAP Score</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Cost/1M</span>
            </div>
            
            {qualityAdjusted.map((model, i) => (
              <motion.div
                key={model.model}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="grid grid-cols-5 border-b border-divider last:border-b-0 py-3 px-4 hover:bg-secondary/30 transition-colors"
              >
                <span className="font-mono text-sm text-text-tertiary">#{model.rank}</span>
                <div className="col-span-2">
                  <span className="font-medium">{model.model}</span>
                  <span className="text-xs text-text-tertiary ml-2">({model.provider})</span>
                </div>
                <span className="font-mono text-sm tabular-nums text-right font-semibold text-index-positive">
                  {model.qap_score.toFixed(1)}
                </span>
                <span className="font-mono text-sm tabular-nums text-right">
                  ${model.cost_per_1m.toFixed(2)}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Exchange Rates */}
      {exchangeRates && (
        <section className="py-16 md:py-24 section-padding">
          <div className="content-max">
            <SectionHeader
              title="Model Exchange Rates"
              subtitle="Relative cost ratios between major model families. How many GPT-4 calls equal one Claude Opus?"
            />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="border border-divider p-6 bg-surface-elevated/50">
                <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                  GPT-4 → Claude Sonnet
                </div>
                <div className="text-3xl font-bold tabular-nums">
                  {exchangeRates.gpt4_to_claude_sonnet.toFixed(2)}×
                </div>
                <p className="text-sm text-text-secondary mt-3">
                  Cost multiplier when switching from GPT-4 to Claude Sonnet
                </p>
              </div>

              <div className="border border-divider p-6 bg-surface-elevated/50">
                <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                  Claude Opus → GPT-4
                </div>
                <div className="text-3xl font-bold tabular-nums">
                  {exchangeRates.claude_opus_to_gpt4.toFixed(2)}×
                </div>
                <p className="text-sm text-text-secondary mt-3">
                  Premium paid for Claude Opus over GPT-4
                </p>
              </div>

              <div className="border border-divider p-6 bg-surface-elevated/50">
                <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                  Gemini Ultra → GPT-4
                </div>
                <div className="text-3xl font-bold tabular-nums">
                  {exchangeRates.gemini_ultra_to_gpt4.toFixed(2)}×
                </div>
                <p className="text-sm text-text-secondary mt-3">
                  Premium paid for Gemini Ultra over GPT-4
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Market Velocity */}
      {marketVelocity && (
        <section className="py-16 md:py-24 section-padding">
          <div className="content-max">
            <SectionHeader
              title="Market Velocity"
              subtitle="Which models are gaining or losing share. Momentum indicators for strategic planning."
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Fastest Growing */}
              <div>
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <span className="text-index-positive">↑</span> Fastest Growing
                </h3>
                <div className="space-y-3">
                  {marketVelocity.fastest_growing.map((model) => (
                    <div key={model.model} className="border border-divider p-4 flex items-center justify-between">
                      <span className="font-medium">{model.model}</span>
                      <span className="font-mono text-sm text-index-positive font-semibold">
                        +{model.growth_rate.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Declining */}
              <div>
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <span className="text-index-negative">↓</span> Declining Share
                </h3>
                <div className="space-y-3">
                  {marketVelocity.declining.map((model) => (
                    <div key={model.model} className="border border-divider p-4 flex items-center justify-between">
                      <span className="font-medium">{model.model}</span>
                      <span className="font-mono text-sm text-index-negative font-semibold">
                        {model.decline_rate.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Market Share */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Market Share by Usage"
            subtitle="Current distribution of AI inference workloads across major models."
          />

          <div className="border border-divider">
            <div className="grid grid-cols-4 border-b border-divider bg-secondary/50 py-3 px-4">
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider">Rank</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider col-span-2">Model</span>
              <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider text-right">Share</span>
            </div>
            
            {marketShare.slice(0, 10).map((model, i) => (
              <motion.div
                key={model.model}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="grid grid-cols-4 border-b border-divider last:border-b-0 py-3 px-4 hover:bg-secondary/30 transition-colors"
              >
                <span className="font-mono text-sm text-text-tertiary">#{model.rank}</span>
                <div className="col-span-2">
                  <span className="font-medium">{model.model}</span>
                  <span className="text-xs text-text-tertiary ml-2">({model.provider})</span>
                </div>
                <div className="text-right">
                  <span className="font-mono text-sm tabular-nums font-semibold">
                    {model.market_share.toFixed(1)}%
                  </span>
                  <span className={`font-mono text-xs ml-2 ${model.mom_change > 0 ? 'text-index-positive' : 'text-index-negative'}`}>
                    {model.mom_change > 0 ? '+' : ''}{model.mom_change.toFixed(1)}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default MarketIntel;
