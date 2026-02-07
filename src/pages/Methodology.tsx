import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

const Methodology = () => {
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
            className="max-w-3xl"
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Index Provider Documentation
            </span>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Methodology
            </h1>
            <p className="text-lg text-text-secondary">
              Technical documentation for the Compute CPI index construction, 
              weighting methodology, and data sourcing protocols.
            </p>
            <div className="mt-6 font-mono text-xs text-text-tertiary">
              Version 2.1 · Last updated January 2026
            </div>
          </motion.div>
        </div>
      </section>

      {/* Overview */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Overview"
          />

          <div className="content-narrow space-y-6 text-text-secondary leading-relaxed">
            <p>
              The Compute CPI (Consumer Price Index for AI Compute) measures changes 
              in the cost of performing standardized cognitive tasks using large language 
              models and related AI systems. The index tracks "inflation" in AI work—how 
              much it costs to do the same job over time.
            </p>
            
            <div className="memo-block">
              <p className="text-foreground">
                <strong>Base Period:</strong> January 2025 = 100
              </p>
              <p className="mt-2">
                All index values are expressed relative to this baseline. A value of 127.4 
                means the cost of the market basket has increased 27.4% since the base period.
              </p>
            </div>

            <p>
              Unlike traditional benchmarks that measure model capabilities, Compute CPI 
              measures the economic cost of achieving specific outcomes. This makes it 
              directly useful for budgeting, procurement, and resource allocation decisions.
            </p>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Market Basket */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Market Basket Construction"
          />

          <div className="content-narrow space-y-6 text-text-secondary leading-relaxed">
            <p>
              The market basket consists of 47 standardized task categories across five 
              major segments. Each task is defined by a reference prompt, expected output 
              characteristics, and quality acceptance criteria.
            </p>

            <h4 className="text-foreground font-semibold mt-8 mb-4">Basket Segments</h4>
            
            <div className="space-y-4">
              <div className="border-l-2 border-foreground pl-4">
                <span className="font-mono text-xs text-text-tertiary">25% weight</span>
                <h5 className="font-semibold text-foreground mt-1">Frontier Reasoning</h5>
                <p className="text-sm mt-1">
                  Complex multi-step reasoning tasks requiring current frontier models. 
                  Includes mathematical proofs, code architecture, and strategic analysis.
                </p>
              </div>

              <div className="border-l-2 border-foreground pl-4">
                <span className="font-mono text-xs text-text-tertiary">35% weight</span>
                <h5 className="font-semibold text-foreground mt-1">General Purpose</h5>
                <p className="text-sm mt-1">
                  Standard knowledge work tasks. Document summarization, Q&A, content 
                  generation, and routine analysis. The bulk of enterprise AI usage.
                </p>
              </div>

              <div className="border-l-2 border-foreground pl-4">
                <span className="font-mono text-xs text-text-tertiary">20% weight</span>
                <h5 className="font-semibold text-foreground mt-1">Budget Inference</h5>
                <p className="text-sm mt-1">
                  High-volume, lower-complexity tasks suitable for smaller models. 
                  Classification, extraction, and formatting operations.
                </p>
              </div>

              <div className="border-l-2 border-foreground pl-4">
                <span className="font-mono text-xs text-text-tertiary">10% weight</span>
                <h5 className="font-semibold text-foreground mt-1">Embedding & Search</h5>
                <p className="text-sm mt-1">
                  Vector embedding generation and semantic search operations. 
                  Measured per million tokens processed.
                </p>
              </div>

              <div className="border-l-2 border-foreground pl-4">
                <span className="font-mono text-xs text-text-tertiary">10% weight</span>
                <h5 className="font-semibold text-foreground mt-1">Multimodal</h5>
                <p className="text-sm mt-1">
                  Vision, audio, and cross-modal tasks. Image analysis, transcription, 
                  and document understanding.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Pricing Methodology */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Pricing Methodology"
          />

          <div className="content-narrow space-y-6 text-text-secondary leading-relaxed">
            <p>
              Prices are collected from the published API pricing of major providers 
              plus observed market rates from enterprise contracts (anonymized). We use 
              a Laspeyres-type index formula with periodic reweighting.
            </p>

            <h4 className="text-foreground font-semibold mt-8 mb-4">Data Sources</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-3">
                <span className="text-foreground">→</span>
                Published API pricing (OpenAI, Anthropic, Google, Mistral, Cohere)
              </li>
              <li className="flex items-start gap-3">
                <span className="text-foreground">→</span>
                Enterprise contract surveys (anonymized, n=150+)
              </li>
              <li className="flex items-start gap-3">
                <span className="text-foreground">→</span>
                Open-source inference cost estimates (GPU-hour equivalents)
              </li>
              <li className="flex items-start gap-3">
                <span className="text-foreground">→</span>
                Cloud provider compute pricing (AWS, GCP, Azure)
              </li>
            </ul>

            <h4 className="text-foreground font-semibold mt-8 mb-4">Quality Adjustment</h4>
            <p>
              When a model is upgraded or replaced, we apply hedonic quality adjustment 
              to separate genuine cost changes from capability improvements. A model that 
              costs the same but performs 20% better on our benchmark suite is treated 
              as a 20% price decrease, all else equal.
            </p>

            <div className="terminal-block mt-6">
              <pre className="text-xs overflow-x-auto">
{`Index(t) = Σ(w_i × p_i,t / p_i,0) × 100

where:
  w_i = category weight
  p_i,t = quality-adjusted price at time t
  p_i,0 = quality-adjusted price at base period`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Sub-Indices */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Sub-Indices"
          />

          <div className="content-narrow space-y-8">
            <div className="space-y-4">
              <h4 className="text-foreground font-semibold">Judgment CPI</h4>
              <p className="text-text-secondary text-sm leading-relaxed">
                Tracks the cost of tasks requiring reliable judgment—decisions where 
                errors have significant consequences. Uses a subset of the basket with 
                stricter quality thresholds and measures the premium for trustworthy outputs.
              </p>
            </div>

            <div className="space-y-4">
              <h4 className="text-foreground font-semibold">Long Context CPI</h4>
              <p className="text-text-secondary text-sm leading-relaxed">
                Measures the cost of extended context operations (100K+ tokens). 
                Particularly relevant for document analysis, codebase understanding, 
                and complex research tasks.
              </p>
            </div>

            <div className="space-y-4">
              <h4 className="text-foreground font-semibold">Budget CPI</h4>
              <p className="text-text-secondary text-sm leading-relaxed">
                Tracks the "commodity" end of the market—simple tasks that can be 
                accomplished by smaller, cheaper models. Useful for understanding 
                the floor of AI compute costs.
              </p>
            </div>
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
            className="max-w-xl"
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Questions about methodology?
            </span>
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-4">
              We're happy to discuss.
            </h2>
            <p className="text-text-secondary mb-8">
              For technical questions about index construction or data sourcing, 
              contact our research team.
            </p>
            <Button asChild variant="hero">
              <Link to="/contact">
                Get in touch
                <ArrowRight size={16} />
              </Link>
            </Button>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Methodology;
