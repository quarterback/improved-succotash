import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

const Work = () => {
  const capabilities = [
    {
      title: "Trust Architecture",
      description: "Design decision systems that remain auditable and contestable as AI capabilities expand. We help organizations build governance that scales.",
      deliverables: ["Decision authority mapping", "Escalation path design", "Audit trail architecture"],
    },
    {
      title: "Procurement Strategy",
      description: "Navigate AI vendor relationships and build procurement frameworks that account for capability evolution and vendor dependency.",
      deliverables: ["Vendor evaluation frameworks", "Contract structure guidance", "Total cost modeling"],
    },
    {
      title: "Decision Budgets",
      description: "Implement decision budget frameworks that allocate judgment authority appropriately between humans and AI systems.",
      deliverables: ["Budget framework design", "Threshold calibration", "Override protocols"],
    },
    {
      title: "Forensic Analysis",
      description: "When AI-augmented decisions fail, understand why. We conduct post-incident analysis and help organizations learn from system failures.",
      deliverables: ["Incident reconstruction", "Root cause analysis", "System hardening recommendations"],
    },
  ];

  const clients = [
    "Government agencies navigating AI procurement",
    "Platform companies building AI-dependent products",
    "Enterprises deploying AI in high-stakes domains",
    "Foundations and research institutions",
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
            className="max-w-2xl"
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Advisory Services
            </span>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Work
            </h1>
            <p className="text-lg text-text-secondary">
              Trust + Decision Engineering for institutions whose judgment now 
              depends on systems they don't control.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Approach */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Our Approach"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16">
            <div className="space-y-6 text-text-secondary leading-relaxed">
              <p>
                We work embedded with organizations, not as outside consultants 
                delivering slide decks. Our engagements typically span 3-12 months 
                and involve direct collaboration with procurement, platform, and 
                leadership teams.
              </p>
              <p>
                The goal is to build lasting capability, not dependency. We help 
                organizations develop internal frameworks and expertise for 
                navigating AI adoption decisions.
              </p>
            </div>
            <div className="space-y-6 text-text-secondary leading-relaxed">
              <p>
                We're selective about engagements. We work with organizations 
                where the stakes are high enough to warrant careful thinking 
                about decision architecture—and where leadership is genuinely 
                committed to getting this right.
              </p>
              <p>
                Our research practice (Compute CPI, public memos) informs our 
                advisory work, and vice versa. Clients benefit from insights 
                developed across the portfolio.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Capabilities */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Capabilities"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {capabilities.map((capability, i) => (
              <motion.div
                key={capability.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="border border-divider p-6 md:p-8"
              >
                <h3 className="text-xl font-semibold mb-3">{capability.title}</h3>
                <p className="text-text-secondary text-sm leading-relaxed mb-6">
                  {capability.description}
                </p>
                <div>
                  <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-3">
                    Deliverables
                  </span>
                  <ul className="space-y-2">
                    {capability.deliverables.map((item) => (
                      <li key={item} className="text-sm text-text-secondary flex items-start gap-2">
                        <span className="text-foreground">→</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Clients */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Who We Work With"
          />

          <div className="max-w-2xl space-y-4">
            {clients.map((client, i) => (
              <motion.div
                key={client}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-4 py-3 border-b border-divider"
              >
                <span className="font-mono text-xs text-text-tertiary">0{i + 1}</span>
                <span className="text-foreground">{client}</span>
              </motion.div>
            ))}
          </div>

          <p className="mt-8 text-text-secondary max-w-xl">
            We don't publish a client list. Confidentiality is core to how we work. 
            References available on request for qualified prospects.
          </p>
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
              Start a conversation
            </span>
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-4">
              Tell us what you're working on.
            </h2>
            <p className="text-text-secondary mb-8">
              If your organization is navigating AI dependency and thinking seriously 
              about decision architecture, we should talk.
            </p>
            <Button asChild variant="hero" size="lg">
              <Link to="/contact">
                Get in touch
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

export default Work;
