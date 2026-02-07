import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

const About = () => {
  const principles = [
    {
      title: "Publish what others assume",
      description: "We create public instruments that make implicit assumptions explicit and measurable. Indices, frameworks, and analysis that reprice what institutions think is real.",
    },
    {
      title: "Clarity over cleverness",
      description: "Complex problems require clear thinking, not jargon. We write and work in plain language because precision demands it.",
    },
    {
      title: "Build lasting capability",
      description: "Advisory work should create internal expertise, not dependency. We help organizations develop frameworks they can own and evolve.",
    },
    {
      title: "Stakes matter",
      description: "We work on problems where getting it wrong has consequences. That focus shapes everything about how we approach our work.",
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
            className="max-w-2xl"
          >
            <span className="brand-bracket font-semibold text-xl mb-6 block">
              [occupant]
            </span>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              About
            </h1>
            <p className="text-lg text-text-secondary leading-relaxed">
              Occupant is a trust + decision engineering practice. We build 
              public instruments and work with institutions navigating AI dependency.
            </p>
          </motion.div>
        </div>
      </section>

      {/* What We Are */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="What We Are"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16 max-w-4xl">
            <div className="space-y-6 text-text-secondary leading-relaxed">
              <p>
                Occupant exists because AI is changing how institutions make decisions, 
                and most organizations aren't thinking carefully enough about what that means.
              </p>
              <p>
                We're not here to help you "adopt AI faster." We're here to help you 
                adopt AI well—in ways that preserve accountability, maintain human 
                judgment where it matters, and build trust that scales.
              </p>
            </div>
            <div className="space-y-6 text-text-secondary leading-relaxed">
              <p>
                Our research practice (Compute CPI, public memos) and advisory practice 
                reinforce each other. Publishing creates accountability. Advisory work 
                grounds our research in real organizational complexity.
              </p>
              <p>
                We're small by design. The work requires depth over breadth, and we're 
                more interested in doing a few things well than scaling a consultancy.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Principles */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="How We Work"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl">
            {principles.map((principle, i) => (
              <motion.div
                key={principle.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="space-y-3"
              >
                <h3 className="font-semibold text-lg">{principle.title}</h3>
                <p className="text-text-secondary text-sm leading-relaxed">
                  {principle.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="section-divider section-padding">
        <div className="content-max h-px bg-divider"></div>
      </div>

      {/* Contact CTA */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <SectionHeader
            title="Get in Touch"
          />

          <div className="max-w-xl">
            <p className="text-text-secondary leading-relaxed mb-8">
              For advisory inquiries, research collaborations, or press inquiries, 
              use the contact form or reach out directly.
            </p>
            
            <div className="space-y-4 mb-8">
              <div>
                <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-1">
                  General
                </span>
                <a href="mailto:hello@occupant.co" className="text-foreground hover:underline">
                  hello@occupant.co
                </a>
              </div>
              <div>
                <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-1">
                  Research
                </span>
                <a href="mailto:research@occupant.co" className="text-foreground hover:underline">
                  research@occupant.co
                </a>
              </div>
            </div>

            <Button asChild variant="hero">
              <Link to="/contact">
                Contact form
                <ArrowRight size={16} />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default About;
