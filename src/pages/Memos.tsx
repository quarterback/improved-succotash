import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import MemoCard from "@/components/MemoCard";
import { memosData } from "@/data/memos";

const Memos = () => {
  const memos = memosData.map((memo) => ({
    number: memo.number,
    title: memo.title,
    date: memo.date,
    excerpt: memo.excerpt,
    href: `/memos/${memo.slug}`,
    tag: memo.tag,
  }));

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
              Research & Analysis
            </span>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Memos
            </h1>
            <p className="text-lg text-text-secondary">
              Public briefs from Occupant research. Index reports, frameworks, 
              and analysis for institutions navigating AI dependency.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Memos List */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <div className="max-w-3xl space-y-0">
            {memos.map((memo) => (
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

          <div className="mt-12 pt-8 border-t border-divider max-w-3xl">
            <p className="text-sm text-text-tertiary">
              Subscribe to receive new memos via email. We publish 2-4 pieces per month.
            </p>
            <form className="mt-4 flex flex-col sm:flex-row gap-3 max-w-md">
              <input
                type="email"
                placeholder="your@email.com"
                className="flex-1 px-4 py-2 border border-divider bg-background font-mono text-sm focus:outline-none focus:border-foreground transition-colors"
              />
              <button
                type="submit"
                className="px-6 py-2 bg-foreground text-background font-mono text-sm hover:bg-foreground/90 transition-colors"
              >
                Subscribe
              </button>
            </form>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Memos;
