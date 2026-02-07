import { motion } from "framer-motion";
import { Link, useParams, Navigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { getMemoBySlug, getRelatedMemos, memosData } from "@/data/memos";

const MemoDetail = () => {
  const { slug } = useParams<{ slug: string }>();
  const memo = slug ? getMemoBySlug(slug) : undefined;

  if (!memo) {
    return <Navigate to="/memos" replace />;
  }

  const relatedMemos = getRelatedMemos(memo);
  
  // Get prev/next memos for navigation
  const currentIndex = memosData.findIndex((m) => m.slug === slug);
  const prevMemo = currentIndex < memosData.length - 1 ? memosData[currentIndex + 1] : null;
  const nextMemo = currentIndex > 0 ? memosData[currentIndex - 1] : null;

  const renderContent = (content: string[]) => {
    return content.map((block, index) => {
      // Check if it's a heading
      if (block.startsWith("## ")) {
        return (
          <h2 key={index} className="text-xl md:text-2xl font-bold mt-12 mb-4">
            {block.replace("## ", "")}
          </h2>
        );
      }
      
      // Check if it's a bold list item pattern
      if (block.startsWith("**") && block.includes("**:")) {
        const [boldPart, rest] = block.split("**:");
        return (
          <p key={index} className="text-text-secondary leading-relaxed mb-4">
            <strong className="text-foreground">{boldPart.replace("**", "")}:</strong>
            {rest}
          </p>
        );
      }

      // Check if it's a list item
      if (block.startsWith("- ")) {
        return (
          <li key={index} className="text-text-secondary leading-relaxed ml-4 mb-2">
            {block.replace("- ", "")}
          </li>
        );
      }

      // Regular paragraph
      return (
        <p key={index} className="text-text-secondary leading-relaxed mb-6">
          {block}
        </p>
      );
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Header */}
      <section className="pt-32 md:pt-40 pb-12 section-padding border-b border-divider">
        <div className="content-max">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Link 
              to="/memos" 
              className="inline-flex items-center gap-2 font-mono text-sm text-text-tertiary hover:text-foreground transition-colors mb-8"
            >
              <ArrowLeft size={14} />
              All memos
            </Link>

            <div className="flex flex-wrap items-center gap-3 mb-4">
              <span className="font-mono text-xs text-text-tertiary">
                {memo.number}
              </span>
              <span className="font-mono text-xs text-text-tertiary px-2 py-0.5 bg-secondary">
                {memo.tag}
              </span>
              <span className="font-mono text-xs text-text-tertiary">
                {memo.date}
              </span>
            </div>

            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight max-w-4xl">
              {memo.title}
            </h1>
          </motion.div>
        </div>
      </section>

      {/* Content */}
      <section className="py-12 md:py-16 section-padding">
        <div className="content-max">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16">
            {/* Main Content */}
            <motion.article
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="lg:col-span-8"
            >
              <div className="prose-occupant">
                {renderContent(memo.content)}
              </div>

              {/* Navigation */}
              <div className="mt-16 pt-8 border-t border-divider">
                <div className="flex flex-col sm:flex-row justify-between gap-6">
                  {prevMemo ? (
                    <Link 
                      to={`/memos/${prevMemo.slug}`}
                      className="group flex-1"
                    >
                      <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2">
                        ← Previous
                      </span>
                      <span className="text-foreground font-medium group-hover:underline">
                        {prevMemo.title}
                      </span>
                    </Link>
                  ) : (
                    <div className="flex-1" />
                  )}
                  
                  {nextMemo ? (
                    <Link 
                      to={`/memos/${nextMemo.slug}`}
                      className="group flex-1 text-right"
                    >
                      <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2">
                        Next →
                      </span>
                      <span className="text-foreground font-medium group-hover:underline">
                        {nextMemo.title}
                      </span>
                    </Link>
                  ) : (
                    <div className="flex-1" />
                  )}
                </div>
              </div>
            </motion.article>

            {/* Sidebar */}
            <motion.aside
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="lg:col-span-4"
            >
              <div className="lg:sticky lg:top-28 space-y-8">
                {/* Related Memos */}
                {relatedMemos.length > 0 && (
                  <div className="border border-divider p-6">
                    <h3 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-4">
                      Related Memos
                    </h3>
                    <ul className="space-y-4">
                      {relatedMemos.map((related) => (
                        <li key={related.slug}>
                          <Link 
                            to={`/memos/${related.slug}`}
                            className="group block"
                          >
                            <span className="font-mono text-xs text-text-tertiary block mb-1">
                              {related.number}
                            </span>
                            <span className="text-sm font-medium text-foreground group-hover:underline">
                              {related.title}
                            </span>
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Subscribe CTA */}
                <div className="bg-secondary/50 p-6 border-l-2 border-foreground">
                  <h3 className="font-semibold mb-2">Subscribe to Memos</h3>
                  <p className="text-text-secondary text-sm mb-4">
                    Get new research delivered to your inbox. 2-4 pieces per month.
                  </p>
                  <form className="space-y-3">
                    <input
                      type="email"
                      placeholder="your@email.com"
                      className="w-full px-3 py-2 border border-divider bg-background font-mono text-sm focus:outline-none focus:border-foreground transition-colors"
                    />
                    <button
                      type="submit"
                      className="w-full px-4 py-2 bg-foreground text-background font-mono text-sm hover:bg-foreground/90 transition-colors"
                    >
                      Subscribe
                    </button>
                  </form>
                </div>

                {/* Tags */}
                <div>
                  <h3 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-3">
                    Categories
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {["Trust Engineering", "Index Report", "Framework", "Announcement"].map((tag) => (
                      <Link
                        key={tag}
                        to="/memos"
                        className={`font-mono text-xs px-3 py-1.5 border transition-colors ${
                          tag === memo.tag 
                            ? "bg-foreground text-background border-foreground" 
                            : "border-divider text-text-secondary hover:border-foreground hover:text-foreground"
                        }`}
                      >
                        {tag}
                      </Link>
                    ))}
                  </div>
                </div>

                {/* Contact CTA */}
                <div className="pt-4">
                  <p className="text-sm text-text-tertiary mb-3">
                    Questions about this research?
                  </p>
                  <Button asChild variant="hero-outline" size="sm" className="w-full">
                    <Link to="/contact">
                      Get in touch
                      <ArrowRight size={14} />
                    </Link>
                  </Button>
                </div>
              </div>
            </motion.aside>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default MemoDetail;
