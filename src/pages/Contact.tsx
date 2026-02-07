import { motion } from "framer-motion";
import { useState } from "react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    organization: "",
    type: "",
    message: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle form submission
    console.log("Form submitted:", formData);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

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
            className="max-w-xl"
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Get in Touch
            </span>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Contact
            </h1>
            <p className="text-lg text-text-secondary">
              For advisory inquiries, research collaborations, or questions 
              about Compute CPI.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Contact Form */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            {/* Form */}
            <motion.form
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              onSubmit={handleSubmit}
              className="space-y-6"
            >
              <div>
                <label 
                  htmlFor="name" 
                  className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2"
                >
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 border border-divider bg-background text-foreground focus:outline-none focus:border-foreground transition-colors"
                  placeholder="Your name"
                />
              </div>

              <div>
                <label 
                  htmlFor="email" 
                  className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2"
                >
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 border border-divider bg-background text-foreground focus:outline-none focus:border-foreground transition-colors"
                  placeholder="your@email.com"
                />
              </div>

              <div>
                <label 
                  htmlFor="organization" 
                  className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2"
                >
                  Organization
                </label>
                <input
                  type="text"
                  id="organization"
                  name="organization"
                  value={formData.organization}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-divider bg-background text-foreground focus:outline-none focus:border-foreground transition-colors"
                  placeholder="Your organization (optional)"
                />
              </div>

              <div>
                <label 
                  htmlFor="type" 
                  className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2"
                >
                  Inquiry Type
                </label>
                <select
                  id="type"
                  name="type"
                  value={formData.type}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 border border-divider bg-background text-foreground focus:outline-none focus:border-foreground transition-colors appearance-none cursor-pointer"
                >
                  <option value="">Select type...</option>
                  <option value="advisory">Advisory engagement</option>
                  <option value="research">Research collaboration</option>
                  <option value="api">Compute CPI API access</option>
                  <option value="press">Press inquiry</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <div>
                <label 
                  htmlFor="message" 
                  className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-2"
                >
                  Message
                </label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows={6}
                  className="w-full px-4 py-3 border border-divider bg-background text-foreground focus:outline-none focus:border-foreground transition-colors resize-none"
                  placeholder="Tell us about your project or question..."
                />
              </div>

              <Button type="submit" variant="hero" size="lg" className="w-full sm:w-auto">
                Send message
                <ArrowRight size={18} />
              </Button>
            </motion.form>

            {/* Contact Info */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="space-y-12"
            >
              <div>
                <h3 className="font-semibold text-lg mb-4">Direct Contact</h3>
                <div className="space-y-4">
                  <div>
                    <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-1">
                      General Inquiries
                    </span>
                    <a href="mailto:hello@occupant.co" className="text-foreground hover:underline">
                      hello@occupant.co
                    </a>
                  </div>
                  <div>
                    <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-1">
                      Research & Data
                    </span>
                    <a href="mailto:research@occupant.co" className="text-foreground hover:underline">
                      research@occupant.co
                    </a>
                  </div>
                  <div>
                    <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-1">
                      Press
                    </span>
                    <a href="mailto:press@occupant.co" className="text-foreground hover:underline">
                      press@occupant.co
                    </a>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-4">Response Time</h3>
                <p className="text-text-secondary text-sm leading-relaxed">
                  We respond to most inquiries within 2-3 business days. For advisory 
                  engagement inquiries, please include context about your organization 
                  and the challenge you're navigating.
                </p>
              </div>

              <div className="p-6 bg-secondary/50 border-l-2 border-foreground">
                <h4 className="font-semibold mb-2">A note on fit</h4>
                <p className="text-text-secondary text-sm leading-relaxed">
                  We're selective about advisory engagements. We work best with 
                  organizations where the stakes are genuinely high and leadership 
                  is committed to getting AI governance right. If you're looking 
                  for someone to validate a predetermined approach, we're probably 
                  not the right fit.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Contact;
