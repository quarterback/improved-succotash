import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t border-divider bg-background">
      <div className="section-padding content-max py-16 md:py-24">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8">
          {/* Brand Column */}
          <div className="md:col-span-2">
            <span className="brand-bracket font-semibold text-base mb-4 block">
              [occupant]
            </span>
            <p className="text-text-secondary text-sm max-w-sm mb-6">
              Trust + Decision Engineering. 
              We build instruments that reprice what institutions think is real.
            </p>
            <p className="font-mono text-xs text-text-tertiary">
              © {new Date().getFullYear()} Occupant. All rights reserved.
            </p>
          </div>

          {/* Products */}
          <div>
            <h4 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-4">
              Products
            </h4>
            <ul className="space-y-3">
              <li>
                <Link to="/compute-cpi" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  Compute CPI
                </Link>
              </li>
              <li>
                <Link to="/methodology" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  Methodology
                </Link>
              </li>
              <li>
                <Link to="/memos" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  Research Memos
                </Link>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-4">
              Company
            </h4>
            <ul className="space-y-3">
              <li>
                <Link to="/about" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  About
                </Link>
              </li>
              <li>
                <Link to="/work" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  Advisory
                </Link>
              </li>
              <li>
                <Link to="/contact" className="text-sm text-text-secondary hover:text-foreground transition-colors">
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Ticker */}
        <div className="mt-16 pt-8 border-t border-divider">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div className="font-mono text-xs text-text-tertiary">
              Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
            </div>
            <div className="flex items-center gap-6">
              <a href="#" className="font-mono text-xs text-text-tertiary hover:text-foreground transition-colors">
                Terms
              </a>
              <a href="#" className="font-mono text-xs text-text-tertiary hover:text-foreground transition-colors">
                Privacy
              </a>
              <a href="#" className="font-mono text-xs text-text-tertiary hover:text-foreground transition-colors">
                RSS
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
