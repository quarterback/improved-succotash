import { motion } from "framer-motion";
import { Link } from "react-router-dom";

interface MemoCardProps {
  number: string;
  title: string;
  date: string;
  excerpt: string;
  href: string;
  tag?: string;
}

export const MemoCard = ({
  number,
  title,
  date,
  excerpt,
  href,
  tag,
}: MemoCardProps) => {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5 }}
      className="group border-t border-divider pt-6"
    >
      <Link to={href} className="block">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-text-tertiary">
              {number}
            </span>
            {tag && (
              <span className="font-mono text-xs text-text-tertiary px-2 py-0.5 bg-secondary">
                {tag}
              </span>
            )}
          </div>
          <span className="font-mono text-xs text-text-tertiary">
            {date}
          </span>
        </div>
        
        <h3 className="text-lg md:text-xl font-semibold mb-2 group-hover:underline decoration-1 underline-offset-4">
          {title}
        </h3>
        
        <p className="text-text-secondary text-sm md:text-base line-clamp-2">
          {excerpt}
        </p>
        
        <span className="inline-flex items-center gap-1 mt-4 text-sm font-medium text-foreground">
          Read memo <span className="transition-transform group-hover:translate-x-1">→</span>
        </span>
      </Link>
    </motion.article>
  );
};

export default MemoCard;
