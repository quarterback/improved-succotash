import { motion } from "framer-motion";

interface SpreadCardProps {
  name: string;
  description: string;
  value: number;
  unit?: string;
  trend?: "up" | "down" | "stable";
}

export const SpreadCard = ({
  name,
  description,
  value,
  unit = "pts",
  trend = "stable",
}: SpreadCardProps) => {
  const getTrendColor = () => {
    switch (trend) {
      case "up":
        return "text-index-positive";
      case "down":
        return "text-index-negative";
      default:
        return "text-text-secondary";
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="border border-divider p-5 md:p-6 bg-surface-elevated/50"
    >
      <div className="flex items-start justify-between mb-3">
        <h4 className="font-mono text-xs text-text-tertiary uppercase tracking-wider">
          {name}
        </h4>
        <span className={`font-mono text-lg font-bold tabular-nums ${getTrendColor()}`}>
          {value > 0 ? "+" : ""}{value.toFixed(1)}<span className="text-xs ml-1">{unit}</span>
        </span>
      </div>
      <p className="text-sm text-text-secondary">
        {description}
      </p>
    </motion.div>
  );
};

export default SpreadCard;
