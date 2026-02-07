import { motion } from "framer-motion";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";

interface IndexTickerProps {
  name: string;
  value: number;
  change: number;
  changePercent: number;
  period?: string;
  compact?: boolean;
}

export const IndexTicker = ({
  name,
  value,
  change,
  changePercent,
  period = "MoM",
  compact = false,
}: IndexTickerProps) => {
  const isPositive = change > 0;
  const isNeutral = change === 0;
  
  const getChangeColor = () => {
    if (isNeutral) return "text-index-neutral";
    return isPositive ? "text-index-positive" : "text-index-negative";
  };

  const getIcon = () => {
    if (isNeutral) return <Minus size={compact ? 12 : 14} />;
    return isPositive ? <ArrowUp size={compact ? 12 : 14} /> : <ArrowDown size={compact ? 12 : 14} />;
  };

  if (compact) {
    return (
      <div className="flex items-center gap-4 py-2">
        <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider min-w-[100px]">
          {name}
        </span>
        <span className="font-mono text-sm font-semibold tabular-nums">
          {value.toFixed(2)}
        </span>
        <span className={`flex items-center gap-1 font-mono text-xs ${getChangeColor()}`}>
          {getIcon()}
          {Math.abs(changePercent).toFixed(1)}%
        </span>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="index-card"
    >
      <div className="flex items-start justify-between mb-4">
        <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider">
          {name}
        </span>
        <span className="font-mono text-xs text-text-tertiary">
          {period}
        </span>
      </div>
      
      <div className="flex items-end gap-3 flex-wrap">
        <span className="text-2xl md:text-3xl font-bold tabular-nums tracking-tight">
          {value.toFixed(2)}
        </span>
        <div className={`flex items-center gap-1 mb-2 ${getChangeColor()}`}>
          {getIcon()}
          <span className="font-mono text-sm font-medium">
            {isPositive ? "+" : ""}{change.toFixed(2)}
          </span>
          <span className="font-mono text-sm">
            ({isPositive ? "+" : ""}{changePercent.toFixed(1)}%)
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default IndexTicker;
