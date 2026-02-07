import { motion } from "framer-motion";
import { useState } from "react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import SectionHeader from "@/components/SectionHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useQuery } from "@tanstack/react-query";
import { fetchCPIData } from "@/services/cpiData";

const Calculator = () => {
  const [monthlySpend, setMonthlySpend] = useState<number>(10000);
  const [workloadType, setWorkloadType] = useState<string>("startup");
  const [currentModel, setCurrentModel] = useState<string>("gpt4");

  const { data: cpiData } = useQuery({
    queryKey: ['cpi-data'],
    queryFn: fetchCPIData,
    staleTime: 5 * 60 * 1000,
  });

  // Model pricing data ($/1M tokens)
  const modelPricing: Record<string, { name: string; price: number; tier: string }> = {
    "gpt4": { name: "GPT-4", price: 10.0, tier: "frontier" },
    "claude-opus": { name: "Claude Opus", price: 15.0, tier: "frontier" },
    "gemini-ultra": { name: "Gemini Ultra", price: 12.5, tier: "frontier" },
    "claude-sonnet": { name: "Claude Sonnet 3.5", price: 3.0, tier: "general" },
    "gpt-3.5": { name: "GPT-3.5 Turbo", price: 0.5, tier: "budget" },
    "claude-haiku": { name: "Claude Haiku", price: 0.25, tier: "budget" },
  };

  // Workload type allocation (how tokens are distributed across tiers)
  const workloadAllocations: Record<string, { frontier: number; general: number; budget: number }> = {
    "startup": { frontier: 0.15, general: 0.35, budget: 0.50 },
    "agentic": { frontier: 0.40, general: 0.45, budget: 0.15 },
    "throughput": { frontier: 0.05, general: 0.25, budget: 0.70 },
  };

  const currentModelData = modelPricing[currentModel];
  const allocation = workloadAllocations[workloadType];
  
  // Calculate optimized spend based on workload type
  const calculateOptimizedSpend = () => {
    // Assume current model is used for everything
    const tokensPerMonth = (monthlySpend / currentModelData.price) * 1_000_000;
    
    // Redistribute tokens based on workload allocation
    const frontierTokens = tokensPerMonth * allocation.frontier;
    const generalTokens = tokensPerMonth * allocation.general;
    const budgetTokens = tokensPerMonth * allocation.budget;
    
    // Use best-in-class for each tier
    const frontierCost = (frontierTokens / 1_000_000) * 10.0; // GPT-4
    const generalCost = (generalTokens / 1_000_000) * 3.0;   // Claude Sonnet
    const budgetCost = (budgetTokens / 1_000_000) * 0.25;    // Claude Haiku
    
    return frontierCost + generalCost + budgetCost;
  };

  const optimizedSpend = calculateOptimizedSpend();
  const monthlySavings = monthlySpend - optimizedSpend;
  const annualSavings = monthlySavings * 12;
  const savingsPercent = (monthlySavings / monthlySpend) * 100;

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
          >
            <span className="font-mono text-xs text-text-tertiary uppercase tracking-wider block mb-4">
              Cost Savings Calculator
            </span>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
              Optimize Your AI Budget
            </h1>
            <p className="text-lg md:text-xl text-text-secondary max-w-2xl">
              Calculate potential savings by matching workload requirements to the right model tiers.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Calculator */}
      <section className="py-16 md:py-24 section-padding">
        <div className="content-max">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Input Panel */}
            <div>
              <h2 className="text-2xl font-bold mb-6">Your Current Setup</h2>
              
              <div className="space-y-6">
                {/* Monthly Spend */}
                <div>
                  <Label htmlFor="monthly-spend" className="text-sm font-medium mb-2 block">
                    Monthly AI Spend
                  </Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary">$</span>
                    <Input
                      id="monthly-spend"
                      type="number"
                      value={monthlySpend}
                      onChange={(e) => setMonthlySpend(Number(e.target.value))}
                      className="pl-7"
                      min={0}
                      step={1000}
                    />
                  </div>
                  <p className="text-xs text-text-tertiary mt-2">
                    Your current monthly spending on AI inference
                  </p>
                </div>

                {/* Current Model */}
                <div>
                  <Label htmlFor="current-model" className="text-sm font-medium mb-2 block">
                    Primary Model
                  </Label>
                  <Select value={currentModel} onValueChange={setCurrentModel}>
                    <SelectTrigger id="current-model">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(modelPricing).map(([key, model]) => (
                        <SelectItem key={key} value={key}>
                          {model.name} (${model.price}/1M tokens)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-text-tertiary mt-2">
                    The model you use most frequently
                  </p>
                </div>

                {/* Workload Type */}
                <div>
                  <Label htmlFor="workload-type" className="text-sm font-medium mb-2 block">
                    Workload Persona
                  </Label>
                  <Select value={workloadType} onValueChange={setWorkloadType}>
                    <SelectTrigger id="workload-type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="startup">
                        Startup (prototyping, iteration)
                      </SelectItem>
                      <SelectItem value="agentic">
                        Agentic (autonomous systems)
                      </SelectItem>
                      <SelectItem value="throughput">
                        Throughput (high-volume processing)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-text-tertiary mt-2">
                    Best describes your use case
                  </p>
                </div>

                {/* Tier Allocation Display */}
                <div className="border border-divider p-4 bg-secondary/30">
                  <h3 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-3">
                    Recommended Allocation
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Frontier Models</span>
                      <span className="font-mono font-semibold">{(allocation.frontier * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span>General Purpose</span>
                      <span className="font-mono font-semibold">{(allocation.general * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span>Budget Tier</span>
                      <span className="font-mono font-semibold">{(allocation.budget * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Results Panel */}
            <div>
              <h2 className="text-2xl font-bold mb-6">Potential Savings</h2>
              
              <div className="space-y-6">
                {/* Savings Summary */}
                <div className="border border-divider p-8 bg-surface-elevated/50">
                  <div className="mb-6">
                    <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                      Monthly Savings
                    </div>
                    <div className="text-5xl font-bold text-index-positive tabular-nums">
                      ${monthlySavings.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    </div>
                    <div className="text-sm text-text-secondary mt-2">
                      {savingsPercent.toFixed(1)}% reduction in costs
                    </div>
                  </div>

                  <div className="border-t border-divider pt-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-1">
                          Current
                        </div>
                        <div className="text-2xl font-bold tabular-nums">
                          ${monthlySpend.toLocaleString()}
                        </div>
                        <div className="text-xs text-text-tertiary mt-1">per month</div>
                      </div>
                      <div>
                        <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-1">
                          Optimized
                        </div>
                        <div className="text-2xl font-bold tabular-nums">
                          ${optimizedSpend.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </div>
                        <div className="text-xs text-text-tertiary mt-1">per month</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Annual Impact */}
                <div className="border border-divider p-6 bg-secondary/30">
                  <div className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                    Annual Impact
                  </div>
                  <div className="text-3xl font-bold text-index-positive tabular-nums">
                    ${annualSavings.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                  </div>
                  <p className="text-sm text-text-secondary mt-3">
                    Total projected savings over 12 months
                  </p>
                </div>

                {/* CTA */}
                <div className="border-l-2 border-foreground pl-4">
                  <p className="text-sm text-text-secondary mb-4">
                    These are estimates based on optimal model selection for your workload type. 
                    Actual savings depend on specific use cases and implementation.
                  </p>
                  <Button variant="hero" size="sm">
                    Get Custom Analysis
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Methodology Note */}
      <section className="py-16 md:py-24 section-padding bg-secondary/30">
        <div className="content-max max-w-3xl">
          <h3 className="font-semibold text-lg mb-4">How We Calculate Savings</h3>
          <div className="text-sm text-text-secondary space-y-3">
            <p>
              The calculator assumes you're currently using a single model for all workloads. It estimates 
              token volume based on your spend and redistributes those tokens across three tiers based on 
              your workload persona.
            </p>
            <p>
              <strong>Startup:</strong> Heavy on iteration and prototyping. More budget-friendly models for drafts.
            </p>
            <p>
              <strong>Agentic:</strong> Autonomous systems requiring reliable judgment. Weighted toward frontier models.
            </p>
            <p>
              <strong>Throughput:</strong> High-volume processing where cost per token is critical.
            </p>
            <p>
              Actual implementation requires workload classification, routing logic, and fallback strategies. 
              Contact us for guidance on model tiering architectures.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Calculator;
