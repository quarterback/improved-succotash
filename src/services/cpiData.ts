// Data service for fetching CPI data from JSON files

export interface CPIData {
  meta: {
    generated_at: string;
    baseline_date: string;
    launch_date: string;
    data_sources: string[];
    methodology_version: string;
    models_count: number;
  };
  compute_cpi: {
    ticker: string;
    name: string;
    value: number;
    mom_change: number | null;
    yoy_change: number | null;
    trend: string;
    basket_cost: number;
  };
  subindices: Record<string, {
    ticker: string;
    name: string;
    value: number;
    description: string;
    mom_change: number;
    weight: number;
  }>;
  spreads: Record<string, {
    ticker: string;
    name: string;
    value: number;
    unit: string;
    description: string;
    trend: string;
  }>;
  persona_cpis: Record<string, {
    ticker: string;
    name: string;
    cpi: number;
    mom_change: number;
    description: string;
  }>;
  basket_components: Array<{
    category: string;
    weight: number;
    examples: string;
    avg_cost_per_1m_tokens: number;
  }>;
  trend_analysis?: {
    direction: string;
    velocity: number;
    forecast_30d: number;
    confidence: string;
    factors: string[];
  };
  yield_curve?: {
    curve: Record<string, number>;
    interpretation: string;
    inversion_risk: string;
  };
  exchange_rates?: {
    gpt4_to_claude_sonnet: number;
    claude_opus_to_gpt4: number;
    gemini_ultra_to_gpt4: number;
    description: string;
  };
}

export interface HistoricalData {
  meta: {
    generated_at: string;
    baseline_date: string;
    start_date: string;
    end_date: string;
    data_points: number;
  };
  historical_series: Array<{
    period: string;
    date: string;
    value: number;
    mom_change: number | null;
    yoy_change: number | null;
  }>;
}

export interface RankingsData {
  meta: {
    generated_at: string;
    period: string;
    models_ranked: number;
  };
  rankings: {
    by_market_share: Array<{
      rank: number;
      model: string;
      provider: string;
      market_share: number;
      mom_change: number;
    }>;
    by_quality_adjusted_price: Array<{
      rank: number;
      model: string;
      provider: string;
      qap_score: number;
      cost_per_1m: number;
    }>;
    market_velocity: {
      fastest_growing: Array<{
        model: string;
        growth_rate: number;
      }>;
      declining: Array<{
        model: string;
        decline_rate: number;
      }>;
    };
  };
}

/**
 * Fetch the main CPI data including headline index, subindices, spreads, and personas
 */
export async function fetchCPIData(): Promise<CPIData> {
  const response = await fetch('/data/compute-cpi.json');
  if (!response.ok) {
    throw new Error(`Failed to fetch CPI data: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch historical time series data
 */
export async function fetchHistoricalData(): Promise<HistoricalData> {
  const response = await fetch('/data/historical.json');
  if (!response.ok) {
    throw new Error(`Failed to fetch historical data: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch model rankings and market share data
 */
export async function fetchRankingsData(): Promise<RankingsData> {
  const response = await fetch('/data/rankings/latest.json');
  if (!response.ok) {
    throw new Error(`Failed to fetch rankings data: ${response.statusText}`);
  }
  return response.json();
}
