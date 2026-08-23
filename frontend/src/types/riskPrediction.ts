export interface RiskPrediction {
  probability: number;
  risk_level: 'Low' | 'High';
  confidence: number;
  inference_time_sec: number;
  shap?: {
    top_features: string[];
    feature_contributions: Record<string, number>;
    feature_directions: Record<string, 'increases' | 'decreases' | 'neutral'>;
    summary: string;
  };
}
