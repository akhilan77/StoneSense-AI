export interface RiskPrediction {
  probability: number;
  risk_level: 'low' | 'medium' | 'high';
  confidence: number;
}
