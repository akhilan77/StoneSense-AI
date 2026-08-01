import type { Patient } from './patient';
import type { RiskPrediction } from './riskPrediction';
import type { StoneDetection } from './stoneDetection';

export interface Explainability {
  shap_values: Record<string, number>;
  lime_explanation: string;
  gradcam_image_url: string;
}

export interface AssessmentResponse {
  patient: Patient;
  risk_prediction: RiskPrediction;
  stone_detection: StoneDetection;
  explainability: Explainability;
  recommendation: string;
}
