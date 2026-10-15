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

// ---------------------------------------------------------------------------------------------------
// NHANES clinical-profile models (nhanes_v003): form definition + four-model comparison
// ---------------------------------------------------------------------------------------------------

export interface NhanesFeatureOption {
  value: number;
  label: string;
}

export interface NhanesFeatureSpec {
  name: string;
  label: string;
  unit: string;
  type: 'continuous' | 'binary' | 'categorical' | 'ordinal' | 'derived';
  min: number;
  max: number;
  default: number | null;
  required: boolean;
  group: string;
  sensitive: boolean;
  is_history: boolean;
  question?: string;
  options?: NhanesFeatureOption[];
  clinical_role?: string;
  derived_from?: string[];
  formula?: string;
}

export interface NhanesFeatureSpecResponse {
  version: string;
  primary_variant: string;
  features: NhanesFeatureSpec[];
  notes: string[];
}

export type NhanesFeatureValues = Record<string, number | null>;

export type NhanesOperatingPointKey = 'accuracy' | 'balanced';

/** Held-out TEST metrics at one operating point (95% CIs are bootstrap intervals from the notebooks). */
export interface NhanesOperatingPointMetrics {
  threshold: number;
  accuracy: number;
  accuracy_ci95?: [number, number];
  balanced_accuracy: number;
  balanced_accuracy_ci95?: [number, number];
  sensitivity: number;
  sensitivity_ci95?: [number, number];
  specificity: number;
  specificity_ci95?: [number, number];
  precision?: number;
  f1?: number;
  mcc?: number;
  positive_rate?: number;
  tp?: number;
  fp?: number;
  tn?: number;
  fn?: number;
}

/** One operating point of one model: validation-chosen threshold, this patient's risk level at it, stored test metrics. */
export interface NhanesOperatingPoint {
  label: string;
  rule: string;
  threshold: number | null;
  risk_level: 'Low' | 'High' | null;
  validation: NhanesOperatingPointMetrics | null;
  test: NhanesOperatingPointMetrics | null;
}

export interface NhanesModelTestMetrics {
  evaluated_on: string;
  test_cycle: string;
  n: number;
  prevalence: number;
  majority_baseline_accuracy: number;
  roc_auc: number;
  roc_auc_ci95: [number, number];
  pr_auc?: number | null;
  brier?: number | null;
  at_accuracy_threshold?: NhanesOperatingPointMetrics | null;
  at_balanced_threshold?: NhanesOperatingPointMetrics | null;
}

export interface NhanesTopFactor {
  feature: string;
  label: string;
  value: number | null;
  value_label?: string | null;
  contribution: number;
  direction: 'increases' | 'decreases' | 'neutral';
}

export interface NhanesModelResult {
  model_id: string;
  model_family: 'ebm' | 'catboost' | 'xgboost' | 'lightgbm';
  model_name: string;
  probability: number;
  risk_level: 'Low' | 'High'; // at the selected operating point
  threshold: number; // threshold of the selected operating point
  operating_point: NhanesOperatingPointKey;
  operating_points: Record<NhanesOperatingPointKey, NhanesOperatingPoint>;
  imbalance_strategy?: string | null;
  top_factors: NhanesTopFactor[];
  explanation_summary: string;
  test_metrics: NhanesModelTestMetrics;
  status: string;
  variant: string;
}

export interface NhanesCompareResponse {
  models: NhanesModelResult[];
  average_probability: number;
  models_agree: boolean;
  n_high: number;
  n_low: number;
  operating_point: NhanesOperatingPointKey;
  n_high_by_point: Record<string, number>;
  probability_min: number;
  probability_max: number;
  warnings: string[];
  metrics_note: string;
  disclaimer: string;
  feature_set: string;
  inference_time_sec: number;
}
